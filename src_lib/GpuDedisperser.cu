#include "../include/pirate/Dedisperser.hpp"
#include "../include/pirate/DedispersionPlan.hpp"
#include "../include/pirate/DedispersionConfig.hpp"
#include "../include/pirate/RingbufCopyKernel.hpp"
#include "../include/pirate/MegaRingbuf.hpp"
#include "../include/pirate/constants.hpp"  // xdiv(), pow2()
#include "../include/pirate/inlines.hpp"  // xdiv(), pow2()

#include <ksgpu/rand_utils.hpp>
#include <ksgpu/test_utils.hpp>

using namespace std;
using namespace ksgpu;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// Helper for GpuDedisperser constructor.
// Prevents constructor from segfaulting, if invoked with empty shared_ptr.
static DedispersionPlan *deref(const shared_ptr<DedispersionPlan> &p)
{
    if (!p)
        throw runtime_error("GpuDedisperser constructor called with empty shared_ptr");
    return p.get();
}


GpuDedisperser::GpuDedisperser(const shared_ptr<DedispersionPlan> &plan_) :
    plan(plan_),
    config(deref(plan_)->config)
{
    // There's some cut-and-paste between this constructor and the ReferenceDedisperser
    // constructor, but not enough to bother defining a common base class.
    
    this->dtype = config.dtype;
    this->input_rank = config.tree_rank;
    this->input_ntime = config.time_samples_per_chunk;
    this->total_beams = config.beams_per_gpu;
    this->beams_per_batch = config.beams_per_batch;
    this->gpu_ringbuf_nelts = plan->mega_ringbuf->gpu_giant_nseg * plan->nelts_per_segment;
    this->host_ringbuf_nelts = plan->mega_ringbuf->host_giant_nseg * plan->nelts_per_segment;
    this->nbatches = xdiv(total_beams, beams_per_batch);
    this->nstreams = config.num_active_batches;

    const DedispersionBufferParams &out_params = plan->stage2_dd_buf_params;
    this->output_ntrees = out_params.nbuf;
    this->output_rank = out_params.buf_rank;
    this->output_ntime = out_params.buf_ntime;
    this->output_ds_level = plan->stage2_ds_level;

    long nbits_per_segment = plan->nelts_per_segment * dtype.nbits;
    xassert_eq(nbits_per_segment, 8 * constants::bytes_per_gpu_cache_line);  // currently assumed in a few places
    
    // Construct, but do not allocate, the following members:
    //
    //   std::vector<DedispersionBuffer> stage1_dd_bufs;  // length nstreams
    //   std::vector<DedispersionBuffer> stage2_dd_bufs;  // length nstreams
    //   std::vector<std::shared_ptr<GpuDedispersionKernel>> stage1_dd_kernels;
    //   std::vector<std::shared_ptr<GpuDedispersionKernel>> stage2_dd_kernels;
    //   std::shared_ptr<GpuLaggedDownsamplingKernel> lds_kernel;
    
    for (long i = 0; i < nstreams; i++) {
        stage1_dd_bufs.push_back(DedispersionBuffer(plan->stage1_dd_buf_params));
        stage2_dd_bufs.push_back(DedispersionBuffer(plan->stage2_dd_buf_params));
    }

    for (const DedispersionKernelParams &kparams: plan->stage1_dd_kernel_params) {
        auto kernel = make_shared<GpuDedispersionKernel> (kparams);
        this->stage1_dd_kernels.push_back(kernel);
        this->bw_per_launch += kernel->bw_per_launch;
    }

    for (const DedispersionKernelParams &kparams: plan->stage2_dd_kernel_params) {
        auto kernel = make_shared<GpuDedispersionKernel> (kparams);
        this->stage2_dd_kernels.push_back(kernel);
        this->bw_per_launch += kernel->bw_per_launch;
    }

    this->lds_kernel = GpuLaggedDownsamplingKernel::make(plan->lds_params);
    this->g2g_copy_kernel = make_shared<GpuRingbufCopyKernel> (plan->g2g_copy_kernel_params);
    this->h2h_copy_kernel = make_shared<CpuRingbufCopyKernel> (plan->h2h_copy_kernel_params);

    this->bw_per_launch += lds_kernel->bw_per_launch;
}


void GpuDedisperser::allocate()
{
    if (this->is_allocated)
        throw runtime_error("double call to GpuDedisperser::allocate()");

    for (auto &buf: stage1_dd_bufs)
        buf.allocate(af_zero | af_gpu);
    
    for (auto &buf: stage2_dd_bufs)
        buf.allocate(af_zero | af_gpu);
    
    for (auto &kernel: this->stage1_dd_kernels)
        kernel->allocate();
    
    for (auto &kernel: this->stage2_dd_kernels)
        kernel->allocate();

    this->gpu_ringbuf = Array<void>(dtype, { gpu_ringbuf_nelts }, af_gpu | af_zero);
    this->host_ringbuf = Array<void>(dtype, { host_ringbuf_nelts }, af_rhost | af_zero);    

    this->lds_kernel->allocate();
    this->g2g_copy_kernel->allocate();

    this->is_allocated = true;
}


void GpuDedisperser::launch(long ibatch, long it_chunk, long istream, cudaStream_t stream)
{
    const long BT = this->config.beams_per_gpu;            // total beams
    const long BB = this->config.beams_per_batch;          // beams per batch
    const long BA = this->config.num_active_batches * BB;  // active beams
    const long SB = constants::bytes_per_gpu_cache_line;   // bytes per segment

    xassert((ibatch >= 0) && (ibatch < nbatches));
    xassert((istream >= 0) && (istream < nstreams));
    xassert(it_chunk >= 0);
    xassert(is_allocated);
    
    long iframe = (it_chunk * BT) + (ibatch * BB);
    
    // Step 1: run LaggedDownsampler.
    lds_kernel->launch(stage1_dd_bufs.at(istream), ibatch, it_chunk, stream);

    // Step 2: run stage1 dedispersion kernels (output to ringbuf)
    for (uint i = 0; i < stage1_dd_kernels.size(); i++) {
        shared_ptr<GpuDedispersionKernel> kernel = stage1_dd_kernels.at(i);
        const DedispersionKernelParams &kp = kernel->params;
        Array<void> dd_buf = stage1_dd_bufs.at(istream).bufs.at(i);

        // See comments in DedispersionKernel.hpp for an explanation of this reshape operation.
        dd_buf = dd_buf.reshape({ kp.beams_per_batch, pow2(kp.amb_rank), pow2(kp.dd_rank), kp.ntime });
        kernel->launch(dd_buf, this->gpu_ringbuf, ibatch, it_chunk, stream);
    }

    // Step 3: extra copying steps needed for early triggers.

    MegaRingbuf::Zone &eth_zone = plan->mega_ringbuf->et_host_zone;
    MegaRingbuf::Zone &etg_zone = plan->mega_ringbuf->et_gpu_zone;
    
    xassert(eth_zone.segments_per_frame == etg_zone.segments_per_frame);
    xassert(eth_zone.num_frames == BA);
    xassert(etg_zone.num_frames == BA);

    long et_off = (iframe % eth_zone.num_frames) * eth_zone.segments_per_frame;
    char *et_src = (char *) this->host_ringbuf.data + (eth_zone.giant_segment_offset + et_off) * SB;
    char *et_dst = (char *) this->gpu_ringbuf.data + (etg_zone.giant_segment_offset + et_off) * SB;
    long et_nbytes = BB * eth_zone.segments_per_frame * SB;
    
    // copy gpu -> xfer
    this->g2g_copy_kernel->launch(this->gpu_ringbuf, ibatch, it_chunk, stream);

    // copy host -> et_host
    this->h2h_copy_kernel->apply(this->host_ringbuf, ibatch, it_chunk);

     // copy et_host -> et_gpu (must come after h2h_copy_kernel)
    CUDA_CALL(cudaMemcpyAsync(et_dst, et_src, et_nbytes, cudaMemcpyHostToDevice, stream)); 

    // Step 4: copy host <-> xfer
    // There is some cut-and-paste with ReferenceDedisperser, but not enough to bother refactoring.
    // FIXME: currently putting copies on the compute stream!
    // This is terrible for performance, but I'm just testing correctness for now.
    
    shared_ptr<MegaRingbuf> mega_ringbuf = plan->mega_ringbuf;
    long max_clag = mega_ringbuf->max_clag;
    xassert(mega_ringbuf->host_zones.size() == uint(max_clag+1));
    xassert(mega_ringbuf->xfer_zones.size() == uint(max_clag+1));
    xassert_divisible(BT, BB);   // assert that length-BB copies don't "wrap"
    
    for (int clag = 0; clag <= max_clag; clag++) {
        MegaRingbuf::Zone &host_zone = mega_ringbuf->host_zones.at(clag);
        MegaRingbuf::Zone &xfer_zone = mega_ringbuf->xfer_zones.at(clag);

        xassert(host_zone.segments_per_frame == xfer_zone.segments_per_frame);
        xassert(host_zone.num_frames == clag*BT + BA);
        xassert(xfer_zone.num_frames == 2*BA);

        if (host_zone.segments_per_frame == 0)
            continue;
        
        char *hp = reinterpret_cast<char *> (this->host_ringbuf.data) + (host_zone.giant_segment_offset * SB);
        char *xp = reinterpret_cast<char *> (this->gpu_ringbuf.data) + (xfer_zone.giant_segment_offset * SB);
        
        long hsrc = (iframe + BA) % host_zone.num_frames;  // host src phase
        long hdst = (iframe) % host_zone.num_frames;       // host dst phase
        long xsrc = (iframe) % xfer_zone.num_frames;       // xfer src phase
        long xdst = (iframe + BA) % xfer_zone.num_frames;  // xfer dst phase
        
        long m = host_zone.segments_per_frame * SB;  // nbytes per frame
        long n = BB * m;                             // nbytes to copy
        
        CUDA_CALL(cudaMemcpyAsync(xp + xdst*m, hp + hsrc*m, n, cudaMemcpyHostToDevice, stream));
        CUDA_CALL(cudaMemcpyAsync(hp + hdst*m, xp + xsrc*m, n, cudaMemcpyDeviceToHost, stream));
    }
    
    // Step 5: run stage2 dedispersion kernels (input from ringbuf)
    for (uint i = 0; i < stage2_dd_kernels.size(); i++) {
        shared_ptr<GpuDedispersionKernel> kernel = stage2_dd_kernels.at(i);
        const DedispersionKernelParams &kp = kernel->params;
        Array<void> dd_buf = stage2_dd_bufs.at(istream).bufs.at(i);

        // See comments in DedispersionKernel.hpp for an explanation of this reshape/transpose operation.
        dd_buf = dd_buf.reshape({ kp.beams_per_batch, pow2(kp.dd_rank), pow2(kp.amb_rank), kp.ntime });
        dd_buf = dd_buf.transpose({0,2,1,3});
        kernel->launch(this->gpu_ringbuf, dd_buf, ibatch, it_chunk, stream);
    }
}


// -------------------------------------------------------------------------------------------------
//
// GpuDedisperser::test()


// Static member function.
void GpuDedisperser::test_one(const DedispersionConfig &config, int nchunks, bool host_only)
{
    cout << "\n" << "GpuDedisperser::test()" << endl;
    config.print(cout, 4);
    print_kv("nchunks", nchunks, cout, 4);
    
    shared_ptr<DedispersionPlan> plan = make_shared<DedispersionPlan> (config);
    print_kv("max_clag", plan->mega_ringbuf->max_clag, cout, 4);
    print_kv("max_gpu_clag", plan->mega_ringbuf->max_gpu_clag, cout, 4);

    if (host_only)
        cout << "!!! Host-only test, GPU code will not be run !!!" << endl;
    
    int nfreq = pow2(config.tree_rank);
    int nt_chunk = config.time_samples_per_chunk;
    int beams_per_batch = config.beams_per_batch;
    int nbatches = xdiv(config.beams_per_gpu, beams_per_batch);
    int nstreams = config.num_active_batches;
    int nout = plan->stage2_trees.size();

    // FIXME test multi-stream logic in the future.
    // For now, we use the default cuda stream, which simplifies things since we can
    // freely mix operations such as Array::to_gpu() which use the default stream.
    xassert(nstreams == 1);
    
    shared_ptr<ReferenceDedisperserBase> rdd0 = ReferenceDedisperserBase::make(plan, 0);
    shared_ptr<ReferenceDedisperserBase> rdd1 = ReferenceDedisperserBase::make(plan, 1);
    shared_ptr<ReferenceDedisperserBase> rdd2 = ReferenceDedisperserBase::make(plan, 2);
    
    shared_ptr<GpuDedisperser> gdd;

    if (!host_only) {
        gdd = make_shared<GpuDedisperser> (plan);
        gdd->allocate();
    }

    // FIXME revisit epsilon if we change the normalization of the dedispersion transform.
    double epsrel_r = 6 * Dtype::native<float>().precision();   // reference
    double epsrel_g = 6 * config.dtype.precision();             // gpu
    double epsabs_r = epsrel_r * pow(1.414, config.tree_rank);  // reference
    double epsabs_g = epsrel_g * pow(1.414, config.tree_rank);  // gpu

    for (int c = 0; c < nchunks; c++) {
        for (int b = 0; b < nbatches; b++) {
            Array<float> arr({beams_per_batch, nfreq, nt_chunk}, af_uhost | af_random);
            // Array<float> arr({nfreq,nt_chunk}, af_uhost | af_zero);
            // arr.at({0,0}) = 1.0;

            rdd0->input_array.fill(arr);
            rdd0->dedisperse(b, c);

            rdd1->input_array.fill(arr);
            rdd1->dedisperse(b, c);

            rdd2->input_array.fill(arr);
            rdd2->dedisperse(b, c);

            if (!host_only) {
                Array<void> &gdd_inbuf = gdd->stage1_dd_bufs.at(0).bufs.at(0);  // (istream,itree) = (0,0)
                gdd_inbuf.fill(arr.convert(config.dtype));
                gdd->launch(b, c, 0, nullptr);  // (ibatch, it_chunk, istream, stream)
            }
            
            for (int iout = 0; iout < nout; iout++) {
                const Array<float> &rdd0_out = rdd0->output_arrays.at(iout);
                const Array<float> &rdd1_out = rdd1->output_arrays.at(iout);
                const Array<float> &rdd2_out = rdd2->output_arrays.at(iout);
                
                // Last two arguments are (epsabs, epsrel).
                assert_arrays_equal(rdd0_out, rdd1_out, "soph0", "soph1", {"beam","dm_brev","t"}, epsabs_r, epsrel_r);
                assert_arrays_equal(rdd0_out, rdd2_out, "soph0", "soph2", {"beam","dm_brev","t"}, epsabs_r, epsrel_r);

                if (!host_only) {
                    const Array<void> &gdd_out = gdd->stage2_dd_bufs.at(0).bufs.at(iout);  // (istream,itree) = (0,iout)
                    assert_arrays_equal(rdd0_out, gdd_out, "soph0", "gpu", {"beam","dm_brev","t"}, epsabs_g, epsrel_g);
                }
            }
        }
    }
    
    cout << endl;
}

// Static member function.
void GpuDedisperser::test()
{
    auto config = DedispersionConfig::make_random();
    config.num_active_batches = 1;   // FIXME currently we only support nstreams==1
    config.validate();
    
    int max_nt = 8192;
    xassert(config.time_samples_per_chunk <= max_nt);
    
    int max_nchunks = max_nt / config.time_samples_per_chunk;  // round down
    int nchunks = ksgpu::rand_int(1, max_nchunks+1);
    
    GpuDedisperser::test_one(config, nchunks);
}


}  // namespace pirate
