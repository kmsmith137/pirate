#include "../include/pirate/Dedisperser.hpp"
#include "../include/pirate/DedispersionPlan.hpp"
#include "../include/pirate/DedispersionConfig.hpp"
#include "../include/pirate/RingbufCopyKernel.hpp"
#include "../include/pirate/TreeGriddingKernel.hpp"
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
    this->nfreq = config.get_total_nfreq();
    this->input_rank = config.tree_rank;
    this->input_ntime = config.time_samples_per_chunk;
    this->total_beams = config.beams_per_gpu;
    this->beams_per_batch = config.beams_per_batch;
    this->gpu_ringbuf_nelts = plan->mega_ringbuf->gpu_global_nseg * plan->nelts_per_segment;
    this->host_ringbuf_nelts = plan->mega_ringbuf->host_global_nseg * plan->nelts_per_segment;
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
    //   std::shared_ptr<GpuTreeGriddingKernel> tree_gridding_kernel;
    //   std::vector<std::shared_ptr<GpuDedispersionKernel>> stage1_dd_kernels;
    //   std::vector<std::shared_ptr<GpuDedispersionKernel>> stage2_dd_kernels;
    //   std::shared_ptr<GpuLaggedDownsamplingKernel> lds_kernel;
    
    for (long i = 0; i < nstreams; i++) {
        stage1_dd_bufs.push_back(DedispersionBuffer(plan->stage1_dd_buf_params));
        stage2_dd_bufs.push_back(DedispersionBuffer(plan->stage2_dd_buf_params));
    }

    // Create tree gridding kernel using plan parameters.
    this->tree_gridding_kernel = make_shared<GpuTreeGriddingKernel> (plan->tree_gridding_kernel_params);
    this->bw_per_launch += tree_gridding_kernel->bw_per_launch;

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

    // Allocate input_arrays (frequency-space, shape (beams_per_batch, nfreq, ntime)).
    for (long i = 0; i < nstreams; i++)
        input_arrays.push_back(Array<void>(dtype, {beams_per_batch, nfreq, input_ntime}, af_gpu | af_zero));

    for (auto &buf: stage1_dd_bufs)
        buf.allocate(af_zero | af_gpu);
    
    for (auto &buf: stage2_dd_bufs)
        buf.allocate(af_zero | af_gpu);

    this->tree_gridding_kernel->allocate();
    
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


void GpuDedisperser::launch(long ichunk, long ibatch, long istream, cudaStream_t stream)
{
    const long BT = this->config.beams_per_gpu;            // total beams
    const long BB = this->config.beams_per_batch;          // beams per batch
    const long BA = this->config.num_active_batches * BB;  // active beams
    const long SB = constants::bytes_per_gpu_cache_line;   // bytes per segment

    xassert((ibatch >= 0) && (ibatch < nbatches));
    xassert((istream >= 0) && (istream < nstreams));
    xassert(ichunk >= 0);
    xassert(is_allocated);
    
    long iframe = (ichunk * BT) + (ibatch * BB);
    
    // Step 0: Run tree gridding kernel (input_arrays[istream] -> stage1_dd_bufs[istream].bufs[0]).
    Array<void> &dd_buf0 = stage1_dd_bufs.at(istream).bufs.at(0);
    tree_gridding_kernel->launch(dd_buf0, input_arrays.at(istream), stream);
    
    // Step 1: run LaggedDownsampler.
    lds_kernel->launch(stage1_dd_bufs.at(istream), ichunk, ibatch, stream);

    // Step 2: run stage1 dedispersion kernels (output to ringbuf)
    for (uint i = 0; i < stage1_dd_kernels.size(); i++) {
        shared_ptr<GpuDedispersionKernel> kernel = stage1_dd_kernels.at(i);
        const DedispersionKernelParams &kp = kernel->params;
        Array<void> dd_buf = stage1_dd_bufs.at(istream).bufs.at(i);

        // See comments in DedispersionKernel.hpp for an explanation of this reshape operation.
        dd_buf = dd_buf.reshape({ kp.beams_per_batch, pow2(kp.amb_rank), pow2(kp.dd_rank), kp.ntime });
        kernel->launch(dd_buf, this->gpu_ringbuf, ichunk, ibatch, stream);
    }

    // Step 3: extra copying steps needed for early triggers.

    MegaRingbuf::Zone &eth_zone = plan->mega_ringbuf->et_host_zone;
    MegaRingbuf::Zone &etg_zone = plan->mega_ringbuf->et_gpu_zone;
    
    xassert(eth_zone.segments_per_frame == etg_zone.segments_per_frame);
    xassert(eth_zone.num_frames == BA);
    xassert(etg_zone.num_frames == BA);

    long et_off = (iframe % eth_zone.num_frames) * eth_zone.segments_per_frame;
    char *et_src = (char *) this->host_ringbuf.data + (eth_zone.global_segment_offset + et_off) * SB;
    char *et_dst = (char *) this->gpu_ringbuf.data + (etg_zone.global_segment_offset + et_off) * SB;
    long et_nbytes = BB * eth_zone.segments_per_frame * SB;
    
    // copy gpu -> xfer
    this->g2g_copy_kernel->launch(this->gpu_ringbuf, ichunk, ibatch, stream);

    // copy host -> et_host
    this->h2h_copy_kernel->apply(this->host_ringbuf, ichunk, ibatch);

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
        
        char *hp = reinterpret_cast<char *> (this->host_ringbuf.data) + (host_zone.global_segment_offset * SB);
        char *xp = reinterpret_cast<char *> (this->gpu_ringbuf.data) + (xfer_zone.global_segment_offset * SB);
        
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
        kernel->launch(this->gpu_ringbuf, dd_buf, ichunk, ibatch, stream);
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

    // I decided that this was the least awkward place to call DedispersionConfig::test().
    config.test();

    long nfreq = config.get_total_nfreq();
    int nt_chunk = config.time_samples_per_chunk;
    int beams_per_batch = config.beams_per_batch;
    int nbatches = xdiv(config.beams_per_gpu, beams_per_batch);
    int nstreams = config.num_active_batches;
    int nout = plan->stage2_trees.size();

    // FIXME test multi-stream logic in the future.
    // For now, we use the default cuda stream, which simplifies things since we can
    // freely mix operations such as Array::to_gpu() which use the default stream.
    xassert(nstreams == 1);
    
    // Some initializations:
    //   ref_kernels_for_weights: only used for ReferencePeakFindingKernel::make_random_weights().
    //   pf_tmp: used to store output from ReferencePeakFindingKernel::eval_tokens().

    vector<long> Dcore(plan->stage2_ntrees);
    vector<Array<float>> pf_tmp(plan->stage2_ntrees);
    vector<Array<float>> subband_variances(plan->stage2_ntrees);
    vector<shared_ptr<ReferencePeakFindingKernel>> ref_kernels_for_weights(plan->stage2_ntrees);

    for (long i = 0; i < plan->stage2_ntrees; i++) {
        const DedispersionPlan::Stage2Tree &st2 = plan->stage2_trees.at(i);
        const PeakFindingKernelParams &pf_params = plan->stage2_pf_params.at(i);
        long Dout = xdiv(pf_params.nt_in, pf_params.nt_out);

        // Logic to initialize Dcore is currently a placeholder.
        Dcore.at(i) = Dout;
        ref_kernels_for_weights.at(i) = make_shared<ReferencePeakFindingKernel> (pf_params, Dcore.at(i));
        pf_tmp.at(i) = Array<float> ({beams_per_batch, pf_params.ndm_out, pf_params.nt_out}, af_uhost | af_zero);

        // s = (conversion between FrequencySubbands::{ilo,ihi} and sample delay)
        FrequencySubbands fs(pf_params.subband_counts);
        long s = pow2(st2.amb_rank + st2.early_dd_rank - fs.pf_rank);

        // Init subband_variances (used in ReferencePeakFindingKernel::make_random_weights()).
        subband_variances.at(i) = Array<float> ({fs.F}, af_uhost | af_zero);
        for (long f = 0; f < fs.F; f++) {
            long ilo = fs.f_to_ilo.at(f);
            long ihi = fs.f_to_ihi.at(f);
            double flo = config.delay_to_frequency(s * ihi);  // note ihi here
            double fhi = config.delay_to_frequency(s * ilo);  // note ilo here
            subband_variances.at(i).at({f}) = fhi - flo;      // frequency range of subband
        }
    }

    shared_ptr<ReferenceDedisperserBase> rdd0 = ReferenceDedisperserBase::make(plan, Dcore, 0);
    shared_ptr<ReferenceDedisperserBase> rdd1 = ReferenceDedisperserBase::make(plan, Dcore, 1);
    shared_ptr<ReferenceDedisperserBase> rdd2 = ReferenceDedisperserBase::make(plan, Dcore, 2);

    shared_ptr<GpuDedisperser> gdd;

    if (!host_only) {
        gdd = make_shared<GpuDedisperser> (plan);
        gdd->allocate();
    }

    // FIXME revisit epsilon if we change the normalization of the dedispersion transform.
    // Note: epsilon accounts for both tree gridding (accumulates nfreq/nchan values on average)
    // and dedispersion (factor of pow(1.414, tree_rank)).
    double tree_gridding_factor = sqrt(double(nfreq) / pow2(config.tree_rank)) + 1.0;
    double dedispersion_factor = pow(1.414, config.tree_rank);
    double epsrel_r = 6 * Dtype::native<float>().precision();                      // reference
    double epsrel_g = 6 * config.dtype.precision();                                // gpu
    double epsabs_r = epsrel_r * tree_gridding_factor * dedispersion_factor;       // reference
    double epsabs_g = epsrel_g * tree_gridding_factor * dedispersion_factor;       // gpu

    for (int ichunk = 0; ichunk < nchunks; ichunk++) {
        for (int ibatch = 0; ibatch < nbatches; ibatch++) {
            // Randomly initialize weights.
            for (int itree = 0; itree < plan->stage2_ntrees; itree++) {
                Array<float> sbv = subband_variances.at(itree);
                Array<float> wt_cpu = ref_kernels_for_weights.at(itree)->make_random_weights(sbv);
                rdd0->wt_arrays.at(itree).fill(wt_cpu);
                rdd1->wt_arrays.at(itree).fill(wt_cpu);
                rdd2->wt_arrays.at(itree).fill(wt_cpu);
            }

            // Frequency-space array with shape (beams_per_batch, nfreq, ntime).
            // Random values uniform over [-1.0, 1.0].
            Array<float> arr({beams_per_batch, nfreq, nt_chunk}, af_uhost);
            for (long i = 0; i < arr.size; i++)
                arr.data[i] = ksgpu::rand_uniform(-1.0, 1.0);

            rdd0->input_array.fill(arr);
            rdd0->dedisperse(ichunk, ibatch);  // (ichunk, ibatch)

            rdd1->input_array.fill(arr);
            rdd1->dedisperse(ichunk, ibatch);  // (ichunk, ibatch)

            rdd2->input_array.fill(arr);
            rdd2->dedisperse(ichunk, ibatch);  // (ichunk, ibatch)

            if (!host_only) {
                gdd->input_arrays.at(0).fill(arr.convert(config.dtype));  // istream=0
                gdd->launch(ichunk, ibatch, 0, nullptr);  // (ichunk, ibatch, istream, stream)
            }
            
            for (int iout = 0; iout < nout; iout++) {
                const Array<float> &rdd0_out = rdd0->output_arrays.at(iout);
                const Array<float> &rdd1_out = rdd1->output_arrays.at(iout);
                const Array<float> &rdd2_out = rdd2->output_arrays.at(iout);
                
                // Last two arguments are (epsabs, epsrel).
                assert_arrays_equal(rdd0_out, rdd1_out, "dd_ref0", "dd_ref1", {"beam","dm_brev","t"}, epsabs_r, epsrel_r);
                assert_arrays_equal(rdd0_out, rdd2_out, "dd_ref0", "dd_ref2", {"beam","dm_brev","t"}, epsabs_r, epsrel_r);

                if (!host_only) {
                    const Array<void> &gdd_out = gdd->stage2_dd_bufs.at(0).bufs.at(iout);  // (istream,itree) = (0,iout)
                    assert_arrays_equal(rdd0_out, gdd_out, "dd_ref0", "dd_gpu", {"beam","dm_brev","t"}, epsabs_g, epsrel_g);
                }

                // Compare peak-finding 'out_max'.
                assert_arrays_equal(rdd1->out_max.at(iout), rdd2->out_max.at(iout), "pfmax_ref1", "pfmax_ref2", {"beam","pfdm","pft"});

                // To check 'out_argmax', we need to jump through some hoops.
                shared_ptr<ReferencePeakFindingKernel> pf_kernel = rdd1->get_pf_kernel(iout);
                pf_kernel->eval_tokens(pf_tmp.at(iout), rdd2->out_argmax.at(iout), rdd1->wt_arrays.at(iout));
                assert_arrays_equal(rdd1->out_max.at(iout), pf_tmp.at(iout), "pfmax_ref1", "pftmp_ref2", {"beam","pfdm","pft"});
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
    
    long ntree = pow2(config.tree_rank);
    long nt_chunk = config.time_samples_per_chunk;
    long min_nchunks = (ntree / nt_chunk) + 2;
    long max_nchunks = (1024*1024) / (ntree * nt_chunk * config.beams_per_gpu);
    max_nchunks = max(min_nchunks, max_nchunks);

    long nchunks = ksgpu::rand_int(1, max_nchunks+1);    
    GpuDedisperser::test_one(config, nchunks);
}


}  // namespace pirate
