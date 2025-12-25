#include "../include/pirate/Dedisperser.hpp"
#include "../include/pirate/DedispersionPlan.hpp"
#include "../include/pirate/DedispersionConfig.hpp"
#include "../include/pirate/CoalescedDdKernel2.hpp"
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

// Helper for GpuDedisperser constructor.
// Concatenate scalar + vector.
template<typename T>
static inline vector<T> svcat(const T &s, const vector<T> &v)
{
    long n = v.size();
    vector<T> ret(n+1);

    ret[0] = s;
    for (long i = 0; i < n; i++)
        ret[i+1] = v[i];

    return ret;
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

    long nbits_per_segment = plan->nelts_per_segment * dtype.nbits;
    xassert_eq(nbits_per_segment, 8 * constants::bytes_per_gpu_cache_line);  // currently assumed in a few places
    
    
    for (long i = 0; i < nstreams; i++)
        stage1_dd_bufs.push_back(DedispersionBuffer(plan->stage1_dd_buf_params));

    // Create tree gridding kernel using plan parameters.
    this->tree_gridding_kernel = make_shared<GpuTreeGriddingKernel> (plan->tree_gridding_kernel_params);
    this->bw_per_launch += tree_gridding_kernel->bw_per_launch;

    for (const DedispersionKernelParams &kparams: plan->stage1_dd_kernel_params) {
        auto kernel = make_shared<GpuDedispersionKernel> (kparams);
        this->stage1_dd_kernels.push_back(kernel);
        this->bw_per_launch += kernel->bw_per_launch;
    }

    for (long itree = 0; itree < output_ntrees; itree++) {
        const DedispersionKernelParams &dd_params = plan->stage2_dd_kernel_params.at(itree);
        const PeakFindingKernelParams &pf_params = plan->stage2_pf_params.at(itree);
        auto cdd2_kernel = make_shared<CoalescedDdKernel2> (dd_params, pf_params);
        this->cdd2_kernels.push_back(cdd2_kernel);
        // this->bw_per_launch += cdd2_kernel->bw_per_launch;
    }

    this->lds_kernel = GpuLaggedDownsamplingKernel::make(plan->lds_params);
    this->g2g_copy_kernel = make_shared<GpuRingbufCopyKernel> (plan->g2g_copy_kernel_params);
    this->h2h_copy_kernel = make_shared<CpuRingbufCopyKernel> (plan->h2h_copy_kernel_params);

    for (long itree = 0; itree < output_ntrees; itree++) {
        const vector<long> &shape = cdd2_kernels.at(itree)->expected_wt_shape;
        const vector<long> &strides = cdd2_kernels.at(itree)->expected_wt_strides;

        // "Extended" shapes with a stream axis added.
        this->extended_wt_shapes.push_back(svcat(nstreams, shape));
        this->extended_wt_strides.push_back(svcat(shape[0] * strides[0], strides));
    }

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

    // wt_arrays
    for (long itree = 0; itree < output_ntrees; itree++) {
        const vector<long> &shape = extended_wt_shapes.at(itree);
        const vector<long> &strides = extended_wt_strides.at(itree);
        wt_arrays.push_back(Array<void> (dtype, shape, strides, af_gpu | af_zero));
    }

    // out_max, out_argmax
    for (long itree = 0; itree < output_ntrees; itree++) {
        const DedispersionTree &tree = plan->trees.at(itree);
        std::initializer_list<long> shape = { nstreams, beams_per_batch, tree.ndm_out, tree.nt_out };
        out_max.push_back(Array<void> (dtype, shape, af_gpu | af_zero));
        out_argmax.push_back(Array<uint> (shape, af_gpu | af_zero));
    }

    this->tree_gridding_kernel->allocate();
    
    for (auto &kernel: this->stage1_dd_kernels)
        kernel->allocate();
    
    for (auto &kernel: this->cdd2_kernels)
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
    
    // Step 5: run cdd2 kernels (input from ringbuf)
    for (long itree = 0; itree < output_ntrees; itree++) {
        Array<void> slice_max = out_max.at(itree).slice(0,istream);
        Array<uint> slice_argmax = out_argmax.at(itree).slice(0,istream);
        Array<void> slice_wt = wt_arrays.at(itree).slice(0,istream);

        shared_ptr<CoalescedDdKernel2> cdd2_kernel = cdd2_kernels.at(itree);
        cdd2_kernel->launch(slice_max, slice_argmax, this->gpu_ringbuf, slice_wt, ichunk, ibatch, stream);
    }
}


// -------------------------------------------------------------------------------------------------
//
// GpuDedisperser::test()


static double variance_upper_bound(const shared_ptr<DedispersionPlan> &plan, long itree, long f)
{
    const DedispersionTree &tree = plan->trees.at(itree);
    const FrequencySubbands &fs = tree.frequency_subbands;

    long ilo = fs.f_to_ilo.at(f);
    long ihi = fs.f_to_ihi.at(f);

    // Frequency range in MHz (note lo/hi swap)
    double flo = fs.i_to_f.at(ihi);
    double fhi = fs.i_to_f.at(ilo);

    // Frequency index range
    double filo = plan->config.frequency_to_index(flo);
    double fihi = plan->config.frequency_to_index(fhi);

    return (fihi - filo) * pow2(tree.ds_level) / 3.0;
}


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
    int nout = plan->trees.size();

    // FIXME test multi-stream logic in the future.
    // For now, we use the default cuda stream, which simplifies things since we can
    // freely mix operations such as Array::to_gpu() which use the default stream.
    xassert(nstreams == 1);
    
    shared_ptr<GpuDedisperser> gdd;

    if (!host_only) {
        gdd = make_shared<GpuDedisperser> (plan);
        gdd->allocate();
    }

    // Some initializations:
    //   Dcore: taken from GPU kernel, passed to reference kernel
    //   pf_tmp: used to store output from ReferencePeakFindingKernel::eval_tokens().
    //   subband_variances: used in ReferencePeakFindingKernel::make_random_weights().
    //   ref_kernels_for_weights: only used for ReferencePeakFindingKernel::make_random_weights().

    vector<long> Dcore(plan->ntrees);
    vector<Array<float>> pf_tmp(plan->ntrees);
    vector<Array<float>> subband_variances(plan->ntrees);
    vector<shared_ptr<ReferencePeakFindingKernel>> ref_kernels_for_weights(plan->ntrees);

    for (long itree = 0; itree < plan->ntrees; itree++) {
        const DedispersionTree &tree = plan->trees.at(itree);
        const PeakFindingKernelParams &pf_params = plan->stage2_pf_params.at(itree);
        long F = tree.frequency_subbands.F;

        Dcore.at(itree) = host_only ? tree.pf.time_downsampling : gdd->cdd2_kernels.at(itree)->Dcore;
        pf_tmp.at(itree) = Array<float> ({beams_per_batch, pf_params.ndm_out, pf_params.nt_out}, af_uhost | af_zero);
        ref_kernels_for_weights.at(itree) = make_shared<ReferencePeakFindingKernel> (pf_params, Dcore.at(itree));
        subband_variances.at(itree) = Array<float> ({F}, af_uhost | af_zero);

        for (long f = 0; f < F; f++)
            subband_variances.at(itree).at({f}) = variance_upper_bound(plan, itree, f);
    }

    shared_ptr<ReferenceDedisperserBase> rdd0 = ReferenceDedisperserBase::make(plan, Dcore, 0);
    shared_ptr<ReferenceDedisperserBase> rdd1 = ReferenceDedisperserBase::make(plan, Dcore, 1);
    shared_ptr<ReferenceDedisperserBase> rdd2 = ReferenceDedisperserBase::make(plan, Dcore, 2);

    for (int ichunk = 0; ichunk < nchunks; ichunk++) {
        for (int ibatch = 0; ibatch < nbatches; ibatch++) {
            // Randomly initialize weights.
            for (int itree = 0; itree < plan->ntrees; itree++) {
                Array<float> sbv = subband_variances.at(itree);
                Array<float> wt_cpu = ref_kernels_for_weights.at(itree)->make_random_weights(sbv);

                rdd0->wt_arrays.at(itree).fill(wt_cpu);
                rdd1->wt_arrays.at(itree).fill(wt_cpu);
                rdd2->wt_arrays.at(itree).fill(wt_cpu);

                if (!host_only) {
                    const GpuPfWeightLayout &wl = gdd->cdd2_kernels.at(itree)->pf_weight_layout;
                    Array<void> wt_gpu = gdd->wt_arrays.at(itree).slice(0,0);  // FIXME istream=0 assumed

                    // FIXME extra copy here (+ another extra copy "hidden" in GpuPfWeightLayout::to_gpu())
                    Array<void> tmp = wl.to_gpu(wt_cpu);
                    wt_gpu.fill(tmp);
                }
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
                long ds_level = plan->trees.at(iout).ds_level;

                // Compute epsabs, epsrel for host and gpu
                Array<float> sbv = subband_variances.at(iout);
                float rms = sqrt(sbv.at({sbv.size-1}));
                float emult = sqrt(config.tree_rank + ds_level + 1);

                Dtype fp32 = Dtype::from_str("float32");
                double epsrel_g = 3 * config.dtype.precision() * emult;
                double epsabs_g = 3 * config.dtype.precision() * emult * rms;
                double epsrel_r = 3 * fp32.precision() * emult;
                double epsabs_r = 3 * fp32.precision() * emult * rms;

                // Last two arguments are (epsabs, epsrel).
                assert_arrays_equal(rdd0_out, rdd1_out, "dd_ref0", "dd_ref1", {"beam","dm_brev","t"}, epsabs_r, epsrel_r);
                assert_arrays_equal(rdd0_out, rdd2_out, "dd_ref0", "dd_ref2", {"beam","dm_brev","t"}, epsabs_r, epsrel_r);

                // Compare peak-finding 'out_max'.
                Array<void> gdd_max = gdd->out_max.at(iout).slice(0,0);  // FIXME istream=0 assumed
                assert_arrays_equal(rdd0->out_max.at(iout), rdd1->out_max.at(iout), "pfmax_ref0", "pfmax_ref1", {"beam","pfdm","pft"});
                assert_arrays_equal(rdd0->out_max.at(iout), rdd2->out_max.at(iout), "pfmax_ref0", "pfmax_ref2", {"beam","pfdm","pft"});
                assert_arrays_equal(rdd0->out_max.at(iout), gdd_max, "pfmax_ref0", "pfmax_gpu", {"beam","pfdm","pft"});

                // To check 'out_argmax', we need to jump through some hoops.
                shared_ptr<ReferencePeakFindingKernel> pf_kernel = rdd0->pf_kernels.at(iout);

                pf_kernel->eval_tokens(pf_tmp.at(iout), rdd1->out_argmax.at(iout), rdd0->wt_arrays.at(iout));
                assert_arrays_equal(rdd0->out_max.at(iout), pf_tmp.at(iout), "pfmax_ref0", "pf_tmp_ref1", {"beam","pfdm","pft"});

                pf_kernel->eval_tokens(pf_tmp.at(iout), rdd2->out_argmax.at(iout), rdd0->wt_arrays.at(iout));
                assert_arrays_equal(rdd0->out_max.at(iout), pf_tmp.at(iout), "pfmax_ref0", "pf_tmp_ref2", {"beam","pfdm","pft"});

                double eps = 5.0 * config.dtype.precision();
                Array<uint> gpu_tokens = gdd->out_argmax.at(iout).slice(0,0).to_host();  // FIXME istream=0 assumed 
                pf_kernel->eval_tokens(pf_tmp.at(iout), gpu_tokens, rdd0->wt_arrays.at(iout));
                assert_arrays_equal(rdd0->out_max.at(iout), pf_tmp.at(iout), "pfmax_ref0", "pf_tmp_gpu", {"beam","pfdm","pft"}, eps, eps);
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
