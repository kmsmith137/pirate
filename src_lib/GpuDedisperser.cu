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
    plan(plan_)
{
    xassert(plan);

    // There's some cut-and-paste between this constructor and the ReferenceDedisperser
    // constructor, but not enough to bother defining a common base class.

    this->config = plan->config;
    this->dtype = plan->dtype;
    this->nfreq = plan->nfreq;
    this->nt_in = plan->nt_in;
    this->total_beams = plan->beams_per_gpu;
    this->beams_per_batch = plan->beams_per_batch;
    this->nstreams = plan->num_active_batches;
    this->nbatches = xdiv(total_beams, beams_per_batch);
    this->ntrees = plan->ntrees;
    this->trees = plan->trees;

    long bytes_per_elt = xdiv(dtype.nbits, 8);
    long nbits_per_segment = plan->nelts_per_segment * dtype.nbits;
    xassert_eq(nbits_per_segment, 8 * constants::bytes_per_gpu_cache_line);  // currently assumed in a few places
    
    // input_arrays: shape (nstreams, beams_per_batch, nfreq, nt_in).
    long input_nbytes = nstreams * beams_per_batch * nfreq * nt_in * bytes_per_elt;
    resource_tracker.add_gmem_footprint("input_arrays", input_nbytes, true);

    // Tree gridding kernel.
    this->tree_gridding_kernel = make_shared<GpuTreeGriddingKernel> (plan->tree_gridding_kernel_params);
    this->resource_tracker += tree_gridding_kernel->resource_tracker;

    // Lagged downsampler.
    this->lds_kernel = GpuLaggedDownsamplingKernel::make(plan->lds_params);
    this->resource_tracker += lds_kernel->resource_tracker;

    // Stage1 dedispersion buffers.
    for (long istream = 0; istream < nstreams; istream++) {
        DedispersionBuffer buf(plan->stage1_dd_buf_params);
        this->resource_tracker.add_gmem_footprint("stage1_dd_bufs", buf.footprint_nbytes, true);
        this->stage1_dd_bufs.push_back(buf);
    }

    // Stage1 dedispersion kernels.
    for (long ids = 0; ids < plan->num_downsampling_levels; ids++) {
        const DedispersionKernelParams &dd_params = plan->stage1_dd_kernel_params.at(ids);
        auto dd_kernel = make_shared<GpuDedispersionKernel> (dd_params);
        this->resource_tracker += dd_kernel->resource_tracker;
        this->stage1_dd_kernels.push_back(dd_kernel);
    }

    // MegaRingbuf.
    this->gpu_ringbuf_nelts = plan->mega_ringbuf->gpu_global_nseg * plan->nelts_per_segment;
    this->host_ringbuf_nelts = plan->mega_ringbuf->host_global_nseg * plan->nelts_per_segment;
    this->resource_tracker.add_gmem_footprint("gpu_ringbuf", gpu_ringbuf_nelts * bytes_per_elt, true);
    this->resource_tracker.add_hmem_footprint("host_ringbuf", host_ringbuf_nelts * bytes_per_elt, true);

    // MegaRingbuf copy kernels.
    this->g2g_copy_kernel = make_shared<GpuRingbufCopyKernel> (plan->g2g_copy_kernel_params);
    this->h2h_copy_kernel = make_shared<CpuRingbufCopyKernel> (plan->h2h_copy_kernel_params);
    this->resource_tracker += g2g_copy_kernel->resource_tracker;
    this->resource_tracker += h2h_copy_kernel->resource_tracker;

    // Main GPU<->host copies.
    long SB = constants::bytes_per_gpu_cache_line;
    for (const MegaRingbuf::Zone &host_zone: plan->mega_ringbuf->host_zones) {
        long nbytes = beams_per_batch * host_zone.segments_per_frame * SB;
        resource_tracker.add_memcpy_g2h("g2h", nbytes);
        resource_tracker.add_memcpy_h2g("h2g", nbytes); 
    }

    // et_host -> et_gpu (early trigger only)
    long et_nbytes = beams_per_batch * plan->mega_ringbuf->et_host_zone.segments_per_frame * SB;
    resource_tracker.add_memcpy_h2g("et_h2g", et_nbytes);

    // cdd2 kernels.
    for (long itree = 0; itree < ntrees; itree++) {
        const DedispersionKernelParams &dd_params = plan->stage2_dd_kernel_params.at(itree);
        const PeakFindingKernelParams &pf_params = plan->stage2_pf_params.at(itree);
        auto cdd2_kernel = make_shared<CoalescedDdKernel2> (dd_params, pf_params);
        this->resource_tracker += cdd2_kernel->resource_tracker;
        this->cdd2_kernels.push_back(cdd2_kernel);
    }

    // Peak-finding weight/output arrays.
    for (long itree = 0; itree < ntrees; itree++) {
        const DedispersionTree &tree = trees.at(itree);
        const vector<long> &wt_shape = cdd2_kernels.at(itree)->expected_wt_shape;
        const vector<long> &wt_strides = cdd2_kernels.at(itree)->expected_wt_strides;

        // "Extended" weight shapes with a stream axis added.
        this->extended_wt_shapes.push_back(svcat(nstreams, wt_shape));
        this->extended_wt_strides.push_back(svcat(wt_shape[0] * wt_strides[0], wt_strides));

        long wt_nbytes = nstreams * wt_shape[0] * wt_strides[0] * bytes_per_elt;
        resource_tracker.add_gmem_footprint("wt_arrays", wt_nbytes, true);

        long out_nelts = nstreams * beams_per_batch * tree.ndm_out * tree.nt_out;
        resource_tracker.add_gmem_footprint("out_max", out_nelts * bytes_per_elt, true);
        resource_tracker.add_gmem_footprint("out_argmax", out_nelts * 4, true);  // uint = 4 bytes
    }

    // XXX should I keep these asserts?
    shared_ptr<MegaRingbuf> mega_ringbuf = plan->mega_ringbuf;
    long max_clag = mega_ringbuf->max_clag;
    xassert(mega_ringbuf->host_zones.size() == uint(max_clag+1));
    xassert(mega_ringbuf->h2g_zones.size() == uint(max_clag+1));
    xassert(mega_ringbuf->g2h_zones.size() == uint(max_clag+1));
    xassert_divisible(config.beams_per_gpu, config.beams_per_batch);   // assert that length-BB copies don't "wrap"
}


void GpuDedisperser::allocate(BumpAllocator &gpu_allocator, BumpAllocator &host_allocator)
{
    if (this->is_allocated)
        throw runtime_error("double call to GpuDedisperser::allocate()");

    if (!(gpu_allocator.aflags & af_gpu))
        throw runtime_error("GpuDedisperser::allocate(): gpu_allocator.aflags must contain af_gpu");
    if (!(gpu_allocator.aflags & af_zero))
        throw runtime_error("GpuDedisperser::allocate(): gpu_allocator.aflags must contain af_zero");

    if (!(host_allocator.aflags & af_rhost))
        throw runtime_error("GpuDedisperser::allocate(): host_allocator.aflags must contain af_rhost");
    if (!(host_allocator.aflags & af_zero))
        throw runtime_error("GpuDedisperser::allocate(): host_allocator.aflags must contain af_zero");

    long gpu_nbytes_before = gpu_allocator.nbytes_allocated.load();
    long host_nbytes_before = host_allocator.nbytes_allocated.load();

    input_arrays = gpu_allocator.allocate_array<void>(dtype, {nstreams, beams_per_batch, nfreq, nt_in});

    for (DedispersionBuffer &buf: stage1_dd_bufs)
        buf.allocate(gpu_allocator);

    // wt_arrays
    for (long itree = 0; itree < ntrees; itree++) {
        const vector<long> &eshape = extended_wt_shapes.at(itree);
        const vector<long> &estrides = extended_wt_strides.at(itree);
        wt_arrays.push_back(gpu_allocator.allocate_array<void>(dtype, eshape, estrides));
    }

    // out_max, out_argmax
    for (long itree = 0; itree < ntrees; itree++) {
        const DedispersionTree &tree = trees.at(itree);
        std::initializer_list<long> shape = { nstreams, beams_per_batch, tree.ndm_out, tree.nt_out };
        out_max.push_back(gpu_allocator.allocate_array<void>(dtype, shape));
        out_argmax.push_back(gpu_allocator.allocate_array<uint>(shape));
    }

    this->tree_gridding_kernel->allocate(gpu_allocator);
    
    for (auto &kernel: this->stage1_dd_kernels)
        kernel->allocate(gpu_allocator);
    
    for (auto &kernel: this->cdd2_kernels)
        kernel->allocate(gpu_allocator);

    this->gpu_ringbuf = gpu_allocator.allocate_array<void>(dtype, { gpu_ringbuf_nelts });
    this->host_ringbuf = host_allocator.allocate_array<void>(dtype, { host_ringbuf_nelts });    

    this->lds_kernel->allocate(gpu_allocator);
    this->g2g_copy_kernel->allocate(gpu_allocator);

    long gpu_nbytes_allocated = gpu_allocator.nbytes_allocated.load() - gpu_nbytes_before;
    long host_nbytes_allocated = host_allocator.nbytes_allocated.load() - host_nbytes_before;
    // cout << "GpuDedisperser: " << gpu_nbytes_allocated << " bytes allocated on GPU" << endl;
    // cout << "GpuDedisperser: " << host_nbytes_allocated << " bytes allocated on host" << endl;
    xassert_eq(gpu_nbytes_allocated, resource_tracker.get_gmem_footprint());
    xassert_eq(host_nbytes_allocated, resource_tracker.get_hmem_footprint());

    this->is_allocated = true;
}


// ------------------------------------------------------------------------------------------


void GpuDedisperser::_launch_tree_gridding(long ichunk, long ibatch, cudaStream_t stream)
{
    long istream = (ichunk * nbatches + ibatch) % nstreams;
    Array<void> &dd_buf0 = stage1_dd_bufs.at(istream).bufs.at(0);
    tree_gridding_kernel->launch(dd_buf0, input_arrays.slice(0,istream), stream);
}


void GpuDedisperser::_launch_lagged_downsampler(long ichunk, long ibatch, cudaStream_t stream)
{
    long istream = (ichunk * nbatches + ibatch) % nstreams;
    lds_kernel->launch(stage1_dd_bufs.at(istream), ichunk, ibatch, stream);
}


void GpuDedisperser::_launch_dd_stage1(long ichunk, long ibatch, cudaStream_t stream)
{
    long istream = (ichunk * nbatches + ibatch) % nstreams;

    for (long ids = 0; ids < plan->num_downsampling_levels; ids++) {
        shared_ptr<GpuDedispersionKernel> kernel = stage1_dd_kernels.at(ids);
        const DedispersionKernelParams &kp = kernel->params;
        Array<void> dd_buf = stage1_dd_bufs.at(istream).bufs.at(ids);

        // See comments in DedispersionKernel.hpp for an explanation of this reshape operation.
        dd_buf = dd_buf.reshape({ kp.beams_per_batch, pow2(kp.amb_rank), pow2(kp.dd_rank), kp.ntime });
        kernel->launch(dd_buf, this->gpu_ringbuf, ichunk, ibatch, stream);
    }
}


void GpuDedisperser::_launch_et_g2g(long ichunk, long ibatch, cudaStream_t stream)
{
    // copies from 'gpu' zones to 'g2h' zones
    this->g2g_copy_kernel->launch(this->gpu_ringbuf, ichunk, ibatch, stream);
}


void GpuDedisperser::_do_et_h2h(long ichunk, long ibatch)
{
    // copy host -> et_host
    // FIXME: in principle this is a bug: running copy kernel without synchronizing
    // wiuth gpu->host copy that produces its input data, or host->gpu copy that
    // consumes its output data. The current unit tests don't detect this!
    this->h2h_copy_kernel->apply(this->host_ringbuf, ichunk, ibatch);
}


void GpuDedisperser::_launch_et_h2g(long ichunk, long ibatch, cudaStream_t stream)
{
    const long BT = this->config.beams_per_gpu;            // total beams
    const long BB = this->config.beams_per_batch;          // beams per batch
    const long BA = this->config.num_active_batches * BB;  // active beams
    const long SB = constants::bytes_per_gpu_cache_line;   // bytes per segment
    const long iframe = (ichunk * BT) + (ibatch * BB);

    MegaRingbuf::Zone &eth_zone = plan->mega_ringbuf->et_host_zone;
    MegaRingbuf::Zone &etg_zone = plan->mega_ringbuf->et_gpu_zone;
    
    xassert(eth_zone.segments_per_frame == etg_zone.segments_per_frame);
    xassert(eth_zone.num_frames == BA);
    xassert(etg_zone.num_frames == BA);

    long soff = eth_zone.segment_offset_of_frame(iframe);
    long doff = etg_zone.segment_offset_of_frame(iframe);
    char *src = (char *) this->host_ringbuf.data + (soff * SB);
    char *dst = (char *) this->gpu_ringbuf.data + (doff * SB);
    long nbytes = beams_per_batch * etg_zone.segments_per_frame * SB;
    CUDA_CALL(cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyHostToDevice, stream));
}


void GpuDedisperser::_launch_g2h(long ichunk, long ibatch, cudaStream_t stream)
{    
    const long BT = this->config.beams_per_gpu;            // total beams
    const long BB = this->config.beams_per_batch;          // beams per batch
    const long BA = this->config.num_active_batches * BB;  // active beams
    const long SB = constants::bytes_per_gpu_cache_line;   // bytes per segment
    const long iframe = (ichunk * BT) + (ibatch * BB);

    for (int clag = 0; clag <= plan->mega_ringbuf->max_clag; clag++) {
        MegaRingbuf::Zone &host_zone = plan->mega_ringbuf->host_zones.at(clag);
        MegaRingbuf::Zone &g2h_zone = plan->mega_ringbuf->g2h_zones.at(clag);

        xassert(host_zone.segments_per_frame == g2h_zone.segments_per_frame);
        xassert(host_zone.num_frames == clag*BT + BA);
        xassert(g2h_zone.num_frames == BA);

        if (host_zone.segments_per_frame > 0) {
            long soff = g2h_zone.segment_offset_of_frame(iframe);
            long doff = host_zone.segment_offset_of_frame(iframe);
            char *src = reinterpret_cast<char *> (this->gpu_ringbuf.data) + (soff * SB);
            char *dst = reinterpret_cast<char *> (this->host_ringbuf.data) + (doff * SB);
            long nbytes = beams_per_batch * host_zone.segments_per_frame * SB;
            CUDA_CALL(cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyDeviceToHost, stream));
        }
    }
}


// XXX clean up cut-and-paste with _launch_g2h().
void GpuDedisperser::_launch_h2g(long ichunk, long ibatch, cudaStream_t stream)
{
    const long BT = this->config.beams_per_gpu;            // total beams
    const long BB = this->config.beams_per_batch;          // beams per batch
    const long BA = this->config.num_active_batches * BB;  // active beams
    const long SB = constants::bytes_per_gpu_cache_line;   // bytes per segment
    const long iframe = (ichunk * BT) + (ibatch * BB);
    const long max_clag = plan->mega_ringbuf->max_clag;

    for (int clag = 0; clag <= plan->mega_ringbuf->max_clag; clag++) {
        MegaRingbuf::Zone &host_zone = plan->mega_ringbuf->host_zones.at(clag);
        MegaRingbuf::Zone &h2g_zone = plan->mega_ringbuf->h2g_zones.at(clag);

        xassert(host_zone.segments_per_frame == h2g_zone.segments_per_frame);
        xassert(host_zone.num_frames == clag*BT + BA);
        xassert(h2g_zone.num_frames == BA);

        if (host_zone.segments_per_frame > 0) {
            long soff = host_zone.segment_offset_of_frame(iframe - clag*BT);
            long doff = h2g_zone.segment_offset_of_frame(iframe);
            char *src = reinterpret_cast<char *> (this->host_ringbuf.data) + (soff * SB);
            char *dst = reinterpret_cast<char *> (this->gpu_ringbuf.data) + (doff * SB);
            long nbytes = beams_per_batch * host_zone.segments_per_frame * SB;
            CUDA_CALL(cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyHostToDevice, stream));
        }
    }
}


void GpuDedisperser::_launch_cdd2(long ichunk, long ibatch, cudaStream_t stream)
{
    long istream = (ichunk * nbatches + ibatch) % nstreams;

    for (long itree = 0; itree < ntrees; itree++) {
        Array<void> slice_max = out_max.at(itree).slice(0,istream);
        Array<uint> slice_argmax = out_argmax.at(itree).slice(0,istream);
        Array<void> slice_wt = wt_arrays.at(itree).slice(0,istream);

        shared_ptr<CoalescedDdKernel2> cdd2_kernel = cdd2_kernels.at(itree);
        cdd2_kernel->launch(slice_max, slice_argmax, this->gpu_ringbuf, slice_wt, ichunk, ibatch, stream);
    }
}


void GpuDedisperser::launch(long ichunk, long ibatch)
{
    xassert(ichunk >= 0);
    xassert((ibatch >= 0) && (ibatch < nbatches));
    xassert(is_allocated);
    
    // XXX temporary kludge: uses default stream for all kernels/copies!!!
    _launch_tree_gridding(ichunk, ibatch, nullptr);
    _launch_lagged_downsampler(ichunk, ibatch, nullptr);
    _launch_dd_stage1(ichunk, ibatch, nullptr);
    _launch_et_g2g(ichunk, ibatch, nullptr);
    _do_et_h2h(ichunk, ibatch);
    _launch_et_h2g(ichunk, ibatch, nullptr);
    _launch_g2h(ichunk, ibatch, nullptr);
    _launch_h2g(ichunk, ibatch, nullptr);
    _launch_cdd2(ichunk, ibatch, nullptr);
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
    config.emit_cpp();

    cout << "    nchunks = " << nchunks << ";\n"
         << "    host_only = " << host_only << ";" << endl;
    
    if (host_only)
         cout << "    !!! Host-only test, GPU code will not be run !!!" << endl;

    // I decided that this was the least awkward place to call DedispersionConfig::test().    
    config.test();   // calls DedispersionConfig::validate()

    shared_ptr<DedispersionPlan> plan = make_shared<DedispersionPlan> (config);

    long ntrees = plan->ntrees;
    long beams_per_batch = plan->beams_per_batch;
    long nbatches = xdiv(plan->beams_per_gpu, plan->beams_per_batch);
    long nstreams = plan->num_active_batches;

    // FIXME test multi-stream logic in the future.
    // For now, we use the default cuda stream, which simplifies things since we can
    // freely mix operations such as Array::to_gpu() which use the default stream.
    xassert(nstreams == 1);
    
    shared_ptr<GpuDedisperser> gdd;

    if (!host_only) {
        gdd = make_shared<GpuDedisperser> (plan);
        BumpAllocator gpu_allocator(af_gpu | af_zero, -1);     // dummy allocator
        BumpAllocator host_allocator(af_rhost | af_zero, -1);  // dummy allocator
        gdd->allocate(gpu_allocator, host_allocator);
    }

    // Dcore: taken from GPU kernel, passed to reference kernel
    vector<long> Dcore(ntrees);
    for (long itree = 0; itree < ntrees; itree++) {
        const DedispersionTree &tree = plan->trees.at(itree);
        Dcore.at(itree) = host_only ? tree.pf.time_downsampling : gdd->cdd2_kernels.at(itree)->Dcore;
    }

    // pf_tmp: used to store output from ReferencePeakFindingKernel::eval_tokens().
    vector<Array<float>> pf_tmp(ntrees);
    for (long itree = 0; itree < ntrees; itree++) {
        const DedispersionTree &tree = plan->trees.at(itree);
        pf_tmp.at(itree) = Array<float> ({beams_per_batch, tree.ndm_out, tree.nt_out}, af_uhost | af_zero);
    }

    // subband_variances: used in ReferencePeakFindingKernel::make_random_weights().
    vector<Array<float>> subband_variances(ntrees);
    for (long itree = 0; itree < ntrees; itree++) {
        long F = plan->trees.at(itree).frequency_subbands.F;
        subband_variances.at(itree) = Array<float> ({F}, af_uhost | af_zero);
        for (long f = 0; f < F; f++)
            subband_variances.at(itree).at({f}) = variance_upper_bound(plan, itree, f);
    }

    // ref_kernels_for_weights: only used for ReferencePeakFindingKernel::make_random_weights().
    vector<shared_ptr<ReferencePeakFindingKernel>> ref_kernels_for_weights(ntrees);
    for (long itree = 0; itree < ntrees; itree++) {
        const PeakFindingKernelParams &pf_params = plan->stage2_pf_params.at(itree);
        ref_kernels_for_weights.at(itree) = make_shared<ReferencePeakFindingKernel> (pf_params, Dcore.at(itree));
    }

    // Create ReferenceDedispersers (must come after Dcore initialization)
    shared_ptr<ReferenceDedisperserBase> rdd0 = ReferenceDedisperserBase::make(plan, Dcore, 0);
    shared_ptr<ReferenceDedisperserBase> rdd1 = ReferenceDedisperserBase::make(plan, Dcore, 1);
    shared_ptr<ReferenceDedisperserBase> rdd2 = ReferenceDedisperserBase::make(plan, Dcore, 2);

    for (int ichunk = 0; ichunk < nchunks; ichunk++) {
        for (int ibatch = 0; ibatch < nbatches; ibatch++) {
            // Randomly initialize weights.
            for (int itree = 0; itree < ntrees; itree++) {
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
            Array<float> arr({beams_per_batch, plan->nfreq, plan->nt_in}, af_uhost);
            for (long i = 0; i < arr.size; i++)
                arr.data[i] = ksgpu::rand_uniform(-1.0, 1.0);

            rdd0->input_array.fill(arr);
            rdd0->dedisperse(ichunk, ibatch);  // (ichunk, ibatch)

            rdd1->input_array.fill(arr);
            rdd1->dedisperse(ichunk, ibatch);  // (ichunk, ibatch)

            rdd2->input_array.fill(arr);
            rdd2->dedisperse(ichunk, ibatch);  // (ichunk, ibatch)

            if (!host_only) {
                gdd->input_arrays.slice(0,0).fill(arr.convert(config.dtype));  // istream=0
                gdd->launch(ichunk, ibatch);
            }
            
            for (int itree = 0; itree < ntrees; itree++) {
                // Compare peak-finding 'out_max'.
                Array<void> gdd_max = gdd->out_max.at(itree).slice(0,0);  // FIXME istream=0 assumed
                assert_arrays_equal(rdd0->out_max.at(itree), rdd1->out_max.at(itree), "pfmax_ref0", "pfmax_ref1", {"beam","pfdm","pft"});
                assert_arrays_equal(rdd0->out_max.at(itree), rdd2->out_max.at(itree), "pfmax_ref0", "pfmax_ref2", {"beam","pfdm","pft"});
                assert_arrays_equal(rdd0->out_max.at(itree), gdd_max, "pfmax_ref0", "pfmax_gpu", {"beam","pfdm","pft"});

                // To check 'out_argmax', we need to jump through some hoops.
                shared_ptr<ReferencePeakFindingKernel> pf_kernel = rdd0->pf_kernels.at(itree);

                pf_kernel->eval_tokens(pf_tmp.at(itree), rdd1->out_argmax.at(itree), rdd0->wt_arrays.at(itree));
                assert_arrays_equal(rdd0->out_max.at(itree), pf_tmp.at(itree), "pfmax_ref0", "pf_tmp_ref1", {"beam","pfdm","pft"});

                pf_kernel->eval_tokens(pf_tmp.at(itree), rdd2->out_argmax.at(itree), rdd0->wt_arrays.at(itree));
                assert_arrays_equal(rdd0->out_max.at(itree), pf_tmp.at(itree), "pfmax_ref0", "pf_tmp_ref2", {"beam","pfdm","pft"});

                double eps = 5.0 * config.dtype.precision();
                Array<uint> gpu_tokens = gdd->out_argmax.at(itree).slice(0,0).to_host();  // FIXME istream=0 assumed 
                pf_kernel->eval_tokens(pf_tmp.at(itree), gpu_tokens, rdd0->wt_arrays.at(itree));
                assert_arrays_equal(rdd0->out_max.at(itree), pf_tmp.at(itree), "pfmax_ref0", "pf_tmp_gpu", {"beam","pfdm","pft"}, eps, eps);
            }
        }
    }
    
    cout << endl;
}

// Static member function.
void GpuDedisperser::test_random()
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
