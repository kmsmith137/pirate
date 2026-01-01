#include "../include/pirate/Dedisperser.hpp"
#include "../include/pirate/BumpAllocator.hpp"
#include "../include/pirate/CudaStreamPool.hpp"
#include "../include/pirate/CudaEventRingbuf.hpp"
#include "../include/pirate/DedispersionPlan.hpp"
#include "../include/pirate/DedispersionConfig.hpp"
#include "../include/pirate/CoalescedDdKernel2.hpp"
#include "../include/pirate/RingbufCopyKernel.hpp"
#include "../include/pirate/TreeGriddingKernel.hpp"
#include "../include/pirate/MegaRingbuf.hpp"
#include "../include/pirate/GpuDequantizationKernel.hpp"
#include "../include/pirate/constants.hpp"  // xdiv(), pow2()
#include "../include/pirate/inlines.hpp"  // xdiv(), pow2()

#include <ksgpu/rand_utils.hpp>
#include <ksgpu/test_utils.hpp>
#include <ksgpu/KernelTimer.hpp>

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


GpuDedisperser::GpuDedisperser(const GpuDedisperser::Params &params_) :
    params(params_)
{
    xassert(params.plan);
    xassert(params.stream_pool);
    xassert_eq(params.plan->num_active_batches, params.stream_pool->num_compute_streams);

    this->plan = params.plan;
    this->stream_pool = params.stream_pool;
    this->mega_ringbuf = plan->mega_ringbuf;
    this->config = plan->config;

    // There's some cut-and-paste between this constructor and the ReferenceDedisperser
    // constructor, but not enough to bother defining a common base class.

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

    // The kernel-launching code makes assumptions about the MegaRingbuf buffer sizes.
    // List all of these assumptions in one place, and check them.

    const long BT = total_beams;                 // total beams
    const long BA = nstreams * beams_per_batch;  // active beams

    shared_ptr<MegaRingbuf> mega_ringbuf = plan->mega_ringbuf;
    long max_clag = mega_ringbuf->max_clag;

    xassert_divisible(config.beams_per_gpu, config.beams_per_batch);   // assert that length-BB copies don't "wrap"
    xassert(mega_ringbuf->gpu_zones.size() == uint(max_clag+1));
    xassert(mega_ringbuf->host_zones.size() == uint(max_clag+1));
    xassert(mega_ringbuf->h2g_zones.size() == uint(max_clag+1));
    xassert(mega_ringbuf->g2h_zones.size() == uint(max_clag+1));

    for (int clag = 0; clag <= mega_ringbuf->max_clag; clag++) {
        MegaRingbuf::Zone &host_zone = mega_ringbuf->host_zones.at(clag);
        MegaRingbuf::Zone &g2h_zone = mega_ringbuf->g2h_zones.at(clag);
        MegaRingbuf::Zone &h2g_zone = mega_ringbuf->h2g_zones.at(clag);

        xassert(host_zone.segments_per_frame == g2h_zone.segments_per_frame);
        xassert(host_zone.segments_per_frame == h2g_zone.segments_per_frame);

        xassert(host_zone.num_frames == clag*BT + BA);
        xassert(g2h_zone.num_frames == BA);
        xassert(h2g_zone.num_frames == BA);
    }

    MegaRingbuf::Zone &eth_zone = mega_ringbuf->et_host_zone;
    MegaRingbuf::Zone &etg_zone = mega_ringbuf->et_gpu_zone;

    xassert(eth_zone.segments_per_frame == etg_zone.segments_per_frame);
    xassert(eth_zone.num_frames == BA);
    xassert(etg_zone.num_frames == BA);

    // Initialize CudaEventRingbufs.
    //
    //   tg: consumers=[acq_input]
    //   g2g: consumers=[g2h]
    //   g2h: consumers=[dd1,h2g,et_h2h]
    //   h2g: consumers=[g2h,cdd2]
    //   cdd2: consumers=[tg,h2g,et_h2g,acq_input,acq_output]
    //   et_h2g: consumers=[et_h2h,cdd2]   (*)
    //   output: consumers=[cdd2]
    //
    // (*) Note that the g2h code waits on et_h2g, via CudaEventRingbuf::synchronize_with_producer(),
    //     but this doesn't count as a "consumer" of the et_h2g ringbuf.

    long tg_nconsumers = params.fixed_weights ? 1 : 0;
    long cdd2_nconsumers = params.fixed_weights ? 4 : 5;

    // FIXME: using huge CudaEventRingbuf capacity for now!
    // Not sure if this will work, but let's try.
    long capacity = 1000; // XXX host_seq_lag + et_seq_headroom + et_seq_lag + (3 * nstreams);

    this->evrb_tree_gridding = make_shared<CudaEventRingbuf> ("tree_gridding", tg_nconsumers, capacity);
    this->evrb_g2g = make_shared<CudaEventRingbuf> ("g2g", 1, capacity);
    this->evrb_g2h = make_shared<CudaEventRingbuf> ("g2h", 3, capacity);
    this->evrb_h2g = make_shared<CudaEventRingbuf> ("h2g", 2, capacity);
    this->evrb_cdd2 = make_shared<CudaEventRingbuf> ("cdd2", cdd2_nconsumers, capacity);
    this->evrb_et_h2g = make_shared<CudaEventRingbuf> ("et_h2g", 2, capacity);
    this->evrb_output = make_shared<CudaEventRingbuf> ("output", 1, capacity);

    bool has_host_ringbuf = (mega_ringbuf->host_global_nseg > 0);
    bool has_early_triggers = (mega_ringbuf->et_host_zone.segments_per_frame > 0);

    // These members help keep track of lags between kernels. See later in this source file for usage.
    // In cases where a placeholder values is needed, we use (2*nstreams).

    this->host_seq_lag = has_host_ringbuf ? (mega_ringbuf->min_host_clag * nbatches) : (2*nstreams);
    this->et_seq_headroom = has_early_triggers ? (mega_ringbuf->min_et_headroom * nbatches) : (2*nstreams);
    this->et_seq_lag = has_early_triggers ? (mega_ringbuf->min_et_clag * nbatches) : (2*nstreams);

    // Note special "prefetch" logic for h2g copies!!
    // I can't decide if this way of doing it is elegant or a hack :)
    //
    // When gpu kernels are queued (in GpuDedisperser::release_input()),
    // instead of queueing h2g(seq_id), we queue h2g(seq_id + nstreams).
    // This should improve throughput, by "prefetching" the h2g data for
    // the next batch of gpu kernels.

    // For h2g prefetching to work, the following assert must be satisfied.
    xassert_ge(host_seq_lag, nstreams);

    // To implement h2g prefetching, we pretend that the first (nstreams) batches
    // of h2g data have already been copied, by calling evrb_h2g.record(). Note
    // that this data is all-zeroes anyway, by the previous assert.
    for (long seq_id = 0; seq_id < nstreams; seq_id++) {
        cudaStream_t s = stream_pool->low_priority_h2g_stream;
        evrb_h2g->record(s, seq_id);
    }

    // Note that we don't implement any sort of prefetching for the et_h2g
    // copy, since it's launched asychronously in a worker thread anyway.
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

    // Create worker thread (thread-backed class pattern).
    this->worker = std::thread(&GpuDedisperser::worker_main, this);
}


GpuDedisperser::~GpuDedisperser()
{
    // Stop the worker thread.
    this->stop();

    // Stop all CudaEventRingbufs before joining the worker thread.
    // This ensures that any blocking waits in the worker thread will unblock.
    if (evrb_tree_gridding) evrb_tree_gridding->stop();
    if (evrb_g2g) evrb_g2g->stop();
    if (evrb_g2h) evrb_g2h->stop();
    if (evrb_h2g) evrb_h2g->stop();
    if (evrb_cdd2) evrb_cdd2->stop();
    if (evrb_et_h2g) evrb_et_h2g->stop();
    if (evrb_output) evrb_output->stop();

    // Join the worker thread.
    if (worker.joinable())
        worker.join();

    // Synchronize all streams in stream_pool after joining the worker thread.
    // This ensures that before GPU arrays are freed (in subsequent destructors),
    // all kernels that perform IO on those arrays have completed. Without this
    // synchronization, we would have a race condition where array memory could
    // be freed while GPU kernels are still accessing it.
    if (stream_pool) {
        cudaStreamSynchronize(stream_pool->low_priority_g2h_stream);
        cudaStreamSynchronize(stream_pool->low_priority_h2g_stream);
        cudaStreamSynchronize(stream_pool->high_priority_g2h_stream);
        cudaStreamSynchronize(stream_pool->high_priority_h2g_stream);
        for (auto &s : stream_pool->compute_streams)
            cudaStreamSynchronize(s);
    }
}


void GpuDedisperser::stop(std::exception_ptr e)
{
    std::lock_guard<std::mutex> lock(mutex);
    if (is_stopped) return;
    is_stopped = true;
    error = e;

    // GpuDedisperser doesn't currently need a condition_variable.
    // See comment in Dedisperser.hpp.
    // cv.notify_all();
}


void GpuDedisperser::_throw_if_stopped(const char *method_name)
{
    // Caller must hold mutex.
    if (error)
        std::rethrow_exception(error);

    if (is_stopped) {
        throw std::runtime_error(std::string("GpuDedisperser::") + method_name + " called on stopped instance");
    }
}


void GpuDedisperser::worker_main()
{
    try {
        _worker_main();  // only returns if GpuDedisperser::is_stopped
    } catch (...) {
        stop(std::current_exception());
    }
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
    const long SB = constants::bytes_per_gpu_cache_line;   // bytes per segment
    const long iframe = (ichunk * total_beams) + (ibatch * beams_per_batch);

    MegaRingbuf::Zone &eth_zone = plan->mega_ringbuf->et_host_zone;
    MegaRingbuf::Zone &etg_zone = plan->mega_ringbuf->et_gpu_zone;

    // Note: there are some relevant asserts in the GpuDedisperser constructor.
    long soff = eth_zone.segment_offset_of_frame(iframe);
    long doff = etg_zone.segment_offset_of_frame(iframe);
    char *src = (char *) this->host_ringbuf.data + (soff * SB);
    char *dst = (char *) this->gpu_ringbuf.data + (doff * SB);
    long nbytes = beams_per_batch * etg_zone.segments_per_frame * SB;
    CUDA_CALL(cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyHostToDevice, stream));
}


void GpuDedisperser::_launch_g2h(long ichunk, long ibatch, cudaStream_t stream)
{    
    const long SB = constants::bytes_per_gpu_cache_line;   // bytes per segment
    const long iframe = (ichunk * total_beams) + (ibatch * beams_per_batch);

    for (int clag = 0; clag <= plan->mega_ringbuf->max_clag; clag++) {
        // Note: there are some relevant asserts in the GpuDedisperser constructor.
        MegaRingbuf::Zone &host_zone = plan->mega_ringbuf->host_zones.at(clag);
        MegaRingbuf::Zone &g2h_zone = plan->mega_ringbuf->g2h_zones.at(clag);

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


void GpuDedisperser::_launch_h2g(long ichunk, long ibatch, cudaStream_t stream)
{
    const long SB = constants::bytes_per_gpu_cache_line;   // bytes per segment
    const long iframe = (ichunk * total_beams) + (ibatch * beams_per_batch);

    for (int clag = 0; clag <= plan->mega_ringbuf->max_clag; clag++) {
        // Note: there are some relevant asserts in the GpuDedisperser constructor.
        MegaRingbuf::Zone &host_zone = plan->mega_ringbuf->host_zones.at(clag);
        MegaRingbuf::Zone &h2g_zone = plan->mega_ringbuf->h2g_zones.at(clag);

        if (host_zone.segments_per_frame > 0) {
            long soff = host_zone.segment_offset_of_frame(iframe -  clag * total_beams);
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


// -------------------------------------------------------------------------------------------------


Array<void> GpuDedisperser::acquire_input(long ichunk, long ibatch, cudaStream_t stream)
{
    xassert(ichunk >= 0);
    xassert((ibatch >= 0) && (ibatch < nbatches));
    xassert(is_allocated);

    std::unique_lock<std::mutex> lock(mutex);
    _throw_if_stopped("acquire_input");

    if ((ichunk != curr_input_ichunk) || (ibatch != curr_input_ibatch)) {
        stringstream ss;
        ss << "GpuDedisperser::acquire_input(): expected (ichunk,ibatch)=(" 
           << curr_input_ichunk << "," << curr_input_ibatch << "), got ("
           << ichunk << "," << ibatch << ")";
        throw runtime_error(ss.str());
    }

    if (curr_input_acquired) {
        stringstream ss;
        ss << "GpuDedisperser::acquire_input(): double call to acquire_input()"
           << " with (ichunk,ibatch)=(" << ichunk << "," << ibatch << ")";
        throw runtime_error(ss.str());
    }

    if (params.detect_deadlocks) {
        long input_seq_id = ichunk * nbatches + ibatch;
        long output_seq_id = curr_output_ichunk * nbatches + curr_output_ibatch;

        if (input_seq_id >= output_seq_id + nstreams) {
            throw runtime_error("GpuDedisperser: deadlock detected (calls to acquire_input()"
                " are too far ahead of calls to release_output(). If the input/output arrays"
                " are handled in different threads, then this error is a false alarm, and you"
                " can suppress it by setting GpuDedisperser::Params::detect_deadlocks = false.");
        }
    }

    curr_input_acquired = true;
    lock.unlock();

    // Argument-checking ends here. If an exception is thrown below, call stop().
    // If fixed_weights=true, then 'stream' should wait for the tree gridding
    // kernel (consumer of input array). If fixed_weights=false, then 'stream'
    // should wait for cdd2 (consumer of pf_weights array).

    try {
        shared_ptr<CudaEventRingbuf> &evrb = params.fixed_weights ? evrb_tree_gridding : evrb_cdd2;

        // This call to wait() can be nonblocking, since we know that the tree_gridding/cdd2
        // kernel was successfully launched by a previous call to release_input().
        long seq_id = ichunk * nbatches + ibatch;
        evrb->wait(stream, seq_id - nstreams);

        // Return input array.
        long istream = seq_id % nstreams;
        return input_arrays.slice(0, istream);
    } catch (...) {
        stop(std::current_exception());
        throw;
    }
}


void GpuDedisperser::release_input(long ichunk, long ibatch, cudaStream_t stream)
{
    xassert(ichunk >= 0);
    xassert((ibatch >= 0) && (ibatch < nbatches));

    std::unique_lock<std::mutex> lock(mutex);
    _throw_if_stopped("release_input");

    if ((ichunk != curr_input_ichunk) || (ibatch != curr_input_ibatch)) {
        stringstream ss;
        ss << "GpuDedisperser::release_input(): expected (ichunk,ibatch)=(" 
           << curr_input_ichunk << "," << curr_input_ibatch << "), got ("
           << ichunk << "," << ibatch << ")";
        throw runtime_error(ss.str());
    }

    if (!curr_input_acquired) {
        stringstream ss;
        ss << "GpuDedisperser::acquire_input(): release_input() called without "
           << " acquire_input(), (ichunk,ibatch)=(" << ichunk << "," << ibatch << ")";
        throw runtime_error(ss.str());
    }

    curr_input_ibatch++;
    curr_input_acquired = false;

    if (curr_input_ibatch == nbatches) {
        curr_input_ichunk++;
        curr_input_ibatch = 0;
    }

    lock.unlock();

    // Argument-checking ends here. If an exception is thrown below, call stop().
    // The rest of release_input() is in its own method _launch_dedispersion_kernels().
    try {
        _launch_dedispersion_kernels(ichunk, ibatch, stream);
    } catch (...) {
        stop(std::current_exception());
        throw;
    }
}


void GpuDedisperser::acquire_output(long ichunk, long ibatch, cudaStream_t stream)
{
    xassert(ichunk >= 0);
    xassert((ibatch >= 0) && (ibatch < nbatches));
    xassert(is_allocated);

    std::unique_lock<std::mutex> lock(mutex);
    _throw_if_stopped("acquire_output");

    if ((ichunk != curr_output_ichunk) || (ibatch != curr_output_ibatch)) {
        stringstream ss;
        ss << "GpuDedisperser::acquire_output(): expected (ichunk,ibatch)=(" 
           << curr_output_ichunk << "," << curr_output_ibatch << "), got ("
           << ichunk << "," << ibatch << ")";
        throw runtime_error(ss.str());
    }

    if (curr_output_acquired) {
        stringstream ss;
        ss << "GpuDedisperser::acquire_output(): double call to acquire_output()"
           << " with (ichunk,ibatch)=(" << ichunk << "," << ibatch << ")";
        throw runtime_error(ss.str());
    }

    if (params.detect_deadlocks) {
        long input_seq_id = curr_input_ichunk * nbatches + curr_input_ibatch;
        long output_seq_id = ichunk * nbatches + ibatch;

        if (output_seq_id >= input_seq_id) {
            throw runtime_error("GpuDedisperser: deadlock detected (calls to acquire_output()"
                " are ahead of calls to release_input(). If the input/output arrays"
                " are handled in different threads, then this error is a false alarm, and you"
                " can suppress it by setting GpuDedisperser::Params::detect_deadlocks = false.");
        }
    }

    curr_output_acquired = true;
    lock.unlock();

    // Argument checking ends here. If an exception is thrown below, call stop().
    // The caller-specified stream waits for 'cdd2' to produce outputs.
    try {
        long seq_id = ichunk * nbatches + ibatch;
        bool blocking = !params.detect_deadlocks;
        evrb_cdd2->wait(stream, seq_id, blocking);
    } catch (...) {
        stop(std::current_exception());
        throw;
    }
}


void GpuDedisperser::release_output(long ichunk, long ibatch, cudaStream_t stream)
{
    xassert(ichunk >= 0);
    xassert((ibatch >= 0) && (ibatch < nbatches));

    long seq_id = ichunk * nbatches + ibatch;

    std::unique_lock<std::mutex> lock(mutex);
    _throw_if_stopped("release_output");

    if ((ichunk != curr_output_ichunk) || (ibatch != curr_output_ibatch)) {
        stringstream ss;
        ss << "GpuDedisperser::release_output(): expected (ichunk,ibatch)=(" 
           << curr_output_ichunk << "," << curr_output_ibatch << "), got ("
           << ichunk << "," << ibatch << ")";
        throw runtime_error(ss.str());
    }

    if (!curr_output_acquired) {
        stringstream ss;
        ss << "GpuDedisperser::acquire_output(): release_output() called without preceding"
           << " acquire_output(), (ichunk,ibatch)=(" << ichunk << "," << ibatch << ")";
        throw runtime_error(ss.str());
    }

    curr_output_ibatch++;
    curr_output_acquired = false;

    if (curr_output_ibatch == nbatches) {
        curr_output_ichunk++;
        curr_output_ibatch = 0;
    }

    lock.unlock();
    
    // Argument-checking ends here. If an exception is thrown below, call stop().
    // We record an event from the caller-specified stream, and put it in 'evrb_output'
    // (a CudaEventRingbuf). The cdd2 kernel will wait for this event later.
    try {
        evrb_output->record(stream, seq_id);
    } catch (...) {
        stop(std::current_exception());
        throw;
    }
}


// ------------------------------------------------------------------------------------
//
// Difficult code starts here!
//
// Dependency graph:
//
//  - 6 MegaRingbuf zones: gpu, host, g2h, h2g, et_host, et_gpu
//
//  - 5+2 MegaRingbuf-adjacent kernels: dd1, g2g, g2h, h2g, cdd2, et_h2h, et_h2g
//
//  - Per-kernel inbufs and outbufs:
//
//      dd1 kernel:     inbufs []                 outbufs [gpu,g2h]
//      g2g kernel:     inbufs [gpu]              outbufs [g2h]
//      g2h kernel:     inbufs [g2h]              outbufs [host]
//      h2g kernel:     inbufs [host]             outbufs [h2g]
//      cdd2 kernel:    inbufs [gpu,h2g,et_gpu]   outbufs []
//      et_h2h kernel:  inbufs [host]             outbufs [et_host]
//      et_h2g kernel:  inbufs [et_host]          outbufs [et_gpu]
//
//  - Per-ringbuf producers and consumers, produced by a mechanical process:
//
//      gpu ringbuf:      producers [dd1]      consumers [g2g,cdd2]
//      host ringbuf:     producers [g2h]      consumers [h2g,et_h2h]
//      g2h ringbuf:      producers [dd1,g2g]  consumers [g2h]
//      h2g ringbuf:      producers [h2g]      consumers [cdd2]
//      et_host ringbuf:  producers [et_h2h]   consumers [et_h2g]
//      et_gpu ringbuf:   producers [et_h2g]   consumers [cdd2]
//
//  - Dependency analysis, produced by very mechanical cut-and-paste:
//
//      dd1 kernel: inbufs=[], outbufs=[gpu,g2h]
//        gpu outbuf: consumers=[g2g,cdd2]
//        g2h outbuf: consumers=[g2h]
//
//      g2g kernel: inbufs=[gpu], outbufs=[g2h]
//        gpu ringbuf: producers=[dd1]
//        g2h ringbuf: consumers=[g2h]
//
//      g2h kernel: inbufs=[g2h], outbufs=[host]
//        g2h ringbuf:  producers=[dd1,g2g]  
//        host ringbuf: consumers=[h2g,et_h2h]
//
//      h2g kernel: inbufs=[host], outbufs=[h2g]
//        host ringbuf: producers=[g2h]
//        h2g ringbuf:  consumers=[cdd2]
//
//      cdd2 kernel: inbufs=[gpu,h2g,et_gpu], outbufs=[]
//        gpu ringbuf:    producers=[dd1]
//        h2g ringbuf:    producers=[h2g]
//        et_gpu ringbuf: producers=[et_h2g]
//
//      et_h2h kernel: inbufs=[host], outbufs=[et_host]
//        host ringbuf:     producers=[g2h]
//        et_host ringbuf:  consumers=[et_h2g]
//
//      et_h2g kernel: inbufs=[et_host], outbufs=[et_gpu]
//        et_host ringbuf:  producers=[et_h2h]
//        et_gpu ringbuf:   consumers=[cdd2]
//
// FIXME now that the dust has settled, this synchronization logic is mechanical
// enough that I could capture it in a KernelGraph helper class, which keeps track
// of lagged dependencies between kernels. A KernelGraph::Node could represent
// a kernel, and contain shared_ptr<CudaEventRingbuf. A KernelGraph::Edge could
// represent a kernel dependency with lags. (There may also be a KernelGraph::Buffer
// used temporarily when building the graph.) Note that initializing CudaEventRingbuf
// capacities may be a sticking point. Don't forget KernelGraph::to_yaml()!
//
// Note that this code always synchronizes with g2h/h2g streams, and always spawns
// an early trigger thread, even in simple cases where there are no host buffers
// (or no early triggers). This small inefficiency is something that a KernelGraph
// could fix.
//
// Note that the synchronization between main thread and et_h2h thread
// is entirely via CudaEventRingbufs:
//
//   - evrb_g2h: produced in main thread, consumed in worker
//   - evrb_cdd2: produced in main thread, consumed in worker
//   - evrb_et_h2g: produced in worker, consumed in main thread


void GpuDedisperser::_launch_dedispersion_kernels(long ichunk, long ibatch, cudaStream_t stream)
{
    // Argument-checking has already been done in release_input().
    long seq_id = ichunk * nbatches + ibatch;
    cudaStream_t g2h_stream = stream_pool->low_priority_g2h_stream;
    cudaStream_t h2g_stream = stream_pool->low_priority_h2g_stream;
    cudaStream_t compute_stream = stream_pool->compute_streams.at(seq_id % nstreams);

    // Compute kernel waits on the caller-specified stream.
    cudaEvent_t event;
    CUDA_CALL(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    CUDA_CALL(cudaEventRecord(event, stream));
    CUDA_CALL(cudaStreamWaitEvent(compute_stream, event, 0));
    CUDA_CALL(cudaEventDestroy(event));

    // This call to wait() ensures that all pstate is up-to-date.
    // (Note that the the previous chunk may have run on a different compute stream.)
    // In principle, one can do better by inserting per-kernel wait() calls below, for
    // every kernel which has pstate. I might revisit this later.
    evrb_cdd2->wait(compute_stream, seq_id - nbatches);

    // Now we can start launching kernels!

    _launch_tree_gridding(ichunk, ibatch, compute_stream);
    evrb_tree_gridding->record(compute_stream, seq_id);

    _launch_lagged_downsampler(ichunk, ibatch, compute_stream);

    // dd1 kernel: inbufs=[], outbufs=[gpu,g2h]
    //   gpu outbuf: consumers=[g2g,cdd2]
    //   g2h outbuf: consumers=[g2h]
    //
    // The [g2g,cdd2] consumers are automatic, since they run earlier on the same stream. 
    // but we need to wait for g2h, before launching dd1.

    evrb_g2h->wait(compute_stream, seq_id - nstreams);    // consumer (g2h)
    _launch_dd_stage1(ichunk, ibatch, compute_stream);

    // g2g kernel: inbufs=[gpu], outbufs=[g2h]
    //   gpu inbuf: producers=[dd1]
    //   g2h outbuf: consumers=[g2h]
    //
    // The dd1 producer is okay, since it's on the same stream.
    // The g2h consumer is okay, since dd1 waited on it (a few lines of code ago).

    _launch_et_g2g(ichunk, ibatch, compute_stream);
    evrb_g2g->record(compute_stream, seq_id);

    // g2h kernel: inbufs=[g2h], outbufs=[host]
    //   g2h inbuf: producers=[dd1,g2g]  
    //   host outbuf: consumers=[h2g,et_h2h]
    //
    // The dd1 producer is okay, since it's (earlier) on the same stream as g2g.
    // Need to wait/synchronize with et_h2h consumer, h2g consumer, g2g producer.
    //
    // To synchronize with the et_h2h consumer, we use a hack: wait on et_h2g producer
    // instead. This is slightly suboptimal, but convenient since we can use CudaEventRingbuf
    // as the only synchronization mechanism. (In practice, I doubt that the suboptimality
    // matters at all.)

    long et_seq_id = seq_id - nstreams - et_seq_headroom;
    evrb_et_h2g->synchronize_with_producer(et_seq_id);   // consumer (et_h2g)
    evrb_h2g->wait(g2h_stream, seq_id - nstreams);       // consumer (h2g)
    evrb_g2g->wait(g2h_stream, seq_id);                  // producer (g2g)
    _launch_g2h(ichunk, ibatch, g2h_stream);
    evrb_g2h->record(g2h_stream, seq_id);

    // Note: h2g kernel postponed until the end -- see below!

    // cdd2 kernel: inbufs=[gpu,h2g,et_gpu], outbufs=[]
    //   gpu inbuf:    producers=[dd1]
    //   h2g inbuf:    producers=[h2g]
    //   et_gpu inbuf: producers=[et_h2g]
    //
    // The dd1 producer is okay, since it's earlier on the same stream.
    // We need to wait on the producers h2g and et_h2g. 
    //
    // Note also that we need to wait on the 'evrb_output' events.
    // These are produced in release_output().
    
    bool out_blocking = !params.detect_deadlocks;
    evrb_h2g->wait(compute_stream, seq_id);                               // producer (h2g)
    evrb_et_h2g->wait(compute_stream, seq_id, true);                      // producer (et_h2g), blocking=true
    evrb_output->wait(compute_stream, seq_id - nstreams, out_blocking);   // consumer (output)
    _launch_cdd2(ichunk, ibatch, compute_stream);
    evrb_cdd2->record(compute_stream, seq_id);

    // Now the h2g kernel. As explained in the constructor above,
    // instead of queueing h2g(seq_id), we queue h2g(seq_id + nstreams).
    // This should improve throughput, by "prefetching" the h2g data for
    // the next batch of gpu kernels.
    //
    // h2g kernel: inbufs=[host], outbufs=[h2g]
    //   host inbuf: producers=[g2h]    (need to wait on this)
    //   h2g outbuf: consumers=[cdd2]   (need to wait on this)

    long prefetch_ichunk = (seq_id + nstreams) / nbatches;
    long prefetch_ibatch = (seq_id + nstreams) % nbatches;
    long producer_seq_id = seq_id + nstreams - host_seq_lag;

    evrb_g2h->wait(h2g_stream, producer_seq_id);  // producer (g2h)
    evrb_cdd2->wait(h2g_stream, seq_id);          // consumer (cdd2)
    _launch_h2g(prefetch_ichunk, prefetch_ibatch, h2g_stream);
    evrb_h2g->record(h2g_stream, seq_id + nstreams);
}


void GpuDedisperser::_worker_main()
{
    xassert(is_allocated);

    cudaStream_t h2g_stream = stream_pool->low_priority_h2g_stream;
    long seq_id = 0;

    // Note that the worker doesn't explicitly check GpuDedisperser::is_stopped.
    // This is okay because calling GpuDedisperser::stop() automatically calls stop()
    // in all the CudaEventRingbufs.

    for (;;) {
        long ichunk = seq_id / nbatches;
        long ibatch = seq_id % nbatches;

        // et_h2h kernel: inbufs=[host], outbufs=[et_host]
        //   host inbuf: producers=[g2h]
        //   et_host outbuf: consumers=[et_h2g]

        evrb_g2h->synchronize(seq_id - et_seq_lag, true);   // producer (blocking=true but farfetched)
        evrb_et_h2g->synchronize(seq_id - nstreams);        // consumer
        _do_et_h2h(ichunk, ibatch);

        // et_h2g kernel: inbufs=[et_host], outbufs=[et_gpu]
        //   et_host inbuf: producers=[et_h2h]
        //   et_gpu outbuf: consumers=[cdd2]
        //
        // The et_h2h producer is okay, since it runs synchronously in the same thread,
        // but we need to wait for the cdd2 consumer, before launching et_h2g.

        evrb_cdd2->wait(h2g_stream, seq_id - nstreams, true);  // consumer (blocking=true)
        _launch_et_h2g(ichunk, ibatch, h2g_stream);
        evrb_et_h2g->record(h2g_stream, seq_id);
        
        seq_id++;
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
        GpuDedisperser::Params params;
        params.plan = plan;
        params.stream_pool = CudaStreamPool::create(nstreams);
        params.detect_deadlocks = true;
        params.fixed_weights = false;

        gdd = make_shared<GpuDedisperser> (params);
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
            // Frequency-space array with shape (beams_per_batch, nfreq, ntime).
            // Random values uniform over [-1.0, 1.0].
            Array<float> arr({beams_per_batch, plan->nfreq, plan->nt_in}, af_uhost);
            for (long i = 0; i < arr.size; i++)
                arr.data[i] = ksgpu::rand_uniform(-1.0, 1.0);

            if (!host_only) {
                // Acquire input (and weights) on default stream.
                Array<void> gpu_in = gdd->acquire_input(ichunk, ibatch, nullptr);
                gpu_in.fill(arr.convert(config.dtype)); 
            }

            // Randomly initialize weights.
            for (int itree = 0; itree < ntrees; itree++) {
                Array<float> sbv = subband_variances.at(itree);
                Array<float> wt_cpu = ref_kernels_for_weights.at(itree)->make_random_weights(sbv);

                rdd0->wt_arrays.at(itree).fill(wt_cpu);
                rdd1->wt_arrays.at(itree).fill(wt_cpu);
                rdd2->wt_arrays.at(itree).fill(wt_cpu);

                if (!host_only) {
                    // Fill weights between GpuDedisperser::acquire_input() and GpuDedisperser::release_input().
                    const GpuPfWeightLayout &wl = gdd->cdd2_kernels.at(itree)->pf_weight_layout;
                    Array<void> wt_gpu = gdd->wt_arrays.at(itree).slice(0,0);  // FIXME istream=0 assumed
                    // FIXME extra copy here (+ another extra copy "hidden" in GpuPfWeightLayout::to_gpu())
                    Array<void> tmp = wl.to_gpu(wt_cpu);
                    wt_gpu.fill(tmp);
                }
            }

            if (!host_only) {
                // Release input on default cuda stream.
                gdd->release_input(ichunk, ibatch, nullptr);
            }

            rdd0->input_array.fill(arr);
            rdd0->dedisperse(ichunk, ibatch);  // (ichunk, ibatch)

            rdd1->input_array.fill(arr);
            rdd1->dedisperse(ichunk, ibatch);  // (ichunk, ibatch)

            rdd2->input_array.fill(arr);
            rdd2->dedisperse(ichunk, ibatch);  // (ichunk, ibatch)

            if (!host_only) {
                // Acquire output on default stream.
                gdd->acquire_output(ichunk, ibatch, nullptr);
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

            if (!host_only) {
                // Release output on default cuda stream.
                gdd->release_output(ichunk, ibatch, nullptr);
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


// ------------------------------------------------------------------------------------------------
//
// Timing


// Static member function.
void GpuDedisperser::time_one(const DedispersionConfig &config, long niterations, bool use_hugepages)
{   
    config.validate();
    xassert(niterations > 2*config.num_active_batches);

    Dtype dtype = config.dtype;
    long B = config.beams_per_batch;
    long F = config.get_total_nfreq();
    long T = config.time_samples_per_chunk;
    long S = config.num_active_batches;
    double Tc = 1.0e-3 * T * config.time_sample_ms;

    Dtype dt_int4 = Dtype::from_str("int4");
    int cpu_aflags = af_rhost | af_zero;
    int gpu_aflags = af_gpu | af_zero;

    if (use_hugepages)
        cpu_aflags |= af_mmap_huge;

    GpuDequantizationKernel dequantization_kernel(dtype, B, F, T);

    cout << "GpuDedisperser::time(): creating GpuDedisperser" << endl;
    GpuDedisperser::Params params;
    params.plan = make_shared<DedispersionPlan> (config);
    params.stream_pool = CudaStreamPool::create(S);
    params.detect_deadlocks = true;
    params.fixed_weights = true;
    shared_ptr<GpuDedisperser> gdd = make_shared<GpuDedisperser> (params);

    ResourceTracker rt = gdd->resource_tracker;  // copy
    rt += dequantization_kernel.resource_tracker;

    long raw_nbytes = xdiv(B * F * T, 2);
    rt.add_memcpy_h2g("raw_data", raw_nbytes);

    double h2g_bw = rt.get_h2g_bw();
    double g2h_bw = rt.get_g2h_bw();
    double gmem_bw = rt.get_gmem_bw();

    // The "multi_" prefix means "one array per stream".
    cout << "GpuDedisperser::time(): allocating" << endl;
    Array<void> multi_raw_cpu(dt_int4, {S,B,F,T}, cpu_aflags);
    Array<void> multi_raw_gpu(dt_int4, {S,B,F,T}, gpu_aflags);

    BumpAllocator dummy_cpu_allocator(cpu_aflags, -1);
    BumpAllocator dummy_gpu_allocator(gpu_aflags, -1);
    gdd->allocate(dummy_gpu_allocator, dummy_cpu_allocator);

    CudaEventRingbuf evrb_raw("raw", 2);  // copy raw data from cpu->gpu
    CudaEventRingbuf evrb_dq("dq", 1);    // dequantization kernel

    // We use a ksgpu::KernelTimer in the timing loop

    KernelTimer kt(niterations, S);
    
    cout << "GpuDedisperser::time(): running" << endl;
    while (kt.next()) {
        long seq_id = kt.curr_iteration;
        long ichunk = seq_id / gdd->nbatches;
        long ibatch = seq_id % gdd->nbatches;

        cudaStream_t h2g_stream = gdd->stream_pool->high_priority_h2g_stream;
        cudaStream_t compute_stream = gdd->stream_pool->compute_streams.at(kt.istream);

        Array<void> raw_cpu = multi_raw_cpu.slice(0, kt.istream);
        Array<void> raw_gpu = multi_raw_gpu.slice(0, kt.istream);

        // Copy raw data cpu->gpu. 
        // First, wait on the dequantization consumer.
        evrb_dq.wait(h2g_stream, seq_id - S);
        CUDA_CALL(cudaMemcpyAsync(raw_gpu.data, raw_cpu.data, raw_nbytes, cudaMemcpyHostToDevice, h2g_stream));
        evrb_raw.record(h2g_stream, seq_id);

        // I decided to synchronize the KernelTimer stream at this point.
        // This is done by making the KernelTimer stream a consumer of 'evrb_raw' (a CudaEventRingbuf).
        evrb_raw.wait(kt.stream, seq_id);

        // Run dequantization kernel on compute stream. 
        // First, wait on the producer (the cpu->gpu copy).
        // First, wait on the consumer (the dedisperser), by calling this->acquire_input().
        evrb_raw.wait(compute_stream, seq_id);
        Array<void> dd_in = gdd->acquire_input(ichunk, ibatch, compute_stream);
        dequantization_kernel.launch(dd_in, raw_gpu, compute_stream);
        evrb_dq.record(compute_stream, seq_id);

        // Launch all the dedispersion kernels.
        // (They will wait for the dequantization kernel.)
        gdd->release_input(ichunk, ibatch, compute_stream);

        // Throw away the dedispersion output.
        gdd->acquire_output(ichunk, ibatch, compute_stream);
        gdd->release_output(ichunk, ibatch, compute_stream);

        if (kt.warmed_up) {
            cout << "  iteration " << kt.curr_iteration
                 << ": real-time beams = " << (B * Tc / kt.dt)
                 << ": gmem_bw = " << (1.0e-9 * gmem_bw / kt.dt)
                 << ", g2h_bw = " << (1.0e-9 * g2h_bw / kt.dt)
                 << ", h2g_bw = " << (1.0e-9 * h2g_bw / kt.dt)
                 << " GB/s" << endl;
        }
    }
}

}  // namespace pirate
