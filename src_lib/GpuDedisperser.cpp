#include "../include/pirate/Dedisperser.hpp"
#include "../include/pirate/BumpAllocator.hpp"
#include "../include/pirate/CudaStreamPool.hpp"
#include "../include/pirate/CudaEventRingbuf.hpp"
#include "../include/pirate/DedispersionPlan.hpp"
#include "../include/pirate/RingbufCopyKernel.hpp"
#include "../include/pirate/TreeGriddingKernel.hpp"
#include "../include/pirate/DedispersionConfig.hpp"
#include "../include/pirate/CoalescedDdKernel2.hpp"
#include "../include/pirate/GpuDequantizationKernel.hpp"
#include "../include/pirate/MegaRingbuf.hpp"
#include "../include/pirate/PfVariance.hpp"
#include "../include/pirate/constants.hpp"  // xdiv(), pow2()
#include "../include/pirate/inlines.hpp"  // xdiv(), pow2()
#include "../include/pirate/utils.hpp"    // safe_memcpy_*

#include <ksgpu/rand_utils.hpp>
#include <ksgpu/test_utils.hpp>
#include <ksgpu/KernelTimer.hpp>

using namespace std;
using namespace ksgpu;

namespace pirate {
#if 0
}  // editor auto-indent
#endif

// General notes on synchronization and ring buffers, useful for reference
// throughout this code.
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
// FIXME now that the dust has settled, this synchronization logic is mechanical
// enough that I could capture it in a KernelGraph helper class, which keeps track
// of lagged dependencies between kernels. A KernelGraph::Node could represent
// a kernel, and contain shared_ptr<CudaEventRingbuf>. A KernelGraph::Edge could
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


// Static factory function.
std::shared_ptr<GpuDedisperser> GpuDedisperser::create(const GpuDedisperser::Params &params)
{
    return std::shared_ptr<GpuDedisperser>(new GpuDedisperser(params));
}


GpuDedisperser::GpuDedisperser(const GpuDedisperser::Params &params_) :
    params(params_)
{
    xassert(params.plan);
    xassert(params.stream_pool);
    xassert(params.cuda_device_id >= 0);
    xassert_eq(params.plan->num_active_batches, params.stream_pool->num_compute_streams);

    if (params.num_consumers < 0) {
        stringstream ss;
        ss << "GpuDedisperser constructor: params.num_consumers=" << params.num_consumers
           << " must be >= 0 (0 is allowed; the default of -1 is a 'you forgot to set it' sentinel)";
        throw runtime_error(ss.str());
    }

    // Size per-consumer state vectors to params.num_consumers. evrb_release_output
    // is filled below alongside the other CudaEventRingbufs.
    this->curr_output_acquire_seq_id.assign(params.num_consumers, 0);
    this->curr_output_release_seq_id.assign(params.num_consumers, 0);
    this->evrb_release_output.assign(params.num_consumers, nullptr);

    if (params.nbatches_out == 0)
        params.nbatches_out = params.plan->num_active_batches;

    if (params.nbatches_wt == 0)
        params.nbatches_wt = params.plan->num_active_batches;

    xassert(params.nbatches_out >= params.plan->num_active_batches);
    xassert(params.nbatches_wt >= params.plan->num_active_batches);

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
    for (long ipri = 0; ipri < plan->num_primary_trees; ipri++) {
        const DedispersionKernelParams &dd_params = plan->stage1_dd_kernel_params.at(ipri);
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

    // Set up the output ringbuf shape metadata (output_ringbuf.allocate() uses
    // it later). Axis 0 has length (nbatches_out * beams_per_batch); see the
    // doc-comment on struct Outputs.
    output_ringbuf.dtype = dtype;
    output_ringbuf.nbeams = params.nbatches_out * beams_per_batch;

    // Peak-finding weight/output arrays.
    for (long itree = 0; itree < ntrees; itree++) {
        const DedispersionTree &tree = trees.at(itree);
        const vector<long> &wt_shape = cdd2_kernels.at(itree)->expected_wt_shape;
        const vector<long> &wt_strides = cdd2_kernels.at(itree)->expected_wt_strides;

        // "Extended" weight shapes with an extra length-(nbatches_wt) axis added.
        this->extended_wt_shapes.push_back(svcat(params.nbatches_wt, wt_shape));
        this->extended_wt_strides.push_back(svcat(wt_shape[0] * wt_strides[0], wt_strides));

        long wt_nbytes = params.nbatches_wt * wt_shape[0] * wt_strides[0] * bytes_per_elt;
        resource_tracker.add_gmem_footprint("wt_arrays", wt_nbytes, true);

        output_ringbuf.ndm_out.push_back(tree.ndm_out);
        output_ringbuf.nt_out.push_back(tree.nt_out);

        long out_nelts = output_ringbuf.nbeams * tree.ndm_out * tree.nt_out;
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

    // Index lags between kernels. Used both for the CudaEventRingbuf capacities
    // below, and for the kernel-launch synchronization later in this file. A
    // placeholder (2*nstreams) is used where a host ringbuf / early triggers are
    // absent. (host_seq_lag >= nstreams is asserted below, see prefetch logic.)

    bool has_host_ringbuf = (mega_ringbuf->host_global_nseg > 0);
    bool has_early_triggers = (mega_ringbuf->et_host_zone.segments_per_frame > 0);

    this->host_seq_lag = has_host_ringbuf ? (mega_ringbuf->min_host_clag * nbatches) : (2*nstreams);
    this->et_seq_headroom = has_early_triggers ? (mega_ringbuf->min_et_headroom * nbatches) : (2*nstreams);
    this->et_seq_lag = has_early_triggers ? (mega_ringbuf->min_et_clag * nbatches) : (2*nstreams);

    // Initialize CudaEventRingbufs. Each ring's max_size is the worst-case
    // host-side span = the max lag between the producer's record() and the
    // slowest consumer's wait()/synchronize(). See "general notes" at the
    // top of this file, and notes/cuda_event_ringbuf.md.
    //
    // Threads below:
    //   D = caller of acquire_input + release_input_and_launch_dd_kernels
    //   W = early-trigger worker (_worker_main)
    //   A/R = acquire_output/release_output caller(s).
    //
    // D and W run in lockstep (D <= W <= D+S, enforced by the blocking et_h2g/cdd2
    // waits in the launch code), so every D<->W cross-thread lag is bounded by S.
    //
    // A ring with a cross-thread consumer records with blocking=true (an
    // under-sized ring then throttles instead of throwing) and adds +S of
    // headroom so the producer does not block in steady state.

    const long S   = nstreams;             // = num_active_batches = #compute streams
    const long nb  = nbatches;             // batches per time chunk
    const long no  = params.nbatches_out;  // output ring depth in batches (>= S)
    const long hsl = host_seq_lag;
    const long esl = et_seq_lag;

    // tg: producer D; consumer acq_input (D, lag S). span = S.
    this->evrb_tree_gridding = make_shared<CudaEventRingbuf> ("tree_gridding", 1, S);

    // g2g: producer D; consumer g2h-launch (D, lag 0). span = 1.
    this->evrb_g2g = make_shared<CudaEventRingbuf> ("g2g", 1, 1);

    // g2h: producer D; consumers dd1 (D, lag S), h2g-launch (D, lag hsl-S),
    // et_h2h (W, lag esl). span = max(S, hsl-S, esl); +S for the cross-thread W.
    this->evrb_g2h = make_shared<CudaEventRingbuf> ("g2h", 3, max(max(S, hsl-S), esl) + S);

    // h2g: producer D records (seq_id+S) (prefetch); consumers cdd2-launch and
    // g2h-launch (both D), the latter at lag 2S behind the recorded index. span = 2S.
    this->evrb_h2g = make_shared<CudaEventRingbuf> ("h2g", 2, 2*S);

    // cdd2: producer D; consumers tg (D, lag nb), h2g-launch (D, lag 0),
    // et_h2g-launch (W, lag S), acquire_output x N (A, lag up to no when output
    // is batched, e.g. test_one). span = max(nb, no); +S for the cross-thread A/W.
    this->evrb_cdd2 = make_shared<CudaEventRingbuf> ("cdd2", params.num_consumers + 3, max(nb, no) + S);

    // et_h2g: producer W; consumers et_h2h (W, lag S), cdd2-launch (D, lag <= S
    // via lockstep). span = S; +S for the cross-thread D. (The g2h code's
    // synchronize_with_producer() on this ring does not count as a consumer.)
    this->evrb_et_h2g = make_shared<CudaEventRingbuf> ("et_h2g", 2, 2*S);

    // output_<k> (one per output consumer): producer release_output (R);
    // consumer cdd2-launch back-pressure (D, lag no). span = no; +S for the
    // cross-thread (R vs D) case.
    for (long c = 0; c < params.num_consumers; c++) {
        string name = "output_" + to_string(c);
        this->evrb_release_output[c] = make_shared<CudaEventRingbuf> (name, 1, no + S);
    }

    // Note special "prefetch" logic for h2g copies!!
    // I can't decide if this way of doing it is elegant or a hack :)
    //
    // When gpu kernels are queued (in GpuDedisperser::release_input_and_launch_dd_kernels()),
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
        // blocking=false: all h2g consumers are on thread D (same thread).
        evrb_h2g->record(s, seq_id, /*blocking=*/false);
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

    // Note: we allocate the output_ringbuf first, so that it will be located as
    // close as possible to the gpu_allocator 'base' pointer. This seems preferable,
    // since we plan to share the output_ringbuf over cuda IPC (but I'm not sure if
    // it's really necessary.)
    //
    // Note: output_ringbuf shape metadata was initialized in the GpuDedisperser constructor.
    output_ringbuf.allocate(gpu_allocator);

    input_arrays = gpu_allocator.allocate_array<void>(dtype, {nstreams, beams_per_batch, nfreq, nt_in});

    for (DedispersionBuffer &buf: stage1_dd_bufs)
        buf.allocate(gpu_allocator);

    // wt_arrays
    for (long itree = 0; itree < ntrees; itree++) {
        const vector<long> &eshape = extended_wt_shapes.at(itree);
        const vector<long> &estrides = extended_wt_strides.at(itree);
        wt_arrays.push_back(gpu_allocator.allocate_array<void>(dtype, eshape, estrides));
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
    // Save the calling thread's current CUDA device so we can restore it
    // on exit. We switch to params.cuda_device_id for the CudaEventRingbuf::stop
    // and cudaStreamSynchronize calls below, since those interact with events
    // and streams that were created on that device. cudaGetDevice and
    // cudaSetDevice are called without CUDA_CALL so we never raise from
    // inside a destructor.
    int saved_device = -1;
    bool saved = (cudaGetDevice(&saved_device) == cudaSuccess);
    cudaSetDevice(params.cuda_device_id);

    // Stop the worker thread. GpuDedisperser::stop() cascades stop() to all the
    // internal CudaEventRingbufs, which unblocks any blocking wait/synchronize in
    // the worker thread so it exits before the worker.join() below. (stop() is
    // idempotent, so this is correct whether or not stop() already ran earlier.)
    this->stop();

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

    // Restore the caller-thread's prior CUDA device. (Member destructors that
    // run after this point will see the restored device; if any of them
    // invoke CUDA APIs they'll do so on the caller's context.)
    if (saved)
        cudaSetDevice(saved_device);
}


void GpuDedisperser::stop(std::exception_ptr e)
{
    std::lock_guard<std::mutex> lock(mutex);
    if (is_stopped) return;
    is_stopped = true;
    error = e;

    // Cascade stop() to the internal CudaEventRingbufs so that any thread
    // parked in a blocking wait / synchronize / synchronize_with_producer
    // (e.g. an external caller of release_input_and_launch_dd_kernels,
    // or our own et_h2h worker) throws and exits promptly. 'e' is forwarded
    // (see "Error reporting" in notes/stoppable_class.md): waiters rethrow
    // the root cause, or throw a generic "called on stopped instance"
    // message on a clean stop. This cascade is also what lets ~GpuDedisperser
    // unblock and join the worker thread by calling stop() alone. NOTE:
    // stopping an evrb does NOT cancel in-flight GPU work or destroy the cuda
    // events -- ~GpuDedisperser still synchronizes all streams before any GPU
    // array is freed.
    if (evrb_tree_gridding) evrb_tree_gridding->stop(e);
    if (evrb_g2g)           evrb_g2g->stop(e);
    if (evrb_g2h)           evrb_g2h->stop(e);
    if (evrb_h2g)           evrb_h2g->stop(e);
    if (evrb_cdd2)          evrb_cdd2->stop(e);
    if (evrb_et_h2g)        evrb_et_h2g->stop(e);
    for (auto &r : evrb_release_output)
        if (r) r->stop(e);

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
        CUDA_CALL(cudaSetDevice(params.cuda_device_id));
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

    for (long ipri = 0; ipri < plan->num_primary_trees; ipri++) {
        shared_ptr<GpuDedispersionKernel> kernel = stage1_dd_kernels.at(ipri);
        const DedispersionKernelParams &kp = kernel->params;
        Array<void> dd_buf = stage1_dd_bufs.at(istream).bufs.at(ipri);

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
    //
    // No synchronization here: the caller (_worker_main) already synchronizes the
    // producer of this kernel's input (the g2h copy, via evrb_g2h->synchronize())
    // and the consumer of its output (the et_h2g copy, via evrb_et_h2g->synchronize())
    // before calling _do_et_h2h(). (This is a plain host-side memcpy, not a stream
    // kernel, so those host-blocking synchronize() calls are what make it safe.)
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
    safe_memcpy_h2g_async(dst, src, nbytes, stream);
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
            safe_memcpy_g2h_async(dst, src, nbytes, stream);
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
            safe_memcpy_h2g_async(dst, src, nbytes, stream);
        }
    }
}


void GpuDedisperser::_launch_cdd2(long ichunk, long ibatch, cudaStream_t stream)
{
    long seq_id = ichunk * nbatches + ibatch;
    long iout = seq_id % params.nbatches_out;
    long iwt = seq_id % params.nbatches_wt;

    // Per-batch output views: slot 'iout' maps to the beam range
    // [iout*beams_per_batch, (iout+1)*beams_per_batch) of output_ringbuf (whose
    // axis 0 has length nbatches_out*beams_per_batch). slice() is metadata-only.
    Outputs batch_out = output_ringbuf.slice(iout * beams_per_batch, (iout+1) * beams_per_batch);

    for (long itree = 0; itree < ntrees; itree++) {
        Array<void> slice_max = batch_out.out_max.at(itree);
        Array<uint> slice_argmax = batch_out.out_argmax.at(itree);
        Array<void> slice_wt = wt_arrays.at(itree).slice(0,iwt);

        shared_ptr<CoalescedDdKernel2> cdd2_kernel = cdd2_kernels.at(itree);
        cdd2_kernel->launch(slice_max, slice_argmax, this->gpu_ringbuf, slice_wt, ichunk, ibatch, stream);
    }
}


// -------------------------------------------------------------------------------------------------


Array<void> GpuDedisperser::acquire_input(long seq_id, cudaStream_t stream)
{
    xassert(seq_id >= 0);
    xassert(is_allocated);

    long istream = seq_id % nstreams;
    Array<void> view;

    {
        std::unique_lock<std::mutex> lock(mutex);
        _throw_if_stopped("acquire_input");

        if (seq_id != curr_input_seq_id) {
            stringstream ss;
            ss << "GpuDedisperser::acquire_input(): expected seq_id="
               << curr_input_seq_id << ", got seq_id=" << seq_id;
            throw runtime_error(ss.str());
        }

        if (curr_input_acquired) {
            stringstream ss;
            ss << "GpuDedisperser::acquire_input(): double call to acquire_input()"
               << " with seq_id=" << seq_id;
            throw runtime_error(ss.str());
        }

        curr_input_acquired = true;

        // Compute the view under the lock, matching the precondition checks
        // that the (now-removed) view_input() method used to do. Slicing is
        // pointer arithmetic + metadata copy, no allocation.
        view = input_arrays.slice(0, istream);
    }

    try {
        // This call to wait() can be nonblocking, since we know that the tree_gridding
        // kernel was successfully launched by a previous call to release_input_and_launch_dd_kernels().
        evrb_tree_gridding->wait(stream, seq_id - nstreams);
    } catch (...) {
        stop(std::current_exception());
        throw;
    }

    return view;
}


void GpuDedisperser::release_input_and_launch_dd_kernels(long seq_id, cudaStream_t stream)
{
    xassert(seq_id >= 0);

    std::unique_lock<std::mutex> lock(mutex);
    _throw_if_stopped("release_input_and_launch_dd_kernels");

    if (seq_id != curr_input_seq_id) {
        stringstream ss;
        ss << "GpuDedisperser::release_input_and_launch_dd_kernels(): expected seq_id="
           << curr_input_seq_id << ", got seq_id=" << seq_id;
        throw runtime_error(ss.str());
    }

    if (!curr_input_acquired) {
        stringstream ss;
        ss << "GpuDedisperser::acquire_input(): release_input_and_launch_dd_kernels() called without "
           << " acquire_input(), seq_id=" << seq_id;
        throw runtime_error(ss.str());
    }

    curr_input_seq_id++;
    curr_input_acquired = false;

    lock.unlock();

    // Argument-checking ends here. If an exception is thrown below, call stop().
    // The rest of release_input_and_launch_dd_kernels() is in its own method _launch_dedispersion_kernels().
    try {
        _launch_dedispersion_kernels(seq_id, stream);
    } catch (...) {
        stop(std::current_exception());
        throw;
    }
}


GpuDedisperser::Outputs GpuDedisperser::acquire_output(long consumer_id, long seq_id, cudaStream_t stream,
                                                       bool sync, bool noreturn)
{
    xassert(seq_id >= 0);
    xassert(is_allocated);

    long iout = seq_id % params.nbatches_out;
    Outputs outputs;

    {
        std::unique_lock<std::mutex> lock(mutex);
        _throw_if_stopped("acquire_output");

        if ((consumer_id < 0) || (consumer_id >= params.num_consumers)) {
            stringstream ss;
            ss << "GpuDedisperser::acquire_output(): consumer_id=" << consumer_id
               << " out of range [0, " << params.num_consumers << ")";
            throw runtime_error(ss.str());
        }

        // acquire_output() seq_ids must be consecutive (0, 1, 2, ...) per consumer.
        if (seq_id != curr_output_acquire_seq_id[consumer_id]) {
            stringstream ss;
            ss << "GpuDedisperser::acquire_output(): consumer_id=" << consumer_id
               << ", expected seq_id=" << curr_output_acquire_seq_id[consumer_id]
               << ", got seq_id=" << seq_id;
            throw runtime_error(ss.str());
        }

        // In synchronous mode, acquire_output() and release_output() must
        // interleave: the previously-acquired batch must be released before the
        // next one is acquired (so the release cursor has caught up to acquire).
        if (params.synchronous &&
            (curr_output_release_seq_id[consumer_id] != curr_output_acquire_seq_id[consumer_id])) {
            stringstream ss;
            ss << "GpuDedisperser::acquire_output(): consumer_id=" << consumer_id
               << ", synchronous consumer called acquire_output() with seq_id=" << seq_id
               << " while batch seq_id=" << curr_output_release_seq_id[consumer_id]
               << " is acquired but not released (set Params::synchronous=false to allow this)";
            throw runtime_error(ss.str());
        }

        curr_output_acquire_seq_id[consumer_id]++;

        // Compute the per-batch output views under the lock (unless the caller
        // passed noreturn=true and doesn't need them). Slot 'iout' maps to the
        // beam range [iout*beams_per_batch, (iout+1)*beams_per_batch) of
        // output_ringbuf; slice() is pointer arithmetic + metadata copy, no
        // allocation. The returned views have nbeams = beams_per_batch.
        if (!noreturn) {
            outputs = output_ringbuf.slice(iout * beams_per_batch, (iout+1) * beams_per_batch);

            // Set the chunk/beam identity fields on the returned slice. These
            // override the values slice() copied from output_ringbuf, because
            // the ring slot 'iout' (= seq_id % nbatches_out) is NOT the true
            // chunk/beam index: seq_id = ichunk*nbatches + ibatch, so the true
            // chunk index is seq_id/nbatches and the true beam index of the
            // first beam is (seq_id % nbatches) * beams_per_batch.
            outputs.ichunk_zero_based = seq_id / nbatches;
            outputs.ichunk_fpga_based = outputs.ichunk_zero_based + params.initial_chunk;
            outputs.ibeam = (seq_id % nbatches) * beams_per_batch;
        }
    }

    // Argument checking ends here. If an exception is thrown below, call stop().
    // Wait for 'cdd2' to produce the output for seq_id. By default this makes the
    // caller-specified 'stream' wait; if sync=true we instead block the host
    // thread (and ignore 'stream'). The wait is blocking on the producer:
    // acquire_output() is assumed to run on a different thread than
    // acquire_input() / release_input_and_launch_dd_kernels().
    try {
        if (sync)
            evrb_cdd2->synchronize(seq_id, /*blocking=*/true);
        else
            evrb_cdd2->wait(stream, seq_id, /*blocking=*/true);
    } catch (...) {
        stop(std::current_exception());
        throw;
    }

    return outputs;
}


void GpuDedisperser::release_output(long consumer_id, long seq_id, cudaStream_t stream)
{
    xassert(seq_id >= 0);

    std::unique_lock<std::mutex> lock(mutex);
    _throw_if_stopped("release_output");

    if ((consumer_id < 0) || (consumer_id >= params.num_consumers)) {
        stringstream ss;
        ss << "GpuDedisperser::release_output(): consumer_id=" << consumer_id
           << " out of range [0, " << params.num_consumers << ")";
        throw runtime_error(ss.str());
    }

    // release_output() seq_ids must be consecutive (0, 1, 2, ...) per consumer.
    if (seq_id != curr_output_release_seq_id[consumer_id]) {
        stringstream ss;
        ss << "GpuDedisperser::release_output(): consumer_id=" << consumer_id
           << ", expected seq_id=" << curr_output_release_seq_id[consumer_id]
           << ", got seq_id=" << seq_id;
        throw runtime_error(ss.str());
    }

    // A consumer can't release a batch it hasn't acquired yet (the release cursor
    // must stay behind the acquire cursor). In synchronous mode this is exactly
    // "release_output() called without a preceding acquire_output()".
    if (curr_output_release_seq_id[consumer_id] >= curr_output_acquire_seq_id[consumer_id]) {
        stringstream ss;
        ss << "GpuDedisperser::release_output(): consumer_id=" << consumer_id
           << ", release_output() called for seq_id=" << seq_id
           << " which has not been acquired (next un-acquired seq_id="
           << curr_output_acquire_seq_id[consumer_id] << ")";
        throw runtime_error(ss.str());
    }

    curr_output_release_seq_id[consumer_id]++;

    lock.unlock();

    // Argument-checking ends here. If an exception is thrown below, call stop().
    // We record an event from the caller-specified stream on the per-consumer
    // ringbuf evrb_release_output[consumer_id]. The cdd2 kernel waits on
    // ALL N rings before reusing the output slot.
    try {
        // blocking=true: the consumer (cdd2-launch on thread D) is cross-thread
        // when release_output() runs on a separate thread R (e.g. FrbServer).
        evrb_release_output[consumer_id]->record(stream, seq_id, /*blocking=*/true);
    } catch (...) {
        stop(std::current_exception());
        throw;
    }
}


// See "general notes" at the top of this file, when thinking through the
// synchronization logic in _launch_dedispersion_kernels().

void GpuDedisperser::_launch_dedispersion_kernels(long seq_id, cudaStream_t stream)
{
    // Argument-checking has already been done in release_input_and_launch_dd_kernels().
    // Translate seq_id -> (ichunk, ibatch) for the per-kernel _launch_* helpers below.
    long ichunk = seq_id / nbatches;
    long ibatch = seq_id % nbatches;
    cudaStream_t g2h_stream = stream_pool->low_priority_g2h_stream;
    cudaStream_t h2g_stream = stream_pool->low_priority_h2g_stream;
    cudaStream_t compute_stream = stream_pool->compute_streams.at(seq_id % nstreams);

    // Compute kernel waits on the caller-specified stream.
    // Note: caller-specified stream isn't used for anything besides creating this event.
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
    // blocking=false: only consumer (acquire_input) is on this thread (D).
    evrb_tree_gridding->record(compute_stream, seq_id, /*blocking=*/false);

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
    // blocking=false: only consumer (g2h-launch) is on this thread (D).
    evrb_g2g->record(compute_stream, seq_id, /*blocking=*/false);

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
    // blocking=true: one consumer (et_h2h) is on the worker thread (W).
    evrb_g2h->record(g2h_stream, seq_id, /*blocking=*/true);

    // Note: h2g kernel postponed until the end -- see below!

    // cdd2 kernel: inbufs=[gpu,h2g,et_gpu], outbufs=[]
    //   gpu inbuf:    producers=[dd1]
    //   h2g inbuf:    producers=[h2g]
    //   et_gpu inbuf: producers=[et_h2g]
    //
    // The dd1 producer is okay, since it's earlier on the same stream.
    // We need to wait on the producers h2g and et_h2g. 
    //
    // Note also that we need to wait on ALL N 'evrb_release_output' events
    // (one per output consumer). These are produced in release_output().
    // If params.num_consumers == 0, the loop body doesn't execute and the
    // cdd2 kernel proceeds with no output-side back-pressure.

    evrb_h2g->wait(compute_stream, seq_id);                                          // producer (h2g)
    evrb_et_h2g->wait(compute_stream, seq_id, true);                                 // producer (et_h2g), blocking=true
    for (long c = 0; c < params.num_consumers; c++)
        evrb_release_output[c]->wait(compute_stream, seq_id - params.nbatches_out, /*blocking=*/true);   // consumer k
    _launch_cdd2(ichunk, ibatch, compute_stream);
    // blocking=true: consumers et_h2g-launch (W) and acquire_output (thread A)
    // are cross-thread.
    evrb_cdd2->record(compute_stream, seq_id, /*blocking=*/true);

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
    // blocking=false: both consumers (g2h-launch, cdd2-launch) are on this thread (D).
    evrb_h2g->record(h2g_stream, seq_id + nstreams, /*blocking=*/false);
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
        // blocking=true: one consumer (cdd2-launch) is on the main thread (D).
        evrb_et_h2g->record(h2g_stream, seq_id, /*blocking=*/true);
        
        seq_id++;
    }
}


// -------------------------------------------------------------------------------------------------
//
// GpuDedisperser::test()


// Fills every weight slot/beam with the SAME non-random analytic weights, computed from
// a PfAvarApproximation. See Dedisperser.hpp.
void GpuDedisperser::fill_analytic_weights(const Array<double> &freq_variances)
{
    xassert(is_allocated);

    // Analytic per-(subband, dm, profile) variances for every tree (validates freq_variances).
    PfAvarApproximation avar(plan, freq_variances);
    const long nslots = params.nbatches_wt * beams_per_batch;

    for (long itree = 0; itree < ntrees; itree++) {
        const DedispersionTree &t = plan->trees.at(itree);
        const long N = t.frequency_subbands.N;

        // Single-beam params (beams_per_batch = total_beams = 1) for the host weights.
        // We deliberately do NOT construct a ReferencePeakFindingKernel here: fill_host_weights()
        // needs only the params, and the kernel ctor would eagerly allocate large per-tree
        // apply()/eval_tokens() scratch buffers (several GB total) that we would never use.
        PeakFindingKernelParams pf_params = plan->stage2_pf_params.at(itree);   // copy
        pf_params.beams_per_batch = 1;
        pf_params.total_beams = 1;

        // Non-random weights from the analytic variances. avar.tree_variance[itree] already has
        // the (N, ndm_wt, nprofiles) shape that fill_host_weights() expects.
        Array<float> host_weights({1, t.ndm_wt, t.nt_wt, t.nprofiles, N}, af_rhost | af_zero);
        pf_params.fill_host_weights(host_weights, avar.tree_variance.at(itree), /*randomize=*/ false);

        // Fill the first beam slot via to_gpu(), then duplicate it to all 'nslots' beam slots with
        // GPU->GPU memcopies. wt_arrays[itree] flattens its (nbatches_wt, beams_per_batch) axes into
        // 'nslots' back-to-back beam blocks, each 'beam_nelts' apart, so beam slot k starts at element
        // offset k*beam_nelts.
        const GpuPfWeightLayout &wl = cdd2_kernels.at(itree)->pf_weight_layout;
        Array<void> dst0 = wt_arrays.at(itree).slice(0, 0).slice(0, 0, 1);   // first beam slot, (1,ndm_wt,...)
        wl.to_gpu(dst0, host_weights);

        const long beam_nelts = extended_wt_strides.at(itree).at(1);   // slot-to-slot (per-beam) stride
        const long beam_nbytes = (beam_nelts * dtype.nbits) / 8;
        xassert_eq(extended_wt_strides.at(itree).at(0), beams_per_batch * beam_nelts);

        // Per-slot copy size = ONE beam block's VALID strided extent (1 + sum over the inner,
        // per-beam axes of (shape-1)*stride), which is <= beam_nelts. This is deliberately smaller
        // than the slot stride 'beam_nelts': the weight layout pads each beam block's trailing bytes
        // (touter_byte_stride is 128-byte aligned; see GpuPfWeightLayout), and wt_arrays is allocated
        // to the array's strided extent (Array.cpp _array_init_dchecked), which trims the LAST beam
        // block's trailing pad. Copying the full 'beam_nbytes' into the last slot would therefore
        // overrun the allocation (-> cudaMemcpy "invalid argument", or silent corruption of a
        // neighboring allocation). The pad bytes are never read by the cdd2 kernel, so copying only
        // the valid extent per slot -- with the dst still offset by the full beam_nbytes stride --
        // reproduces the weights exactly.
        const vector<long> &bshape   = cdd2_kernels.at(itree)->expected_wt_shape;    // [beams_per_batch, ...inner]
        const vector<long> &bstrides = cdd2_kernels.at(itree)->expected_wt_strides;
        long beam_valid_nelts = 1;
        for (size_t i = 1; i < bshape.size(); i++)   // inner (single-beam) axes only; skip axis 0 (beams)
            beam_valid_nelts += (bshape.at(i) - 1) * bstrides.at(i);
        const long beam_valid_nbytes = (beam_valid_nelts * dtype.nbits) / 8;

        char *base = (char *) wt_arrays.at(itree).data;
        for (long k = 1; k < nslots; k++)
            CUDA_CALL(cudaMemcpy(base + k * beam_nbytes, base, beam_valid_nbytes, cudaMemcpyDeviceToDevice));
    }

    // Ensure all GPU copies have completed before returning.
    CUDA_CALL(cudaDeviceSynchronize());
}


// Copies host-side peak-finding weights to the GPU for one tree, filling all nbatches_wt
// weight slots. See Dedisperser.hpp.
void GpuDedisperser::fill_all_weights(long itree, const Array<float> &pf_weights)
{
    xassert(is_allocated);
    xassert((itree >= 0) && (itree < ntrees));

    const DedispersionTree &t = plan->trees.at(itree);
    xassert_shape_eq(pf_weights, ({params.nbatches_wt, beams_per_batch, t.ndm_wt, t.nt_wt, t.nprofiles, t.frequency_subbands.N}));

    // wt_arrays[itree] flattens (nbatches_wt, beams_per_batch, ...inner); slot 's' is the
    // (beams_per_batch, ...inner) block, matching one pf_weights[s] of logical shape
    // (beams_per_batch, ndm_wt, nt_wt, nprofiles, N) after to_gpu()'s layout transform
    // (which also converts fp32 -> fp16 if needed).
    const GpuPfWeightLayout &wl = cdd2_kernels.at(itree)->pf_weight_layout;
    for (long s = 0; s < params.nbatches_wt; s++) {
        Array<void> wgpu = wt_arrays.at(itree).slice(0, s);
        wl.to_gpu(wgpu, pf_weights.slice(0, s));
    }
}


// Static member function.
void GpuDedisperser::test_one(const DedispersionConfig &config, long nchunks, long nbatches_out, long nbatches_wt, bool host_only)
{
    cout << "\n" << "GpuDedisperser::test()" << endl;
    config.emit_cpp();
    config.test();  // calls config.validate()

    Dtype dtype = config.dtype;
    long beams_per_batch = config.beams_per_batch;
    long nbatches = xdiv(config.beams_per_gpu, config.beams_per_batch);
    long nstreams = config.num_active_batches;
    long nt_in = config.time_samples_per_chunk;
    long nfreq = config.get_total_nfreq();
    std::mt19937 &rng = ksgpu::default_rng();

    cout << "    nchunks = " << nchunks << ";\n"
         << "    nbatches_out = " << nbatches_out << ";"
         << "       // nstreams = " << nstreams << ", (chunks * batches) = " << (nchunks*nbatches) << "\n"
         << "    nbatches_wt = " << nbatches_wt << ";\n"
         << "    host_only = " << host_only << ";" << endl;

    if (host_only)
         cout << "    !!! Host-only test, GPU code will not be run !!!" << endl;

    if (nbatches_out == 0)
        nbatches_out = nstreams;
    if (nbatches_wt == 0)
        nbatches_wt = nbatches_out;

    xassert(nchunks > 0);
    xassert(nbatches_out > 0);
    xassert(nbatches_out <= nchunks * nbatches);
    xassert(nbatches_wt > 0);   // GpuDedisperser ctor additionally checks nbatches_wt >= nstreams

    Array<double> freq_variance({nfreq}, af_uhost | af_zero);  // for PfAvarApproximation
    for (long ifreq = 0; ifreq < nfreq; ifreq++)
        freq_variance.data[ifreq] = 1.0f;

    shared_ptr<DedispersionPlan> plan = make_shared<DedispersionPlan> (config);
    shared_ptr<PfAvarApproximation> avar = make_shared<PfAvarApproximation> (plan, freq_variance);
    shared_ptr<GpuDedisperser> gdd;
    long ntrees = plan->ntrees;
    
    if (!host_only) {
        // We use compute_stream_priority=-1 so that cudaMemcpyAsync(..., compute_stream)
        // will fill the GpuDedisperser input arrays as quickly as possible. See below.
        int compute_stream_priority = -1;

        GpuDedisperser::Params params;
        params.plan = plan;
        params.stream_pool = CudaStreamPool::create(nstreams, compute_stream_priority);
        params.nbatches_out = nbatches_out;
        params.nbatches_wt = nbatches_wt;
        params.num_consumers = 1;
        params.cuda_device_id = 0;
        params.initial_chunk = 0;   // test harness: outputs are zero-based

        gdd = GpuDedisperser::create(params);
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

    // Create ReferenceDedispersers (must come after Dcore initialization).
    // Pass an explicit Dcore (from the GPU kernels), not the host-only default.
    ReferenceDedisperserBase::Params rdd_params;
    rdd_params.plan = plan;
    rdd_params.Dcore = Dcore;
    rdd_params.sophistication = 0;  shared_ptr<ReferenceDedisperserBase> rdd0 = ReferenceDedisperserBase::make(rdd_params);
    rdd_params.sophistication = 1;  shared_ptr<ReferenceDedisperserBase> rdd1 = ReferenceDedisperserBase::make(rdd_params);
    rdd_params.sophistication = 2;  shared_ptr<ReferenceDedisperserBase> rdd2 = ReferenceDedisperserBase::make(rdd_params);

    // Dedispersion input array. We randomly generate it (nbatches_out) batches at a time.
    Array<float> dd_in_cpu({nbatches_out, beams_per_batch, nfreq, nt_in}, af_rhost | af_zero);
    Array<void> dd_in_gpu(dtype, {nbatches_out, beams_per_batch, nfreq, nt_in}, af_gpu | af_zero);

    // Peak-finding weights arrays. The GPU weight ring has 'nbatches_wt' slots (independent of
    // nbatches_out); we hold a host copy of the whole ring and re-randomize it every group.
    vector<Array<float>> pf_wt_cpu(ntrees);
    for (long itree = 0; itree < ntrees; itree++) {
        const DedispersionTree &t = plan->trees.at(itree);
        long B = beams_per_batch;
        long D = t.ndm_wt;
        long T = t.nt_wt;
        long P = t.nprofiles;
        long N = t.frequency_subbands.N;
        pf_wt_cpu.at(itree) = Array<float> ({nbatches_wt,B,D,T,P,N}, af_rhost | af_zero);
    }
    
    for (long ichunk = 0; ichunk < nchunks; ichunk++) {
        for (long ibatch = 0; ibatch < nbatches; ibatch++) {
            long seq_id = ichunk * nbatches + ibatch;
            long seq_base = (seq_id / nbatches_out) * nbatches_out;

            // Every (nbatches_out) batches, we do a large computation, to simulate
            // dd_in and pf_wt arrays, copy to the GPU, and launch all GPU compute.
            // This stress-tests the GPU synchronization logic.

            if (seq_id == seq_base) {
                long seq_end = min(seq_base + nbatches_out, nchunks * nbatches);
                long ns = seq_end - seq_base;

                // Simulate dedispersion input.
                // Random values uniform over [-1.0, 1.0].
                xassert(dd_in_cpu.is_fully_contiguous());
                for (long i = 0; i < ns * beams_per_batch * nfreq * nt_in; i++)
                    dd_in_cpu.data[i] = ksgpu::rand_uniform(-1.0, 1.0, rng);

                // Copy dedispersion input to GPU.
                if (!host_only) {
                    Array<void> src = dd_in_cpu.slice(0,0,ns);
                    Array<void> dst = dd_in_gpu.slice(0,0,ns);
                    dst.fill(src.convert(dtype));
                }

                // Simulate peak-finding weights (host-side), then copy to the GPU.
                // As noted in Dedisperser.hpp, there's no explicit API protecting
                // the weights, and the caller is responsible for thinking through race conditions.
                // Out of paranoia, we put cudaDeviceSynchronize() here, but it shouldn't be necessary.

                CUDA_CALL(cudaDeviceSynchronize());

                for (long itree = 0; itree < ntrees; itree++) {
                    Array<double> variances = avar->tree_variance.at(itree);

                    // Re-randomize the entire weight ring (all nbatches_wt slots), so every slot
                    // that any batch in this group might select (seq_id % nbatches_wt) is valid.
                    for (long s = 0; s < nbatches_wt; s++) {
                        Array<float> wcpu = pf_wt_cpu.at(itree).slice(0,s);
                        plan->stage2_pf_params.at(itree).fill_host_weights(wcpu, variances, /*randomize=*/ true);
                    }

                    // Refresh the whole weight ring on the GPU (all nbatches_wt slots).
                    if (!host_only)
                        gdd->fill_all_weights(itree, pf_wt_cpu.at(itree));
                }

                if (!host_only) {

                    // After all this setup, launch all compute on GPU. This loop is intended
                    // to be fast, e.g. no activity on default stream, in order to stress-test
                    // the kernel-queueing logic.

                    for (long s = 0; s < ns; s++) {
                        long iseq_gpu = seq_base + s;

                        long istream = iseq_gpu % nstreams;
                        cudaStream_t compute_stream = gdd->stream_pool->compute_streams.at(istream);

                        Array<void> src = dd_in_gpu.slice(0,s);
                        Array<void> dst = gdd->acquire_input(iseq_gpu, compute_stream);

                        // Some paranoid asserts.
                        xassert(src.dtype == dst.dtype);
                        xassert_shape_eq(src, ({beams_per_batch, nfreq, nt_in}));
                        xassert_shape_eq(dst, ({beams_per_batch, nfreq, nt_in}));
                        xassert(src.is_fully_contiguous());
                        xassert(dst.is_fully_contiguous());
                        xassert(src.on_gpu());
                        xassert(dst.on_gpu());

                        // FIXME using cudaMemcpyAsync() here instead of ksgpu::launch_memcpy_kernel(),
                        // due to alignment issues.

                        long nbytes = dst.size * xdiv(dst.dtype.nbits, 8);
                        cudaMemcpyAsync(dst.data, src.data, nbytes, cudaMemcpyDeviceToDevice, compute_stream);
                        gdd->release_input_and_launch_dd_kernels(iseq_gpu, compute_stream);
                    }
                }
            }
            
            // End of "Every (nbatches_out) batches, we do a large computation...".
            // Back to the outer loops over (ichunk, ibatch).

            for (long itree = 0; itree < ntrees; itree++) {
                // Match the GPU's weight-slot selection in _launch_cdd2(): iwt = seq_id % nbatches_wt.
                Array<float> wt_cpu = pf_wt_cpu.at(itree).slice(0, seq_id % nbatches_wt);
                rdd0->wt_arrays.at(itree).fill(wt_cpu);
                rdd1->wt_arrays.at(itree).fill(wt_cpu);
                rdd2->wt_arrays.at(itree).fill(wt_cpu);
            }

            Array<float> dd_in = dd_in_cpu.slice(0, seq_id - seq_base);

            rdd0->input_array.fill(dd_in);
            rdd0->dedisperse(ichunk, ibatch);  // (ichunk, ibatch)

            rdd1->input_array.fill(dd_in);
            rdd1->dedisperse(ichunk, ibatch);  // (ichunk, ibatch)

            rdd2->input_array.fill(dd_in);
            rdd2->dedisperse(ichunk, ibatch);  // (ichunk, ibatch)
            
            if (!host_only)
                gdd->acquire_output(0, seq_id, nullptr, /*sync=*/false, /*noreturn=*/true);

            for (long itree = 0; itree < ntrees; itree++) {
                const DedispersionTree &tree = plan->trees.at(itree);

                // Compare peak-finding 'out_max'.
                assert_arrays_equal(rdd0->out_max.at(itree), rdd1->out_max.at(itree), "pfmax_ref0", "pfmax_ref1", {"beam","pfdm","pft"});
                assert_arrays_equal(rdd0->out_max.at(itree), rdd2->out_max.at(itree), "pfmax_ref0", "pfmax_ref2", {"beam","pfdm","pft"});
               
                // To check 'out_argmax', we need to jump through some hoops.
                shared_ptr<ReferencePeakFindingKernel> pf_kernel = rdd0->pf_kernels.at(itree);

                pf_kernel->eval_tokens(pf_tmp.at(itree), rdd1->out_argmax.at(itree), rdd0->wt_arrays.at(itree));
                assert_arrays_equal(rdd0->out_max.at(itree), pf_tmp.at(itree), "pfmax_ref0", "pf_tmp_ref1", {"beam","pfdm","pft"});

                pf_kernel->eval_tokens(pf_tmp.at(itree), rdd2->out_argmax.at(itree), rdd0->wt_arrays.at(itree));
                assert_arrays_equal(rdd0->out_max.at(itree), pf_tmp.at(itree), "pfmax_ref0", "pf_tmp_ref2", {"beam","pfdm","pft"});

                if (host_only)
                    continue;

                long iout = seq_id % nbatches_out;                
                Outputs gdd_out = gdd->output_ringbuf.slice(iout * beams_per_batch, (iout+1) * beams_per_batch);
                Array<void> gdd_max = gdd_out.out_max.at(itree);
                Array<uint> gpu_tokens = gdd_out.out_argmax.at(itree).to_host();

                long n = tree.primary_tree_index + tree.total_rank();
                double eps = 3.0 * config.dtype.precision() * sqrt(n+2);
                assert_arrays_equal(rdd0->out_max.at(itree), gdd_max, "pfmax_ref0", "pfmax_gpu", {"beam","pfdm","pft"}, eps, eps);

                pf_kernel->eval_tokens(pf_tmp.at(itree), gpu_tokens, rdd0->wt_arrays.at(itree));
                assert_arrays_equal(rdd0->out_max.at(itree), pf_tmp.at(itree), "pfmax_ref0", "pf_tmp_gpu", {"beam","pfdm","pft"}, eps, eps);
            }

            if (!host_only)
                gdd->release_output(0, seq_id, nullptr);
        }
    }
    
    cout << endl;
}

// Static member function.
void GpuDedisperser::test_random()
{
    auto config = DedispersionConfig::make_random();
    config.validate();
    
    long ntree = pow2(config.toplevel_tree_rank);
    long nt_chunk = config.time_samples_per_chunk;
    long min_nchunks = (ntree / nt_chunk) + 2;
    long max_nchunks = (1024*1024) / (ntree * nt_chunk * config.beams_per_gpu);
    max_nchunks = max(min_nchunks, max_nchunks);

    long nchunks = ksgpu::rand_int(1, max_nchunks+1);

    long nfreq = config.get_total_nfreq();
    long beams_per_batch = config.beams_per_batch;
    long nbatches = xdiv(config.beams_per_gpu, config.beams_per_batch);
    long min_nbatches_out = config.num_active_batches;
    long max_nbatches_out = (1024*1024*1024) / (beams_per_batch * nfreq * nt_chunk);
    max_nbatches_out = min(max_nbatches_out, nchunks * nbatches);
    max_nbatches_out = max(min_nbatches_out, max_nbatches_out);

    double t = rand_uniform(log(min_nbatches_out) + 1.0e-3, log(max_nbatches_out) + 1.0);
    long nbatches_out = min(long(exp(t)), max_nbatches_out);

    // Independent weight-ring depth. Lower-bounded by nstreams (= min_nbatches_out), since the
    // GpuDedisperser ctor requires nbatches_wt >= num_active_batches.
    long nbatches_wt = ksgpu::rand_int(min_nbatches_out, 2*nbatches_out);

    GpuDedisperser::test_one(config, nchunks, nbatches_out, nbatches_wt);
}


// ------------------------------------------------------------------------------------------------
//
// Timing


void GpuDedisperser::time(BumpAllocator &gpu_allocator, BumpAllocator &cpu_allocator, long niterations)
{
    xassert(is_allocated);
    xassert(niterations > 2*nstreams);

    long B = beams_per_batch;
    long F = nfreq;
    long T = nt_in;
    long S = nstreams;
    double Tc = 1.0e-3 * T * config.time_sample_ms;

    xassert_divisible(T, 256);

    Dtype dt_int4 = Dtype::from_str("int4");
    GpuDequantizationKernel dequantization_kernel(dtype, B, F, T);

    ResourceTracker rt = resource_tracker;  // copy
    rt += dequantization_kernel.resource_tracker;

    long raw_nbytes   = xdiv(B * F * T, 2);
    long scoff_nbytes = B * F * xdiv(T, 256) * 2 * sizeof(__half);
    rt.add_memcpy_h2g("raw_data",       raw_nbytes);
    rt.add_memcpy_h2g("scales_offsets", scoff_nbytes);

    double h2g_bw = rt.get_h2g_bw();
    double g2h_bw = rt.get_g2h_bw();
    double gmem_bw = rt.get_gmem_bw();
    double hmem_bw = rt.get_hmem_bw();

    // The "multi_" prefix means "one array per stream".
    cout << "GpuDedisperser::time(): allocating raw data + scales_offsets arrays" << endl;
    Array<void>   multi_raw_cpu   = cpu_allocator.allocate_array<void>(dt_int4, {S,B,F,T});
    Array<void>   multi_raw_gpu   = gpu_allocator.allocate_array<void>(dt_int4, {S,B,F,T});
    Array<__half> multi_scoff_cpu = cpu_allocator.allocate_array<__half>({S,B,F,T/256,2});
    Array<__half> multi_scoff_gpu = gpu_allocator.allocate_array<__half>({S,B,F,T/256,2});

    // max_size: raw has 2 lag-0 consumers on this thread (span 1); dq has one
    // consumer at lag S on this thread (span S). All records/waits here are
    // same-thread, so blocking stays false (the default).
    CudaEventRingbuf evrb_raw("raw", 2, 1);  // copy raw data + scales_offsets from cpu->gpu
    CudaEventRingbuf evrb_dq("dq", 1, S);    // dequantization kernel

    // We use a ksgpu::KernelTimer in the timing loop

    KernelTimer kt(niterations, S);

    cout << "GpuDedisperser::time(): running" << endl;
    while (kt.next()) {
        long seq_id = kt.curr_iteration;

        cudaStream_t h2g_stream = stream_pool->high_priority_h2g_stream;
        cudaStream_t compute_stream = stream_pool->compute_streams.at(kt.istream);

        Array<void>   raw_cpu   = multi_raw_cpu.slice(0, kt.istream);
        Array<void>   raw_gpu   = multi_raw_gpu.slice(0, kt.istream);
        Array<__half> scoff_cpu = multi_scoff_cpu.slice(0, kt.istream);
        Array<__half> scoff_gpu = multi_scoff_gpu.slice(0, kt.istream);

        // Copy raw data + scales_offsets cpu->gpu.
        // First, wait on the dequantization consumer.
        evrb_dq.wait(h2g_stream, seq_id - S);
        safe_memcpy_h2g_async(raw_gpu.data,   raw_cpu.data,   raw_nbytes,   h2g_stream);
        safe_memcpy_h2g_async(scoff_gpu.data, scoff_cpu.data, scoff_nbytes, h2g_stream);
        evrb_raw.record(h2g_stream, seq_id);

        // I decided to synchronize the KernelTimer stream at this point.
        // This is done by making the KernelTimer stream a consumer of 'evrb_raw' (a CudaEventRingbuf).
        evrb_raw.wait(kt.stream, seq_id);

        // Run dequantization kernel on compute stream.
        // First, wait on the producer (the cpu->gpu copy).
        // Then, wait on the consumer (the dedisperser), by calling this->acquire_input().
        evrb_raw.wait(compute_stream, seq_id);
        Array<void> dd_in = acquire_input(seq_id, compute_stream);
        dequantization_kernel.launch(dd_in, scoff_gpu, raw_gpu, compute_stream);
        evrb_dq.record(compute_stream, seq_id);

        // Launch all the dedispersion kernels.
        // (They will wait for the dequantization kernel.)
        release_input_and_launch_dd_kernels(seq_id, compute_stream);

        // Wait for dedispersion output (then throw it away).
        acquire_output(0, seq_id, compute_stream, /*sync=*/false, /*noreturn=*/true);
        release_output(0, seq_id, compute_stream);

        if (kt.warmed_up) {
            cout << "  iteration " << kt.curr_iteration
                 << ": real-time beams = " << (B * Tc / kt.dt)
                 << ", gmem_bw = " << (1.0e-9 * gmem_bw / kt.dt)
                 << ", hmem_bw = " << (1.0e-9 * hmem_bw / kt.dt)
                 << ", g2h_bw = " << (1.0e-9 * g2h_bw / kt.dt)
                 << ", h2g_bw = " << (1.0e-9 * h2g_bw / kt.dt)
                 << " GB/s" << endl;
        }
    }
}


// -------------------------------------------------------------------------------------------------
//
// GpuDedisperser::Outputs helpers


void GpuDedisperser::Outputs::allocate(BumpAllocator &gpu_allocator)
{
    long nt = ndm_out.size();
    xassert(nbeams > 0);
    xassert_eq(ndm_out.size(), nt_out.size());
    xassert(out_max.empty() && out_argmax.empty());   // not already allocated

    out_max.resize(nt);
    out_argmax.resize(nt);

    for (long itree = 0; itree < nt; itree++) {
        std::initializer_list<long> shape = { nbeams, ndm_out[itree], nt_out[itree] };
        out_max[itree]    = gpu_allocator.allocate_array<void>(dtype, shape);
        out_argmax[itree] = gpu_allocator.allocate_array<uint>(shape);
    }
}


GpuDedisperser::Outputs GpuDedisperser::Outputs::slice(long start_beam_index, long end_beam_index) const
{
    xassert(start_beam_index >= 0);
    xassert(start_beam_index <= end_beam_index);
    xassert(end_beam_index <= nbeams);

    long nt = out_max.size();

    Outputs ret;
    ret.dtype = dtype;
    ret.nbeams = end_beam_index - start_beam_index;

    // A beam-axis slice leaves the chunk indices unchanged and shifts the
    // first-beam index by start_beam_index. (Callers that slice by ring slot
    // rather than by true beam index -- GpuDedisperser/FrbGrouper acquire_output()
    // -- overwrite these fields afterwards; see those methods.)
    ret.ichunk_zero_based = ichunk_zero_based;
    ret.ichunk_fpga_based = ichunk_fpga_based;
    ret.ibeam = ibeam + start_beam_index;

    ret.ndm_out = ndm_out;
    ret.nt_out = nt_out;
    ret.out_max.resize(nt);
    ret.out_argmax.resize(nt);

    for (long itree = 0; itree < nt; itree++) {
        ret.out_max[itree]    = out_max.at(itree).slice(0, start_beam_index, end_beam_index);
        ret.out_argmax[itree] = out_argmax.at(itree).slice(0, start_beam_index, end_beam_index);
    }

    return ret;
}


}  // namespace pirate
