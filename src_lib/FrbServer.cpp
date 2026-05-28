#include "../include/pirate/FrbServer.hpp"
#include "../include/pirate/FileWriter.hpp"
#include "../include/pirate/DedispersionPlan.hpp"
#include "../include/pirate/Dedisperser.hpp"        // GpuDedisperser
#include "../include/pirate/CudaStreamPool.hpp"

#include <cstring>   // strstr

#include <ksgpu/string_utils.hpp>  // tuple_str()

#include <chrono>    // duration<double> (processing-thread delay) + ms (MonitorRingbuf timeout)
#include <thread>    // this_thread::sleep_for (processing-thread delay)
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "../grpc/frb_search.grpc.pb.h"
#include <grpcpp/grpcpp.h>

using namespace std;
using namespace ksgpu;


namespace pirate {
#if 0
}  // editor auto-indent
#endif

namespace fs = frb::search::v1;


// GRPC service implementation. See bottom of file for implementations of individual RPCs.

class FrbRpcService final : public fs::FrbSearch::Service {
public:
    std::weak_ptr<FrbServer> state;

    FrbRpcService(const weak_ptr<FrbServer> &s) : state(s) {}

    // These functions implement the individual RPCs.
    void _GetStatus(const fs::GetStatusRequest *request, fs::GetStatusResponse *response);
    void _GetXEngineMetadata(const fs::GetXEngineMetadataRequest *request, fs::GetXEngineMetadataResponse *response);
    void _WriteFiles(const fs::WriteFilesRequest *request, fs::WriteFilesResponse *response);
    void _SubscribeFiles(grpc::ServerContext* context, grpc::ServerWriter<fs::SubscribeFilesResponse>* writer);
    void _GetConfig(const fs::GetConfigRequest *request, fs::GetConfigResponse *response);
    void _MonitorRingbuf(grpc::ServerContext* context, grpc::ServerWriter<fs::MonitorRingbufResponse>* writer);

    // Helper to lock the weak_ptr. Throws if the server is exiting.
    inline shared_ptr<FrbServer> _lock_state();

    // Try-catch wrappers, to gracefully return an error status to the client 
    // (instead of crashing the server).

    grpc::Status GetStatus(
        grpc::ServerContext* context,
        const fs::GetStatusRequest* request,
        fs::GetStatusResponse* response) override;

    grpc::Status GetXEngineMetadata(
        grpc::ServerContext* context,
        const fs::GetXEngineMetadataRequest* request,
        fs::GetXEngineMetadataResponse* response) override;

    grpc::Status WriteFiles(
        grpc::ServerContext* context,
        const fs::WriteFilesRequest* request,
        fs::WriteFilesResponse* response) override;

    grpc::Status SubscribeFiles(
        grpc::ServerContext* context,
        const fs::SubscribeFilesRequest* request,
        grpc::ServerWriter<fs::SubscribeFilesResponse>* writer) override;

    grpc::Status GetConfig(
        grpc::ServerContext* context,
        const fs::GetConfigRequest* request,
        fs::GetConfigResponse* response) override;

    grpc::Status MonitorRingbuf(
        grpc::ServerContext* context,
        const fs::MonitorRingbufRequest* request,
        grpc::ServerWriter<fs::MonitorRingbufResponse>* writer) override;
};


// -------------------------------------------------------------------------------------------------
//
// FrbServer


std::shared_ptr<FrbServer> FrbServer::create(const Params &params)
{
    // Note: can't use make_shared since constructor is private.
    return std::shared_ptr<FrbServer>(new FrbServer(params));
}


FrbServer::FrbServer(const Params &p) : params(p)
{
    xassert(params.file_writer);
    xassert(params.receivers.size() > 0);
    xassert(params.rpc_server_address.size() > 0);  // check that string was initialized
    xassert(params.ringbuf_nchunks > 0);
    xassert(params.min_data_mtu > 0);
    xassert(params.host_allocator);
    xassert(params.gpu_allocator);
    xassert(params.cuda_device_id >= 0);

    // Fail fast on a malformed user-supplied config. (The four metadata-
    // dependent members are overwritten later by the processing thread,
    // but params.config_prefilled must already be self-consistent enough
    // to pass validate() -- DedispersionConfig::from_yaml requires those
    // members to be present in the YAML.)
    params.config_prefilled.validate();

    // Check that all recivers use the same allocator, and consumer IDs are consistent with ordering.
    for (uint i = 0; i < params.receivers.size(); i++) {
        xassert(params.receivers[i]);
        xassert(params.receivers[i]->params.allocator == params.receivers[0]->params.allocator);
        xassert(params.receivers[i]->params.consumer_id == i);
    }

    this->frame_allocator = params.receivers[0]->params.allocator;

    // Verbose consistency check: the dedispersion config and the
    // frame_allocator must agree on time_samples_per_chunk. In the normal
    // run_server.py path the frame_allocator is constructed FROM the
    // dedispersion config, so this guards against future mis-wiring or
    // direct Python callers.
    if (params.config_prefilled.time_samples_per_chunk != frame_allocator->time_samples_per_chunk) {
        stringstream ss;
        ss << "FrbServer: DedispersionConfig::time_samples_per_chunk ("
           << params.config_prefilled.time_samples_per_chunk
           << ") does not match AssembledFrameAllocator::time_samples_per_chunk ("
           << frame_allocator->time_samples_per_chunk << ")";
        throw runtime_error(ss.str());
    }

    // Note: rpc_service is created in start(), not here, because we need shared_from_this().
}


FrbServer::~FrbServer()
{
    stop();

    for (auto &w : workers)
        if (w.joinable())
            w.join();

    if (reaper_thread.joinable())
        reaper_thread.join();

    if (processing_thread.joinable())
        processing_thread.join();

    if (rpc_server)
        rpc_server->Wait();
}


// Helper: returns true if the exception is a "normal shutdown signaling"
// exception (i.e., "called on stopped instance"). These get thrown by entry
// points like Receiver::get_frame_set / AssembledFrameAllocator::* once
// FrbServer::stop() has cascaded, and they're how the worker/reaper threads
// notice the shutdown and exit. We don't want to print them as errors.
static bool _is_cascade_stop_exception(const std::exception &e)
{
    return std::strstr(e.what(), "called on stopped instance") != nullptr;
}


void FrbServer::_throw_if_stopped(const char *method_name)
{
    if (error)
        std::rethrow_exception(error);

    if (is_stopped) {
        throw runtime_error(string(method_name) + " called on stopped instance");
    }
}


void FrbServer::start()
{
    unique_lock<std::mutex> lock(mutex);
    _throw_if_stopped("FrbServer::start");

    if (is_started)
        throw runtime_error("FrbServer::start() called twice");

    // Sanity check: all three async-eligible allocators must have finished
    // initializing before start(). If any of these is false the build code
    // forgot to call wait_until_initialized() before start() -- async-init
    // failures would surface from the processing/receiver threads later
    // rather than cleanly from the build path.
    xassert(frame_allocator->is_initialized());
    xassert(params.host_allocator->is_initialized());
    xassert(params.gpu_allocator->is_initialized());

    is_started = true;
    lock.unlock();

    try {
        // Create the RPC service (needs shared_from_this(), so can't be done in constructor).
        this->rpc_service = make_unique<FrbRpcService> (weak_from_this());

        // Start the RPC server.
        grpc::ServerBuilder builder;
        builder.AddListeningPort(params.rpc_server_address, grpc::InsecureServerCredentials());
        builder.RegisterService(rpc_service.get());
        this->rpc_server = builder.BuildAndStart();

        // Start all receivers.
        for (auto &r : params.receivers)
            r->start();

        // Spawn one worker thread per receiver.
        int nreceivers = params.receivers.size();
        for (int i = 0; i < nreceivers; i++)
            workers.emplace_back(&FrbServer::worker_main, this, i);

        // Spawn reaper thread iff frame_allocator is not in dummy mode.
        if (!frame_allocator->is_dummy())
            reaper_thread = std::thread(&FrbServer::reaper_thread_main, this);

        // Spawn processing thread (builds DedispersionPlan once metadata arrives).
        // Spawned unconditionally -- doesn't depend on dummy/non-dummy mode.
        processing_thread = std::thread(&FrbServer::processing_thread_main, this);
    } catch (...) {
        stop(std::current_exception());
        throw;
    }
}


void FrbServer::stop(std::exception_ptr e)
{
    unique_lock<std::mutex> lock(mutex);

    if (is_stopped)
        return;

    is_stopped = true;
    error = e;
    cv.notify_all();
    lock.unlock();

    // Stop all receivers.
    for (auto &r : params.receivers)
        r->stop();

    // Stop the frame_allocator (which will unblock num_total_frames).
    frame_allocator->stop();

    // Shutdown RPC server.
    if (rpc_server)
        rpc_server->Shutdown();
}


// -------------------------------------------------------------------------------------------------
//
// Worker threads


void FrbServer::_worker_main(int receiver_index)
{
    auto &receiver = params.receivers.at(receiver_index);
    long num_receivers = params.receivers.size();

    // Get the canonical metadata from the frame_allocator (blocking until
    // some Receiver's reader thread has parsed its first peer's YAML). The
    // frame_allocator's initialize_metadata() has already enforced
    // cross-sender consistency, so no check_sender_consistency() call is
    // needed here.
    shared_ptr<const XEngineMetadata> m = frame_allocator->get_metadata(true);

    // Also wait for the canonical initial_time_chunk to be established
    // (set by the first Receiver-reader to parse a per-minichunk header).
    // Frame ids in the FrbServer ringbuf are absolute -- frame_id =
    // time_chunk_index * nbeams + ibeam -- so the very first frame has
    // frame_id = initial_time_chunk * nbeams. We seed rb_* with that
    // offset below; the worker's frame_id loop starts there too.
    long initial_time_chunk = frame_allocator->wait_for_initial_chunk();
    long nbeams = m->get_nbeams();
    long rb_size = params.ringbuf_nchunks * nbeams;
    long initial_frame_id = initial_time_chunk * nbeams;

    unique_lock<std::mutex> lock(mutex);

    if (!metadata) {
        metadata = m;
        // The frame_ringbuf and beam_id_to_index are initialized at the same time
        // as the metadata, without dropping the lock in between. Correctness of
        // worker_main() and reaper_main() depend on this property.
        frame_ringbuf.resize(rb_size);
        for (int i = 0; i < nbeams; i++)
            beam_id_to_index[m->beam_ids[i]] = i;

        // Seed the ring-buffer indices at initial_frame_id (rather than 0).
        // Otherwise the worker's first install would fail the xassert
        // rb_end == frame_id below.
        rb_start        = initial_frame_id;
        rb_reaped       = initial_frame_id;
        rb_processed    = initial_frame_id;
        rb_assembled    = initial_frame_id;
        rb_end          = initial_frame_id;
        rb_initialized  = true;

        cv.notify_all();
    } else {
        xassert(long(frame_ringbuf.size()) == rb_size);
    }

    lock.unlock();

    // The Receiver now hands us one whole AssembledFrameSet (= one time
    // chunk, all beams) per call. Each outer iteration of the loop
    // processes one set: we fire cross-Receiver evict() once (each set
    // is a new time chunk by construction), then iterate the set's
    // nbeams frames into frame_ringbuf.
    long expected_frame_id = initial_frame_id;

    for (;;) {
        shared_ptr<AssembledFrameSet> set = receiver->get_frame_set();
        long ichunk = set->time_chunk_index;
        xassert(ichunk == expected_frame_id / nbeams);
        xassert(long(set->frames.size()) == nbeams);

        // Each get_frame_set() returns a fresh time chunk from this
        // Receiver, so we ask every other Receiver to evict the same
        // chunk index (keeping multiple Receivers synchronized so
        // chunk `ichunk` gets finalized promptly in the FrbServer
        // ring buffer downstream). evict() is non-blocking and
        // idempotent / monotone.
        for (size_t j = 0; j < params.receivers.size(); j++) {
            if (j == size_t(receiver_index))
                continue;
            params.receivers.at(j)->evict(ichunk);
        }

        // Insert this set's nbeams frames into frame_ringbuf. Each
        // frame must be received from all Receivers before it is
        // "finalized".
        for (long b = 0; b < nbeams; b++) {
            long frame_id = expected_frame_id + b;
            long rb_slot = frame_id % rb_size;
            long expected_beam_id = m->beam_ids.at(b);

            shared_ptr<AssembledFrame> frame = set->frames[b];
            xassert(frame->time_chunk_index == ichunk);
            xassert(frame->beam_id == expected_beam_id);

            lock.lock();

            // In principle, this assert can fail if one Receiver is running far
            // behind, OR if the GPU processing thread is running far behind. Either
            // way, something has gone off the rails and needs human debugging.
            // (Note: rb_processed <= rb_assembled, so asserting on rb_processed
            // subsumes the older "assembly keeps up" assertion as well.)
            xassert(rb_processed >= frame_id - rb_size + 1);

            unique_lock<std::mutex> frame_lock(frame->mutex);
            frame->finalize_count++;

            if (frame->finalize_count == 1) {
                // Frame received from first Receiver. Check that it is not in
                // the ringbuf already, and put it at the end of the ringbuf.
                xassert(rb_end == frame_id);
                frame_ringbuf[rb_slot] = frame;
                // Floor rb_start at initial_frame_id (not 0). Otherwise,
                // during ringbuf warm-up, rb_start would point below the
                // oldest actually-populated slot -- the formula
                // 'frame_id - rb_size + 1' assumes the ringbuf has been
                // filling since frame_id 0, but our worker seeds rb_start
                // at initial_frame_id (which can be large, e.g. when the
                // sender starts at a nonzero minichunk index). Floor at
                // initial_frame_id makes [rb_start, rb_end) contain only
                // populated slots in every phase.
                rb_start = max(frame_id - rb_size + 1, initial_frame_id);
                rb_reaped = max(rb_start, rb_reaped);
                rb_end = frame_id + 1;
                // Note that frame is not finalized yet (see below).
            }
            else {
                // Frame has previously been received from another Receiver.
                // Check that it is already in the ringbuf, but not assembled yet.
                xassert(frame_id >= rb_assembled);
                xassert(frame_id < rb_end);
                xassert(frame_ringbuf[rb_slot] == frame);
            }

            if (frame->finalize_count == num_receivers) {
                // Frame received from last Receiver, so mark it assembled.
                xassert(rb_assembled == frame_id);
                rb_assembled++;
            }

            _check_rb_invariants();

            lock.unlock();
            frame_lock.unlock();
            cv.notify_all();
        }

        expected_frame_id += nbeams;
    }
}


void FrbServer::worker_main(int receiver_index)
{
    try {
        _worker_main(receiver_index);
    } catch (const std::exception &e) {
        if (!_is_cascade_stop_exception(e)) {
            std::cerr << "FrbServer: worker thread " << receiver_index
                      << " terminated with exception: " << e.what() << std::endl;
        }
        stop(std::current_exception());
    } catch (...) {
        std::cerr << "FrbServer: worker thread " << receiver_index
                  << " terminated with unknown exception" << std::endl;
        stop(std::current_exception());
    }
}


// -------------------------------------------------------------------------------------------------
//
// Reaper thread


void FrbServer::_reaper_thread_main()
{
    // Wait for the canonical metadata via the frame_allocator. (Blocks
    // until some Receiver's reader thread has parsed its first peer's
    // YAML and called frame_allocator->initialize().)
    shared_ptr<const XEngineMetadata> m = frame_allocator->get_metadata(true);
    long nbeams = m->get_nbeams();

    // Compute rb_size from nbeams directly. Reading frame_ringbuf.size()
    // would be racy here: a FrbServer worker is responsible for the
    // frame_ringbuf.resize() call, and that may not have happened yet at
    // this point. The frame_ringbuf[rb_slot] access further down is still
    // safe because it sits inside the cv-wait on (rb_reaped < rb_processed)
    // -- for that to become true, some worker has assembled and the
    // processing thread has advanced past a frame, which means the first
    // worker has already executed its resize.
    long rb_size = params.ringbuf_nchunks * nbeams;

    // Get total number of frames (blocking until frame_allocator is initialized).
    long total_frames = frame_allocator->num_total_frames(/*blocking=*/ true);
    xassert(total_frames >= 6 * nbeams);

    for (;;) {
        frame_allocator->block_until_low_memory(2 * nbeams);

        unique_lock<std::mutex> lock(mutex);

        // Wait for a reapable frame (i.e. rb_reaped < rb_processed). The
        // processing thread signals the cv when it advances rb_processed.
        // Worker threads also signal on every frame insertion, but those
        // wakeups only become actionable for the reaper once the processing
        // thread has caught up (rb_processed has advanced).
        for (;;) {
            if (is_stopped)
                return;
            if (rb_reaped < rb_processed)
                break;
            cv.wait(lock);
        }

        long rb_slot = rb_reaped % rb_size;
        shared_ptr<AssembledFrame> frame = frame_ringbuf[rb_slot];
        rb_reaped++;
        lock.unlock();

        lock_guard<std::mutex> frame_lock(frame->mutex);
        frame->_reap_locked();
        // frame_lock dropped (by going out of scope)
    }
}


void FrbServer::reaper_thread_main()
{
    try {
        _reaper_thread_main();
    } catch (const std::exception &e) {
        if (!_is_cascade_stop_exception(e)) {
            std::cerr << "FrbServer: reaper thread terminated with exception: "
                      << e.what() << std::endl;
        }
        stop(std::current_exception());
    } catch (...) {
        std::cerr << "FrbServer: reaper thread terminated with unknown exception" << std::endl;
        stop(std::current_exception());
    }
}


// -------------------------------------------------------------------------------------------------
//
// Processing thread
//
// Pins to params.cuda_device_id first thing, then blocks until X-engine
// metadata is available. After metadata arrives, this thread overwrites
// the four metadata-dependent members of params.config_prefilled
// (zone_nfreq, zone_freq_edges, time_sample_ms, beams_per_gpu) to build a
// "postfilled" DedispersionConfig, validates it, and constructs a
// DedispersionPlan. It then builds a CudaStreamPool and a GpuDedisperser
// on top of the plan and calls allocate() to attach the FrbServer's
// host/gpu allocators. All three are published via the lock-protected
// 'plan' / 'dedisperser' members in a single critical section.
//
// After publishing plan + dedisperser, the thread enters a wait-and-advance
// loop: it blocks on cv until rb_processed < rb_assembled, advances
// rb_processed by one, and notifies cv (waking the reaper). The "processing"
// itself is trivial for now -- a future prompt will pull the current frame
// off the ringbuf and do real GPU work off-lock before advancing the counter.


void FrbServer::_processing_thread_main()
{
    // Pin this thread to the right CUDA device BEFORE any CudaStreamPool /
    // CudaEventRingbuf / cudaMemcpyAsync calls below: those all bind to
    // the "current device" of the thread that issues them. (Setting it
    // here also means the thread is parked on the correct device while
    // blocked in get_metadata() below; harmless but consistent.)
    CUDA_CALL(cudaSetDevice(params.cuda_device_id));

    // Block until X-engine metadata is available.
    // frame_allocator->get_metadata() is an entry point: it throws if the
    // frame_allocator has been stopped (which happens via cascade from
    // FrbServer::stop()), so we get a clean exit path through
    // processing_thread_main()'s try/catch.
    shared_ptr<const XEngineMetadata> m = frame_allocator->get_metadata(/*blocking=*/true);

    // Build config_postfilled by copying config_prefilled and overwriting the
    // four metadata-dependent members. Print a one-line message when an
    // overwrite changes the value.
    DedispersionConfig config_postfilled = params.config_prefilled;

    if (config_postfilled.zone_nfreq != m->zone_nfreq) {
        cout << "FrbServer: config_postfilled.zone_nfreq changed from "
             << tuple_str(config_postfilled.zone_nfreq) << " to "
             << tuple_str(m->zone_nfreq) << endl;
        config_postfilled.zone_nfreq = m->zone_nfreq;
    }

    if (config_postfilled.zone_freq_edges != m->zone_freq_edges) {
        cout << "FrbServer: config_postfilled.zone_freq_edges changed from "
             << tuple_str(config_postfilled.zone_freq_edges) << " to "
             << tuple_str(m->zone_freq_edges) << endl;
        config_postfilled.zone_freq_edges = m->zone_freq_edges;
    }

    // time_sample_ms is derived from XMD's seq-based timekeeping.
    double new_time_sample_ms = (double(m->dt_ns_per_seq) * double(m->seq_per_frb_time_sample)) / 1.0e6;
    if (config_postfilled.time_sample_ms != new_time_sample_ms) {
        cout << "FrbServer: config_postfilled.time_sample_ms changed from "
             << config_postfilled.time_sample_ms << " to " << new_time_sample_ms << endl;
        config_postfilled.time_sample_ms = new_time_sample_ms;
    }

    // beams_per_gpu = number of beams in the XMD.
    long new_beams_per_gpu = m->get_nbeams();
    if (config_postfilled.beams_per_gpu != new_beams_per_gpu) {
        cout << "FrbServer: config_postfilled.beams_per_gpu changed from "
             << config_postfilled.beams_per_gpu << " to " << new_beams_per_gpu << endl;
        config_postfilled.beams_per_gpu = new_beams_per_gpu;
    }

    // Will throw if (e.g.) beams_per_gpu is not divisible by beams_per_batch,
    // or if num_active_batches * beams_per_batch > beams_per_gpu. Exception
    // propagates to the wrapper which calls stop().
    config_postfilled.validate();

    // Construct the DedispersionPlan OUTSIDE the mutex -- this is the slow
    // step (can take seconds), and we don't want to block other threads
    // (e.g. RPC handlers reading 'plan' or 'is_stopped') while it runs.
    auto plan_p = make_shared<DedispersionPlan>(config_postfilled);
    cout << "FrbServer: DedispersionPlan constructed"
         << " (nfreq=" << plan_p->nfreq << ", ntrees=" << plan_p->ntrees << ")" << endl;

    // Build the CudaStreamPool and GpuDedisperser (also OUTSIDE the mutex;
    // these create CUDA streams and other GPU-side resources, so they should
    // not block lock-only RPC handlers).
    //
    // num_compute_streams = nstreams = config.num_active_batches by design:
    // one compute stream per active batch lets the dedispersion pipeline
    // overlap a batch's compute with the next batch's I/O. compute_stream_priority
    // = -1 (high priority) -- dedispersion compute is the time-critical path,
    // and the host<->device copy streams in the pool already run at the
    // default priority of 0.
    auto stream_pool = CudaStreamPool::create(
        /*num_compute_streams=*/ config_postfilled.num_active_batches,
        /*compute_stream_priority=*/ -1);

    // num_consumers=0: the downstream consumer of the dedispersion output is
    // not yet wired up (added in a future prompt). num_consumers=0 makes the
    // dedisperser drop outputs as soon as cdd2 produces them.
    //
    // detect_deadlocks=false: the FrbServer's processing thread drives
    // acquire_input / release_input_and_launch_dedispersion_kernels, and a separate downstream consumer
    // (added in a future prompt) drives acquire_output / release_output --
    // so the deadlock-detector's same-thread assumption would fire spuriously.
    GpuDedisperser::Params dd_params;
    dd_params.plan = plan_p;
    dd_params.stream_pool = stream_pool;
    dd_params.nbatches_out = config_postfilled.num_active_batches;
    dd_params.detect_deadlocks = false;
    dd_params.num_consumers = 0;
    dd_params.cuda_device_id = params.cuda_device_id;
    auto dedisperser_p = GpuDedisperser::create(dd_params);
    cout << "FrbServer: GpuDedisperser constructed"
         << " (nstreams=" << dedisperser_p->nstreams << ")" << endl;

    // Allocate GpuDedisperser resources from the FrbServer's dedicated
    // host/gpu BumpAllocators. allocate() also spawns the GpuDedisperser
    // worker thread, which sets cudaSetDevice on itself.
    dedisperser_p->allocate(*params.gpu_allocator, *params.host_allocator);
    cout << "FrbServer: GpuDedisperser::allocate() done"
         << " (gmem=" << dedisperser_p->resource_tracker.get_gmem_footprint() << " B"
         << ", hmem=" << dedisperser_p->resource_tracker.get_hmem_footprint() << " B)" << endl;

    // Publish plan + dedisperser atomically under the mutex. Notify cv so
    // future waiters (if any) observe the new values.
    {
        lock_guard<std::mutex> lock(mutex);
        plan = plan_p;
        dedisperser = dedisperser_p;
        cv.notify_all();
    }

    // Wait-and-advance loop. Mirrors the reaper pattern: cv-wait until
    // either is_stopped or rb_processed < rb_assembled, then bump
    // rb_processed by one and notify cv (waking the reaper). The
    // "processing" is trivial for now; a future prompt will pull
    // frame_ringbuf[rb_processed % rb_size] under the lock, drop the
    // lock to issue real GPU work off-lock, then re-take the lock to
    // advance rb_processed.
    for (;;) {
        unique_lock<std::mutex> lock(mutex);

        for (;;) {
            if (is_stopped)
                return;
            if (rb_processed < rb_assembled)
                break;
            cv.wait(lock);
        }

        // Optional artificial per-frame delay (config: processing_delay_sec).
        // Used to simulate slow GPU processing for testing the FakeXEngine
        // pacing path. Done off-lock so workers can keep advancing
        // rb_assembled during the "work".
        if (params.processing_delay_sec > 0.0) {
            lock.unlock();
            std::this_thread::sleep_for(
                std::chrono::duration<double>(params.processing_delay_sec));
            lock.lock();
        }

        rb_processed++;
        _check_rb_invariants();

        lock.unlock();
        cv.notify_all();
    }
}


void FrbServer::processing_thread_main()
{
    try {
        _processing_thread_main();
    } catch (const std::exception &e) {
        if (!_is_cascade_stop_exception(e)) {
            std::cerr << "FrbServer: processing thread terminated with exception: "
                      << e.what() << std::endl;
        }
        stop(std::current_exception());
    } catch (...) {
        std::cerr << "FrbServer: processing thread terminated with unknown exception" << std::endl;
        stop(std::current_exception());
    }
}


// ------------------------------------------------------------------------------------------
//
// FrbRpcService implementation

    
// Helper to lock the weak_ptr. Throws if the server is exiting.
shared_ptr<FrbServer> FrbRpcService::_lock_state()
{
    auto s = state.lock();
    if (!s)
        throw runtime_error("FrbServer is in the process of exiting");
    return s;
}

// ---- GetStatus ----

void FrbRpcService::_GetStatus(const fs::GetStatusRequest *request, fs::GetStatusResponse *response)
{
    shared_ptr<FrbServer> s = _lock_state();

    // Call Receiver::get_status() for each receiver,
    // and sum the results over receivers.
    long total_conn = 0, total_bytes = 0;
    for (auto &r : s->params.receivers) {
        long nc, nb;
        r->get_status(nc, nb);
        total_conn += nc;
        total_bytes += nb;
    }
    response->set_num_connections(total_conn);
    response->set_num_bytes(total_bytes);

    // Get ring buffer state under lock.
    {
        lock_guard<std::mutex> lock(s->mutex);
        response->set_rb_start(s->rb_start);
        response->set_rb_reaped(s->rb_reaped);
        response->set_rb_processed(s->rb_processed);
        response->set_rb_assembled(s->rb_assembled);
        response->set_rb_end(s->rb_end);
    }

    // Get num_free_frames from the allocator (permissive=true to handle uninitialized state).
    auto &allocator = s->params.receivers[0]->params.allocator;
    response->set_num_free_frames(allocator->num_free_frames(/*permissive=*/true));
}

grpc::Status FrbRpcService::GetStatus(
    grpc::ServerContext* context,
    const fs::GetStatusRequest* request,
    fs::GetStatusResponse* response) 
{
    try {
        _GetStatus(request, response);
        return grpc::Status::OK;
    } catch (const std::exception &e) {
        return grpc::Status(grpc::StatusCode::INTERNAL, e.what());
    } catch (...) {
        return grpc::Status(grpc::StatusCode::INTERNAL, "Unknown error in GetStatus");
    }
}

// ---- GetXEngineMetadata ----

void FrbRpcService::_GetXEngineMetadata(const fs::GetXEngineMetadataRequest *request, fs::GetXEngineMetadataResponse *response)
{
    shared_ptr<FrbServer> s = _lock_state();

    // Pull the canonical metadata from the allocator (non-blocking; if no
    // peer has yet sent YAML, returns nullptr and we return an empty
    // yaml_string to the RPC client).
    shared_ptr<const XEngineMetadata> m = s->frame_allocator->get_metadata(/*blocking=*/false);
    if (m)
        response->set_yaml_string(m->to_yaml_string(request->verbose()));
    else
        response->set_yaml_string("");
}

grpc::Status FrbRpcService::GetXEngineMetadata(
    grpc::ServerContext* context,
    const fs::GetXEngineMetadataRequest* request,
    fs::GetXEngineMetadataResponse* response)
{
    try {
        _GetXEngineMetadata(request, response);
        return grpc::Status::OK;
    } catch (const std::exception &e) {
        return grpc::Status(grpc::StatusCode::INTERNAL, e.what());
    } catch (...) {
        return grpc::Status(grpc::StatusCode::INTERNAL, "Unknown error in GetXEngineMetadata");
    }
}

// ---- WriteFiles ----

void FrbRpcService::_WriteFiles(const fs::WriteFilesRequest *request, fs::WriteFilesResponse *response)
{
    shared_ptr<FrbServer> s = _lock_state();
    shared_ptr<FileWriter> file_writer = s->params.file_writer;

    // Convert beam_ids to beam_indices.
    vector<int> beam_indices;
    beam_indices.reserve(request->beams_size());

    for (int i = 0; i < request->beams_size(); i++) {
        long beam_id = request->beams(i);
        auto it = s->beam_id_to_index.find(beam_id);
        if (it == s->beam_id_to_index.end()) {
            stringstream ss;
            ss << "WriteFiles: unknown beam_id " << beam_id;
            throw runtime_error(ss.str());
        }
        beam_indices.push_back(it->second);
    }

    // Validate time_chunk_index range.
    long min_time_chunk_index = request->min_time_chunk_index();
    long max_time_chunk_index = request->max_time_chunk_index();

    if ((min_time_chunk_index < 0) || (min_time_chunk_index > max_time_chunk_index)) {
        stringstream ss;
        ss << "WriteFiles: invalid time_chunk_index range [" << min_time_chunk_index
            << ", " << max_time_chunk_index << "]";
        throw runtime_error(ss.str());
    }

    // Construct FilenamePattern (validates that pattern contains "(BEAM)" and "(CHUNK)").
    FilenamePattern filename_pattern(request->filename_pattern());

    // Get frames from ring buffer.
    // We compute frame_id = time_chunk_index * nbeams + beam_index directly for each
    // (time_chunk_index, beam_index) pair, rather than iterating over the entire ring
    // buffer and checking each frame's metadata. This is O(num_time_chunks * num_beams)
    // instead of O(ringbuf_size).

    vector<shared_ptr<AssembledFrame>> local_frames;

    long num_time_chunks = max_time_chunk_index - min_time_chunk_index + 1;
    local_frames.reserve(num_time_chunks * beam_indices.size());

    // Pull nbeams from the canonical metadata (frame_allocator) before
    // taking FrbServer::mutex. If metadata is not yet available, there are
    // no frames to write and the filename list returned to the client is
    // empty.
    shared_ptr<const XEngineMetadata> m = s->frame_allocator->get_metadata(/*blocking=*/false);
    if (!m)
        return;
    long rb_nbeams = m->get_nbeams();

    unique_lock<std::mutex> server_lock(s->mutex);

    // Use rb_processed (not rb_assembled) as the upper bound: frames in
    // [rb_processed, rb_assembled) are fully assembled but the GPU may
    // still be mutating them, so they are NOT rpc-writeable.
    long rb_size      = s->frame_ringbuf.size();
    long rb_start     = s->rb_start;
    long rb_processed = s->rb_processed;

    min_time_chunk_index = max(min_time_chunk_index, rb_start / rb_nbeams);
    max_time_chunk_index = min(max_time_chunk_index, rb_processed / rb_nbeams);

    for (long t = min_time_chunk_index; t <= max_time_chunk_index; t++) {
        for (int b : beam_indices) {
            long frame_id = t * rb_nbeams + b;
            if ((frame_id >= rb_start) && (frame_id < rb_processed)) {
                long rb_slot = frame_id % rb_size;
                local_frames.push_back(s->frame_ringbuf[rb_slot]);
            }
        }
    }

    server_lock.unlock();

    // Process frames in reverse order: push filenames onto frame->save_paths,
    // then call file_writer->process_frame(). Reverse order ensures that frames
    // with lower time_chunk_index are processed last (and appear earlier in queues).

    vector<string> filename_list;
    filename_list.reserve(local_frames.size());

    for (auto it = local_frames.rbegin(); it != local_frames.rend(); ++it) {
        auto &frame = *it;

        unique_lock<std::mutex> frame_lock(frame->mutex);

        // Skip if frame has been reaped without ever being written.
        // If save_paths is non-empty, then either the data is still
        // in memory (data.size > 0), or it's on disk (on_ssd, or
        // copied to NFS and tracked via nfs_count). In the latter
        // case FileWriter's NFS thread can hardlink from the
        // primary save_path to a new save_path (see _nfs_thread_main
        // -- both _hardlink_in_nfs and the save_error path operate
        // off save_paths[0]). So we should only skip when save_paths
        // is empty AND data has been reaped.
        if (frame->data.size == 0 && frame->save_paths.empty())
            continue;

        string filename = filename_pattern.expand(frame);
        frame->save_paths.push_back(filename);
        frame_lock.unlock();

        file_writer->process_frame(frame);
        filename_list.push_back(std::move(filename));
    }

    // Return filename list to RPC caller.
    for (const auto &fn : filename_list)
        response->add_filename_list(fn);
}

grpc::Status FrbRpcService::WriteFiles(
    grpc::ServerContext* context,
    const fs::WriteFilesRequest* request,
    fs::WriteFilesResponse* response) 
{
    try {
        _WriteFiles(request, response);
        return grpc::Status::OK;
    } catch (const std::exception &e) {
        return grpc::Status(grpc::StatusCode::INTERNAL, e.what());
    } catch (...) {
        return grpc::Status(grpc::StatusCode::INTERNAL, "Unknown error in WriteFiles");
    }
}

// ---- SubscribeFiles ----

// Subscribes to file write notifications from FileWriter.
// If the client closes the connection, exits gracefully.
//
// Note: gRPC's server-streaming model provides one thread of execution per connection
// via gRPC's internal thread pool.
//
// Each response has (filename, error_message). Empty error_message indicates success.
void FrbRpcService::_SubscribeFiles(grpc::ServerContext* context, grpc::ServerWriter<fs::SubscribeFilesResponse>* writer)
{
    shared_ptr<FrbServer> s = _lock_state();
    shared_ptr<FileWriter> file_writer = s->params.file_writer;

    // Create subscriber and register with FileWriter.
    auto subscriber = make_shared<FileWriter::RpcSubscriber>();
    file_writer->add_subscriber(subscriber);

    // Ready sentinel. Must be the FIRST response on every stream --
    // the client constructor reads this synchronously to confirm
    // subscription before returning. Ordering matters: add_subscriber()
    // runs before this Write(), so every notification that pops out of
    // subscriber->queue below was enqueued by a write that completed
    // AFTER the client had a chance to issue (and observe) any
    // subsequent WriteFiles call.
    {
        fs::SubscribeFilesResponse ready;
        ready.mutable_ready();          // sets the oneof variant
        if (!writer->Write(ready))
            return;
    }

    // Loop over entries in subscriber's queue.
    for (;;) {
        // Check if client has disconnected.
        if (context->IsCancelled())
            return;

        unique_lock<std::mutex> subscriber_lock(subscriber->mutex);

        // Wait for queue entry, stop signal, or error (with 0.5 sec timeout).
        // Timeout ensures context->IsCancelled() is checked regularly.
        subscriber->cv.wait_for(subscriber_lock, std::chrono::milliseconds(500), [&subscriber]() {
            return !subscriber->queue.empty() || subscriber->is_stopped || subscriber->error;
        });

        // Check for error condition.
        if (subscriber->error)
            std::rethrow_exception(subscriber->error);

        // If queue is empty, loop back to check context->IsCancelled().
        if (subscriber->queue.empty()) {
            if (subscriber->is_stopped)
                return;
            continue;
        }

        // Pop entry from queue.
        FileWriter::WriteStatus write_status = std::move(subscriber->queue.front());
        subscriber->queue.pop();

        subscriber_lock.unlock();

        // Build response with (filename, error_message) inside the
        // 'notification' arm of the oneof. Empty error_message
        // indicates success.
        fs::SubscribeFilesResponse response;
        auto *notif = response.mutable_notification();
        notif->set_filename(write_status.save_path.string());

        if (write_status.error) {
            try {
                std::rethrow_exception(write_status.error);
            } catch (const std::exception &e) {
                // e.what() may return empty string; replace with "Unknown error".
                const char *msg = e.what();
                notif->set_error_message((msg && msg[0]) ? msg : "Unknown error");
            } catch (...) {
                notif->set_error_message("Unknown error");
            }
        } else {
            notif->set_error_message("");  // Empty error_message indicates success.
        }

        // Write() returns false if the stream has been closed by the client.
        if (!writer->Write(response))
            return;
    }
}

grpc::Status FrbRpcService::SubscribeFiles(
    grpc::ServerContext* context,
    const fs::SubscribeFilesRequest* request,
    grpc::ServerWriter<fs::SubscribeFilesResponse>* writer)
{
    try {
        _SubscribeFiles(context, writer);
        return grpc::Status::OK;
    } catch (const std::exception &e) {
        return grpc::Status(grpc::StatusCode::INTERNAL, e.what());
    } catch (...) {
        return grpc::Status(grpc::StatusCode::INTERNAL, "Unknown error in SubscribeFiles");
    }
}

// ---- GetConfig ----

void FrbRpcService::_GetConfig(const fs::GetConfigRequest *request, fs::GetConfigResponse *response)
{
    shared_ptr<FrbServer> s = _lock_state();

    response->set_rpc_ip_addr(s->params.rpc_server_address);

    for (auto &r : s->params.receivers)
        response->add_data_ip_addrs(r->params.address);

    response->set_time_samples_per_chunk(s->frame_allocator->time_samples_per_chunk);
    response->set_ringbuf_nchunks(s->params.ringbuf_nchunks);
    response->set_min_data_mtu(s->params.min_data_mtu);

    const auto &fwp = s->params.file_writer->params;
    response->set_ssd_dir(fwp.ssd_root.string());
    response->set_nfs_dir(fwp.nfs_root.string());
    response->set_ssd_threads(fwp.num_ssd_threads);
    response->set_nfs_threads(fwp.num_nfs_threads);

    // Dedispersion / fake X-engine fields from the pre-metadata config.
    // (The processing thread overwrites four members of config_prefilled
    // into a local config_postfilled; the prefilled values are what the
    // fake X-engine sender should mimic.)
    const DedispersionConfig &c = s->params.config_prefilled;
    response->set_tree_rank(c.tree_rank);
    response->set_beams_per_batch(c.beams_per_batch);
    for (long v : c.zone_nfreq)
        response->add_fake_zone_nfreq(v);
    for (double v : c.zone_freq_edges)
        response->add_fake_zone_freq_edges(v);
    response->set_fake_time_sample_ms(c.time_sample_ms);
    response->set_fake_nbeams(c.beams_per_gpu);
}

grpc::Status FrbRpcService::GetConfig(
    grpc::ServerContext* context,
    const fs::GetConfigRequest* request,
    fs::GetConfigResponse* response)
{
    try {
        _GetConfig(request, response);
        return grpc::Status::OK;
    } catch (const std::exception &e) {
        return grpc::Status(grpc::StatusCode::INTERNAL, e.what());
    } catch (...) {
        return grpc::Status(grpc::StatusCode::INTERNAL, "Unknown error in GetConfig");
    }
}

// ---- MonitorRingbuf ----
//
// Special-purpose push stream of rb_processed updates, intended for use by
// the FakeXEngine "pacing" feature (see plans/fake_xengine_pacing.md). The
// handler sends one message as soon as rb_initialized becomes true (carrying
// the current value of rb_processed), then one message per change. The stream
// ends when the client closes the connection or when FrbServer::stop() is
// called.
//
// Synchronization: reuses FrbServer::mutex + FrbServer::cv. The cv is already
// notified on every event that matters (rb_initialized flip, rb_processed
// advance, stop). Spurious wake-ups (e.g., from worker frame-insert notifies)
// just re-check the predicate and re-wait; the cost is microseconds per notify.
//
// The 500ms wait timeout is a safety net for "silent disconnect during an
// idle server" -- if a client disconnects with no Cancel() call AND
// rb_processed is not advancing, we'd otherwise block forever on cv.wait.
// The timeout lets us periodically re-poll context->IsCancelled().
void FrbRpcService::_MonitorRingbuf(
    grpc::ServerContext* context,
    grpc::ServerWriter<fs::MonitorRingbufResponse>* writer)
{
    shared_ptr<FrbServer> s = _lock_state();

    // rb_processed is always >= 0; -1 is a safe sentinel for "no value sent yet".
    long last_sent = -1;

    for (;;) {
        if (context->IsCancelled())
            return;

        long current;
        {
            unique_lock<std::mutex> lock(s->mutex);

            // Wait for: stop, OR (initialized AND value changed). Timeout
            // bounds the latency of silent-disconnect detection.
            s->cv.wait_for(lock, std::chrono::milliseconds(500), [&]() {
                return s->is_stopped
                    || (s->rb_initialized && s->rb_processed != last_sent);
            });

            if (s->is_stopped)
                return;
            if (!s->rb_initialized)
                continue;   // pre-init: timeout fired, nothing to send
            if (s->rb_processed == last_sent)
                continue;   // timeout fired but value unchanged

            current = s->rb_processed;
        }

        // Send outside the lock (Write is blocking I/O).
        fs::MonitorRingbufResponse response;
        response.set_rb_processed(current);
        if (!writer->Write(response))
            return;   // client closed the stream
        last_sent = current;
    }
}

grpc::Status FrbRpcService::MonitorRingbuf(
    grpc::ServerContext* context,
    const fs::MonitorRingbufRequest* request,
    grpc::ServerWriter<fs::MonitorRingbufResponse>* writer)
{
    try {
        _MonitorRingbuf(context, writer);
        return grpc::Status::OK;
    } catch (const std::exception &e) {
        return grpc::Status(grpc::StatusCode::INTERNAL, e.what());
    } catch (...) {
        return grpc::Status(grpc::StatusCode::INTERNAL, "Unknown error in MonitorRingbuf");
    }
}


}  // namespace pirate
