#include "../include/pirate/FrbServer.hpp"
#include "../include/pirate/BumpAllocator.hpp"    // gpu_allocator / host_allocator (complete type)
#include "../include/pirate/FileWriter.hpp"
#include "../include/pirate/DedispersionPlan.hpp"
#include "../include/pirate/Dedisperser.hpp"        // GpuDedisperser
#include "../include/pirate/CudaStreamPool.hpp"
#include "../include/pirate/CudaEventRingbuf.hpp"
#include "../include/pirate/GpuDequantizationKernel.hpp"
#include "../include/pirate/utils.hpp"               // safe_memcpy_h2g_async
#include "../include/pirate/network_utils.hpp"       // parse_ip_address, is_loopback_address

#include <cstring>   // strstr

#include <ksgpu/string_utils.hpp>  // tuple_str()

#include <algorithm> // find (stream beam matching), remove_if (stream expiry)
#include <chrono>    // duration<double> (processing-thread delay) + ms (MonitorRingbuf timeout)
#include <thread>    // this_thread::sleep_for (processing-thread delay)
#include <iomanip>   // setprecision (throughput suffix on the per-chunk line)
#include <iostream>
#include <sstream>
#include <stdexcept>

// grpc/protobuf headers pull in conda-forge's libabseil, which was built
// with -DNDEBUG. absl::Mutex::Dtor() is only inlined when NDEBUG is
// defined at the include site; otherwise it becomes an undefined external
// symbol that the abseil DSO does not export, and libpirate.so fails to
// load. So we push_macro NDEBUG on, include grpc, then pop it back. See
// notes/build.md.
#pragma push_macro("NDEBUG")
#ifndef NDEBUG
#  define NDEBUG
#endif
// Silence -Wdeprecated-declarations warnings emitted from grpc's own public
// headers (e.g. IdentityKeyCertPair, set_certificate_provider). They come
// from grpc-internal code that pirate does not call directly.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "../grpc/frb_search.grpc.pb.h"
#include "../grpc/frb_grouper.grpc.pb.h"
#include "../grpc/frb_grouper.pb.h"
#include <grpcpp/grpcpp.h>
#pragma GCC diagnostic pop
#pragma pop_macro("NDEBUG")

using namespace std;
using namespace ksgpu;


namespace pirate {
#if 0
}  // editor auto-indent
#endif

namespace fs = frb::search::v1;
namespace fg = frb::grouper::v1;


// pImpl for the FrbGrouper gRPC client state (see FrbServer.hpp). Holds the
// channel, stub, ClientContext, and the bidi ClientReaderWriter for the active
// Session RPC. Built + published by grouper_send_thread; cancelled by stop().
struct FrbServer::GrouperClient {
    std::shared_ptr<grpc::Channel> channel;
    std::unique_ptr<fg::FrbGrouper::Stub> stub;
    std::unique_ptr<grpc::ClientContext> context;
    std::unique_ptr<grpc::ClientReaderWriter<fg::ProducerMessage,
                                             fg::ConsumerMessage>> stream;

    // Idempotent; safe to call from FrbServer::stop() on any thread. TryCancel()
    // unblocks any in-flight stream->Read / stream->Write (they return false).
    void cancel() { if (context) context->TryCancel(); }
};


// GRPC service implementation. See bottom of file for implementations of individual RPCs.

class FrbRpcService final : public fs::FrbSearch::Service {
public:
    std::weak_ptr<FrbServer> state;

    FrbRpcService(const weak_ptr<FrbServer> &s) : state(s) {}

    // These functions implement the individual RPCs.
    void _GetStatus(const fs::GetStatusRequest *request, fs::GetStatusResponse *response);
    void _GetXEngineMetadata(const fs::GetXEngineMetadataRequest *request, fs::GetXEngineMetadataResponse *response);
    void _WriteFiles(const fs::WriteFilesRequest *request, fs::WriteFilesResponse *response);
    void _SubscribeFiles(grpc::ServerContext* context, const fs::SubscribeFilesRequest *request, grpc::ServerWriter<fs::SubscribeFilesResponse>* writer);
    void _GetConfig(const fs::GetConfigRequest *request, fs::GetConfigResponse *response);
    void _MonitorRingbuf(grpc::ServerContext* context, grpc::ServerWriter<fs::MonitorRingbufResponse>* writer);
    void _StartStream(const fs::StartStreamRequest *request, fs::StartStreamResponse *response);
    void _ShowStreams(const fs::ShowStreamsRequest *request, fs::ShowStreamsResponse *response);
    void _CancelStream(const fs::CancelStreamRequest *request, fs::CancelStreamResponse *response);

    // Helper to lock the weak_ptr. Throws if the server is exiting.
    inline shared_ptr<FrbServer> _lock_state();

    // Try-catch wrappers, to gracefully return an error status to the client
    // (instead of crashing the server). Each wrapper first checks the request's
    // protocol_version against the server's PROTOCOL_VERSION_CURRENT (see
    // _check_protocol_version); a mismatch throws and is reported to the client.

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

    grpc::Status StartStream(
        grpc::ServerContext* context,
        const fs::StartStreamRequest* request,
        fs::StartStreamResponse* response) override;

    grpc::Status ShowStreams(
        grpc::ServerContext* context,
        const fs::ShowStreamsRequest* request,
        fs::ShowStreamsResponse* response) override;

    grpc::Status CancelStream(
        grpc::ServerContext* context,
        const fs::CancelStreamRequest* request,
        fs::CancelStreamResponse* response) override;
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

    // --no-dedispersion implies --no-grouper: the processing thread skips all
    // GPU work (so the dedisperser produces no output), leaving nothing for a
    // grouper to consume. run_server.py enforces this by forcing an empty
    // grouper_ip_addr; assert it here to catch direct Python callers / future
    // mis-wiring.
    if (params.no_dedispersion)
        xassert(params.grouper_ip_addr.empty());

    if (!params.grouper_ip_addr.empty()) {
        // CUDA IPC requires producer + consumer on the same physical GPU, so the
        // grouper must be local. Enforce a loopback address.
        string ip; uint16_t port;
        parse_ip_address(params.grouper_ip_addr, ip, port);   // network_utils.hpp
        if (!is_loopback_address(ip)) {
            stringstream ss;
            ss << "FrbServer: grouper_ip_addr=" << params.grouper_ip_addr
               << " is not a loopback address (CUDA IPC requires the grouper to be "
               << "on the same node / GPU)";
            throw runtime_error(ss.str());
        }

        // The handshake exports an IPC handle via params.gpu_allocator->get_base().
        // Reject a dummy (capacity < 0) OR empty (capacity == 0) gpu_allocator early
        // so the error is at construction rather than mid-handshake on a worker
        // thread. Dummy mode makes get_base() throw; empty mode makes it return a
        // null shared_ptr (cudaIpcGetMemHandle(nullptr) would then fail). Neither
        // can back output_ringbuf, so require capacity > 0.
        if (params.gpu_allocator->capacity <= 0) {
            throw runtime_error("FrbServer: grouper_ip_addr requires a non-empty, "
                                "non-dummy gpu_allocator (CUDA IPC needs a real GPU "
                                "allocation; capacity < 0 is dummy mode, capacity == 0 "
                                "allocates nothing)");
        }
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

    if (frame_finalizing_thread.joinable())
        frame_finalizing_thread.join();

    if (grouper_send_thread.joinable())
        grouper_send_thread.join();

    if (grouper_receive_thread.joinable())
        grouper_receive_thread.join();

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

        // Spawn one worker thread per receiver.
        int nreceivers = params.receivers.size();
        for (int i = 0; i < nreceivers; i++)
            workers.emplace_back(&FrbServer::worker_main, this, i);

        // Spawn reaper thread iff frame_allocator is not in dummy mode.
        if (!frame_allocator->is_dummy())
            reaper_thread = std::thread(&FrbServer::reaper_thread_main, this);

        // Spawn processing thread (builds DedispersionPlan once metadata arrives)
        // and the frame-finalizing thread. Spawned unconditionally -- don't
        // depend on dummy/non-dummy mode.
        processing_thread       = std::thread(&FrbServer::processing_thread_main,       this);
        frame_finalizing_thread = std::thread(&FrbServer::frame_finalizing_thread_main, this);

        if (params.grouper_ip_addr.empty()) {
            // No grouper: start receivers now.
            for (auto &r : params.receivers)
                r->start();
        } else {
            // Grouper enabled: do NOT start receivers here. grouper_send_thread
            // starts them after the grouper connection is established (so we
            // don't ingest data until the grouper is connected -- see the
            // "Thread ordering" discussion in plans/grouper_client.md).
            grouper_send_thread    = std::thread(&FrbServer::grouper_send_thread_main,    this);
            grouper_receive_thread = std::thread(&FrbServer::grouper_receive_thread_main, this);
        }
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

    // Snapshot the lock-protected shared_ptrs under the lock (the
    // processing_thread publishes them under this same mutex). We call their
    // stop()s below, after unlocking -- reading them outside the lock would be
    // a data race, and we don't want to hold 'mutex' across the cascades.
    shared_ptr<GpuDedisperser>   dd   = dedisperser;
    shared_ptr<CudaEventRingbuf> edq  = evrb_dq;
    shared_ptr<CudaEventRingbuf> eh2g = evrb_h2g;

    // Snapshot the grouper client pointer under the lock. grouper_send_thread
    // publishes grouper_client under this same mutex *after* checking is_stopped,
    // so either we ran first (it never publishes) or it published before this
    // snapshot -- no lost wakeup. The raw pointer stays valid for the duration of
    // stop() because grouper_client is destroyed only in ~FrbServer (after joins).
    GrouperClient *gc = grouper_client.get();

    cv.notify_all();
    lock.unlock();

    // Cancel the grouper Session RPC: TryCancel() unblocks any in-flight
    // stream->Read (receive thread) / stream->Write (send thread).
    if (gc) gc->cancel();

    // Unblock the two FrbServer-owned evrbs.
    if (edq) edq->stop();
    if (eh2g) eh2g->stop();

    // Stop all receivers.
    for (auto &r : params.receivers)
        r->stop();

    // Stop the frame_allocator (which will unblock num_total_frames).
    frame_allocator->stop();

    // Cascade to the dedisperser. GpuDedisperser::stop() also stops the
    // dedisperser's internal CudaEventRingbufs, which is how the
    // processing_thread is unblocked if it is parked in a blocking evrb_et_h2g
    // wait inside release_input_and_launch_dd_kernels.
    if (dd)
        dd->stop();

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
            // (The per-chunk "FrbServer: ..." status line is printed by the
            // frame_finalizing_thread, when a chunk is fully PROCESSED by the GPU,
            // not here at network-assembly time.)
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
// host/gpu allocators. plan + dedisperser (along with evrb_dq /
// evrb_h2g and dedisperser_is_initialized) are published via the
// lock-protected members in a single critical section.
//
// After publishing, the thread allocates per-stream GPU scratch + a
// GpuDequantizationKernel, snapshots rb_curr = rb_processed once
// rb_initialized is true, then loops over (ichunk, ibatch, beam): per-beam
// H2G copy of the assembled frame into the scratch (bumping the local
// rb_curr), per-batch dequantization into the dedisperser input buffer, and
// per-batch release_input_and_launch_dd_kernels. It does NOT bump
// rb_processed -- the frame_finalizing_thread does that (in batches of
// beams_per_batch) once the H2G copies are observably complete on the GPU.


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
    auto t_plan0 = std::chrono::steady_clock::now();
    auto plan_p = make_shared<DedispersionPlan>(config_postfilled);
    auto t_plan1 = std::chrono::steady_clock::now();
    double dt_plan = std::chrono::duration<double>(t_plan1 - t_plan0).count();
    cout << "FrbServer: DedispersionPlan constructed in " << dt_plan << " sec"
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

    // num_consumers: 1 when a grouper is configured (the grouper_send_thread is
    // the single output consumer), else 0 (the dedisperser drops outputs as soon
    // as cdd2 produces them). synchronous=false: the consumer's acquire_output
    // (send thread) and release_output (receive thread) run on different threads,
    // and the send thread legitimately gets several seq_ids ahead of the receive
    // thread (pipeline depth ~ nbatches_out).
    GpuDedisperser::Params dd_params;
    dd_params.plan = plan_p;
    dd_params.stream_pool = stream_pool;
    dd_params.nbatches_out = 2 * config_postfilled.num_active_batches;  // leave a little headroom for grouper
    dd_params.nbatches_wt = (params.nbatches_wt > 0) ? params.nbatches_wt
                                                     : config_postfilled.num_active_batches;
    dd_params.num_consumers = params.grouper_ip_addr.empty() ? 0 : 1;
    dd_params.synchronous = false;
    dd_params.cuda_device_id = params.cuda_device_id;
    
    // initial_chunk: the canonical initial_time_chunk (time-chunk index of the
    // very first frame, relative to FPGA seq 0). Lets the dedisperser stamp each
    // output with its FPGA-based chunk index (Outputs::ichunk_fpga_based). Blocks
    // until the first Receiver-reader has parsed a per-minichunk header; harmless
    // here since this thread already blocked on get_metadata() above.
    dd_params.initial_chunk = frame_allocator->wait_for_initial_chunk();
    auto t_dd0 = std::chrono::steady_clock::now();
    auto dedisperser_p = GpuDedisperser::create(dd_params);
    auto t_dd1 = std::chrono::steady_clock::now();
    double dt_dd = std::chrono::duration<double>(t_dd1 - t_dd0).count();
    cout << "FrbServer: GpuDedisperser constructed in " << dt_dd << " sec"
         << " (nstreams=" << dedisperser_p->nstreams << ")" << endl;

    // Allocate GpuDedisperser resources from the FrbServer's dedicated
    // host/gpu BumpAllocators. allocate() also spawns the GpuDedisperser
    // worker thread, which sets cudaSetDevice on itself.
    auto t_alloc0 = std::chrono::steady_clock::now();
    dedisperser_p->allocate(*params.gpu_allocator, *params.host_allocator);
    auto t_alloc1 = std::chrono::steady_clock::now();
    double dt_alloc = std::chrono::duration<double>(t_alloc1 - t_alloc0).count();
    cout << "FrbServer: GpuDedisperser::allocate() done in " << dt_alloc << " sec"
         << " (gmem=" << dedisperser_p->resource_tracker.get_gmem_footprint() << " B"
         << ", hmem=" << dedisperser_p->resource_tracker.get_hmem_footprint() << " B)" << endl;

    // One-time fill of the dedisperser's peak-finding weights from analytic variances,
    // done after allocate() but before the dedisperser is "published" below (so no other
    // thread can touch the weights concurrently). 'freq_variances' is the per-channel noise
    // variance, expanded from the X-engine metadata's per-zone noise_variance.
    {
        const long nfreq = m->get_total_nfreq();
        const long nzones = (long) m->zone_nfreq.size();
        xassert_eq((long) m->noise_variance.size(), nzones);

        Array<double> freq_variances({nfreq}, af_uhost);
        long ifreq = 0;
        for (long z = 0; z < nzones; z++)
            for (long i = 0; i < m->zone_nfreq[z]; i++)
                freq_variances.data[ifreq++] = m->noise_variance[z];
        xassert_eq(ifreq, nfreq);

        cout << "FrbServer: calling fill_analytic_weights() ..." << endl;
        auto t0 = std::chrono::steady_clock::now();
        dedisperser_p->fill_analytic_weights(freq_variances);
        auto t1 = std::chrono::steady_clock::now();
        double dt = std::chrono::duration<double>(t1 - t0).count();
        cout << "FrbServer: filled analytic dedisperser weights in " << dt << " sec" << endl;
    }

    // Per-stream GPU scratch + dequantization kernel + the two new
    // CudaEventRingbufs. These are local to processing_thread; the evrb_*'s
    // are "published" to FrbServer's lock-protected members below.
    //
    // B = beams_per_batch, F = nfreq, T = nt_in, S = nstreams.
    const long S         = dedisperser_p->nstreams;
    const long B         = dedisperser_p->beams_per_batch;
    const long F         = dedisperser_p->nfreq;
    const long T         = dedisperser_p->nt_in;
    const Dtype dtype    = dedisperser_p->dtype;
    const Dtype dt_int4  = Dtype::from_str("int4");

    // Note that we allocate these arrays after calling GpuDedisperser::allocate().
    // This is so that GpuDedisperser::output_ringbuf will be located as close as
    // possible to the gpu_allocator 'base' pointer. This seems preferable, since
    // we plan to share the output_ringbuf over cuda IPC (but I'm not sure if it's
    // really necessary.)
    Array<void>   int4_data_gpu      = params.gpu_allocator->allocate_array<void>  (dt_int4, {S, B, F, T});
    Array<__half> scales_offsets_gpu = params.gpu_allocator->allocate_array<__half>({S, B, F, T / 256, 2});

    GpuDequantizationKernel dequantization_kernel(dtype, B, F, T);

    // CudaEventRingbuf capacities. As in GpuDedisperser, each ring's max_size is
    // the worst-case host-side span = max lag between the producer's record() and
    // the slowest consumer's wait()/synchronize() (see notes/cuda_event_ringbuf.md).
    // Threads here: P = processing_thread (this thread; producer + same-thread
    // consumer), F = frame_finalizing_thread. A ring with a cross-thread consumer
    // records with blocking=true (an under-sized ring then throttles instead of
    // throwing) and adds +S of headroom so the producer doesn't block steady-state.
    //
    //   dequant: producer P; consumer dq-wait (P, lag S). span = S.
    //   h2g: producer P; consumers h2g-wait (P, lag 0), frame_finalizing
    //        (F, lag ~S = input pipeline depth). span ~ S; +S headroom -> 2S.
    auto evrb_dq_p = std::make_shared<CudaEventRingbuf>(
        "dequant", /*nconsumers=*/1, /*max_size=*/S, /*blocking_sync=*/true);
    auto evrb_h2g_p = std::make_shared<CudaEventRingbuf>(
        "h2g", /*nconsumers=*/2, /*max_size=*/2*S, /*blocking_sync=*/true);

    // Publish plan + dedisperser + evrb_* atomically under the mutex, and set
    // dedisperser_is_initialized (which wakes the frame_finalizing_thread).
    {
        lock_guard<std::mutex> lock(mutex);
        plan         = plan_p;
        dedisperser  = dedisperser_p;
        evrb_dq      = evrb_dq_p;
        evrb_h2g     = evrb_h2g_p;
        dedisperser_is_initialized = true;
        cv.notify_all();
    }

    // Snapshot rb_processed into rb_curr (local). Indexes the processing
    // thread's current position in the ring buffer, in lockstep with the loop
    // indices (ichunk, ibatch, b) below.
    //
    // At this point the frame_finalizing_thread (the only other writer of
    // rb_processed) is necessarily blocked in
    // evrb_h2g->synchronize(0, true) -- we haven't called record()
    // yet -- so rb_processed is still at its seed value and the snapshot is
    // well-defined.
    long rb_curr;
    {
        unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [&]{ return is_stopped || rb_initialized; });
        if (is_stopped) return;
        rb_curr = rb_processed;
    }

    // NOTE on index conventions:
    //
    // FrbServer's rb_* counters use ABSOLUTE frame IDs of the form
    // (time_chunk_index * nbeams + ibeam), where time_chunk_index is measured
    // from fpga_seq=0 (in general nonzero). GpuDedisperser uses a ZERO-BASED
    // batch index seq_id = 0, 1, 2, ...
    //
    // THIS PROCESSING CODE USES ZERO-BASED seq_id THROUGHOUT. seq_id and the
    // per-beam index b are used only for the dedispersion-side arithmetic
    // (acquire_input, evrb_*, dequant kernel). Ring-buffer access uses
    // rb_curr % rb_size (an absolute index); the only bridge between the two
    // worlds is the snapshot above.
    const long rb_size = long(frame_ringbuf.size());

    // Index-convention helpers for the per-frame consistency check below.
    //   nbeams   = beams per time chunk (= m->get_nbeams() = beams_per_gpu).
    //   nbatches = batches per time chunk (B = beams_per_batch beams each).
    //   initial_time_chunk = FPGA-based chunk index of seq_id=0 (the same
    //     canonical value the workers seed rb_* with, and that we passed to
    //     GpuDedisperser::Params::initial_chunk). wait_for_initial_chunk() is
    //     idempotent and already resolved here, so this does not block.
    const long nbeams             = m->get_nbeams();
    const long nbatches           = nbeams / B;
    const long initial_time_chunk = frame_allocator->wait_for_initial_chunk();

    for (long seq_id = 0; ; seq_id++) {
        const long istream = seq_id % S;
        cudaStream_t compute_stream = dedisperser_p->stream_pool->compute_streams.at(istream);
        cudaStream_t h2g_stream     = dedisperser_p->stream_pool->high_priority_h2g_stream;

        // --no-dedispersion threads through this loop as a set of skips rather
        // than a separate path: we run the SAME frame-consuming inner loop (so
        // rb_curr advances in lockstep with rb_assembled, and frame_finalizing_thread
        // can keep advancing rb_processed), but skip every GPU step -- the evrb_dq
        // wait, the per-beam host->device copies, and the dequant + dedispersion
        // launches. The dedisperser is still fully built and allocated above; under
        // --no-dedispersion it is simply never fed (and --no-dedispersion implies
        // --no-grouper, so there is no output consumer to starve). The key reason
        // we keep the inner loop is that evrb_h2g must be recorded only AFTER all B
        // frames are assembled -- never ahead of assembly -- or frame_finalizing_thread
        // would advance rb_processed past rb_assembled and trip its invariant (and
        // stall the reaper / trip the worker's slot-reuse assert).

        // Before queueing the h2g copies, wait on its output buffer, by waiting on
        // the dequant kernel with (seq_id - nstreams). Skipped under
        // --no-dedispersion: evrb_dq is never recorded there, so this wait
        // (blocking=false) would throw once seq_id >= nstreams.
        if (!params.no_dedispersion)
            evrb_dq_p->wait(h2g_stream, seq_id - S);

        // Consume B assembled frames and queue their h2g copies (under
        // --no-dedispersion, consume only -- no copies are queued).
        for (long b = 0; b < B; b++) {
            std::shared_ptr<AssembledFrame> frame;
            {
                unique_lock<std::mutex> lock(mutex);
                cv.wait(lock, [&]{
                    return is_stopped || (rb_curr < rb_assembled);
                });
                if (is_stopped)
                    return;
                xassert(rb_processed <= rb_curr);    // defensive
                frame = frame_ringbuf[rb_curr % rb_size];
                rb_curr++;
            }

            // Optional artificial per-frame delay, off-lock.
            if (params.processing_delay_sec > 0.0) {
                std::this_thread::sleep_for(
                    std::chrono::duration<double>(params.processing_delay_sec));
            }

            // Defensive: the h2g copies use raw pointers + byte counts (no
            // shape validation), so assert the frame matches the dedisperser
            // geometry (nfreq, nt_in). A mismatch would silently corrupt GPU
            // memory. (Also runs under --no-dedispersion, as a harmless
            // frame-geometry sanity check.)
            xassert_eq(frame->nfreq, F);
            xassert_eq(frame->ntime, T);

            // Defensive: confirm the frame we just pulled is exactly the one
            // this (seq_id, b) is supposed to feed into dedispersion. The
            // dedisperser stamps each output with an FPGA-based chunk index and
            // a beam index derived SOLELY from seq_id:
            //     ichunk_fpga_based = initial_time_chunk + seq_id / nbatches
            //     ibeam             = (seq_id % nbatches) * B + b
            // (see GpuDedisperser::acquire_output / Outputs). This asserts the
            // INPUT frame carries the matching time_chunk_index / beam_id, so
            // any drift between the absolute frame ring (rb_curr) and the
            // zero-based seq_id world -- which would silently mislabel every
            // output -- is caught here rather than downstream. Mirrors the
            // worker-side insertion check (frame_id = ichunk*nbeams + ibeam).
            const long expected_chunk   = initial_time_chunk + seq_id / nbatches;
            const long expected_beam_id = m->beam_ids.at((seq_id % nbatches) * B + b);
            xassert_eq(frame->time_chunk_index, expected_chunk);
            xassert_eq(frame->beam_id, expected_beam_id);

            // --no-dedispersion: the frame has been consumed (so rb_processed can
            // advance); skip the actual host->device copy and move to the next beam.
            if (params.no_dedispersion)
                continue;

            // Per-beam destinations and byte counts.
            Array<void>   raw_dst   = int4_data_gpu     .slice(0, istream).slice(0, b);
            Array<__half> scoff_dst = scales_offsets_gpu.slice(0, istream).slice(0, b);
            const long raw_nbytes   = F * (T / 2);
            const long scoff_nbytes = F * (T / 256) * 2 * long(sizeof(__half));

            safe_memcpy_h2g_async(raw_dst.data,   frame->data.data,           raw_nbytes,   h2g_stream);
            safe_memcpy_h2g_async(scoff_dst.data, frame->scales_offsets.data, scoff_nbytes, h2g_stream);

            // Note: we do NOT bump rb_processed here -- the
            // frame_finalizing_thread does that, in batches of B, after
            // evrb_h2g fires for this seq_id.
        }

        // After queueing the h2g copies (none under --no-dedispersion, but the
        // inner loop waited for all B frames to be assembled either way), produce
        // the h2g event. frame_finalizing_thread consumes it to advance
        // rb_processed in both modes.
        // blocking=true: the second consumer (frame_finalizing_thread) is cross-thread.
        evrb_h2g_p->record(h2g_stream, seq_id, /*blocking=*/true);

        // Consume the h2g event on compute_stream. In the normal path this gates
        // the dequantization kernel on its source buffer; under --no-dedispersion
        // no kernel follows, but the wait is still required as evrb_h2g's SECOND
        // consumer (nconsumers=2: this wait + frame_finalizing_thread) so its
        // ring-buffer slots recycle.
        evrb_h2g_p->wait(compute_stream, seq_id);

        // --no-dedispersion: dedisperser built but never fed -- skip all remaining
        // GPU compute (dequant + dedispersion launches, and the evrb_dq event that
        // gates h2g-scratch reuse, which nothing consumes under --no-dedispersion).
        if (params.no_dedispersion)
            continue;

        // Wait on the dequant kernel's destination buffer via GpuDedisperser::acquire_input().
        Array<void> dd_inbuf = dedisperser_p->acquire_input(seq_id, compute_stream);

        // Launch the dequantization kernel.
        Array<void>   raw_batch   = int4_data_gpu     .slice(0, istream);
        Array<__half> scoff_batch = scales_offsets_gpu.slice(0, istream);
        dequantization_kernel.launch(dd_inbuf, scoff_batch, raw_batch, compute_stream);

        // After launching the dequantization kernel, we produce a dq event.
        // blocking=false: the only consumer (dq-wait above) is on this thread (P).
        evrb_dq_p->record(compute_stream, seq_id, /*blocking=*/false);

        // Launch the rest of the dedispersion kernels.
        dedisperser_p->release_input_and_launch_dd_kernels(seq_id, compute_stream);
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


// -------------------------------------------------------------------------------------------------
//
// Frame-finalizing thread
//
// Bridges H2G-copy completion to the FrbServer ringbuf accounting. Blocks
// until the processing_thread publishes evrb_h2g, then for each
// seq_id = 0, 1, 2, ... blocks (on the host) in
// evrb_h2g->synchronize() until the GPU has completed batch
// seq_id's H2G copies, then advances rb_processed by beams_per_batch.


void FrbServer::_frame_finalizing_thread_main()
{
    // Wait for processing_thread to finish post-allocate setup and publish evrb_h2g.
    shared_ptr<CudaEventRingbuf> evrb_h2g_p;
    {
        unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [&]{ return is_stopped || dedisperser_is_initialized; });
        if (is_stopped) return;
        evrb_h2g_p = evrb_h2g;
        xassert(evrb_h2g_p);
    }

    // beams_per_batch is fixed at construction; cache it. (dedisperser is
    // non-null and immutable once dedisperser_is_initialized was observed true
    // above, so reading it here outside the lock is safe.)
    const long B = dedisperser->beams_per_batch;

    // Per-chunk status-line geometry. Metadata is available by now (the plan was
    // built from it before dedisperser_is_initialized, which we waited on above), so
    // get_metadata(true) does not block. A chunk is fully PROCESSED when rb_processed
    // reaches a chunk boundary (multiple of nbeams); rb_processed advances by B, a
    // divisor of nbeams, in both the normal and --no-dedispersion paths, so it lands
    // exactly on each boundary. bytes_per_chunk / chunk_len_sec feed Gbps + rt_beams.
    shared_ptr<const XEngineMetadata> m = frame_allocator->get_metadata(true);
    const long   nbeams          = m->get_nbeams();
    const long   ntime           = frame_allocator->time_samples_per_chunk;
    const long   seq_per_chunk   = ntime * m->seq_per_frb_time_sample;
    const long   bytes_per_chunk = nbeams * AssembledFrameAllocator::slab_nbytes(m->get_total_nfreq(), ntime);
    const double chunk_len_sec   = double(seq_per_chunk) * double(m->dt_ns_per_seq) * 1.0e-9;

    // Average throughput since the first PRINTED chunk. Only this single thread
    // touches these, so no locking is needed. t0_valid latches on the first printed
    // chunk (which reports no rate -- there is no interval yet).
    bool t0_valid = false;
    std::chrono::steady_clock::time_point t0;
    long ichunk0 = 0;

    // Scratch for the stream-capture hook (reused across batches).
    std::vector<std::pair<std::shared_ptr<AssembledFrame>,
                          std::shared_ptr<FileStream>>> stream_matches;

    for (long seq_id = 0; ; seq_id++) {
        // Blocks (on the host) until the GPU completes batch seq_id's H2G
        // copies. Throws "called on stopped instance" if stop() cascaded into
        // evrb_h2g; the wrapper below catches it.
        evrb_h2g_p->synchronize(seq_id, /*blocking=*/true);

        // The batch's frames [rb_processed, rb_processed + B) are about to
        // transition from "assembled" to "processed". Stream capture happens
        // in three phases:
        //
        //   A. Under 'mutex': deactivate expired streams (move to the
        //      inactive ring), record matching (frame, stream) pairs, bump
        //      the queued-counters. rb_processed is NOT advanced yet.
        //   B. Off-lock: expand patterns and queue the writes
        //      (_queue_frame_write; disk I/O happens on FileWriter threads,
        //      process_frame only enqueues).
        //   C. Under 'mutex': advance rb_processed.
        //
        // Queueing BEFORE advancing rb_processed (B before C) is load-bearing,
        // not cosmetic: the reaper only reaps frames with frame_id <
        // rb_processed, and AssembledFrame::_reap_locked() refuses to free
        // data while an unwritten save_path is pending. So pushing the
        // save_path first guarantees the reaper can never free the data
        // underneath a pending stream write. (Data is stable here: the
        // batch's H2G copy has completed, in both the normal and
        // --no-dedispersion paths.) A corollary: _queue_frame_write's
        // reaped-and-never-written skip can never fire on this path, so
        // every counted match really is queued.
        stream_matches.clear();

        // Phase A.
        {
            unique_lock<std::mutex> lock(mutex);
            if (is_stopped) return;

            // Each fired event was preceded by B inner-loop iterations in the
            // processing_thread, each gated on rb_curr < rb_assembled, so this
            // should always hold. If it doesn't, it's a logic bug.
            if (rb_assembled < rb_processed + B) {
                std::stringstream ss;
                ss << "FrbServer::frame_finalizing_thread observed rb_assembled="
                   << rb_assembled << " < rb_processed + beams_per_batch="
                   << (rb_processed + B)
                   << "; this should never happen and indicates a logic bug";
                throw std::runtime_error(ss.str());
            }

            if (!active_streams.empty()) {
                // All B frames lie in this chunk: rb_processed is a multiple
                // of B, and B divides nbeams.
                long ichunk = rb_processed / nbeams;

                // Deactivate expired streams (they can never match chunk >=
                // ichunk); they remain visible in the inactive ring.
                for (const auto &st : active_streams) {
                    if (st->chunk_last < ichunk)
                        _deactivate_stream(st, /*cancelled=*/ false);
                }
                active_streams.erase(
                    std::remove_if(active_streams.begin(), active_streams.end(),
                                   [&](const std::shared_ptr<FileStream> &st)
                                   { return st->chunk_last < ichunk; }),
                    active_streams.end());

                for (long frame_id = rb_processed; frame_id < rb_processed + B; frame_id++) {
                    int ibeam = int(frame_id % nbeams);
                    for (const auto &st : active_streams) {
                        if ((ichunk < st->chunk_first) || (ichunk > st->chunk_last))
                            continue;
                        auto it = std::find(st->beam_indices.begin(), st->beam_indices.end(), ibeam);
                        if (it == st->beam_indices.end())
                            continue;

                        // Rule R1 (see FileStream's thread-safety comment):
                        // counted at MATCH time, before the Phase-B push, so
                        // written + errored <= queued at every instant, and
                        // equality on a deactivated stream means "fully
                        // drained".
                        st->num_files_queued++;
                        long rb_slot = frame_id % long(frame_ringbuf.size());
                        stream_matches.emplace_back(frame_ringbuf[rb_slot], st);
                    }
                }
            }
        }

        // Phase B.
        for (const auto &[frame, st] : stream_matches) {
            string relpath = st->pattern.expand(frame);
            _queue_frame_write(frame, relpath, st);
        }

        // Phase C.
        long completed_ichunk = -1;   // >= 0 once rb_processed crosses a chunk boundary
        {
            unique_lock<std::mutex> lock(mutex);
            if (is_stopped) return;

            rb_processed += B;
            _check_rb_invariants();

            // rb_processed just reached (ichunk+1)*nbeams => chunk 'ichunk' is done.
            if (rb_processed % nbeams == 0)
                completed_ichunk = rb_processed / nbeams - 1;
        }
        cv.notify_all();

        // Announce a fully-processed chunk (once per chunk), unless params.quiet.
        // Printed off-lock. See the assembly-time note in _worker_main: the status
        // line lives here (processed by the GPU), not there (assembled by the net).
        if ((completed_ichunk >= 0) && !params.quiet) {
            long ichunk = completed_ichunk;
            auto now = std::chrono::steady_clock::now();
            double chunks_per_sec = -1.0;   // < 0 => not available yet (first printed chunk)
            
            if (!t0_valid) {
                t0_valid = true;
                t0 = now;
                ichunk0 = ichunk;
            } else {
                double elapsed = std::chrono::duration<double>(now - t0).count();
                long dchunks = ichunk - ichunk0;
                if ((elapsed > 0.0) && (dchunks > 0))
                    chunks_per_sec = double(dchunks) / elapsed;
            }

            std::ostringstream line;
            line << "FrbServer: beamset=" << m->beamset
                 << ", ichunk=" << ichunk
                 << ", fpga=[" << (ichunk * seq_per_chunk)
                 << ":" << ((ichunk + 1) * seq_per_chunk) << "]";
            
            if (chunks_per_sec >= 0.0) {
                double gbps = chunks_per_sec * double(bytes_per_chunk) * 8.0 / 1.0e9;
                double rt_beams = chunks_per_sec * chunk_len_sec * double(nbeams);
                line << std::fixed << std::setprecision(2)
                     << ", Gbps=" << gbps
                     << ", beams=" << rt_beams
                     << "/" << nbeams << "\n";
            }
            std::cout << line.str() << std::flush;
        }
    }
}


void FrbServer::frame_finalizing_thread_main()
{
    try {
        // Same cudaSetDevice as processing_thread_main: this thread calls
        // CudaEventRingbuf::synchronize(), which dispatches to CUDA runtime
        // calls bound to the current device.
        CUDA_CALL(cudaSetDevice(params.cuda_device_id));
        _frame_finalizing_thread_main();
    } catch (const std::exception &e) {
        if (!_is_cascade_stop_exception(e)) {
            std::cerr << "FrbServer: frame_finalizing thread terminated with exception: "
                      << e.what() << std::endl;
        }
        stop(std::current_exception());
    } catch (...) {
        std::cerr << "FrbServer: frame_finalizing thread terminated with unknown exception" << std::endl;
        stop(std::current_exception());
    }
}


// Queue one file write on 'frame'. Shared by the WriteFiles RPC handler
// (stream = nullptr) and the frame_finalizing_thread's stream-capture hook.
// See declaration in FrbServer.hpp for the contract.
bool FrbServer::_queue_frame_write(const shared_ptr<AssembledFrame> &frame,
                                   const string &relpath,
                                   const shared_ptr<FileStream> &stream)
{
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
    //
    // NOTE: on the stream path (stream != nullptr) this skip can never fire
    // (frames at the assembled->processed transition are not yet reapable),
    // which is what keeps FileStream::num_files_queued -- already bumped at
    // match time, rule R1 -- consistent with the pushes.
    if (frame->data.size == 0 && frame->save_paths.empty())
        return false;

    // Duplicate paths are legal here (e.g. two streams with the same
    // filename_pattern, or a repeated WriteFiles): FileWriter's NFS thread
    // skips the filesystem operation for a duplicate of an earlier entry,
    // but still emits its WriteStatus.
    frame->save_paths.push_back({relpath, stream});
    frame_lock.unlock();

    params.file_writer->process_frame(frame);
    return true;
}


// (file-local) Wall-clock unix time in ns, for FileStream timestamps.
static long _unix_time_ns()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}


// Deactivate one stream. Caller must hold 'mutex' and removes 'st' from
// active_streams itself. See declaration in FrbServer.hpp for the contract.
void FrbServer::_deactivate_stream(const shared_ptr<FileStream> &st, bool cancelled)
{
    st->cancelled = cancelled;
    st->deactivated_at_unix_ns = _unix_time_ns();
    inactive_streams[num_deactivated_streams % constants::inactive_file_stream_capacity] = st;
    num_deactivated_streams++;
}


// -------------------------------------------------------------------------------------------------
//
// FrbGrouper client: send thread + receive thread + handshake builder.
//
// Only active when params.grouper_ip_addr is non-empty. The send thread is the
// single GpuDedisperser output consumer (consumer_id 0): it connects to the
// grouper, starts the receivers, builds + sends the Handshake, then streams
// produced_seq_ids (one per cdd2-produced batch). The receive thread reads
// consumed_seq_ids and releases each batch back to the dedisperser. See
// plans/grouper_client.md for the full design / thread-ordering rationale.


void FrbServer::_fill_handshake(fg::Handshake *hs,
                                const shared_ptr<GpuDedisperser> &dd,
                                const shared_ptr<DedispersionPlan> &plan_snap)
{
    hs->set_protocol_version(fg::PROTOCOL_VERSION_CURRENT);   // proto enum (shared w/ consumer)

    // IPC handle for the base GPU allocation. The base + size come from the
    // gpu_allocator (not a GpuDedisperser accessor); the FrbServer constructor
    // already guaranteed it is non-empty/non-dummy when a grouper is configured.
    //
    // Note: cudaIPC requires the base pointer to be allocated with cudaMalloc
    // (not cudaMallocAsync). The BumpAllocator code is writen so that the initial
    // allocation is always synchronous, even if the BumpAllocator is constructed
    // with async=true. (The 'async' arg controls whether zeroing is done in a
    // separate thread, but does not affect the initial allocation.)
    std::shared_ptr<void> base_sp = params.gpu_allocator->get_base();
    void *base = base_sp.get();
    xassert(base != nullptr);   // guaranteed by the constructor's capacity > 0 check
    
    // Check that the output ringbuf is the first thing allocated by the gpu_allocator.
    // I don't know if this is really necessary, but it's true in our current implementation.
    // This assert is just intended to flag if this changes in the future, so that we can
    // revisit (if this assert ever fails, the most likely scenario is that it's unintentional
    // and can made to pass by reordering a few lines of code).
    void *ringbuf_base = dd->output_ringbuf.out_max.at(0).data;
    xassert(base == ringbuf_base);
    
    cudaIpcMemHandle_t handle;
    CUDA_CALL(cudaIpcGetMemHandle(&handle, base));
    static_assert(sizeof(cudaIpcMemHandle_t) == 64, "");
    hs->set_ipc_mem_handle(reinterpret_cast<const char*>(&handle), sizeof(handle));
    hs->set_cuda_device_id(params.cuda_device_id);
    hs->set_base_nbytes(params.gpu_allocator->capacity);

    // Array descriptors: out_max + out_argmax for each tree. Layout-agnostic:
    // each array located by byte_offset (relative to base) + shape + strides.
    const char *cbase = static_cast<const char*>(base);
    const auto &orb = dd->output_ringbuf;
    auto add_array = [&](const std::string &name, long itree,
                         const ksgpu::Array<void> &arr) {
        fg::ArrayDescriptor *ad = hs->add_arrays();
        ad->set_name(name);
        ad->set_tree_index(itree);
        ad->set_dtype(arr.dtype.str());   // canonical ksgpu dtype string
        ad->set_byte_offset(static_cast<const char*>(arr.data) - cbase);
        for (int i = 0; i < arr.ndim; i++) ad->add_shape(arr.shape[i]);
        for (int i = 0; i < arr.ndim; i++) ad->add_strides(arr.strides[i]);
    };
    for (long t = 0; t < dd->ntrees; t++) {
        add_array("out_max",    t, orb.out_max[t]);
        add_array("out_argmax", t, orb.out_argmax[t]);   // Array<uint> -> const Array<void>&
    }

    // Geometry. (num_batch_slots == producer nbatches_out; the output_ringbuf
    // leading axis is num_batch_slots * beams_per_batch.)
    hs->set_num_trees(dd->ntrees);
    hs->set_num_batch_slots(dd->params.nbatches_out);
    hs->set_beams_per_batch(dd->beams_per_batch);
    hs->set_initial_chunk(dd->params.initial_chunk);   // -> Outputs::ichunk_fpga_based

    // The producer's own RPC endpoint, so the consumer can reach back to us.
    hs->set_rpc_ip_addr(params.rpc_server_address);

    // Run context (YAML). metadata is available by now (the dedisperser needed
    // it to initialize). NOTE: frame_allocator->get_metadata() returns the
    // FrbServer's canonical copy, which is FREQUENCY-SCRUBBED (empty
    // freq_channels); we send it as-is. The consumer gets frequency info from
    // dedispersion_config_yaml's zone structure, not from this. (See the
    // xengine_metadata_yaml comment in frb_grouper.proto.)
    shared_ptr<const XEngineMetadata> m = frame_allocator->get_metadata(/*blocking=*/true);
    hs->set_xengine_metadata_yaml(m->to_yaml_string());
    hs->set_dedispersion_config_yaml(plan_snap->config.to_yaml_string());
    hs->set_dedispersion_plan_yaml(plan_snap->to_yaml_string());
}


void FrbServer::_grouper_send_thread_main()
{
    CUDA_CALL(cudaSetDevice(params.cuda_device_id));   // cudaIpcGetMemHandle, acquire_output(sync)

    // (1) Build the gRPC client WITHOUT the lock (CreateCustomChannel/NewStub
    // allocate gRPC internals + may start name resolution -- too heavy to hold
    // the hot FrbServer::mutex across), then publish it under the lock.
    //
    // Cap the channel's connection-reconnect backoff at 1s. By default gRPC
    // backs off exponentially between TCP connect attempts (1s, 1.6s, ... up to
    // 120s), so if the grouper isn't up yet the producer can be slow to notice
    // it appear -- the wait loop below polls once/sec, but GetState(
    // try_to_connect=true) does NOT shorten an in-progress backoff. With
    // initial == max == 1s, the channel retries ~once per second.
    grpc::ChannelArguments chan_args;
    chan_args.SetInt(GRPC_ARG_INITIAL_RECONNECT_BACKOFF_MS, 1000);
    chan_args.SetInt(GRPC_ARG_MAX_RECONNECT_BACKOFF_MS, 1000);

    auto gc = make_unique<GrouperClient>();
    gc->channel = grpc::CreateCustomChannel(params.grouper_ip_addr,
                                            grpc::InsecureChannelCredentials(),
                                            chan_args);
    gc->stub    = fg::FrbGrouper::NewStub(gc->channel);
    gc->context = make_unique<grpc::ClientContext>();
    {
        unique_lock<std::mutex> lock(mutex);
        if (is_stopped) return;            // stop() ran during construction; drop gc, return
        grouper_client = std::move(gc);    // publish (O(1) move)
    }

    // (2) Wait for the channel to become READY, printing once/sec.
    //     (try_to_connect=true kicks the channel out of IDLE.)
    auto channel = grouper_client->channel;
    while (channel->GetState(/*try_to_connect=*/true) != GRPC_CHANNEL_READY) {
        { unique_lock<std::mutex> lock(mutex); if (is_stopped) return; }
        std::cout << "FrbServer: waiting for grouper at "
                  << params.grouper_ip_addr << " ..." << std::endl;
        // Block up to 1s for a state change (bounds is_stopped re-check latency).
        auto deadline = std::chrono::system_clock::now() + std::chrono::seconds(1);
        channel->WaitForStateChange(channel->GetState(false), deadline);
    }

    std::cout << "FrbServer: connected to grouper at " << params.grouper_ip_addr << std::endl;
    std::cout << "FrbServer: waiting for X-engine node(s) to connect at "
              << params.rpc_server_address << " (Ctrl-C to stop)" << std::endl;

    // (3) Open the Session stream. Recheck is_stopped first (see
    //     plans/grouper_client.md 4g): avoids starting an RPC we're about to
    //     abandon if stop() landed during the connect loop.
    { unique_lock<std::mutex> lock(mutex); if (is_stopped) return; }
    grouper_client->stream = grouper_client->stub->Session(grouper_client->context.get());

    // (4) Start receivers NOW (see "Thread ordering"). Data ingest begins ->
    //     metadata arrives -> processing_thread builds the dedisperser.
    { unique_lock<std::mutex> lock(mutex); if (is_stopped) return; }
    for (auto &r : params.receivers)
        r->start();

    // (5) Block until the dedisperser is initialized + allocated.
    shared_ptr<GpuDedisperser> dd;
    shared_ptr<DedispersionPlan> plan_snap;
    {
        unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [&]{ return is_stopped || dedisperser_is_initialized; });
        if (is_stopped) return;
        dd = dedisperser;
        plan_snap = plan;
    }

    // (6) Build + send the Handshake.
    fg::ProducerMessage msg;
    fg::Handshake *hs = msg.mutable_handshake();
    _fill_handshake(hs, dd, plan_snap);
    if (!grouper_client->stream->Write(msg))
        throw runtime_error("FrbServer: grouper Write(Handshake) failed (stream closed)");

    // (7) Read the single HandshakeReply (the only Read done by this thread;
    //     receive_thread does all subsequent Reads).
    fg::ConsumerMessage reply;
    if (!grouper_client->stream->Read(&reply))
        throw runtime_error("FrbServer: grouper closed stream before HandshakeReply");
    if (reply.body_case() != fg::ConsumerMessage::kHandshakeReply)
        throw runtime_error("FrbServer: expected HandshakeReply as first grouper message");
    if (!reply.handshake_reply().ready())
        throw runtime_error("FrbServer: grouper rejected handshake: "
                            + reply.handshake_reply().error_message());

    // (8) Wake receive_thread.
    { lock_guard<std::mutex> lock(mutex); grouper_handshake_done = true; cv.notify_all(); }

    // (9) Steady-state produced loop.
    for (long seq_id = 0; ; seq_id++) {
        // Blocks the host until cdd2 has produced seq_id (or throws on stop).
        // sync=true -> 'stream' arg ignored; noreturn=true -> no Outputs built.
        dd->acquire_output(/*consumer_id=*/0, seq_id, /*stream=*/nullptr,
                           /*sync=*/true, /*noreturn=*/true);

        fg::ProducerMessage pm;
        pm.set_produced_seq_id(seq_id);
        if (!grouper_client->stream->Write(pm))
            throw runtime_error("FrbServer: grouper Write(produced_seq_id) failed");
    }
}


void FrbServer::grouper_send_thread_main()
{
    try {
        _grouper_send_thread_main();
    } catch (const std::exception &e) {
        if (!_is_cascade_stop_exception(e)) {
            std::cerr << "FrbServer: grouper send thread terminated with exception: "
                      << e.what() << std::endl;
        }
        stop(std::current_exception());
    } catch (...) {
        std::cerr << "FrbServer: grouper send thread terminated with unknown exception" << std::endl;
        stop(std::current_exception());
    }
}


void FrbServer::_grouper_receive_thread_main()
{
    CUDA_CALL(cudaSetDevice(params.cuda_device_id));   // release_output -> cudaEventRecord

    // Dedicated stream for release_output()'s cudaEventRecord.
    //
    // HACK (artifact of synchronous, non-event-driven coordination): we ONLY
    // ever cudaEventRecord() on rel_stream and NEVER enqueue any kernel on it.
    // Because the stream is always idle, every event recorded on it is
    // considered "transpired" the instant it is recorded -- so any downstream
    // cudaStreamWaitEvent / cudaEventSynchronize on these events is a GPU-level
    // no-op. That is intentional today: the real consumer<->producer ordering is
    // enforced at the HOST level (the grouper sends CONSUMED only after it has
    // finished reading the batch; we then call release_output(); cdd2's
    // host-blocking wait on evrb_release_output is the actual back-pressure). The
    // recorded event exists only to satisfy the GpuDedisperser API, which expects
    // one recorded event per released seq_id. If we later switch to event-driven
    // (true cross-process GPU) coordination, this hack goes away.
    ksgpu::CudaStreamWrapper rel_stream = ksgpu::CudaStreamWrapper::create();

    // (1) Wait until send_thread has completed the handshake.
    shared_ptr<GpuDedisperser> dd;
    {
        unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [&]{ return is_stopped || grouper_handshake_done; });
        if (is_stopped) return;
        dd = dedisperser;
        xassert(dd);
    }

    // grouper_client->stream was published by send_thread before it set
    // grouper_handshake_done (same mutex), so it is safe to read here. gRPC
    // permits one concurrent Read (here) + one concurrent Write (send thread).
    auto *stream = grouper_client->stream.get();

    // (2) Steady-state consumed loop.
    for (long seq_id = 0; ; seq_id++) {
        fg::ConsumerMessage msg;
        if (!stream->Read(&msg)) {
            // Stream closed. If we initiated it (stop), exit quietly; else throw.
            { unique_lock<std::mutex> lock(mutex); if (is_stopped) return; }
            throw runtime_error("FrbServer: grouper Session stream closed unexpectedly");
        }
        if (msg.body_case() != fg::ConsumerMessage::kConsumedSeqId)
            throw runtime_error("FrbServer: expected consumed_seq_id from grouper");

        long got = msg.consumed_seq_id();
        xassert_eq(got, seq_id);   // consumer must ack in order 0,1,2,...

        // Host-ordered release (records event on rel_stream). cdd2 will not reuse
        // this slot until this returns (see the "Cross-process GPU
        // synchronization" discussion in plans/grouper_client.md).
        dd->release_output(/*consumer_id=*/0, seq_id, rel_stream);
    }
}


void FrbServer::grouper_receive_thread_main()
{
    try {
        _grouper_receive_thread_main();
    } catch (const std::exception &e) {
        if (!_is_cascade_stop_exception(e)) {
            std::cerr << "FrbServer: grouper receive thread terminated with exception: "
                      << e.what() << std::endl;
        }
        stop(std::current_exception());
    } catch (...) {
        std::cerr << "FrbServer: grouper receive thread terminated with unknown exception" << std::endl;
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

// Every RPC request carries a protocol_version field (see notes/grpc.md); each
// wrapper below calls this first. Throws (mapped to a gRPC error by the wrapper)
// if the client's version does not match the server's, so client/server
// wire-format skew (out-of-sync pirate builds) surfaces as a clear error rather
// than silent misbehavior. Compared as integers -- protocol_version is a uint32
// on the wire (see the proto comment) so an out-of-range value from a newer
// client round-trips rather than decoding to the proto3 "unknown enum" sentinel.
static void _check_protocol_version(uint32_t client_version, const char *rpc_name)
{
    if (long(client_version) != long(fs::PROTOCOL_VERSION_CURRENT)) {
        stringstream ss;
        ss << rpc_name << ": protocol version mismatch (client sent protocol_version="
           << client_version << ", but this server requires "
           << int(fs::PROTOCOL_VERSION_CURRENT)
           << "; client and server pirate builds are out of sync)";
        throw runtime_error(ss.str());
    }
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
        _check_protocol_version(request->protocol_version(), "GetStatus");
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
        _check_protocol_version(request->protocol_version(), "GetXEngineMetadata");
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

    // RPC callers express the time range as fpga sequence numbers (aka fpga
    // counts), half-open: files are written for all chunks overlapping
    // [fpga_seq_start, fpga_seq_end). The FrbServer works internally in
    // absolute time-chunk indices, so we translate to a chunk-index range
    // here, up front, and use chunk indices for everything below.
    long fpga_seq_start = request->fpga_seq_start();
    long fpga_seq_end   = request->fpga_seq_end();

    if ((fpga_seq_start < 0) || (fpga_seq_end <= fpga_seq_start)) {
        stringstream ss;
        ss << "WriteFiles: invalid fpga_seq range [" << fpga_seq_start << ", "
            << fpga_seq_end << ") (require 0 <= fpga_seq_start < fpga_seq_end)";
        throw runtime_error(ss.str());
    }

    // Construct FilenamePattern (validates that pattern contains "(BEAM)" and "(CHUNK)").
    FilenamePattern filename_pattern(request->filename_pattern());

    vector<int> beam_indices;
    beam_indices.reserve(request->beams_size());

    vector<shared_ptr<AssembledFrame>> local_frames;

    {
        unique_lock<std::mutex> server_lock(s->mutex);

        // The fpga-seq <-> chunk mapping, the ring buffer, and beam_id_to_index
        // only exist once the server has locked onto the stream: X-engine
        // metadata parsed AND the initial fpga chunk established (both are
        // prerequisites for rb_initialized). Fail if the server isn't there yet.
        if (!s->rb_initialized)
            throw runtime_error("WriteFiles: server has not yet established an initial fpga chunk");

        // s->metadata is non-null once rb_initialized (published together).
        long rb_nbeams = s->metadata->get_nbeams();

        // Chunk t spans fpga seqs [t*seq_per_chunk, (t+1)*seq_per_chunk),
        // measured from fpga seq 0 (time_chunk_index is absolute), so fpga seq f
        // lives in chunk floor(f / seq_per_chunk). Map the half-open fpga range
        // to the inclusive range of chunks it touches.
        long seq_per_chunk = s->frame_allocator->time_samples_per_chunk * s->metadata->seq_per_frb_time_sample;
        xassert(seq_per_chunk > 0);
        long min_time_chunk_index = fpga_seq_start / seq_per_chunk;
        long max_time_chunk_index = (fpga_seq_end - 1) / seq_per_chunk;

        // Convert beam_ids to beam_indices.
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

        // Get frames from the ring buffer. We compute frame_id =
        // time_chunk_index * nbeams + beam_index directly for each
        // (time_chunk_index, beam_index) pair, rather than iterating over the
        // entire ring buffer and checking each frame's metadata. This is
        // O(num_time_chunks * num_beams) instead of O(ringbuf_size).
        //
        // Use rb_processed (not rb_assembled) as the upper bound: frames in
        // [rb_processed, rb_assembled) are fully assembled but the GPU may
        // still be mutating them, so they are NOT rpc-writeable.
        long rb_size      = s->frame_ringbuf.size();
        long rb_start     = s->rb_start;
        long rb_processed = s->rb_processed;

        min_time_chunk_index = max(min_time_chunk_index, rb_start / rb_nbeams);
        max_time_chunk_index = min(max_time_chunk_index, rb_processed / rb_nbeams);

        long num_time_chunks = max(0L, max_time_chunk_index - min_time_chunk_index + 1);
        local_frames.reserve(num_time_chunks * beam_indices.size());

        for (long t = min_time_chunk_index; t <= max_time_chunk_index; t++) {
            for (int b : beam_indices) {
                long frame_id = t * rb_nbeams + b;
                if ((frame_id >= rb_start) && (frame_id < rb_processed)) {
                    long rb_slot = frame_id % rb_size;
                    local_frames.push_back(s->frame_ringbuf[rb_slot]);
                }
            }
        }
    }

    // Process frames in reverse order: push filenames onto frame->save_paths,
    // then call file_writer->process_frame() (via _queue_frame_write, shared
    // with the frame_finalizing_thread's stream-capture hook). Reverse order
    // ensures that frames with lower time_chunk_index are processed last
    // (and appear earlier in queues).

    vector<string> filename_list;
    filename_list.reserve(local_frames.size());

    for (auto it = local_frames.rbegin(); it != local_frames.rend(); ++it) {
        auto &frame = *it;
        string filename = filename_pattern.expand(frame);

        // stream = nullptr: WriteFiles-triggered (as opposed to a stream).
        // Returns false if the frame was reaped without ever being written.
        if (s->_queue_frame_write(frame, filename, nullptr))
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
        _check_protocol_version(request->protocol_version(), "WriteFiles");
        _WriteFiles(request, response);
        return grpc::Status::OK;
    } catch (const std::exception &e) {
        return grpc::Status(grpc::StatusCode::INTERNAL, e.what());
    } catch (...) {
        return grpc::Status(grpc::StatusCode::INTERNAL, "Unknown error in WriteFiles");
    }
}

// ---- StartStream ----

void FrbRpcService::_StartStream(const fs::StartStreamRequest *request, fs::StartStreamResponse *response)
{
    shared_ptr<FrbServer> s = _lock_state();

    const string &acq_name = request->acq_name();
    if (acq_name.empty())
        throw runtime_error("StartStream: acq_name must be a nonempty string");

    long fpga_seq_start = request->fpga_seq_start();
    long fpga_seq_end   = request->fpga_seq_end();

    if ((fpga_seq_start < 0) || (fpga_seq_end <= fpga_seq_start)) {
        stringstream ss;
        ss << "StartStream: invalid fpga_seq range [" << fpga_seq_start << ", "
            << fpga_seq_end << ") (require 0 <= fpga_seq_start < fpga_seq_end)";
        throw runtime_error(ss.str());
    }

    if (request->beam_ids_size() == 0)
        throw runtime_error("StartStream: beam_ids must be nonempty"
                            " (there is no all-beams convention; list beams explicitly)");

    // Validates that the pattern contains "(BEAM)" and "(CHUNK)", and is a
    // safe relative path.
    FilenamePattern pattern(request->filename_pattern());

    auto st = make_shared<FileStream>(pattern);
    st->acq_name = acq_name;
    st->pattern_string = request->filename_pattern();
    st->fpga_seq_start = fpga_seq_start;
    st->fpga_seq_end = fpga_seq_end;

    // Wall-clock start time. Stamped before publication (the push into
    // active_streams below, under s->mutex), so it is immutable-after-
    // publication like the other args -- see FileStream's comment.
    st->started_at_unix_ns = _unix_time_ns();

    unique_lock<std::mutex> server_lock(s->mutex);

    // Same gate as WriteFiles: the fpga-seq <-> chunk mapping and
    // beam_id_to_index only exist once rb_initialized.
    if (!s->rb_initialized)
        throw runtime_error("StartStream: server has not yet established an initial fpga chunk");

    for (const auto &other : s->active_streams) {
        if (other->acq_name == acq_name) {
            stringstream ss;
            ss << "StartStream: acq_name '" << acq_name << "' is already in use by an active stream";
            throw runtime_error(ss.str());
        }
    }

    // Same fpga-seq -> chunk conversion as WriteFiles: chunk t matches iff it
    // overlaps [fpga_seq_start, fpga_seq_end). The (fpga_seq_end - 1) form
    // avoids overflow when fpga_seq_end = INT64_MAX ("run indefinitely").
    long nbeams = s->metadata->get_nbeams();
    long seq_per_chunk = s->frame_allocator->time_samples_per_chunk * s->metadata->seq_per_frb_time_sample;
    xassert(seq_per_chunk > 0);
    st->chunk_first = fpga_seq_start / seq_per_chunk;
    st->chunk_last  = (fpga_seq_end - 1) / seq_per_chunk;

    // Convert beam_ids to beam_indices (unknown or repeated beams are errors).
    for (int i = 0; i < request->beam_ids_size(); i++) {
        long beam_id = request->beam_ids(i);
        auto it = s->beam_id_to_index.find(beam_id);
        if (it == s->beam_id_to_index.end()) {
            stringstream ss;
            ss << "StartStream: unknown beam_id " << beam_id;
            throw runtime_error(ss.str());
        }
        if (std::find(st->beam_ids.begin(), st->beam_ids.end(), beam_id) != st->beam_ids.end()) {
            stringstream ss;
            ss << "StartStream: beam_id " << beam_id << " appears more than once";
            throw runtime_error(ss.str());
        }
        st->beam_ids.push_back(beam_id);
        st->beam_indices.push_back(it->second);
    }

    // Streams are not retroactive (chunks already processed are never
    // captured), so a range entirely in the past could never match anything.
    if (st->chunk_last < s->rb_processed / nbeams) {
        stringstream ss;
        ss << "StartStream: fpga_seq_end=" << fpga_seq_end
           << " is entirely in the past (stream would never match); "
           << "use WriteFiles for retroactive dumps within the ring buffer";
        throw runtime_error(ss.str());
    }

    s->active_streams.push_back(st);
}

grpc::Status FrbRpcService::StartStream(
    grpc::ServerContext* context,
    const fs::StartStreamRequest* request,
    fs::StartStreamResponse* response)
{
    try {
        _check_protocol_version(request->protocol_version(), "StartStream");
        _StartStream(request, response);
        return grpc::Status::OK;
    } catch (const std::exception &e) {
        return grpc::Status(grpc::StatusCode::INTERNAL, e.what());
    } catch (...) {
        return grpc::Status(grpc::StatusCode::INTERNAL, "Unknown error in StartStream");
    }
}

// ---- ShowStreams ----

void FrbRpcService::_ShowStreams(const fs::ShowStreamsRequest *request, fs::ShowStreamsResponse *response)
{
    shared_ptr<FrbServer> s = _lock_state();

    unique_lock<std::mutex> server_lock(s->mutex);

    if (!s->rb_initialized)
        throw runtime_error("ShowStreams: server has not yet established an initial fpga chunk");

    long nbeams = s->metadata->get_nbeams();
    long seq_per_chunk = s->frame_allocator->time_samples_per_chunk * s->metadata->seq_per_frb_time_sample;
    long ichunk = s->rb_processed / nbeams;   // first not-fully-processed chunk

    // Deactivate expired streams here too (the capture hook only runs when
    // a batch completes, so without this an expired stream would linger as
    // ACTIVE whenever data stalls).
    for (const auto &st : s->active_streams) {
        if (st->chunk_last < ichunk)
            s->_deactivate_stream(st, /*cancelled=*/ false);
    }
    s->active_streams.erase(
        std::remove_if(s->active_streams.begin(), s->active_streams.end(),
                       [&](const std::shared_ptr<FileStream> &st)
                       { return st->chunk_last < ichunk; }),
        s->active_streams.end());

    // "All data before this fpga seq has been fully processed."
    response->set_current_fpga_seq(ichunk * seq_per_chunk);

    for (long beam_id : s->metadata->beam_ids)
        response->add_beam_ids(beam_id);

    response->set_num_deactivated_streams(s->num_deactivated_streams);

    // Fills one StreamInfo. Counter reads: written/errored are loaded BEFORE
    // queued, so the reported triple always satisfies written + errored <=
    // queued (each written/errored file was queued strictly earlier, rules
    // R1/R2). For a deactivated stream, queued is final, so equality means
    // fully drained (INACTIVE); short of it, writes are in flight (DRAINING).
    auto fill_stream_info = [](fs::StreamInfo *info,
                               const std::shared_ptr<FileStream> &st, bool active) {
        long written = st->num_files_written;
        long errored = st->num_files_errored;
        long queued  = st->num_files_queued;

        fs::StreamStatus status =
            active ? fs::STREAM_STATUS_ACTIVE
                   : ((written + errored == queued) ? fs::STREAM_STATUS_INACTIVE
                                                    : fs::STREAM_STATUS_DRAINING);

        fs::StartStreamRequest *args = info->mutable_args();
        args->set_acq_name(st->acq_name);
        args->set_filename_pattern(st->pattern_string);
        for (long beam_id : st->beam_ids)
            args->add_beam_ids(beam_id);
        args->set_fpga_seq_start(st->fpga_seq_start);
        args->set_fpga_seq_end(st->fpga_seq_end);

        info->set_status(status);
        info->set_cancelled(st->cancelled);
        info->set_started_at_unix_ns(st->started_at_unix_ns);
        info->set_deactivated_at_unix_ns(st->deactivated_at_unix_ns);
        info->set_num_files_queued(queued);
        info->set_num_files_written(written);
        info->set_num_files_errored(errored);
    };

    for (const auto &st : s->active_streams)
        fill_stream_info(response->add_streams(), st, /*active=*/ true);

    // Inactive ring, oldest to newest (deactivation order).
    constexpr long cap = constants::inactive_file_stream_capacity;
    long n = s->num_deactivated_streams;
    for (long i = std::max(0L, n - cap); i < n; i++)
        fill_stream_info(response->add_streams(), s->inactive_streams[i % cap], /*active=*/ false);
}

grpc::Status FrbRpcService::ShowStreams(
    grpc::ServerContext* context,
    const fs::ShowStreamsRequest* request,
    fs::ShowStreamsResponse* response)
{
    try {
        _check_protocol_version(request->protocol_version(), "ShowStreams");
        _ShowStreams(request, response);
        return grpc::Status::OK;
    } catch (const std::exception &e) {
        return grpc::Status(grpc::StatusCode::INTERNAL, e.what());
    } catch (...) {
        return grpc::Status(grpc::StatusCode::INTERNAL, "Unknown error in ShowStreams");
    }
}

// ---- CancelStream ----

void FrbRpcService::_CancelStream(const fs::CancelStreamRequest *request, fs::CancelStreamResponse *response)
{
    shared_ptr<FrbServer> s = _lock_state();

    unique_lock<std::mutex> server_lock(s->mutex);

    if (!s->rb_initialized)
        throw runtime_error("CancelStream: server has not yet established an initial fpga chunk");

    // Note: cancellation only stops future matching. File writes already
    // queued to the FileWriter still complete, and still notify
    // SubscribeFiles subscribers with this stream's acq_name. Cancelled
    // streams remain visible in ShowStreams (inactive ring history).

    if (request->cancel_all()) {
        // Deactivate ALL active streams, in active_streams (registration)
        // order, however many there are: num_cancelled counts every one,
        // even when more than the ring capacity are cancelled in a single
        // request and the ring immediately evicts the oldest. (Ring
        // residency is display-only history; cancellation semantics never
        // depend on it.)
        long n = long(s->active_streams.size());
        for (const auto &st : s->active_streams)
            s->_deactivate_stream(st, /*cancelled=*/ true);
        s->active_streams.clear();
        response->set_num_cancelled(n);
        return;
    }

    const string &acq_name = request->acq_name();
    for (auto it = s->active_streams.begin(); it != s->active_streams.end(); ++it) {
        if ((*it)->acq_name == acq_name) {
            s->_deactivate_stream(*it, /*cancelled=*/ true);
            s->active_streams.erase(it);
            response->set_num_cancelled(1);
            return;
        }
    }

    // Not active. Distinguish "already inactive" (present in the ring) from
    // "never heard of" for a clearer error message.
    constexpr long cap = constants::inactive_file_stream_capacity;
    long n = s->num_deactivated_streams;
    for (long i = std::max(0L, n - cap); i < n; i++) {
        if (s->inactive_streams[i % cap]->acq_name == acq_name) {
            stringstream ss;
            ss << "CancelStream: stream '" << acq_name << "' is already inactive";
            throw runtime_error(ss.str());
        }
    }

    stringstream ss;
    ss << "CancelStream: no active stream with acq_name '" << acq_name << "'";
    throw runtime_error(ss.str());
}

grpc::Status FrbRpcService::CancelStream(
    grpc::ServerContext* context,
    const fs::CancelStreamRequest* request,
    fs::CancelStreamResponse* response)
{
    try {
        _check_protocol_version(request->protocol_version(), "CancelStream");
        _CancelStream(request, response);
        return grpc::Status::OK;
    } catch (const std::exception &e) {
        return grpc::Status(grpc::StatusCode::INTERNAL, e.what());
    } catch (...) {
        return grpc::Status(grpc::StatusCode::INTERNAL, "Unknown error in CancelStream");
    }
}

// ---- SubscribeFiles ----

// Subscribes to file write notifications from FileWriter.
// If the client closes the connection, exits gracefully.
//
// Note: gRPC's server-streaming model provides one thread of execution per connection
// via gRPC's internal thread pool.
//
// Each response has (filename, error_message, acq_name). Empty error_message
// indicates success; nonempty acq_name means the file was triggered by a
// stream (StartStream RPC) rather than a WriteFiles call. Stream-triggered
// notifications are delivered only if request->subscribe_streams() is true.
void FrbRpcService::_SubscribeFiles(grpc::ServerContext* context, const fs::SubscribeFilesRequest *request, grpc::ServerWriter<fs::SubscribeFilesResponse>* writer)
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

        // Stream-triggered notifications (nonempty acq_name) are delivered
        // only if the subscriber opted in via subscribe_streams.
        if (!request->subscribe_streams() && !write_status.acq_name.empty())
            continue;

        // Build response with (filename, error_message, acq_name) inside
        // the 'notification' arm of the oneof. Empty error_message
        // indicates success.
        fs::SubscribeFilesResponse response;
        auto *notif = response.mutable_notification();
        notif->set_filename(write_status.save_path.string());
        notif->set_acq_name(write_status.acq_name);

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
        _check_protocol_version(request->protocol_version(), "SubscribeFiles");
        _SubscribeFiles(context, request, writer);
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
    for (long v : c.frequency_subband_counts)
        response->add_frequency_subband_counts(v);
    for (long v : c.zone_nfreq)
        response->add_fake_zone_nfreq(v);
    for (double v : c.zone_freq_edges)
        response->add_fake_zone_freq_edges(v);
    response->set_fake_time_sample_ms(c.time_sample_ms);
    response->set_fake_nbeams(c.beams_per_gpu);

    // Search reach, used by the fake X-engine to bound the simulated FRBs it injects.
    // (max_width is in frame time samples, not ms -- see DedispersionConfig.)
    response->set_max_dm_of_all_trees(c.max_dm_of_all_trees());
    response->set_max_width_of_base_tree(c.max_width_of_base_tree());
}

grpc::Status FrbRpcService::GetConfig(
    grpc::ServerContext* context,
    const fs::GetConfigRequest* request,
    fs::GetConfigResponse* response)
{
    try {
        _check_protocol_version(request->protocol_version(), "GetConfig");
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
        _check_protocol_version(request->protocol_version(), "MonitorRingbuf");
        _MonitorRingbuf(context, writer);
        return grpc::Status::OK;
    } catch (const std::exception &e) {
        return grpc::Status(grpc::StatusCode::INTERNAL, e.what());
    } catch (...) {
        return grpc::Status(grpc::StatusCode::INTERNAL, "Unknown error in MonitorRingbuf");
    }
}


}  // namespace pirate
