#include "../include/pirate/FrbServer.hpp"

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


// Service implementation (forward-declared in header)
// Note: All RPC methods should wrap their logic in try-catch, to gracefully return
// an error status to the client (instead of crashing the server).
class FrbRpcService final : public fs::FrbSearch::Service {
public:
    std::shared_ptr<FrbServerImpl> state;

    FrbRpcService(const shared_ptr<FrbServerImpl> &s) : state(s) {}

    // ---- GetStatus ----

    void _GetStatus(const fs::GetStatusRequest *request, fs::GetStatusResponse *response)
    {
        // Call Receiver::get_status() for each receiver,
        // and sum the results over receivers.
        long total_conn = 0, total_bytes = 0;
        for (auto &r : state->params.receivers) {
            long nc, nb;
            r->get_status(nc, nb);
            total_conn += nc;
            total_bytes += nb;
        }
        response->set_num_connections(total_conn);
        response->set_num_bytes(total_bytes);

        // Get ring buffer state under lock.
        {
            lock_guard<std::mutex> lock(state->mutex);
            response->set_rb_start(state->rb_start);
            response->set_rb_reaped(state->rb_reaped);
            response->set_rb_finalized(state->rb_finalized);
            response->set_rb_end(state->rb_end);
        }

        // Get num_free_frames from the allocator (permissive=true to handle uninitialized state).
        auto &allocator = state->params.receivers[0]->params.allocator;
        response->set_num_free_frames(allocator->num_free_frames(/*permissive=*/true));
    }

    grpc::Status GetStatus(
        grpc::ServerContext* context,
        const fs::GetStatusRequest* request,
        fs::GetStatusResponse* response) override
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

    // ---- GetMetadata ----

    void _GetMetadata(const fs::GetMetadataRequest *request, fs::GetMetadataResponse *response)
    {
        // Check has_metadata under lock, then release.
        // Safe because has_metadata transitions false->true exactly once,
        // and metadata is immutable after being set.
        bool has_metadata;
        {
            lock_guard<std::mutex> lock(state->mutex);
            has_metadata = state->has_metadata;
        }

        if (has_metadata) {
            response->set_yaml_string(state->metadata.to_yaml_string(request->verbose()));
        } else {
            response->set_yaml_string("");
        }
    }

    grpc::Status GetMetadata(
        grpc::ServerContext* context,
        const fs::GetMetadataRequest* request,
        fs::GetMetadataResponse* response) override
    {
        try {
            _GetMetadata(request, response);
            return grpc::Status::OK;
        } catch (const std::exception &e) {
            return grpc::Status(grpc::StatusCode::INTERNAL, e.what());
        } catch (...) {
            return grpc::Status(grpc::StatusCode::INTERNAL, "Unknown error in GetMetadata");
        }
    }
};



// -------------------------------------------------------------------------------------------------
//
// FrbServer


FrbServer::FrbServer(const Params &params)
{
    // Note: Params argument-checking happens in the FrbServerImpl constructor.
    this->state = make_shared<FrbServerImpl> (params);
    this->rpc_service = make_unique<FrbRpcService> (state);

    // Spawn watchdog thread (waits for state to stop, then propagates error).
    watchdog_thread = std::thread(&FrbServer::watchdog_thread_main, this);
}


FrbServer::~FrbServer()
{
    stop();

    if (watchdog_thread.joinable())
        watchdog_thread.join();

    if (rpc_server)
        rpc_server->Wait();
}


void FrbServer::_throw_if_stopped(const char *method_name)
{
    if (error)
        std::rethrow_exception(error);

    if (is_stopped) {
        throw runtime_error(string(method_name) + " called on stopped instance");
    }
}


void FrbServer::_watchdog_thread_main()
{
    // Wait for FrbServerImpl::stop().
    // Note that this also waits for FrbServer::stop(), since FrbServer::stop() calls FrbServerImpl::stop().

    unique_lock<std::mutex> state_lock(state->mutex);

    while (!state->is_stopped)
        state->cv.wait(state_lock);

    std::exception_ptr e = state->error;
    state_lock.unlock();

    // Propagate to FrbServer.
    this->stop(e);
}


void FrbServer::watchdog_thread_main()
{
    try {
        _watchdog_thread_main();
    } catch (...) {
        stop(std::current_exception());
    }
}


void FrbServer::start()
{
    unique_lock<std::mutex> lock(mutex);
    _throw_if_stopped("FrbServer::start");

    if (is_started)
        throw runtime_error("FrbServer::start() called twice");

    is_started = true;
    lock.unlock();

    try {
        // Start the RPC server.
        grpc::ServerBuilder builder;
        builder.AddListeningPort(state->params.rpc_server_address, grpc::InsecureServerCredentials());
        builder.RegisterService(rpc_service.get());
        this->rpc_server = builder.BuildAndStart();

        // Start FrbServerImpl (starts receivers, spawns worker/reaper threads).
        state->start();
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

    // Stop FrbServerImpl (stops receivers, allocator, notifies threads).
    state->stop();

    // Shutdown RPC server.
    if (rpc_server)
        rpc_server->Shutdown();
}


// -------------------------------------------------------------------------------------------------
//
// FrbServerImpl


FrbServerImpl::FrbServerImpl(const Params &p) : params(p) 
{ 
    xassert(params.receivers.size() > 0);
    xassert(params.rpc_server_address.size() > 0);  // check that string was initialized

    // Check that all recivers use the same allocator, and consumer IDs are consistent with ordering.
    for (uint i = 0; i < params.receivers.size(); i++) {
        xassert(params.receivers[i]);
        xassert(params.receivers[i]->params.allocator == params.receivers[0]->params.allocator);
        xassert(params.receivers[i]->params.consumer_id == i);
    }

    this->allocator = params.receivers[0]->params.allocator;
}


FrbServerImpl::~FrbServerImpl()
{
    stop();

    for (auto &w : workers)
        if (w.joinable())
            w.join();

    if (reaper_thread.joinable())
        reaper_thread.join();
}


void FrbServerImpl::_throw_if_stopped(const char *method_name)
{
    if (error)
        std::rethrow_exception(error);

    if (is_stopped) {
        throw runtime_error(string(method_name) + " called on stopped instance");
    }
}


void FrbServerImpl::_worker_main(int receiver_index)
{
    auto &receiver = params.receivers.at(receiver_index);
    long num_receivers = params.receivers.size();

    // Get metadata from receiver (blocking).
    XEngineMetadata m = receiver->get_metadata(true);  // blocking=true
    long rb_size = FrbServerImpl::ringbuf_nchunks * m.nbeams;
    long nbeams = m.nbeams;

    // Check consistency with other receivers.
    unique_lock<std::mutex> lock(mutex);

    if (!has_metadata) {
        metadata = m;
        has_metadata = true;
        // The frame_ringbuf is initialized at the same time as the metadata,
        // without dropping the lock in between. Correctness of worker_main() 
        // and reaper_main() depend on this property.
        frame_ringbuf.resize(rb_size);
        cv.notify_all();  // wake up reaper thread
    } else {
        XEngineMetadata::check_sender_consistency(metadata, m);
        xassert(long(frame_ringbuf.size()) == rb_size);
    }

    lock.unlock();

    for (long frame_id = 0; true; frame_id++) {
        long rb_slot = frame_id % rb_size;
        long expected_time_chunk_index = frame_id / nbeams;
        long expected_beam_id = m.beam_ids.at(frame_id % nbeams);

        shared_ptr<AssembledFrame> frame = receiver->get_frame();
        xassert(frame->time_chunk_index == expected_time_chunk_index);
        xassert(frame->beam_id == expected_beam_id);

        // Insert the new frame into frame_ringbuf. Note that each frame
        // must be received from all Receivers before it is "finalized".

        lock.lock();

        // In principle, this assert can fail if one Receiver is running far behind.
        // I decided it was best to "handle" this condition by throwing an exception,
        // since something has gone off the rails, and needs human debugging.
        xassert(rb_finalized >= frame_id - rb_size + 1);

        unique_lock<std::mutex> frame_lock(frame->mutex);
        frame->finalize_count++;

        if (frame->finalize_count == 1) {
            // Frame received from first Receiver. Check that it is not in
            // the ringbuf already, and put it at the end of the ringbuf.
            xassert(rb_end == frame_id);
            frame_ringbuf[rb_slot] = frame;
            rb_start = max(frame_id - rb_size + 1, 0L);
            rb_reaped = max(rb_start, rb_reaped);
            rb_end = frame_id + 1;
            // Note that frame is not finalized yet (see below).
        }
        else {
            // Frame has previously been received from another Receiver.
            // Check that it is already in the ringbuf, but not finalized yet.
            xassert(frame_id >= rb_finalized);
            xassert(frame_id < rb_end);
            xassert(frame_ringbuf[rb_slot] == frame);
        }

        if (frame->finalize_count == num_receivers) {
            // Frame received from last Receiver, so finalize it.
            xassert(rb_finalized == frame_id);
            rb_finalized++;
        }

        _check_rb_invariants();

        lock.unlock();
        frame_lock.unlock();
        cv.notify_all();
    }
}


void FrbServerImpl::worker_main(int receiver_index)
{
    try {
        _worker_main(receiver_index);
    } catch (...) {
        stop(std::current_exception());
    }
}


void FrbServerImpl::_reaper_thread_main()
{
    unique_lock<std::mutex> lock(mutex);

    // Wait for metadata.
    for (;;) {
        if (is_stopped)
            return;
        if (has_metadata)
            break;
        cv.wait(lock);
    }

    lock.unlock();

    // It's okay to access metadata.nbeams and frame_ringbuf.size() after dropping
    // the lock, since these are initialized once and constant thereafter.
    long nbeams = metadata.nbeams;
    long rb_size = frame_ringbuf.size();

    // Get total number of frames (blocking until allocator is initialized).
    long total_frames = allocator->num_total_frames(/*blocking=*/ true);
    xassert(total_frames >= 6 * nbeams);

    for (;;) {
        allocator->block_until_low_memory(2 * nbeams);

        unique_lock<std::mutex> lock(mutex);

        // Wait for a reapable frame (i.e. rb_reaped < rb_finalized).
        // Note that worker_main() signals the condition_variable when new frames are added.
        for (;;) {
            if (is_stopped)
                return;
            if (rb_reaped < rb_finalized)
                break;
            cv.wait(lock);
        }

        long rb_slot = rb_reaped % rb_size;
        shared_ptr<AssembledFrame> frame = frame_ringbuf[rb_slot];
        rb_reaped++;
        lock.unlock();

        lock_guard<std::mutex> frame_lock(frame->mutex);
        frame->data = Array<void> ();
        // frame_lock dropped (by going out of scope)
    }
}


void FrbServerImpl::reaper_thread_main()
{
    try {
        _reaper_thread_main();
    } catch (...) {
        stop(std::current_exception());
    }
}


void FrbServerImpl::start()
{
    unique_lock<std::mutex> lock(mutex);
    _throw_if_stopped("FrbServerImpl::start");

    if (is_started)
        throw runtime_error("FrbServerImpl::start() called twice");

    is_started = true;
    lock.unlock();

    try {
        // Start all receivers.
        for (auto &r : params.receivers)
            r->start();

        // Spawn one worker thread per receiver.
        int nreceivers = params.receivers.size();
        for (int i = 0; i < nreceivers; i++)
            workers.emplace_back(&FrbServerImpl::worker_main, this, i);

        // Spawn reaper thread iff allocator is not in dummy mode.
        if (!allocator->is_dummy())
            reaper_thread = std::thread(&FrbServerImpl::reaper_thread_main, this);
    } catch (...) {
        stop(std::current_exception());
        throw;
    }
}


void FrbServerImpl::stop(std::exception_ptr e)
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

    // Stop the allocator (which will unblock num_total_frames).
    allocator->stop();
}


}  // namespace pirate