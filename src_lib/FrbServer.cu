#include "../include/pirate/FrbServer.hpp"

#include "../include/pirate/Receiver.hpp"
#include "../include/pirate/AssembledFrame.hpp"
#include "../include/pirate/XEngineMetadata.hpp"

// Suppress nvcc warning #970 from gRPC headers ("qualifier on friend declaration ignored")
#pragma nv_diagnostic push
#pragma nv_diag_suppress 970
#include "../grpc/frb_search.grpc.pb.h"
#include <grpcpp/grpcpp.h>
#pragma nv_diagnostic pop

#include <stdexcept>
#include <ksgpu/xassert.hpp>

using namespace std;


namespace pirate {
#if 0
}  // editor auto-indent
#endif

namespace fs = frb::search::v1;


struct FrbServer::State
{
    FrbServer::Params params;

    // Metadata from receivers (protected by 'mutex').
    // Used to check that all receivers send consistent metadata.
    std::mutex mutex;
    bool has_metadata = false;
    XEngineMetadata metadata;

    // The frame_ringbuf is initialized at the same time as the metadata.
    // Ring buffer has length (FrbServer::ringbuf_nchunks * metadata.nbeams).
    vector<shared_ptr<AssembledFrame>> frame_ringbuf;

    // "Frame ids" are defined as (time_chunk_index * nbeams + ibeam).
    // Invariant: rb_start <= rb_finalized <= rb_end <= (rb_start + frame_ringbuf.size()).
    long rb_start = 0;       // (first frame_id in ringbuf)
    long rb_finalized = 0;   // (last finalized frame_id in ringbuf) + 1
    long rb_end = 0;         // (last frame_id in ringbuf) + 1

    State(const FrbServer::Params &p) : params(p) { }

    // Call with lock held.
    inline void _check_rb_invariants()
    {
        xassert(rb_start >= 0);
        xassert(rb_start <= rb_finalized);
        xassert(rb_finalized <= rb_end);
        xassert(rb_end <= rb_start + long(frame_ringbuf.size()));
    }
};


// -------------------------------------------------------------------------------------------------


// Service implementation (forward-declared in header)
class FrbRpcService final : public fs::FrbSearch::Service {
public:
    std::shared_ptr<FrbServer::State> state;

    FrbRpcService(const shared_ptr<FrbServer::State> &s) : state(s) {}

    grpc::Status GetStatus(
        grpc::ServerContext* context,
        const fs::GetStatusRequest* request,
        fs::GetStatusResponse* response) override
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
        return grpc::Status::OK;
    }

    grpc::Status GetMetadata(
        grpc::ServerContext* context,
        const fs::GetMetadataRequest* request,
        fs::GetMetadataResponse* response) override
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
        return grpc::Status::OK;
    }
};


// -------------------------------------------------------------------------------------------------


FrbServer::FrbServer(const Params &params)
{
    xassert(params.receivers.size() > 0);
    xassert(params.rpc_server_address.size() > 0);  // check that string was initialized

    // Check that all recivers use the same allocator, and consumer IDs are consistent with ordering.
    for (uint i = 0; i < params.receivers.size(); i++) {
        xassert(params.receivers[i]);
        xassert(params.receivers[i]->params.allocator == params.receivers[0]->params.allocator);
        xassert(params.receivers[i]->params.consumer_id == i);
    }

    this->state = make_shared<FrbServer::State> (params);
    this->rpc_service = make_unique<FrbRpcService> (state);

    grpc::ServerBuilder builder;
    builder.AddListeningPort(params.rpc_server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(rpc_service.get());
    this->rpc_server = builder.BuildAndStart();
}


FrbServer::~FrbServer()
{
    stop();

    for (auto &w : workers)
        if (w.joinable())
            w.join();

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


void FrbServer::_worker_main(int receiver_index)
{
    auto &receiver = state->params.receivers.at(receiver_index);
    long num_receivers = state->params.receivers.size();

    // Get metadata from receiver (blocking).
    XEngineMetadata m = receiver->get_metadata(true);  // blocking=true
    long rb_size = FrbServer::ringbuf_nchunks * m.nbeams;
    long nbeams = m.nbeams;

    // Check consistency with other receivers.
    unique_lock<std::mutex> state_lock(state->mutex);

    if (!state->has_metadata) {
        state->metadata = m;
        state->has_metadata = true;
        // The frame_ringbuf is initialized at the same time as the metadata.
        state->frame_ringbuf.resize(rb_size);
    } else {
        XEngineMetadata::check_sender_consistency(state->metadata, m);
        xassert(long(state->frame_ringbuf.size()) == rb_size);
    }

    state_lock.unlock();

    for (long frame_id = 0; true; frame_id++) {
        long rb_slot = frame_id % rb_size;
        long expected_time_chunk_index = frame_id / nbeams;
        long expected_beam_id = m.beam_ids.at(frame_id % nbeams);

        shared_ptr<AssembledFrame> frame = receiver->get_frame();
        xassert(frame->time_chunk_index == expected_time_chunk_index);
        xassert(frame->beam_id == expected_beam_id);

        // Insert the new frame into state->frame_ringbuf. Note that each frame
        // must be received from all Receivers before it is "finalized".

        state_lock.lock();

        // In principle, this assert can fail if one Receiver is running far behind.
        // I decided it was best to "handle" this condition by throwing an exception,
        // since something has gone off the rails, and needs human debugging.
        xassert(state->rb_finalized >= frame_id - rb_size + 1);

        lock_guard<std::mutex> frame_lock(frame->mutex);
        frame->finalize_count++;

        if (frame->finalize_count == 1) {
            // Frame received from first Receiver. Check that it is not in
            // the ringbuf already, and put it at the end of the ringbuf.
            xassert(state->rb_end == frame_id);
            state->frame_ringbuf[rb_slot] = frame;
            state->rb_start = max(frame_id - rb_size + 1, 0L);
            state->rb_end = frame_id + 1;
            // Note that frame is not finalized yet (see below).
        }
        else {
            // Frame has previously been received from another Receiver.
            // Check that it is already in the ringbuf, but not finalized yet.
            xassert(frame_id >= state->rb_finalized);
            xassert(frame_id < state->rb_end);
            xassert(state->frame_ringbuf[rb_slot] == frame);
        }

        if (frame->finalize_count == num_receivers) {
            // Frame received from last Receiver, so finalize it.
            xassert(state->rb_finalized == frame_id);
            state->rb_finalized++;
        }

        state->_check_rb_invariants();

        state_lock.unlock();
        // Note that frame_lock is also dropped here.
    }
}


void FrbServer::worker_main(int receiver_index)
{
    try {
        _worker_main(receiver_index);
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

    // Start all receivers.
    for (auto &r : state->params.receivers)
        r->start();

    // Spawn one worker thread per receiver.
    int nreceivers = state->params.receivers.size();
    for (int i = 0; i < nreceivers; i++)
        workers.emplace_back(&FrbServer::worker_main, this, i);
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
    for (auto &r : state->params.receivers)
        r->stop();

    // Shutdown RPC server.
    if (rpc_server)
        rpc_server->Shutdown();
}


}  // namespace pirate