#include "../include/pirate/FrbServer.hpp"
#include "../include/pirate/Receiver.hpp"
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
    mutex mutex;
    bool has_metadata = false;
    XEngineMetadata metadata;

    State(const FrbServer::Params &p) : params(p) { }
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
};


// -------------------------------------------------------------------------------------------------


FrbServer::FrbServer(const Params &params)
{
    xassert(params.receivers.size() > 0);
    xassert(params.rpc_server_address.size() > 0);  // check that string was initialized

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

    // Get metadata from receiver (blocking).
    XEngineMetadata m = receiver->get_metadata(true);  // blocking=true

    // Check consistency with other receivers.
    lock_guard<std::mutex> lock(state->mutex);

    if (!state->has_metadata) {
        state->metadata = m;
        state->has_metadata = true;
    } else {
        XEngineMetadata::check_sender_consistency(state->metadata, m);
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