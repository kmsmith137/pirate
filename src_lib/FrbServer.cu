#include "../include/pirate/FrbServer.hpp"
#include "../include/pirate/Receiver.hpp"

// Suppress nvcc warning #970 from gRPC headers ("qualifier on friend declaration ignored")
#pragma nv_diagnostic push
#pragma nv_diag_suppress 970
#include "../grpc/frb_search.grpc.pb.h"
#include <grpcpp/grpcpp.h>
#pragma nv_diagnostic pop

#include <stdexcept>

namespace pirate {
#if 0
}  // editor auto-indent
#endif

namespace fs = frb::search::v1;

// Service implementation (forward-declared in header)
class FrbRpcService final : public fs::FrbSearch::Service {
public:
    std::shared_ptr<FrbServer::Params> params;

    FrbRpcService(const std::shared_ptr<FrbServer::Params> &p) : params(p) {}

    grpc::Status GetStatus(
        grpc::ServerContext* context,
        const fs::GetStatusRequest* request,
        fs::GetStatusResponse* response) override
    {
        // Call Receiver::get_status() for each receiver,
        // and sum the results over receivers.
        long total_conn = 0, total_bytes = 0;
        for (auto &r : params->receivers) {
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


FrbServer::FrbServer(const std::shared_ptr<Params> &p)
    : params(p),
      rpc_service(std::make_unique<FrbRpcService>(p))
{
    grpc::ServerBuilder builder;
    builder.AddListeningPort(params->rpc_server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(rpc_service.get());
    rpc_server = builder.BuildAndStart();
}


FrbServer::~FrbServer()
{
    stop();
    if (rpc_server)
        rpc_server->Wait();
}


void FrbServer::start()
{
    std::unique_lock<std::mutex> lock(mutex);

    if (is_started)
        throw std::runtime_error("FrbServer::start() called twice");
    if (is_stopped)
        throw std::runtime_error("FrbServer::start() called after stop()");

    is_started = true;
    lock.unlock();

    for (auto &r : params->receivers)
        r->start();
}


void FrbServer::stop()
{
    std::unique_lock<std::mutex> lock(mutex);

    if (is_stopped)
        return;

    is_stopped = true;
    lock.unlock();

    if (rpc_server)
        rpc_server->Shutdown();
    for (auto &r : params->receivers)
        r->stop();
}


}  // namespace pirate