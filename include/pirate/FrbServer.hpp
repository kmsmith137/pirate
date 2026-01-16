#ifndef _PIRATE_FRB_SERVER_HPP
#define _PIRATE_FRB_SERVER_HPP

#include <condition_variable>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// Forward declarations
namespace grpc { class Server; }

namespace pirate {
#if 0
}  // editor auto-indent
#endif

// Forward declare (defined in FrbServer.cpp)
struct Receiver;       // defined in Receiver.{hpp.cu}
struct FrbRpcService;  // defined in FrbServer.cu


// FrbServer: a thread-backed class that manages Receivers and an RPC service.
// One worker thread is created per Receiver.

struct FrbServer
{
    // The Params struct does double duty: it contains constructor
    // arguments, and also allows the FrbServer and FrbRpcService
    // to share state (via shared_ptr<Params> members).

    struct Params {
        std::vector<std::shared_ptr<Receiver>> receivers;
        std::string rpc_server_address;
    };

    FrbServer(const std::shared_ptr<Params> &params);
    ~FrbServer();  // calls stop(), joins workers, then rpc_server->Wait()

    // Start/stop the Receivers and the RPC service.
    // Asynchronous: neither start() nor stop() calls rpc_server->Wait().
    void start();  // entry point
    void stop(std::exception_ptr e = nullptr);  // idempotent

    std::shared_ptr<Params> params;
    std::unique_ptr<FrbRpcService> rpc_service;
    std::unique_ptr<grpc::Server> rpc_server;

    // Thread-backed class state (protected by mutex).
    std::mutex mutex;
    std::condition_variable cv;
    bool is_started = false;
    bool is_stopped = false;
    std::exception_ptr error;

    // Worker threads (one per receiver).
    std::vector<std::thread> workers;

    // ----- Noncopyable, nonmoveable -----

    FrbServer(const FrbServer &) = delete;
    FrbServer &operator=(const FrbServer &) = delete;
    FrbServer(FrbServer &&) = delete;
    FrbServer &operator=(FrbServer &&) = delete;

private:
    // Helper for entry points. Caller must hold mutex.
    void _throw_if_stopped(const char *method_name);

    // Worker thread functions.
    void _worker_main(int receiver_index);
    void worker_main(int receiver_index);
};


}  // namespace pirate

#endif // _PIRATE_FRB_SERVER_HPP