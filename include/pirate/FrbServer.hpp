#ifndef _PIRATE_FRB_SERVER_HPP
#define _PIRATE_FRB_SERVER_HPP

#include "Receiver.hpp"
#include "AssembledFrame.hpp"
#include "XEngineMetadata.hpp"

#include <ksgpu/xassert.hpp>

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

// Forward declaration (defined in FrbServer.cpp)
struct FrbServerImpl;  // defined later in this file
struct FrbRpcService;  // defined in FrbServer.cpp



// FrbServer: a thread-backed class that manages Receivers and an RPC service.
// One worker thread is created per Receiver.

struct FrbServer
{
    struct Params {
        std::vector<std::shared_ptr<Receiver>> receivers;
        std::string rpc_server_address;
    };

    FrbServer(const Params &params);
    ~FrbServer();  // calls stop(), joins workers, then rpc_server->Wait()

    // Start/stop the Receivers and the RPC service.
    // Asynchronous: neither start() nor stop() calls rpc_server->Wait().
    void start();  // entry point
    void stop(std::exception_ptr e = nullptr);  // idempotent

    // Shared state between FrbServer and FrbRpcService
    std::shared_ptr<FrbServerImpl> state;  

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

    // Reaper thread (only spawned in non-dummy mode).
    std::thread reaper_thread;

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

    // Reaper thread functions.
    void _reaper_thread_main();
    void reaper_thread_main();
};


// FrbServerImpl: Shared state between FrbServer and FrbRpcService.

struct FrbServerImpl
{
    using Params = FrbServer::Params;

    Params params;

    // Metadata from receivers (protected by 'mutex').
    // Used to check that all receivers send consistent metadata.
    std::mutex mutex;
    std::condition_variable cv;  // signaled when metadata becomes available
    bool has_metadata = false;
    XEngineMetadata metadata;

    // The frame_ringbuf is initialized at the same time as the metadata.
    // Ring buffer has length (ringbuf_nchunks * metadata.nbeams).
    static constexpr int ringbuf_nchunks = 512;
    std::vector<std::shared_ptr<AssembledFrame>> frame_ringbuf;

    // "Frame ids" are defined as (time_chunk_index * nbeams + ibeam).
    // Invariant: rb_start <= rb_finalized <= rb_end <= (rb_start + frame_ringbuf.size()).
    long rb_start = 0;       // (first frame_id in ringbuf)
    long rb_finalized = 0;   // (last finalized frame_id in ringbuf) + 1
    long rb_end = 0;         // (last frame_id in ringbuf) + 1

    FrbServerImpl(const Params &p) : params(p) { }

    // Call with lock held.
    inline void _check_rb_invariants()
    {
        xassert(rb_start >= 0);
        xassert(rb_start <= rb_finalized);
        xassert(rb_finalized <= rb_end);
        xassert(rb_end <= rb_start + long(frame_ringbuf.size()));
    }
};


}  // namespace pirate

#endif // _PIRATE_FRB_SERVER_HPP