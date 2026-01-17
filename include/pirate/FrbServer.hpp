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


struct FrbServerImpl;  // defined later in this file
struct FrbRpcService;  // defined in FrbServer.cpp



// FrbServer: a thread-backed class that manages Receivers and an RPC service.
// Contains a watchdog thread that propagates errors from FrbServerImpl.

struct FrbServer
{
    struct Params {
        std::vector<std::shared_ptr<Receiver>> receivers;
        std::string rpc_server_address;
    };

    FrbServer(const Params &params);
    ~FrbServer();  // calls stop(), joins watchdog, then rpc_server->Wait()

    // Start/stop the Receivers and the RPC service.
    // Asynchronous: neither start() nor stop() calls rpc_server->Wait().
    void start();  // entry point
    void stop(std::exception_ptr e = nullptr);  // idempotent

    std::shared_ptr<FrbServerImpl> state;
    std::unique_ptr<FrbRpcService> rpc_service;
    std::unique_ptr<grpc::Server> rpc_server;

    // Thread-backed class state (protected by mutex).
    std::mutex mutex;
    std::condition_variable cv;
    bool is_started = false;
    bool is_stopped = false;
    std::exception_ptr error;

    // Watchdog thread: waits for FrbServerImpl to stop, then propagates the error.
    std::thread watchdog_thread;

    // ----- Noncopyable, nonmoveable -----

    FrbServer(const FrbServer &) = delete;
    FrbServer &operator=(const FrbServer &) = delete;
    FrbServer(FrbServer &&) = delete;
    FrbServer &operator=(FrbServer &&) = delete;

private:
    // Helper for entry points. Caller must hold mutex.
    void _throw_if_stopped(const char *method_name);

    // Watchdog thread functions.
    void _watchdog_thread_main();
    void watchdog_thread_main();
};


// FrbServerImpl: Thread-backed class containing worker and reaper threads.
// Shared state between FrbServer and FrbRpcService.

struct FrbServerImpl
{
    using Params = FrbServer::Params;

    FrbServerImpl(const Params &p);
    ~FrbServerImpl();  // calls stop(), joins workers and reaper

    void start();  // entry point
    void stop(std::exception_ptr e = nullptr);  // idempotent

    Params params;
    std::shared_ptr<AssembledFrameAllocator> allocator;

    // Thread-backed class state (protected by 'mutex').
    std::mutex mutex;
    std::condition_variable cv;  // signaled on: stop, metadata available
    bool is_started = false;
    bool is_stopped = false;
    std::exception_ptr error;

    // Metadata from receivers (protected by 'mutex').
    // Used to check that all receivers send consistent metadata.
    bool has_metadata = false;
    XEngineMetadata metadata;

    // The frame_ringbuf is initialized at the same time as the metadata.
    // Ring buffer has length (ringbuf_nchunks * metadata.nbeams).
    static constexpr int ringbuf_nchunks = 512;
    std::vector<std::shared_ptr<AssembledFrame>> frame_ringbuf;

    // "Frame ids" are defined as (time_chunk_index * nbeams + ibeam).
    // See below for invariants.
    long rb_start = 0;       // (first frame_id in ringbuf)
    long rb_reaped = 0;      // (last reaped frame_id in ringbuf) + 1
    long rb_finalized = 0;   // (last finalized frame_id in ringbuf) + 1
    long rb_end = 0;         // (last frame_id in ringbuf) + 1

    // Worker threads (one per receiver).
    std::vector<std::thread> workers;

    // Reaper thread (only spawned in non-dummy mode).
    std::thread reaper_thread;

    // ----- Noncopyable, nonmoveable -----

    FrbServerImpl(const FrbServerImpl &) = delete;
    FrbServerImpl &operator=(const FrbServerImpl &) = delete;
    FrbServerImpl(FrbServerImpl &&) = delete;
    FrbServerImpl &operator=(FrbServerImpl &&) = delete;

    // Call with lock held.
    inline void _check_rb_invariants()
    {
        xassert(rb_start >= 0);
        xassert(rb_start <= rb_reaped);
        xassert(rb_reaped <= rb_finalized);
        xassert(rb_finalized <= rb_end);
        xassert(rb_end <= rb_start + long(frame_ringbuf.size()));
    }

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


}  // namespace pirate

#endif // _PIRATE_FRB_SERVER_HPP