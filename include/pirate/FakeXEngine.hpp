#ifndef _PIRATE_FAKE_XENGINE_HPP
#define _PIRATE_FAKE_XENGINE_HPP

#include <atomic>
#include <condition_variable>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "XEngineMetadata.hpp"


namespace pirate {
#if 0
}   // pacify editor auto-indent
#endif


// FakeXEngine: simulates 64 upstream X-engine nodes sending data to a single receiver.
//
// This class is a "thread-backed class" (see notes/thread_backed_class.md) that spawns
// multiple worker threads, each sending data over a TCP connection following the X->FRB
// network protocol (see notes/network_protocol.md).
//
// Usage:
//   XEngineMetadata xmd = XEngineMetadata::from_yaml_file("...");
//   FakeXEngine fxe(xmd, "127.0.0.1", 8787, 64);
//   fxe.start();  // creates worker threads and begins sending
//   // ... wait ...
//   fxe.stop();   // signals threads to stop
//
// Worker threads are created in start(), not the constructor. Each worker thread:
//   - Opens a TCP connection to the receiver
//   - Sends protocol header (magic number + YAML metadata)
//   - Sends shape-(nbeams, nfreq, 256) int4 data arrays (all zeros for now)
//
// Frequency channels are assigned round-robin to worker threads. Before sending
// the N-th data array, each worker waits until all threads have finished sending
// the (N-2)-th array, to keep threads synchronized.

struct FakeXEngine
{
    // Constructor args.
    const XEngineMetadata xmd;
    const std::string ip_addr;
    const uint16_t port;
    const int nthreads;

    // Thread-backed class state (protected by 'mutex').
    std::mutex mutex;
    std::condition_variable cv;
    bool is_stopped = false;
    std::exception_ptr error;

    // Worker threads (created in start()).
    std::vector<std::thread> workers;

    // Synchronization: tracks how many arrays have been completed across all threads.
    // After sending array i, each thread increments this counter.
    // Before sending array N, each thread waits until arrays_completed >= nthreads * (N-1).
    std::atomic<long> arrays_completed{0};

    // Timeout for send operations (milliseconds).
    static constexpr int send_timeout_ms = 10;

    // ----- Public interface -----

    // Constructor. Does not create worker threads (call start() for that).
    FakeXEngine(const XEngineMetadata &xmd, const std::string &ip_addr, uint16_t port, int nthreads);

    // Destructor calls stop() and joins worker threads.
    ~FakeXEngine();

    // Create worker threads and begin sending data.
    // Entry point: throws if already started or stopped.
    void start();

    // Put FakeXEngine into stopped state. Worker threads exit promptly.
    // If 'e' is non-null, it represents an error; otherwise normal termination.
    void stop(std::exception_ptr e = nullptr);

    // ----- Noncopyable, nonmoveable -----

    FakeXEngine(const FakeXEngine &) = delete;
    FakeXEngine &operator=(const FakeXEngine &) = delete;
    FakeXEngine(FakeXEngine &&) = delete;
    FakeXEngine &operator=(FakeXEngine &&) = delete;

private:
    // Helper: create XEngineMetadata for a specific worker thread (with subset of freq channels).
    XEngineMetadata make_worker_metadata(int thread_id) const;

    // Helper: check if stopped, with lock held by caller.
    void _throw_if_stopped(const char *method_name);

    // Worker thread main function.
    void _worker_main(int thread_id);

    // Wrapper that catches exceptions and calls stop().
    void worker_main(int thread_id);

    // Helper: wait for synchronization barrier before sending array N.
    // Returns false if stopped while waiting.
    bool wait_for_sync(long array_index);
};


}  // namespace pirate

#endif // _PIRATE_FAKE_XENGINE_HPP
