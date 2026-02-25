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


struct Socket;  // forward declaration (defined in network_utils.hpp)


// FakeXEngine: simulates 64 upstream X-engine nodes sending data to a single receiver.
//
// This class is a "thread-backed class" (see notes/thread_backed_class.md) that spawns
// multiple worker threads, each sending data over a TCP connection following the X->FRB
// network protocol (see notes/network_protocol.md).
//
// Usage:
//   XEngineMetadata xmd = XEngineMetadata::from_yaml_file("...");
//   FakeXEngine fxe(xmd, {"10.0.0.2:5000", "10.0.1.2:5000"}, 64);
//   fxe.start();  // creates worker threads and begins sending
//   // ... wait ...
//   fxe.stop();   // signals threads to stop
//
// Worker threads are created in start(), not the constructor. Each worker thread:
//   - Opens a TCP connection to the receiver
//   - Sends protocol header (magic number + YAML metadata)
//   - Sends shape-(nbeams, nfreq, 256) int4 data arrays (all zeros for now)
//
// Threads are assigned round-robin to IP addresses (nthreads must be a multiple
// of ip_addrs.size()). Frequency channels are assigned round-robin to worker
// threads. Before sending the N-th data array, each worker waits until all
// threads have finished sending the (N-2)-th array, to keep threads synchronized.

struct FakeXEngine
{
    // Protocol magic number (little-endian): 0xf4bf4b01 where 01 is the version number.
    static constexpr uint32_t protocol_magic = 0xf4bf4b01;
    // Timeout for send operations (milliseconds).
    static constexpr int send_timeout_ms = 10;

    // Constructor args.
    const XEngineMetadata xmd;
    const std::vector<std::string> ip_addrs;  // each element is "ip:port"
    const int nthreads;

    // Thread-backed class state (protected by 'mutex').
    std::mutex mutex;
    std::condition_variable cv;
    bool is_stopped = false;
    std::exception_ptr error;

    // Worker threads (created in start()).
    std::vector<std::thread> workers;

    // Synchronization: tracks completion of even/odd arrays.
    // barrier[0] counts completions of arrays 0, 2, 4, ...
    // barrier[1] counts completions of arrays 1, 3, 5, ...
    // Before sending array N, wait for barrier[N%2] >= nthreads * (N/2).
    // Non-atomic, protected by lock.

    long barrier[2] = {0,0};

    // ----- Public interface -----

    // Constructor. Does not create worker threads (call start() for that).
    // Each element of 'ip_addrs' is "ip:port" format. nthreads must be a multiple of ip_addrs.size().
    FakeXEngine(const XEngineMetadata &xmd, const std::vector<std::string> &ip_addrs, int nthreads);

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

    // Helper: send all bytes from buffer, using short timeouts to allow prompt exit.
    // Returns false if stopped or connection reset.
    bool _send_all(Socket &sock, const void *buf, long nbytes);
};


}  // namespace pirate

#endif // _PIRATE_FAKE_XENGINE_HPP
