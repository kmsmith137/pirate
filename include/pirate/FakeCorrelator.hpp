#ifndef _PIRATE_INTERNALS_FAKE_CORRELATOR_HPP
#define _PIRATE_INTERNALS_FAKE_CORRELATOR_HPP

#include <condition_variable>
#include <exception>
#include <mutex>
#include <vector>
#include <string>
#include <thread>

namespace pirate {
#if 0
}  // editor auto-indent
#endif

struct Socket;  // forward declaration (defined in network_utils.hpp)


// FakeCorrelator: simulates a correlator by sending data over TCP connections
// to one or more receiver endpoints. Used for testing/benchmarking the receiver.
//
// This class is a "thread-backed class" (see notes/thread_backed_class.md) that
// spawns one worker thread per endpoint, each sending data as fast as possible
// (optionally rate-limited) over TCP.
//
// Usage:
//   auto fc = make_shared<FakeCorrelator>(send_bufsize, ...);
//   fc->add_endpoint("10.1.1.2", 1, 0.0, vcpu_list);
//   fc->start();   // creates worker threads and begins sending
//   // ... wait ...
//   fc->stop();    // signals threads to stop
//   fc->join();    // waits for threads to exit
//
// Worker threads are created in start(), not the constructor. Each worker thread:
//   - Opens TCP connection(s) to the endpoint on port 8787
//   - Sends data in a loop until stopped or connection is closed

struct FakeCorrelator
{
    // Timeout for send operations (milliseconds).
    static constexpr int send_timeout_ms = 10;

    FakeCorrelator(
        long send_bufsize = 64*1024,
        bool use_zerocopy = true,
        bool use_mmap = false,
        bool use_hugepages = true
    );

    // Destructor calls stop() and joins worker threads.
    ~FakeCorrelator();

    // Add an endpoint. Must be called before start().
    // total_gbps=0 means "no rate limit".
    void add_endpoint(const std::string &ip_addr, long num_tcp_connections, double total_gbps, const std::vector<int> &vcpu_list);

    // Create worker threads and begin sending data.
    // Entry point: throws if already started or stopped.
    void start();

    // Put FakeCorrelator into stopped state. Worker threads exit promptly.
    // If 'e' is non-null, it represents an error; otherwise normal termination.
    void stop(std::exception_ptr e = nullptr);

    // Block until all worker threads have exited.
    void join();

    // ----- Noncopyable, nonmoveable -----

    FakeCorrelator(const FakeCorrelator &) = delete;
    FakeCorrelator &operator=(const FakeCorrelator &) = delete;
    FakeCorrelator(FakeCorrelator &&) = delete;
    FakeCorrelator &operator=(FakeCorrelator &&) = delete;

private:
    struct Endpoint {
        std::string ip_addr;
        long num_tcp_connections;
        double total_gbps;
        std::vector<int> vcpu_list;
    };

    // Constructor args.
    long send_bufsize;
    bool use_zerocopy;
    bool use_mmap;
    bool use_hugepages;

    std::vector<Endpoint> endpoints;

    // Thread-backed class state (protected by 'mutex').
    std::mutex mutex;
    std::condition_variable cv;
    bool is_stopped = false;
    bool is_started = false;
    std::exception_ptr error;

    // Worker threads (created in start()).
    std::vector<std::thread> workers;

    // Helper: check if stopped, with lock held by caller.
    void _throw_if_stopped(const char *method_name);

    // Worker thread main function. Returns total bytes sent.
    long _worker_main(long endpoint_index);

    // Wrapper that catches exceptions and calls stop().
    void worker_main(long endpoint_index);

    // Helper: send all bytes from buffer, using short timeouts to allow prompt exit.
    // Returns false if stopped or connection reset.
    bool _send_all(Socket &sock, const void *buf, long nbytes);
};


}  // namespace pirate

#endif // _PIRATE_INTERNALS_FAKE_CORRELATOR_HPP
