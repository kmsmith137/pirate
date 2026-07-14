#ifndef _PIRATE_INTERNALS_HWTEST_HPP
#define _PIRATE_INTERNALS_HWTEST_HPP

#include "Barrier.hpp"

#include <condition_variable>
#include <exception>
#include <mutex>
#include <vector>
#include <string>
#include <thread>
#include <memory> // shared_ptr


namespace pirate {
#if 0
}  // editor auto-indent
#endif


struct Socket;  // forward declaration (defined in network_utils.hpp), used by HwtestSender


// Hwtest: hardware performance tests (memory bandwidth, network bandwidth, etc.)
// Thread-backed class (see notes/thread_backed_class.md).

// WARNING on enable_shared_from_this: shared_from_this() is currently unused,
// and must NEVER be handed to worker threads -- workers hold a bare Hwtest*
// by design. If workers held shared_ptr<Hwtest>, the destructor (which is
// what joins the threads) could never run: each worker's reference would keep
// the object alive until the worker exits, and the worker exits only when
// joined.

struct Hwtest : public std::enable_shared_from_this<Hwtest>
{
    // TCP port for the hwtest network test: Hwtest's TcpReceiver binds it, and
    // HwtestSender connects to it.
    static constexpr int tcp_port = 8787;

    // Factory method (constructor is protected).
    static std::shared_ptr<Hwtest> create(const std::string &server_name="Hwtest", bool use_hugepages=true);

    // I've been using 512KB as a default 'recv_bufsize', but I haven't explored this systematically.
    void add_tcp_receiver(const std::string &ip_addr, long num_tcp_connections, long recv_bufsize, const std::vector<int> &vcpu_list, int cpu, int inic);

    // Dedispersion on one GPU, using simplified CHIME parameters.
    void add_chime_dedisperser(int device, const std::vector<int> &vcpu_list, int cpu);

    // The 'src_device' and 'dst_device' args can be (-1) for "host".
    // Important note: it turns out that cudaMemcpy() runs slow for sizes >4GB (!!)
    // Empirically, any blocksize between 1MB and 4GB works pretty well.
    //
    // The 'use_copy_engine' argument is only meaningful for GPU->GPU copies.
    // If use_copy_engine=False, then we copy memory using a GPU kernel, rather than cudaMemcpyAsync().
    // This is useful in a situation where both GPU "compute engines" are being used for GPU->host and host->GPU transfers.
    void add_memcpy_thread(int src_device, int dst_device, long blocksize, bool use_copy_engine, const std::vector<int> &vcpu_list, int cpu);

    // To get multiple threads per SSD, use multiple workers with different 'root_dir' args.
    // If write_asdf=true, writes AssembledFrame ASDF files instead of binary blobs.
    void add_ssd_writer(const std::string &root_dir, long nbytes_per_file, bool write_asdf, const std::vector<int> &vcpu_list, int cpu, int issd);

    // Runs the AVX2 downsampling kernel
    void add_downsampling_thread(int src_bit_depth, long src_nelts, const std::vector<int> &vcpu_list, int cpu);

    // Called by python code, to control server.
    double show_stats();  // returns elapsed time in seconds
    double _show_stats();  // helper, called without lock held.

    void join();
    void start();
    void stop(std::exception_ptr e = nullptr) const;

    // Defined in src_lib/Hwtest.cpp
    struct Stats;
    struct Worker;

    std::string server_name;
    bool use_hugepages;

    // Stop-pattern state ('mutable' since stop() is const -- see
    // notes/stoppable_class.md). is_stopped/error are protected by 'mutex'.
    mutable std::mutex mutex;
    mutable bool is_stopped = false;
    mutable std::exception_ptr error;

    bool is_started = false;
    Barrier barrier;

    // After server is started, 'workers' is immutable.
    // Before server is started, 'workers' is protected by mutex.
    std::vector<std::shared_ptr<Worker>> workers;

    // Published by start() under the mutex; joined via _join_threads().
    std::vector<std::thread> threads;

    void _add_worker(const std::shared_ptr<Worker> &worker, const std::string &caller);

    // Joins all worker threads (synchronizes with start()'s publication of
    // 'threads' via the mutex, then joins with the mutex released). Called
    // by join() and the destructor.
    void _join_threads();

    // Helper for entry points. Caller must hold mutex.
    void _throw_if_stopped(const char *method_name);

    // Helper for add_*() entry points: throws (without stopping the server)
    // if the server is stopped or already started. Caller must hold mutex.
    void _throw_unless_addable(const std::string &caller);


    // ----- Noncopyable, nonmoveable -----

    Hwtest(const Hwtest &) = delete;
    Hwtest &operator=(const Hwtest &) = delete;
    Hwtest(Hwtest &&) = delete;
    Hwtest &operator=(Hwtest &&) = delete;

    // Destructor calls stop(), then joins the worker threads inline (via
    // _join_threads). It deliberately does NOT call Hwtest::join(), which
    // rethrows the saved error -- fatal in a destructor.
    ~Hwtest();

protected:
    Hwtest(const std::string &server_name, bool use_hugepages);
};


// -------------------------------------------------------------------------------------------------
//
// HwtestSender


// HwtestSender: simulates a correlator by sending data over TCP connections
// to one or more receiver endpoints. Used for testing/benchmarking the receiver.
//
// This class is a "thread-backed class" (see notes/thread_backed_class.md) that
// spawns one worker thread per endpoint, each sending data as fast as possible
// (optionally rate-limited) over TCP.
//
// Usage:
//   auto fc = make_shared<HwtestSender>(send_bufsize, ...);
//   fc->add_endpoint("10.1.1.2", 1, 0.0, vcpu_list);
//   fc->start();   // creates worker threads and begins sending
//   // ... wait ...
//   fc->stop();    // signals threads to stop
//   fc->join();    // waits for threads to exit
//
// Worker threads are created in start(), not the constructor. Each worker thread:
//   - Opens TCP connection(s) to the endpoint on port 8787
//   - Sends data in a loop until stopped or connection is closed

struct HwtestSender
{
    // Timeout for send operations (milliseconds).
    static constexpr int send_timeout_ms = 10;

    HwtestSender(
        long send_bufsize = 64*1024,
        bool use_zerocopy = true,
        bool use_mmap = false,
        bool use_hugepages = true
    );

    // Destructor calls stop() and joins worker threads.
    ~HwtestSender();

    // Add an endpoint. Must be called before start().
    // total_gbps=0 means "no rate limit".
    void add_endpoint(const std::string &ip_addr, long num_tcp_connections, double total_gbps, const std::vector<int> &vcpu_list);

    // Create worker threads and begin sending data.
    // Entry point: throws if already started or stopped.
    void start();

    // Put HwtestSender into stopped state. Worker threads exit promptly.
    // If 'e' is non-null, it represents an error; otherwise normal termination.
    void stop(std::exception_ptr e = nullptr) const;

    // Block until all worker threads have exited.
    void join();

    // Wait until all workers have exited, or timeout expires.
    // Returns true if all workers exited, false on timeout.
    bool wait(int timeout_ms);

    // ----- Noncopyable, nonmoveable -----

    HwtestSender(const HwtestSender &) = delete;
    HwtestSender &operator=(const HwtestSender &) = delete;
    HwtestSender(HwtestSender &&) = delete;
    HwtestSender &operator=(HwtestSender &&) = delete;

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

    // Thread-backed class state (protected by 'mutex'). The stop-pattern
    // members are 'mutable' since stop() is const (see notes/stoppable_class.md).
    mutable std::mutex mutex;
    mutable std::condition_variable cv;
    mutable bool is_stopped = false;
    mutable std::exception_ptr error;

    bool is_started = false;
    long num_workers_exited = 0;

    // Worker threads (created in start()).
    std::vector<std::thread> workers;

    // Helper: check if stopped, with lock held by caller.
    void _throw_if_stopped(const char *method_name);

    // Helper: returns true if stop() has been called. (Acquires mutex.)
    bool _stopped();

    // Worker thread main function. Returns total bytes sent.
    long _worker_main(long endpoint_index);

    // Wrapper that catches exceptions and calls stop().
    void worker_main(long endpoint_index);

    // Helper: send all bytes from buffer, using short timeouts to allow prompt exit.
    // Returns false if stopped or connection reset.
    bool _send_all(Socket &sock, const void *buf, long nbytes);
};


}  // namespace pirate

#endif // _PIRATE_INTERNALS_HWTEST_HPP
