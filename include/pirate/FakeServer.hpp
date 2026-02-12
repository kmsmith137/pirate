#ifndef _PIRATE_INTERNALS_FAKE_SERVER_HPP
#define _PIRATE_INTERNALS_FAKE_SERVER_HPP

#include <condition_variable>
#include <exception>
#include <mutex>
#include <vector>
#include <string>
#include <thread>
#include <memory> // shared_ptr

#include <ksgpu/Barrier.hpp>


namespace pirate {
#if 0
}  // editor auto-indent
#endif


// Defined in include/pirate/DedispersionPlan.hpp
struct DedispersionPlan;


// FakeServer: benchmarking tool that runs multiple worker threads for testing
// network receive, GPU memcpy, SSD writes, AVX2 kernels, and dedispersion.
//
// This class is a "thread-backed class" (see notes/thread_backed_class.md) that
// spawns one thread per worker. Workers are added before start() with the add_*()
// methods, then started with start().
//
// Usage:
//   FakeServer server("Node test");
//   server.add_tcp_receiver(...);
//   server.add_memcpy_thread(...);
//   server.start();
//   // ... periodically call show_stats() ...
//   server.stop();
//   server.join();

struct FakeServer
{
    // Timeout for blocking calls in worker threads (milliseconds).
    static constexpr int worker_timeout_ms = 10;

    FakeServer(const std::string &server_name="FakeServer", bool use_hugepages=true);

    // Destructor calls stop() and joins worker threads.
    ~FakeServer();

    // I've been using 512KB as a default 'recv_bufsize', but I haven't explored this systematically.
    void add_tcp_receiver(const std::string &ip_addr, long num_tcp_connections, long recv_bufsize, bool use_epoll, const std::vector<int> &vcpu_list, int cpu, int inic);

    // Dedispersion on one GPU, using simplified CHIME parameters.
    void add_chime_dedisperser(int device, int beams_per_gpu, int num_active_batches, int beams_per_batch, bool use_copy_engine, const std::vector<int> &vcpu_list, int cpu);

    // The 'src_device' and 'dst_device' args can be (-1) for "host".
    // Important note: it turns out that cudaMemcpy() runs slow for sizes >4GB (!!)
    // Empirically, any blocksize between 1MB and 4GB works pretty well.
    //
    // The 'use_copy_engine' argument is only meaningful for GPU->GPU copies.
    // If use_copy_engine=False, then we copy memory using a GPU kernel, rather than cudaMemcpyAsync().
    // This is useful in a situation where both GPU "compute engines" are being used for GPU->host and host->GPU transfers.
    void add_memcpy_thread(int src_device, int dst_device, long blocksize, bool use_copy_engine, const std::vector<int> &vcpu_list, int cpu);

    // To get multiple threads per SSD, use multiple workers with different 'root_dir' args.
    void add_ssd_writer(const std::string &root_dir, long nbytes_per_file, const std::vector<int> &vcpu_list, int cpu, int issd);

    // Runs the AVX2 downsampling kernel
    void add_downsampling_thread(int src_bit_depth, long src_nelts, const std::vector<int> &vcpu_list, int cpu);

    // Create worker threads and begin running.
    // Entry point: throws if already started or stopped.
    void start();

    // Put FakeServer into stopped state. Worker threads exit promptly.
    // If 'e' is non-null, it represents an error; otherwise normal termination.
    void stop(std::exception_ptr e = nullptr);

    // Block until all worker threads have exited.
    void join();

    // Wait until all workers have exited, or timeout expires.
    // Returns true if all workers exited, false on timeout.
    bool wait(int timeout_ms);

    // Show bandwidth stats. Returns elapsed time in seconds.
    // Entry point: rethrows worker errors.
    double show_stats();

    // ----- Noncopyable, nonmoveable -----

    FakeServer(const FakeServer &) = delete;
    FakeServer &operator=(const FakeServer &) = delete;
    FakeServer(FakeServer &&) = delete;
    FakeServer &operator=(FakeServer &&) = delete;

    // ----- Internal types (defined in FakeServer.cpp) -----

    struct Stats;
    struct Worker;

private:
    std::string server_name;
    bool use_hugepages;

    // Thread-backed class state (protected by 'mutex').
    std::mutex mutex;
    std::condition_variable cv;
    bool is_stopped = false;
    bool is_started = false;
    std::exception_ptr error;
    long num_workers_exited = 0;

    // Worker-to-worker synchronization (initialized in start()).
    ksgpu::Barrier barrier;

    // After start(), 'workers' is immutable.
    // Before start(), 'workers' is protected by 'mutex'.
    std::vector<std::shared_ptr<Worker>> workers;

    std::vector<std::thread> threads;

    // Helper: check if stopped, with lock held by caller.
    void _throw_if_stopped(const char *method_name);

    void _add_worker(const std::shared_ptr<Worker> &worker, const std::string &caller);

    // Worker thread entry point (static so it has access to private members).
    static void _worker_thread_main(FakeServer *server, std::shared_ptr<Worker> worker);
};


}  // namespace pirate

#endif // _PIRATE_INTERNALS_FAKE_SERVER_HPP
