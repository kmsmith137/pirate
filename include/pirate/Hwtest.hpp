#ifndef _PIRATE_INTERNALS_HWTEST_HPP
#define _PIRATE_INTERNALS_HWTEST_HPP

#include "Barrier.hpp"

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


// Hwtest: hardware performance tests (memory bandwidth, network bandwidth, etc.)
// Thread-backed class (see notes/thread_backed_class.md).

struct Hwtest : public std::enable_shared_from_this<Hwtest>
{
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

    // Destructor calls stop() and join().
    ~Hwtest();
    
protected:
    Hwtest(const std::string &server_name, bool use_hugepages);
};


}  // namespace pirate

#endif // _PIRATE_INTERNALS_HWTEST_HPP
