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


struct Hwtest : public std::enable_shared_from_this<Hwtest>
{
    // Factory method (constructor is protected).
    static std::shared_ptr<Hwtest> create(const std::string &server_name="Hwtest", bool use_hugepages=true);

    // I've been using 512KB as a default 'recv_bufsize', but I haven't explored this systematically.
    void add_tcp_receiver(const std::string &ip_addr, long num_tcp_connections, long recv_bufsize, bool use_epoll, const std::vector<int> &vcpu_list, int cpu, int inic);

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
    void add_ssd_writer(const std::string &root_dir, long nbytes_per_file, const std::vector<int> &vcpu_list, int cpu, int issd);

    // Runs the AVX2 downsampling kernel
    void add_downsampling_thread(int src_bit_depth, long src_nelts, const std::vector<int> &vcpu_list, int cpu);

    // Called by python code, to control server.
    double show_stats();  // returns elapsed time in seconds

    void join();
    void start();
    void stop(std::exception_ptr e = nullptr);

    // Defined in src_lib/Hwtest.cpp
    struct Stats;
    struct Worker;

    std::string server_name;
    bool is_started = false;
    bool is_stopped = false;
    std::exception_ptr error;
    std::mutex mutex;
    bool use_hugepages;
    Barrier barrier;

    // After server is started, 'workers' is immutable.
    // Before server is started, 'workers' is protected by mutex.
    std::vector<std::shared_ptr<Worker>> workers;

    std::vector<std::thread> threads;

    void _add_worker(const std::shared_ptr<Worker> &worker, const std::string &caller);

    // Helper for entry points. Caller must hold mutex.
    void _throw_if_stopped(const char *method_name);

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
