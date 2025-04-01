#ifndef _PIRATE_INTERNALS_FAKE_SERVER_HPP
#define _PIRATE_INTERNALS_FAKE_SERVER_HPP

#include <mutex>
#include <vector>
#include <string>
#include <thread>
#include <memory> // shared_ptr


namespace pirate {
#if 0
}  // editor auto-indent
#endif


// Defined in include/pirate/DedispersionPlan.hpp
struct DedispersionPlan;


struct FakeServer
{
    FakeServer(const std::string &server_name="FakeServer", bool use_hugepages=true);

    // I've been using 512KB as a default 'recv_bufsize', but I haven't explored this systematically.
    void add_tcp_receiver(const std::string &ip_addr, long num_tcp_connections, long recv_bufsize, bool use_epoll, const std::vector<int> &vcpu_list);

    // Dedispersion on one GPU, using simplified CHIME parameters.
    void add_chime_dedisperser(int device, int beams_per_gpu, int num_active_batches, int beams_per_batch, bool use_copy_engine, const std::vector<int> &vcpu_list);
    
    // The 'src_device' and 'dst_device' args can be (-1) for "host".
    // Important note: it turns out that cudaMemcpy() runs slow for sizes >4GB (!!)
    // Empirically, any blocksize between 1MB and 4GB works pretty well.
    //
    // The 'use_copy_engine' argument is only meaningful for GPU->GPU copies.
    // If use_copy_engine=False, then we copy memory using a GPU kernel, rather than cudaMemcpyAsync().
    // This is useful in a situation where both GPU "compute engines" are being used for GPU->host and host->GPU transfers.
    void add_memcpy_thread(int src_device, int dst_device, long blocksize, bool use_copy_engine, const std::vector<int> &vcpu_list);
    
    // To get multiple threads per SSD, use multiple workers with different 'root_dir' args.
    void add_ssd_writer(const std::string &root_dir, long nbytes_per_file, const std::vector<int> &vcpu_list);

    // Runs the AVX2 downsampling kernel
    void add_downsampling_thread(int src_bit_depth, long src_nelts, const std::vector<int> &vcpu_list);

    // Called by python code, to control server.
    double show_stats();  // returns elapsed time in seconds
    void abort(const std::string &abort_msg);
    void join_threads();
    void start();
    void stop();

    // Defined in src_lib/FakeServer.cu
    struct State;    
    struct Stats;
    struct Worker;

    std::string server_name;
    std::shared_ptr<State> state;

    // After server is started, 'workers' is immutable after server is started.
    // Before server is started, 'workers' is protected by state->lock (kinda awkward but turns out to be simplest).
    std::vector<std::shared_ptr<Worker>> workers;

    std::vector<std::thread> threads;
    std::mutex thread_lock;  // protects 'threads'
    
    void _add_worker(const std::shared_ptr<Worker> &worker, const std::string &caller);
};


}  // namespace pirate

#endif // _PIRATE_INTERNALS_FAKE_SERVER_HPP
