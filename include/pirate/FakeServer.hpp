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
    
    // The "network_sync_cadence" arg determines how often the receiver threads synchronize counters.
    void add_receiver(
        const std::string &ip_addr,      // e.g. "10.1.1.2"
	long num_tcp_connections,        // for this IP address
	long recv_bufsize,               // bytes (512 KiB is a good default)
	bool use_epoll,                  // recommend 'true'
	long network_sync_cadence,       // 16 MiB is a good default
	const std::vector<int> &vcpu_list
    );

    // The 'src_device' and 'dst_device' args can be (-1) for "host".
    // For a host->host copy, the memory bandwidth is (2 * nbytes_per_iteration).
    // Important note: it turns out that cudaMemcpy() runs slow for sizes >4GB (!!)
    // Empirically, any blocksize between 1MB and 4GB works pretty well.
    void add_memcpy_worker(int src_device, int dst_device, long nbytes_per_iteration, long blocksize, const std::vector<int> &vcpu_list);

    // GPU copy kernel: consumes memory bandwidth on GPU, with a kernel instead of cudaMemcpyDeviceToDevice().
    // This is sometimes useful if both "copy engines" are being used for PCIe transfers.
    // Suggest nbytes=100GB and blocksize=2GB.
    void add_gpu_copy_kernel(int device, long nbytes, long blocksize, const std::vector<int> &vcpu_list);
    
    // To get multiple threads per SSD, use multiple workers with different 'root_dir' args.
    void add_ssd_worker(const std::string &root_dir, long nfiles_per_iteration, long nbytes_per_file, long nbytes_per_write, const std::vector<int> &vcpu_list);

    void add_downsampling_worker(int src_bit_depth, long src_nelts, const std::vector<int> &vcpu_list);

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
