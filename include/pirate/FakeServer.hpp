#ifndef _PIRATE_INTERNALS_FAKE_SERVER_HPP
#define _PIRATE_INTERNALS_FAKE_SERVER_HPP

#include <mutex>
#include <vector>
#include <string>
#include <memory> // shared_ptr
#include <condition_variable>
#include <ksgpu/Barrier.hpp>


namespace pirate {
#if 0
}  // editor auto-indent
#endif


// Defined in include/pirate/DedispersionPlan.hpp
struct DedispersionPlan;


// Currently, we only expose the top-level FakeServer class in the .hpp file.
//
// In src_lib/FakeServer.cu, more helper classes are defined:
//
//    - Receiver
//    - Worker
//       - SleepyWorker
//       - DownsamplingWorker
//       - MemcpyWorker
//       - SsdWorker
//
// If needed, these helper classes could also be exposed in the .hpp file.


struct FakeServer
{
    FakeServer(const std::string &server_name, bool use_hugepages=true);
    
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

    // Uses memory bandwidth on GPU, but from a kernel instead of cudaMemcpyDeviceToDevice().
    // Note: the 'nbytes_per_iteration' argument is the memory bandwidth, not (memory_bw / 2)!
    void add_gmem_worker(int device, long nbytes_per_iteration, long blocksize, const std::vector<int> &vcpu_list);
    
    // To get multiple threads per SSD, use multiple workers with different 'root_dir' args.
    void add_ssd_worker(const std::string &root_dir, long nfiles_per_iteration, long nbytes_per_file, long nbytes_per_write, const std::vector<int> &vcpu_list);

    void add_downsampling_worker(int src_bit_depth, long src_nelts, const std::vector<int> &vcpu_list);

    // This is useful either for a network-only test with no workers, or if you want to rate-limit workers.
    void add_sleepy_worker(long sleep_usec);

    // After adding workers, this method runs the server.
    void run(long num_iterations);


    // -------------------------------------------------------------------------------------------------
    //
    // Lower-level interface follows.

    // Defined in src_lib/FakeServer.cu
    struct Receiver;
    struct Worker;

    void _add_worker(const std::shared_ptr<Worker> &worker, const std::string &caller);
    
    void increment_counter(int ix, int expected_value);
    int wait_for_counters(int threshold);
    int peek_at_counter();
    void abort(const std::string &msg);
    void receiver_main(int irecv, long num_iterations);
    void worker_main(int iworker, long num_iterations);
    void announcer_main(long num_iterations);
    void _show_all(bool show_vcpus, bool show_stats);

    std::string server_name;
    bool use_hugepages = false;
    
    std::vector<std::shared_ptr<Receiver>> receivers;
    std::vector<std::shared_ptr<Worker>> workers;

    // All threads (receivers + workers + announcer) wait at the barrier three times:
    //   - after initialization (e.g. allocating memory)
    //   - after all TCP connections have been accepted
    //   - after all iterations have finished.
    ksgpu::Barrier barrier;
    
    std::mutex lock;
    std::condition_variable cv;
    std::vector<int> counters;  // length num_workers
    int min_counter = 0;        // invariant: always equal to min(counters)

    bool running = false;
    bool aborted = false;
    std::string abort_msg;

    // Set in announcer thread
    double total_time = 0.0;
    double total_gbps = 0.0;
};


}  // namespace pirate

#endif // _PIRATE_INTERNALS_FAKE_SERVER_HPP
