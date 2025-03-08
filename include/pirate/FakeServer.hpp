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
    struct Params
    {
	std::string server_name = "Server";
	long num_iterations = 20;
	bool use_hugepages = true;
	
	long nbytes_h2h = 0;  // memcpy host->host
	long nbytes_h2g = 0;  // memcpy host->gpu (per GPU)
	long nbytes_g2h = 0;  // memcpy host->gpu (per GPU)
	long memcpy_blocksize = 0;
	
	long nbytes_gmem_kernel = 0;  // use memory bandwidth on GPU, using a GPU kernel (not cudaMemcpyDeviceToDevice())
	long gmem_kernel_blocksize = 0;
	
	long nbytes_per_ssd = 0;
	long nthreads_per_ssd = 0;
	long nwrites_per_file = 0;
	long nbytes_per_file = 16L * 1024L * 1024L;
	std::vector<std::string> ssd_list;
	
	// We use one downsampling thread for each value of 'src_bit_depth'.
	long nbytes_downsample = 0;
	
	// Network parameters.
	long nconn_per_ipaddr = 0;    // tcp connections per ip address
	std::vector<std::string> ipaddr_list;      // e.g { "10.1.1.2", "10.1.2.2" }
	
	// If specified, include a SleepyWorker which sleeps for the specified number of seconds.
	// This is useful either if there are no workers (e.g. test_03_receive_data()), or if you want to rate-limit workers.
	long sleep_usec = 0;
	
	// Networking options. I'll probably add more later (e.g. zero-copy TCP)
	long recv_bufsize = 512 * 1024;
	bool use_epoll = true;
	
	// This parameter just determines how often the receiver threads synchronize counters.
	long network_sync_cadence = 16 * 1024 * 1024;  // bytes per ip address
	
	// If negative, all GPUs will be used.
	int ngpu = -1;
    };

    
    // High-level interface for running FakeServer.
    //
    //   - creates FakeServer (via shared_ptr)
    //   - creates announcer thread
    //   - creates receiver threads to receive TCP data
    //   - creates worker threads to perform tasks specified in Params
    //   - joins all threads
    //   - destroys FakeServer.
    
    static void run(const Params &params);

    
    // Lower-level interface follows.

    
    FakeServer(const Params &params);
    
    void increment_counter(int ix, int expected_value);
    int wait_for_counters(int threshold);
    int peek_at_counter();
    void abort(const std::string &msg);
    void receiver_main(int irecv);
    void worker_main(int iworker);
    void _show_all(bool show_stats=true);
    void announcer_main();

    
    Params params;

    // Defined in src_lib/FakeServer.cu
    struct Receiver;
    struct Worker;
    
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

    bool aborted = false;
    std::string abort_msg;

    // Set in announcer thread
    double total_time = 0.0;
    double total_gbps = 0.0;
};


}  // namespace pirate

#endif // _PIRATE_INTERNALS_FAKE_SERVER_HPP
