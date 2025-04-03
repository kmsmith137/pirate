#ifndef _PIRATE_INTERNALS_FAKE_CORRELATOR_HPP
#define _PIRATE_INTERNALS_FAKE_CORRELATOR_HPP

#include <mutex>
#include <vector>
#include <string>

#include <ksgpu/Barrier.hpp>

namespace pirate {
#if 0
}  // editor auto-indent
#endif


struct FakeCorrelator
{
    FakeCorrelator(
        long send_bufsize = 512 * 1024,
	bool use_zerocopy = true,
	bool use_mmap = false,
	bool use_hugepages = true
    );

    // total_gbps=0 means "no rate limit".
    void add_endpoint(const std::string &ip_addr, long num_tcp_connections, double total_gbps, const std::vector<int> &vcpu_list);
    
    void run();
    
    // Lower-level interface.

    struct Endpoint {
	std::string ip_addr;
	long num_tcp_connections;
	double total_gbps;
	std::vector<int> vcpu_list;
    };

    ksgpu::Barrier barrier;
    
    std::vector<Endpoint> endpoints;
    long send_bufsize;
    bool use_zerocopy;
    bool use_mmap;
    bool use_hugepages;
    
    void abort(const std::string &msg);
    void throw_exception_if_aborted();
    void sender_main(long endpoint_index);

    std::mutex lock;
    bool aborted = false;
    std::string abort_msg;
};


}  // namespace pirate

#endif // _PIRATE_INTERNALS_FAKE_CORRELATOR_HPP
