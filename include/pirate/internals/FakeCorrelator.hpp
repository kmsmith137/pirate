#ifndef _PIRATE_INTERNALS_FAKE_CORRELATOR_HPP
#define _PIRATE_INTERNALS_FAKE_CORRELATOR_HPP

#include <mutex>
#include <vector>
#include <string>

namespace pirate {
#if 0
}  // editor auto-indent
#endif


struct FakeCorrelator
{
    struct Params
    {
	long nconn_per_ipaddr = 0;      // tcp connections per ip address
	std::vector<std::string> ipaddr_list;   // e.g { "10.1.1.2", "10.1.2.2" }
	
	double gbps_per_ipaddr = 0.0;      // Zero means "no rate limit"    
	long send_bufsize = 512 * 1024;
	bool use_zerocopy = true;
	bool use_mmap = false;
	bool use_hugepages = true;
    };

    
    // High-level interface for running FakeCorrelator.
    //
    //   - creates FakeCorrelator (via shared_ptr)
    //   - creates sender threads which hold references to Correlator
    //   - joins sender threads (after FakeServer closes connections)
    //   - destroys FakeCorrelator.
    
    static void run(const Params &params);

    
    // Lower-level interface.
    
    FakeCorrelator(const Params &params);
    
    void abort(const std::string &msg);
    void throw_exception_if_aborted();
    void sender_main(const std::string &ipaddr);

        
    const Params params;
    
    bool aborted = false;
    std::string abort_msg;
    std::mutex lock;
};


}  // namespace pirate

#endif // _PIRATE_INTERNALS_FAKE_CORRELATOR_HPP
