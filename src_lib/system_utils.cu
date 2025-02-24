#include "../include/pirate/internals/system_utils.hpp"

#include <thread>
#include <string>
#include <sstream>
#include <cassert>
#include <stdexcept>

#include <errno.h>
#include <sched.h>
#include <unistd.h>

// #include <thread>
// #include <cassert>
// #include <sstream>
// #include <stdexcept>
// #include <pthread.h>
// #include <sys/stat.h>

using namespace std;


namespace pirate {
#if 0
};  // pacify emacs c-mode!
#endif


void sys_mlockall(int flags)
{
    // FIXME low-priority things to add:
    //   - test that 'flags' is a subset of (MCL_CURRENT | MCL_FUTURE | MCL_ONFAULT)
    //   - if mlockall() fails, then exception text should pretty-print flags
    
    int err = mlockall(flags);

    if (err < 0)
	throw runtime_error(string("mlockall: ") + strerror(errno));
}


void sys_usleep(ssize_t usec)
{
    // According to usleep() manpage, sleeping for longer than this is an error!
    static constexpr ssize_t max_usleep = 1000000;
	
    assert(usec >= 0);

    while (usec > 0) {
	ssize_t n = std::min(usec, max_usleep);
	usec -= n;
	
	int err = usleep(n);

	if (err < 0)
	    throw runtime_error(string("usleep: ") + strerror(errno));
    }
}


// -------------------------------------------------------------------------------------------------
//
// pin_thread_to_vcpus(vcpu_list)
//
// The 'vcpu_list' argument is a list of integer vCPUs, where I'm defining a vCPU
// to be the scheduling unit in pthread_setaffinity_np() or sched_setaffinity().
//
// If hyperthreading is disabled, then there should be one vCPU per core.
// If hyperthreading is enabled, then there should be two vCPUs per core
// (empirically, always with vCPU indices 2*n and 2*n+1?)
//
// I think that the number of vCPUs and their location in the NUMA hierarchy
// always follows the output of 'lscpu -ae', but AFAIK this isn't stated anywhere.
//
// If 'vcpu_list' is an empty vector, then pin_thread_to_vcpus() is a no-op.


void pin_thread_to_vcpus(const vector<int> &vcpu_list)
{
    if (vcpu_list.size() == 0)
	return;

    // I wanted to argument-check 'vcpu_list', by comparing with the number of VCPUs available.
    //
    // FIXME Calling std::thread::hardware_concurrency() doesn't seem quite right, but doing the "right"
    // thing seems nontrivial. According to 'man sched_setaffinity()':
    //
    //     "There are various ways of determining the number of CPUs available on the system, including:
    //      inspecting the contents of /proc/cpuinfo; using sysconf(3) to obtain the values of the
    //      _SC_NPROCESSORS_CONF and _SC_NPROCESSORS_ONLN parameters; and inspecting the list of CPU
    //      directories under /sys/devices/system/cpu/."
    
    int num_vcpus = std::thread::hardware_concurrency();

    cpu_set_t cs;
    CPU_ZERO(&cs);

    for (int vcpu: vcpu_list) {
	if ((vcpu < 0) || (vcpu >= num_vcpus)) {
	    stringstream ss;
	    ss << "pirate::pin_thread_to_vcpus(): vcpu=" << vcpu
	       << " is out of range (num_vcpus=" << num_vcpus <<  to_string(num_vcpus) + ")";
	    throw runtime_error(ss.str());
	}
	CPU_SET(vcpu, &cs);
    }

    // Note: pthread_self() always succeeds, no need to check its return value.
    int err = pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs);

    if (err != 0) {
	// If pthread_setaffinity_np() fails, then according to its manpage,
	// it returns an error code, rather than setting 'errno'.
	throw runtime_error(string("pthread_setaffinity_np(): ") + strerror(err));
    }
}


}  // namespace pirate
