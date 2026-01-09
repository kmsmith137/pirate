#ifndef _PIRATE_SYSTEM_UTILS_HPP
#define _PIRATE_SYSTEM_UTILS_HPP

#include <vector>
#include <sys/mman.h>  // MCL_* flags


namespace pirate {
#if 0
}   // pacify editor auto-indent
#endif


extern void sys_mlockall(int flags = MCL_CURRENT | MCL_FUTURE | MCL_ONFAULT);

// FIXME usleep() is deprecated. In hindsight the I think the best interface is
//    void sys_sleep(double seconds);  // fills nanosleep() data structures "under the hood"
extern void sys_usleep(long usec);


// -------------------------------------------------------------------------------------------------
//
// set_thread_affinity(vcpu_list)
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
// If 'vcpu_list' is an empty vector, then set_thread_affinity() is a no-op.

extern void set_thread_affinity(const std::vector<int> &vcpu_list);

// get_thread_affinity()
//
// Returns the current thread's CPU affinity mask as a vector of vCPU indices.
// An empty vector means the thread can run on any CPU.

extern std::vector<int> get_thread_affinity();


} // namespace pirate

#endif  // _PIRATE_SYSTEM_UTILS_HPP
