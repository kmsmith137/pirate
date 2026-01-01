#ifndef _PIRATE_RESOURCE_TRACKER_HPP
#define _PIRATE_RESOURCE_TRACKER_HPP

#include <string>
#include <unordered_map>
#include <yaml-cpp/yaml.h>

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// The ResourceTracker is a self-contained utility class that keeps track of resources
// (e.g. GPU global memory bandwidth, PCIe bandwidth) consumed by computational tasks.
//
// For fine-grained tracking, the ResourceTracker represents each resource by a
// dictionary (user-defined key) -> (integer count).

struct ResourceTracker
{
    using Dict = std::unordered_map<std::string, long>;

    // Bandwidth tracking.
    Dict gmem_bw_nbytes;       // GPU memory bandwidth, including PCIe transfers
    Dict hmem_bw_nbytes;       // Host memory bandwidth, including PCIe transfers
    Dict h2g_bw_nbytes;        // PCIe bandwidth, (host -> GPU) direction
    Dict g2h_bw_nbytes;        // PCIe bandwidth, (GPU -> host) direction
    Dict memcpy_h2g_calls;     // calls to cudaMemcpy() in (host -> GPU) direction
    Dict memcpy_g2h_calls;     // calls to cudaMemcpy() in (GPU -> host) direction
    Dict kernel_launches;      // user-defined kernels

    // Footprint tracking.
    Dict gmem_footprint_nbytes;  // total GPU memory consumption
    Dict hmem_footprint_nbytes;  // total host memory consumption

    // add_kernel() updates 'kernel_launches', 'gmem_bw_nbytes'
    // add_memcpy_h2g() updates 'memcpy_h2g_calls', 'h2g_bw_nbytes', 'gmem_bw_nbytes', 'hmem_bw_nbytes'.
    // add_memcpy_g2h() updates 'memcpy_g2h_calls', 'g2h_bw_nbytes', 'gmem_bw_nbytes', 'hmem_bw_nbytes'.
    void add_kernel(const std::string &key, long gmem_bw_nbytes);
    void add_memcpy_h2g(const std::string &key, long nbytes);  // no-op if nbytes==0
    void add_memcpy_g2h(const std::string &key, long nbytes);  // no-op if nbytes==0

    // These methods update a single Dict.
    // If align=true, then 'nbytes' is aligned to BumpAllocator::nalign.
    void add_gmem_bw(const std::string &key, long gmem_bw_nbytes);
    void add_hmem_bw(const std::string &key, long hmem_bw_nbytes);
    void add_gmem_footprint(const std::string &key, long gmem_footprint_nbytes, bool align);
    void add_hmem_footprint(const std::string &key, long hmem_footprint_nbytes, bool align);

    // If key is non-empty, returns value for that key (throws exception if not found).
    // If key is empty, returns sum over all keys.
    long get_gmem_bw(const std::string &key = "") const;
    long get_hmem_bw(const std::string &key = "") const;
    long get_h2g_bw(const std::string &key = "") const;
    long get_g2h_bw(const std::string &key = "") const;
    long get_gmem_footprint(const std::string &key = "") const;
    long get_hmem_footprint(const std::string &key = "") const;

    // Returns a copy of this ResourceTracker.
    ResourceTracker clone() const;

    // Used to accumulate "child" ResourceTrackers into their parent.
    ResourceTracker &operator+=(const ResourceTracker &);

    // to_yaml(): produces YAML of schematic form
    //
    // BW: XX GB/s  # for BW in [ gmem_bw, hmem_bw, h2g_bw, g2h_bw ]
    //   key1: XX GB/s
    //   key2: XX GB/s
    //     ...
    //
    // COUNTS: XX calls/s   # for COUNTS in [ memcpy_h2g, memcpy_g2h, kernel_launches ]
    //   key1: XX calls/s
    //     ...
    // 
    // FOOTPRINT: XX GiB  # for FOOTPRINT in [ gmem_footprint, hmem_footprint ]
    //   key1: XX GiB
    //     ...
    //
    // The 'multiplier' arg gives the conversion between integer rate counts in the ResourceTracker
    // and floating-point rates with units (sec^{-1}).
    //
    // "Top-level" numbers are sums over all keys, and the constituent key-value pairs
    // are printed iff fine_grained=true. Key-value pairs are sorted first by decreasing
    // values, and second alphabetically by key.

    void to_yaml(YAML::Emitter &emitter, double multiplier, bool fine_grained) const;
    std::string to_yaml_string(double multiplier, bool fine_grained) const;

    // Called internally by add_*() methods and operator+=(). No-op if value==0.
    // Throws an exception if key is empty.
    static void _update_dict(Dict &d, const std::string &key, long value);

    // Helper for get_*() methods. If key is empty, returns sum over all dict values.
    // If key is non-empty, returns value for that key (throws exception if not found).
    long _get_dict(const Dict &d, const std::string &key, const char *method_name) const;
};


}  // namespace pirate

#endif // _PIRATE_RESOURCE_TRACKER_HPP
