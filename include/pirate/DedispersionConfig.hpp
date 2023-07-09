#ifndef _PIRATE_DEDISPERSION_CONFIG_HPP
#define _PIRATE_DEDISPERSION_CONFIG_HPP

#include <vector>
#include <iostream>

namespace pirate {
#if 0
}  // editor auto-indent
#endif


struct DedispersionConfig
{
    // Core dedispersion parameters.
    ssize_t tree_rank = -1;
    ssize_t num_downsampling_levels = -1;
    ssize_t time_samples_per_chunk = 0;

    std::string uncompressed_dtype;  // "float32" or "float16"
    std::string compressed_dtype;    // "float32", "float16", or "int8"
    
    // Early triggers.
    struct EarlyTrigger
    {
	ssize_t ds_level = -1;
	ssize_t tree_rank = 0;
    };

    std::vector<EarlyTrigger> early_triggers;

    // GPU configuration.
    ssize_t beams_per_gpu = 0;
    ssize_t beams_per_batch = 0;
    ssize_t num_active_batches = 0;
    ssize_t gmem_nbytes_per_gpu = 0;

    // Intended for debugging!
    bool use_hugepages = true;
    bool force_ring_buffers_to_host = false;   // increases PCI-E bandwidth but simplifies logic
    bool bloat_dedispersion_plan = false;      // include extra data for debugging
    int planner_verbosity = 0;

    // Note: rather than calling these functions directly, you probably want the
    // DedispersionPlan (not DedispersionConfig) members 'nelts_per_segment',
    // 'uncompressed_dtype_size', and 'bytes_per_compressed_segment'.
    int get_nelts_per_segment() const;
    int get_uncompressed_dtype_size() const;
    int get_bytes_per_compressed_segment() const;
    // Note: no 'get_bytes_per_uncompressed_segment()' (use constants::bytes_per_segment).
    
    void validate() const;
    void print(std::ostream &os = std::cout, int indent=0) const;
    
    // Helper functions for constructing DedispersionConfig instances.
    void add_early_trigger(ssize_t ds_level, ssize_t tree_rank);
    void add_early_triggers(ssize_t ds_level, std::initializer_list<ssize_t> tree_ranks);

    // For unit tests.
    static DedispersionConfig make_random();
};

extern bool operator==(const DedispersionConfig::EarlyTrigger &x, const DedispersionConfig::EarlyTrigger &y);
extern bool operator>(const DedispersionConfig::EarlyTrigger &x, const DedispersionConfig::EarlyTrigger &y);
extern std::ostream &operator<<(std::ostream &os, const DedispersionConfig::EarlyTrigger &et);


}  // namespace pirate

#endif // _PIRATE_DEDISPERSION_CONFIG_HPP
