#ifndef _PIRATE_DEDISPERSION_CONFIG_HPP
#define _PIRATE_DEDISPERSION_CONFIG_HPP

#include <vector>
#include <string>
#include <iostream>

namespace YAML { class Emitter; }      // #include <yaml-cpp/yaml.h>
namespace pirate { struct YamlFile; }  // #include <pirate/internals/YamlFile.hpp>


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
    
    struct EarlyTrigger
    {
	ssize_t ds_level = -1;
	ssize_t tree_rank = 0;
    };

    // Sorted (by ds_level first, then tree_rank).
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
    
    int get_nelts_per_segment() const;             // returns constants::bytes_per_segment / sizeof(uncompressed_dtype)
    int get_uncompressed_dtype_size() const;       // returns sizeof(uncompressed_type)
    int get_bytes_per_compressed_segment() const;  // nontrivial (e.g. returns 66 if uncompressed=float16 and compressed=int8)
    // Note: no 'get_bytes_per_uncompressed_segment()' (use constants::bytes_per_segment).
    
    void validate() const;

    // Write in informal text format (e.g. for log files)
    // FIXME I might phase this out, in favor of yaml everywhere.
    void print(std::ostream &os = std::cout, int indent=0) const;

    // Write in YAML format.
    void to_yaml(YAML::Emitter &emitter) const;
    void to_yaml(const std::string &filename) const;    
    std::string to_yaml_string() const;

    // Construct from YAML file.
    // The 'verbosity' argument has the following meaning:
    //   0 = quiet
    //   1 = announce default values for all unspecified parameters
    //   2 = announce all parameters

    static DedispersionConfig from_yaml(const std::string &filename, int verbosity=0);
    static DedispersionConfig from_yaml(const YamlFile &file);
    
    // Helper functions for constructing DedispersionConfig instances.
    // Add early triggers, while maintaining invariant that 'early_triggers' is sorted.
    void add_early_trigger(ssize_t ds_level, ssize_t tree_rank);
    void add_early_triggers(ssize_t ds_level, std::initializer_list<ssize_t> tree_ranks);

    // make_random(): used for unit tests.
    //
    // If reference=true, then impose some constraints needed by ReferenceDedisperser:
    //   - uncompressed_dtype == compressed_dtype == "float32"
    //   - beams_per_gpu == beams_per_batch == num_active_batches == 1
    //   - force_ring_buffers_to_host = true
    //
    // (Eventually, I hope to relax these constrains and remove the 'reference' argument).
    
    static DedispersionConfig make_random(bool reference=false);
};

extern bool operator==(const DedispersionConfig::EarlyTrigger &x, const DedispersionConfig::EarlyTrigger &y);
extern bool operator>(const DedispersionConfig::EarlyTrigger &x, const DedispersionConfig::EarlyTrigger &y);
extern std::ostream &operator<<(std::ostream &os, const DedispersionConfig::EarlyTrigger &et);


}  // namespace pirate

#endif // _PIRATE_DEDISPERSION_CONFIG_HPP
