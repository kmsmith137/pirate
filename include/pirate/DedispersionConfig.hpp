#ifndef _PIRATE_DEDISPERSION_CONFIG_HPP
#define _PIRATE_DEDISPERSION_CONFIG_HPP

#include <vector>
#include <string>
#include <iostream>

#include <ksgpu/Dtype.hpp>

namespace YAML { class Emitter; }      // #include <yaml-cpp/yaml.h>
namespace pirate { struct YamlFile; }  // #include <pirate/internals/YamlFile.hpp>


namespace pirate {
#if 0
}  // editor auto-indent
#endif


struct DedispersionConfig
{
    // Core dedispersion parameters.
    long tree_rank = -1;
    long num_downsampling_levels = -1;
    long time_samples_per_chunk = 0;

    // For now, there is only one dtype, which can be either float32 or float16.
    // Later, I might split this into "compute" and "ringbuf" dtypes, and allow compressed
    // dtypes (e.g. float8, int7).

    ksgpu::Dtype dtype;
    
    struct EarlyTrigger
    {
        long ds_level = -1;
        long tree_rank = 0;
    };

    // Sorted (by ds_level first, then tree_rank).
    std::vector<EarlyTrigger> early_triggers;

    // GPU configuration.
    long beams_per_gpu = 0;
    long beams_per_batch = 0;
    long num_active_batches = 0;

    // For testing: limit on-gpu ring buffers to (clag) <= (gpu_clag_maxfrac) * (max_clag)
    double gpu_clag_maxfrac = 1.0;   // set to 1 to disable (default)
    
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
    void add_early_trigger(long ds_level, long tree_rank);
    void add_early_triggers(long ds_level, std::initializer_list<long> tree_ranks);

    // Note: rather than calling this function directly, you probably want the
    // DedispersionPlan (not DedispersionConfig) member 'nelts_per_segment'.
    int get_nelts_per_segment() const;

    // make_random(): used for unit tests.
    static DedispersionConfig make_random(bool allow_early_triggers=true);
};

extern bool operator==(const DedispersionConfig::EarlyTrigger &x, const DedispersionConfig::EarlyTrigger &y);
extern bool operator>(const DedispersionConfig::EarlyTrigger &x, const DedispersionConfig::EarlyTrigger &y);
extern std::ostream &operator<<(std::ostream &os, const DedispersionConfig::EarlyTrigger &et);


}  // namespace pirate

#endif // _PIRATE_DEDISPERSION_CONFIG_HPP
