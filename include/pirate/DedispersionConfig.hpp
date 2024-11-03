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

    // For now, there is only one dtype, which can be either "float32" or "float16".
    // Later, I might split this into "compute" and "ringbuf" dtypes, and allow compressed
    // dtypes (e.g. float8, int7).
    
    std::string dtype;  // "float32" or "float16"
    
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

    // Note: rather than calling this function directly, you probably want the
    // DedispersionPlan (not DedispersionConfig) member 'nelts_per_segment'.
    int get_nelts_per_segment() const;

    // make_random(): used for unit tests.
    static DedispersionConfig make_random();
};

extern bool operator==(const DedispersionConfig::EarlyTrigger &x, const DedispersionConfig::EarlyTrigger &y);
extern bool operator>(const DedispersionConfig::EarlyTrigger &x, const DedispersionConfig::EarlyTrigger &y);
extern std::ostream &operator<<(std::ostream &os, const DedispersionConfig::EarlyTrigger &et);


}  // namespace pirate

#endif // _PIRATE_DEDISPERSION_CONFIG_HPP
