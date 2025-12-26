#ifndef _PIRATE_DEDISPERSION_CONFIG_HPP
#define _PIRATE_DEDISPERSION_CONFIG_HPP

#include <vector>
#include <string>
#include <iostream>

#include <ksgpu/Dtype.hpp>
#include <ksgpu/Array.hpp>

namespace YAML { class Emitter; }      // #include <yaml-cpp/yaml.h>
namespace pirate { struct YamlFile; }  // #include <pirate/internals/YamlFile.hpp>


namespace pirate {
#if 0
}  // editor auto-indent
#endif


struct DedispersionConfig
{
    // Frequency channels. The observed frequency band is divided into "zones".
    // Within each zone, all frequency channels have the same width, but the 
    // channel width may differ between zones. For example:
    //
    //   zone_nfreq = {N}      zone_freq_edges={400,800}      one zone, channel width (400/N)
    //   zone_nfreq = {2*N,N}  zone_freq_edges={400,600,800}  width (100/N), (200/N) in lower/upper band

    std::vector<long> zone_nfreq;         // length (nzones)
    std::vector<double> zone_freq_edges;  // length (nzones+1), monotone increasing, in MHz.

    // Time sample length in milliseconds.
    float time_sample_ms = 0.0f;

    // Core dedispersion parameters.
    // The number of "tree" channels is ntree = 2^tree_rank.
    // The first tree searches to dispersion delay given by 2^tree_rank time samples.
    // Downsampled trees (0 < ids < num_downsampling_levels) downsample in time by 2^ids,
    // then search delay range 2^(tree_rank+ids-1) <= delay <= 2^(tree_rank+ids).

    long tree_rank = -1;
    long num_downsampling_levels = 0;
    long time_samples_per_chunk = 0;

    // For now, there is only one dtype, which can be either float32 or float16.
    // Later, I might split this into "compute" and "ringbuf" dtypes, and allow compressed
    // dtypes (e.g. float8, int7).

    ksgpu::Dtype dtype;

    // Defines frequency sub-bands for search. This can improve SNR for bursts that don't
    // span the full frequency range. For documentation, see FrequencySubbands.hpp.
    // To disable subbands (and search only the full band), set to {1}.

    std::vector<long> frequency_subband_counts;

    // Each downsampling level has its own PeakFindingConfig.
    // All members must be powers of two.
    //   max_width: max width of peak-finding kernel, in "tree" time samples
    //   {dm,time}_downsampling: downsampling factors of coarse-grained array, relative to tree
    //   wt_{dm,downsampling}: downsampling factors of weights array, relative to tree.

    struct PeakFindingConfig
    {
        long max_width = 0;             // required
        long dm_downsampling = 0;       // optional (default = "2^ceil(tree_rank/4)")
        long time_downsampling = 0;     // optional (default = "use value of dm_downsampling")
        long wt_dm_downsampling = 0;    // required (must be >= dm_downsampling)
        long wt_time_downsampling = 0;  // required (must be >= time_downsampling)
    };

    std::vector<PeakFindingConfig> peak_finding_params;  // length (num_downsampling_levels)

    // Early triggers: search a subset [fmid,fmax] of the full frequency range [flo,fhi]
    // at reduced latency. Each downsampling level has an independent set of early triggers.
    //
    // Early triggers are parameterized by EarlyTrigger::tree_rank, which must be less 
    // than the rank of the main (i.e. non-early) dedispersion tree. The rank of the main 
    // tree is (DedispersionConfig::tree_rank - S), where S=0 at ds_level=0, and S=1 for
    // ds_level > 0. (Detail: the downsampled trees have one lower rank because they search
    // a DM range which does not start at zero, see above.)
    //
    // Early triggers are optional (i.e. 'early_triggers' can be an empty vector).

    struct EarlyTrigger
    {
        long ds_level = -1;    // 0 <= ds_level < num_downsampling_levels
        long delta_rank = 0;   // must be > 0, specifies "early-ness" of trigger
    };

    // Sorted first by increasing ds_level, second by decreasing delta_rank.
    std::vector<EarlyTrigger> early_triggers;

    // GPU configuration.
    long beams_per_gpu = 0;
    long beams_per_batch = 0;
    long num_active_batches = 0;

    // For testing: limit on-gpu ring buffers to (clag) <= (gpu_clag_maxfrac) * (max_clag)
    // Set to 1.0 to disable (this is the default).
    double gpu_clag_maxfrac = 1.0;
    
    void validate() const;

    // Write in informal text format (e.g. for log files)
    // FIXME I might phase this out, in favor of yaml everywhere.
    void print(std::ostream &os = std::cout, int indent=0) const;

    // Write in YAML format.
    // If 'verbose' is true, include comments explaining the meaning of each field.
    void to_yaml(YAML::Emitter &emitter, bool verbose = false) const;
    void to_yaml(const std::string &filename, bool verbose = false) const;    
    std::string to_yaml_string(bool verbose = false) const;

    // Construct from YAML file.
    static DedispersionConfig from_yaml(const std::string &filename);
    static DedispersionConfig from_yaml(const YamlFile &file);
    
    // Helper functions for constructing DedispersionConfig instances.
    // Add early triggers, while maintaining invariant that 'early_triggers' is sorted.
    void add_early_trigger(long ds_level, long tree_rank);
    void add_early_triggers(long ds_level, std::initializer_list<long> tree_ranks);

    // Note: rather than calling this function directly, you probably want the
    // DedispersionPlan (not DedispersionConfig) member 'nelts_per_segment'.
    int get_nelts_per_segment() const;

    // Converts between frequency and (fractional) frequency channel.
    // Returns the fractional frequency channel corresponding to frequency f.
    // E.g. f=zone_freq_edges[i] corresponds to index = sum_{j<i} zone_nfreq[j].
    // Throws an exception if out-of-range (but allows a little roundoff error).
    // Uses linear search (not binary search) since the number of zones is assumed small.
    float frequency_to_index(float f) const;
    float index_to_frequency(float index) const;

    // Converts between frequency and "delay" (a scaled version of f^(-2)).
    // Delay is defined so that d=0 corresponds to f=fhi, and d=ntree corresponds to f=flo,
    // where flo=zone_freq_edges.front(), fhi=zone_freq_edges.back(), and ntree=2^tree_rank.
    // Valid delay range is [0, 2^tree_rank], valid frequency range is [flo, fhi].
    float delay_to_frequency(float delay) const;
    float frequency_to_delay(float f) const;

    // Returns the DM (in standard units, pc cm^{-3}) of an FRB whose dispersion delay
    // across the full band (zone_freq_edges.front() < f < zone_freq_edges.back()) is
    // equal to one time sample.
    double dm_per_unit_delay() const;

    // Returns sum of zone_nfreq (i.e. total number of frequency channels across all zones).
    long get_total_nfreq() const;

    // Returns channel_map array of length (2^tree_rank + 1), stored in CPU memory.
    // The channel_map defines the mapping between "tree" channels and frequency channels.
    // Given tree channel 0 <= n < ntree, the values (channel_map[n+1], channel_map[n])
    // define the edges of the tree channel in frequency space. (Note: channel_map is
    // monotonically decreasing, so channel_map[n+1] < channel_map[n].)
    ksgpu::Array<float> make_channel_map() const;

    // Test that frequency_to_index/index_to_frequency and delay_to_frequency/frequency_to_delay
    // are inverses of each other, by sampling random values and checking endpoints.
    // Called by 'python -m pirate_frb test --dd' -> GpuDedisperser::test() -> DedispersionConfig::test().
    // Also called by 'python -m pirate_frb show_dedisperser ...'.
    void test() const;

    // make_random(): used for unit tests.

    struct RandomArgs 
    {
        int max_rank = 10;
        int max_early_triggers = 5;  // set to zero to disable early triggers
        bool gpu_valid = true;
        bool verbose = false;
    };
    
    static DedispersionConfig make_random(const RandomArgs &args);
    static DedispersionConfig make_random() { return make_random(RandomArgs()); }
};

extern bool operator==(const DedispersionConfig::EarlyTrigger &x, const DedispersionConfig::EarlyTrigger &y);
extern bool operator>(const DedispersionConfig::EarlyTrigger &x, const DedispersionConfig::EarlyTrigger &y);
extern std::ostream &operator<<(std::ostream &os, const DedispersionConfig::EarlyTrigger &et);


}  // namespace pirate

#endif // _PIRATE_DEDISPERSION_CONFIG_HPP
