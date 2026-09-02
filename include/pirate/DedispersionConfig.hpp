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
    double time_sample_ms = 0.0f;

    // Core dedispersion parameters.
    long toplevel_tree_rank = -1;      // rank of "toplevel" tree (non-downsampled, no early trigger)
    long time_samples_per_chunk = 0;

    // For now, there is only one dtype, which can be either float32 or float16.
    // Later, I might split this into "compute" and "ringbuf" dtypes, and allow compressed
    // dtypes (e.g. float8, int7).

    ksgpu::Dtype dtype;

    // Defines frequency sub-bands for search. This can improve SNR for bursts that don't
    // span the full frequency range. For documentation, see FrequencySubbands.hpp.
    // To disable subbands (and search only the full band), set to {1}.
    //
    // Note: these are the 'top-level' frequency subbands; fewer subbands may be searched in 
    // individual trees. To see which subbands are searched in which trees, use the command
    // 'python -m pirate_frb show dedisperser --verbose <config.yml>'.

    std::vector<long> frequency_subband_counts;

    // Each "primary tree" searches a different DM range, ordered from low to high
    // (primary tree p downsamples the input in time by 2^p, see 'toplevel_tree_rank' above).
    // Each primary tree is expanded into (num_early_triggers+1) "dedispersion trees".
    // See the dedispersion tex notes for more info.
    //
    // The remaining members configure peak-finding, and must be powers of two:
    //   max_width: max width of peak-finding kernel, in "tree" time samples
    //   wt_{dm,time}_downsampling: downsampling factors of weights array, relative to tree.

    struct PrimaryTree
    {
        // Member ORDER is load-bearing: operator<<(ostream&, const PrimaryTree&) emits an
        // aggregate-initializer brace list in this order, which DedispersionConfig::emit_cpp()
        // pastes into generated C++. Reordering silently misassigns those fields.

        long num_early_triggers = 0;    // required (can be zero)
        long max_width = 0;             // required

        // Both must be >= DedispersionTree::{dm,time}_downsampling of every tree in this
        // family, i.e. >= pow2(dd_rank1) of its early_trigger_level=0 tree. Checked in
        // validate(), and again in the DedispersionPlan constructor where the tree's own
        // values are known.
        long wt_dm_downsampling = 0;    // required
        long wt_time_downsampling = 0;  // required
    };

    std::vector<PrimaryTree> primary_trees;  // one entry per DM range searched

    // Number of primary trees. (Not a member -- inferred from 'primary_trees'.)
    long num_primary_trees() const { return primary_trees.size(); }

    // The DedispersionTree enumeration. Trees are ordered by primary tree, then by
    // DECREASING early-trigger level (earliest trigger first, then the main
    // early_trigger_level=0 tree), so 'itree' is a prefix sum over
    // primary_trees[].num_early_triggers. These three are the size, the forward map and the
    // inverse map of that enumeration. They are how a caller with no plan in hand names a
    // tree, and the DedispersionPlan constructor asserts its own loop against them.

    // Number of DedispersionTrees, i.e. sum over primary trees of (num_early_triggers+1).
    // Equal to DedispersionPlan::ntrees.
    long num_dedispersion_trees() const;

    // itree -> (primary_tree_index, early_trigger_level). Throws if itree is out of range.
    void locate_dedispersion_tree(long itree, long *primary_tree_index,
                                  long *early_trigger_level) const;

    // (primary_tree_index, early_trigger_level) -> itree. The inverse of
    // locate_dedispersion_tree(); throws if either argument is out of range.
    long dedispersion_tree_index(long primary_tree_index, long early_trigger_level) const;

    // GPU configuration.
    long beams_per_gpu = 0;
    long beams_per_batch = 0;
    long num_active_batches = 0;

    // Bound on how far a WriteFiles RPC may extend past the current
    // processing threshold, in time samples; the excess is silently
    // truncated. Rounded up to a whole number of chunks by the FrbServer.
    // 0 = WriteFiles requests never extend into the future (but note that
    // StartStream requests can always extend arbitrarily far).
    long future_write_max_samples = 0;

    // For testing: limit on-gpu ring buffers to (clag) <= max_gpu_clag.
    // Set to 10000 to disable (this is the default).
    long max_gpu_clag = 10000;
    
    void validate() const;

    // Returns a deep copy of this config. Useful when a caller wants to modify a
    // config (e.g. override the beam geometry) without mutating the caller's original.
    DedispersionConfig clone() const { return *this; }

    // Write in YAML format.
    // If 'verbose' is true, include comments explaining the meaning of each field.
    void to_yaml(YAML::Emitter &emitter, bool verbose = false) const; 
    std::string to_yaml_string(bool verbose = false) const;

    // Construct from YAML file.
    static DedispersionConfig from_yaml(const std::string &filename);
    static DedispersionConfig from_yaml(const YamlFile &file);

    // Construct from a YAML string, i.e. the inverse of to_yaml_string(). Needed wherever a
    // config travels as a string rather than a file -- over the wire (see the grouper
    // handshake in FrbGrouper.cpp), or embedded in a container format
    // (see pirate_frb.varmap.asdf_io).
    static DedispersionConfig from_yaml_string(const std::string &yaml_string);

    // Note: rather than calling this function directly, you probably want the
    // DedispersionPlan (not DedispersionConfig) member 'nelts_per_segment'.
    int get_nelts_per_segment() const;

    // Converts between frequency and (fractional) frequency channel.
    // Returns the fractional frequency channel corresponding to frequency freq.
    // E.g. freq=zone_freq_edges[i] corresponds to index = sum_{j<i} zone_nfreq[j].
    // Throws an exception if out-of-range (but allows a little roundoff error).
    // Uses linear search (not binary search) since the number of zones is assumed small.
    double frequency_to_index(double freq) const;
    double index_to_frequency(double index) const;

    // Converts between frequency and "delay" (a scaled version of freq^(-2)).
    // Delay is defined so that d=0 corresponds to freq=freq_hi, and d=ntree corresponds to freq=freq_lo,
    // where freq_lo=zone_freq_edges.front(), freq_hi=zone_freq_edges.back(), and ntree=2^toplevel_tree_rank.
    // Valid delay range is [0, 2^toplevel_tree_rank], valid frequency range is [freq_lo, freq_hi].
    double delay_to_frequency(double delay) const;
    double frequency_to_delay(double freq) const;

    // Returns the DM (in standard units, pc cm^{-3}) of an FRB whose dispersion delay
    // across the full band (zone_freq_edges.front() < freq < zone_freq_edges.back()) is
    // equal to one time sample.
    double dm_per_unit_delay() const;

    // Returns the largest DM (pc cm^{-3}) searched by any dedispersion tree. Mirrors the
    // per-tree dm_max = dm_per_unit_delay() * 2^toplevel_tree_rank * 2^p computed in the
    // DedispersionPlan constructor; this is monotonic in the primary tree index p and
    // independent of early_trigger_level, so the maximum is at p = num_primary_trees()-1.
    // (Depends only on pre-metadata config fields, so it is valid on config_prefilled.)
    double max_dm_of_all_trees() const;

    // Returns the peak-finding kernel max_width of the base (non-downsampled, p=0)
    // tree, in time samples. At p=0 the tree's time sampling equals the native
    // (frame) time sampling, so this is a number of frame time samples (NOT milliseconds).
    long max_width_of_base_tree() const;

    // Returns sum of zone_nfreq (i.e. total number of frequency channels across all zones).
    long get_total_nfreq() const;

    // Returns channel_map array of length (2^toplevel_tree_rank + 1), stored in CPU memory.
    // The channel_map defines the mapping between "tree" channels and frequency channels.
    // Given tree channel 0 <= n < ntree, the values (channel_map[n+1], channel_map[n])
    // define the edges of the tree channel in frequency space. (Note: channel_map is
    // monotonically decreasing, so channel_map[n+1] < channel_map[n].)
    //
    // NOTE: we use double precision, since weights are computed by differencing
    // (channel_map[i+1] - channel_map[i]), which loses a lot of relative precision.

    ksgpu::Array<double> make_channel_map() const;

    // make_random_freq_variances(): for testing/debugging (its caller is
    // pirate_frb/fast_varmap/test_fast_varmap.py). Assigns one random variance in [0,1] to
    // each frequency zone, and returns a length-nfreq array of per-channel variances
    // (constant within each zone).
    // If 'noisy' is true, prints the length-nzones per-zone array.
    ksgpu::Array<double> make_random_freq_variances(bool noisy=false) const;

    // Test that frequency_to_index/index_to_frequency and delay_to_frequency/frequency_to_delay
    // are inverses of each other, by sampling random values and checking endpoints.
    // Called by 'python -m pirate_frb test --dd' (special iteration-0 logic in __main__.py).
    // Also called by 'python -m pirate_frb show dedisperser ...'.
    void test() const;

    // Emit C++ code to initialize this DedispersionConfig.
    // (Sometimes convenient in unit tests.)
    void emit_cpp(std::ostream &os=std::cout, const char *name="config", int indent=4) const;

    // make_random(): used for unit tests.

    struct RandomArgs
    {
        int max_toplevel_rank = 10;  // bounds toplevel_tree_rank
        int max_early_triggers = 5;  // set to zero to disable early triggers
        bool gpu_valid = true;
        bool verbose = false;
        bool force_float32 = false;
        bool no_host_mega_ringbuf = false;   // MegaRingbuf gpu-only, no host<->gpu copies
    };
    
    static DedispersionConfig make_random(const RandomArgs &args);
    static DedispersionConfig make_random() { return make_random(RandomArgs()); }

    // make_mini_chord(): returns a "throwaway" CHORD-like DedispersionConfig.
    // Useful for testing and timing kernels that need a valid config.
    static DedispersionConfig make_mini_chord(ksgpu::Dtype dtype);
};


}  // namespace pirate

#endif // _PIRATE_DEDISPERSION_CONFIG_HPP
