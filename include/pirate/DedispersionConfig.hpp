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

        // Lower bound on num_primary_trees(), in 1..constants::max_primary_trees. Honoured BY
        // CONSTRUCTION -- the base cdd2 key and toplevel_tree_rank are drawn only from those
        // that admit this many primary trees -- so a caller who needs a multi-tree config gets
        // one on every call, instead of rejection-sampling at the ~35-50% rate the
        // unconstrained draw gives. To get an EXACT tree count, ask for it as a minimum and
        // truncate 'primary_trees': validate()'s two dependences on the count both weaken
        // under truncation (see the comments at its 'min_nt' and 'min_dd_rank1' checks).
        int min_primary_trees = 1;

        // If true, beams_per_gpu = beams_per_batch = num_active_batches = 1, and the whole
        // (8192 / nt_divisor) budget goes to time_samples_per_chunk.
        //
        // FOR CALLERS WHO WOULD OTHERWISE OVERWRITE THE BEAM FIELDS AFTER THE DRAW. Doing that
        // by hand has a trap -- overwrite only beams_per_gpu and beams_per_batch, and
        // num_active_batches keeps a value validate() then rejects on 55% of draws -- and it
        // leaves time_samples_per_chunk conditioned on a beam draw that no longer exists: one
        // factor of a three-way bounded product, so its median is ~n^(1/3) rather than ~n/2.
        //
        // BUT NOT FOR THE VARIANCE-MAP SWEEP TESTS, which is worth saying because they are the
        // obvious candidates and the choice was measured. They overwrite the beam fields, and
        // moving them to this flag raises their median time_samples_per_chunk from 256 to
        // ~4000 -- which collapses the number of CHUNKS the sweep streams (P(ndata_chunks == 1)
        // goes from 0.28 to 0.89), and multiplies the predicted sweep work by 4x to 15x. Chunk
        // boundaries are most of what those tests exist to check, so for them the short chunk
        // is the useful corner and the three-way split is doing them a favour by accident.
        bool single_beam = false;

        // The next three exist so that a caller with a hard requirement can ASK FOR IT, rather
        // than drawing configs and discarding the ones that miss. That is not just a saving on
        // draws: all three correlate strongly with num_primary_trees and dtype (both feed
        // nt_divisor below), so filtering afterwards silently reshapes the population under
        // test. Measured on the loopback tests before this existed, requiring
        // time_samples_per_chunk % 256 == 0 together with beams_per_gpu <= 8 took
        // P(one primary tree) from 61% to 16% and P(toplevel_tree_rank == 5) from 49% to 13%.

        // Draw time_samples_per_chunk as a multiple of this. 1 = no constraint. Used by the
        // loopback tests, whose network protocol sends in units of 256 time samples.
        long tspc_multiple = 1;

        // Upper bound on beams_per_gpu. 0 = no bound. When set, the beam geometry is drawn
        // FIRST, on its own scale, and the chunk length takes what is left of the budget.
        long max_beams_per_gpu = 0;

        // Reserve this many of the GPU's batch slots per ACTIVE batch, i.e. guarantee
        //
        //    beams_per_gpu >= min_batch_slots * num_active_batches * beams_per_batch.
        //
        // 1 (the default) is what make_random() gives anyway. 2 is what a grouper-enabled
        // FrbServer needs: it builds its dedisperser with nbatches_out = 2*num_active_batches,
        // and FrbGrouper requires that output ring to fit within one chunk.
        int min_batch_slots = 1;

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
