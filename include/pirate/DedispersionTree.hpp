#ifndef _PIRATE_DEDISPERSION_TREE_HPP
#define _PIRATE_DEDISPERSION_TREE_HPP

#include "DedispersionConfig.hpp"
#include "FrequencySubbands.hpp"

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// DedispersionTree: a simple "data" class (only trivial member functions).
// Represents the output of the dedisperser, for one choice of (primary tree, early trigger).
//
// A vector of DedispersionTrees is created in the DedispersionPlan constructor, and gets copied
// into the dedisperser classes (GpuDedisperser, ReferenceDedisperser).

struct DedispersionTree
{
    int primary_tree_index = -1;   // Also identifies associated stage1 tree (input downsampled in time by 2^primary_tree_index).
    int early_trigger_level = -1;  // "Earliness" of trigger: 0 for the main tree, 1..num_early_triggers for early triggers.
    int amb_rank = 0;              // Ambient rank of this DedispersionTree (= dd_rank of associated stage1 tree)
    int dd_rank = 0;               // Active rank of this DedispersionTree (= amb_rank of stage1 tree, minus early_trigger_level)
    int nt_ds = 0;                 // Downsampled time samples per chunk (= config.time_samples_per_chunk / pow2(primary_tree_index))

    // Total tree rank. Equal to (config.toplevel_tree_rank - early_trigger_level - (primary_tree_index ? 1 : 0)).
    long total_rank() const { return amb_rank + dd_rank; }

    // Subbands searched in this tree.
    // Can differ from DedispersionConfig::frequency_subbands, due to early triggers and downsampling.
    FrequencySubbands frequency_subbands;

    // Contains members: num_early_triggers, max_width, {dm,time}_downsampling, wt_{dm,time}_downsampling.
    // Note that {dm,time}_downsampling can be 0 in the config, but are filled with nonzero values here.
    DedispersionConfig::PrimaryTree pf;

    // Internal time-downsampling ("core") factor of this tree's peak-finding kernel; sets
    // the time granularity of out_argmax tokens (see PeakFindingKernelParams::Dcore).
    // A property of the compiled cdd2 kernel (registry value), NOT derivable from the
    // config; equals pf.time_downsampling if the kernel is not compiled into this build.
    long Dcore = 0;

    // Number of time profiles used in peak-finder. (Equal to 1 + 3*log2(pf.max_width).)
    long nprofiles = 0;

    // For peak-finding array shapes.
    // 'wt' array shape is (beams_per_batch, ndm_wt, nt_wt, nprofiles, frequency_subbands.N).
    // 'out_max', 'out_argmax' shapes are (beams_per_batch, ndm_out, nt_out).
    long ndm_out = 0;
    long ndm_wt = 0;
    long nt_out = 0;
    long nt_wt = 0;

    // Currently, these informational members are just used in print-statements.
    double dm_min = 0.0;
    double dm_max = 0.0;
    double trigger_frequency = 0.0f;
};


}  // namespace pirate

#endif // _PIRATE_DEDISPERSION_TREE_HPP

