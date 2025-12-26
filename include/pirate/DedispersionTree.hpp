#ifndef _PIRATE_DEDISPERSION_TREE_HPP
#define _PIRATE_DEDISPERSION_TREE_HPP

#include "DedispersionConfig.hpp"
#include "FrequencySubbands.hpp"

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// DedispersionTree: a simple "data" class with no member functions.
// Represents the output of the dedisperser, for one choice of (downsampling level, early trigger).
//
// A vector of DedispersionTrees is created in the DedispersionPlan constructor, and gets copied
// into the dedisperser classes (GpuDedisperser, ReferenceDedisperser).

struct DedispersionTree
{
    // Note: for most purposes, you want 'early_dd_rank', not 'pri_dd_rank'.
    int ds_level = -1;       // Downsampling level, also identifies associated stage1 tree.
    int amb_rank = 0;        // Ambient rank of DedispersionTree (= dd_rank of associated stage1 tree)
    int pri_dd_rank = 0;     // Active rank of primary DedispersionTree (= amb_rank of associated stage1 tree)
    int early_dd_rank = 0;   // Active rank of this DedispersionTree (always <= pri_dd_rank)
    int nt_ds = 0;           // Downsampled time samples per chunk (= config.time_samples_per_chunk / pow2(ds_level))

    // Subbands searched in this tree.
    // Can differ from DedispersionConfig::frequency_subbands, due to early triggers and downsampling.
    FrequencySubbands frequency_subbands;
    
    // Contains members: max_width, {dm,time}_downsampling, wt_{dm,time}_downsampling.
    // Note that {dm,time}_downsampling can be 0 in the config, but are filled with nonzero values here.
    DedispersionConfig::PeakFindingConfig pf;

    // Number of time profiles used in peak-finder. (Equal to 1 + 3*log2(pf.max_width).)
    long nprofiles = 0;

    // For peak-finding array shapes.
    // 'wt' array shape is (beams_per_batch, ndm_wt, nt_wt, nprofiles, frequency_subbands.F).
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

