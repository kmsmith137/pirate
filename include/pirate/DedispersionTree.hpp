#ifndef _PIRATE_DEDISPERSION_TREE_HPP
#define _PIRATE_DEDISPERSION_TREE_HPP

#include "DedispersionConfig.hpp"
#include "FrequencySubbands.hpp"

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// DedispersionTree: a "dataclass" representing the output of the dedisperser, for one
// choice of (primary tree, early trigger).
//
// Trees are created in the DedispersionPlan constructor, which is the only place the geometry
// is implemented, and get copied out of DedispersionPlan::trees by the dedisperser classes
// (GpuDedisperser, ReferenceDedisperser). Everything that INTERPRETS a tree -- decoding
// out_argmax tokens, the subband index mappings, yaml I/O -- is a DedispersionPlan method.

struct DedispersionTree
{
    // Each DedispersionTree corresponds to an index pair (primary_tree_index, early_trigger_level).
    long primary_tree_index = -1;   // Input is downsampled in time by 2^primary_tree_index.
    long early_trigger_level = -1;  // "Earliness" of trigger: 0 for the main tree, 1..num_early_triggers for early triggers.

    // PrimaryTree contains members: num_early_triggers, max_width, wt_{dm,time}_downsampling.
    DedispersionConfig::PrimaryTree primary_tree;

    // Subbands searched in this tree (can differ from DedispersionConfig::frequency_subbands, due to early triggers).
    FrequencySubbands frequency_subbands;

    // Tree geometry. (Some of these are redundant, and not written to yaml.)
    // Note: tree_rank == (amb_rank + dd_rank).
    long toplevel_tree_rank = 0;    // = DedispersionConfig::toplevel_tree_rank.
    long primary_tree_rank = 0;     // = toplevel_tree_rank - (primary_tree_index ? 1 : 0)
    long tree_rank = 0;             // = primary_tree_rank - early_trigger_level
    long amb_rank = 0;              // Ambient rank of this DedispersionTree (= stage1_dd_rank)
    long dd_rank = 0;               // Active rank of this DedispersionTree (= stage1_amb_rank - et_level)
    long dd_rank1 = 0;              // = (dd_rank+1)/2, mirroring GPU kernel
    long xdm_rank = 0;              // K = dd_rank1 - frequency_subbands.pf_rank
    long nt_ds = 0;                 // = config.time_samples_per_chunk / pow2(primary_tree_index)

    // Downsampling factors in coarse-grained dedispersion output.
    long dm_downsampling = 0;       // = pow2(dd_rank1), mirroring GPU kernel
    long time_downsampling = 0;     // = pow2(dd_rank1), mirroring GPU kernel

    // Array shapes for peak-finding and related kernels.
    // 'wt' array shape is (beams_per_batch, ndm_wt, nt_wt, nprofiles, frequency_subbands.N).
    // 'out_max', 'out_argmax' shapes are (beams_per_batch, ndm_out, nt_out).
    long nprofiles = 0;             // = 1 + 3*log2(primary_tree.max_width)
    long ndm_out = 0;               // = pow2(tree_rank) / dm_downsampling
    long ndm_wt = 0;                // = pow2(tree_rank) / primary_tree.wt_dm_downsampling
    long nt_out = 0;                // = nt_ds / time_downsampling
    long nt_wt = 0;                 // = nt_ds / primary_tree.wt_time_downsampling

    // Currently, these informational members are just used in print-statements.
    double dm_min = 0.0;
    double dm_max = 0.0;
    double trigger_frequency = 0.0f;

    // TOPLEVEL half-open tree-freq range spanned by subband 0 <= n < fs.N.
    // 0 <= flo < fhi <= pow2(toplevel_tree_rank - early_trigger_level).
    // Note: defined in DedispersionPlan.cpp (we don't have a DedispersionTree.cpp).
    long n_to_toplevel_flo(long n) const;
    long n_to_toplevel_fhi(long n) const;
};


}  // namespace pirate

#endif // _PIRATE_DEDISPERSION_TREE_HPP

