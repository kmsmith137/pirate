#ifndef _PIRATE_DEDISPERSION_TREE_HPP
#define _PIRATE_DEDISPERSION_TREE_HPP

#include "DedispersionConfig.hpp"
#include "FrequencySubbands.hpp"

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// DedispersionTree: a "data" class representing the output of the dedisperser, for one
// choice of (primary tree, early trigger). Its member functions are all trivial derived
// quantities.
//
// Trees are created in the DedispersionPlan constructor, which is the only place the geometry
// is implemented, and get copied out of DedispersionPlan::trees by the dedisperser classes
// (GpuDedisperser, ReferenceDedisperser). Everything that INTERPRETS a tree -- decoding
// out_argmax tokens, the subband index mappings, yaml I/O -- is a DedispersionPlan method.
//
// A plan built with DedispersionPlan::Params::minimal() needs no GPU, so trees are available
// on any machine -- but only through a plan.

struct DedispersionTree
{
    int primary_tree_index = -1;   // Also identifies associated stage1 tree (input downsampled in time by 2^primary_tree_index).
    int early_trigger_level = -1;  // "Earliness" of trigger: 0 for the main tree, 1..num_early_triggers for early triggers.
    int amb_rank = 0;              // Ambient rank of this DedispersionTree (= dd_rank of associated stage1 tree)
    int dd_rank = 0;               // Active rank of this DedispersionTree (= amb_rank of stage1 tree, minus early_trigger_level)
    int nt_ds = 0;                 // Downsampled time samples per chunk (= config.time_samples_per_chunk / pow2(primary_tree_index))

    // Total tree rank. (For its relation to the config's toplevel rank, see
    // toplevel_tree_rank() just below.)
    long total_rank() const { return amb_rank + dd_rank; }

    // The toplevel_tree_rank of the DedispersionConfig this tree came from, recovered from
    // the tree's own members. Named to match DedispersionConfig::toplevel_tree_rank.
    //
    // Note '(primary_tree_index > 0)' rather than a bare 'primary_tree_index': the member is
    // -1 in a default-constructed tree, which the bare form would silently read as 1.
    long toplevel_tree_rank() const {
        return total_rank() + ((primary_tree_index > 0) ? 1 : 0) + early_trigger_level;
    }

    // Subbands searched in this tree.
    // Can differ from DedispersionConfig::frequency_subbands, due to early triggers and downsampling.
    FrequencySubbands frequency_subbands;

    // Toplevel-relative tree-freq range spanned by subband 'n', i.e. channels of the
    // rank-toplevel_tree_rank() gridding. HALF-OPEN, matching FrequencySubbands::n_to_f{lo,hi},
    // of which these are just a rescaling:
    //
    //     0 <= flo < fhi <= pow2(toplevel_tree_rank())
    //
    // with the sharper per-tree bound fhi <= pow2(toplevel_tree_rank() - early_trigger_level).
    //
    // NOTE DedispersionPlan::decode_argmax() reports an INCLUSIVE upper channel, i.e. its
    // 'fmax' output is n_to_toplevel_fhi(n) - 1.
    long n_to_toplevel_flo(long n) const;
    long n_to_toplevel_fhi(long n) const;

    // Contains members: num_early_triggers, max_width, wt_{dm,time}_downsampling. An EXACT
    // copy of config.primary_trees[primary_tree_index]: the DedispersionPlan constructor
    // resolves nothing into it (the resolved factors are the tree members just below).
    DedispersionConfig::PrimaryTree pf;

    // The GPU kernel's second-stage rank: the DedispersionPlan constructor pins
    // dm_downsampling to pow2(dd_rank1), and it is what xdm_rank() is measured against.
    long dd_rank1() const { return (dd_rank + 1) / 2; }

    // DM downsampling factor of the coarse-grained array, relative to this tree.
    // Currently "hardwired" to pow2(dd_rank1()).
    long dm_downsampling = 0;

    // Time downsampling factor of the coarse-grained array, relative to this tree.
    // Currently "hardwired" to pow2(dd_rank1())  [= dm_downsampling]
    long time_downsampling = 0;

    // K = dd_rank1() - frequency_subbands.pf_rank: the number of "extra DM" bits that this
    // tree's cdd2 kernel folds into the argmax token's m-field (see CoalescedDdKernel2.hpp).
    // Zero unless the tree has an early trigger.
    //
    // Derived from dm_downsampling (which the DedispersionPlan constructor pins to
    // pow2(dd_rank1())) rather than stored, so it is not one more field for the plan yaml
    // and its verifier to keep honest.
    long xdm_rank() const;

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

