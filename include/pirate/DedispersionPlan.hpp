#ifndef _PIRATE_DEDISPERSION_PLAN_HPP
#define _PIRATE_DEDISPERSION_PLAN_HPP

#include "DedispersionConfig.hpp"

namespace YAML { class Emitter; }  // #include <yaml-cpp/yaml.h>
#include "DedispersionBuffer.hpp"        // struct DedispersionBufferParams
#include "DedispersionKernel.hpp"        // struct DedispersionKernelParams
#include "LaggedDownsamplingKernel.hpp"  // struct LaggedDownsamplingKernelParams
#include "RingbufCopyKernel.hpp"         // struct RingbufCopyKernelParams
#include "TreeGriddingKernel.hpp"        // struct TreeGriddingKernelParams
#include "PeakFindingKernel.hpp"         // struct PeakFindingKernelParams
#include "MegaRingbuf.hpp"               // struct MegaRingbuf

#include <vector>
#include <memory>  // shared_ptr
#include <ksgpu/Array.hpp>


namespace pirate {
#if 0
}  // editor auto-indent
#endif


struct DedispersionPlan
{
    DedispersionPlan(const DedispersionConfig &config);
    
    const DedispersionConfig config;

    // Dedispersion is a two-stage process. In the first stage, there is one tree for each
    // downsampling level. In the second stage, each downsampling level has a "primary" tree,
    // plus one "early" tree for each EarlyTrigger (see DedispersionConfig.hpp). Thus,
    // there is a many-to-one mapping from stage2 trees to stage1 trees.

    long stage1_ntrees = 0;  // = (config.num_downsampling_levels)
    long stage2_ntrees = 0;  // = (config.num_downsampling_levels + total number of early triggers)

    int nelts_per_segment = 0;   // currently always constants::bytes_per_gpu_cache_line / (sizeof config dtype)
    int nbytes_per_segment = 0;  // currently always constants::bytes_per_gpu_cache_line

    void print(std::ostream &os=std::cout, int indent=0) const;

    // Write in YAML format.
    void to_yaml(YAML::Emitter &emitter, bool verbose = false) const;
    std::string to_yaml_string(bool verbose = false) const;

    // -------------------------------------------------------------------------------------------------
    //
    // Stage1Trees, Stage2Trees.

    struct Stage1Tree
    {
        // Note: total tree rank (dd_rank + amb_rank) is equal to (config.tree_rank - (ds_level ? 1 : 0)).        
        int ds_level = -1;  // Downsampling level (downsampling "factor" is 2^level)
        int dd_rank = 0;    // "Active" dedispersion rank of Stage1Tree 
        int amb_rank = 0;   // "Ambient" rank of Stage1Tree (= number of coarse freq channels)
        int nt_ds = 0;      // Downsampled time samples per chunk (= config.time_samples_per_chunk / pow2(ds_level))
    };

    struct Stage2Tree
    {
        // Note: for most purposes, you want 'early_dd_rank', not 'pri_dd_rank'.
        int ds_level = -1;       // Downsampling level, also identifies associated Stage1Tree.
        int amb_rank = 0;        // Ambient rank of Stage2Tree (= dd_rank of associated Stage1Tree)
        int pri_dd_rank = 0;     // Active rank of primary Stage2Tree (= amb_rank of associated Stage1Tree)
        int early_dd_rank = 0;   // Active rank of this Stage2Tree (always <= pri_dd_rank)
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
    
    std::vector<Stage1Tree> stage1_trees;  // length stage1_ntrees
    std::vector<Stage2Tree> stage2_trees;  // length stage2_ntrees

    // -------------------------------------------------------------------------------------------------
    //
    // Low-level data needed for compute kernels.

    // MegaRingbuf: this data structure is the "nerve center" of the real-time FRB search.
    // I have written a short novel explaining how it works, in MegaRingbuf.hpp.
    std::shared_ptr<MegaRingbuf> mega_ringbuf;

    TreeGriddingKernelParams tree_gridding_kernel_params;
    LaggedDownsamplingKernelParams lds_params;

    DedispersionBufferParams stage1_dd_buf_params;  // (number of buffers) = stage1_ntrees
    DedispersionBufferParams stage2_dd_buf_params;  // (number of buffers) = stage2_ntrees
    
    std::vector<DedispersionKernelParams> stage1_dd_kernel_params;  // length stage1_ntrees
    std::vector<DedispersionKernelParams> stage2_dd_kernel_params;  // length stage2_ntrees
    std::vector<PeakFindingKernelParams> stage2_pf_params;          // length stage2_ntrees

    // Only needed if early triggers are used.
    RingbufCopyKernelParams g2g_copy_kernel_params;
    RingbufCopyKernelParams h2h_copy_kernel_params;
};


}  // namespace pirate

#endif // _PIRATE_DEDISPERSION_PLAN_HPP
