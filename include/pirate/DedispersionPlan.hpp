#ifndef _PIRATE_DEDISPERSION_PLAN_HPP
#define _PIRATE_DEDISPERSION_PLAN_HPP

#include "DedispersionConfig.hpp"
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

    // Defines the mapping from stage2 trees to stage1 trees.
    // Length (stage2_ntrees), each element is an index 0 <= ds_level < stage1_ntrees.
    std::vector<long> stage2_ds_level;

    int nelts_per_segment = 0;   // currently always constants::bytes_per_gpu_cache_line / (sizeof config dtype)
    int nbytes_per_segment = 0;  // currently always constants::bytes_per_gpu_cache_line

    void print(std::ostream &os=std::cout, int indent=0) const;

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

        // It's convenient to put some peak-finding info here.
        // More peak-finding info is in DeispersionPlan::stage2_pf_params (Wmax, ndm_{out,wt}, nt_{in,out,wt})
        long nprofiles = 0;      // same as PeakFindingKernel::nprofiles, equal to 1 + 3*log2(Wmax)
        long nsubbands = 0;      // same as FrequencySubbands::F
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
