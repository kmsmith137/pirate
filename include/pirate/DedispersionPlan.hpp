#ifndef _PIRATE_DEDISPERSION_PLAN_HPP
#define _PIRATE_DEDISPERSION_PLAN_HPP

#include "DedispersionConfig.hpp"
#include "DedispersionBuffer.hpp"        // struct DedispersionBufferParams
#include "DedispersionKernel.hpp"        // struct DedispersionKernelParams
#include "LaggedDownsamplingKernel.hpp"  // struct LaggedDownsamplingKernelParams
#include "RingbufCopyKernel.hpp"         // struct RingbufCopyKernelParams
#include "TreeGriddingKernel.hpp"        // struct TreeGriddingKernelParams

#include <vector>
#include <memory>  // shared_ptr
#include <ksgpu/Array.hpp>


namespace pirate {
#if 0
}  // editor auto-indent
#endif

// Defined in MegaRingbuf.hpp.
struct MegaRingbuf;


struct DedispersionPlan
{
    DedispersionPlan(const DedispersionConfig &config);
    
    const DedispersionConfig config;

    long stage1_ntrees = 0;  // same as config.num_downsampling_levels
    long stage2_ntrees = 0;
    
    TreeGriddingKernelParams tree_gridding_kernel_params;
    
    DedispersionBufferParams stage1_dd_buf_params;  // (number of buffers) = stage1_ntrees
    DedispersionBufferParams stage2_dd_buf_params;  // (number of buffers) = stage2_ntrees
    
    std::vector<DedispersionKernelParams> stage1_dd_kernel_params;  // length stage1_ntrees
    std::vector<DedispersionKernelParams> stage2_dd_kernel_params;  // length stage2_ntrees
    std::vector<long> stage2_ds_level;  // length stage2_ntrees
    
    LaggedDownsamplingKernelParams lds_params;

    // Only needed if early triggers are used.
    RingbufCopyKernelParams g2g_copy_kernel_params;
    RingbufCopyKernelParams h2h_copy_kernel_params;

    void print(std::ostream &os=std::cout, int indent=0) const;

    // Members after this should not be needed "from the outside".
    // FIXME: reorganize code to reflect this?
    
    // --------------------  Helper classes  --------------------
    
    struct Stage1Tree
    {
        // Note: total tree rank (rank0 + rank1) is equal to (config.tree_rank - (ds_level ? 1 : 0)).        
        int ds_level = -1;  // downsampling level (downsampling "factor" is 2^level)
        int rank0 = 0;      // rank of Stage1Tree
        int rank1 = 0;      // rank of subsequent Stage2Tree (if no early trigger)
        int nt_ds = 0;      // downsampled time samples per chunk (= config.time_samples_per_chunk / pow2(ds_level))
    };

    struct Stage2Tree
    {
        int ds_level = -1;       // Same as Stage1Tree::ds_level
        int rank0 = 0;           // Same as Stage1Tree::rank0
        int rank1_ambient = 0;   // Same as Stage1Tree::rank1
        int rank1_trigger = 0;   // Can be smaller than rank1_ambient, for early trigger
        int nt_ds = 0;           // Same as Stage1Tree::nt_ds
    };

    int nelts_per_segment = 0;   // currently always constants::bytes_per_gpu_cache_line / (sizeof config dtype)
    int nbytes_per_segment = 0;  // currently always constants::bytes_per_gpu_cache_line
    
    std::vector<Stage1Tree> stage1_trees;  // length stage1_ntrees
    std::vector<Stage2Tree> stage2_trees;  // length stage2_ntrees

    std::shared_ptr<MegaRingbuf> mega_ringbuf;
};


}  // namespace pirate

#endif // _PIRATE_DEDISPERSION_PLAN_HPP
