#ifndef _PIRATE_DEDISPERSION_PLAN_HPP
#define _PIRATE_DEDISPERSION_PLAN_HPP

#include "DedispersionConfig.hpp"

namespace YAML { class Emitter; }  // #include <yaml-cpp/yaml.h>
#include "DedispersionTree.hpp"           // struct DedispersionTree
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

    long num_downsampling_levels = 0;  // = (config.num_downsampling_levels)
    long ntrees = 0;  // = (config.num_downsampling_levels + total number of early triggers)

    int nelts_per_segment = 0;   // currently always constants::bytes_per_gpu_cache_line / (sizeof config dtype)
    int nbytes_per_segment = 0;  // currently always constants::bytes_per_gpu_cache_line

    void print(std::ostream &os=std::cout, int indent=0) const;

    // Write in YAML format.
    void to_yaml(YAML::Emitter &emitter, bool verbose = false) const;
    std::string to_yaml_string(bool verbose = false) const;

    // -------------------------------------------------------------------------------------------------


    // Stage1 tree info (vectors have length == num_downsampling_levels).
    // Note: total tree rank (dd_rank + amb_rank) is equal to (config.tree_rank - (ds_level ? 1 : 0)).     
    std::vector<long> stage1_dd_rank;    // "Active" dedispersion rank of each stage1 tree.
    std::vector<long> stage1_amb_rank;   // "Ambient" rank of each stage1 tree (= number of coarse freq channels)

    std::vector<DedispersionTree> trees;  // length ntrees

    // -------------------------------------------------------------------------------------------------
    //
    // Low-level data needed for compute kernels.

    // MegaRingbuf: this data structure is the "nerve center" of the real-time FRB search.
    // I have written a short novel explaining how it works, in MegaRingbuf.hpp.
    std::shared_ptr<MegaRingbuf> mega_ringbuf;

    TreeGriddingKernelParams tree_gridding_kernel_params;
    LaggedDownsamplingKernelParams lds_params;

    DedispersionBufferParams stage1_dd_buf_params;  // (number of buffers) = num_downsampling_levels
    DedispersionBufferParams stage2_dd_buf_params;  // (number of buffers) = ntrees
    
    std::vector<DedispersionKernelParams> stage1_dd_kernel_params;  // length num_downsampling_levels
    std::vector<DedispersionKernelParams> stage2_dd_kernel_params;  // length ntrees
    std::vector<PeakFindingKernelParams> stage2_pf_params;          // length ntrees

    // Only needed if early triggers are used.
    RingbufCopyKernelParams g2g_copy_kernel_params;
    RingbufCopyKernelParams h2h_copy_kernel_params;
};


}  // namespace pirate

#endif // _PIRATE_DEDISPERSION_PLAN_HPP
