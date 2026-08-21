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


// Dedispersion is a two-stage process. In the first stage, there is one tree for each
// primary tree (see DedispersionConfig.hpp). In the second stage, each primary tree is
// expanded into (num_early_triggers+1) trees: the main (full-band) tree, plus one "early"
// tree for each early_trigger_level = 1..num_early_triggers. Thus, there is a many-to-one mapping
// from stage2 trees to stage1 trees.

struct DedispersionPlan
{
    // cdd2_kernel_required: if true, then Dcore values will be taken from the cdd2 kernel
    // registry, and an exception will be thrown if a cdd2 kernel is missing. If false,
    // then default Dcore values will be assigned (Dcore = pf.time_downsampling).
    //
    // Since the Dcore values are then not the compiled kernels' values, a false plan
    // cannot be used in a GpuDedisperser. It is still perfectly usable on the GPU by
    // callers that drive kernels themselves without going through cdd2 -- see
    // varmap.brute_force._GpuSweep, which runs GpuDedispersionKernel,
    // GpuSbDedispersionKernel and GpuPfSquare -- as well as in host-only contexts such
    // as the 'pirate_frb show_dedisperser' CLI.
    //
    // Not to be confused with the config-level 'gpu_valid' flag in
    // DedispersionConfig::make_random(), which restricts random configs to precompiled
    // cdd2 kernels.
    explicit DedispersionPlan(const DedispersionConfig &config,
                              bool cdd2_kernel_required = true);

    const DedispersionConfig config;
    const bool cdd2_kernel_required;

    // Some key members of DedispersionConfig, copied into DedispersionPlan for convenience.
    ksgpu::Dtype dtype;                  // same as config.dtype
    long nfreq = 0;                      // same as config.get_total_nfreq()
    long nt_in = 0;                      // same as config.time_samples_per_chunk
    long num_primary_trees = 0;          // same as config.num_primary_trees()
    long beams_per_gpu = 0;              // same as config.beams_per_gpu
    long beams_per_batch = 0;            // same as config.beams_per_batch
    long num_active_batches = 0;         // same as config.num_active_batches
    long nbits = 0;                      // same as config.dtype.nbits

    // Stage1 trees. These trees are "internal" to dedispersion, and can probably be ignored "from outside".
    // Total tree rank (dd_rank + amb_rank) is equal to (toplevel_tree_rank - (primary_tree_index ? 1 : 0)).
    // Both vectors have length (num_primary_trees).
    std::vector<long> stage1_dd_rank;    // "Active" dedispersion rank of each stage1 tree.
    std::vector<long> stage1_amb_rank;   // "Ambient" rank of each stage1 tree (= number of coarse freq channels)

    // Stage2 trees. These trees contain the output of the dedispersion, and are useful "from outside".
    // There is a lot of per-tree data, so I defined a helper class 'DedispersionTree'.
    // The number of trees is (config.num_primary_trees() + total number of early triggers).
    //
    // Reminder: DedispersionTree defines the following methods (see DedispersionTree.hpp):
    //    compute_steady_state_it0(config)
    //    decode_argmax(argmax_token, ...)
    //    decode_argmax2(config, ...)    
    long ntrees = 0;
    std::vector<DedispersionTree> trees;  // length ntrees

    // 'verbose' controls explanatory comments; 'zones' independently controls
    // whether the mega_ringbuf per-clag host/gpu zone breakdown is emitted.
    void to_yaml(YAML::Emitter &emitter, bool verbose = false, bool zones = false) const;
    std::string to_yaml_string(bool verbose = false, bool zones = false) const;


    // -------------------------------------------------------------------------------------------------
    //
    // Low-level data needed for compute kernels.


    int nelts_per_segment = 0;   // currently always constants::bytes_per_gpu_cache_line / (sizeof config dtype)
    int nbytes_per_segment = 0;  // currently always constants::bytes_per_gpu_cache_line
    
    // MegaRingbuf: this data structure is the "nerve center" of the real-time FRB search.
    // I have written a short novel explaining how it works, in MegaRingbuf.hpp.
    std::shared_ptr<MegaRingbuf> mega_ringbuf;

    TreeGriddingKernelParams tree_gridding_kernel_params;
    LaggedDownsamplingKernelParams lds_params;

    DedispersionBufferParams stage1_dd_buf_params;  // (number of buffers) = num_primary_trees
    DedispersionBufferParams stage2_dd_buf_params;  // (number of buffers) = ntrees

    std::vector<DedispersionKernelParams> stage1_dd_kernel_params;  // length num_primary_trees
    std::vector<DedispersionKernelParams> stage2_dd_kernel_params;  // length ntrees

    // Note: stage2_pf_params[:].Dcore is copied from trees[:].Dcore (which the constructor
    // fills from the cdd2 kernel registry if cdd2_kernel_required, else from a default; see
    // Part 1), so that peak-finders built from the plan (GPU or reference) agree on
    // out_argmax token granularity.
    std::vector<PeakFindingKernelParams> stage2_pf_params;          // length ntrees

    // Only needed if early triggers are used.
    RingbufCopyKernelParams g2g_copy_kernel_params;
    RingbufCopyKernelParams h2h_copy_kernel_params;
};


}  // namespace pirate

#endif // _PIRATE_DEDISPERSION_PLAN_HPP
