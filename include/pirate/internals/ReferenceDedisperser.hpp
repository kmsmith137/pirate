#ifndef _PIRATE_INTERNALS_REFERENCE_DEDISPERSER_HPP
#define _PIRATE_INTERNALS_REFERENCE_DEDISPERSER_HPP

#include <vector>
#include <memory>  // shared_ptr
#include <iostream>
#include <gputils/Array.hpp>

#include "../DedispersionConfig.hpp"
#include "../DedispersionPlan.hpp"

#include "ReferenceTree.hpp"
#include "ReferenceLagbuf.hpp"
#include "ReferenceLaggedDownsampler.hpp"
#include "ReferenceDedispersionKernel.hpp"


namespace pirate {
#if 0
}  // editor auto-indent
#endif


struct ReferenceDedisperserBase
{
    // Constructor not intended to be called directly -- use make() below.
    ReferenceDedisperserBase(const std::shared_ptr<DedispersionPlan> &plan_, int sophistication_);
    
    std::shared_ptr<DedispersionPlan> plan;
    const DedispersionConfig config;   // same as plan->config
    const int sophistication;
        
    int config_rank = 0;         // same as config.tree_rank
    int config_ntime = 0;        // same as config.time_samples_per_chunk
    int total_beams;             // same as config.beams_per_gpu
    int beams_per_batch;         // same as config.beams_per_batch
    
    int nds = 0;                 // same as plan->stage0_trees.size(), i.e. number of downsampling (ids) values
    int output_ntrees = 0;       // same as plan->stage1_trees.size(), i.e. number of (ids, early_trigger) pairs
    int nelts_per_segment = 0;   // same as plan->nelts_per_segment

    // To process multiple chunks, call the dedisperse() method in a loop.
    void dedisperse(long itime, long ibeam);

    // Before calling dedisperse(), caller should fill 'input_array'.
    // Shape is (beams_per_batch, 2^config_rank, input_nt).
    // 
    // After dedisperse() completes, dedispersion output is stored in 'output_arrays'.
    // output_arrays[itree] has shape (beams_per_batch, 2^output_rank, config_ntime / pow2(ids)), where:
    //
    //   ids = downsampling factor of tree 'itree' (same as DedispersionPlan::Stage0Tree::ds_level)
    //   output_rank = rank of tree 'itree' (same as Stage1Tree::rank0 + Stage1Tree::rank1_trigger)
    
    gputils::Array<float> input_array;
    std::vector<gputils::Array<float>> output_arrays;

    // Factory function -- constructs ReferenceDedisperser of specified sophistication.
    static std::shared_ptr<ReferenceDedisperserBase> make(const std::shared_ptr<DedispersionPlan> &plan_, int sophistication);
    
    // Dedispersion implementation supplied by subclass.
    virtual void _dedisperse(const gputils::Array<float> &in) = 0;
};


// -------------------------------------------------------------------------------------------------
//
// Sophistication == 0:
//
//   - Uses one-stage dedispersion instead of two stages.
//   - In downsampled trees, compute twice as many DMs as necessary, then drop the bottom half.
//   - Each early trigger is computed in an independent tree, by disregarding some input channels.


struct ReferenceDedisperser0 : public ReferenceDedisperserBase
{
    ReferenceDedisperser0(const std::shared_ptr<DedispersionPlan> &plan_);

    virtual void _dedisperse(const gputils::Array<float> &in) override;

    // Step 1: downsample input array (straightforward downsample, not "lagged" downsample).
    // Outer length is nds, inner shape is (beams_per_batch, 2^config_rank, input_nt / pow2(ids)).
    std::vector<gputils::Array<float>> downsampled_inputs;
    
    std::vector<gputils::Array<float>> dedispersion_buffers;  // length output_ntrees
    std::vector<std::shared_ptr<ReferenceTree>> trees;        // length output_ntrees
};


// -------------------------------------------------------------------------------------------------
//
// Sophistication == 1:
//
//   - Uses same two-stage tree/lag structure as plan.
//   - Lags are split into segments + residuals, but not further split into chunks.
//   - Lags are applied with a per-tree ReferenceLagbuf, rather than using ring/staging buffers.


struct ReferenceDedisperser1 : public ReferenceDedisperserBase
{
    ReferenceDedisperser1(const std::shared_ptr<DedispersionPlan> &plan_);

    virtual void _dedisperse(const gputils::Array<float> &in) override;

    gputils::Array<float> stage0_iobuf;
    gputils::Array<float> stage1_iobuf;

    std::shared_ptr<ReferenceLaggedDownsampler> lagged_downsampler;
    std::vector<std::shared_ptr<ReferenceDedispersionKernel>> stage0_kernels;   // length nds
    std::vector<std::shared_ptr<ReferenceDedispersionKernel>> stage1_kernels;   // length output_ntrees
    std::vector<std::shared_ptr<ReferenceLagbuf>> stage1_lagbufs;               // length output_ntrees
};    


// -------------------------------------------------------------------------------------------------
//
// Sophistication == 2: as close to GPU implementation as possible.


struct ReferenceDedisperser2 : public ReferenceDedisperserBase
{
    ReferenceDedisperser2(const std::shared_ptr<DedispersionPlan> &plan_);

    virtual void _dedisperse(const gputils::Array<float> &in) override;

    gputils::Array<float> stage0_iobuf;
    gputils::Array<float> stage1_iobuf;
    gputils::Array<float> gpu_ringbuf;

    std::shared_ptr<ReferenceLaggedDownsampler> lagged_downsampler;
    std::vector<std::shared_ptr<ReferenceDedispersionKernel>> stage0_kernels;   // length nds
    std::vector<std::shared_ptr<ReferenceDedispersionKernel>> stage1_kernels;   // length output_ntrees
};


}  // namespace pirate

#endif // _PIRATE_INTERNALS_REFERENCE_DEDISPERSER_HPP
