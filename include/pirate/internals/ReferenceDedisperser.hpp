#ifndef _PIRATE_INTERNALS_REFERENCE_DEDISPERSER_HPP
#define _PIRATE_INTERNALS_REFERENCE_DEDISPERSER_HPP

#include <vector>
#include <memory>  // shared_ptr
#include <iostream>
#include <ksgpu/Array.hpp>

#include "../DedispersionConfig.hpp"
#include "../DedispersionPlan.hpp"


namespace pirate {
#if 0
}  // editor auto-indent
#endif


// -------------------------------------------------------------------------------------------------
//
// Sophistication == 0:
//
//   - Uses one-stage dedispersion instead of two stages.
//   - In downsampled trees, compute twice as many DMs as necessary, then drop the bottom half.
//   - Each early trigger is computed in an independent tree, by disregarding some input channels.
//
// Sophistication == 1:
//
//   - Uses same two-stage tree/lag structure as plan.
//   - Lags are applied with a per-tree ReferenceLagbuf, rather than using ring/staging buffers.
//   - Lags are split into segments + residuals, but not further split into chunks.
//
// Sophistication == 2:
//
//   - As close to GPU implementation as possible!


struct ReferenceDedisperserBase
{
    // Constructor not intended to be called directly -- use make() below.
    ReferenceDedisperserBase(const std::shared_ptr<DedispersionPlan> &plan_, int sophistication_);
    
    std::shared_ptr<DedispersionPlan> plan;
    
    const DedispersionConfig config;   // same as plan->config
    const int sophistication;
        
    int config_rank = 0;         // same as config.tree_rank
    int config_ntime = 0;        // same as config.time_samples_per_chunk
    int total_beams = 0;         // same as config.beams_per_gpu
    int beams_per_batch = 0;     // same as config.beams_per_batch
    int nbatches = 0;            // = (total_beams / beams_per_batch)
    
    int nds = 0;                 // same as plan->stage0_trees.size(), i.e. number of downsampling (ids) values
    int nout = 0;                // same as plan->stage1_trees.size(), i.e. number of (ids, early_trigger) pairs
    int nelts_per_segment = 0;   // same as plan->nelts_per_segment
    
    // To process multiple chunks, call the dedisperse() method in a loop.
    virtual void dedisperse(long ibatch, long it_chunk) = 0;

    // Before calling dedisperse(), caller should fill 'input_array'.
    // Shape is (beams_per_batch, 2^config_rank, config_ntime).
    // 
    // After dedisperse() completes, dedispersion output is stored in 'output_arrays'.
    // output_arrays[iout] has shape (beams_per_batch, 2^output_rank, config_ntime / pow2(ids)), where:
    //
    //   ids = downsampling factor of tree 'iout' (same as DedispersionPlan::Stage0Tree::ds_level)
    //   output_rank = rank of tree 'iout' (same as Stage1Tree::rank0 + Stage1Tree::rank1_trigger)
    //
    // Warning: dedisperse() may modify 'input_array'!
    
    ksgpu::Array<float> input_array;
    std::vector<ksgpu::Array<float>> output_arrays;

    // Factory function -- constructs ReferenceDedisperser of specified sophistication.
    static std::shared_ptr<ReferenceDedisperserBase> make(const std::shared_ptr<DedispersionPlan> &plan_, int sophistication);

    // Helper function called by subclass
    void check_iobuf_shapes();
};


}  // namespace pirate

#endif // _PIRATE_INTERNALS_REFERENCE_DEDISPERSER_HPP
