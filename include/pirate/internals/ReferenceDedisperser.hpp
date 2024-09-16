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


namespace pirate {
#if 0
}  // editor auto-indent
#endif


// -------------------------------------------------------------------------------------------------


struct ReferenceDedisperser
{
    // The ReferenceDedisperser has a 'sophistication' argument, which is one of {0,1,2,3}.
    //
    // sophistication=0:
    //   - Uses one-stage dedispersion instead of two stages.
    //   - In downsampled trees, compute twice as many DMs as necessary, then drop the bottom half.
    //   - Each early trigger is computed in an independent tree, by disregarding some input channels.
    //
    // sophistication=1:
    //   - use same tree/lag structure as plan, but don't use LaggedCacheLines
    //
    // sophistication=2:
    //   - using Stage0/Stage1 trees and LaggedCacheLines from plan
    //   - but buffering entire chunk to max clag, rather than using CacheLineRingbuf
    //
    // sophistication=3:
    //   - Using CacheLineRingBuf
    //
    // Note: the ReferenceDedisperser uses float32 everywhere (i.e. ignores values of
    // DedispersionConfig::uncompressed_dtype and DedispersionConfig::compressed_dtype).
    
    ReferenceDedisperser(const std::shared_ptr<DedispersionPlan> &plan_, int sophistication);

    const int sophistication;
    const DedispersionConfig config;
    std::shared_ptr<DedispersionPlan> plan;
    
    int input_rank = 0;
    int input_nt = 0;
    int output_ntrees = 0;
    int nds = 0;
    
    // Counts cumulative calls to dedisperse()
    ssize_t pos = 0;

    // The 'in' array represents one "chunk", with shape (2^input_rank, input_nt).
    // To process multiple chunks, call the dedipserse() method in a loop.
    void dedisperse(const gputils::Array<float> &in);
    
    void print(std::ostream &os=std::cout, int indent=0) const;
    
    std::vector<gputils::Array<float>> output_arrays;  // length output_ntrees
    gputils::Array<float> output_flattened;
    void _allocate_output_arrays();

    
    // -------------------------------------------------------------------------------------------------
    //
    // Sophistication 0
    //
    //   - Uses one-stage dedispersion instead of two stages.
    //   - In downsampled trees, compute twice as many DMs as necessary, then drop the bottom half.
    //   - Each early trigger is computed in an independent tree, by disregarding some input channels.
    //   - First, make a copy of the data at each downsampling factor
    //   - Then, for each output tree (i.e. choice of downsampling factor and early trigger), dedisperse.

    
    struct Soph0Tree
    {
	Soph0Tree(const DedispersionPlan::Stage1Tree &st1);

	// Input array will be an element of this->soph0_ds_inputs (see below)
	// Output array will be an element of this->output_arrays.
	void dedisperse(const gputils::Array<float> &in, gputils::Array<float> &out);

	const bool is_downsampled;    // (st1.ds_level > 0)
	const int output_rank;        // (st1.rank0 + st1.rank1_trigger)
	const int nt_ds;              // (st1.nt_ds)
	
	std::shared_ptr<ReferenceTree> rtree;  // rank = is_downsampled ? (output_rank+1) : output_rank
	gputils::Array<float> rstate;          // length rtree->nrstate
	gputils::Array<float> scratch;         // length rtree->nscratch
	gputils::Array<float> iobuf;           // shape (pow2(rtree->rank), nt_ds)
    };

    // soph0_ds_inputs[ids] has shape (2^input_rank, input_nt/2^ids), where 0 <= ids < nds.
    // It contains the input array after downsampling by a factor 2^ids.
    
    std::vector<gputils::Array<float>> soph0_ds_inputs;  // length nds
    void _allocate_soph0_ds_inputs();
    void _compute_soph0_ds_inputs(const gputils::Array<float> &in);
    
    // Used if sophistication == 0.
    std::vector<Soph0Tree> soph0_trees;
    void _init_soph0_trees();
    void _apply_soph0_trees();

    
    // -------------------------------------------------------------------------------------------------
    //
    // General structure for (sophistication >= 1).
    //
    // The following logic is shared by all cases with sophistication >= 1.
    //
    //   - Dedispersion is done in two stages, with classes Pass0Tree and Pass1Tree.
    //     These are in 1-1 correspondence with DedispersionPlan::{Stage0Tree,Stage1Tree}.
    //     (For no good reason, We use "Pass" in the ReferenceDedipserse, and "Stage" in the DedispersionPlan.)
    //
    //   - The Pass0Trees all operate on a 'pass0_iobuf' array, and the Pass1Trees all operate
    //     on the 'output_array' (see above).
    //
    //   - First, we run a ReferenceLaggedDownsampler, which populates the pass0_iobuf.
    //
    //   - Second, we run the Pass0Trees, which operate in-place on the pass0_iobuf.
    //
    //   - Third, we populate the output_array with current+previous elements of the pass0_iobuf,
    //     using some lagging logic.
    //
    //     **NOTE** The different sophistication values {1,2,3} just differ in the details of
    //      how this lagging logic is implemented.
    //
    //   - Fourth, we run the Pass1Trees, which operate in-place on the output_arrays.
    
    
    struct Pass0Tree
    {
	Pass0Tree(const DedispersionPlan::Stage0Tree &st0);

	// Array argument will be an element of this-pass0_iobufs.
	void dedisperse(gputils::Array<float> &arr);
	
	const bool is_downsampled;
	const int output_rank0;   // same as DedispersionPlan::Stage0Tree::rank0
	const int output_rank1;   // same as DedispersionPlan::Stage0Tree::rank1
	const int nt_ds;
	    
	std::shared_ptr<ReferenceTree> rtree;       // rank = output_rank0
	gputils::Array<float> rstate;
	gputils::Array<float> scratch;
    };


    struct Pass1Tree
    {
	Pass1Tree(const DedispersionPlan::Stage1Tree &st1);

	// Array argument will be an element of this->output_arrays.
	void dedisperse(gputils::Array<float> &arr);

	const int rank0;
	const int rank1;
	const int nt_ds;
	
	std::shared_ptr<ReferenceTree> rtree;
	gputils::Array<float> rstate;
	gputils::Array<float> scratch;
    };

    
    std::vector<gputils::Array<float>> pass0_iobufs;   // length nds, elements are 2-d arrays
    gputils::Array<float> pass0_iobuf_flattened;       // 1-d array, for addressing by segment id
    void _allocate_pass0_iobuf();

    std::shared_ptr<ReferenceLaggedDownsampler> lagged_downsampler;
    void _init_lagged_downsampler();
    void _apply_lagged_downsampler(const gputils::Array<float> &in);   // populates pass0_iobufs

    std::vector<Pass0Tree> pass0_trees;
    void _init_pass0_trees();
    void _apply_pass0_trees();    // operates on pass0_iobufs
    
    std::vector<Pass1Tree> pass1_trees;
    void _init_pass1_trees();
    void _apply_pass1_trees();    // operates on pass0_iobufs


    // -------------------------------------------------------------------------------------------------

    
    // Used if sophistication == 1.
    std::vector<std::shared_ptr<ReferenceLagbuf>> big_lagbufs;
    void _init_big_lagbufs();
    void _apply_big_lagbufs();

    // Used if sophistication == 2.
    int max_clag = -1;
    int max_ringpos = 0;
    gputils::Array<float> max_ringbuf;   // shape (max_clag+1, len(intermediate_flattened))
    void _init_max_ringbuf();
    void _apply_max_ringbuf();

    // Used if sophistication >= 2.
    std::vector<std::shared_ptr<ReferenceLagbuf>> residual_lagbufs;
    void _init_residual_lagbufs();
    void _apply_residual_lagbufs();

    // Used if sophistication == 3.
    // The number of _proper_*() methods is awkwardly large, but I wanted each method to roughly
    // correspond to a GPU kernel (or sequence of cudaMemcpyAsync() calls).
    
    std::vector<gputils::Array<float>> proper_ringbufs;
    std::vector<gputils::Array<float>> staging_inbufs;
    std::vector<gputils::Array<float>> staging_outbufs;

    void _init_proper_ringbufs();
    void _proper_ringbuf_to_staging();  // (oldest ringbuf entries) -> (staging_inbuf)
    void _proper_s0_to_staging();       // (stage0 iobufs) -> (staging outbuf), based on CacheLineRingbuf::PrimaryEntry::src_segment
    void _proper_staging_to_staging();  // (staging inbuf) -> (staging outbuf), based on CacheLineRingbuf::SecondaryEntry::src_*
    void _proper_staging_to_ringbuf();  // (staging outbuf) -> (newest ringbuf entries)
    void _proper_staging_to_s1();       // (staging inbuf) -> (stage1 iobufs), based on CacheLineRingbuf::*Entry::dst_segment
    void _proper_s0_to_s1();            // (stage0 iobufs) -> (stage1 iobufs), based on CacheLineRingbuf::stage0_stage1_copies
    void _proper_s1_to_s1();            // (stage1 iobufs) -> (stage1 iobufs), based on CacheLineRingbuf::stage0_stage1_copies

    // Helpers for print().
    ssize_t _print_array(const std::string &name, const gputils::Array<float> &arr, 
			 std::ostream &os, int indent, bool active_beams_only) const;
    
    ssize_t _print_ringbuf(const std::string &name,
			   const std::vector<gputils::Array<float>> &arr_vec, 
			   std::ostream &os, int indent, bool active_beams_only) const;
};


}  // namespace pirate

#endif // _PIRATE_INTERNALS_REFERENCE_DEDISPERSER_HPP
