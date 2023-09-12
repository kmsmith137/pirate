#ifndef _PIRATE_INTERNALS_REFERENCE_DEDISPERSER_HPP
#define _PIRATE_INTERNALS_REFERENCE_DEDISPERSER_HPP

#include <vector>
#include <memory>  // shared_ptr
#include <iostream>
#include <gputils/Array.hpp>

#include "../DedispersionConfig.hpp"
#include "../DedispersionPlan.hpp"


namespace pirate {
#if 0
}  // editor auto-indent
#endif

// Defined later in this file.
class ReferenceTree;
class ReferenceLagbuf;
class ReferenceReducer;


// -------------------------------------------------------------------------------------------------


struct ReferenceDedisperser
{
    // The ReferenceDedisperser has a 'sophistication' argument, which is one of {0,1,2,3}.
    //
    // sophistication=0:
    //   - downsampling implemented by throwing away bottom half of output
    //   - early triggers implemented by independent trees
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
    ssize_t pos = 0;  // counts cumulative calls to dedisperse()

    // The 'in' array represents one "chunk", with shape (2^input_rank, input_nt).
    // To process multiple chunks, call the dedipserse() method in a loop.
    void dedisperse(const gputils::Array<float> &in);
    
    void print(std::ostream &os=std::cout, int indent=0) const;

    // downsampled_inputs: only used if sophistication == 0.
    // downsampled_inputs[ids] has shape (2^input_rank, input_nt/2^ids), where 0 <= ids < nds.
    // It contains the input array after downsampling by a factor 2^ids.
    
    std::vector<gputils::Array<float>> downsampled_inputs;  // length nds
    void _allocate_downsampled_inputs();
    void _compute_downsampled_inputs(const gputils::Array<float> &in);

    // lagged_downsampled_inputs: used if sophistication >= 1.
    // It contains the input array after applying the ReferenceLaggedDownsampler.
    //
    // lagged_downsampled_inputs[ids] has shape:
    //   (2^input_rank, input_nt)            if ids == 0
    //   (2^(input_rank-1), input_nt/2^ids)  if ids > 0

    std::vector<gputils::Array<float>> lagged_downsampled_inputs;  // length nds
    void _allocate_lagged_downsampled_inputs();
    void _compute_lagged_downsampled_inputs(const gputils::Array<float> &in);

    // FIXME temporary kludge
    std::vector<std::shared_ptr<ReferenceReducer>> reducer_hack;

    // The "intermediate" arrays are the iobufs of the Stage0Trees.
    std::vector<gputils::Array<float>> intermediate_arrays;   // length nds
    gputils::Array<float> intermediate_flattened;
    void _allocate_intermediate_arrays();
    
    std::vector<gputils::Array<float>> output_arrays;  // length output_ntrees
    gputils::Array<float> output_flattened;
    void _allocate_output_arrays();

    
    // -------------------------------------------------------------------------------------------------

    
    struct SimpleTree
    {
	// SimpleTree: only used if sophistication==0.
	//
	//  - Uses one-stage dedispersion instead of two stages.
	//  - Assumes that caller has applied appropriate downsampling before calling dedisperse().
	//  - If tree is downsampled, then we compute twice as many DMs as necessary, then drop the bottom half.
	//  - Each early trigger is computed independently "from scratch", by disregarding some input channels.
	
	SimpleTree(const DedispersionPlan::Stage1Tree &st1);

	// Input array will be an element of this->downsampled_inputs.
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

    
    struct FirstTree
    {
	FirstTree(const DedispersionPlan::Stage0Tree &st0);

	// Input array will be an element of this->lagged_downsampled_inputs.
	// Output array will be an element of this->intermediate_arrays.
	
	void dedisperse(gputils::Array<float> &in, gputils::Array<float> &out);
	
	const bool is_downsampled;
	const int output_rank0;   // same as DedispersionPlan::Stage0Tree::rank0
	const int output_rank1;   // same as DedispersionPlan::Stage0Tree::rank1
	const int nt_ds;
	    
	std::shared_ptr<ReferenceTree> rtree;       // rank = output_rank0
	gputils::Array<float> rstate;
	gputils::Array<float> scratch;
    };


    struct SecondTree
    {
	SecondTree(const DedispersionPlan::Stage1Tree &st1);

	// Array argument will be an element of this->output_arrays.
	void dedisperse(gputils::Array<float> &arr);

	const int rank0;
	const int rank1;
	const int nt_ds;
	
	std::shared_ptr<ReferenceTree> rtree;
	gputils::Array<float> rstate;
	gputils::Array<float> scratch;
    };
    
    // Used if sophistication == 0.
    std::vector<SimpleTree> simple_trees;
    void _init_simple_trees();
    void _apply_simple_trees();

    // Used if sophistication > 0.
    std::vector<FirstTree> first_trees;
    void _init_first_trees();
    void _apply_first_trees();
    
    // Used if sophistication > 0.
    std::vector<SecondTree> second_trees;
    void _init_second_trees();
    void _apply_second_trees();

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


// -------------------------------------------------------------------------------------------------
//
// Helper classes: ReferenceTree, ReferenceLagbuf, ReferenceReducer


class ReferenceTree
{
    // ReferenceTree: simple, self-contained reference implementation of tree dedispersion.
    // Processes input incrementally in chunks of shape (2^rank, ntime).
    //
    // The RefrerenceTree is unaware of the larger dedispersion plan (stage0/stage1 split,
    // early triggers, downsampling, etc.) but can be used as a "building block" to implement
    // these features.
    
public:
    ReferenceTree(int rank, int ntime);

    int rank = 0;
    int ntime = 0;
    int nrstate = 0;
    int nscratch = 0;

    // 2-d array of shape (2^rank, ntime).
    // Dedispersion is done in place -- output index is a bit-reversed delay.
    void dedisperse(gputils::Array<float> &arr, float *rstate, float *scratch) const; 
    void dedisperse(float *arr, int stride, float *rstate, float *scratch) const;

protected:
    std::shared_ptr<ReferenceTree> prev_tree;
    std::vector<int> lags;  // length 2^(rank-1)
};



class ReferenceLagbuf
{
public:
    // ReferenceLagbuf: a very simple class which applies a channel-dependent lag
    // (specified by a length-nchan integer-valued vector of lags) incrementally to
    // an input array of shape (nchan, ntime).
    
    ReferenceLagbuf(const std::vector<int> &lags, int ntime);

    int nchan = 0; // lags.size()
    int ntime = 0;
    int nrstate = 0;

    // 2-d array of shape (nchan, ntime).
    // Lags are applied in place.
    void apply_lags(gputils::Array<float> &arr) const;
    void apply_lags(float *arr, int stride) const;

protected:
    std::vector<int> lags;  // length nchan
    gputils::Array<float> rstate;
    gputils::Array<float> scratch;
};


// FIXME write comment explaining the ReferenceReducer!

class ReferenceReducer
{
public:
    ReferenceReducer(int rank0_out, int rank1, int ntime);

    int rank0_out = 0;
    int rank1 = 0;
    int ntime = 0;
    int nrstate = 0;

    // Warning: modifies input array!!
    void reduce(gputils::Array<float> &in, gputils::Array<float> &out) const;

protected:
    std::shared_ptr<ReferenceLagbuf> lagbuf0; // applied to input array before freq downsampling (all lags 0 or 1)
    std::shared_ptr<ReferenceLagbuf> lagbuf1; // applied to input array after freq downsampling (all lags < 2^rank0_out)
};


// -------------------------------------------------------------------------------------------------
//
// Helper functions


// Downsamples (freq,time) array by a factor 2 along either frequency or time axis.
// Each pair of elements will be averaged/summed, depending on whether the 'normalize' flag is true/false.
extern void reference_downsample_freq(const gputils::Array<float> &in, gputils::Array<float> &out, bool normalize);
extern void reference_downsample_time(const gputils::Array<float> &in, gputils::Array<float> &out, bool normalize);

// Reduces (dm_brev, time) array by a factor 2, by keeping only odd (dm_brev)-indices.
// FIXME if I ever implement Array<float>::slice() with strides, then this would be a special case.
extern void reference_extract_odd_channels(const gputils::Array<float> &in, gputils::Array<float> &out);

// lag_non_incremental() is only used for testing the ReferenceLagbuf.
// Lagging is done in place.
extern void lag_non_incremental(gputils::Array<float> &arr, const std::vector<int> &lags);

// dedisperse_non_incremental() is only used for testing the ReferenceTree.
// Dedispersion is done in place -- output index is a bit-reversed delay.
extern void dedisperse_non_incremental(gputils::Array<float> &arr);
				       

}  // namespace pirate

#endif // _PIRATE_INTERNALS_REFERENCE_DEDISPERSER_HPP
