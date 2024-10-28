#ifndef _PIRATE_DEDISPERSION_PLAN_HPP
#define _PIRATE_DEDISPERSION_PLAN_HPP

#include "DedispersionConfig.hpp"

#include <vector>
#include <memory>  // shared_ptr
#include <gputils/Array.hpp>


namespace pirate {
#if 0
}  // editor auto-indent
#endif


struct DedispersionPlan
{
    DedispersionPlan(const DedispersionConfig &config);
    
    // --------------------  Helper classes  --------------------
    
    struct Stage0Tree
    {
	int ds_level = -1;  // downsampling level (downsampling "factor" is 2^level)
	int rank0 = 0;      // rank of Stage0Tree
	int rank1 = 0;      // rank of subsequent Stage1Tree (if no early trigger)
	int nt_ds = 0;      // number of time samples per chunk (after downsampling)

	int num_stage1_trees = 0;        // number of associated Stage1Trees (= num_early_triggers + 1)
	int stage1_base_tree_index = 0;  // base Stage1Tree index, within this->stage1_trees.
	
	int segments_per_row = 0;    // equal to (nt_ds / nelts_per_segment)
	int segments_per_beam = 0;   // equal to pow2(rank0+rank1) * segments_per_row
        int iobuf_base_segment = 0;  // base segment index, within single-beam stage0_iobuf

	// "rstate" = dedispersion state kept in registers, stored persistently on GPU between chunks
	// Initialized in _init_footprints().
	ssize_t rstate_nbytes_per_beam = 0;   // FIXME placeholder value -- improve
    };

    struct Stage1Tree
    {
	int ds_level = -1;       // Same as Stage0Tree::ds_level
	int rank0 = 0;           // Same as Stage0Tree::rank0
	int rank1_ambient = 0;   // Same as Stage0Tree::rank1
	int rank1_trigger = 0;   // Can be smaller than rank1_ambient, for early trigger
	int nt_ds = 0;           // Same as Stage0Tree::nt_ds
		
	int segments_per_row = 0;    // equal to (nt_ds / nelts_per_segment)
	int segments_per_beam = 0;   // equal to pow2(rank0 + rank1_trigger) * segments_per_row
        int iobuf_base_segment = 0;  // base segment index, within single-beam stage1_iobuf

	// For each row of the tree, we compute a lag (in time samples), then split the lag
	// into a 'segment_lag' (slag) and 'residual_lag' (rlag) as:
	//
	//    lag = N*slag + rlag     (where N = nelts_per_segment, and 0 <= rlag < N)
	
	gputils::Array<int> segment_lags;   // Length 2^rank, initialized in _init_lags()
	gputils::Array<int> residual_lags;  // Length 2^rank, initialized in _init_lags()
		
	// "rstate" = dedispersion state kept in registers, stored persistently on GPU between chunks.
	// Initialized in _init_footprints().
	
	// FIXME approximate value used for bandwidth analysis -- either improve or move elsewhere.
	ssize_t rstate_nbytes_per_beam = 0;    // FIXME placeholder value -- improve
    };

    struct Ringbuf
    {
	long rb_len = 0;           // number of (time chunk, beam) pairs
	long nseg_per_beam = 0;
	long base_segment = 0;
    };
    
    // --------------------  Members  --------------------

    
    const DedispersionConfig config;

    // Initialized at beginning of constructor.
    int nelts_per_segment = 0;            // constants::bytes_per_segment / sizeof(uncompressed_dtype)
    int uncompressed_dtype_size = 0;      // sizeof(uncompressed_type)
    int bytes_per_compressed_segment = 0; // nontrivial (e.g. 66 if uncompressed=float16 and compressed=int8)
    // Note: no 'bytes_per_uncompressed_segment' (use constants::bytes_per_segment).
    
    // Initialized in _init_trees() and _init_lags().
    std::vector<Stage0Tree> stage0_trees;
    std::vector<Stage1Tree> stage1_trees;

    // Iniitialized in _init_trees().
    ssize_t stage0_iobuf_segments_per_beam = 0;
    ssize_t stage1_iobuf_segments_per_beam = 0;

    // Everything after this is initialized in _init_ring_buffers().

    int max_clag = 0;
    long gmem_ringbuf_nseg = 0;

    // All vector<Ringbuf> objects have length (max_clag + 1).
    // T = total beams, A = active beams, B = beams per batch.
    
    std::vector<Ringbuf> gmem_ringbufs;    // rb_size = (clag*T + A), on GPU
    std::vector<Ringbuf> g2h_ringbufs;     // rb_size = min(A+B, T), on GPU
    std::vector<Ringbuf> h2g_ringbufs;     // rb_size = min(A+B, T), on GPU
    std::vector<Ringbuf> h2h_ringbufs;     // rb_size = (clag*T + B), on host

    // stage0_output_rb_locs, stage1_input_rb_locs.
    //
    // These arrays contain GPU ringbuf locations, represented as 4 uint32s:
    //  uint rb_offset;  // in segments, not bytes
    //  uint rb_phase;   // index of (time chunk, beam) pair, relative to current pair
    //  uint rb_len;     // number of (time chunk, beam) pairs in ringbuf (same as Ringbuf::rb_len)
    //  uint rb_nseg;    // number of segments per (time chunk, beam) (same as Ringbuf::nseg_per_beam)
    //
    // The arrays are indexed by:
    //  iseg0 -> (time/nelts_per_segment, 2^rank1, 2^rank0)
    //  iseg1 -> (time/nelts_per_segment, 2^rank0, 2^rank1)   note transpose

    gputils::Array<uint> stage0_rb_locs;   // shape (stage0_iobuf_segments_per_beam, 4)
    gputils::Array<uint> stage1_rb_locs;   // shape (stage1_iobuf_segments_per_beam, 4)


    // --------------------  Methods  --------------------
    
    // Helper functions called by constructor
    void _init_trees();
    void _init_lags();
    void _init_ring_buffers();
    
    void print(std::ostream &os=std::cout, int indent=0) const;
};


}  // namespace pirate

#endif // _PIRATE_DEDISPERSION_PLAN_HPP
