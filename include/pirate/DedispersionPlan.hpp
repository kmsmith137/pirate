#ifndef _PIRATE_DEDISPERSION_PLAN_HPP
#define _PIRATE_DEDISPERSION_PLAN_HPP

#include "DedispersionConfig.hpp"

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
    
    void print(std::ostream &os=std::cout, int indent=0) const;
    
    // --------------------  Helper classes  --------------------
    
    struct Stage0Tree
    {
	// Note: total tree rank (rank0 + rank1) is equal to (config.tree_rank - (ds_level ? 1 : 0)).
	
	int ds_level = -1;  // downsampling level (downsampling "factor" is 2^level)
	int rank0 = 0;      // rank of Stage0Tree
	int rank1 = 0;      // rank of subsequent Stage1Tree (if no early trigger)
	int nt_ds = 0;      // downsampled time samples per chunk (= config.time_samples_per_chunk / pow2(ds_level))
	
	int segments_per_beam = 0;   // equal to pow2(rank0+rank1) * (nt_ds / nelts_per_segment)
        int base_segment = 0;        // cumulative (over all Stage0Trees) segment count
    };

    struct Stage1Tree
    {
	int ds_level = -1;       // Same as Stage0Tree::ds_level
	int rank0 = 0;           // Same as Stage0Tree::rank0
	int rank1_ambient = 0;   // Same as Stage0Tree::rank1
	int rank1_trigger = 0;   // Can be smaller than rank1_ambient, for early trigger
	int nt_ds = 0;           // Same as Stage0Tree::nt_ds
		
	int segments_per_beam = 0;   // equal to pow2(rank0 + rank1_trigger) * (nt_ds / nelts_per_segment)
        int base_segment = 0;        // cumulative (over all Stage1Trees) segment count
    };

    struct Ringbuf
    {
	long rb_len = 0;           // number of (time chunk, beam) pairs
	long nseg_per_beam = 0;
	long base_segment = 0;
    };
    
    // --------------------  Members  --------------------
    
    const DedispersionConfig config;

    int nelts_per_segment = 0;   // currently always constants::bytes_per_gpu_cache_line / (sizeof config dtype)
    int nbytes_per_segment = 0;  // currently always constants::bytes_per_gpu_cache_line
    
    std::vector<Stage0Tree> stage0_trees;
    std::vector<Stage1Tree> stage1_trees;

    ssize_t stage0_total_segments_per_beam = 0;
    ssize_t stage1_total_segments_per_beam = 0;

    int max_clag = 0;
    long gmem_ringbuf_nseg = 0;    // includes gmem + g2h + h2g

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

    ksgpu::Array<uint> stage0_rb_locs;   // shape (stage0_total_segments_per_beam, 4)
    ksgpu::Array<uint> stage1_rb_locs;   // shape (stage1_total_segments_per_beam, 4)
};


}  // namespace pirate

#endif // _PIRATE_DEDISPERSION_PLAN_HPP
