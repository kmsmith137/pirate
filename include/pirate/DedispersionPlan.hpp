#ifndef _PIRATE_DEDISPERSION_PLAN_HPP
#define _PIRATE_DEDISPERSION_PLAN_HPP

#include "DedispersionConfig.hpp"
#include "DedispersionBuffer.hpp"        // struct DedispersionBufferParams
#include "DedispersionKernel.hpp"        // struct DedispersionKernelParams
#include "LaggedDownsamplingKernel.hpp"  // struct LaggedDownsamplingKernelParams
#include "RingbufCopyKernel.hpp"         // struct RingbufCopyKernelParams

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

    long stage1_ntrees = 0;  // same as config.num_downsampling_levels
    long stage2_ntrees = 0;
    
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
	
	int segments_per_beam = 0;   // equal to pow2(rank0+rank1) * (nt_ds / nelts_per_segment)
        int base_segment = 0;        // cumulative (over all Stage1Trees) segment count
    };

    struct Stage2Tree
    {
	int ds_level = -1;       // Same as Stage1Tree::ds_level
	int rank0 = 0;           // Same as Stage1Tree::rank0
	int rank1_ambient = 0;   // Same as Stage1Tree::rank1
	int rank1_trigger = 0;   // Can be smaller than rank1_ambient, for early trigger
	int nt_ds = 0;           // Same as Stage1Tree::nt_ds
		
	int segments_per_beam = 0;   // equal to pow2(rank0 + rank1_trigger) * (nt_ds / nelts_per_segment)
        int base_segment = 0;        // cumulative (over all Stage2Trees) segment count
    };

    struct Ringbuf
    {
	long rb_len = 0;           // number of (time chunk, beam) pairs, see below
	long nseg_per_beam = 0;    // size (in segments) per (time chunk, beam) pair
	long base_segment = -1;    // offset (in segments) relative to base memory address on either GPU or CPU
    };
    
    // --------------------  Members  --------------------

    int nelts_per_segment = 0;   // currently always constants::bytes_per_gpu_cache_line / (sizeof config dtype)
    int nbytes_per_segment = 0;  // currently always constants::bytes_per_gpu_cache_line
    
    std::vector<Stage1Tree> stage1_trees;  // length stage1_ntrees
    std::vector<Stage2Tree> stage2_trees;  // length stage2_ntrees

    long stage1_total_segments_per_beam = 0;
    long stage2_total_segments_per_beam = 0;

    int max_clag = 0;
    int max_gpu_clag = 0;
    
    long gmem_ringbuf_nseg = 0;    // total size (gpu_ringbufs + xfer_ringbufs + et_gpu_ringbuf)
    long hmem_ringbuf_nseg = 0;    // total size (host_ringbufs + et_host_ringbuf)

    // All vector<Ringbuf> objects have length (max_clag + 1).
    // BT = total beams, BA = active beams, BB = beams per batch.
    
    std::vector<Ringbuf> gpu_ringbufs;    // rb_len = (clag*BT + BA), on GPU
    std::vector<Ringbuf> host_ringbufs;   // rb_len = (clag*BT + BA), on host
    std::vector<Ringbuf> xfer_ringbufs;   // rb_len = (2*BA), on GPU

    // If early triggers are used, need one more pair of buffers.
    Ringbuf et_host_ringbuf;  // rb_len = 2*BA, on host (send buffer)
    Ringbuf et_gpu_ringbuf;   // rb_len = 2*BA, on GPU (recv buffer)

    // stage1_output_rb_locs, stage2_input_rb_locs: used in dedispersion kernels.
    //
    // These arrays contain GPU ringbuf locations, represented as four uint32s:
    //  uint rb_offset;  // in segments, not bytes
    //  uint rb_phase;   // index of (time chunk, beam) pair, relative to current pair
    //  uint rb_len;     // number of (time chunk, beam) pairs in ringbuf (same as Ringbuf::rb_len)
    //  uint rb_nseg;    // number of segments per (time chunk, beam) (same as Ringbuf::nseg_per_beam)
    //
    // The arrays are indexed by:
    //  iseg0 -> (time/nelts_per_segment, 2^rank1, 2^rank0)
    //  iseg1 -> (time/nelts_per_segment, 2^rank0, 2^rank1)   note transpose

    ksgpu::Array<uint> stage1_rb_locs;   // shape (stage1_total_segments_per_beam, 4)
    ksgpu::Array<uint> stage2_rb_locs;   // shape (stage2_total_segments_per_beam, 4)

    // Only needed if early triggers are used.    
    ksgpu::Array<uint> g2g_rb_locs;      // copy from gpu_ringbufs to xfer_ringbufs
    ksgpu::Array<uint> h2h_rb_locs;      // copy from host_ringbufs to et_host_ringbuf
};


}  // namespace pirate

#endif // _PIRATE_DEDISPERSION_PLAN_HPP
