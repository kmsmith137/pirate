#ifndef _PIRATE_DEDISPERSION_PLAN_HPP
#define _PIRATE_DEDISPERSION_PLAN_HPP

#include "DedispersionConfig.hpp"

#include <vector>
#include <memory>  // shared_ptr
#include <gputils/Array.hpp>

#include "internals/LaggedCacheLine.hpp"

namespace pirate {
#if 0
}  // editor auto-indent
#endif

// Defined in pirate/internals/CacheLineRingbuf.hpp
struct CacheLineRingbuf;


struct DedispersionPlan
{
    DedispersionPlan(const DedispersionConfig &config);
    
    // --------------------  Helper classes  --------------------
    // FIXME int -> ssize_t in some places
    
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
	
	ssize_t rstate_nbytes_per_beam = 0;
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
	
	ssize_t rstate_nbytes_per_beam = 0;
    };

    
    // --------------------  Members  --------------------

    
    const DedispersionConfig config;

    // Initialized at beginning of constructor.
    int nelts_per_segment = 0;
    int uncompressed_dtype_size = 0;
    int bytes_per_compressed_segment = 0;
    // Note: no 'bytes_per_uncompressed_segment' (use constants::bytes_per_segment).
    
    // Initialized in _init_trees() and _init_lags().
    std::vector<Stage0Tree> stage0_trees;
    std::vector<Stage1Tree> stage1_trees;

    // Iniitialized in _init_trees().
    ssize_t stage0_iobuf_segments_per_beam = 0;
    ssize_t stage1_iobuf_segments_per_beam = 0;

    // Initialized in _init_ring_buffers().
    // Only present if DedispersionConfig::bloat_dedispersion_plan == true.
    std::vector<LaggedCacheLine> lagged_cache_lines;

    // Initialized in _init_ring_buffers().
    std::shared_ptr<CacheLineRingbuf> cache_line_ringbuf;

    // Initialized in _init_trees(), _init_rstate_footprints(), _init_ring_buffers().
    ssize_t gmem_nbytes_tot = 0;
    ssize_t gmem_nbytes_stage0_iobufs = 0;  // Core dedispersion array, including factor 'active_beams_per_gpu'
    ssize_t gmem_nbytes_stage1_iobufs = 0;  // Core dedispersion array, including factor 'active_beams_per_gpu'
    ssize_t gmem_nbytes_stage0_rstate = 0;  // Saved kernel register state, including factor 'total_beams_per_gpu'
    ssize_t gmem_nbytes_stage1_rstate = 0;  // Saved kernel register state, including factor 'total_beams_per_gpu'
    ssize_t gmem_nbytes_staging_buf = 0;    // GPU staging buffer, including factor 'active_beams_per_gpu'
    ssize_t gmem_nbytes_ringbuf = 0;        // GPU ring buffer, including factor 'total_beams_per_gpu'
    ssize_t hmem_nbytes_ringbuf = 0;        // host ring buffer, including factor 'total_beams_per_gpu'
    ssize_t pcie_nbytes_per_chunk = 0;      // host <-> GPU bandwidth per chunk (EACH WAY), including factor 'total_beams_per_gpu'
    ssize_t pcie_memcopies_per_chunk = 0;   // host <-> GPU memcopy call count per chunk (EACH WAY)

    // GPU global memory bandwidth, in bytes/chunk (not bytes/sec!), including factor 'total_beams_per_gpu'.
    struct {
	ssize_t init_stage0_ds0 = 0;        // FIXME underestimates, since input (4-bit?) array is not included
	ssize_t init_stage0_higher_ds = 0;
	ssize_t dedisperse_stage0_main = 0;
	ssize_t dedisperse_stage0_rstate = 0;
	ssize_t copy_stage0_to_staging = 0;
	ssize_t copy_staging_to_staging = 0;
	ssize_t copy_staging_to_stage1 = 0;
	ssize_t copy_staging_to_gmem_ringbuf = 0;
	ssize_t copy_staging_to_hmem_ringbuf = 0;
	ssize_t copy_gmem_ringbuf_to_staging = 0;
	ssize_t copy_hmem_ringbuf_to_staging = 0;
	ssize_t copy_stage0_to_stage1 = 0;
	ssize_t copy_stage1_to_stage1 = 0;
	ssize_t dedisperse_stage1_main = 0;
	ssize_t dedisperse_stage1_rstate = 0;
	ssize_t peak_finding = 0;           // FIXME underestimates, since output coarse-grained array is not included
    } gmem_bw_nbytes_per_chunk;
    
    // FIXME locator arrays
    //   stage0 -> compressed staging
    //   compressed staging -> gpu ring buffer
    //   compressed staging -> host ring buffer
    //   gpu ring buffer -> host ring buffer
    //   stage0 -> stage1 direct
    //   host ring buffer -> compressed staging
    //   gpu ring buffer -> compressed staging
    //   compressed staging -> stage1
    //   stage1 -> stage1 (subtle)
    
    // --------------------  Methods  --------------------
    
    // Helper functions called by constructor
    void _init_trees();
    void _init_lags();
    void _init_rstate_footprints();
    void _init_ring_buffers();
    
    void print(std::ostream &os=std::cout, int indent=0) const;
    void print_segment_info(std::ostream &os=std::cout, int indent=0) const;
    void print_trees(std::ostream &os=std::cout, int indent=0) const;
    void print_footprints(std::ostream &os=std::cout, int indent=0) const;
};


}  // namespace pirate

#endif // _PIRATE_DEDISPERSION_PLAN_HPP
