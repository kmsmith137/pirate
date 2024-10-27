#ifndef _PIRATE_INTERNALS_CACHE_LINE_RINGBUF_HPP
#define _PIRATE_INTERNALS_CACHE_LINE_RINGBUF_HPP

#include <vector>
#include <iostream>

#include "../DedispersionConfig.hpp"

namespace pirate {
#if 0
}  // editor auto-indent
#endif

// Defined in ./LaggedCacheLinehpp
struct LaggedCacheLine;


// Ring Buffer Memory layout
//
//   for each rb_lag:
//      for time in range(rb_lag):
//          for each beam (active or inactive):
//	        primary entries
//              secondary entries
//
// Staging buffer Memory layout
//
//   for each rb_lag:
//       for direction in [ 'h2g', 'g2h' ]:
//           for each active beam:
//	         primary entries
//               secondary entries


struct CacheLineRingbuf
{
    struct ConstructorArgs {
	ssize_t num_src_segments = 0;   // FIXME currently unused, may use later to test coverage
	ssize_t num_dst_segments = 0;   // FIXME currently unused, may use later to test coverage
	ssize_t gmem_nbytes_used_so_far = 0;   // bytes used so far (dedispersion iobufs and rstate bufs)
    };
    
    const DedispersionConfig config;
    const ConstructorArgs cargs;

    // Initialized in finalize().
    ssize_t gmem_nbytes_staging_buf = 0;   // GPU staging buffer, including factor 'active_beams_per_gpu'
    ssize_t gmem_nbytes_ringbuf = 0;       // GPU ring buffer, including factor 'total_beams_per_gpu'
    ssize_t hmem_nbytes_ringbuf = 0;       // host ring buffer, including factor 'total_beams_per_gpu'
    ssize_t pcie_nbytes_per_chunk = 0;     // host <-> GPU bandwidth per chunk (EACH WAY), including factor 'total_beams_per_gpu'
    ssize_t pcie_memcopies_per_chunk = 0;  // host <-> GPU memcpy call count per chunk (EACH WAY)

    
    CacheLineRingbuf(const DedispersionConfig &config, const ConstructorArgs &cargs);
    
    void add(const LaggedCacheLine &cl);
    void finalize();
    
    void print(std::ostream &os=std::cout, int indent=0) const;
    

    // -------------------------------------------------------------------------------------------------
    //
    // The CacheLineRingbuf consists of:
    //
    //   - a list of stage0 -> stage1 copies
    //   - a list of stage1 -> stage1 copies
    //   - for each rb_lag, one Buffer, which contains:
    //      - list of PrimaryEntries, representing ring buffer inputs from the stage0 iobuf
    //      - list of SecondaryEntries, representing ring buffer inputs from the output of another ring buffer
    

    struct PrimaryEntry
    {
	int src_segment = -1;
	int dst_segment = -1;
    };

    struct SecondaryEntry
    {
	int src_rb_lag = -1;
	int src_rb_index = -1;
	bool src_is_primary = true;
	int dst_segment = -1;
    };
    
    struct Buffer
    {
	std::vector<PrimaryEntry> primary_entries;
	std::vector<SecondaryEntry> secondary_entries;

	// Fields below are initialized in CacheLineRingbuf::finalize().

	// These fields do not include the factor 2 from [ 'h2g', 'g2h' ].
	// (This factor of 2 is appropriate for the staging buffer, but not the ring buffer.)
	ssize_t primary_nbytes_per_beam_per_chunk = 0;    // cache-aligned
	ssize_t secondary_nbytes_per_beam_per_chunk = 0;  // cache-aligned
	ssize_t total_nbytes_per_beam_per_chunk = 0;      // sum of prev two fields

	ssize_t total_ringbuf_nbytes = 0;         // = (rb_lag * total_beams_per_gpu * total_nbytes_per_beam_per_chunk)
	ssize_t total_staging_buffer_nbytes = 0;  // = (2 * active_beams_per_gpu * total_nbytes_per_beam_per_chunk)
	ssize_t staging_buffer_byte_offset = -1;
	
	bool on_gpu = false;
	ssize_t gmem_ringbuf_byte_offset = -1;   // only >= 0 if on_gpu
	ssize_t hmem_ringbuf_byte_offset = -1;   // only >= 0 if !on_gpu
	ssize_t pcie_xfer_size = 0;              // only nonzero if !on_gpu
	//   ... more to come
    };
    
    std::vector<Buffer> buffers;   // indexed by rb_lag (entry 0 is empty placeholder)
    std::vector<PrimaryEntry> stage0_stage1_copies;
    std::vector<PrimaryEntry> stage1_stage1_copies;

    bool finalized = false;
};


}  // namespace pirate

#endif // _PIRATE_INTERNALS_CACHE_LINE_RINGBUF_HPP
