#include "../include/pirate/internals/DedispersionKernel.hpp"
#include "../include/pirate/internals/inlines.hpp"   // pow2()
#include "../include/pirate/constants.hpp"

#include <ksgpu/Dtype.hpp>
#include <ksgpu/xassert.hpp>

using namespace std;
using namespace ksgpu;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


void DedispersionKernelParams::validate(bool on_gpu) const
{
    xassert_ge(dd_rank, 0);
    xassert_le(dd_rank, 8);
    xassert_ge(amb_rank, 0);
    xassert_le(amb_rank, 8);
    xassert_gt(total_beams, 0);
    xassert_gt(beams_per_batch, 0);
    xassert_le(beams_per_batch, constants::cuda_max_y_blocks);
    xassert_ge(ntime, 0);

    xassert((dtype == Dtype::native<float>()) || (dtype == Dtype::native<__half>()));

    // (ringbuf -> ringbuf) doesn't make sense.
    xassert(!input_is_ringbuf || !output_is_ringbuf);
    
    // Currently assumed throughout the pirate code.
    xassert_divisible(total_beams, beams_per_batch);

    // Currently assumed by the GPU kernels.
    int nelts_per_cache_line = xdiv(8 * constants::bytes_per_gpu_cache_line, dtype.nbits);
    xassert_eq(nelts_per_segment, nelts_per_cache_line);
    xassert_divisible(ntime, nelts_per_segment);
    
    if (input_is_ringbuf || output_is_ringbuf) {
	long nsegments_per_tree = pow2(dd_rank+amb_rank) * xdiv(ntime,nelts_per_segment);
	xassert_shape_eq(ringbuf_locations, ({ nsegments_per_tree, 4 }));
	xassert(ringbuf_locations.is_fully_contiguous());
	xassert(ringbuf_nseg > 0);
	xassert(ringbuf_nseg <= UINT_MAX);

	if (on_gpu) {
	    xassert(ringbuf_locations.on_gpu());
	    return;
	}

	xassert(ringbuf_locations.on_host());
	
	for (long iseg = 0; iseg < nsegments_per_tree; iseg++) {
	    const uint *rb_locs = ringbuf_locations.data + (4*iseg);
	    long rb_offset = rb_locs[0];  // in segments, not bytes
	    // long rb_phase = rb_locs[1];   // index of (time chunk, beam) pair, relative to current pair
	    long rb_len = rb_locs[2];     // number of (time chunk, beam) pairs in ringbuf (same as Ringbuf::rb_len)
	    long rb_nseg = rb_locs[3];    // number of segments per (time chunk, beam) (same as Ringbuf::nseg_per_beam)
	    xassert(rb_offset + (rb_len-1)*rb_nseg < ringbuf_nseg);
	}
    }
}


}  // namespace pirate
