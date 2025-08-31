#include "../include/pirate/DedispersionKernel.hpp"
#include "../include/pirate/constants.hpp"
#include "../include/pirate/inlines.hpp"   // pow2()

#include <ksgpu/Dtype.hpp>
#include <ksgpu/xassert.hpp>

using namespace std;
using namespace ksgpu;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


void DedispersionKernelParams::validate() const
{
    xassert_ge(dd_rank, 0);
    xassert_le(dd_rank, 8);
    xassert_ge(amb_rank, 0);
    xassert_le(amb_rank, 8);
    xassert_gt(total_beams, 0);
    xassert_gt(beams_per_batch, 0);
    xassert_le(beams_per_batch, constants::cuda_max_y_blocks);
    xassert_ge(nt_per_segment, 0);
    xassert_ge(nspec, 0);
    xassert_ge(ntime, 0);

    xassert((dtype == Dtype::native<float>()) || (dtype == Dtype::native<__half>()));
    xassert_divisible(ntime, nt_per_segment);

    // (ringbuf -> ringbuf) doesn't make sense.
    xassert(!input_is_ringbuf || !output_is_ringbuf);
    
    // Currently assumed throughout the pirate code.
    xassert_divisible(total_beams, beams_per_batch);
    
    if (input_is_ringbuf || output_is_ringbuf) {
	long nsegments_per_tree = pow2(dd_rank+amb_rank) * xdiv(ntime,nt_per_segment);
	xassert_shape_eq(ringbuf_locations, ({ nsegments_per_tree, 4 }));
	xassert(ringbuf_locations.is_fully_contiguous());
	xassert(ringbuf_nseg > 0);
	xassert(ringbuf_nseg <= UINT_MAX);
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


DedispersionKernelIobuf::DedispersionKernelIobuf(const DedispersionKernelParams &params, const Array<void> &arr, bool is_ringbuf_, bool on_gpu_)
{
    this->buf = arr.data;
    this->is_ringbuf = is_ringbuf_;
    this->on_gpu = on_gpu_;

    // Note: we don't call params.validate(), since it can be a "heavyweight" operation
    // (since ringbuf_locations are checked), and the DedispersionKernelIobuf constructor
    // needs to be "lightweight" (since called on every kernel launch).
    
    xassert_eq(arr.dtype, params.dtype);
    
    // Check alignment. Not strictly necessary, but failure would be unintentional and indicate a bug somewhere.
    xassert(is_aligned(buf, constants::bytes_per_gpu_cache_line));   // also checks non_NULL
    
    if (on_gpu)
	xassert(arr.on_gpu());
    else
	xassert(arr.on_host());

    // FIXME constructor should include overflow checks on strides.
    // (Check on act_stride is nontrivial, since it gets multiplied by a small integer in the kernel.)
    
    if (is_ringbuf) {
	// Case 1: ringbuf, 1-d array of length (ringbuf_nseg * nt_per_segment * nspec).
	xassert_shape_eq(arr, ({ params.ringbuf_nseg * params.nt_per_segment * params.nspec }));
	xassert(arr.get_ncontig() == 1);  // fully contiguous
	return;
    }
    
    // Case 2: simple buf. Shape is either:
    //     (beams_per_batch, pow2(amb_rank), pow2(dd_rank), ntime, nspec)
    //  or (beams_per_batch, pow2(amb_rank), pow2(dd_rank), ntime)  if nspec==1

    long B = params.beams_per_batch;
    long A = pow2(params.amb_rank);
    long N = pow2(params.dd_rank);
    long T = params.ntime;
    long S = params.nspec;

    bool valid1 = arr.shape_equals({B,A,N,T,S});
    bool valid2 = (S==1) && arr.shape_equals({B,A,N,T});

    // FIXME should compose helpful error message if this assert fails.
    xassert(valid1 || valid2);

    // Valid for both shapes.
    xassert(arr.get_ncontig() >= arr.ndim-3);

    long denom = xdiv(32, arr.dtype.nbits);
    this->beam_stride32 = xdiv(arr.strides[0], denom);   // 32-bit stride
    this->amb_stride32 = xdiv(arr.strides[1], denom);    // 32-bit stride
    this->act_stride32 = xdiv(arr.strides[2], denom);    // 32-bit stride
    
    // FIXME could improve these checks, by verifying that strides are non-overlapping.
    xassert((params.beams_per_batch == 1) || (beam_stride32 != 0));
    xassert((params.amb_rank == 0) || (amb_stride32 != 0));
    xassert((params.dd_rank == 0) || (act_stride32 != 0));

    if (on_gpu) {
	// Check alignment. Not strictly necessary, but failure would be unintentional and indicate a bug somewhere.
	xassert_divisible(beam_stride32 * 4, constants::bytes_per_gpu_cache_line);
	xassert_divisible(amb_stride32 * 4, constants::bytes_per_gpu_cache_line);
	xassert_divisible(act_stride32 * 4, constants::bytes_per_gpu_cache_line);
    }
}


}  // namespace pirate
