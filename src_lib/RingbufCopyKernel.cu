#include "../include/pirate/RingbufCopyKernel.hpp" 
#include "../include/pirate/constants.hpp"  // bytes_per_gpu_cache_line
#include "../include/pirate/inlines.hpp"    // xdiv()

#include <ksgpu/xassert.hpp>

using namespace std;
using namespace ksgpu;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// -------------------------------------------------------------------------------------------------


const RingbufCopyKernelParams &RingbufCopyKernelParams::validate() const
{
    xassert_gt(total_beams, 0);
    xassert_gt(beams_per_batch, 0);
    xassert(locations.on_host());
    
    // Currently assumed throughout the pirate code.
    xassert_divisible(total_beams, beams_per_batch);

    // Locations array can either be size-zero, or shape-(2N,4) contiguous.
    if (locations.size != 0) {
	xassert_eq(locations.ndim, 2);
	xassert_eq(locations.shape[1], 4);
	xassert_divisible(locations.shape[0], 2);
	xassert(locations.is_fully_contiguous());
    }

    return *this;
}


// -------------------------------------------------------------------------------------------------


CpuRingbufCopyKernel::CpuRingbufCopyKernel(const RingbufCopyKernelParams &params_) :
    params(params_.validate()),
    nlocations(xdiv(params_.locations.size, 8))
{
    long nbytes_per_location = (2 * params.beams_per_batch * constants::bytes_per_gpu_cache_line) + 8;
    bw_per_launch.nbytes_hmem = nlocations * nbytes_per_location;
}


void CpuRingbufCopyKernel::apply(ksgpu::Array<void> &ringbuf, long ibatch, long it_chunk)
{
    xassert(ringbuf.on_host());
    xassert(ringbuf.ndim == 1);
    xassert(ringbuf.is_fully_contiguous());
    
    xassert(ibatch >= 0);
    xassert(ibatch * params.beams_per_batch < params.total_beams);
    xassert(it_chunk >= 0);
    
    ulong irb = (it_chunk * params.total_beams) + (ibatch * params.beams_per_batch);   // ulong is important here
    char *rp = reinterpret_cast<char *> (ringbuf.data);
    const uint *lp = this->params.locations.data;
    
    for (long i = 0; i < nlocations; i++) {
	uint src_offset = lp[8*i];     // in segments, not bytes
	uint src_phase = lp[8*i+1];    // index of (time chunk, beam) pair, relative to current pair
	uint src_len = lp[8*i+2];      // number of (time chunk, beam) pairs in ringbuf (same as Ringbuf::rb_len)
	uint src_nseg = lp[8*i+3];     // number of segments per (time chunk, beam) (same as Ringbuf::nseg_per_beam)
	
	uint dst_offset = lp[8*i+4];
	uint dst_phase = lp[8*i+5];
	uint dst_len = lp[8*i+6];
	uint dst_nseg = lp[8*i+7];

	// Absorb irb into phases. (Note that 'irb' has ulong type.)
	src_phase = (ulong(src_phase) + irb) % ulong(src_len);
	dst_phase = (ulong(dst_phase) + irb) % ulong(dst_len);

	for (int j = 0; j < params.beams_per_batch; j++) {
	    ulong s = ulong(src_offset + src_phase * src_nseg) * constants::bytes_per_gpu_cache_line;
	    ulong d = ulong(dst_offset + dst_phase * dst_nseg) * constants::bytes_per_gpu_cache_line;

	    // FIXME is memmove() fastest here?
	    memmove(rp+d, rp+s, constants::bytes_per_gpu_cache_line);
	    
	    // Equivalent to (phase = (phase+1) % len), but avoids cost of %-operator.
	    src_phase = (src_phase == src_len-1) ? 0 : (src_phase+1);
	    dst_phase = (dst_phase == dst_len-1) ? 0 : (dst_phase+1);
	}
    }
}


}  // namespace pirate
