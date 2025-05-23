#include "../include/pirate/RingbufCopyKernel.hpp" 
#include "../include/pirate/constants.hpp"  // bytes_per_gpu_cache_line
#include "../include/pirate/inlines.hpp"    // xdiv()

#include <ksgpu/xassert.hpp>
#include <ksgpu/cuda_utils.hpp>
#include <ksgpu/device_transposes.hpp>   // FULL_MASK

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

    // Currently this is all we need.
    xassert((nelts_per_segment == 32) || (nelts_per_segment == 64));
    
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
    // Note: only correct if (nbytes_per_segment == constants::bytes_per_gpu_cache_line).
    long nbytes_per_location = (2 * params.beams_per_batch * constants::bytes_per_gpu_cache_line) + 8;
    bw_per_launch.nbytes_hmem = nlocations * nbytes_per_location;
}


// Helper for CpuRingbufCopyKernel::apply()
// B = bytes per segment

template<int B>
static void _cpu_copy(void *ringbuf, const uint *locations, long nlocations, int nbeams, ulong iframe)
{
    char *rp = reinterpret_cast<char *> (ringbuf);
    
    for (long i = 0; i < nlocations; i++) {
	uint src_offset = locations[8*i];     // in segments, not bytes
	uint src_phase = locations[8*i+1];    // index of (time chunk, beam) pair, relative to current pair
	uint src_len = locations[8*i+2];      // number of (time chunk, beam) pairs in ringbuf (same as Ringbuf::rb_len)
	uint src_nseg = locations[8*i+3];     // number of segments per (time chunk, beam) (same as Ringbuf::nseg_per_beam)
	
	uint dst_offset = locations[8*i+4];
	uint dst_phase = locations[8*i+5];
	uint dst_len = locations[8*i+6];
	uint dst_nseg = locations[8*i+7];

	// Absorb iframe into phases. (Note that 'iframe' has ulong type.)
	src_phase = (ulong(src_phase) + iframe) % ulong(src_len);
	dst_phase = (ulong(dst_phase) + iframe) % ulong(dst_len);

	for (int j = 0; j < nbeams; j++) {
	    ulong s = ulong(src_offset + src_phase * src_nseg) * B;
	    ulong d = ulong(dst_offset + dst_phase * dst_nseg) * B;

	    // FIXME is memmove() fastest here?
	    memmove(rp + d, rp + s, B);
	    
	    // Equivalent to (phase = (phase+1) % len), but avoids cost of %-operator.
	    src_phase = (src_phase == src_len-1) ? 0 : (src_phase+1);
	    dst_phase = (dst_phase == dst_len-1) ? 0 : (dst_phase+1);
	}
    }
}


void CpuRingbufCopyKernel::apply(ksgpu::Array<void> &ringbuf, long ibatch, long it_chunk)
{
    xassert(ringbuf.on_host());
    xassert(ringbuf.ndim == 1);
    xassert(ringbuf.is_fully_contiguous());
    
    xassert(ibatch >= 0);
    xassert(ibatch * params.beams_per_batch < params.total_beams);
    xassert(it_chunk >= 0);
    
    ulong iframe = (it_chunk * params.total_beams) + (ibatch * params.beams_per_batch);
    long nbits_per_segment = params.nelts_per_segment * ringbuf.dtype.nbits;

    // These two cases are all we currently need.
    if (nbits_per_segment == 1024)
	_cpu_copy<128> (ringbuf.data, params.locations.data, nlocations, params.beams_per_batch, iframe);
    else if (nbits_per_segment == 2048)
	_cpu_copy<256> (ringbuf.data, params.locations.data, nlocations, params.beams_per_batch, iframe);
    else {
	stringstream ss;
	ss << "CpuRingbufCopyKernel: expected nbits_per_segment in {1024,2048}, got "
	   << params.nelts_per_segment << " (nelts_per_segment=" << params.nelts_per_segment
	   << ", dtype=" << ringbuf.dtype << ")";
	throw runtime_error(ss.str());
    }
}


// -------------------------------------------------------------------------------------------------


GpuRingbufCopyKernel::GpuRingbufCopyKernel(const RingbufCopyKernelParams &params_) :
    params(params_.validate()),
    nlocations(xdiv(params_.locations.size, 8))
{
    long nbytes_per_location = (2 * params.beams_per_batch * constants::bytes_per_gpu_cache_line) + 8;
    bw_per_launch.nbytes_gmem = nlocations * nbytes_per_location;
}


void GpuRingbufCopyKernel::allocate()
{
    if (is_allocated)
	throw runtime_error("double call to GpuRingbufCopyKernel::allocate()");

    // Copy host -> GPU.
    this->gpu_locations = params.locations.to_gpu();    
    this->is_allocated = true;
}


// Thread grid: { 32*W, 1, 1 }.
// Block grid: { B, 1, 1 }.

__global__ void gpu_copy_kernel(uint4 *ringbuf, const uint *locations, long nlocations, int nbeams, ulong iframe)
{
    // Global thread ID
    long tid = long(blockIdx.x) * long(blockDim.x) + threadIdx.x;

    // "Regulated" thread ID (avoids out-of-range)
    long treg = ((nlocations-1) << 3) + (threadIdx.x & 0x7);
    treg = min(tid, treg);

    // Each warp reads 4 locations (i.e. 4 src+dst pairs).
    uint loc_data = locations[treg];

    // Absorb iframe into phases (only valid on laneId=1 mod 4).
    // Note: currently using 9 __shfl_syncs in this function, optimal number is 7.
    uint loc_len = __shfl_sync(FULL_MASK, loc_data, (threadIdx.x & 0x1c) + 2);
    uint loc_phase = (ulong(loc_data) + iframe) % ulong(loc_len);

    // "Allgather" src/dst offsets in groups of 8.
    uint src_offset = __shfl_sync(FULL_MASK, loc_data, (threadIdx.x & 0x18));
    uint src_phase = __shfl_sync(FULL_MASK, loc_phase, (threadIdx.x & 0x18) + 1);   // Note loc_phase here
    uint src_len = __shfl_sync(FULL_MASK, loc_data, (threadIdx.x & 0x18) + 2);
    uint src_nseg = __shfl_sync(FULL_MASK, loc_data, (threadIdx.x & 0x18) + 3);

    uint dst_offset = __shfl_sync(FULL_MASK, loc_data, (threadIdx.x & 0x18) + 4);
    uint dst_phase = __shfl_sync(FULL_MASK, loc_phase, (threadIdx.x & 0x18) + 5);   // Note loc_phase here
    uint dst_len = __shfl_sync(FULL_MASK, loc_data, (threadIdx.x & 0x18) + 6);
    uint dst_nseg = __shfl_sync(FULL_MASK, loc_data, (threadIdx.x & 0x18) + 7);
    
    for (int b = 0; b < nbeams; b++) {
	ulong s = ulong(src_offset + src_phase * src_nseg) << 3;   // int4 offset
	ulong d = ulong(dst_offset + dst_phase * dst_nseg) << 3;   // int4 offset
	uint4 rb_data = ringbuf[s + (threadIdx.x & 0x7)];
	
	if (tid == treg)
	    ringbuf[d + (threadIdx.x & 0x7)] = rb_data;

	// Equivalent to (phase = (phase+1) % len), but avoids cost of %-operator.
	src_phase = (src_phase == src_len-1) ? 0 : (src_phase+1);
	dst_phase = (dst_phase == dst_len-1) ? 0 : (dst_phase+1);
    }
}


void GpuRingbufCopyKernel::launch(ksgpu::Array<void> &ringbuf, long ibatch, long it_chunk, cudaStream_t stream)
{
    xassert(ringbuf.on_gpu());
    xassert(ringbuf.ndim == 1);
    xassert(ringbuf.is_fully_contiguous());

    xassert(ibatch >= 0);
    xassert(ibatch * params.beams_per_batch < params.total_beams);
    xassert(it_chunk >= 0);
    
    long nbits_per_segment = params.nelts_per_segment * ringbuf.dtype.nbits;
    xassert_eq(nbits_per_segment, 1024);  // currently assuemd in gpu kernel
    xassert(this->is_allocated);

    if (nlocations == 0)
	return;
    
    int W = 4;
    int B = (nlocations + 4*W - 1) / (4*W);  // each block does 4*W locations
    ulong iframe = (it_chunk * params.total_beams) + (ibatch * params.beams_per_batch);
    
    gpu_copy_kernel<<< B, 32*W, 0, stream >>> (
	reinterpret_cast<uint4 *> (ringbuf.data),   // uint4 *ringbuf
	gpu_locations.data,                         // const uint *locations
	nlocations,                                 // long nlocations
	params.beams_per_batch,                     // int nbeams
	iframe                                      // ulong iframe
    );

    CUDA_PEEK("gpu_copy_kernel");
}


}  // namespace pirate
