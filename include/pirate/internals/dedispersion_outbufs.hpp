#ifndef _PIRATE_INTERNALS_DEDISPERSION_OUTBUFS_HPP
#define _PIRATE_INTERNALS_DEDISPERSION_OUTBUFS_HPP

#include "inlines.hpp"                // simd32_type
#include "GpuDedispersionKernel.hpp"  // UntypedArray, GpuDedispersionKernel::Params

namespace pirate {
#if 0
}   // pacify editor auto-indent
#endif


// T = either float or __half
template<typename T>
struct dedispersion_simple_outbuf
{
    // If T==float, then T32 is also 'float'.
    // If T==__half, then T32 is '__half2'.
    using T32 = typename simd32_type<T>::type;

    struct device_args
    {
	T32 *out;
	long beam_stride32;     // 32-bit stride
	long ambient_stride32;  // 32-bit stride
	long dm_stride32;       // 32-bit stride
	
	// Defined in GpuDedispersionKernel.cu
	__host__ device_args(const UntypedArray &uarr, const GpuDedispersionKernel::Params &params);
    };

    struct device_state
    {
	T32 *out;
	long dm_stride32;
	
	__device__ __forceinline__ device_state(const device_args &args, int active_rank, long rb_pos)
	{
	    const int ambient_ix = blockIdx.x;
	    const int beam_ix = blockIdx.y;
	    dm_stride32 = args.dm_stride32;
	    
	    // Apply (beam, ambient) strides to iobuf. (Note laneId shift)
	    out = args.out;
	    out += beam_ix * args.beam_stride32;
	    out += ambient_ix * args.ambient_stride32;
	    out += (threadIdx.x >> 5) * dm_stride32;
	    out += (threadIdx.x & 0x1f);  // laneId
	}

	__device__ __forceinline__ void store(int dm, T32 x)
	{
	    out[dm * dm_stride32] = x;
	}

	__device__ __forceinline__ void advance()
	{
	    out += 32;
	}
    };
};


// T = either float or __half
template<typename T>
struct dedispersion_ring_outbuf
{
    // If T==float, then T32 is also 'float'.
    // If T==__half, then T32 is '__half2'.
    using T32 = typename simd32_type<T>::type;

    struct device_args
    {
	T32 *rb_base;
	const uint4 *rb_loc;  // indexed by (seg, ambient, dm)
	
	// Defined in GpuDedispersionKernel.cu
	__host__ device_args(const UntypedArray &uarr, const GpuDedispersionKernel::Params &params);
    };

    struct device_state
    {
	T32 *rb_base;
	const uint4 *rb_loc;     // indexed by (seg, ambient, dm)
	uint rb_loc_seg_stride;  // uint4 stride
	long rb_pos;
	
	__device__ __forceinline__ device_state(const device_args &args, int active_rank, long rb_pos)
	{
	    const uint ambient_ix = blockIdx.x;
	    const uint nambient = gridDim.x;
	    const uint beam_ix = blockIdx.y;
	    
	    this->rb_base = args.rb_base;
	    this->rb_loc = args.rb_loc + (ambient_ix << active_rank);
	    this->rb_loc_seg_stride = (nambient << active_rank);
	    this->rb_pos = rb_pos + beam_ix;
	}

	__device__ __forceinline__ void store(int dm, T32 x)
	{
	    // FIXME super inefficient -- repeats same computation on all warps!
	    
	    uint4 q = rb_loc[dm];
	    uint rb_offset = q.x;  // in segments, not bytes
	    uint rb_phase = q.y;   // index of (time chunk, beam) pair, relative to current pair
	    uint rb_len = q.z;     // number of (time chunk, beam) pairs in ringbuf (same as Ringbuf::rb_len)
	    uint rb_nseg = q.w;    // number of segments per (time chunk, beam)

	    uint p = (rb_pos + rb_phase) % rb_len;
	    uint s = rb_offset + (p * rb_nseg);
	    long i = (long(s) << 5) + (threadIdx.x & 0x1f);
	    
	    rb_base[i] = x;
	}

	__device__ __forceinline__ void advance()
	{
	    rb_loc += rb_loc_seg_stride;
	}
    };
};


}  // namespace pirate

#endif // _PIRATE_INTERNALS_DEDISPERSION_OUTBUFS_HPP
