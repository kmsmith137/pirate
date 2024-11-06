#ifndef _PIRATE_INTERNALS_DEDISPERSION_INBUFS_HPP
#define _PIRATE_INTERNALS_DEDISPERSION_INBUFS_HPP

#include "inlines.hpp"                // simd32_type
#include "GpuDedispersionKernel.hpp"  // UntypedArray, GpuDedispersionKernel::Params

namespace pirate {
#if 0
}   // pacify editor auto-indent
#endif


// T = either float or __half
template<typename T, bool Lagged>
struct dedispersion_simple_inbuf
{
    static constexpr bool is_lagged = Lagged;
    
    // If T==float, then T32 is also 'float'.
    // If T==__half, then T32 is '__half2'.
    using T32 = typename simd32_type<T>::type;

    struct device_args
    {
	const T32 *in;
	long beam_stride32;     // 32-bit stride
	long ambient_stride32;  // 32-bit stride
	long freq_stride32;     // 32-bit stride
	bool is_downsampled;    // only used if Lagged=True.
	
	__device__ __forceinline__ bool _is_downsampled() { return is_downsampled; }

	// Defined in GpuDedispersionKernel.cu
	__host__ device_args(const UntypedArray &uarr, const GpuDedispersionKernel::Params &params);
    };

    struct device_state
    {
	const T32 *in;
	long freq_stride32;
	
	__device__ __forceinline__ device_state(const device_args &args, int freqs_per_warp, int active_rank, long rb_pos)
	{
	    const int ambient_ix = blockIdx.x;
	    const int beam_ix = blockIdx.y;
	    freq_stride32 = args.freq_stride32;
	    
	    // Apply (beam, ambient) strides to iobuf. (Note laneId shift)
	    in = args.in;
	    in += beam_ix * args.beam_stride32;
	    in += ambient_ix * args.ambient_stride32;
	    in += (threadIdx.x >> 5) * freqs_per_warp * freq_stride32;
	    in += (threadIdx.x & 0x1f);  // laneId
	}

	__device__ __forceinline__ T32 load(int freq)
	{
	    return in[freq * freq_stride32];
	}

	__device__ __forceinline__ void advance()
	{
	    in += 32;
	}
    };
};


// T = either float or __half.
// Note: currently taking Lagged=true, rather than having a Lagged template argument.

template<typename T>
struct dedispersion_ring_inbuf
{
    static constexpr bool is_lagged = true;
        
    // If T==float, then T32 is also 'float'.
    // If T==__half, then T32 is '__half2'.
    using T32 = typename simd32_type<T>::type;

    struct device_args
    {
	const T32 *rb_base;
        const uint4 *rb_loc;  // indexed by (seg, ambient, freq)
	bool is_downsampled;

	__device__ __forceinline__ bool _is_downsampled() { return is_downsampled; }

	// Defined in GpuDedispersionKernel.cu
	__host__ device_args(const UntypedArray &uarr, const GpuDedispersionKernel::Params &params);
    };

    struct device_state
    {
	const T32 *rb_base;
	const uint4 *rb_loc;     // indexed (seg, ambient, freq)
	uint rb_loc_seg_stride;  // uint4 stride
	long rb_pos;

	__device__ __forceinline__ device_state(const device_args &args, int freqs_per_warp, int active_rank, long rb_pos)
        {
	    const uint ambient_ix = blockIdx.x;
	    const uint nambient = gridDim.x;
	    const uint beam_ix = blockIdx.y;
	    
	    this->rb_base = args.rb_base;
	    this->rb_loc = args.rb_loc + (ambient_ix << active_rank) + freqs_per_warp;
	    this->rb_loc_seg_stride = (nambient << active_rank);
	    this->rb_pos = rb_pos + beam_ix;
	}

	__device__ __forceinline__ T32 load(int freq)
        {
	    // FIXME super inefficient -- repeats same computation on all warps!
	    
	    uint4 q = rb_loc[freq];
	    uint rb_offset = q.x;  // in segments, not bytes
	    uint rb_phase = q.y;   // index of (time chunk, beam) pair, relative to current pair
	    uint rb_len = q.z;     // number of (time chunk, beam) pairs in ringbuf (same as Ringbuf::rb_len)
	    uint rb_nseg = q.w;    // number of segments per (time chunk, beam)

	    uint p = (rb_pos + rb_phase) % rb_len;
	    uint s = rb_offset + (p * rb_nseg);
	    long i = (long(s) << 5) + (threadIdx.x & 0x1f);

	    return rb_base[i];
        }

        __device__ __forceinline__ void advance()
        {
	    rb_loc += rb_loc_seg_stride;
        }
    };
};


}  // namespace pirate

#endif // _PIRATE_INTERNALS_DEDISPERSION_INBUFS_HPP
