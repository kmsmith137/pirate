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
	
	__device__ __forceinline__ device_state(const device_args &args)
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

	__device__ __forceinline__ void store(int freq, T32 x)
	{
	    out[freq * dm_stride32] = x;
	}

	__device__ __forceinline__ void advance()
	{
	    out += 32;
	}
    };
};


}  // namespace pirate

#endif // _PIRATE_INTERNALS_DEDISPERSION_OUTBUFS_HPP
