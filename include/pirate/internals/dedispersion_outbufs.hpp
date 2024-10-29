#ifndef _PIRATE_INTERNALS_DEDISPERSION_OUTBUFS_HPP
#define _PIRATE_INTERNALS_DEDISPERSION_OUTBUFS_HPP

#include "inlines.hpp"  // simd32_type
#include <gputils/Array.hpp>

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

    struct host_args
    {
	gputils::Array<T> out;
    };

    struct device_args
    {
	T32 *out;
	long beam_stride;     // 32-bit stride
	long ambient_stride;  // 32-bit stride
	long dm_stride;     // 32-bit stride
    };

    struct device_state
    {
	T32 *out;
	long dm_stride;
	
	__device__ __forceinline__ device_state(const device_args &args)
	{
	    const int ambient_ix = blockIdx.x;
	    const int beam_ix = blockIdx.y;
	    dm_stride = args.dm_stride;
	    
	    // Apply (beam, ambient) strides to iobuf. (Note laneId shift)
	    out = args.out;
	    out += beam_ix * args.beam_stride;
	    out += ambient_ix * args.ambient_stride;
	    out += (threadIdx.x >> 5) * dm_stride;
	    out += (threadIdx.x & 0x1f);  // laneId
	}

	__device__ __forceinline__ void store(int freq, T32 x)
	{
	    out[freq * dm_stride] = x;
	}

	__device__ __forceinline__ void advance()
	{
	    out += 32;
	}
    };
};


}  // namespace pirate

#endif // _PIRATE_INTERNALS_DEDISPERSION_OUTBUFS_HPP
