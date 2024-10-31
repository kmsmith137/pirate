#ifndef _PIRATE_INTERNALS_DEDISPERSION_INBUFS_HPP
#define _PIRATE_INTERNALS_DEDISPERSION_INBUFS_HPP

#include "inlines.hpp"  // simd32_type
#include "GpuDedispersionKernel.hpp"

namespace pirate {
#if 0
}   // pacify editor auto-indent
#endif


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
    };

    struct device_state
    {
	const T32 *in;
	long freq_stride32;
	
	__device__ __forceinline__ device_state(const device_args &args, int freqs_per_warp)
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

    struct host_args
    {
	device_args kernel_args;
	long nbeams;
	long ntime;

	__host__ host_args(const UntypedArray &Uarr, const GpuDedispersionKernel::Params &params);
    };
};


}  // namespace pirate

#endif // _PIRATE_INTERNALS_DEDISPERSION_INBUFS_HPP
