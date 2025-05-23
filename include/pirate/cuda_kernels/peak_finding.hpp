#ifndef _PIRATE_CUDA_KERNELS_PEAK_FINDING_HPP
#define _PIRATE_CUDA_KERNELS_PEAK_FINDING_HPP

#include <ksgpu/Array.hpp>
#include <ksgpu/constexpr_functions.hpp>
#include <ksgpu/device_transposes.hpp>


namespace pirate {
#if 0
}   // pacify editor auto-indent
#endif


// -------------------------------------------------------------------------------------------------
//
// pf_ringbuf: represents a logical ring buffer, in registers on a single warp.
// Uses warp shuffles to advance ring buffer.
//
// I = "inner" ringbuf size
// O = "outer" ringbuf size


template<typename T32, int I_, int O_>
struct pf_ringbuf
{
    static constexpr int I = I_;
    static constexpr int O = O_;
    
    static_assert(I <= 32);
    static_assert(sizeof(T32) == 4);
    static_assert(ksgpu::constexpr_is_pow2(I));

    static constexpr int nelts_per_warp = I*O;            // total ring buffer size
    static constexpr int R = (nelts_per_warp + 31) >> 5;  // registers per thread
    T32 data[R];

    __device__ inline pf_ringbuf() { }

    // Advance ring buffer
    // Must call sequentially with J = 0, ..., (O-1)!
    
    template<int J>
    __device__ inline void advance(T32 &x)
    {
	static_assert((J >= 0) && (J < O));
	
	constexpr int IR = (I*J) >> 5;                 // ring buffer register index
	constexpr int S = (J < O-1) ? I : (32*R-J*I);  // phase shift at end
	
	T32 xrot = __shfl_sync(ksgpu::FULL_MASK, x, threadIdx.x + (32-I));

	bool low = (threadIdx.x & 0x1f) < I;
	x = low ? data[IR] : xrot;
	data[IR] = low ? xrot : data[IR];

	if constexpr (S != 32)
	    data[IR] = __shfl_sync(ksgpu::FULL_MASK, data[IR], threadIdx.x + S);
    }

    // Advance ring buffer M times, using x[0:M] and ignoring x[M:N].
    // To use the entire x-array, set M=N (which is the default).
    
    template<int J, int N, int M=N>
    __device__ inline void multi_advance(T32 x[N])
    {
	static_assert(M <= N);
	
	if constexpr (M > 0) {
	    this->template multi_advance<J, N, M-1> (x);
	    this->template advance<J+M-1> (x[M-1]);
	}
    }
    
    // Load ring buffer from memory (either global or shared).
    // The 'p_warp' argument should have per-warp offests applied, but
    // not per-thread offsets. The total number of elements (not bytes)
    // written is (pf_ringbuf::nelts_per_warp).
    
    __device__ inline void load(const T32 *p_warp)
    {
	#pragma unroll
	for (int r = 0; r < R; r++) {
	    int i = 32*r + (threadIdx.x & 0x1f);
	    data[r] = (i < nelts_per_warp) ? p_warp[i] : 0;
	}
    }
    
    // Write ring buffer to memory (either global or shared).
    // The 'p_warp' argument should have per-warp offests applied, but
    // not per-thread offsets. The total number of elements (not bytes)
    // written is (pf_ringbuf::nelts_per_warp).
    
    __device__ inline void store(T32 *p_warp)
    {
	#pragma unroll
	for (int r = 0; r < R; r++) {
	    int i = 32*r + (threadIdx.x & 0x1f);
	    if (i < nelts_per_warp)
		p_warp[i] = data[r];
	}
    }
};


}  // namespace pirate

#endif // _PIRATE_CUDA_KERNELS_PEAK_FINDING_HPP

