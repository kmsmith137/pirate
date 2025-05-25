#ifndef _PIRATE_CUDA_KERNELS_PEAK_FINDING_HPP
#define _PIRATE_CUDA_KERNELS_PEAK_FINDING_HPP

#include <ksgpu/Array.hpp>
#include <ksgpu/constexpr_functions.hpp>
#include <ksgpu/device_transposes.hpp>
#include <ksgpu/device_dtype_ops.hpp>

#include "../constants.hpp"   // constants::pf_a, constants::pfb

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
    
    __device__ inline void save(T32 *p_warp)
    {
	#pragma unroll
	for (int r = 0; r < R; r++) {
	    int i = 32*r + (threadIdx.x & 0x1f);
	    if (i < nelts_per_warp)
		p_warp[i] = data[r];
	}
    }
};


// -------------------------------------------------------------------------------------------------
//
// class pf_core
//
//  - Lowest-level part of the code, where peak-finding kernels are applied,
//    and pf_out, pf_ssq are computed. This is the only part of the code which
//    is "aware" of the kernel profiles.
//
//  - Coarse-grains in time but not DM.
//    Therefore, the output of pf_core::advance() is a lot of registers:
//
//       T32 pf_out[P];
//       T32 pf_ssq[P];
//
//  - Does not apply weights, or write outputs to global memory.
//
//  - Treats trial DMs as abstract spectator indices.
//    (Later, these abstract spectator indices may contain additional data,
//    such as sub-bands, spectral indices, etc.)
//
//  - Number of spectator indices, and register layout, is prescribed by
//    the pf_core. Caller is responsible for transposes and outer loops.
//
//  - The pf_core always uses Dt registers per thread, so that time
//    coarse-graining is a "thread-local" operation (except for "neighbors",
//    which are exchanged with __shfl_sync()).
//
//  - We almost always use the following register layout (where W is
//    the "simd_width": 1 for float32, or 2 for float16):
//
//      - Dt registers per thread (corresponding to consecutive times)
//      - ((32*W)/Dt "outer" thread indices) <-> (coarse-grained times)
//      - ((Dt/W) "inner" thread indices) <-> (spectators)
//      - (W simd lanes) <-> (spectators)
//
//  - Nuisance issue: this doesn't work in the special case (float16, Dt=1).
//    In this case we use the following register layout:
//
//      - 1 register per thread
//      - (32 thread indices) <-> (time samples)
//      - (2 simd lanes) <-> (spectator)


template<typename T32, int Dt, int E, int S>
struct pf_core
{
    static_assert(sizeof(T32) == 4);
    static_assert(ksgpu::constexpr_is_pow2(Dt));
    static_assert(ksgpu::constexpr_is_pow2(E));
    static_assert(Dt <= 16);
    static_assert(E <= Dt);

    // Tin = number of input time samples processed by one call to pf_core::advance().
    // Tout = number of output time samples, after dividing by Dt.
    
    static constexpr int W = ksgpu::dtype_ops<T32>::simd_width;
    static constexpr int Tin = (Dt > 1) ? (32*W) : 32;   // see above
    static constexpr int Tout = Tin / Dt;

    // SS = number of spectator indices assigned to simd lanes, per call to pf_core::advance().
    // ST = number of spectator indices assigned to threads, per call to pf_core::advance().
    // Souter = number of calls to pf_core::advance() needed to process all S spectator indices.
    
    static constexpr int SS = W;
    static constexpr int ST = (32/Tout);
    static constexpr int Souter = S / (SS*ST);

    static_assert(Tout * ST == 32);
    static_assert(SS * ST * Souter == S);

    // P = number of peak-finding kernels.
    //
    //   - If E=1: P=1 (single sample)
    //   - If E=2: P=4 (+ length-2 boxcar + length-3 Gaussian + length-4 Gaussian)
    //   - If E=4: P=7 (+ length-4 boxcar + length-6 Gaussian + length-12 Gaussian)
    //   - If E=8: P=10 (+ length-8 boxcar + length-12 Gaussian + length-16 Gaussian)
    //   - If E=16: P=13 (+ length-16 boxcar + length-24 Gaussian + length-32 Gaussian)

    static constexpr int P = 3*ksgpu::constexpr_ilog2(E) + 1;
    
    // NL, NR = number of left/right neighbors
    static constexpr int NL = ksgpu::constexpr_ilog2(E);
    static constexpr int NR = (E > 1) ? (NL+1) : 0;

    // Now we can declare the ring buffer.
    using Ringbuf = pf_ringbuf<T32, ST, Souter*(Dt+NL)>;
    Ringbuf ringbuf;

    // Ring buffer persistent state (per S spectator indices).
    static constexpr int pstate_n32_per_warp = Ringbuf::nelts_per_warp;
    static constexpr int pstate_nbytes_per_warp = 4 * pstate_n32_per_warp;

    // These registers are set by pf_core::advance().
    T32 pf_out[P];
    T32 pf_ssq[P];  // "sum of squares", i.e. variance before dividing by nsamples.

    const T32 a = constants::pf_a;
    const T32 b = constants::pf_b;


    void load_pstate(const T32 *p_warp)
    {
	ringbuf.load(p_warp);
    }

    void save_pstate(T32 *p_warp)
    {
	ringbuf.save(p_warp);
    }
    
    // Helper for advance().
    template<int N, bool Reverse>
    __device__ inline void _compute_neighbors(T32 x[Dt], T32 y[N])
    {
	constexpr int A = Reverse ? (Dt-1) : 0;  // index of initial register
	constexpr int B = Reverse ? -1 : 1;      // associated step
	
	static_assert(N <= 5);
	
	if constexpr (N >= 1)
	    y[0] = x[A];
	if constexpr (N >= 2)
	    y[1] = x[A+B];
	if constexpr (N >= 3)
	    y[2] = x[A+2*B] + x[A+3*B];
	if constexpr (N >= 4)
	    y[3] = x[A+4*B] + x[A+5*B] + x[A+6*B] + x[A+7*B];
	if constexpr (N >= 5)
	    y[4] = x[A+8*B] + x[A+9*B] + x[A+10*B] + x[A+11*B] + x[A+12*B] + x[A+13*B] + x[A+14*B] + x[A+15*B];
    }


    // Helper for _eval_all_kernels() and _eval_kernels_Emin().
    __device__ inline void _update_pf(T32 x, int d, T32 &out, T32 &ssq)
    {
	out = d ? max(out,x) : x;
	ssq = d ? (ssq+x*x) : (x*x);
    }
	
    // Helper for advance().
    // Fills pf_out[0] and pf_ssq[0] (single sample).
    
    __device__ inline void _eval_all_kernels(T32 x[Dt], T32 yl[NL], T32 yr[NR])
    {
	#pragma unroll
	for (int d = 0; d < Dt; d++)
	    _update_pf(x[d], d, pf_out[0], pf_ssq[0]);

	if constexpr (E >= 2) {
	    T32 xpad[Dt+3];

	    #pragma unroll
	    for (int d = 0; d < Dt; d++)
		xpad[d+1] = x[d];

	    xpad[0] = yl[0];
	    xpad[Dt+1] = yr[0];
	    xpad[Dt+2] = yr[1];
	    
	    this->_eval_kernels_Emin<2,Dt+3> (xpad, yl, yr);
	}
    }

    // Helper for advance().
    // Fills pf_out[] and pf_ssq[].
    // The template parameter Emin means "process all peak-finders >= Emin".
    
    template<int Emin, int D>
    __device__ inline void _eval_kernels_Emin(T32 x[D], T32 yl[NL], T32 yr[NR])
    {
	static_assert(Emin >= 2);
	static_assert(E >= Emin);
	static_assert(ksgpu::constexpr_is_pow2(Emin));

	constexpr int I = ksgpu::constexpr_ilog2(Emin);
	constexpr int D0 = (2*Dt)/Emin;
	static_assert(D == D0+3);

	#pragma unroll
	for (int d = 0; d < D0; d++) {
	    T32 b2 = (x[d+1] + x[d+2]);
	    T32 g3 = (x[d+1]) + a * (x[d] + x[d+1]);
	    T32 g4 = (x[d+1] + x[d+2]) + b * (x[d] + x[d+3]);

	    _update_pf(b2, d, pf_out[3*I-2], pf_ssq[3*I-2]);
	    _update_pf(g3, d, pf_out[3*I-1], pf_ssq[3*I-1]);
	    _update_pf(g4, d, pf_out[3*I], pf_ssq[3*I]);
	}
	
	// Call recursively with Emin -> (2 * Emin).
	
	if constexpr (E > Emin) {
	    constexpr int D1 = D0 >> 2;
	    
	    T32 xnext[D1+3];

	    #pragma unroll
	    for (int d = 0; d < D1+1; d++)
		xnext[d+1] = x[2*d+1] + x[2*d+2];
	    
	    xnext[0] = x[0] + yl[I];
	    xnext[D1+2] = yr[I+1];

	    this->_eval_kernels_Emin<2*Emin, D1+3> (xnext, yl, yr);
	}
    }

    // Call sequentially with J = 0, ..., Souter, before moving on to
    // the next time chunk.
    
    template<int J>
    __device__ inline void advance(T32 x[Dt])
    {
	// Compute right neighbors, before applying lag.
	T32 yr[NR];
	template _compute_neighbors<NR,false> (x,yr);   // reverse=false

	// Apply lag.
	ringbuf.template multi_advance<J*(Dt+NL), Dt> (x);

	// Compute left neighbors, after applying lag.
	// Note call to Ringbuf::multi_advance() here.
	T32 yl[NL];
	template _compute_neighbors<NL,true> (x,yl);   // reverse=true
	ringbuf.template multi_advance<J*(Dt+NL) + Dt, NL> (yl);

	this->_eval_all_kernels(x, yl, yr);
    }
};


// -------------------------------------------------------------------------------------------------
//
// class pf_tile: slightly higher level than pf_core
//
//  - Input is an array x[M], where indices 0 <= m < M represent trial
//    DMs, and threads/simd-lanes represent time. The pf_tile supplies
//    the transposes and outer loops needed by pf_core.
//
//  - Fully coarse-grains, applies weights, and writes results to GPU
//    global memory. Therefore, pf_tile::advance() returns void.
//
//  - Works with trial DMs, whereas 'pf_core' works with abstract spectator
//    indices. When we implement subbands, pf_core should stay the same,
//    whereas pf_tile may change.
//
//  - The value of M (number of trial DMs per warp) is supplied by the
//    caller, but there is a technical constraint that (M % Core::S) == 0.
//    (If this constraint is violated, then a static_assert will fail.)
//
//  - Designed for either "standalone" use in pf_kernel, or coalescing
//    with the dedispersion kernels.


#if 0

template<typename T32, int Dd, int Dt, int E, int M>
struct pf_tile
{
    static_assert((Dd==1) || (Dd==Dt));

    using Core = pf_core<T32,Dt,E,M>;
    Core core;


    __device__ inline void transpose(T32 x[M])
    {	
	// Case 1: float32
	// Transpose t0 <-> r0, t1 <-> r1, ..., t(L-1) <-> r(L-1), where L = log2(Dt)
	// Call Core::advance() in contiguous blocks of (M/Dt) registers

	// Case 2: float16, Dt > 1
	// Transpose b0 <-> r0, t0 <-> r1, ..., t(L-2) <-> r(L-1)
	// Call Core::advance() in contiguous blocks of (M/Dt) registers

	// Case 3: float16, Dt==1 (which implies Dd==1)
	// Group x's into consecutive pairs (representing consecutive trial DMs).
	// Do a special-case transpose
	// Call Core::advance() on even m's, then odd m's.
    }

    
    template<int J0=0>
    __device__ inline void _advance(T32 x[M])
    {
	if constexpr (J0 < Core::Souter) {
	    T32 x0[Dt];

	    #pragma unroll
	    for (int d = 0; d < Dt; d++)
		x0[d] = x[J0*Dt + d];

	    core.template advance(x0);

	    xxx;
	    // Apply weights
	    // Coarse-grain? Write?

	    this->template _advance<J0+1> (x);
	}
    }
    
    __device__ inline void advance(T32 x[M])
    {
	this->_transpose(x);

	// Two calls to _advance() in (float16 Dt=1) case.
	this->template _advance(x);
    }
};

#endif


}  // namespace pirate

#endif // _PIRATE_CUDA_KERNELS_PEAK_FINDING_HPP
