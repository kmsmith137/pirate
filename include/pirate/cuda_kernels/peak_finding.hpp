#ifndef _PIRATE_CUDA_KERNELS_PEAK_FINDING_HPP
#define _PIRATE_CUDA_KERNELS_PEAK_FINDING_HPP

#include <ksgpu/Array.hpp>
#include <ksgpu/device_basics.hpp>
#include <ksgpu/device_transposes.hpp>
#include <ksgpu/constexpr_functions.hpp>

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
//  - We almost always use the following register layout
//    (where L is the number of simd "lanes": 1 for float32, or 2 for float16)
//
//      - Dt registers per thread (corresponding to consecutive times)
//      - ((32*L)/Dt "outer" thread indices) <-> (coarse-grained times)
//      - ((Dt/L) "inner" thread indices) <-> (spectators)
//      - (L simd lanes) <-> (spectators)
//
//  - Nuisance issue: this doesn't work in the special case (float16, Dt=1).
//    In this case we use the following register layout:
//
//      - 1 register per thread
//      - (32 thread indices) <-> (time samples)
//      - (2 simd lanes) <-> (spectator)


template<typename T32, int Dt_, int E_, int S_>
struct pf_core
{
    static constexpr int Dt = Dt_;
    static constexpr int E = E_;
    static constexpr int S = S_;
    
    static_assert(sizeof(T32) == 4);
    static_assert(ksgpu::constexpr_is_pow2(Dt));
    static_assert(ksgpu::constexpr_is_pow2(E));
    static_assert(Dt <= 16);
    static_assert(E <= Dt);

    // Tin = number of input time samples processed by one call to pf_core::advance().
    // Tout = number of output time samples, after dividing by Dt.
    
    static constexpr int L = ksgpu::dtype_ops<T32>::simd_width;
    static constexpr int Tin = (Dt > 1) ? (32*L) : 32;   // see above
    static constexpr int Tout = Tin / Dt;

    // SS = number of spectator indices assigned to simd lanes, per call to pf_core::advance().
    // ST = number of spectator indices assigned to threads, per call to pf_core::advance().
    // Souter = number of calls to pf_core::advance() needed to process all S spectator indices.
    
    static constexpr int SS = L;
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


    __device__ inline void load_pstate(const T32 *p_warp)
    {
	ringbuf.load(p_warp);
    }

    __device__ inline void save_pstate(T32 *p_warp)
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
    
    __device__ inline void _eval_all_kernels(T32 x[Dt], ksgpu::RegisterArray<T32,NL> &yl, ksgpu::RegisterArray<T32,NR> &yr)
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
    __device__ inline void _eval_kernels_Emin(T32 x[D], ksgpu::RegisterArray<T32,NL> &yl, ksgpu::RegisterArray<T32,NR> &yr)
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
	ksgpu::RegisterArray<T32,NR> yr;
	if constexpr (NR > 0)
	    this->template _compute_neighbors<NR,false> (x, yr.data);   // reverse=false

	// Apply lag.
	ringbuf.template multi_advance<J*(Dt+NL), Dt> (x);

	// Compute left neighbors, after applying lag.
	// Note call to Ringbuf::multi_advance() here.
	ksgpu::RegisterArray<T32,NL> yl;
	if constexpr (NL > 0) {
	    this->template _compute_neighbors<NL,true> (x, yl.data);   // reverse=true
	    ringbuf.template multi_advance<J*(Dt+NL) + Dt, NL> (yl.data);
	}

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


template<typename T32, int Dd, int Dt, int E, int M>
struct pf_tile
{
    // For now!
    static_assert(Dd == 1);
    static_assert(M >= 2);

    using Core = pf_core<T32,Dt,E,M>;

    static constexpr int P = Core::P;
    static constexpr int L = ksgpu::dtype_ops<T32>::simd_width;
    static constexpr int pstate_n32_per_warp = Core::pstate_n32_per_warp;
    static constexpr int weights_n32_per_warp = M * P;
    
    Core core;

    T32 *out_th;
    T32 *ssq_th;
    T32 *weights_th;

    long out_mcore_stride;   // stride per call to pf_core::advance()
    long out_pstride;
    int wt_pstride;

    // The 'out' and 'ssq' arrays have shape (P,M/Dd,Tout).
    // The 'pstate' array is a 1-d array of length (pstate_n32_per_warp), and may be in shared memory.
    // The 'weights' array has shape (P,M), and should be in shared memory (otherwise kernel will be slow).
    
    __device__ inline pf_tile(T32 *out_warp, T32 *ssq_warp, T32 *pstate_warp, T32 *weights_warp, long out_pstride_, long out_mstride_, int wt_pstride_)
    {
	constexpr int ST = Core::ST;

	// These per-thread pointer offsets must be kept in sync with _apply_weights_and_advance().
	int st = threadIdx.x & (ST-1);
	int mout = (st << 1) + (threadIdx.x & ST);
	out_th = out_warp + (mout * out_mstride_);
	ssq_th = ssq_warp + (mout * out_mstride_);
	weights_th = weights_warp + st;
	
	out_mcore_stride = (Core::ST * Core::SS) * out_mstride_;   // stride per call to pf_core::advance().
	out_pstride = out_pstride_;
	wt_pstride = wt_pstride_;

	core.load(pstate_warp);
    }
    
    // _do_warp_transposes(): morally equivalent to the following (but written
    // in a weird way, since "for constexpr" doesn't actually exist):
    //
    //   for constexpr (RS = RegisterStride0; RS < RegisterStride1; RS *= 2) {
    //       ksgpu::multi_warp_transpose<RS> (x, thread_stride0);
    //       thread_stride0 *= 2;
    //   }

    template<int RegisterStride0, int RegisterStride1>
    __device__ inline void _do_warp_transposes(T32 x[M], int thread_stride0)
    {
	if constexpr (RegisterStride0 < RegisterStride1) {
	    ksgpu::multi_warp_transpose<RegisterStride0> (x, thread_stride0);
	    this->template _do_warp_transposes<2*RegisterStride0, RegisterStride1> (x, 2*thread_stride0);
	}
    }

    
    // _apply_weights_and_write(): called after pf_core::advance().
    //
    // Setup: we've processed (Core::ST * Core::SS DMs) x (Core::Tin times)
    // and have computed:
    //
    //   T32 core.pf_out[P];
    //   T32 core.pf_ssq[P];
    //
    // We downsample in DM (if Dd > 1), apply weights, and write to
    // GPU global memory.
    
    __device__ inline void _apply_weights_and_write(int souter)
    {
	// For now, assume no downsampling in DM.
	static_assert(Dd == 1);

	#pragma unroll
	for (int p = 0; p < P; p++) {
	    // Reminder: weights array has shape (P,M).
	    // Note that per-thread pointer offset has already been applied in constructor.
	    int wix = (souter * Core::ST) + (p * wt_pstride);
	    T32 w = weights_th[wix];

	    out_pf[p] *= w;
	    out_ssq[p] *= (w*w);
	    
	    // Case 1: float32 (where "T" is a downsampled time)
	    //   t0 t1 t2 t3 t4 <-> m0 ... m(K-1) T0 ... T(4-K)
	    //
	    // Case 2: float16
	    //    b0 t1 t2 t3 t4 <-> m0 m1 ... m(K) T0 ... T(4-K)

	    if constexpr (L > 1) {  // float16
		// b0 t0 t1 t2 t3 t4 <-> m0 m1 ... mJ T0 ... TK
		ksgpu::warp_and_half2_transpose(core.out_pf[p], core.out_ssq[p], Core::ST);
	    }
	    
	    // Partial cache line writes -- slow!

	    long ix = (souter * out_mcore_stride) + (p * out_pstride);
	    out_th[ix] = out_pf[p];
	    ssq_th[ix] = out_ssq[p];
	}

	out_th += Core::Tout;
	out_ssq += Core::Tout;
    }

    
    // _advance(): fully processes (M DMs) x (Core::Tin times), where
    //
    //     Core::Tin = (Dt > 1) ? (32*L) : 32
    //
    // The register assignment is awkward to write down! We split trial DMs into:
    //   - (Dt) "inner" DMs, denoted m0 ... m(K-1) where K=log2(Dt)
    //   - (M/Dt) "outer" DMs, denoted s0 ... 
    //
    // Then the register assignment is:
    //
    //  - Case 1: float32
    //      r0 ... r(K-1) t0 ... t4 <-> T0 ... T(K-1) m0 ... m(K-1) T(K) ... T4
    // 
    //  - Case 2: float16 and Dt > 1
    //      r0 ... r(K-1) b0 t0 ... t4 <-> T0 ... T(K-1) m0 m1 ... m(K-1) T(K) ... T5
    //
    //  - Case 3: float16 and Dt==1
    //     r0 b0 t0 t1 t2 t3 t4 <-> UNUSED s0 T0 T1 T2 T3 T4
    //
    // where registers not shown map to "outer" spectator indices. 
    //
    // The (X0, XStride) template arguments select a subset of the length-M register
    // array x[X0::XStride]. In cases 1+2, we take X0=0 and XStride=1, i.e. select the
    // entire array. In case 3, we take XStride=2 and X0 = (either 0 or 1), to select
    // half the array (corresponding to the UNUSED r0 bit above).
    //
    // The (X0, XStride) arguments are also used to implement "for constexpr" logic
    // which calls pf_core::advance() multiple times (number of calls is Core::Souter).
    
    template<int X0, int XStride>
    __device__ inline void _advance(T32 x[M])
    {
	// Consistency check: if we make Core::Souter calls to pf_core::advance(),
	// and each call advances the register array by (Core::Dt * XStride) indices,
	// then we should precisely consume M registers.
	
	static_assert(Core::Dt * Core::Souter * XStride == M);

	if constexpr (X0 < M) {
	    static_assert(X0 + (Dt-1)*XStride < M);
	    
	    T32 xslice[Dt];

	    #pragma unroll
	    for (int d = 0; d < Dt; d++)
		xslice[d] = x[d*XStride + X0];

	    constexpr int souter = X0 / (Dt*XStride);
	    core.template advance<souter> (xslice);
	    
	    this->_apply_weights_and_write(souter);
	    
	    this->template _advance<XStride, X0 + Dt*XStride> (x);
	}
    }


    // advance(): fully processes (M trial DMs) x (32L times), in the
    // "natural" register assignmnent, where registers represent DMs,
    // and threads/simd-lanes represent times.
    
    __device__ inline void advance(T32 x[M])
    {
	// When denoting register assignments, we split the trial DMs into:
	//   - (Dt) "inner" DMs, denoted m0 ... m(K-1) where K=log2(Dt)
	//   - (M/Dt) "outer" DMs, denoted s0 ... 
	//
	// The initial register assignment is (where not all spectators are
	// shown):
	//
	//  - Case 1: float32
	//      r0 ... r(K-1) t0 ... t4 <-> m0 ... m(K-1) T0 ... T4
	// 
	//  - Case 2: float16 and (Dt > 1)
	//      r0 ... r(K-1) b0 t0 ... t4 <-> m0 ... m(K-1) T0 T1 ... T5
	//
	//  - Case 3: float16 and (Dt == 1)	
	//      r0 b0 t0 ... t4 <-> s0 T0 T1 ... T5

	// If float16, exchange r0 and b0.
	if constexpr (L > 1)
	    ksgpu::multi_half2_transpose<1> (x);

	// If float32, exchange (t0,...,t(K-1)) with (r0,...,r(K-1)).
	// If float16, exchange (t0,...,t(K-2)) with (r1,...,r(K-1)).
	this->_do_warp_transposes<L,Dt> (x, 1);
       
	if constexpr (Core::Tin == 32*L) {

	    // Case 1: float32
	    //    r0 ... r(K-1) t0 ... t4 <-> T0 ... T(K-1) m0 ... m(K-1) T(K) ... T4
	    // 
	    // Case 2: float16 and Dt > 1
	    //    r0 ... r(K-1) b0 t0 ... t4 <-> T0 ... T(K-1) m0 m1 ... m(K-1) T(K) ... T5
	    //
	    // We have the correct register assignment for _advance().
	    
	    this->template _advance<0,1> (x);
	}
	else {

	    // Case 3: float16 and Dt == 1
	    //   r0 b0 t0 ... t4 <-> T0 s0 T1 ... T5
	    
	    static_assert(L == 2);
	    static_assert(Dt == 1);
	    static_assert(Core::Tin == 32);
	    
	    ksgpu::warp_transpose(x[2*d], x[2*d+1], 16);
	    
	    //  r0 b0 t0 t1 t2 t3 t4 <-> T5 s0 T1 T2 T3 T4 T0
	    
	    const int src_lane = ((threadIdx.x & 0x1f) >> 1) + (threadIdx.x << 4);
	    
	    #pragma unroll
	    for (int d = 0; d < M; d++)
		x[d] = __shfl_sync(ksgpu::FULL_MASK, x[d], src_lane);
	    
	    // r0 b0 t0 t1 t2 t3 t4 <-> T5 s0 T0 T1 T2 T3 T4
	    // Now we have the correct register assignment for two calls to _advance().
	    
	    this->template _advance<0,2> (x);
	    this->template _advance<1,2> (x);
	}	    
    }

    __device__ inline finalize(T32 *pstate_warp)
    {
	static_assert(Dd == 1);
	core.save(pstate_warp);
    }
};


// -------------------------------------------------------------------------------------------------
//
// pf_kernel

#if 0

template<typename T32>
struct pf_kernel
{
    // global_kernel(out_pf, out_ssq, pstate, in, wt)
    using global_kernel_t = void (*)(T32 *, T32 *, T32 *, const T32 *, const T32 *);

    long num_profiles = 0;
    long ndm_per_block = 0;
    long threads_per_block = 0;
    long shmem_nbytes_per_block = 0;
    long ntime_per_loop_iteration = 0;
    
    global_kernel_t kernel;
};


// Launch with {32*W,1,1} threads.
template<typename Tile, int W, int B>
__global__ void __launch_bounds__(32*W, B)
pf_global_kernel(Tile::T32 *out_pf, Tile::T32 *out_ssq, Tile::T32 *pstate_glo, const Tile::T32 *in, const Tile::T32 *wt_glo, int nseg)
{
    constexpr int M = Tile::M;    // input DMs per warp
    constexpr int P = Tile::P;

    int w = threadIdx.x >> 5;  // warp ID (each warp corresponds to M trial DMs)
    int mb = blockIdx.x;       // "outer" or block-based DM index (one index corresponds to W*M trial DMs)
    int MB = gridDim.x;        // number of outer or block-based DM indices
    int b = blockIdx.y;        // beam index
    
    // Apply per-warp offsets to 'out_pf' and 'out_ssq'.
    // Shapes are (B, P, Mout, Nout) where Mout = (MB*L*M)/Dd, and Nout = (32*nseg)/Dt
    int out_mstride = nseg * ksgpu::constexpr_idiv(32, Tile::Dt);
    int out_pstride = out_mstride * MB * ksgpu::constexpr_idiv(W*M, Tile::Dd);
    int out_woff = out_mstride * (W*mb+w) * ksgpu::constexpr_idiv(M, Tile::Dd);
    out_pf += out_woff;    // per-warp offset
    out_ssq += out_woff;   // per-warp offset
    
    // Apply per-block offset to 'pstate_glo'
    // Shape is (B, MB, pstate_n32_per_block)
    pstate_glo += (b*MB + mb) * Tile::pstate_n32_per_block;

    // Apply per-thread offsets to 'in'.
    // Shape is (B, Min, Nin) where Min = (MB*W*M) and Nin = (32*nseg)
    int wglo = (b*MB + mb) * W + w;
    in += (in_off * M * 32 * nseg) + (threadIdx.x & 0x1f);
    
    // Weights are stored in global memory with shape (B, P, MB*W*M).
    wt_glo += (b*P + mb) * (W*M);    // per-block offset
    
    // Copy pstate from global memory to shared memory.
    for (int i = threadIdx.x; i < pstate_n32_per_block; i += blockDim.x)
	shmem[i] = pstate_glo[i];

    // Copying weights from global memory to shared is nontrivial.
    // We want to copy a subarray with shape (P, W*M).
    // Weights are stored in global memory with shape (B, P, W*M*MB).
    // The array is discontiguous in global memory, but contiguous in shared memory.

    constexpr int WM = W*M;
    static_assert(ksgpu::constexpr_is_pow2(WM));
    static_assert(WM >= 32);

    for (int i = threadIdx.x; i < P*WM; i += blockDim.x) {
	// Separate i = p*WM + q, where 0 <= p < P, and 0 <= q < W*M
	int q = i & xx;   // index 0 <= q < W*M
	int pwm = i - q;  // index 0 <= 
	wt_sh[p*(W*M) + ] = wt_glo[p * (W*M*MB) + q];
    }

    // Construct tile (reads pstate from shared memory)
    T32 *pstate_warp = shmem + (threadIdx.x >> 5) * Tile::pstate_n32_per_warp;
    Tile tile(out_warp, ssq_warp, pstate_warp, weights_warp, out_pstride, out_mstride, wt_pstride);

    for (int iseg = 0; iseg < nseg; iseg++) {
	T32 x[M];

	#pragma unroll
	for (int m = 0; m < M; m++)
	    x[m] += in[(32*m) * nseg];

	tile.advance(x);
	in += 32;
    }
    
    // Write pstate to shared memory.
    tile.finalize();

    __syncthreads();

    // Copy pstate from shared memory to global memory.
    for (int i = threadIdx.x; i < pstate_n32_per_block; i += blockDim.x)
	pstate_glo[i] = shmem[i];
}


template<typename T32, int Dd, int Dt, int E, int M>
pf_kernel get_pf_kernel()
{
    using Tile = pf_tile<T32, Dd, Dt, E, M>;

    pf_kernel k;
    k.num_profiles = xx;
    k.ndm_per_block = xx;
    k.shmem_nbytes_per_block = xx;
    k.kernel = pf_global_kernel<Tile>;
}

#endif


}  // namespace pirate

#endif // _PIRATE_CUDA_KERNELS_PEAK_FINDING_HPP
