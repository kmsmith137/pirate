#ifndef _PIRATE_INTERNALS_DEDISPERSION_KERNEL_IMPLEMENTATION_HPP
#define _PIRATE_INTERNALS_DEDISPERSION_KERNEL_IMPLEMENTATION_HPP

#include <cuda_fp16.h>

using namespace std;

namespace pirate {
#if 0
}  // editor auto-indent
#endif

// FIXME important but deprioritized: dedispersion kernels should use "wide" global memory loads/stores
//
// FIXME factors of 0.5, to make everything more overflow-resistant! Should be in reference kernels too?
// Note that when we put in these factors of 0.5, we'll want to change values of (epsabs, epsrel) in
// src_bin/test-gpu-dedispersion-kernels.cu (see comment in that source file).
//
// FIXME(?): I'm slightly overallocating persistent state ("rstate") in the case where input is lagged,
// since I'm saving entire cache lines, rather than residual cache lines.


// The following line evaluates to 'true' if T==float32, and 'false' if T==__half2.
//   constexpr bool is_float32 = _is_float32<T>::value;

template<typename T> struct _is_float32 { };
template<> struct _is_float32<float>   { static constexpr bool value = true; };
template<> struct _is_float32<__half>  { static constexpr bool value = false; };
template<> struct _is_float32<__half2> { static constexpr bool value = false; };


// The following line returns a (T*) pointer to shared memory
//   T *shmem = _shmem_base<T>();

template<typename T> T* _shmem_base();
template<> __device__ inline float* _shmem_base<float> () { extern __shared__ float shmem_f[]; return shmem_f; }
template<> __device__ inline __half2* _shmem_base<__half2> () { extern __shared__ __half2 shmem_h2[]; return shmem_h2; }


// -------------------------------------------------------------------------------------------------
//
// float16 permutations.
// FIXME move to its own .hpp file to expose for unit testing?
//
// Given __half2 variables a = [a0,a1] and b = [b0,b1]:
//
//    f16_align() returns [a1,b0]
//    f16_blend() returns [a0,b1]
//    __lows2half2() returns [a0,b0]
//    __highs2half2() returns [a1,b1]


__device__ __forceinline__
__half2 f16_align(__half2 a, __half2 b)
{
    __half2 d;
    
    // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-prmt
    // Note: I chose to use prmt.b32.f4e(d,a,b,2) but I think prmt.b32(d,a,b,0x5432) is equivalent.
    
    asm("prmt.b32.f4e %0, %1, %2, %3;" :
	"=r" (*(unsigned int *) &d) :
	"r" (*(const unsigned int *) &a),
	"r" (*(const unsigned int *) &b),
	"n"(2)
    );

    return d;
}


__device__ __forceinline__
__half2 f16_blend(__half2 a, __half2 b)
{
    __half2 d;
        
    // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-prmt
    asm("prmt.b32 %0, %1, %2, %3;" :
	"=r" (*(unsigned int *) &d) :
	"r" (*(const unsigned int *) &a),
	"r" (*(const unsigned int *) &b),
	"n"(0x7610)
    );

    return d;
}


// -------------------------------------------------------------------------------------------------
//
// Dedispersion "microkernels"
// Arguments beginning with 'x' are input/output data.
// The 'rs' argument is ring buffer state held in registers.


static constexpr int CycleNone = 0;
static constexpr int CycleFwd = 1;
static constexpr int CycleRev = 2;


template<typename T>
__device__ __forceinline__ void dd_sum(T &x0, T &x1, T y, T z)
{
    x0 = z + x1;
    x1 += y;
}


// Uses (N+1) registers of 'rs'.
//  - CycleNone or CycleRev: does not advance rs
//  - CycleFwd: advances rs by (N+1) lanes

template<int N, int C>
__device__ __forceinline__ void dd_step(float &x0, float &x1, float &rs)
{
    static_assert(N >= 0);
    static_assert(N < 32);
    
    float y, z;
    const int laneId = threadIdx.x & 0x1f;
    
    if constexpr (N == 0)
	z = x0;
    else {
	z = (laneId >= (32-N)) ? rs : x0;
	z = __shfl_sync(0xffffffff, z, laneId + (32-N));
    }
    
    y = (laneId >= (31-N)) ? rs : x0;
    y = __shfl_sync(0xffffffff, y, laneId + (31-N));
    
    rs = (laneId >= (31-N)) ? x0 : rs;

    if constexpr (C == CycleFwd)
	rs = __shfl_sync(0xffffffff, rs, laneId + (31-N));
    
    dd_sum(x0, x1, y, z);
}


// apply_rlag(), float version (see below for __half2 version)
__device__ __forceinline__ float apply_rlag(float x, float &xprev, int rlag)
{
    rlag = rlag & 0x1f;   // (rlag % 32)

    const int laneId = (threadIdx.x & 0x1f);
    float y = ((laneId + rlag) >= 32) ? xprev : x;
    
    xprev = x;
    return __shfl_sync(0xffffffff, y, threadIdx.x - rlag);
}


// Dedisperse float32 shape-(2,32) array, where indices are (freq/dm, time).
// Uses one register of 'rs'.
//  - CycleNone or CycleRev: does not advance rs
//  - CycleFwd: advances rs by 1 lane

template<int C>
__device__ __forceinline__ void dd_r1(float &x0, float &x1, float &rs)
{
    dd_step<0,C> (x0, x1, rs);
}


// Dedisperse float32 shape-(4,32) array, where indices are (freq/dm, time).
// Uses five registers of 'rs'.
//   - CycleNone: advances rs by 3 lanes
//   - CycleFwd: advances rs by 5 lanes

template<int C>
__device__ inline void dd_r2(float &x0, float &x1, float &x2, float &x3, float &rs)
{
    dd_r1<CycleFwd> (x0, x1, rs);      // cycles rs by 1 lane
    dd_r1<CycleFwd> (x2, x3, rs);      // cycles rs by 1 lane
    
    dd_step<0,CycleFwd> (x0, x2, rs);  // cycles rs by 1 lane
    dd_step<1,CycleNone> (x1, x3, rs);

    if constexpr (C == CycleFwd)
	rs = __shfl_sync(0xffffffff, rs, threadIdx.x + 30);
    else if constexpr (C == CycleRev)
	rs = __shfl_sync(0xffffffff, rs, threadIdx.x + 3);    
}


// Dedisperse float32 shape-(8,32) array, where indices are (freq/dm, time).
// Uses 20 registers of 'rs'.
//   - CycleNone: advances rs by 16 lanes
//   - CycleFwd: advances rs by 20 lanes

template<int C>
__device__ inline void dd_r3(float &x0, float &x1, float &x2, float &x3,
			     float &x4, float &x5, float &x6, float &x7,
			     float &rs)
{
    dd_r2<CycleFwd> (x0, x1, x2, x3, rs);  // advances rs by 5 lanes
    dd_r2<CycleFwd> (x4, x5, x6, x7, rs);  // advances rs by 5 lanes
    
    dd_step<0,CycleFwd> (x0, x4, rs);   // advances rs by 1 lane
    dd_step<2,CycleFwd> (x1, x5, rs);   // advances rs by 3 lanes (note bit-reverse here)
    dd_step<1,CycleFwd> (x2, x6, rs);   // advances rs by 2 lanes (note bit-reverse here)
    dd_step<3,CycleNone> (x3, x7, rs);  // does not advance rs by 4 lanes

    if constexpr (C == CycleFwd)
	rs = __shfl_sync(0xffffffff, rs, threadIdx.x + 28);
    else if constexpr (C == CycleRev)
	rs = __shfl_sync(0xffffffff, rs, threadIdx.x + 16);    
}


// Dedisperse float32 shape-(16,32) array, where indices are (freq/dm, time).
// Uses 32 registers of 'rs' and 'rs2', and 12 registers of 'rs3'.
//   - CycleNone: advances rs3 by 4 lanes
//   - CycleFwd: advances rs3 by 12 lanes

template<int C>
__device__ inline void dd_r4(float &x0, float &x1, float &x2, float &x3,
			     float &x4, float &x5, float &x6, float &x7,
			     float &x8, float &x9, float &x10, float &x11,
			     float &x12, float &x13, float &x14, float &x15,
			     float &rs, float &rs2, float &rs3)
{
    dd_r3<CycleFwd> (x0, x1, x2, x3, x4, x5, x6, x7, rs);         // advances rs by 20 lanes
    dd_r3<CycleFwd> (x8, x9, x10, x11, x12, x13, x14, x15, rs2);  // advances rs2 by 20 lanes

    // Note bit-reverse here.
    dd_step<0,CycleFwd> (x0, x8, rs);     // advances rs by 1 lane (21 cumulative)
    dd_step<4,CycleFwd> (x1, x9, rs);     // advances rs by 5 lanes (26 cumulative)
    dd_step<2,CycleFwd> (x2, x10, rs2);   // advances rs2 by 3 lanes (23 cumulative)
    dd_step<6,CycleFwd> (x3, x11, rs2);   // advances rs2 by 7 lanes (30 cumulative)
    dd_step<1,CycleFwd> (x4, x12, rs2);   // advances rs2 by 2 lanes (32 cumulative)
    dd_step<5,CycleFwd> (x5, x13, rs);    // advances rs by 6 lanes (32 cumulative)
    dd_step<3,CycleFwd> (x6, x14, rs3);   // advances rs3 by 4 lanes
    dd_step<7,CycleNone> (x7, x15, rs3);  // does not advance rs3 by 8 lanes

    if constexpr (C == CycleFwd)
	rs3 = __shfl_sync(0xffffffff, rs3, threadIdx.x + 24);
    else if constexpr (C == CycleRev)
	rs3 = __shfl_sync(0xffffffff, rs3, threadIdx.x + 4);        
}


// -------------------------------------------------------------------------------------------------


template<int N, int C>
__device__ __forceinline__ __half2 lag_half2(__half2 x, __half2 &rs)
{
    const int laneId = (threadIdx.x & 0x1f);
    
    __half2 t = (laneId >= (32-N)) ? rs : x;
    t = __shfl_sync(0xffffffff, t, threadIdx.x + (32-N));
    rs = (laneId >= (32-N)) ? x : rs;

    if constexpr (C == CycleFwd)
	rs = __shfl_sync(0xffffffff, rs, threadIdx.x + (32-N));

    return t;
}


// apply_rlag(), __half2 version (see above for float version)
__device__ inline __half2 apply_rlag(__half2 x, __half2 &xprev, int rlag)
{
    rlag = rlag & 0x3f;  // (rlag % 64)

    const int rlag32 = rlag >> 1;
    const int shifted_lane = (threadIdx.x & 0x1f) + rlag32;
    
    __half2 y = (shifted_lane >= 32) ? xprev : x;

    if (rlag & 1) {
	__half2 z = f16_blend(x, xprev);
	y = (shifted_lane == 31) ? z : y;

	__half2 t = __shfl_sync(0xffffffff, y, threadIdx.x + 31);  // (y[-2], y[-1])
	y = f16_align(t, y);  // (y[-1], y[0])
    }
    
    xprev = x;
    return __shfl_sync(0xffffffff, y, threadIdx.x - rlag32);
}


// Dedisperse float16 shape-(2,64) array, where indices are (freq/dm, time).
// Uses one register of 'rs'.
//  - CycleNone or CycleRev: does not advance rs
//  - CycleFwd: advances rs by 1 lane

template<int C>
__device__ __forceinline__ void dd_r1(__half2 &x0, __half2 &x1, __half2 &rs)
{
    __half2 t = lag_half2<1,C> (x0, rs);  // (x0[-2], x0[-1])
    __half2 y = f16_align(t, x0);         // (x0[-1], x0[0])
    
    dd_sum(x0, x1, y, x0);
}


// Dedisperse float16 shape-(2,2,64) array, where indices are (freq/dm, spectator, time).
// Uses one register of 'rs'.
//  - CycleNone or CycleRev: does not advance rs
//  - CycleFwd: advances rs by 1 lane

template<int C>
__device__ inline void dd_r1_s2(__half2 &x00, __half2 &x10, __half2 &x01, __half2 &x11, __half2 &rs)
{
    __half2 t = __highs2half2(x00, x01);  // (x00[1], x01[1])
    __half2 u = lag_half2<1,C> (t,rs);    // (x00[-1], x01[-1]) and maybe cycle rs by 1
    __half2 y0 = __lows2half2(u, x00);    // (x00[-1], x00[0])
    __half2 y1 = f16_align(u, x01);       // (x01[-1], x01[0])

    dd_sum(x00, x10, y0, x00);
    dd_sum(x01, x11, y1, x01);
}


// Partially dedisperse shape (4,64) array, where indices are (freq/dm, time):
//   - First tree dedispersion iteration is completely applied.
//   - Second tree dedispersion iteration is applied to [x1,x3] but not [x0,x2].
//
// Uses 2 registers of 'rs', and cycles rs by 2.

__device__ inline void dd_r2_top(__half2 &x0, __half2 &x1, __half2 &x2, __half2 &x3, __half2 &rs)
{
    // First tree iteration (applied independently to x0+x1 and x2+x3)
    dd_r1_s2<CycleFwd> (x0, x1, x2, x3, rs);   // cycles rs by 1

    __half2 y1 = lag_half2<1,CycleFwd> (x1,rs);  // (x1[-2], x1[-1]) and cycles rs by 1
    __half2 z1 = f16_align(y1, x1);              // (x1[-1], x1[0])

    dd_sum(x1, x3, y1, z1);
}


// Dedisperse shape (4,64) array, where indices are (freq/dm, time).
// Three registers of 'rs' are used.
//   - CycleNone: advances rs by 2 lanes.
//   - CycleFwd: advances rs by 3 lanes.

template<int C>
__device__ inline void dd_r2(__half2 &x0, __half2 &x1, __half2 &x2, __half2 &x3, __half2 &rs)
{
    dd_r2_top(x0, x1, x2, x3, rs);  // cycles rs by 2
    dd_r1<CycleNone> (x0, x2, rs);  // does not cycle rs

    if constexpr (C == CycleFwd)
	rs = __shfl_sync(0xffffffff, rs, threadIdx.x + 31);
    else if constexpr (C == CycleRev)
	rs = __shfl_sync(0xffffffff, rs, threadIdx.x + 2);    
}


// Dedisperse shape (4,2,64) array, where indices are (freq/dm, spectator, time).
// Five registers of 'rs' are used, and rs is cycled by either 4 (Cycle=false) or 5 (Cycle=true).
//   - CycleNone: advances rs by 4 lanes.
//   - CycleFwd: advances rs by 5 lanes.

template<int C>
__device__ inline void dd_r2_s2(__half2 &x00, __half2 &x10, __half2 &x20, __half2 &x30,
				__half2 &x01, __half2 &x11, __half2 &x21, __half2 &x31,
				__half2 &rs)
{
    dd_r2_top(x00, x10, x20, x30, rs);  // cycles rs by 2
    dd_r2_top(x01, x11, x21, x31, rs);  // cycles rs by 2
    
    dd_r1_s2<CycleNone> (x00, x20, x01, x21, rs);
    
    if constexpr (C == CycleFwd)
	rs = __shfl_sync(0xffffffff, rs, threadIdx.x + 31);
    else if constexpr (C == CycleRev)
	rs = __shfl_sync(0xffffffff, rs, threadIdx.x + 4);
}


// Partially dedisperse shape (8,64) array, where indices are (freq/dm, time):
//
//   - First two tree dedispersion iterations are completely applied.
//   - Third tree dedispersion iteration is applied to [x0,x4], [x2,x6], [x3,x7].
//   - Caller is responsible for applying third tree iteration to [x1,x5].
//     (Note that due to DM bit reversal, this will involve lags 2+3, not lags 1+2.)
//
// Uses 9 registers of 'rs', and cycles rs by 9.

__device__ inline void dd_r3_top(__half2 &x0, __half2 &x1, __half2 &x2, __half2 &x3,
				 __half2 &x4, __half2 &x5, __half2 &x6, __half2 &x7,
				 __half2 &rs)
{
    // First two tree iterations (applied independently to x0..x3 and x4..x7)
    dd_r2_s2<CycleFwd> (x0, x1, x2, x3, x4, x5, x6, x7, rs);   // cycles rs by 5
    
    __half2 t = f16_align(x0, x3);       // (x0[1], x3[0])
    t = lag_half2<1,CycleFwd> (t, rs);   // (x0[-1], x3[-2]) and cycles rs by 1

    __half2 y2 = lag_half2<1,CycleFwd> (x2,rs);   // (x2[-2], x2[-1]) and cycles rs by 1
    __half2 y3 = lag_half2<2,CycleFwd> (x3,rs);   // (x3[-4], x3[-3]) and cycles rs by 2
    __half2 y0 = __lows2half2(t, x0);             // (x0[-1], x0[0])
    __half2 z2 = f16_align(y2, x2);               // (x2[-1], x2[0])
    __half2 z3 = __highs2half2(y3, t);            // (x3[-3], x3[-2])

    dd_sum(x0, x4, y0, x0);
    dd_sum(x2, x6, y2, z2);
    dd_sum(x3, x7, y3, z3);
}


// Dedisperse shape (8,64) array, where indices are (freq/dm, time).
// Uses 12 registers of 'rs'.
//   - CycleNone: advances rs by 11.
//   - CycleFwd: advances rs by 12.

template<int C>
__device__ inline void dd_r3(__half2 &x0, __half2 &x1, __half2 &x2, __half2 &x3,
			     __half2 &x4, __half2 &x5, __half2 &x6, __half2 &x7,
			     __half2 &rs)
{
    dd_r3_top(x0, x1, x2, x3, x4, x5, x6, x7, rs);  // cycles rs by 9
    
    __half2 t = lag_half2<2,CycleFwd> (x1,rs);     // (x1[-4], x1[-3]) and cycles rs by 2
    __half2 z1 = lag_half2<1,CycleNone> (x1,rs);   // (x1[-2], x1[-1]) and does not cycle rs by 1
    __half2 y1 = f16_align(t, z1);                 // (x1[-3], x1[-2])

    dd_sum(x1, x5, y1, z1);
    
    if constexpr (C == CycleFwd)
	rs = __shfl_sync(0xffffffff, rs, threadIdx.x + 31);
    else if constexpr (C == CycleRev)
	rs = __shfl_sync(0xffffffff, rs, threadIdx.x + 11);
}


// Dedisperse shape (8,2,64) array, where indices are (freq/dm, spectator, time).
// Uses 22 registers of 'rs', and cycles rs by 22.

__device__ inline void dd_r3_s2(__half2 &x00, __half2 &x10, __half2 &x20, __half2 &x30,
				__half2 &x40, __half2 &x50, __half2 &x60, __half2 &x70,
				__half2 &x01, __half2 &x11, __half2 &x21, __half2 &x31,
				__half2 &x41, __half2 &x51, __half2 &x61, __half2 &x71,
				__half2 &rs)
{
    dd_r3_top(x00, x10, x20, x30, x40, x50, x60, x70, rs);  // cycle rs by 9
    dd_r3_top(x01, x11, x21, x31, x41, x51, x61, x71, rs);  // cycle rs by 9

    __half2 t = __highs2half2(x10, x11);           // (x10[1], x11[1])
    __half2 u = lag_half2<2,CycleFwd> (t,rs);      // (x10[-3], x11[-3]) and cycle rs by 2
    __half2 z10 = lag_half2<1,CycleFwd> (x10,rs);  // (x10[-2], x10[-1]) and cycle rs by 1
    __half2 z11 = lag_half2<1,CycleFwd> (x11,rs);  // (x11[-2], x11[-1]) and cycle rs by 1
    __half2 y10 = __lows2half2(u, z10);            // (x10[-3], x10[-2])
    __half2 y11 = f16_align(u, z11);               // (x11[-3], x11[-2])

    dd_sum(x10, x50, y10, z10);
    dd_sum(x11, x51, y11, z11);
}


// Dedisperse shape (16,64) array, where indices are (freq/dm, time).
// Uses all 32 registers of 'rs', and 16 registers of 'rs2'.
//   - CycleNone: advances rs by 12.
//   - CycleFwd: advances rs by 16.

template<int C>
__device__ inline void dd_r4(__half2 &x0, __half2 &x1, __half2 &x2, __half2 &x3,
			     __half2 &x4, __half2 &x5, __half2 &x6, __half2 &x7,
			     __half2 &x8, __half2 &x9, __half2 &x10, __half2 &x11,
			     __half2 &x12, __half2 &x13, __half2 &x14, __half2 &x15,
			     __half2 &rs, __half2 &rs_unused, __half2 &rs2)
{
    dd_r3_s2(x0, x1, x2, x3, x4, x5, x6, x7,
	     x8, x9, x10, x11, x12, x13, x14, x15,
	     rs);  // cycle rs by 22

    __half2 t06 = f16_align(x0, x6);  // (x0[1], x6[0])
    __half2 t25 = f16_align(x2, x5);  // (x2[1], x5[0])
    __half2 t17 = f16_align(x1, x7);  // (x1[1], x7[0])

    __half2 u06 = lag_half2<1,CycleFwd> (t06,rs);  // (x0[-1], x6[-2]) and cycle rs by 1 (cumulative 23)
    __half2 u25 = lag_half2<2,CycleFwd> (t25,rs);  // (x2[-3], x5[-4]) and cycle rs by 2 (cumulative 25)
    __half2 u17 = lag_half2<3,CycleFwd> (t17,rs);  // (x1[-5], x7[-6]) and cycle rs by 3 (cumulative 28)
    __half2 u3 = lag_half2<4,CycleFwd> (x3,rs);    // (x3[-8], x3[-7]) and cycle rs by 4 (cumulative 32 -- switch to rs2 here)

    __half2 y4 = lag_half2<1,CycleFwd> (x4,rs2);   // (x4[-2], x4[-1]) and cycle rs2 by 1
    __half2 z2 = lag_half2<1,CycleFwd> (x2,rs2);   // (x2[-2], x2[-1]) and cycle rs2 by 1 (cumulative 2)
    __half2 y6 = lag_half2<2,CycleFwd> (x6,rs2);   // (x6[-4], x6[-3]) and cycle rs2 by 2 (cumulative 4)
    __half2 z1 = lag_half2<2,CycleFwd> (x1,rs2);   // (x1[-4], x1[-3]) and cycle rs2 by 2 (cumulative 6)
    __half2 y5 = lag_half2<3,CycleFwd> (x5,rs2);   // (x5[-6], x5[-5]) and cycle rs2 by 3 (cumulative 9)
    __half2 z3 = lag_half2<3,CycleFwd> (x3,rs2);   // (x3[-6], x3[-5]) and cycle rs2 by 3 (cumulative 12)
    __half2 y7 = lag_half2<4,CycleNone> (x7,rs2);  // (x7[-8], x7[-7]) and don't cycle rs2 by 4

    __half2 y0 = __lows2half2(u06, x0);
    __half2 y2 = __lows2half2(u25, z2);
    __half2 y1 = __lows2half2(u17, z1);
    __half2 y3 = f16_align(u3, z3);

    __half2 z4 = f16_align(y4, x4);
    __half2 z6 = __highs2half2(y6, u06);
    __half2 z5 = __highs2half2(y5, u25);
    __half2 z7 = __highs2half2(y7, u17);

    dd_sum(x0, x8, y0, x0);
    dd_sum(x1, x9, y1, z1);
    dd_sum(x2, x10, y2, z2);
    dd_sum(x3, x11, y3, z3);
    dd_sum(x4, x12, y4, z4);
    dd_sum(x5, x13, y5, z5);
    dd_sum(x6, x14, y6, z6);
    dd_sum(x7, x15, y7, z7);

    if constexpr (C == CycleFwd)
	rs2 = __shfl_sync(0xffffffff, rs2, threadIdx.x + 28);
    else if constexpr (C == CycleRev)
	rs2 = __shfl_sync(0xffffffff, rs2, threadIdx.x + 12);
}


// -------------------------------------------------------------------------------------------------
//
// Full dedispersion kernels (i.e. cuda __global__).
// Note: There's a lot of cut-and-paste here, but I'm not seeing a good way to reduce it!
//
// For now, dedispersion kernels have interface
//
//   dedisperse_rRANK(__half2 *iobuf, __half2 *rstate,
//                    long beam_stride, long ambient_stride,
//                    int row_stride, int nt_cl,
//                    uint *integer_constants,
//                    uint flags);
//
// The 'iobuf' and 'rstate' arrays have logical shapes:
//
//   float16 iobuf[nbeams][nambient][2^rank][ntime];
//   float16 rstate[nbeams][nambient][state_nelts_per_small_tree]
//
// The iobuf array has strides (beam_stride, ambient_stride, row_stride, 1),
// and the rstate array is fully contiguous.
//
// Kernel grid dimensions are (x,y,z) = (ambient_ix, beam_ix, 1),
// and thread block dimensions are (x,y,z) = (32 * warps_per_threadblock, 1, 1).
//
// The kernels are not externally visible, but get called via GpuDedispersionKernel::make() below.
// FIXME replace 'integer_constants' argument with constant memory.


template<typename T, class Inbuf, class Outbuf>
__global__ void dedisperse_r1(typename Inbuf::device_args inbuf_args, typename Outbuf::device_args outbuf_args, T *rstate, int nt_cl, uint *integer_constants)
{
    static_assert(sizeof(T) == 4);    
    // assert(blockDim.x == 32);

    constexpr int gmem_ncl = Inbuf::is_lagged ? 2 : 1;
    const int ambient_ix = blockIdx.x;
    const int beam_ix = blockIdx.y;
    
    typename Inbuf::device_state inbuf(inbuf_args, 0);
    typename Outbuf::device_state outbuf(outbuf_args);

    // Apply (beam, ambient) strides to rstate. (Note no laneId shift here.)
    rstate += beam_ix * gridDim.x * (32 * gmem_ncl);
    rstate += ambient_ix * (32 * gmem_ncl);
    
    T rs = rstate[threadIdx.x];
    
    // Temporarily disable nvcc warning "...variable was declared but never referenced"
#pragma nv_diag_suppress 177
    T xp0;
    int dm;
#pragma nv_diag_default 177

    if constexpr (Inbuf::is_lagged) {
	xp0 = rstate[threadIdx.x + 32];
	dm = __brev(blockIdx.x) >> (33 - __ffs(gridDim.x));  // bit-reversed DM, see below
	dm += inbuf_args._is_downsampled() ? gridDim.x : 0;
    }

    for (int it_cl = 0; it_cl < nt_cl; it_cl++) {
	T x0 = inbuf.load(0);
	T x1 = inbuf.load(1);
	inbuf.advance();

	if constexpr (Inbuf::is_lagged) {
	    // "Row" index represents a coarse frequency 0 <= f < 2^(rank).
	    // Residual lag is computed as folows:
	    //   int ff = 2^rank - 1 - f;
	    //   int N = is_float32 ? 32 : 64;
	    //   int rlag = (ff * d) % N
	
	    x0 = apply_rlag(x0, xp0, dm);
	}

	dd_r1<CycleRev> (x0, x1, rs);

	outbuf.store(0, x0);
	outbuf.store(1, x1);
	outbuf.advance();
    }

    rstate[threadIdx.x] = rs;
    
    if constexpr (Inbuf::is_lagged) {
	rstate[threadIdx.x + 32] = xp0;
    }
}


template<typename T, class Inbuf, class Outbuf>
__global__ void dedisperse_r2(typename Inbuf::device_args inbuf_args, typename Outbuf::device_args outbuf_args, T *rstate, int nt_cl, uint *integer_constants)
{
    static_assert(sizeof(T) == 4);
    // assert(blockDim.x == 32);
    
    constexpr int gmem_ncl = Inbuf::is_lagged ? 4 : 1;
    const int ambient_ix = blockIdx.x;
    const int beam_ix = blockIdx.y;
    
    typename Inbuf::device_state inbuf(inbuf_args, 0);
    typename Outbuf::device_state outbuf(outbuf_args);

    // Apply (beam, ambient) strides to rstate. (Note no laneId shift here.)
    rstate += beam_ix * gridDim.x * (32 * gmem_ncl);
    rstate += ambient_ix * (32 * gmem_ncl);
    
    T rs = rstate[threadIdx.x];
    
    // Temporarily disable nvcc warning "...variable was declared but never referenced"
#pragma nv_diag_suppress 177
    T xp0, xp1, xp2;
    int dm;
#pragma nv_diag_default 177

    if constexpr (Inbuf::is_lagged) {
	xp0 = rstate[threadIdx.x + 32];
	xp1 = rstate[threadIdx.x + 64];
	xp2 = rstate[threadIdx.x + 96];
	dm = __brev(blockIdx.x) >> (33 - __ffs(gridDim.x));  // bit-reversed DM, see below
	dm += inbuf_args._is_downsampled() ? gridDim.x : 0;
    }

    for (int it_cl = 0; it_cl < nt_cl; it_cl++) {
	T x0 = inbuf.load(0);
	T x1 = inbuf.load(1);
	T x2 = inbuf.load(2);
	T x3 = inbuf.load(3);
	inbuf.advance();

	if constexpr (Inbuf::is_lagged) {
	    // "Row" index represents a coarse frequency 0 <= f < 2^(rank).
	    // Residual lag is computed as folows:
	    //   int ff = 2^rank - 1 - f;
	    //   int N = is_float32 ? 32 : 64;
	    //   int rlag = (ff * d) % N
	
	    x0 = apply_rlag(x0, xp0, 3*dm);
	    x1 = apply_rlag(x1, xp1, 2*dm);
	    x2 = apply_rlag(x2, xp2, dm);
	}

	dd_r2<CycleRev> (x0, x1, x2, x3, rs);

	outbuf.store(0, x0);
	outbuf.store(1, x1);
	outbuf.store(2, x2);
	outbuf.store(3, x3);
	outbuf.advance();
    }

    rstate[threadIdx.x] = rs;

    if constexpr (Inbuf::is_lagged) {
	rstate[threadIdx.x + 32] = xp0;
	rstate[threadIdx.x + 64] = xp1;
	rstate[threadIdx.x + 96] = xp2;
    }
}


template<typename T, class Inbuf, class Outbuf>
__global__ void dedisperse_r3(typename Inbuf::device_args inbuf_args, typename Outbuf::device_args outbuf_args, T *rstate, int nt_cl, uint *integer_constants)
{
    static_assert(sizeof(T) == 4);
    // assert(blockDim.x == 32);

    constexpr int gmem_ncl = Inbuf::is_lagged ? 8 : 1;
    const int ambient_ix = blockIdx.x;
    const int beam_ix = blockIdx.y;
    
    typename Inbuf::device_state inbuf(inbuf_args, 0);
    typename Outbuf::device_state outbuf(outbuf_args);

    // Apply (beam, ambient) strides to rstate. (Note no laneId shift here.)
    rstate += beam_ix * gridDim.x * (32 * gmem_ncl);
    rstate += ambient_ix * (32 * gmem_ncl);
    
    T rs = rstate[threadIdx.x];
    
    // Temporarily disable nvcc warning "...variable was declared but never referenced"
#pragma nv_diag_suppress 177
    T xp0, xp1, xp2, xp3, xp4, xp5, xp6;
    int dm;
#pragma nv_diag_default 177

    if constexpr (Inbuf::is_lagged) {
	xp0 = rstate[threadIdx.x + 32];
	xp1 = rstate[threadIdx.x + 2*32];
	xp2 = rstate[threadIdx.x + 3*32];
	xp3 = rstate[threadIdx.x + 4*32];
	xp4 = rstate[threadIdx.x + 5*32];
	xp5 = rstate[threadIdx.x + 6*32];
	xp6 = rstate[threadIdx.x + 7*32];
	dm = __brev(blockIdx.x) >> (33 - __ffs(gridDim.x));  // bit-reversed DM, see below
	dm += inbuf_args._is_downsampled() ? gridDim.x : 0;
    }
    
    for (int it_cl = 0; it_cl < nt_cl; it_cl++) {
	T x0 = inbuf.load(0);
	T x1 = inbuf.load(1);
	T x2 = inbuf.load(2);
	T x3 = inbuf.load(3);
	T x4 = inbuf.load(4);
	T x5 = inbuf.load(5);
	T x6 = inbuf.load(6);
	T x7 = inbuf.load(7);
	inbuf.advance();

	if constexpr (Inbuf::is_lagged) {
	    // "Row" index represents a coarse frequency 0 <= f < 2^(rank).
	    // Residual lag is computed as folows:
	    //   int ff = 2^rank - 1 - f;
	    //   int N = is_float32 ? 32 : 64;
	    //   int rlag = (ff * d) % N
	
	    x0 = apply_rlag(x0, xp0, 7*dm);
	    x1 = apply_rlag(x1, xp1, 6*dm);
	    x2 = apply_rlag(x2, xp2, 5*dm);
	    x3 = apply_rlag(x3, xp3, 4*dm);
	    x4 = apply_rlag(x4, xp4, 3*dm);
	    x5 = apply_rlag(x5, xp5, 2*dm);
	    x6 = apply_rlag(x6, xp6, dm);
	}

	dd_r3<CycleRev> (x0, x1, x2, x3, x4, x5, x6, x7, rs);
	
	outbuf.store(0, x0);
	outbuf.store(1, x1);
	outbuf.store(2, x2);
	outbuf.store(3, x3);
	outbuf.store(4, x4);
	outbuf.store(5, x5);
	outbuf.store(6, x6);
	outbuf.store(7, x7);
	outbuf.advance();
    }

    rstate[threadIdx.x] = rs;
    
    if constexpr (Inbuf::is_lagged) {
	rstate[threadIdx.x + 32] = xp0;
	rstate[threadIdx.x + 2*32] = xp1;
	rstate[threadIdx.x + 3*32] = xp2;
	rstate[threadIdx.x + 4*32] = xp3;
	rstate[threadIdx.x + 5*32] = xp4;
	rstate[threadIdx.x + 6*32] = xp5;
	rstate[threadIdx.x + 7*32] = xp6;
    }
}


template<typename T, class Inbuf, class Outbuf>
__global__ void dedisperse_r4(typename Inbuf::device_args inbuf_args, typename Outbuf::device_args outbuf_args, T *rstate, int nt_cl, uint *integer_constants)
{
    static_assert(sizeof(T) == 4);  // float or __half2
    // assert(blockDim.x == 32);
    
    constexpr bool is_float32 = _is_float32<T>::value;    
    constexpr int nrs_per_thread = is_float32 ? 3 : 2;
    constexpr int nrp_per_thread = Inbuf::is_lagged ? 15 : 0;
    constexpr int gmem_ncl = nrs_per_thread + nrp_per_thread;
    
    const int ambient_ix = blockIdx.x;
    const int beam_ix = blockIdx.y;
    
    typename Inbuf::device_state inbuf(inbuf_args, 0);
    typename Outbuf::device_state outbuf(outbuf_args);

    // Apply (beam, ambient) strides to rstate. (Note no laneId shift here.)
    rstate += beam_ix * gridDim.x * (32 * gmem_ncl);
    rstate += ambient_ix * (32 * gmem_ncl);
    
    T rs = rstate[threadIdx.x];
    T rs2 = rstate[threadIdx.x + 32];
    T rs3;

    if constexpr (is_float32)
	rs3 = rstate[threadIdx.x + 64];
        
    // Temporarily disable nvcc warning "...variable was declared but never referenced"
#pragma nv_diag_suppress 177
    T xp0, xp1, xp2, xp3, xp4, xp5, xp6, xp7, xp8, xp9, xp10, xp11, xp12, xp13, xp14;
#pragma nv_diag_default 177

    if constexpr (Inbuf::is_lagged) {
	xp0 = rstate[threadIdx.x + 32 * (nrs_per_thread)];
	xp1 = rstate[threadIdx.x + 32 * (nrs_per_thread+1)];
	xp2 = rstate[threadIdx.x + 32 * (nrs_per_thread+2)];
	xp3 = rstate[threadIdx.x + 32 * (nrs_per_thread+3)];
	xp4 = rstate[threadIdx.x + 32 * (nrs_per_thread+4)];
	xp5 = rstate[threadIdx.x + 32 * (nrs_per_thread+5)];
	xp6 = rstate[threadIdx.x + 32 * (nrs_per_thread+6)];
	xp7 = rstate[threadIdx.x + 32 * (nrs_per_thread+7)];
	xp8 = rstate[threadIdx.x + 32 * (nrs_per_thread+8)];
	xp9 = rstate[threadIdx.x + 32 * (nrs_per_thread+9)];
	xp10 = rstate[threadIdx.x + 32 * (nrs_per_thread+10)];
	xp11 = rstate[threadIdx.x + 32 * (nrs_per_thread+11)];
	xp12 = rstate[threadIdx.x + 32 * (nrs_per_thread+12)];
	xp13 = rstate[threadIdx.x + 32 * (nrs_per_thread+13)];
	xp14 = rstate[threadIdx.x + 32 * (nrs_per_thread+14)];
    }    

    for (int it_cl = 0; it_cl < nt_cl; it_cl++) {
	T x0 = inbuf.load(0);
	T x1 = inbuf.load(1);
	T x2 = inbuf.load(2);
	T x3 = inbuf.load(3);
	T x4 = inbuf.load(4);
	T x5 = inbuf.load(5);
	T x6 = inbuf.load(6);
	T x7 = inbuf.load(7);
	T x8 = inbuf.load(8);
	T x9 = inbuf.load(9);
	T x10 = inbuf.load(10);
	T x11 = inbuf.load(11);
	T x12 = inbuf.load(12);
	T x13 = inbuf.load(13);
	T x14 = inbuf.load(14);
	T x15 = inbuf.load(15);
	inbuf.advance();

	if constexpr (Inbuf::is_lagged) {
	    // Ambient index represents a bit-reversed DM 0 <= d < 2^(ambient_rank).
	    // Residual lag is computed as folows:
	    //   int ff = 2^rank - 1 - f;
	    //   int N = is_float32 ? 32 : 64;
	    //   int rlag = (ff * d) % N

	    int dm = __brev(blockIdx.x) >> (33 - __ffs(gridDim.x));  // bit-reversed DM
	    dm += inbuf_args._is_downsampled() ? gridDim.x : 0;
	    
	    x0 = apply_rlag(x0, xp0, 15*dm);
	    x1 = apply_rlag(x1, xp1, 14*dm);
	    x2 = apply_rlag(x2, xp2, 13*dm);
	    x3 = apply_rlag(x3, xp3, 12*dm);
	    x4 = apply_rlag(x4, xp4, 11*dm);
	    x5 = apply_rlag(x5, xp5, 10*dm);
	    x6 = apply_rlag(x6, xp6, 9*dm);
	    x7 = apply_rlag(x7, xp7, 8*dm);
	    x8 = apply_rlag(x8, xp8, 7*dm);
	    x9 = apply_rlag(x9, xp9, 6*dm);
	    x10 = apply_rlag(x10, xp10, 5*dm);
	    x11 = apply_rlag(x11, xp11, 4*dm);
	    x12 = apply_rlag(x12, xp12, 3*dm);
	    x13 = apply_rlag(x13, xp13, 2*dm);
	    x14 = apply_rlag(x14, xp14, dm);
	}

	dd_r4<CycleRev> (x0, x1, x2, x3, x4, x5, x6, x7,
			 x8, x9, x10, x11, x12, x13, x14, x15,
			 rs, rs3, rs2);
	
	outbuf.store(0, x0);
	outbuf.store(1, x1);
	outbuf.store(2, x2);
	outbuf.store(3, x3);
	outbuf.store(4, x4);
	outbuf.store(5, x5);
	outbuf.store(6, x6);
	outbuf.store(7, x7);
	outbuf.store(8, x8);
	outbuf.store(9, x9);
	outbuf.store(10, x10);
	outbuf.store(11, x11);
	outbuf.store(12, x12);
	outbuf.store(13, x13);
	outbuf.store(14, x14);
	outbuf.store(15, x15);
	outbuf.advance();
    }

    rstate[threadIdx.x] = rs;
    rstate[threadIdx.x + 32] = rs2;
    
    if constexpr (is_float32)
	rstate[threadIdx.x + 64] = rs3;

    if constexpr (Inbuf::is_lagged) {
	rstate[threadIdx.x + 32 * (nrs_per_thread)] = xp0;
	rstate[threadIdx.x + 32 * (nrs_per_thread+1)] = xp1;
	rstate[threadIdx.x + 32 * (nrs_per_thread+2)] = xp2;
	rstate[threadIdx.x + 32 * (nrs_per_thread+3)] = xp3;
	rstate[threadIdx.x + 32 * (nrs_per_thread+4)] = xp4;
	rstate[threadIdx.x + 32 * (nrs_per_thread+5)] = xp5;
	rstate[threadIdx.x + 32 * (nrs_per_thread+6)] = xp6;
	rstate[threadIdx.x + 32 * (nrs_per_thread+7)] = xp7;
	rstate[threadIdx.x + 32 * (nrs_per_thread+8)] = xp8;
	rstate[threadIdx.x + 32 * (nrs_per_thread+9)] = xp9;
	rstate[threadIdx.x + 32 * (nrs_per_thread+10)] = xp10;
	rstate[threadIdx.x + 32 * (nrs_per_thread+11)] = xp11;
	rstate[threadIdx.x + 32 * (nrs_per_thread+12)] = xp12;
	rstate[threadIdx.x + 32 * (nrs_per_thread+13)] = xp13;
	rstate[threadIdx.x + 32 * (nrs_per_thread+14)] = xp14;
    }    
}


// -------------------------------------------------------------------------------------------------
//
// Start higher-rank stuff here!
// Just hacking for now, will clean up and define nice data structures later.
//
// A note on indexing, since I always get confused by this!!
//
// We split rank = rank0 + rank1
//   where rank0 = floor(rank/2)
//     and rank1 = ceil(rank/2)
//
// Number of warps W = 2^rank0.
// Number of data registers/thread R = 2^rank1.
//
// Throughout the kernel,
//
//    - Index i runs over 0 <= i < 2^rank1
//    - Index j runs over 0 <= j < 2^rank0
//
// This indexing convention is most similar to other parts of the code (e.g. ReferenceDedisperser).
// In the first stage,
//
//    - Each warp corresponds to one or two values of 'i', which represent coarse frequencies:
//        - If rank1==rank0, we take i = warpId.
//        - If rank1==rank0, we take i = 2*warpId and i = 2*warpId+1.
//
//    - Registers in the warp represent different values of 'j'.
//      Initially, 'j' represents a fine frequency.
//      At the end, 'j' represents a coarse bit-reversed DM.
//
//    - When reading the input array, we read from array index (2^rank0)*i + j
//
// In the second stage,
//
//    - Each warp corresponds to a fixed value of 'j', which represents a bit-reversed coarse DM.
//      Currently we take j = warpId, but this might change later.
//
//    - Registers in the warp represent different values of 'i'.
//      Initially, 'i' represents a coarse frequency
//      At the end, 'i' represents a bit-reversed fine DM.
//
//    - When writing to the output array, we write to array index (2^rank0)*i + j
//
// The intermdiate ring buffer delay in channel (i,j) is:
//
//     (2^rank1-1-i) * bit_reverse(j,rank0)    (*)
//
// Since the ring buffer can only supply lags which are even numbers, we supply an additional lag=1
// in channels (i,j) where (*) is odd, or equivalently:
//
//    (i is even) and (j >= 2^(rank0-1))


// When shared memory ring buffer is saved/restored in global memory, how many cache lines do we need?
template<typename T, int Rank> struct _gs_ncl { };

// Precomputed in git/chord/frb_search/r8_hacking.py
template<> struct _gs_ncl<float,5>    { static constexpr int value = 6; };
template<> struct _gs_ncl<__half2,5>  { static constexpr int value = 3; };
template<> struct _gs_ncl<float,6>    { static constexpr int value = 25; };
template<> struct _gs_ncl<__half2,6>  { static constexpr int value = 12; };
template<> struct _gs_ncl<float,7>    { static constexpr int value = 105; };
template<> struct _gs_ncl<__half2,7>  { static constexpr int value = 52; };
template<> struct _gs_ncl<float,8>    { static constexpr int value = 450; };
template<> struct _gs_ncl<__half2,8>  { static constexpr int value = 224; };


// The "integer constants" array looks like this:
//
//   uint32 control_words[2^rank1][2^rank0];  // indexed by (i,j)
//   uint32 gmem_specs[gs_ncl][2];
//
// A ring buffer "control word" consists of:
//
//   uint15 rb_base;   // base shared memory location of ring buffer (in 32-bit registers)
//   uint9  rb_pos;    // current position, satisfying 0 <= rb_pos < (rb_lag + 32)
//   uint8  rb_lag;    // ring buffer lag (in 32-bit registers), note that capacity = lag + 32.
//
// Depending on context, 'shmem_curr_pos' may point to either the end of the buffer
// (writer thread context), or be appropriately lagged (reader thread context).
//
// A "gmem spec" is a pair describing how a global memory cache line gets scattered into shared memory.
//
//   uint32 shmem_base;  // in 32-bit registers, will always be a multiple of 32
//   uint32 gap_bits;    // FIXME write comment explaining this
//
// FIXME it would probably be better to keep the integer constants array in GPU constant memory.
// (Currently we keep it in global memory.) Before doing this, I wanted to answer some initial
// questions about constant memory (search "CHORD TODO" google doc for "constant memory").


    
// Called at very beginning of kernel, to read control words into registers.
//
// Recall the control words are a shape (2^rank1, 2^rank0) array:
//   uint32 control_words[2^rank1][2^rank0];   // indexed by (i,j)
//
// On warp w, we want the following control words:
//
//   - Lanes 0 <= l < 16    store control_words[i][j] at "writer offset"
//                           where i = w + ((l & 2^rank0) % 2^rank1)
//                                 j = l % 2^rank0
//
//   - Lanes 16 <= l < 32   store control_words[i][j] at "reader offset"
//                           where i = (l-16) % 2^rank1
//                                 j = w
//
// (Note that this register assigment could change, if mapping of warpIds to
//  i,j values changes.)

template<int Rank>
__device__ inline uint read_control_words(const uint *integer_constants)
{
    static_assert((Rank >= 5) && (Rank <= 8));
    
    constexpr int Rank0 = (Rank >> 1);   // round down
    constexpr int Rank1 = Rank - Rank0; 
    constexpr int N0 = 1 << Rank0;
    constexpr int N1 = 1 << Rank1;
    constexpr int N = 1 << Rank;
   
    extern __shared__ uint shmem_i[];
    
    // (Global memory) -> (Shared memory).
    // Note: two-to-one shared memory bank conflict here!
    // This is okay since it's just one-time initialization at the beginning of the kernel.
    
    if (threadIdx.x < N) {
	int j = threadIdx.x & (N0-1);
	int i = threadIdx.x >> Rank0;
	int s = (N0+1)*i + j;   // Note shared memory layout
	shmem_i[s] = integer_constants[threadIdx.x];
    }
    
    __syncthreads();

    // (Shared memory) -> (Registers).
    // Note: two-to-one shared memory bank conflict here!
    // This is okay since it's just one-time initialization at the beginning of the kernel.

    const int w = (threadIdx.x >> 5);
    const int l = (threadIdx.x & 0x1f);
    
    // Assign (i,j) for shared memory.
    // Recall that i represents a coarse frequency 0 <= i < 2^rank1,
    // and j represents a bit-reversed DM 0 <= j < 2^rank0.

#if 0
    // This version has warp divergence but is easier to read.
    int i, j;
    
    if (l < 16) {
	int r = l & (N1-1);  // Register index 0 <= r < 2^rank1.
	i = ((w << Rank1) + r) >> Rank0;
	j = r & (N0-1);
    }
    else {
	int r = (l-16) & (N1-1);  // Register index 0 <= r < 2^rank1.
	i = r;
	j = w;
    }
#else
    // Version without warp divergence.
    bool ff = (l < 16);
    int r = (ff ? l : (l-16)) & (N1-1);
    int i = ff ? (((w << Rank1) + r) >> Rank0) : r;
    int j = ff ? (r & (N0-1)) : w;
#endif

    // Control words are stored in global memory at "writer offset".
    // To get "reader offset", set pos=0 by applying mask 0xff007fff.

    int s = (N0+1)*i + j;   // Note shared memory layout
    uint mask = (l < 16) ? 0xffffffff : 0xff007fff;
    uint control_word = shmem_i[s] & mask;

    return control_word;
}


// gmem_shmem_exchange(): copies (global memory ring buffer) <-> (shared memory).
//
// It's a bit ugly, but we use a boolean template argument to indicate copy direction:
//
//   gmem_shmem_exchange<true>   copy gmem -> shmem at beginning of kernel
//   gmem_shmem_exchange<false>  copy shmem -> gmem at end of kernel
//
// The 'gmem' pointer argument should point to the global memory ring buffer.
// This will be a subset of the rstate, and may be offset relative to the 'rstate'
// base pointer (depending on the rstate memory layout).
//
// On the other hand, the 'integer_constants' argument should be the base pointer,
// i.e. not offset by 256 to skip the control words. (I think this convention will
// be more convenient if I switch to using constant memory later.)
//
// Note: before calling gmem_shmem_exchange<false>() at end of kernel, must call
// align_ring_buffers(), see below.


template<typename T, int Rank, bool GmemToShmem>
__device__ inline void gmem_shmem_exchange(T *gmem, const uint *integer_constants)
{
    T *shmem = _shmem_base<T>();
    constexpr int gs_ncl = _gs_ncl<T,Rank>::value;
	
    const int warpId = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;
    const int laneId = threadIdx.x & 0x1f;
    const uint lane_mask = (1U << (laneId+1)) - 1;
    
    // FIXME each warp processes cache lines in "batches" of 16, which leads
    // to suboptimal load-balancing. I think this will naturally fix itself
    // if I switch to using constant memory.
    
    for (int icl0 = 16*warpId; icl0 < gs_ncl; icl0 += 16*nwarps) {
	uint gmem_spec = integer_constants[(1 << Rank) + 2*icl0 + laneId];
	int ninner = min(16, gs_ncl - icl0);
	
	for (int i = 0; i < ninner; i++) {
	    // Global memory index
	    int g = 32*(icl0+i) + laneId;

	    // Shared memory index
	    uint shmem_base = __shfl_sync(0xffffffff, gmem_spec, 2*i);
	    uint gap_bits = __shfl_sync(0xffffffff, gmem_spec, 2*i+1);
	    uint s = shmem_base + (__popc(gap_bits & lane_mask) << 5) + laneId;

	    if constexpr (GmemToShmem)
		shmem[s] = gmem[g];
	    else
		gmem[g] = shmem[s];
	}
    }
}


// Helper for align_ring_buffer().
// Warning: if the 'step' argument is > 32, this function will silently fail!!

__device__ inline int rb_advance(int rb_pos, int rb_size, int step)
{
    rb_pos += step;
    return (rb_pos >= rb_size) ? (rb_pos - rb_size) : rb_pos;
}


// Note: align_ring_buffer() assumes the ring buffer lag is <= 256 (or equivalently
// rb_size <= 288), so that eight 32-bit registers suffice to hold the contents of
// the ring buffer. This condition is checked in make_integer_constants() above.

__device__ inline void align_ring_buffer(uint control_word, int control_lane)
{
    const int laneId = (threadIdx.x & 0x1f);

    // Unpack control_word into (rb_base, rb_pos, rb_size).
    uint w = __shfl_sync(0xffffffff, control_word, control_lane);
    int rb_base = (w & 0x7fff);
    int rb_pos = ((w >> 15) & 0x1ff);
    int rb_size = (w >> 24) + 32;  // Note (+32) here (rb_lag -> rb_size)

    if (rb_size <= 32)
	return;

    rb_pos = rb_advance(rb_pos, rb_size, laneId);

    // We use 32-bit uints to hold data from the ring buffer. (The actual data is either
    // float or __half2, but it's opaque to align_ring_buffer().)

    extern __shared__ uint shmem_i[];
    uint x0, x1, x2, x3, x4, x5, x6, x7;

    // Read ring buffer.
    // The do-while loop here is ugly, but I thought a goto-statement was even uglier :)
    
    do {
	x0 = shmem_i[rb_base + rb_pos];
	
	if (rb_size <= 64)
	    break;

	rb_pos = rb_advance(rb_pos, rb_size, 32);
	x1 = shmem_i[rb_base + rb_pos];

	if (rb_size <= 96)
	    break;
	
	rb_pos = rb_advance(rb_pos, rb_size, 32);
	x2 = shmem_i[rb_base + rb_pos];
	
	if (rb_size <= 128)
	    break;
	
	rb_pos = rb_advance(rb_pos, rb_size, 32);
	x3 = shmem_i[rb_base + rb_pos];
	
	if (rb_size <= 160)
	    break;
	
	rb_pos = rb_advance(rb_pos, rb_size, 32);
	x4 = shmem_i[rb_base + rb_pos];
	
	if (rb_size <= 192)
	    break;
	
	rb_pos = rb_advance(rb_pos, rb_size, 32);
	x5 = shmem_i[rb_base + rb_pos];
	
	if (rb_size <= 224)
	    break;
	
	rb_pos = rb_advance(rb_pos, rb_size, 32);
	x6 = shmem_i[rb_base + rb_pos];
	
	if (rb_size <= 256)
	    break;
	
	rb_pos = rb_advance(rb_pos, rb_size, 32);
	x7 = shmem_i[rb_base + rb_pos];
    } while (0);

    // Write ring buffer.
    // Note: no __syncthreads() needed, since read/write is done on the same warp.

    shmem_i[rb_base + laneId] = x0;

    if (rb_size <= 64)
	return;

    shmem_i[rb_base + laneId + 32] = x1;
    
    if (rb_size <= 96)
	return;
    
    shmem_i[rb_base + laneId + 64] = x2;
    
    if (rb_size <= 128)
	return;
    
    shmem_i[rb_base + laneId + 96] = x3;
    
    if (rb_size <= 160)
	return;
    
    shmem_i[rb_base + laneId + 128] = x4;
    
    if (rb_size <= 192)
	return;
    
    shmem_i[rb_base + laneId + 160] = x5;
    
    if (rb_size <= 224)
	return;
    
    shmem_i[rb_base + laneId + 192] = x6;
    
    if (rb_size <= 256)
	return;
    
    shmem_i[rb_base + laneId + 224] = x7;
}

// align_all_ring_buffers(): called at end of kernel, to align all ring buffers to pos=0,
// before copying ring buffers to shared memory with gmem_shmem_exchange<false> ();

template<int Rank>
__device__ inline void align_all_ring_buffers(uint control_word)
{
    constexpr int Rank1 = Rank - (Rank >> 1);
    constexpr int N1 = (1 << Rank1);
    
    // Note: we use the "reader" control_words (16 <= laneId < 16+N1),
    // not the "writer" control words (0 <= laneId < N1), which have
    // the wrong positions.
    
    for (int control_lane = 16; control_lane < 16+N1; control_lane++)
	align_ring_buffer(control_word, control_lane);
}


// The next three device-side helper functions are for accessing the shared memory
// ring buffer using control words. See dedisperse_r8() below for usage.
//
// rbloc(): broadcasts control_word from specified control_lane to all threads,
//  and returns associated address in ring buffer (with laneId added).
//
// advance_control_word(): advances ring buffer position by 32.
//
// print_control_word(): for debugging
//
// Reminder: a ring buffer "control word" consists of
//   uint16 rb_base;   // base shared memory location of ring buffer (in 32-bit __half2s)
//   uint8  rb_pos;    // current position, satisfying 0 <= pos < size
//   uint8  rb_size;   // ring buffer size (in 32-bit __half2s), always equal to (lag+32)


__device__ inline int rbloc(uint control_word, int control_lane)
{    
    const int laneId = (threadIdx.x & 0x1f);
    
    uint w = __shfl_sync(0xffffffff, control_word, control_lane);
    uint rb_base = (w & 0x7fff);
    uint rb_pos = ((w >> 15) & 0x1ff) + laneId;   // Note laneId here
    uint rb_size = (w >> 24) + 32;                // Note (+32) here (rb_lag -> rb_size)
    uint downshift = (rb_pos >= rb_size) ? rb_size : 0;

    return rb_base + rb_pos - downshift;
}


__device__ inline uint advance_control_word(uint control_word)
{
    int pos15 = (control_word & 0xff8000);
    int lag15 = ((control_word >> 9) & 0xff8000);
    int dpos15 = (pos15 >= lag15) ? lag15 : (-(32 << 15));
    
    return (control_word & 0xff007fff) | (pos15 - dpos15);
}


// __device__ void print_control_word(uint cw)
// {
//     uint rb_base = (cw & 0x7fff);
//     uint rb_pos = ((cw >> 15) & 0x1ff);
//     uint rb_lag = (cw >> 24);
//     printf("rb_base=%d rb_pos=%d, rb_lag=%d\n", rb_base, rb_pos, rb_lag);
// }


// The next device-side helper functions are for applying one-sample lag.
// (The reason for this is explained near (*) above.)


// align1_s2():
//  - CycleFwd: advances 'rs' by 1
//  - CycleNone or CycleRev: do not advance 'rs'.

template<int C>
__device__ inline void align1_s2(__half2 &x, __half2 &y, __half2 &rs)
{
    __half2 t = __highs2half2(x, y);     // (x[1], y[1])
    __half2 u = lag_half2<1,C> (t, rs);  // (x[-1], y[-1])
    x = __lows2half2(u, x);  // (x[-1], x[0])
    y = f16_align(u, y);     // (y[-1], y[0])
}

// align1_s4()
//  - CycleFwd: advances 'rs' by 2
//  - CycleNone: advances 'rs' by 1

template<int C>
__device__ inline void align1_s4(__half2 &x0, __half2 &x1, __half2 &x2, __half2 &x3, __half2 &rs)
{
    align1_s2<CycleFwd> (x0, x1, rs);   // advances rs by 1
    align1_s2<CycleNone> (x2, x3, rs);  // does not advance rs

    if constexpr (C == CycleFwd)
	rs = __shfl_sync(0xffffffff, rs, threadIdx.x + 31);
    else if constexpr (C == CycleRev)
	rs = __shfl_sync(0xffffffff, rs, threadIdx.x + 1);
}

// align1_s4()
//  - CycleFwd: advances 'rs' by 2
//  - CycleNone: advances 'rs' by 1

template<int C>
__device__ inline void align1_s8(__half2 &x0, __half2 &x1, __half2 &x2, __half2 &x3,
				 __half2 &x4, __half2 &x5, __half2 &x6, __half2 &x7,
				 __half2 &rs)
{
    align1_s4<CycleFwd> (x0, x1, x2, x3, rs);   // advances rs by 2
    align1_s4<CycleNone> (x4, x5, x6, x7, rs);  // advances rs by 1
    
    if constexpr (C == CycleFwd)
	rs = __shfl_sync(0xffffffff, rs, threadIdx.x + 31);
    else if constexpr (C == CycleRev)
	rs = __shfl_sync(0xffffffff, rs, threadIdx.x + 3);
}


// -------------------------------------------------------------------------------------------------


template<typename T, class Inbuf, class Outbuf>
__global__ void __launch_bounds__(128, 8)
dedisperse_r5(typename Inbuf::device_args inbuf_args, typename Outbuf::device_args outbuf_args, T *rstate, int nt_cl, uint *integer_constants)
{
    static_assert(sizeof(T) == 4);  // float or __half2
    
    // 4 warps, 8 data registers/warp.
    constexpr int nwarps = 4;  // 2^rank0
    constexpr int nrdata = 8;  // 2^rank1
    constexpr int nthreads = 32 * nwarps;
    
    constexpr bool is_float32 = _is_float32<T>::value;
    constexpr int gs_ncl = _gs_ncl<T,5>::value;
    constexpr int nrs_per_thread = 1;
    constexpr int nrp_per_thread = Inbuf::is_lagged ? nrdata : 0;
    constexpr int nr_per_thread = nrs_per_thread + nrp_per_thread;
    constexpr int gmem_ncl = gs_ncl + nwarps * nr_per_thread;

    const int ambient_ix = blockIdx.x;
    const int beam_ix = blockIdx.y;
    
    typename Inbuf::device_state inbuf(inbuf_args, nrdata);
    typename Outbuf::device_state outbuf(outbuf_args);

    // Apply (beam, ambient) strides to rstate. (Note no laneId shift here.)
    rstate += beam_ix * gridDim.x * (32 * gmem_ncl);
    rstate += ambient_ix * (32 * gmem_ncl);
    
    // read_control_words() uses shared memory, so must precede gmem_shmem_exchange<true>().
    uint cw = read_control_words<5> (integer_constants);

    // Need __syncthreads() between read_control_words() and gmem_shmem_exchange().
    __syncthreads();

    // Restore register state (from previous chunk) from global memory.    
    T rs = rstate[threadIdx.x];
    
    // Temporarily disable nvcc warning "...variable was declared but never referenced"
#pragma nv_diag_suppress 177
    T xp0, xp1, xp2, xp3, xp4, xp5, xp6, xp7;
#pragma nv_diag_default 177

    if constexpr (Inbuf::is_lagged) {
	xp0 = rstate[threadIdx.x + nthreads * (nrs_per_thread)];
	xp1 = rstate[threadIdx.x + nthreads * (nrs_per_thread+1)];
	xp2 = rstate[threadIdx.x + nthreads * (nrs_per_thread+2)];
	xp3 = rstate[threadIdx.x + nthreads * (nrs_per_thread+3)];
	xp4 = rstate[threadIdx.x + nthreads * (nrs_per_thread+4)];
	xp5 = rstate[threadIdx.x + nthreads * (nrs_per_thread+5)];
	xp6 = rstate[threadIdx.x + nthreads * (nrs_per_thread+6)];
	xp7 = rstate[threadIdx.x + nthreads * (nrs_per_thread+7)];
    }
    
    // Restore shared memory ring buffer (from previous chunk) from global memory.
    gmem_shmem_exchange<T,5,true> (rstate + nthreads * nr_per_thread, integer_constants);
    __syncthreads();
    
    for (int it_cl = 0; it_cl < nt_cl; it_cl++) {
	
	// When reading the input array, we read from array index (2^rank0)*i + j.
	// Currently, i = warpId (might change later), and j = register index.
	
	T x0 = inbuf.load(0);
	T x1 = inbuf.load(1);
	T x2 = inbuf.load(2);
	T x3 = inbuf.load(3);
	T x4 = inbuf.load(4);
	T x5 = inbuf.load(5);
	T x6 = inbuf.load(6);
	T x7 = inbuf.load(7);
	inbuf.advance();

	if constexpr (Inbuf::is_lagged) {
	    // Ambient index represents a bit-reversed DM 0 <= d < 2^(ambient_rank).
	    // "Row" index represents a coarse frequency 0 <= f < 2^(rank).
	    // Residual lag is computed as folows:
	    //   int ff = 2^rank - 1 - f;
	    //   int N = is_float32 ? 32 : 64;
	    //   int rlag = (ff * d) % N

	    int ff0 = 31 - ((threadIdx.x & ~0x1f) >> 2);             // 31 - (8 * warpId)
	    int dm = __brev(blockIdx.x) >> (33 - __ffs(gridDim.x));  // bit-reversed DM
	    dm += inbuf_args._is_downsampled() ? gridDim.x : 0;

	    x0 = apply_rlag(x0, xp0, dm * ff0);
	    x1 = apply_rlag(x1, xp1, dm * (ff0-1));
	    x2 = apply_rlag(x2, xp2, dm * (ff0-2));
	    x3 = apply_rlag(x3, xp3, dm * (ff0-3));
	    x4 = apply_rlag(x4, xp4, dm * (ff0-4));
	    x5 = apply_rlag(x5, xp5, dm * (ff0-5));
	    x6 = apply_rlag(x6, xp6, dm * (ff0-6));
	    x7 = apply_rlag(x7, xp7, dm * (ff0-7));
	}

	// Rank-2 dedispersion stages.
	//   - Float32: 'rs' is advanced by 2*5=10 lanes.
	//   - Float16: 'rs' is advanced by 2*3=6 lanes.

	dd_r2<CycleFwd> (x0, x1, x2, x3, rs);
	dd_r2<CycleFwd> (x4, x5, x6, x7, rs);
	
	// Write/read shared memory ring buffer.

	T *shmem = _shmem_base<T>();

	shmem[ rbloc(cw,0) ] = x0;
	shmem[ rbloc(cw,1) ] = x1;
	shmem[ rbloc(cw,2) ] = x2;
	shmem[ rbloc(cw,3) ] = x3;
	shmem[ rbloc(cw,4) ] = x4;
	shmem[ rbloc(cw,5) ] = x5;
	shmem[ rbloc(cw,6) ] = x6;
	shmem[ rbloc(cw,7) ] = x7;
	
	__syncthreads();

	x0 = shmem[ rbloc(cw,16) ];
	x1 = shmem[ rbloc(cw,17) ];
	x2 = shmem[ rbloc(cw,18) ];
	x3 = shmem[ rbloc(cw,19) ];
	x4 = shmem[ rbloc(cw,20) ];
	x5 = shmem[ rbloc(cw,21) ];
	x6 = shmem[ rbloc(cw,22) ];
	x7 = shmem[ rbloc(cw,23) ];
	
	// In principle, we need one call to __syncthreads() somewhere after these
	// shared memory reads, and before the shared memory writes in the next loop
	// iteration. I wonder where the best place is to put it?
	
	__syncthreads();

	// As explained near (*) above, we need to supply additional one-sample lag
	// in channels where (register index i is even) and (warp index j >= 2).
	// Advances 'rs' by 2 lanes (cumulative 8).

	if constexpr (!is_float32) {
	    if (threadIdx.x >= 64)
		align1_s4<CycleFwd> (x0, x2, x4, x6, rs);
	}

	// Rank-3 dedispersion stage.
	//   - Float32: rs is advanced by 16 lanes (cumulative 26)
	//   - Float16: rs is advanced by 11 lanes (cumulative 19 or 17, depending on whether threadIdx.x >= 64)
	
	dd_r3<CycleNone> (x0, x1, x2, x3, x4, x5, x6, x7, rs);

	// Fully cycle 'rs'. This is a little awkward!
	if constexpr (is_float32) {
	    rs = __shfl_sync(0xffffffff, rs, threadIdx.x + 26);
	}
	else {
	    int ss = (threadIdx.x >= 64) ? 19 : 17;
	    rs = __shfl_sync(0xffffffff, rs, threadIdx.x + ss);
	}

	// When writing to the output array, we write to array index (2^rank0)*i + j.
	// Currently, j = warpId (might change later) and i = (register index).
	
	outbuf.store(0, x0);
	outbuf.store(4, x1);
	outbuf.store(2*4, x2);
	outbuf.store(3*4, x3);
	outbuf.store(4*4, x4);
	outbuf.store(5*4, x5);
	outbuf.store(6*4, x6);
	outbuf.store(7*4, x7);
	outbuf.advance();
	
	cw = advance_control_word(cw);
    }

    align_all_ring_buffers<5> (cw);

    // Need __syncthreads() between align_all_ring_buffers() and gmem_shmem_exchange().
    __syncthreads();

    // Save register state to global memory.
    rstate[threadIdx.x] = rs;
    
    if constexpr (Inbuf::is_lagged) {
	rstate[threadIdx.x + nthreads * (nrs_per_thread)] = xp0;
	rstate[threadIdx.x + nthreads * (nrs_per_thread+1)] = xp1;
	rstate[threadIdx.x + nthreads * (nrs_per_thread+2)] = xp2;
	rstate[threadIdx.x + nthreads * (nrs_per_thread+3)] = xp3;
	rstate[threadIdx.x + nthreads * (nrs_per_thread+4)] = xp4;
	rstate[threadIdx.x + nthreads * (nrs_per_thread+5)] = xp5;
	rstate[threadIdx.x + nthreads * (nrs_per_thread+6)] = xp6;
	rstate[threadIdx.x + nthreads * (nrs_per_thread+7)] = xp7;
    }

    // Save shared memory ring buffer to global memory.
    gmem_shmem_exchange<T,5,false> (rstate + nthreads * nr_per_thread, integer_constants);
}


template<typename T, class Inbuf, class Outbuf>
__global__ void __launch_bounds__(256, 4)
dedisperse_r6(typename Inbuf::device_args inbuf_args, typename Outbuf::device_args outbuf_args, T *rstate, int nt_cl, uint *integer_constants)
{
    static_assert(sizeof(T) == 4);  // float or __half2
    // assert(blockDim.x == 256);

    constexpr bool is_float32 = _is_float32<T>::value;
    constexpr int gs_ncl = _gs_ncl<T,6>::value;
    constexpr int nrs_per_thread = is_float32 ? 2 : 1;
    constexpr int nrp_per_thread = Inbuf::is_lagged ? 8 : 0;
    constexpr int nr_per_thread = nrs_per_thread + nrp_per_thread;
    constexpr int gmem_ncl = gs_ncl + 8 * nr_per_thread;

    const int ambient_ix = blockIdx.x;
    const int beam_ix = blockIdx.y;
    
    typename Inbuf::device_state inbuf(inbuf_args, 8);
    typename Outbuf::device_state outbuf(outbuf_args);

    // Apply (beam, ambient) strides to rstate. (Note no laneId shift here.)
    rstate += beam_ix * gridDim.x * (32 * gmem_ncl);
    rstate += ambient_ix * (32 * gmem_ncl);
    
    // read_control_words() uses shared memory, so must precede gmem_shmem_exchange<true>().
    uint cw = read_control_words<6> (integer_constants);

    // Need __syncthreads() between read_control_words() and gmem_shmem_exchange().
    __syncthreads();

    // Restore register state (from previous chunk) from global memory.    
    T rs = rstate[threadIdx.x];
    T rs2;

    if constexpr (is_float32)
	rs2 = rstate[threadIdx.x + 256];
    
    // Temporarily disable nvcc warning "...variable was declared but never referenced"
#pragma nv_diag_suppress 177
    T xp0, xp1, xp2, xp3, xp4, xp5, xp6, xp7;
#pragma nv_diag_default 177

    if constexpr (Inbuf::is_lagged) {
	xp0 = rstate[threadIdx.x + 256 * (nrs_per_thread)];
	xp1 = rstate[threadIdx.x + 256 * (nrs_per_thread+1)];
	xp2 = rstate[threadIdx.x + 256 * (nrs_per_thread+2)];
	xp3 = rstate[threadIdx.x + 256 * (nrs_per_thread+3)];
	xp4 = rstate[threadIdx.x + 256 * (nrs_per_thread+4)];
	xp5 = rstate[threadIdx.x + 256 * (nrs_per_thread+5)];
	xp6 = rstate[threadIdx.x + 256 * (nrs_per_thread+6)];
	xp7 = rstate[threadIdx.x + 256 * (nrs_per_thread+7)];
    }
    
    // Restore shared memory ring buffer (from previous chunk) from global memory.
    gmem_shmem_exchange<T,6,true> (rstate + 256*nr_per_thread, integer_constants);
    __syncthreads();
    
    for (int it_cl = 0; it_cl < nt_cl; it_cl++) {
	
	// When reading the input array, we read from array index (2^rank0)*i + j.
	// Currently, i = warpId (might change later), and j = register index.
	
	T x0 = inbuf.load(0);
	T x1 = inbuf.load(1);
	T x2 = inbuf.load(2);
	T x3 = inbuf.load(3);
	T x4 = inbuf.load(4);
	T x5 = inbuf.load(5);
	T x6 = inbuf.load(6);
	T x7 = inbuf.load(7);
	inbuf.advance();

	if constexpr (Inbuf::is_lagged) {
	    // Ambient index represents a bit-reversed DM 0 <= d < 2^(ambient_rank).
	    // "Row" index represents a coarse frequency 0 <= f < 2^(rank).
	    // Residual lag is computed as folows:
	    //   int ff = 2^rank - 1 - f;
	    //   int N = is_float32 ? 32 : 64;
	    //   int rlag = (ff * d) % N

	    int ff0 = 63 - ((threadIdx.x & ~0x1f) >> 2);             // 63 - (8 * warpId)
	    int dm = __brev(blockIdx.x) >> (33 - __ffs(gridDim.x));  // bit-reversed DM
	    dm += inbuf_args._is_downsampled() ? gridDim.x : 0;
	    
	    x0 = apply_rlag(x0, xp0, dm * ff0);
	    x1 = apply_rlag(x1, xp1, dm * (ff0-1));
	    x2 = apply_rlag(x2, xp2, dm * (ff0-2));
	    x3 = apply_rlag(x3, xp3, dm * (ff0-3));
	    x4 = apply_rlag(x4, xp4, dm * (ff0-4));
	    x5 = apply_rlag(x5, xp5, dm * (ff0-5));
	    x6 = apply_rlag(x6, xp6, dm * (ff0-6));
	    x7 = apply_rlag(x7, xp7, dm * (ff0-7));
	}
	
	// First rank-3 dedispersion stage.
	//   - Float32: 'rs' is fully consumed
	//   - Float16: 'rs' is advanced by 12 lanes.

	constexpr int C1 = is_float32 ? CycleRev : CycleFwd;
	dd_r3<C1> (x0, x1, x2, x3, x4, x5, x6, x7, rs);
	
	// Write/read shared memory ring buffer.

	T *shmem = _shmem_base<T>();

	shmem[ rbloc(cw,0) ] = x0;
	shmem[ rbloc(cw,1) ] = x1;
	shmem[ rbloc(cw,2) ] = x2;
	shmem[ rbloc(cw,3) ] = x3;
	shmem[ rbloc(cw,4) ] = x4;
	shmem[ rbloc(cw,5) ] = x5;
	shmem[ rbloc(cw,6) ] = x6;
	shmem[ rbloc(cw,7) ] = x7;

	__syncthreads();

	x0 = shmem[ rbloc(cw,16) ];
	x1 = shmem[ rbloc(cw,17) ];
	x2 = shmem[ rbloc(cw,18) ];
	x3 = shmem[ rbloc(cw,19) ];
	x4 = shmem[ rbloc(cw,20) ];
	x5 = shmem[ rbloc(cw,21) ];
	x6 = shmem[ rbloc(cw,22) ];
	x7 = shmem[ rbloc(cw,23) ];

	// In principle, we need one call to __syncthreads() somewhere after these
	// shared memory reads, and before the shared memory writes in the next loop
	// iteration. I wonder where the best place is to put it?
	
	__syncthreads();

	// As explained near (*) above, we need to supply additional one-sample lag
	// in channels where (register index i is even) and (warp index j >= 3).
	// Advances 'rs' by 2 lanes (cumulative 14).

	if constexpr (!is_float32) {
	    if (threadIdx.x >= 128)
		align1_s4<CycleFwd> (x0, x2, x4, x6, rs);
	}

	// Second rank-3 dedispersion stage.
	// The register-cycling logic here is a little awward!
	//   - Float32: (rs,rs2) are fully consumed
	//   - Float16: rs2 is an alias for rs, and is advanced by 11 lanes (cumulative 23 or 25).
	
	if constexpr (!is_float32)
	    rs2 = rs;
	
	constexpr int C2 = is_float32 ? CycleRev : CycleNone;
	dd_r3<C2> (x0, x1, x2, x3, x4, x5, x6, x7, rs2);
	
	if constexpr (!is_float32) {
	    int ss = (threadIdx.x >= 128) ? 25 : 23;
	    rs = __shfl_sync(0xffffffff, rs2, threadIdx.x + ss);  // Note rs = __shfl_sync(..., rs2, ...) here
	}

	// When writing to the output array, we write to array index (2^rank0)*i + j.
	// Currently, j = warpId (might change later) and i = (register index).
	
	outbuf.store(0, x0);
	outbuf.store(8, x1);
	outbuf.store(2*8, x2);
	outbuf.store(3*8, x3);
	outbuf.store(4*8, x4);
	outbuf.store(5*8, x5);
	outbuf.store(6*8, x6);
	outbuf.store(7*8, x7);
	outbuf.advance();

	cw = advance_control_word(cw);
    }

    align_all_ring_buffers<6> (cw);

    // Need __syncthreads() between align_all_ring_buffers() and gmem_shmem_exchange().
    __syncthreads();

    // Save register state to global memory.
    rstate[threadIdx.x] = rs;

    if constexpr (is_float32)
	rstate[threadIdx.x + 256] = rs2;
    
    if constexpr (Inbuf::is_lagged) {
	rstate[threadIdx.x + 256 * (nrs_per_thread)] = xp0;
	rstate[threadIdx.x + 256 * (nrs_per_thread+1)] = xp1;
	rstate[threadIdx.x + 256 * (nrs_per_thread+2)] = xp2;
	rstate[threadIdx.x + 256 * (nrs_per_thread+3)] = xp3;
	rstate[threadIdx.x + 256 * (nrs_per_thread+4)] = xp4;
	rstate[threadIdx.x + 256 * (nrs_per_thread+5)] = xp5;
	rstate[threadIdx.x + 256 * (nrs_per_thread+6)] = xp6;
	rstate[threadIdx.x + 256 * (nrs_per_thread+7)] = xp7;
    }

    // Save shared memory ring buffer to global memory.
    gmem_shmem_exchange<T,6,false> (rstate + 256*nr_per_thread, integer_constants);
}


template<typename T, class Inbuf, class Outbuf>
__global__ void __launch_bounds__(256, 3)
dedisperse_r7(typename Inbuf::device_args inbuf_args, typename Outbuf::device_args outbuf_args, T *rstate, int nt_cl, uint *integer_constants)
{
    static_assert(sizeof(T) == 4);  // float or __half2

    // 8 warps, 16 data registers/warp
    constexpr int nwarps = 8;   // 2^rank0
    constexpr int nrdata = 16;  // 2^rank1
    constexpr int nthreads = 32 * nwarps;
    
    constexpr bool is_float32 = _is_float32<T>::value;
    constexpr int gs_ncl = _gs_ncl<T,7>::value;
    constexpr int nrs_per_thread = is_float32 ? 4 : 3;
    constexpr int nrp_per_thread = Inbuf::is_lagged ? nrdata : 0;
    constexpr int nr_per_thread = nrs_per_thread + nrp_per_thread;
    constexpr int gmem_ncl = gs_ncl + nwarps * nr_per_thread;

    const int ambient_ix = blockIdx.x;
    const int beam_ix = blockIdx.y;
    
    typename Inbuf::device_state inbuf(inbuf_args, nrdata);
    typename Outbuf::device_state outbuf(outbuf_args);
    
    // Apply (beam, ambient) strides to rstate. (Note no laneId shift here.)
    rstate += beam_ix * gridDim.x * (32 * gmem_ncl);
    rstate += ambient_ix * (32 * gmem_ncl);
    
    // read_control_words() uses shared memory, so must precede gmem_shmem_exchange<true>().
    uint cw = read_control_words<7> (integer_constants);

    // Need __syncthreads() between read_control_words() and gmem_shmem_exchange().
    __syncthreads();

    // Restore register state (from previous chunk) from global memory.
    // FIXME global memory layout for 'rs4' is a little wasteful in float16 case,
    // but this is awkward to change.
    
    T rs = rstate[threadIdx.x];
    T rs2 = rstate[threadIdx.x + nthreads];
    T rs3 = rstate[threadIdx.x + 2*nthreads];
    T rs4;

    if constexpr (is_float32)
	rs4 = rstate[threadIdx.x + 3*nthreads];
    
    // Temporarily disable nvcc warning "...variable was declared but never referenced"
#pragma nv_diag_suppress 177
    T xp0, xp1, xp2, xp3, xp4, xp5, xp6, xp7, xp8, xp9, xp10, xp11, xp12, xp13, xp14, xp15;
#pragma nv_diag_default 177

    if constexpr (Inbuf::is_lagged) {
	xp0 = rstate[threadIdx.x + nthreads * (nrs_per_thread)];
	xp1 = rstate[threadIdx.x + nthreads * (nrs_per_thread+1)];
	xp2 = rstate[threadIdx.x + nthreads * (nrs_per_thread+2)];
	xp3 = rstate[threadIdx.x + nthreads * (nrs_per_thread+3)];
	xp4 = rstate[threadIdx.x + nthreads * (nrs_per_thread+4)];
	xp5 = rstate[threadIdx.x + nthreads * (nrs_per_thread+5)];
	xp6 = rstate[threadIdx.x + nthreads * (nrs_per_thread+6)];
	xp7 = rstate[threadIdx.x + nthreads * (nrs_per_thread+7)];
	xp8 = rstate[threadIdx.x + nthreads * (nrs_per_thread+8)];
	xp9 = rstate[threadIdx.x + nthreads * (nrs_per_thread+9)];
	xp10 = rstate[threadIdx.x + nthreads * (nrs_per_thread+10)];
	xp11 = rstate[threadIdx.x + nthreads * (nrs_per_thread+11)];
	xp12 = rstate[threadIdx.x + nthreads * (nrs_per_thread+12)];
	xp13 = rstate[threadIdx.x + nthreads * (nrs_per_thread+13)];
	xp14 = rstate[threadIdx.x + nthreads * (nrs_per_thread+14)];
	xp15 = rstate[threadIdx.x + nthreads * (nrs_per_thread+15)];
    }
    
    // Restore shared memory ring buffer (from previous chunk) from global memory.
    gmem_shmem_exchange<T,7,true> (rstate + nthreads * nr_per_thread, integer_constants);
    __syncthreads();
    
    for (int it_cl = 0; it_cl < nt_cl; it_cl++) {
	
	// When reading the input array, we read from array index (2^rank0)*i + j.
	// Currently, i = warpId (might change later), and j = register index.

	T x0 = inbuf.load(0);
	T x1 = inbuf.load(1);
	T x2 = inbuf.load(2);
	T x3 = inbuf.load(3);
	T x4 = inbuf.load(4);
	T x5 = inbuf.load(5);
	T x6 = inbuf.load(6);
	T x7 = inbuf.load(7);
	T x8 = inbuf.load(8);
	T x9 = inbuf.load(9);
	T x10 = inbuf.load(10);
	T x11 = inbuf.load(11);
	T x12 = inbuf.load(12);
	T x13 = inbuf.load(13);
	T x14 = inbuf.load(14);
	T x15 = inbuf.load(15);
	inbuf.advance();

	if constexpr (Inbuf::is_lagged) {
	    // Ambient index represents a bit-reversed DM 0 <= d < 2^(ambient_rank).
	    // "Row" index represents a coarse frequency 0 <= f < 2^(rank).
	    // Residual lag is computed as folows:
	    //   int ff = 2^rank - 1 - f;
	    //   int N = is_float32 ? 32 : 64;
	    //   int rlag = (ff * d) % N

	    int ff0 = 127 - ((threadIdx.x & ~0x1f) >> 1);            // 127 - (16 * warpId)
	    int dm = __brev(blockIdx.x) >> (33 - __ffs(gridDim.x));  // bit-reversed DM
	    dm += inbuf_args._is_downsampled() ? gridDim.x : 0;

	    x0 = apply_rlag(x0, xp0, dm * ff0);
	    x1 = apply_rlag(x1, xp1, dm * (ff0-1));
	    x2 = apply_rlag(x2, xp2, dm * (ff0-2));
	    x3 = apply_rlag(x3, xp3, dm * (ff0-3));
	    x4 = apply_rlag(x4, xp4, dm * (ff0-4));
	    x5 = apply_rlag(x5, xp5, dm * (ff0-5));
	    x6 = apply_rlag(x6, xp6, dm * (ff0-6));
	    x7 = apply_rlag(x7, xp7, dm * (ff0-7));
	    x8 = apply_rlag(x8, xp8, dm * (ff0-8));
	    x9 = apply_rlag(x9, xp9, dm * (ff0-9));
	    x10 = apply_rlag(x10, xp10, dm * (ff0-10));
	    x11 = apply_rlag(x11, xp11, dm * (ff0-11));
	    x12 = apply_rlag(x12, xp12, dm * (ff0-12));
	    x13 = apply_rlag(x13, xp13, dm * (ff0-13));
	    x14 = apply_rlag(x14, xp14, dm * (ff0-14));
	    x15 = apply_rlag(x15, xp15, dm * (ff0-15));
	}

	// Rank-3 dedispersion stages.
	
	dd_r3<CycleRev> (x0, x1, x2, x3, x4, x5, x6, x7, rs);
	dd_r3<CycleFwd> (x8, x9, x10, x11, x12, x13, x14, x15, rs2);

	// Register lane counts at this point in the code:
	//   - Float32: (rs1,rs2,rs3,rs4) = (32,20,0,0).
	//   - Float16: (rs1,rs2,rs3) = (32,12,0).

	// Write/read shared memory ring buffer.

	T *shmem = _shmem_base<T>();

	shmem[ rbloc(cw,0) ] = x0;
	shmem[ rbloc(cw,1) ] = x1;
	shmem[ rbloc(cw,2) ] = x2;
	shmem[ rbloc(cw,3) ] = x3;
	shmem[ rbloc(cw,4) ] = x4;
	shmem[ rbloc(cw,5) ] = x5;
	shmem[ rbloc(cw,6) ] = x6;
	shmem[ rbloc(cw,7) ] = x7;
	shmem[ rbloc(cw,8) ] = x8;
	shmem[ rbloc(cw,9) ] = x9;
	shmem[ rbloc(cw,10) ] = x10;
	shmem[ rbloc(cw,11) ] = x11;
	shmem[ rbloc(cw,12) ] = x12;
	shmem[ rbloc(cw,13) ] = x13;
	shmem[ rbloc(cw,14) ] = x14;
	shmem[ rbloc(cw,15) ] = x15;

	__syncthreads();

	x0 = shmem[ rbloc(cw,16) ];
	x1 = shmem[ rbloc(cw,17) ];
	x2 = shmem[ rbloc(cw,18) ];
	x3 = shmem[ rbloc(cw,19) ];
	x4 = shmem[ rbloc(cw,20) ];
	x5 = shmem[ rbloc(cw,21) ];
	x6 = shmem[ rbloc(cw,22) ];
	x7 = shmem[ rbloc(cw,23) ];
	x8 = shmem[ rbloc(cw,24) ];
	x9 = shmem[ rbloc(cw,25) ];
	x10 = shmem[ rbloc(cw,26) ];
	x11 = shmem[ rbloc(cw,27) ];
	x12 = shmem[ rbloc(cw,28) ];
	x13 = shmem[ rbloc(cw,29) ];
	x14 = shmem[ rbloc(cw,30) ];
	x15 = shmem[ rbloc(cw,31) ];

	// In principle, we need one call to __syncthreads() somewhere after these
	// shared memory reads, and before the shared memory writes in the next loop
	// iteration. I wonder where the best place is to put it?
	
	__syncthreads();

	// As explained near (*) above, we need to supply additional one-sample lag
	// in channels where (register index i is even) and (warp index j >= 4).

	if constexpr (!is_float32) {
	    // Advances 'rs2' by 4 lanes.
	    if (threadIdx.x >= 128)
		align1_s8<CycleFwd> (x0, x2, x4, x6, x8, x10, x12, x14, rs2);
	}

	// Register lane counts at this point in the code:
	//   - Float32: (rs1,rs2,rs3,rs4) = (32,20,0,0).
	//   - Float16 (threadId >= 128): (rs1,rs2,rs3) = (32,16,0).
	//   - Float16 (threadId < 128):  (rs1,rs2,rs3) = (32,12,0).

	// Rank-4 dedispersion stage.
	
	constexpr int C = is_float32 ? CycleFwd : CycleNone;
	dd_r4<C> (x0, x1, x2, x3, x4, x5, x6, x7,
		  x8, x9, x10, x11, x12, x13, x14, x15,
		  rs3, rs4, rs2);
	
	// Register lane counts at this point in the code:
	//   - Float32: (rs1,rs2,rs3,rs4) = (32,32,32,32).
	//   - Float16 (threadId >= 128): (rs1,rs2,rs3) = (32,28,32).
	//   - Float16 (threadId < 128):  (rs1,rs2,rs3) = (32,24,32).

	if constexpr (!is_float32) {
	    int ss = (threadIdx.x >= 128) ? 28 : 24;
	    rs2 = __shfl_sync(0xffffffff, rs2, threadIdx.x + ss);
	}

	// When writing to the output array, we write to array index (2^rank0)*i + j.
	// Currently, j = warpId (might change later) and i = (register index).

	outbuf.store(0, x0);
	outbuf.store(8, x1);
	outbuf.store(2*8, x2);
	outbuf.store(3*8, x3);
	outbuf.store(4*8, x4);
	outbuf.store(5*8, x5);
	outbuf.store(6*8, x6);
	outbuf.store(7*8, x7);
	outbuf.store(8*8, x8);
	outbuf.store(9*8, x9);
	outbuf.store(10*8, x10);
	outbuf.store(11*8, x11);
	outbuf.store(12*8, x12);
	outbuf.store(13*8, x13);
	outbuf.store(14*8, x14);
	outbuf.store(15*8, x15);
	outbuf.advance();

	cw = advance_control_word(cw);
    }

    align_all_ring_buffers<7> (cw);

    // Need __syncthreads() between align_all_ring_buffers() and gmem_shmem_exchange().
    __syncthreads();

    // Save register state to global memory.
    rstate[threadIdx.x] = rs;
    rstate[threadIdx.x + nthreads] = rs2;
    rstate[threadIdx.x + 2*nthreads] = rs3;
    
    if constexpr (is_float32)
	rstate[threadIdx.x + 3*nthreads] = rs4;
    
    if constexpr (Inbuf::is_lagged) {
	rstate[threadIdx.x + nthreads * (nrs_per_thread)] = xp0;
	rstate[threadIdx.x + nthreads * (nrs_per_thread+1)] = xp1;
	rstate[threadIdx.x + nthreads * (nrs_per_thread+2)] = xp2;
	rstate[threadIdx.x + nthreads * (nrs_per_thread+3)] = xp3;
	rstate[threadIdx.x + nthreads * (nrs_per_thread+4)] = xp4;
	rstate[threadIdx.x + nthreads * (nrs_per_thread+5)] = xp5;
	rstate[threadIdx.x + nthreads * (nrs_per_thread+6)] = xp6;
	rstate[threadIdx.x + nthreads * (nrs_per_thread+7)] = xp7;
	rstate[threadIdx.x + nthreads * (nrs_per_thread+8)] = xp8;
	rstate[threadIdx.x + nthreads * (nrs_per_thread+9)] = xp9;
	rstate[threadIdx.x + nthreads * (nrs_per_thread+10)] = xp10;
	rstate[threadIdx.x + nthreads * (nrs_per_thread+11)] = xp11;
	rstate[threadIdx.x + nthreads * (nrs_per_thread+12)] = xp12;
	rstate[threadIdx.x + nthreads * (nrs_per_thread+13)] = xp13;
	rstate[threadIdx.x + nthreads * (nrs_per_thread+14)] = xp14;
	rstate[threadIdx.x + nthreads * (nrs_per_thread+15)] = xp15;
    }

    // Save shared memory ring buffer to global memory.
    gmem_shmem_exchange<T,7,false> (rstate + nthreads * nr_per_thread, integer_constants);
}


template<typename T, class Inbuf, class Outbuf>
__global__ void __launch_bounds__(512, 1)
dedisperse_r8(typename Inbuf::device_args inbuf_args, typename Outbuf::device_args outbuf_args, T *rstate, int nt_cl, uint *integer_constants)
{
    static_assert(sizeof(T) == 4);  // float or __half2
    // assert(blockDim.x == 512);

    constexpr bool is_float32 = _is_float32<T>::value;
    constexpr int gs_ncl = _gs_ncl<T,8>::value;
    constexpr int nrs_per_thread = is_float32 ? 5 : 4;
    constexpr int nrp_per_thread = Inbuf::is_lagged ? 16 : 0;
    constexpr int nr_per_thread = nrs_per_thread + nrp_per_thread;
    constexpr int gmem_ncl = gs_ncl + 16 * nr_per_thread;

    const int ambient_ix = blockIdx.x;
    const int beam_ix = blockIdx.y;
    
    typename Inbuf::device_state inbuf(inbuf_args, 16);
    typename Outbuf::device_state outbuf(outbuf_args);

    // Apply (beam, ambient) strides to rstate. (Note no laneId shift here.)
    rstate += beam_ix * gridDim.x * (32 * gmem_ncl);
    rstate += ambient_ix * (32 * gmem_ncl);
    
    // read_control_words() uses shared memory, so must precede gmem_shmem_exchange<true>().
    uint cw = read_control_words<8> (integer_constants);

    // Need __syncthreads() between read_control_words() and gmem_shmem_exchange().
    __syncthreads();

    // Restore register state (from previous chunk) from global memory.
    // FIXME global memory layout for 'rs4' is a little wasteful in float16 case,
    // but this is awkward to change.
    
    T rs = rstate[threadIdx.x];
    T rs2 = rstate[threadIdx.x + 512];
    T rs3 = rstate[threadIdx.x + 2*512];
    T rs4 = rstate[threadIdx.x + 3*512];
    T rs5;

    if constexpr (is_float32)
	rs5 = rstate[threadIdx.x + 4*512];
    
    // Temporarily disable nvcc warning "...variable was declared but never referenced"
#pragma nv_diag_suppress 177
    T xp0, xp1, xp2, xp3, xp4, xp5, xp6, xp7, xp8, xp9, xp10, xp11, xp12, xp13, xp14, xp15;
#pragma nv_diag_default 177

    if constexpr (Inbuf::is_lagged) {
	xp0 = rstate[threadIdx.x + 512 * (nrs_per_thread)];
	xp1 = rstate[threadIdx.x + 512 * (nrs_per_thread+1)];
	xp2 = rstate[threadIdx.x + 512 * (nrs_per_thread+2)];
	xp3 = rstate[threadIdx.x + 512 * (nrs_per_thread+3)];
	xp4 = rstate[threadIdx.x + 512 * (nrs_per_thread+4)];
	xp5 = rstate[threadIdx.x + 512 * (nrs_per_thread+5)];
	xp6 = rstate[threadIdx.x + 512 * (nrs_per_thread+6)];
	xp7 = rstate[threadIdx.x + 512 * (nrs_per_thread+7)];
	xp8 = rstate[threadIdx.x + 512 * (nrs_per_thread+8)];
	xp9 = rstate[threadIdx.x + 512 * (nrs_per_thread+9)];
	xp10 = rstate[threadIdx.x + 512 * (nrs_per_thread+10)];
	xp11 = rstate[threadIdx.x + 512 * (nrs_per_thread+11)];
	xp12 = rstate[threadIdx.x + 512 * (nrs_per_thread+12)];
	xp13 = rstate[threadIdx.x + 512 * (nrs_per_thread+13)];
	xp14 = rstate[threadIdx.x + 512 * (nrs_per_thread+14)];
	xp15 = rstate[threadIdx.x + 512 * (nrs_per_thread+15)];
    }
    
    // Restore shared memory ring buffer (from previous chunk) from global memory.
    gmem_shmem_exchange<T,8,true> (rstate + 512*nr_per_thread, integer_constants);
    __syncthreads();
    
    for (int it_cl = 0; it_cl < nt_cl; it_cl++) {
	
	// When reading the input array, we read from array index (2^rank0)*i + j.
	// Currently, i = warpId (might change later), and j = register index.
	
	T x0 = inbuf.load(0);
	T x1 = inbuf.load(1);
	T x2 = inbuf.load(2);
	T x3 = inbuf.load(3);
	T x4 = inbuf.load(4);
	T x5 = inbuf.load(5);
	T x6 = inbuf.load(6);
	T x7 = inbuf.load(7);
	T x8 = inbuf.load(8);
	T x9 = inbuf.load(9);
	T x10 = inbuf.load(10);
	T x11 = inbuf.load(11);
	T x12 = inbuf.load(12);
	T x13 = inbuf.load(13);
	T x14 = inbuf.load(14);
	T x15 = inbuf.load(15);
	inbuf.advance();

	if constexpr (Inbuf::is_lagged) {
	    // Ambient index represents a bit-reversed DM 0 <= d < 2^(ambient_rank).
	    // "Row" index represents a coarse frequency 0 <= f < 2^(rank).
	    // Residual lag is computed as folows:
	    //   int ff = 2^rank - 1 - f;
	    //   int N = is_float32 ? 32 : 64;
	    //   int rlag = (ff * d) % N

	    int ff0 = 255 - ((threadIdx.x & ~0x1f) >> 1);            // 255 - (16 * warpId)
	    int dm = __brev(blockIdx.x) >> (33 - __ffs(gridDim.x));  // bit-reversed DM
	    dm += inbuf_args._is_downsampled() ? gridDim.x : 0;

	    x0 = apply_rlag(x0, xp0, dm * ff0);
	    x1 = apply_rlag(x1, xp1, dm * (ff0-1));
	    x2 = apply_rlag(x2, xp2, dm * (ff0-2));
	    x3 = apply_rlag(x3, xp3, dm * (ff0-3));
	    x4 = apply_rlag(x4, xp4, dm * (ff0-4));
	    x5 = apply_rlag(x5, xp5, dm * (ff0-5));
	    x6 = apply_rlag(x6, xp6, dm * (ff0-6));
	    x7 = apply_rlag(x7, xp7, dm * (ff0-7));
	    x8 = apply_rlag(x8, xp8, dm * (ff0-8));
	    x9 = apply_rlag(x9, xp9, dm * (ff0-9));
	    x10 = apply_rlag(x10, xp10, dm * (ff0-10));
	    x11 = apply_rlag(x11, xp11, dm * (ff0-11));
	    x12 = apply_rlag(x12, xp12, dm * (ff0-12));
	    x13 = apply_rlag(x13, xp13, dm * (ff0-13));
	    x14 = apply_rlag(x14, xp14, dm * (ff0-14));
	    x15 = apply_rlag(x15, xp15, dm * (ff0-15));
	}
	
	// First rank-4 dedispersion stage.
	// The register-cycling logic here is a little convoluted!
	//   - Float32: 'rs' and 'rs4' are fully consumed, and 'rs3' is advanced by 12 lanes.
	//   - Float16: 'rs' is fully consumed, 'rs4' is an unused placeholder, and 'rs3' is advanced by 16 lanes.
		
	dd_r4<CycleFwd> (x0, x1, x2, x3, x4, x5, x6, x7,
			 x8, x9, x10, x11, x12, x13, x14, x15,
			 rs, rs4, rs3);
	
	// Write/read shared memory ring buffer.

	T *shmem = _shmem_base<T>();

	shmem[ rbloc(cw,0) ] = x0;
	shmem[ rbloc(cw,1) ] = x1;
	shmem[ rbloc(cw,2) ] = x2;
	shmem[ rbloc(cw,3) ] = x3;
	shmem[ rbloc(cw,4) ] = x4;
	shmem[ rbloc(cw,5) ] = x5;
	shmem[ rbloc(cw,6) ] = x6;
	shmem[ rbloc(cw,7) ] = x7;
	shmem[ rbloc(cw,8) ] = x8;
	shmem[ rbloc(cw,9) ] = x9;
	shmem[ rbloc(cw,10) ] = x10;
	shmem[ rbloc(cw,11) ] = x11;
	shmem[ rbloc(cw,12) ] = x12;
	shmem[ rbloc(cw,13) ] = x13;
	shmem[ rbloc(cw,14) ] = x14;
	shmem[ rbloc(cw,15) ] = x15;

	__syncthreads();

	x0 = shmem[ rbloc(cw,16) ];
	x1 = shmem[ rbloc(cw,17) ];
	x2 = shmem[ rbloc(cw,18) ];
	x3 = shmem[ rbloc(cw,19) ];
	x4 = shmem[ rbloc(cw,20) ];
	x5 = shmem[ rbloc(cw,21) ];
	x6 = shmem[ rbloc(cw,22) ];
	x7 = shmem[ rbloc(cw,23) ];
	x8 = shmem[ rbloc(cw,24) ];
	x9 = shmem[ rbloc(cw,25) ];
	x10 = shmem[ rbloc(cw,26) ];
	x11 = shmem[ rbloc(cw,27) ];
	x12 = shmem[ rbloc(cw,28) ];
	x13 = shmem[ rbloc(cw,29) ];
	x14 = shmem[ rbloc(cw,30) ];
	x15 = shmem[ rbloc(cw,31) ];

	// In principle, we need one call to __syncthreads() somewhere after these
	// shared memory reads, and before the shared memory writes in the next loop
	// iteration. I wonder where the best place is to put it?
	
	__syncthreads();

	// As explained near (*) above, we need to supply additional one-sample lag
	// in channels where (register index i is even) and (warp index j >= 8).

	if constexpr (!is_float32) {
	    if (threadIdx.x >= 256)
		align1_s8<CycleRev> (x0, x2, x4, x6, x8, x10, x12, x14, rs4);
	}

	// Second rank-4 dedispersion stage.
	// The register-cycling logic here is a little convoluted!
	//   - Float32: (rs,rs2,rs4,rs5) are fully consumed, and rs3 is advanced by 12+4=16 lanes.
	//   - Float16: (rs,rs2) are fully consumed, (rs4,rs5) are unused placeholders, rs3 is advanced by 16+12=28 lanes.

	dd_r4<CycleNone> (x0, x1, x2, x3, x4, x5, x6, x7,
			  x8, x9, x10, x11, x12, x13, x14, x15,
			  rs2, rs5, rs3);
	
	// To finish cycling rs3, we need this awkward line of code.
	constexpr int ss = is_float32 ? 16 : 28;
	rs3 = __shfl_sync(0xffffffff, rs3, threadIdx.x + ss);

	// When writing to the output array, we write to array index (2^rank0)*i + j.
	// Currently, j = warpId (might change later) and i = (register index).
	
	outbuf.store(0, x0);
	outbuf.store(16, x1);
	outbuf.store(2*16, x2);
	outbuf.store(3*16, x3);
	outbuf.store(4*16, x4);
	outbuf.store(5*16, x5);
	outbuf.store(6*16, x6);
	outbuf.store(7*16, x7);
	outbuf.store(8*16, x8);
	outbuf.store(9*16, x9);
	outbuf.store(10*16, x10);
	outbuf.store(11*16, x11);
	outbuf.store(12*16, x12);
	outbuf.store(13*16, x13);
	outbuf.store(14*16, x14);
	outbuf.store(15*16, x15);
	outbuf.advance();

	cw = advance_control_word(cw);
    }

    align_all_ring_buffers<8> (cw);

    // Need __syncthreads() between align_all_ring_buffers() and gmem_shmem_exchange().
    __syncthreads();

    // Save register state to global memory.
    rstate[threadIdx.x] = rs;
    rstate[threadIdx.x + 512] = rs2;
    rstate[threadIdx.x + 2*512] = rs3;
    rstate[threadIdx.x + 3*512] = rs4;
    
    if constexpr (is_float32)
	rstate[threadIdx.x + 4*512] = rs5;
    
    if constexpr (Inbuf::is_lagged) {
	rstate[threadIdx.x + 512 * (nrs_per_thread)] = xp0;
	rstate[threadIdx.x + 512 * (nrs_per_thread+1)] = xp1;
	rstate[threadIdx.x + 512 * (nrs_per_thread+2)] = xp2;
	rstate[threadIdx.x + 512 * (nrs_per_thread+3)] = xp3;
	rstate[threadIdx.x + 512 * (nrs_per_thread+4)] = xp4;
	rstate[threadIdx.x + 512 * (nrs_per_thread+5)] = xp5;
	rstate[threadIdx.x + 512 * (nrs_per_thread+6)] = xp6;
	rstate[threadIdx.x + 512 * (nrs_per_thread+7)] = xp7;
	rstate[threadIdx.x + 512 * (nrs_per_thread+8)] = xp8;
	rstate[threadIdx.x + 512 * (nrs_per_thread+9)] = xp9;
	rstate[threadIdx.x + 512 * (nrs_per_thread+10)] = xp10;
	rstate[threadIdx.x + 512 * (nrs_per_thread+11)] = xp11;
	rstate[threadIdx.x + 512 * (nrs_per_thread+12)] = xp12;
	rstate[threadIdx.x + 512 * (nrs_per_thread+13)] = xp13;
	rstate[threadIdx.x + 512 * (nrs_per_thread+14)] = xp14;
	rstate[threadIdx.x + 512 * (nrs_per_thread+15)] = xp15;
    }

    // Save shared memory ring buffer to global memory.
    gmem_shmem_exchange<T,8,false> (rstate + 512*nr_per_thread, integer_constants);
}


#define INSTANTIATE_DEDISPERSION_KERNELS(T, Inbuf, Outbuf) \
    template __global__ void dedisperse_r1<T, Inbuf, Outbuf> (Inbuf::device_args, Outbuf::device_args, T *rstate, int nt_cl, uint *integer_constants); \
    template __global__ void dedisperse_r2<T, Inbuf, Outbuf> (Inbuf::device_args, Outbuf::device_args, T *rstate, int nt_cl, uint *integer_constants); \
    template __global__ void dedisperse_r3<T, Inbuf, Outbuf> (Inbuf::device_args, Outbuf::device_args, T *rstate, int nt_cl, uint *integer_constants); \
    template __global__ void dedisperse_r4<T, Inbuf, Outbuf> (Inbuf::device_args, Outbuf::device_args, T *rstate, int nt_cl, uint *integer_constants); \
    template __global__ void dedisperse_r5<T, Inbuf, Outbuf> (Inbuf::device_args, Outbuf::device_args, T *rstate, int nt_cl, uint *integer_constants); \
    template __global__ void dedisperse_r6<T, Inbuf, Outbuf> (Inbuf::device_args, Outbuf::device_args, T *rstate, int nt_cl, uint *integer_constants); \
    template __global__ void dedisperse_r7<T, Inbuf, Outbuf> (Inbuf::device_args, Outbuf::device_args, T *rstate, int nt_cl, uint *integer_constants); \
    template __global__ void dedisperse_r8<T, Inbuf, Outbuf> (Inbuf::device_args, Outbuf::device_args, T *rstate, int nt_cl, uint *integer_constants)


}  // namespace pirate

#endif // _PIRATE_INTERNALS_DEDISPERSION_KERNEL_IMPLEMENTATION_HPP
