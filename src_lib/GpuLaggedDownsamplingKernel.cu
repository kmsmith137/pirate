#include "../include/pirate/internals/GpuLaggedDownsamplingKernel.hpp"
#include "../include/pirate/internals/inlines.hpp"   // pow2(), simd32_type
#include "../include/pirate/constants.hpp"

#include <sstream>
#include <gputils/cuda_utils.hpp>    // CUDA_CALL()

using namespace std;
using namespace gputils;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// FIXME use "wide" loads/stores (including when loading/restoring state?)

// LaggedDownsampler
//
// -------
//  Input
// -------
//
// Input array can be viewed either as:
//   - a 4-d array with shape (B, A, 2^r, ntime)     [ more convenient in gpu kernel ]
//   - a 3-d array with shape (B, A 2^r, ntime)      [ reflects high-level launch() interface ]
//
// The index 0 <= b < B is intended to represent a beam, and the index 0 <= a < A represents
// the index of a rank-r tree in a higher-rank ambient tree.
//
// We assume that all input array strides are contiguous, except the beam index, whose stride
// 'bstride_in' is a kernel argument. (Could generalize this assumption by introducing more strides.)
//
// We'll map this onto a 3-d grid of threadblocks:
//   gridDim.z = B    (number of beams)
//   gridDim.y = A_B  (see below)
//   gridDim.x = M_B  (see below)
//
// Each threadblock is a 3-d array of threads:
//    blockDim.z = A_W  (see below)
//    blockDim.y = M_W  (see below)
//    blockDim.x = 32
//
// Each warp processes four rows of the input tree, and writes two rows of the output tree.
// We define M = 2^(r-2). Thus, each "small" tree corresponds to M warps, and each "large"
// (ambient) tree corresponds to A*M warps. This factor of A*M can be factored arbitrarily
// between warps in a threadblock, and threadblocks in a kernel:
//
//   A = A_B * A_W     M = M_B * M_W
//
// In particular, the per-warp indices 0 <= i < M and 0 <= a < A are computed as follows:
//
//   int i = threadIdx.y + blockIdx.x * blockDim.y;   // 0 <= i < M
//   int a = threadIdx.z + blockIdx.y * blockDim.z;   // 0 <= a < A
//
// Input rows on one warp: (2i), (2i+1), (2^r-2i-2), (2^r-2i-1)   [ + a*(4*M) ]
// Output rows on one warp: i, (2^(r-1)-i-1)                      [ + a*(2*M) ]
//
// Note: we do not pass the rank r, or values of A,B as kernel arguments, since the kernel
// can infer these from grid/block dims.
//
// --------
//  Output
// --------
//
// For each beam 0 <= b < B, the output consists of D 3-d arrays, labelled by d=1,...,D:
//
//   d=1   output shape (A, 2^(r-1), ntime/2)
//   d=2   output shape (A, 2^(r-1), ntime/4)
//     ...
//   d=D   output shape (A, 2^(r-1), ntime/2^D)
//
// We assume that for each beam 0 <= b < B, the above arrays are contiguous, and the
// arrays are 'adjacent', in the sense that the base index of the d-th output array is:
//
//   sum_{1<=e<d} A 2^(r-1) ntime/2^e
//        = A 2^(r-1) ntime (1 - 1/2^(d-1))
//
// However, the beam index 0 <= b < B has stride 'bstride_out' (a kernel argument).
// Thus, the output array can be viewed as a 2-d array:
//
//    out[B][L]   where L = A 2^(r-1) ntime (1 - 1/2^d), with stride (bstride_out, 1)
//
// although this obscures the structure of the per-beam data (a length-D list of 3-d
// arrays of different shapes).
//
// ------------------------------------
//  Shared memory and persistent state
// ------------------------------------
//
// If T=float32, then unpadded ring buffers have length (i) and (2^(r-1)-i-1).
// Total length S = 2^(r-1) - 1.
//
// If T=__half2, then ring buffers have length div2(i) and div2(2^(r-1)-i-1), where div2(n) = (n >> 1).
// Total length S = 2^(r-2) - 1.
//
// It will be convenient to define
//   S = 2^(r-1) - 1    if T = float32
//     = 2^(r-2) - 1    if T = __half2
//
// Then shared memory ring buffer is an array of shape (D,W,S), where W is warps/threadblock:
//
//   W = blockDim.y * blockDim.z
//
// For convenience when saving/restoring state, we append in shared memory the 'rstate' (ring buffer
// register state), with shape (W,2*D). Thus, the total shared memory footprint is:
//
//    shared memory nbytes SB = align_up(W * D * (S+2) * 4, 128)
//
// This persistent state gets saved/restored in global memory:
//
//    global memory nbytes GB = B * A * W2 * SB
//
// ------------------
//  Kernel interface
// ------------------
//
// The __global__ kernel has the following syntax (where T=float or T=__half2):
//
//   template<typename T, int D>
//   lagged_downsample(const T *in, T *out, int ntime, long ntime_cumulative,
//                     long bstride_in, long bstride_out, T *persistent_state);
//
// The 'ntime_cumulative' argument is the total number of time samples processed
// by previous calls. (This is convenient when saving/restoring shmem state.)


// -------------------------------------------------------------------------------------------------
//
// FIXME should have an .hpp file with float16 cuda helpers.


// FIXME add "wide" float16 loads/stores (not obvious how to do this!)


// Given __half2 variables a = [a0,a1] and b = [b0,b1]:
// f16_align() returns [a1,b0]

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


// -------------------------------------------------------------------------------------------------


// dtype_shift<T>(): returns (0 if float32, or 1 if __half2).
// Useful in a few places below.
template<typename T> __device__ int dtype_shift();
template<> __device__ int dtype_shift<float>() { return 0; }
template<> __device__ int dtype_shift<__half2>() { return 1; }

extern __shared__ float shmem_f[];
extern __shared__ __half2 shmem_h2[];

// shmem_base<T>(): returns a (T*) pointer to shared memory.
template<typename T> __device__ T *shmem_base();
template<> __device__ float *shmem_base<float> () { return shmem_f; }
template<> __device__ __half2 *shmem_base<__half2> () { return shmem_h2; }

// zero<T>(): returns zero
template<typename T> __device__ T zero();
template<> __device__ float zero<float>() { return 0.0f; }
template<> __device__ __half2 zero<__half2>() { return __half2half2(0.0f); }
    

// -------------------------------------------------------------------------------------------------


// 'rb' is the base address of the shared memory ring buffer (same on all threads).
// The 'rpos' argument keeps track of the ring buffer location (only used if lag > 32).

template<typename T>
__device__ T shmem_rb_cycle(T x, T *rb, int lag, int &rpos)
{
    static_assert(sizeof(T) == 4);
    
    if (lag > 32) {
	
	// Case 1: lag > 32.
	// In this case, rpos = (nreg_cumulative + laneId) % lag.
	// Here, nreg_cumulative is the number of registers processed so far:
	//   nreg_cumulative = ntime_cumulative      for float32
	//   nreg_cumulative = ntime_cumulative/2    for float16
	
	T y = rb[rpos];
	rb[rpos] = x;
	rpos += 32;
	rpos = (rpos < lag) ? rpos : (rpos - lag);
	return y;
    }

    // Case 2: 0 <= lag < 32.
    // In this case, 'rpos' is not used.

    int laneId = threadIdx.x;
    T y = __shfl_sync(0xffffffff, x, laneId + 32 - lag);

    // Warp divergence!
    if (laneId < lag) {
	T z = y;
	y = rb[laneId];
	rb[laneId] = z;
    }

    __syncwarp();
    return y;
}


template<typename T>
__device__ int init_rpos(int lag, long ntime_cumulative)
{
    // init_rpos<T>: Used to initialize 'rpos' argument to shmem_rb_cycle().
    // As explained above, we return rpos = (nreg_cumulative + laneId) % lag.
    // Here, nreg_cumulative is the number of registers processed so far:
    //   nreg_cumulative = ntime_cumulative      for T=float.
    //   nreg_cumulative = ntime_cumulative/2    for T=__half2.

    int laneId = threadIdx.x;
    long nreg_cumulative = ntime_cumulative >> dtype_shift<T>();
    return (nreg_cumulative + laneId) % lag;
}


// -------------------------------------------------------------------------------------------------


// wparams: per-warp parameters which apply to all downsampling levels 1 <= d <= D.

template<typename T>
struct wparams
{
    // "Extended" row indices (iext, jext) and number of extended rows (nrext) in output array.
    //  We use the term "extended" to refer to a row index 0 <= i < (2*A*M).
    //  A "non-extended" index is a row index 0 <= i < (2*M).
    //
    //  Reminder: each output tree can be viewed as a 3-d array with shape (A, 2*M, ntime/2^d),
    //  or a 2-d array with shape (2*A*M, ntime/2^d).
    
    int iext;
    int jext;
    int nrext;

    // Ring buffer lags (in 32-bit registers)
    int lag0;
    int lag1;
    
    __device__ wparams(int D)
    {
	const int A = blockDim.z * gridDim.y;
	const int M = blockDim.y * gridDim.x;  // 2^(r-2)
	const int nr = 2*M;                    // 2^(r-1)

	// (i,j) are non-extended row indices.
	const int a = threadIdx.z + blockIdx.y * blockDim.z;   // 0 <= a < A
	const int i = threadIdx.y + blockIdx.x * blockDim.y;   // 0 <= i < M
	const int j = nr - i - 1;                              // M <= j < 2*M

	iext = a*nr + i;
	jext = a*nr + j;
	nrext = A*nr;
	
	lag0 = (j >> dtype_shift<T>());  // note j here
	lag1 = (i >> dtype_shift<T>());  // note i here
    }
};


// -------------------------------------------------------------------------------------------------


__device__ float ld_downsample(float x, float y)
{
    return (x + y);
}

// Input: x=[z0 z1] and y=[z2 z3].
// Output: [ (z0+z1) (z2+z3) ]
__device__ half2 ld_downsample(__half2 x, __half2 y)
{
    __half2 t = __lows2half2(x, y);  // [ z0 z2 ]
    __half2 u = __highs2half2(x, y); // [ z1 z3 ]
    return (t + u);        // [ (z0+z1) (z2+z3) ]
}


// Helper for ld_lag_pair(), ld_lag2().
// Cycles 'rs' by one lane.
template<typename T>
__device__ T lag_register(T x, T &rs)
{
    const int laneId = threadIdx.x;
    const bool tail_flag = (laneId == 31);

    T t = tail_flag ? rs : x;
    t = __shfl_sync(0xffffffff, t, laneId + 31);
    rs = tail_flag ? x : rs;
    rs = __shfl_sync(0xffffffff, rs, laneId + 31);
    return t;
}


// Lags both 'x' and 'y' by one element (float32).
// Cycles 'rs' by two lanes.
__device__ void ld_lag_pair(float &x, float &y, float &rs)
{
    x = lag_register(x, rs);
    y = lag_register(y, rs);
}


// Lags both 'x' and 'y' by one element (float16, i.e. half of a __half2).
// Cycles 'rs' by one lane.
__device__ void ld_lag_pair(__half2 &x, __half2 &y, __half2 &rs)
{
    __half2 t = __highs2half2(x, y);  // (x[1], y[1])
    t = lag_register(t, rs);          // (x[-1], y[-1])

    x = __lows2half2(t, x);   // (x[-1], x[0])
    y = f16_align(t, y);     // (y[-1], y[0])
}


// This oddball function lags either 'x' or 'y' by one element (float16, i.e. half
// of a __half2), depending on whether yflag is false or true. Cycles 'rs' by one lane.
__device__ void ld_lag_switch(__half2 &x, __half2 &y, __half2 &rs, bool yflag)
{
    __half2 t = yflag ? y : x;
    __half2 u = lag_register(t, rs);  // t[-2] t[-1]
    u = f16_align(u, t);              // t[-1] t[0]
    x = yflag ? x : u;
    y = yflag ? u : y;
}

// In the float32 case, ld_lag_switch() is still defined, but is a no-op.
__device__ void ld_lag_switch(float &x, float &y, float &rs, bool yflag)
{
    return;
}


// ld_transpose()
//
// Input: a 64-element vector, with register assignment
//   x = [ v0 v1 ... v31 ]    (list notation is indexed by laneId)
//   y = [ v32 v33 ... v63 ]
//
// Output: regsister assignment
//   x = [ v0 v2 v4 ... v62 ]
//   y = [ v1 v3 v5 ... v63 ]

template<typename T>
__device__ void ld_transpose(T &x, T &y)
{
    int laneId = threadIdx.x;
    
    // r <-> i5
    // t4 t3 t2 t1 t0 <-> i4 i3 i2 i1 i0

    // Swap registers on threads where i0=1.
    bool i0 = laneId & 0x1;
    T t = (i0 ? y : x);
    T u = (i0 ? x : y);
    
    // r <-> (i5 + i0)
    // t4 t3 t2 t1 t0 <-> i4 i3 i2 i1 i0

    // lane = [ 0 2 4 ... 30 1 3 ... 31 ]
    int lane = (laneId << 1) | (laneId >> 4);
    t = __shfl_sync(0xffffffff, t, lane);
    u = __shfl_sync(0xffffffff, u, lane ^ 0x1);

    // r <-> (i5 + i0)
    // t4 t3 t2 t1 t0 <-> i5 i4 i3 i2 i1

    // Swap registers on threads where i5=1.
    bool i5 = laneId & 0x10;
    x = i5 ? u : t;
    y = i5 ? t : u;

    // r <-> i0
    // t4 t3 t2 t1 t0 <-> i5 i4 i3 i2 i1
}


// -------------------------------------------------------------------------------------------------


template<typename T, int D>
struct ld_half_kernel;


// ld_kernel<T,D>
//   T = either float or __half2
//   D = number of downsampling levels.
//
// The main function here is process(), which processes a shape (2,2,2) logical
// array x_{ijk} on each thread:
//
//   i = output row
//   j = input row within output row
//   k = time
//
// and writes two elements (per thread) to the output tree with d=1.
// The process() function recursively calls lower-D kernels, which writes at
// slower cadence.
//
// If RestoreRs==true, then 'rs' is fully cycled (by 32 registers).
// If RestoreRs==false, then 'rs' is cycled by (2*D) registers.

template<typename T, int D, bool RestoreRs>
struct ld_kernel
{
    static_assert(D >= 1);
    static_assert(D <= 16);  // required by register-cycling logic
    static_assert(sizeof(T) == 4);

    // Recursive kernels which run at slower cadence (more downsampling)
    ld_half_kernel<T,D-1> next_kernel;
    
    // Ring buffer positions
    int rpos0;
    int rpos1;


    __device__ ld_kernel(const wparams<T> &wp, long ntime_cumulative)
	: next_kernel(wp, ntime_cumulative >> 1)   // Note right-shift by 1 here!
    {
	// FIXME uses more %-operators than necessary.
	// Seems very unlikely to be a bottleneck!
	this->rpos0 = init_rpos<T> (wp.lag0, ntime_cumulative);
	this->rpos1 = init_rpos<T> (wp.lag1, ntime_cumulative);
    }
    
	
    // x_{ijk} arguments are a shape (2,2,2) logical array
    //  i = output row
    //  j = input row within output row
    //  k = time
    //
    // The 'counter' argument starts at zero, and is incremented every time
    // process() is called.
    //
    // If RestoreRs==true, then 'rs' is fully cycled (by 32 registers).
    // If RestoreRs==false, then 'rs' is cycled by (2*D) registers.

    __device__ void process(const wparams<T> &wp, T *out, long ntime_out, int counter, T &rs,
			    T x000, T x001, T x010, T x011, T x100, T x101, T x110, T x111)
    {
	const int laneId = threadIdx.x;
	const int W = blockDim.y * blockDim.z;
	const int w = threadIdx.y + threadIdx.z * blockDim.y;
	
	// Downsample in time.
	// Get shape (2,2) register array
	//   i = output row
	//   j = input row within output row
	
	T x00 = ld_downsample(x000, x001);
	T x01 = ld_downsample(x010, x011);
	T x10 = ld_downsample(x100, x101);
	T x11 = ld_downsample(x110, x111);

	// Apply one-sample lag to x00 and x01.
	// Cycles 'rs' by either 1 or 2 lanes, depending on whether float16 or float32.
	
	ld_lag_pair(x00, x10, rs);

	// Downsample in frequency.
	// Get 1-d register array, indexed by output frequency.
	
	T x0 = x00 + x01;
	T x1 = x10 + x11;

	// If float16, need to apply one-sample lag to either x0 or x1
	// (depending on whether j or i is odd).
	//
	// Cycles 'rs' by either 1 or 0 lanes, depending on whether float16 or float32.
	// Note that 'rs' have been cycled by 2 lanes so far, in both float16/float32 cases.
	
	bool flag = (wp.iext & 1);  // if iext is odd, then lag x1, else lag x0
	ld_lag_switch(x0, x1, rs, flag);

	// Go through shared memory ring buffer.
	// Reminder: shared memory ring buffer is an array of shape (D,W,S).

	int S = wp.lag0 + wp.lag1;
	T *rb0 = shmem_base<T>() + ((D-1)*W + w) * S;  // length wp.lag0
	T *rb1 = rb0 + wp.lag0;                        // length wp.lag1
	
	x0 = shmem_rb_cycle(x0, rb0, wp.lag0, rpos0);
	x1 = shmem_rb_cycle(x1, rb1, wp.lag1, rpos1);
	
	// Write output.
	// Output array has shape (wp.nrext, ntime_out)

	out[wp.iext * ntime_out + 32*counter + laneId] = x0;
	out[wp.jext * ntime_out + 32*counter + laneId] = x1;

	int ntime2 = ntime_out >> 1;
	T *out2 = out + (wp.nrext * ntime_out);
	next_kernel.process(wp, out2, ntime2, counter, rs, x00, x01, x10, x11);

	if constexpr (RestoreRs)
	    rs = __shfl_sync(0xffffffff, rs, laneId + 2*D);
    }
};


// ld_half_kernel<T,D>: "half-speed" version of ld_kernel<T,D>.
//
// The process() function receives a shape (2,2) logical array x_{ij} on each thread:
//   i = output row
//   j = input row within output row

template<typename T, int D>
struct ld_half_kernel
{
    static_assert(D >= 1);

    T y00, y01, y10, y11;
    ld_kernel<T,D,false> base_kernel;  // RestoreRs=false

    __device__ ld_half_kernel(const wparams<T> &wp, long ntime_cumulative)
	: base_kernel(wp, ntime_cumulative)
    { }

    // x_{ij} arguments are a shape (2,2) logical array
    //   i = output row
    //   j = input row within output row
    //
    // The 'counter' argument starts at zero, and is incremented every time
    // process() is called.
    //
    // Cycles 'rs' by (2*D) registers.

    __device__ void process(const wparams<T> &wp, T *out, long ntime_out, int counter, T &rs, T x00, T x01, T x10, T x11)
    {
	if (counter & 1) {
	    ld_transpose(y00, x00);
	    ld_transpose(y01, x01);
	    ld_transpose(y10, x10);
	    ld_transpose(y11, x11);

	    int counter2 = counter >> 1;
	    base_kernel.process(wp, out, ntime_out, counter2, rs, y00, x00, y01, x01, y10, x10, y11, x11);
	}
	else {
	    y00 = x00;
	    y01 = x01;
	    y10 = x10;
	    y11 = y11;
	    rs = __shfl_sync(0xffffffff, rs, threadIdx.x + 32 - 2*D);
	}
    }
};


template<typename T>
struct ld_half_kernel<T,0>
{
    __device__ ld_half_kernel(const wparams<T> &wp, long ntime_cumulative) { }
    __device__ void process(const wparams<T> &wp, T *out, long ntime_out, int counter, T &rs, T x00, T x01, T x10, T x11) { }
};


// -------------------------------------------------------------------------------------------------


// Helper for restore_state() and save_state().
template<typename T, int D>
struct state_params
{
    int threadId;
    int nthreads;
    int shmem_nelts;  // total shared memory size (in 32-bit registers)
    int rs_idx;       // shared memory index of 'rs' on this thread
    bool rs_flag;     // is 'rs' valid on this thread?

    __device__ state_params(const wparams<T> &wp)
    {
	const int W = blockDim.y * blockDim.z;
	const int w = threadIdx.y + threadIdx.z * blockDim.y;
	const int S = wp.lag0 + wp.lag1;
	const int ri = (int)threadIdx.x + (2*D-32);
	
	threadId = threadIdx.x + 32 * (threadIdx.y + threadIdx.z * blockDim.y);
	nthreads = blockDim.x * blockDim.y * blockDim.z;
	rs_idx = (W*D*S) + (2*D*w) + ri;
	rs_flag = (ri >= 0);
	
	shmem_nelts = D * W * (S+2);
	shmem_nelts = (shmem_nelts + 31) & ~0x1f;
    }
};


template<typename T, int D>
__device__ T restore_state(const wparams<T> &wp, const T *persistent_state)
{
    T *shmem = shmem_base<T> ();
    state_params<T,D> sp(wp);

    for (int s = sp.threadId; s < sp.shmem_nelts; s += sp.nthreads)
	shmem[s] = persistent_state[s];

    __syncthreads();

    // Warp divergence
    T rs = sp.rs_flag ? shmem[sp.rs_idx] : zero<T>();
    __syncwarp();
    
    return rs;
}


template<typename T, int D>
__device__ void save_state(const wparams<T> &wp, T *persistent_state, T rs)
{
    T *shmem = shmem_base<T> ();
    state_params<T,D> sp(wp);

    // Warp divergence
    if (sp.rs_flag)
	shmem[sp.rs_idx] = rs;

    __syncthreads();
    
    for (int s = sp.threadId; s < sp.shmem_nelts; s += sp.nthreads)
	persistent_state[s] = shmem[s];
}


template<typename T, int D>
__global__ void __launch_bounds__(512, 2)  // FIXME rethink launch_bounds
lagged_downsample(const T *in, T *out, int ntime, long ntime_cumulative, long bstride_in, long bstride_out, T *persistent_state)
{
    static_assert(D >= 1);
    static_assert(sizeof(T) == 4);
    
    const wparams<T> wp(D);
    T rs = restore_state<T,D> (wp, persistent_state);

    ld_kernel<T,D,true> kernel(wp, ntime_cumulative);  // last template argument is RestoreRs=true

    in += (blockIdx.z * bstride_in) + threadIdx.x;  // laneId included in 'in'
    out += (blockIdx.z * bstride_out);              // laneId not included in 'out'

    const int i_off = (2*wp.iext) * ntime;
    const int j_off = (2*wp.jext) * ntime;
    const int ntime_out = ntime >> 1;
    int counter = 0;

    for (int it = 0; it < ntime_out; it += 32) {
	// FIXME use wide loads/stores here
	T x000 = in[i_off];                // (2*i, 0)
	T x001 = in[i_off + 32];           // (2*i, 32)
	T x010 = in[i_off + ntime];        // (2*i+1, 0)
	T x011 = in[i_off + ntime + 32];   // (2*i+1, 32)
	T x100 = in[j_off];                // (2*j, 0)
	T x101 = in[j_off + 32];           // (2*j, 32)
	T x110 = in[j_off + ntime];        // (2*j+1, 0)
	T x111 = in[j_off + ntime + 32];   // (2*j+1, 32)

	ld_transpose(x000, x001);
	ld_transpose(x010, x011);
	ld_transpose(x100, x101);
	ld_transpose(x110, x111);

	kernel.process(wp, out, ntime_out, counter, rs,
		       x000, x001, x010, x011, x100, x101, x110, x111);
	
	in += 64;
	counter++;
    }

    save_state<T,D> (wp, persistent_state, rs);
}


// -------------------------------------------------------------------------------------------------


template<typename T>
using kernel_t = typename GpuLaggedDownsamplingKernel<T>::kernel_t;


template<typename T, int Dmax>
static vector<kernel_t<T>> _make_kernels()
{
    using T32 = typename simd32_type<T>::type;
    
    if constexpr (Dmax > 0) {
	kernel_t<T> k = lagged_downsample<T32,Dmax>;
	vector<kernel_t<T>> ret = _make_kernels<T,Dmax-1> ();
	ret.push_back(k);
	return ret;
    }
    else
	return vector<kernel_t<T>> (1, nullptr);
}


template<typename T>
static vector<kernel_t<T>> make_kernels()
{
    constexpr int Dmax = constants::max_downsampling_level;
    vector<kernel_t<T>> ret = _make_kernels<T,Dmax> ();
    
    for (int d = 1; d <= Dmax; d++) {
	CUDA_CALL(cudaFuncSetAttribute(
	    ret[d],
	    cudaFuncAttributeMaxDynamicSharedMemorySize,
	    99 * 1024
	));
    }

    return ret;
}	


template<typename T>
static int get_shmem_nbytes(int r, int D, int W, bool align=true)
{
    int S = (pow2(r-2) * (sizeof(T)/2)) - 1;
    int nbytes = W * D * (S+2) * 4;
    return align ? align_up(nbytes,128) : nbytes;
}


template<typename T>
static int target_warps_per_threadblock(int r, int D)
{
    // Shared memory bytes per warp, without 128-byte alignment
    int nb1 = get_shmem_nbytes<T> (r, D, 1, false);

    // Aim for 8 warps per threadblock (semi-arbitrary), but use fewer warps
    // if we get into trouble with shared memory.
    for (int W = 1; W <= 4; W++)
	if (W*nb1 > 48*1024)
	    return W;

    return 8;
}


// Static member function
template<typename T>
shared_ptr<GpuLaggedDownsamplingKernel<T>>
GpuLaggedDownsamplingKernel<T>::make(int small_input_rank, int large_input_rank, int num_downsampling_levels)
{
    // FIXME do something thread-safe here
    static vector<kernel_t> kernels;

    if (kernels.size() == 0)
	kernels = make_kernels<T> ();
	
    assert(small_input_rank >= 2);
    assert(small_input_rank <= 8);
    assert(large_input_rank >= small_input_rank);
    assert(large_input_rank <= constants::max_tree_rank);
    assert(num_downsampling_levels >= 1);
    assert(num_downsampling_levels <= constants::max_downsampling_level);

    Params params;
    params.small_input_rank = small_input_rank;
    params.large_input_rank = large_input_rank;
    params.num_downsampling_levels = num_downsampling_levels;
    
    int M = pow2(small_input_rank - 2);
    int A = pow2(large_input_rank - small_input_rank);
    int W_target = target_warps_per_threadblock<T> (small_input_rank, num_downsampling_levels);

    // Each "large" tree corresponds to A*M warps. This factor of A*M can be factored
    // arbitrarily between warps in a threadblock, and threadblocks in a kernel:
    //
    //   A = A_B * A_W     M = M_B * M_W

    params.M_W = min(M, W_target);
    params.A_W = min(A, (int)xdiv(W_target, params.M_W));
    params.M_B = xdiv(M, params.M_W);
    params.A_B = xdiv(A, params.A_W);

    int nb = get_shmem_nbytes<T> (small_input_rank, num_downsampling_levels, params.warps_per_threadblock(), true);  // align=true
    params.ntime_divisibility_requirement = (128 / sizeof(T)) * pow2(num_downsampling_levels);
    params.state_nelts_per_beam = params.threadblocks_per_beam() * (nb / sizeof(T));
    params.shmem_nbytes = nb;

    kernel_t k = kernels.at(num_downsampling_levels);
    GpuLaggedDownsamplingKernel<T> *kernel = new GpuLaggedDownsamplingKernel (params, k);
    return shared_ptr<GpuLaggedDownsamplingKernel<T>> (kernel);
}


template<typename T>
GpuLaggedDownsamplingKernel<T>::GpuLaggedDownsamplingKernel(const Params &params_, kernel_t kernel_)
    : params(params_), kernel(kernel_)
{
    // Assume "real" argument-checking has been done in GpuLaggedDownsamplingKernel::make().
    // Here, just some asserts to make sure I didn't forget to initialize something!

    assert(params.small_input_rank >= 2);   // currently required by GPU kernel
    assert(params.large_input_rank >= params.small_input_rank);
    assert(params.num_downsampling_levels > 0);
    assert(params.ntime_divisibility_requirement > 0);
    assert(params.state_nelts_per_beam > 0);
    assert(params.shmem_nbytes > 0);
    assert(params.M_W > 0);
    assert(params.M_B > 0);
    assert(params.A_W > 0);
    assert(params.A_B > 0);
    assert(kernel != nullptr);

    assert(params.M_W * params.M_B == pow2(params.small_input_rank - 2));
    assert(params.A_W * params.A_B == pow2(params.large_input_rank - params.small_input_rank));
    assert(params.warps_per_threadblock() <= 32);
}


template<typename T>
void GpuLaggedDownsamplingKernel<T>::launch(
    const Array<T> &in,
    vector<Array<T>> &out,
    Array<T> &persistent_state,
    long ntime_cumulative,
    cudaStream_t stream) const
{
    int r = params.large_input_rank;
    
    assert(in.ndim == 3);
    assert(in.shape[1] == pow2(params.large_input_rank));
    assert(in.get_ncontig() >= 2);

    long nbeams = in.shape[0];
    long ntime_in = in.shape[2];

    assert(nbeams > 0);
    assert(ntime_in > 0);
    assert(ntime_cumulative >= 0);
    assert(ntime_in < 2L * 1024L * 1024L * 1024L);
    assert((ntime_in % params.ntime_divisibility_requirement) == 0);
    assert((ntime_cumulative % params.ntime_divisibility_requirement) == 0);
    
    assert(out.size() == params.num_downsampling_levels);
    long nout = 0;

    for (int i = 0; i < (int)out.size(); i++) {
	int ntime_out = ntime_in >> (i+1);
	int ntree_out = pow2(params.large_input_rank - 1);

	if (!out[i].shape_equals({ nbeams, ntree_out, ntime_out })) {
	    stringstream ss;
	    ss << "GpuLaggedDownsamplingKernel::launch(): out[" << i
	       << "]: expected shape (" << nbeams << "," << ntree_out << "," << ntime_out << ")"
	       << ", got shape "
	       << out[i].shape_str();
	    throw runtime_error(ss.str());
	}

	if (out[i].get_ncontig() < 2)
	    throw runtime_error("GpuLaggedDownsamplingKernel::launch(): output arrays must have contiguous freq/time axes");
	
	if (out[i].data != (out[0].data + nout)) {
	    throw runtime_error("GpuLaggedDownsamplingKernel::launch(): output arrays must be 'adjacent' "
				"(see comment at beginning of pirate/src_lib/GpuLaggedDownsampler.cu)");
	}

	if (out[i].strides[0] != out[0].strides[0])
	    throw runtime_error("GpuLaggedDownsamplingKernel::launch(): all output arrays must have same beam_stride");

	nout += nbeams * ntree_out * ntime_out;
    }
	
    assert(persistent_state.shape_equals({ nbeams, params.state_nelts_per_beam }));
    assert(persistent_state.is_fully_contiguous());

    // Required by CUDA (max alllowed value of gridDims.z)
    assert(nbeams < 65536);

    dim3 block_dims;
    block_dims.z = params.A_W;
    block_dims.y = params.M_W;
    block_dims.x = 32;

    dim3 grid_dims;
    grid_dims.z = nbeams;
    grid_dims.y = params.A_B;
    grid_dims.x = params.M_B;
    
    this->kernel
	<<< grid_dims, block_dims, params.shmem_nbytes, stream >>>
	(reinterpret_cast<const T32 *> (in.data),
	 reinterpret_cast<T32 *> (out[0].data),
	 ntime_in,
	 ntime_cumulative,
	 in.strides[0],      // bstride_in,
	 out[0].strides[0],  // bstride_out,
	 reinterpret_cast<T32 *> (persistent_state.data));

    CUDA_PEEK("lagged downsampling kernel");
}


template<typename T>
void GpuLaggedDownsamplingKernel<T>::print(ostream &os, int indent) const
{
    // Usage reminder: print_kv(key, val, os, indent, units=nullptr)
    print_kv("small_input_rank", params.small_input_rank, os, indent);
    print_kv("large_input_rank", params.large_input_rank, os, indent);
    print_kv("num_downsampling_levels", params.num_downsampling_levels, os, indent);
    print_kv("ntime_divisibility_requirement", params.ntime_divisibility_requirement, os, indent);
    print_kv("state_nelts_per_beam", params.state_nelts_per_beam, os, indent);
    print_kv("shmem_nbytes", params.shmem_nbytes, os, indent);

    stringstream sw;
    sw << params.warps_per_threadblock() << " (M_W=" << params.M_W << ", A_W=" << params.A_W << ")";
    print_kv("warps_per_threadblock", sw.str(), os, indent);
    
    stringstream sb;
    sb << params.threadblocks_per_beam() << " (M_B=" << params.M_B << ", A_B=" << params.A_B << ")";
    print_kv("threadblocks_per_beam", sb.str(), os, indent);
}


#define INSTANTIATE(T) \
    template GpuLaggedDownsamplingKernel<T>::GpuLaggedDownsamplingKernel(const Params &, kernel_t); \
    template shared_ptr<GpuLaggedDownsamplingKernel<T>> GpuLaggedDownsamplingKernel<T>::make(int,int,int); \
    template void GpuLaggedDownsamplingKernel<T>::launch(const Array<T>&, vector<Array<T>>&, Array<T>&, long, cudaStream_t) const; \
    template void GpuLaggedDownsamplingKernel<T>::print(ostream &, int) const

INSTANTIATE(float);
INSTANTIATE(__half);


}  // namespace pirate
