#include "../include/pirate/casm.hpp"

#include <cassert>
#include <ksgpu/Array.hpp>
#include <ksgpu/cuda_utils.hpp>
#include <ksgpu/rand_utils.hpp>

using namespace std;
using namespace ksgpu;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// Complex out += x*y
__device__ void zma(float &out_re, float &out_im, float xre, float xim, float yre, float yim)
{
    out_re += (xre*yre - xim*yim);
    out_im += (xre*yim + xim*yre);
}

// Complex out += e^(i*theta)*y
__device__ void zma_expi(float &out_re, float &out_im, float theta, float yre, float yim)
{
    float xre, xim;
    sincosf(theta, &xim, &xre);   // note ordering (im,re)
    zma(out_re, out_im, xre, xim, yre, yim);
}

// FIXME by calling warp_shuffle() in pairs, am I being suboptimal?
__device__ void warp_shuffle(float &x, float &y, uint bit)
{
    bool flag = threadIdx.x & bit;
    float z = __shfl_sync(0xffffffff, (flag ? x : y), threadIdx.x ^ bit);
    x = flag ? z : x;  // compiles to conditional (predicated) move, not branch
    y = flag ? y : z;  // compiles to conditional (predicated) move, not branch
}


// -------------------------------------------------------------------------------------------------
//
// fft_c2c (helper for "FFT1" kernel)


__device__ void fft0(float &xre, float &xim)
{
    float t = xre - xim;
    xre += xim;
    xim = t;
}


template<int R>
struct fft_c2c_state
{
    // Implements a c2c FFT with 2^R elements.
    //
    // Input register assignment:
    //   r1 r0 <-> x_{r-1} ReIm
    //   t4 t3 t2 t1 t0 <-> s_{5-r} ... s0 x_{r-2} ... x_0
    //
    // Output register assignment:
    //   r1 r0 <-> y_{r-1} ReIm
    //   t4 t3 t2 t1 t0 <-> s_{5-r} ... s0 y_0 ... y_{r-2}

    fft_c2c_state<R-1> next_fft;
    float cre, cim;
    
    __device__ fft_c2c_state()
    {
	constexpr float a = 6.283185307f / (1<<R);   // 2*pi/2^r
	uint t = threadIdx.x & ((1 << (R-1)) - 1);   // 0 <= t < 2^{r-1}
	sincosf(a*t, &cim, &cre);                    // phase is exp(2*pi*t / 2^r)
    }

    __device__ void apply(float  &x0_re, float &x0_im, float &x1_re, float &x1_im)
    {
	// (x0,x1) = (x0+x1,x0-x1)
	fft0(x0_re, x1_re);
	fft0(x0_im, x1_im);

	// x1 *= phase
	float yre = cre * x1_re - cim * x1_im;
	float yim = cim * x1_re + cre * x1_im;
	x1_re = yre;
	x1_im = yim;

	// swap "01" register bit with thread bit (R-2)
	warp_shuffle(x0_re, x1_re, (1 << (R-2)));
	warp_shuffle(x0_im, x1_im, (1 << (R-2)));
	
	next_fft.apply(x0_re, x0_im, x1_re, x1_im);
    }
};


template<>
struct fft_c2c_state<1>
{
    __device__ void apply(float  &x0_re, float &x0_im, float &x1_re, float &x1_im)
    {
	fft0(x0_re, x1_re);
	fft0(x0_im, x1_im);
    }
};


// Call with {1,32} threads.
// Input and output arrays have shape (2^(6-R), 2^R, 2)
template<int R>
__global__ void fft_c2c_test_kernel(const float *in, float *out)
{
    // Input register assignment:
    //   r1 r0 <-> x_{r-1} ReIm
    //   t4 t3 t2 t1 t0 <-> s_{5-r} ... s0 x_{r-2} ... x_0
    
    int ss = threadIdx.x >> (R-1);            // spectator index
    int sx = threadIdx.x & ((1<<(R-1)) - 1);  // x-index
    int sin = (ss << (R+1)) | (sx << 1);
    
    float x0_re = in[sin];
    float x0_im = in[sin + 1];
    float x1_re = in[sin + (1<<R)];
    float x1_im = in[sin + (1<<R) + 1];

    fft_c2c_state<R> fft;
    fft.apply(x0_re, x0_im, x1_re, x1_im);

    // Output register assignment:
    //   r1 r0 <-> y_{r-1} ReIm
    //   t4 t3 t2 t1 t0 <-> s_{5-r} ... s0 y_0 ... y_{r-2}

    int sy = __brev(threadIdx.x << (33-R));
    int sout = (ss << (R+1)) | (sy << 1);

    out[sout] = x0_re;
    out[sout + 1] = x0_im;
    out[sout + (1<<R)] = x1_re;
    out[sout + (1<<R) + 1] = x1_im;
}


void test_casm_fft_c2c()
{
    constexpr int R = 6;
    constexpr int N = (1 << R);
    constexpr int S = (1 << (6-R));
    
    Array<float> in({S,N,2}, af_random | af_rhost);
    Array<float> out({S,N,2}, af_zero | af_rhost);
    Array<float> out_gpu({S,N,2}, af_random | af_gpu);
    
    for (int j = 0; j < N; j++) {
	for (int k = 0; k < N; k++) {
	    float theta = (2*M_PI/N) * ((j*k) % N);
	    float cth = cosf(theta);
	    float sth = sinf(theta);
	    
	    for (int s = 0; s < S; s++) {
		float xre = in.at({s,k,0});
		float xim = in.at({s,k,1});
		
		out.at({s,j,0}) += (cth*xre - sth*xim);
		out.at({s,j,1}) += (sth*xre + cth*xim);
	    }
	}
    }

    in = in.to_gpu();
    fft_c2c_test_kernel<R> <<<1,32>>> (in.data, out_gpu.data);
    CUDA_PEEK("fft_c2c_test_kernel");

    assert_arrays_equal(out, out_gpu, "cpu", "gpu", {"s","i","reim"}, 1.0e-3);
    cout << "test_casm_fft_c2c: pass" << endl;
}


// -------------------------------------------------------------------------------------------------
//
// FFT1


#if 0
template<bool Debug>
struct fft1_state
{
    // Implements a zero-padded c2c FFT with 64 inputs and 128 outputs
    //
    // Input register assignment:
    //   r1 r0 <-> x5 ReIm
    //   t4 t3 t2 t1 t0 <-> x4 x3 x2 x1 x0
    //
    // Outputs are written to shared memory:
    //   float G[W][2][128]   strides (257,128,1)   W=warps per threadblock
    //
    // Kernel assumes that threads are a {W,32} grid (not a {1,32*W} grid).

    fft_c2c_state<6> next_fft;
    float cre, cim;   // "twiddle" factor exp(2*pi*i t / 128)
    int sbase;

    fft1_state()
    {
	float x = 0.04908738521234052f * threadIdx.x;   // constant is (2pi)/128
	sincosf(x, &cim, &cre);                         // note ordering (im,re)
	
	sbase = (threadIdx.y) * 257;              // warp id
	sbase += (threadIdx.x & 1);               // t0 <-> y0
	sbase += __brev(threadIdx.x >> 1) >> 27;  // t4 t3 t2 t1 <-> y1 y2 y3 y4

	// Check bank conflict free
	xxx;
    }
    
    __device__ void fft1(float x0_re, float x0_im, float x1_re, float x1_im, float *sp)
    {
	// y0 = x0 * exp(2*pi*i t / 128)   where t = 0, ..., 31
	// y1 = x1 * exp(2*pi*i (t+32) / 128)
	
	float y0_re = cre*x0_re - cim*x0_im;
	float y0_im = cim*x0_re + cre*x0_im;

	float y0_re = -cre*x0_re + cre*x0_im;
	float y1_im = cre*x1_re - cim*x1_im;

	// xy r1 r0 <-> y0 x5 ReIm
	// t4 t3 t2 t1 t0 <-> x4 x3 x2 x1 x0

	next_fft.apply(x0_re, x0_im, x1_re, x1_im);
	next_fft.apply(y0_re, y0_im, y1_re, y1_im);
	
	// xy r1 r0 <-> y0 y6 ReIm
	// t4 t3 t2 t1 t0 <-> y1 y2 y3 y4 y5

	// Exchange "xy" and "thread 0" bits
	warp_shuffle(x0_re, y0_re, 1);
	warp_shuffle(x0_im, y0_im, 1);
	warp_shuffle(x1_re, y1_re, 1);
	warp_shuffle(x1_im, y1_im, 1);
	
	// xy r1 r0 <-> y5 y6 ReIm
	// t4 t3 t2 t1 t0 <-> y1 y2 y3 y4 y0

	// Strides: xy=32, 01=64, ReIm=128
	sp[sbase] = x0_re;
	sp[sbase+32] = y0_re;
	sp[sbase+64] = x1_re;
	sp[sbase+96] = y1_re;
	sp[sbase+128] = x0_im;
	sp[sbase+160] = y0_im;
	sp[sbase+192] = x1_im;	
	sp[sbase+224] = y1_im;	
    }
};  // fft1_state
#endif


// -------------------------------------------------------------------------------------------------
//
// FFT2
//
// There are 6 east-west feeds. It's convenient to parameterize them by
// 0 <= fouter < 2 and 0 <= finner < 3 as follows:
//
//   fouter=1  finner=(2,1,0)   negative feed locations
//   fouter=0  finner=(0,1,2)   positive feed locations
//
// Similarly, there are 24 east-west beams. It's convenient to parameterize by
// 0 <= bouter < 2 and 0 <= binner < 12 as follows:
//
//   bouter=1  binner=(11,...,2,1,0)   negative beam locations
//   bouter=0  binner=(0,1,2,...11)    positive beam locations
//
// Thus, if we flip 'bouter' at fixed 'binner', the beam location goes to its
// negative, and likewise for feeds.
//
// We sometimes write 'binner' as:
//
//   binner = 4*b2 + 2*b1 + b0    0 <= b2 < 3   0 <= b1 < 2   0 <= b0 < 2
//
// NS beam locations are represented by an index 0 <= ns < 128,
// which we sometimes decompose into its base-2 digits [ns6,...,ns0].
// These are spectator indices, as far as the FFT2 kernel is concerned.
//
// We store EW beamforming phases for fouter=bouter=0 only, since flipping
// 'bouter' or 'fouter' sends the phase to its complex conjugate. The phase
// can be written in the form (where alpha[3] is a kernel argument):
//
//   phase[binner,finner] = exp( i * (1+2*binner) * alpha[finner] )
//
// The I-array is distributed as follows:
//
//   r1 r0 <-> (bouter) (b0)
//   t4 t3 t2 t1 t0 <-> (b1) (ns5) (ns4) (ns3) (ns2)
//   24 warps <-> (b2) (ns6) (ns1) (ns0)
//
// The G-array shared memory layout is (note brev!)
//
//   float G[2][6][128]   strides (6*129,129,1)   bit-reversed ns
//
// We read from the G-array in register assignment:
//
//   t4 t3 t2 t1 t0 <-> (fouter) (ns5) (ns4) (ns3) (ns2)
//
// This is bank conflict free, since flipping 'fouter' always produces
// an odd change in f, the f-stride is odd (129), and the (ns5, ..., ns2)
// strides are (2, 4, 8, 16) after bit reversal.
//
// The I-array shared memory layout is:
//
//   float I[24][128]   strides (133,1)
//
// We write to the I-array in register assignment:
//
//   t4 t3 t2 t1 t0 <-> (b1) (bouter) (ns4) (ns3) (ns2)
//
// This is bank conflict free, since flipping 'bouter' always produces
// an odd change in b, the b-stride is odd (133), and the (ns4, ns3, ns2)
// strides are (16, 8, 4).


#if 0

template<bool Debug>
struct fft2_state
{
    // Note: we use 12 persistent registers/thread to store beamforming
    // phases, but the number of distinct phases is 24/warp or 72/block.
    // Maybe better to distribute registers as needed with __shfl_sync()?
     
    float I[2][2];      // beams are indexed by (bouter, binner)
    float pcos[2][3];   // beamforming phases are indexed by (b0, finner)
    float psin[2][3];   // beamforming phases are indexed by (b0, finner)
    
    int soff_g;    // base shared memory offset in G-array.
    int soff_i;    // base shared memory offset in I-array.
    
    __device__ fft2_state(float alpha[3])
    {
	if constexpr (Debug) {
	    assert(blockDim.x == 32);
	    assert(blockDim.y == 24);
	    assert(blockDim.z == 1);
	}
	
	I[0][0] = I[0][1] = I[1][0] = I[1][1] = 0.0f;

	// Each warp maps to an (b2, ns6, ns01) triple.
	uint b2 = (threadIdx.y >> 3);       // 0 <= b2 < 3
	uint ns6 = (threadIdx.y >> 2) & 1;  // 0 <= ns6 < 2
	uint ns01 = (threadIdx.y & 3);      // 0 <= ns01 < 4

	// Each thread maps to a (b1, ns25) pair.
	uint b1 = (threadIdx.x & 1);
	uint ns25 = (threadIdx.x >> 1);

	// Beamforming phases are indexed by (b0, finner)
	#pragma unroll
	for (uint b0 = 0; b0 < 2; b0++) {
	    // Beamforming phase is exp(i*t*alpha[finner]) where t = 1+2*binner
	    float t = (b2 << 3) | (b1 << 2) | (b0 << 1) | 1;
	    
	    #pragma unroll
	    for (uint finner = 0; finner < 3; finner++)
		sincosf(t * alpha[finner], psin[b0][finner], pcos[b0][finner]);
	}

	// Shared memory offset for reading G-array:
	//
	//   float G[2][6][128]   strides (6*129,129,1)   bit-reversed ns
	//
	// When we read from the G-array, we read it as:
	//
	//   t0 t1 t2 t3 t4 <-> (fouter) (ns5) (ns4) (ns3) (ns2)
	//
	// 'soff_g' uses per-warp values of (ns01,ns6), and (finner,ReIm)=(0,0).
	
	uint fouter = threadIdx.x & 1;
	uint ns = (ns6 << 6) | (ns25 << 2) | ns01;
	soff_g = (__brev(ns) >> 25) + (fouter ? (5*129) : 0);

	// Shared memory offset for writing I-array:
	//
	//   float I[24][128]   strides (133,1)
	//
	// When we write to the I-array, we write as
	//
	//   t0 t1 t2 t3 t4 <-> (b1) (bouter) (ns4) (ns3) (ns2)
	//
	// 'soff_i' uses        
    }

    // Accumulates one (time,pol) into I-registers.
    __device__ void step(const float *sp)
    {
	float Fre[2][2];   // (b1, bflip)
	float Fim[2][2];   // (b1, bflip)

	Fre[0][0] = Fre[0][1] = Fre[1][0] = Fre[1][1] = 0.0f;
	Fim[0][0] = Fim[0][1] = Fim[1][0] = Fim[1][1] = 0.0f;

	int ds = (threadIdx.x & 1) ? (-129) : 129;
	
        #pragma unroll
	for (int f1 = 0; f1 < 3; f1++) {
	    float Gre[2];  // fflip
	    float Gim[2];  // fflip

	    // FIXME loading logic could be improved.
	    int s = soff_g + f1*ds;
	    float tre = sp[s];
	    float tim = sp[s + 6*129];
	    Gre[0] = __shfl_sync(0xffffffff, tre, threadIdx.x & ~1);
	    Gre[1] = __shfl_sync(0xffffffff, tre, threadIdx.x | 1);
	    Gim[0] = __shfl_sync(0xffffffff, tim, threadIdx.x & ~1);
	    Gim[1] = __shfl_sync(0xffffffff, tim, threadIdx.x | 1);

	    #pragma unroll
	    for (int b1 = 0; b1 < 2; b1++) {
		// FIXME can be sped up with FFT-style tricks.

		// F[0] += (phase) (G[0])
		// F[0] += (phase^*) (G[1])
		// F[1] += (phase^*) (G[0])
		// F[1] += (phase) (G[1])

		// FIXME I don't think zma() will be called here, in the final kernel.
		zma(Fre[b1][0], Fim[b1][0], pcos[b1][f1],  psin[b1][f1], Gre[0], Gim[0]);
		zma(Fre[b1][0], Fim[b1][0], pcos[b1][f1], -psin[b1][f1], Gre[1], Gim[1]);
		zma(Fre[b1][1], Fim[b1][1], pcos[b1][f1], -psin[b1][f1], Gre[0], Gim[0]);
		zma(Fre[b1][1], Fim[b1][1], pcos[b1][f1],  psin[b1][f1], Gre[1], Gim[1]);
	    }
	}

	#pragma unroll
	for (int b1 = 0; b1 < 2; b1++) {
	    #pragma unroll
	    for (int bflip = 0; bflip < 2; bflip++)
		I[b1][bflip] += (Fre[b1][bflip] * Fre[b1][bflip]) + (Fim[b1][bflip] * Fim[b1][bflip]);
	}
    }

    // Writes to shared memory.
    __device__ void write(float *sp)
    {
	// r1 r0 <-> (bouter) (b0)
	// t4 t3 t2 t1 t0 <-> (b1) (ns5) (ns4) (ns3) (ns2)

	xxx;  // warp shuffle
	
	// r1 r0 <-> (ns5) (b0)
	// t4 t3 t2 t1 t0 <-> (b1) (bouter) (ns4) (ns3) (ns2)

	// ns5_stride = 32, b0_stride = 133
	sp[soff_i] = I[0][0];
	sp[soff_i + 133] = I[0][1];
	sp[soff_i + 32] = I[1][0];
	sp[soff_i + 32+133] = I[1][1];

	// Reset
	I[0][0] = I[0][1] = I[1][0] = I[1][1] = 0.0f;
    }
};

#endif  // FFT2


// -------------------------------------------------------------------------------------------------
//
// Interpolation


// Caller must check that 1 <= x <= (n-2) within roundoff error.
// (FIXME comment on how this happens in full kernel.)

__device__ void grid_interpolation_site(float x, int n, int &ix, float &dx)
{
    ix = int(x);
    ix = (ix >= 1) ? ix : 0;
    ix = (ix <= (n-3)) ? ix : (n-3);
    dx = x - float(ix);
}


__device__ void compute_interpolation_weights(float dx, float &w0, float &w1, float &w2, float &w3)
{
    static constexpr float one_sixth = 1.0f / 6.0f;
    static constexpr float one_half = 1.0f / 2.0f;
    
    w0 = -one_sixth * (dx) * (dx-1.0f) * (dx-2.0f);
    w1 = one_half * (dx+1.0f) * (dx-1.0f) * (dx-2.0f);
    w2 = -one_half * (dx+1.0f) * (dx) * (dx-2.0f);
    w3 = one_sixth * (dx+1.0f) * (dx) * (dx-1.0f);
}


// Helper for interpolate_slow()
__device__ float _interpolate_slow_1d(const float *sp, float wy0, float wy1, float wy2, float wy3)
{
    return wy0*sp[0] + wy1*sp[1] + wy2*sp[2] + wy3*sp[3];
}


// Interpolate on (24,128) grid in shared memory, stride=133.
// Caller must check that 1 <= x <= 22, and 1 <= y <= 126, within roundoff error.
__device__ float interpolate_slow(const float *sp, float x, float y)
{
    int ix, iy;
    float dx, dy;

    grid_interpolation_site(x, 24, ix, dx);
    grid_interpolation_site(y, 128, iy, dy);
    sp += 133*(ix-1) + (iy-1);

    float wx0, wx1, wx2, wx3, wy0, wy1, wy2, wy3;
    compute_interpolation_weights(dx, wx0, wx1, wx2, wx3);
    compute_interpolation_weights(dy, wy0, wy1, wy2, wy3);

    float ret = wx0 * _interpolate_slow_1d(sp, wy0, wy1, wy2, wy3);
    ret += wx1 * _interpolate_slow_1d(sp+133, wy0, wy1, wy2, wy3);
    ret += wx2 * _interpolate_slow_1d(sp+2*133, wy0, wy1, wy2, wy3);
    ret += wx3 * _interpolate_slow_1d(sp+3*133, wy0, wy1, wy2, wy3);

    return ret;
}


// Factor interpolation weight as w_j = pf * (x+a) * (x+b) * (x+c), where 0 <= j < 4
__device__ void compute_abc(int j, float &pf, float &a, float &b, float &c)
{
    static constexpr float one_sixth = 1.0f / 6.0f;
    static constexpr float one_half = 1.0f / 2.0f;

    pf = ((j==0) || (j==3)) ? one_sixth : one_half;
    pf = (j & 1) ? pf : (-pf);
	
    a = (j > 0) ? 1.0f : 0.0f;
    b = (j > 1) ? 0.0f : -1.0f;
    c = (j > 2) ? -1.0f : -2.0f;
}


template<bool Debug>
__device__ float interpolate_fast(const float *sp, float x, float y)
{
    int ix_g, iy_g;
    float dx_g, dy_g;
    float ret = 0.0;
    
    grid_interpolation_site(x, 24, ix_g, dx_g);
    grid_interpolation_site(y, 128, iy_g, dy_g);
    
    int jx = (threadIdx.x >> 2) & 3;
    int jy = (threadIdx.x & 3);
    int ds = 133*(jx-1) + (jy-1);
    int sg = 133*ix_g + iy_g;

    float pfx, pfy, ax, bx, cx, ay, by, cy;
    compute_abc(jx, pfx, ax, bx, cx);
    compute_abc(jy, pfy, ay, by, cy);
    pfx *= pfy;  // save one register
    
    for (int iouter = 0; iouter < 16; iouter++) {
	int src_lane = (threadIdx.x & 0x10) | iouter;
	
	int s = __shfl_sync(0xffffffff, sg, src_lane) + ds;
	float dx = __shfl_sync(0xffffffff, dx_g, src_lane);
	float dy = __shfl_sync(0xffffffff, dy_g, src_lane);

	if constexpr (Debug) {
	    assert(s >= 0);
	    assert(s < 24*133);
	    uint m = __match_any_sync(0xffffffff, s & 31);
	    assert(__popc(m) <= 2);  // at most 2:1 bank conflict
	}
	
	float w = pfx * (dx+ax) * (dx+bx) * (dx+cx) * (dy+ay) * (dy+by) * (dy+cy);
	float t = w * sp[s];

	// FIXME placeholder for fast reduce
	t += __shfl_sync(0xffffffff, t, threadIdx.x ^ 1);
	t += __shfl_sync(0xffffffff, t, threadIdx.x ^ 2);
	t += __shfl_sync(0xffffffff, t, threadIdx.x ^ 4);
	t += __shfl_sync(0xffffffff, t, threadIdx.x ^ 8);
	ret = ((threadIdx.x & 15) == iouter) ? t : ret;
    }

    return ret;
}


// Launch with 32 threads, 1 block.
//   - out_slow: shape (32,)
//   - out_fast: shape (32,)
//   - xy: shape (2,32)
//   - grid: shape (24,133)

__global__ void casm_interpolation_test_kernel(float *out_slow, float *out_fast, const float *xy, const float *grid)
{
    __shared__ float sgrid[24*133];

    for (int i = threadIdx.x; i < 24*133; i += 32)
	sgrid[i] = grid[i];
    
    float x = xy[threadIdx.x];
    float y = xy[threadIdx.x + 32];
    
    out_slow[threadIdx.x] = interpolate_slow(sgrid, x, y);
    out_fast[threadIdx.x] = interpolate_fast<true> (sgrid, x, y);
}


static void test_casm_interpolation()
{
    Array<float> xy({64}, af_rhost);
    Array<float> grid({24,133}, af_random | af_gpu);
    Array<float> out_slow({32}, af_random | af_gpu);
    Array<float> out_fast({32}, af_random | af_gpu);

    for (int i = 0; i < 32; i++) {
	xy.data[i] = rand_uniform(1.0f, 22.0f);
	xy.data[i+32] = rand_uniform(1.0f, 126.0f);
    }

    xy = xy.to_gpu();
    
    casm_interpolation_test_kernel<<<1,32>>> (out_slow.data, out_fast.data, xy.data, grid.data);
    CUDA_PEEK("casm_interpolation_test_kernel");
    
    assert_arrays_equal(out_slow, out_fast, "slow", "fast", {"i"});
    cout << "test_casm_interpolation: pass" << endl;
}


// -------------------------------------------------------------------------------------------------


void test_casm()
{
    test_casm_fft_c2c();
    test_casm_interpolation();
}


}  // namespace pirate
