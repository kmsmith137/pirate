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


// Shared memory layout:
//
//  uint E[24][259];
//  float G[4][2][6][2][128];   // (time,pol,ew,reim,ns), ew-stride = 257
//  float I[24][128];           // (ew,ns), ew-stride = 133
//
// Total bytes = 4 * (6216 + 12336 + 3192) = 86976.


// -------------------------------------------------------------------------------------------------


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
template<typename T>
__device__ void warp_shuffle(T &x, T &y, uint bit)
{
    static_assert(sizeof(T) == 4);

    bool flag = (threadIdx.x & bit) != 0;
    T z = __shfl_sync(0xffffffff, (flag ? x : y), threadIdx.x ^ bit);
    x = flag ? z : x;  // compiles to conditional (predicated) move, not branch
    y = flag ? y : z;  // compiles to conditional (predicated) move, not branch
}


template<bool Debug>
__device__ void check_bank_conflict_free(int offset, int max_conflicts=1)
{
    if constexpr (Debug) {
	uint m = __match_any_sync(0xffffffff, offset & 31);
	assert(__popc(m) <= max_conflicts);
	assert(offset >= 0);
    }
}


// -------------------------------------------------------------------------------------------------
//
// casm_shuffle_state
//
// Reindex (time,pol) as (i,j) where
//   i1 i0 <-> t2 t1
//   j2* j1 j0 <-> t3* t0 pol
//
// load_ungridded_e(): Delta(t)=24
// write_ungridded_e(): Delta(t)=24
// grid_shared_e(): Delta(t)=48
// load_gridded_e(): Delta(t)=8
// unpack_e(): Delta(t)=2
//
// Shared memory layout:
//
//   uint E[24][259];   // first index is j, corresponds to 48 time samples
//
// The length-259 axis represents either a shape-(256,) or shape-(6,43) array,
// padded to stride 259.
//
// Assumes {32,24,1} thread grid, not {32*24,1,1}.

__device__ void double_byte_perm(uint &x, uint &y, uint s1, uint s2)
{
    uint xx = x;
    x = __byte_perm(xx, y, s1);
    y = __byte_perm(xx, y, s2);
}

__device__ float unpack_int4(uint x, uint s)
{
    x = ((x >> s) ^ 0x88888888) & 0xf;
    return float(x) - 8.0f;
}


template<bool Debug>
struct casm_shuffle_state
{
    // Managed by setup_global_to_shared().
    const uint4 *ep4;
    uint soff_gs;
    uint4 e4;

    // Managed by setup_gridding(), grid_shared_e().
    uint grid_dst;

    // Managed by setup_unpacking(), load_gridded_e(), unpack_e().
    uint soff_ge;
    uint e0, e1;

    // 'global_e' points to (T,F,2,256)
    // 'global_gridding' points to length-256, elements are (43*ew+ns).
    
    __device__ casm_shuffle_state(const uint8_t *global_e, const uint *global_gridding)
    {
	if constexpr (Debug) {
	    assert(blockDim.x == 32);
	    assert(blockDim.y == 24);
	    assert(blockDim.z == 1);
	}

	setup_global_to_shared(global_e);
	setup_gridding(global_gridding);
	setup_unpacking();
    }

    // These member functions manage copying the ungridded E-array from global
    // memory to shared memory, with Delta(t)=24.
    //
    //   setup_global_to_shared()
    //   load_ungridded_e()
    //   write_ungridded_e()
    //
    // Data is read from global memory with (T=24, I=4, J=48, D=128):
    //
    //   b1 b0 <-> d1 d0
    //   r1 r0 <-> d3 d2
    //   l4 l3 l2 l1 l0 <-> i1 i0 d6 d5 d4   (= t2 t1 d6 d5 d4)
    //   w1* w0 <-> j0* d7                   (= ... t3* t0 pol d7)
    //
    // Then some shuffling operations are performed, to obtain:
    //
    //   b1 b0 <-> i1 i0
    //   r1 r0 <-> d6 d5
    //   l4 l3 l2 l1 l0 <-> d3 d2 d1 d0 d4
    //   w1* w0 <-> j0* d7
    //
    // before writing to shared memory.
    
    __device__ void setup_global_to_shared(const uint8_t *global_e)
    {
	uint l = threadIdx.x;  // lane id
	uint w = threadIdx.y;  // warp id
	uint f = blockIdx.x;   // frequency channel for this threadblock
	uint F = gridDim.x;    // total frequency channels

	// Initialize the 'ep4' global memory pointer.
	//
	//   b1 b0 <-> d1 d0
	//   r1 r0 <-> d3 d2
	//   l4 l3 l2 l1 l0 <-> i1 i0 d6 d5 d4   (= t2 t1 d6 d5 d4)
	//   w1* w0 <-> j0* d7                   (= ... t3* t0 pol d7)

	// (warpId, laneId) -> (i, j, d16)
	uint i = (l >> 3);
	uint j = (w >> 1);
	uint d16 = (l & 0x7) | ((w & 1) << 3);

	// (i, j) -> (pol, t)
	//   i1 i0 <-> t2 t1
	//   j2* j1 j0 <-> t3* t0 pol
	
	uint pol = (j & 1);
	uint t = (i << 1) | ((j & 2) >> 1) | ((j >> 2) << 3);
	
	// uint8 E[T][F][2][256];
	// uint128 E[T][F][2][16];
	ep4 = (const uint4 *) global_e;
	ep4 += 32*(t*F+f) + 16*pol + d16;

	// Initialize the 'goff_gs' shared memory offset.
	//   b1 b0 <-> i1 i0
	//   r1 r0 <-> d6 d5
	//   l4 l3 l2 l1 l0 <-> d3 d2 d1 d0 d4   (note permutation)
	//   w1* w0 <-> j0* d7

	uint d = (l >> 1) | ((l & 0x1) << 4) | ((w & 0x1) << 7);
	soff_gs = 259*j + d;   // note that j was previously initialized above
	
	check_bank_conflict_free<Debug>(soff_gs);
    }
    
    __device__ void load_ungridded_e()
    {
	e4 = *ep4;
    }

    __device__ void write_ungridded_e(uint *sp)
    {
	// At top, 'e4' has register assignment:
	//
	//   b1 b0 <-> d1 d0
	//   r1 r0 <-> d3 d2
	//   l4 l3 l2 l1 l0 <-> i1 i0 d6 d5 d4
	//   w1* w0 <-> j0* d7

	// swap (r1,r0) <-> (l4,l3)
	warp_shuffle(e4.x, e4.y, 8);
	warp_shuffle(e4.z, e4.w, 8);
	warp_shuffle(e4.x, e4.z, 16);
	warp_shuffle(e4.y, e4.w, 16);

	// b1 b0 <-> d1 d0
	// r1 r0 <-> i1 10
	// l4 l3 l2 l1 l0 <-> d3 d2 d6 d5 d4
	// w1* w0 <-> j0* d7

	// swap (b1,b0) <-> (r1,r0)
	double_byte_perm(e4.x, e4.y, 0x6240, 0x7351);
	double_byte_perm(e4.z, e4.w, 0x6240, 0x7351);
	double_byte_perm(e4.x, e4.z, 0x5410, 0x7632);
	double_byte_perm(e4.y, e4.w, 0x5410, 0x7632);
	
	// b1 b0 <-> i1 i0
	// r1 r0 <-> d1 d0
	// l4 l3 l2 l1 l0 <-> d3 d2 d6 d5 d4
	// w1* w0 <-> j0* d7

	// swap (r1,r0) <-> (l2,l1)
	warp_shuffle(e4.x, e4.y, 2);
	warp_shuffle(e4.z, e4.w, 2);
	warp_shuffle(e4.x, e4.z, 4);
	warp_shuffle(e4.y, e4.w, 4);
	
	// At bottom, we have the register assignment that was assumed
	// when computing 'goff_gs' above:
	//
	//   b1 b0 <-> i1 i0
	//   r1 r0 <-> d6 d5
	//   l4 l3 l2 l1 l0 <-> d3 d2 d1 d0 d4   (note permutation)
	//   w1* w0 <-> j0* d7
	
	sp[soff_gs] = e4.x;
	sp[soff_gs + 32] = e4.y;
	sp[soff_gs + 64] = e4.z;
	sp[soff_gs + 96] = e4.w;

	// Advance global E-array pointer 'ep4'.

	uint F = gridDim.x;    // total frequency channels
	ep4 += 24*32*F;        // advance by 24 time samples
    }

    // These member functions are responsible for "gridding" the E-array
    // in shared memory.
    //
    //   setup_gridding()
    //   grid_shared_e()
    //
    // A little awkward, since we want to loop over 256 dishes with 24 warps.
    // Note that 256 = 10*240 + 16.

    __device__ void setup_gridding(const uint *global_gridding)
    {
	// FIXME loading directly from global memory -- rethink?
	uint l = threadIdx.x;
	uint w = threadIdx.y;
	uint nd = (w < 16) ? 11 : 10;
	uint d0 = (w < 16) ? (11*w) : (10*w+16);
	grid_dst = (l < nd) ? global_gridding[d0+l] : 100000;
    }
    
    __device__ void grid_shared_e(uint *sp)
    {
	uint e[11];
	uint j = threadIdx.x;  // lane
	uint w = threadIdx.y;  // warp
	uint d0 = (w < 16) ? (11*w) : (10*w+16);

	#pragma unroll
	for (int i = 0; i < 10; i++)
	    e[i] = (j < 24) ? sp[259*j + d0+i] : 0;
	
	if (w < 16)
	    e[10] = (j < 24) ? sp[259*j + d0+10] : 0;
	
	__syncthreads();

	// FIXME temporary convenience in testing!
	for (int i = 32*w + j; i < 24*259; i += 24*32)
	    sp[i] = 0;
	__syncthreads();
	
	#pragma unroll
	for (int i = 0; i < 10; i++) {
	    uint dst = __shfl_sync(0xffffffff, grid_dst, i);
	    if (j < 24)
		sp[259*j + dst] = e[i];
	}

	if (w < 16) {
	    uint dst = __shfl_sync(0xffffffff, grid_dst, 10);
	    if (j < 24)
		sp[259*j + dst] = e[10];
	}
    }

    // The member functions
    //
    //   setup_unpacking()
    //   load_gridded_e()
    //   unpack_e()
    //
    // manage reading the int4+4 gridded E-array from shared memory with
    // Delta(t)=8, and "unpacking" to float registers with Delta(t)=2.
    //
    // The int4+4 E-array uses register assignment (T=8, P=2, EW=6, NS=64):
    //
    //   b2 b1 b0 <-> t2 t1 ReIm     (= i1 i0 ReIm)
    //   r0 <-> ns5
    //   l4 l3 l2 l1 l0 <-> ns4 ns3 ns2 ns1 ns0
    //   w2* w1 w0 <-> ew t0 pol  (= j1 j0)
    //
    // and is "unpacked" to floats with assignment (T=2, P=2, EW=6, NS=64):
    //
    //   r1 r0 <-> ns5 ReIm
    //   l4 l3 l2 l1 l0 <-> ns4 ns3 ns2 ns1 ns0
    //   w2* w1 w0 <-> ew t0 pol  (= j1 j0)

    __device__ void setup_unpacking()
    {
	uint j = (threadIdx.y & 3);
	uint ew = (threadIdx.y >> 2);
	soff_ge = 259*j + 43*ew + threadIdx.x;
    }

    __device__ void load_gridded_e(const uint *sp)
    {
	e0 = sp[soff_ge];
	e1 = (threadIdx.x < 11) ? sp[soff_ge+32] : 0;
    }
    
    __device__ void unpack_e(int i, float &e0_re, float &e0_im, float &e1_re, float &e1_im)
    {
	// i is the length-4 index (t2 t1).
	// FIXME save a few cycles by offset encoding earlier?

	e0_re = unpack_int4(e0, 8*i);
	e0_im = unpack_int4(e0, 8*i+4);
	e1_re = unpack_int4(e1, 8*i);
	e1_im = unpack_int4(e1, 8*i+4);
    }
};


// Launch with {32,24,1} threads and {F,1,1} blocks.
// T must be a multiple of 48.
//
//   uint8 e_in[T][F][2][128];
//   uint gridding[256];
//   float out[T][F][2][6][64][2];  // (time,freq,pol,ew,ns,reim)

__global__ void casm_shuffle_test_kernel(const uint8_t *e_in, const uint *gridding, float *out, int T)
{
    __shared__ uint shmem[24*259];  // Delta(t)=48

    assert(blockDim.x == 32);
    assert(blockDim.y == 24);
    assert(blockDim.z == 1);
    assert(gridDim.y == 1);
    assert(gridDim.z == 1);
    assert((T % 48) == 0);
    
    casm_shuffle_state<true> shuffle(e_in, gridding);  // Debug=true

    // Set up writing to the 'out' array, with Delta(t)=2.
    // Output from casm_shuffle_state::unpack_e() has register assignment:
    //
    //   r1 r0 <-> ns5 ReIm
    //   l4 l3 l2 l1 l0 <-> ns4 ns3 ns2 ns1 ns0
    //   w2* w1 w0 <-> ew t0 pol

    uint f = blockIdx.x;   // frequency channel of threadblock
    uint F = gridDim.x;    // total frequency channels
    uint pol = (threadIdx.y & 1);
    uint t0 = (threadIdx.y & 2) >> 1;
    uint ew = (threadIdx.y >> 2);
    uint ns = threadIdx.x;

    // float out[T][F][2][6][64][2];   // time,freq,pol,ew,ns,reim
    out += 2 * ns;
    out += 64*2 * ew;
    out += 6*64*2 * pol;
    out += 2*6*64*2 * f;
    out += F*2*6*64*2 * t0;
    
    for (int touter = 0; touter < T; touter += 48) {
	for (int s = 0; s < 2; s++) {
	    // Delta(t)=24, Delta(j)=12
	    shuffle.load_ungridded_e();
	    shuffle.write_ungridded_e(shmem + 12*259*s);
	}

	__syncthreads();
	
	shuffle.grid_shared_e(shmem);
	
	__syncthreads();

	for (int s = 0; s < 6; s++) {
	    // Delta(t)=8, Delta(j)=4
	    shuffle.load_gridded_e(shmem + 4*259*s);
	    
	    for (int i = 0; i < 4; i++) {
		// Delta(t)=2
		float e0_re, e0_im, e1_re, e1_im;
		shuffle.unpack_e(i, e0_re, e0_im, e1_re, e1_im);
		
		// r1 r0 <-> ns5 ReIm
		// l4 l3 l2 l1 l0 <-> ns4 ns3 ns2 ns1 ns0
		// w2* w1 w0 <-> ew t0 pol

		// float out[T][F][2][6][64][2]
		out[0] = e0_re;
		out[1] = e0_im;
		out[64] = e1_re;
		out[65] = e1_im;

		// Delta(t)=2
		out += (2*F*2*6*64*2);
	    }
	}

	__syncthreads();
    }
}

// uint8 e_in[T][F][2][256];
// float out[T][F][2][6][64][2];  // (time,freq,pol,ew,ns,reim)
// uint gridding[256];

void casm_shuffle_reference_kernel(const uint8_t *e_in, const uint *gridding, float *out, int T, int F)
{
    int TFP = 2*T*F;
    memset(out, 0, TFP * 6*64*2 * sizeof(float));

    vector<uint> regridding(256);
    
    // Convert 'gridding' indices (43*ew+ns) to 'regridding' indices (64*ew+ns)
    for (int d = 0; d < 256; d++) {
	uint g = gridding[d];
	uint ew = g / 43;
	uint ns = g - (43*ew);

	assert((ew < 6) && (ns < 43));
	regridding[d] = 64*ew + ns;
    }
    
    for (int tfp = 0; tfp < TFP; tfp++) {
	const uint8_t *e2 = e_in + 256*tfp;  // points to shape (256,)
	float *out2 = out + 6*64*2*tfp;    // points to shape (6,64,2)

	for (int d = 0; d < 256; d++) {
	    uint8_t e = e2[d] ^ 0x88888888;
	    float e_re = float(e & 0xf) - 8.0f;
	    float e_im = float(e >> 4) - 8.0f;
		
	    uint rg = regridding[d];
	    out2[2*rg] = e_re;
	    out2[2*rg+1] = e_im;
	}
    }
}


// Helper for test_casm_shuffle()
static Array<uint8_t> make_random_e_array(int T, int F)
{
    Array<uint8_t> ret({T,F,2,256}, af_rhost | af_zero);

    uint *p = (uint *) (ret.data);
    int n = (ret.size >> 2);

    for (int i = 0; i < n; i++) {
	uint t1 = ksgpu::default_rng();
	uint t2 = ksgpu::default_rng();
	p[i] = t1 ^ (t2 << 16);
    }

    return ret;
}


// Helper for test_casm_shuffle()
static Array<uint> make_random_gridding()
{
    vector<int> v(6*43);
    for (int i = 0; i < 6*43; i++)
	v[i] = (i < 256) ? i : -1;

    ksgpu::randomly_permute(v);

    Array<uint> ret({256}, af_rhost);
    for (int d = 0; d < 256; d++)
	ret.data[d] = 6*43;
    
    for (int i = 0; i < 6*43; i++)
	if (v[i] >= 0)
	    ret.at({v[i]}) = i;
    
    for (int d = 0; d < 256; d++)
	assert(ret.data[d] < 6*43);

    return ret;
}


void test_casm_shuffle()
{
    auto v = random_integers_with_bounded_product(2, 100);
    int T = v[0] * 48;  // T must be a multiple of 48
    int F = v[1];
    
    Array<uint8_t> e = make_random_e_array(T,F);
    Array<uint> gridding = make_random_gridding();
    Array<float> out_cpu({T,F,2,6,64,2}, af_random | af_rhost);
    Array<float> out_gpu({T,F,2,6,64,2}, af_random | af_gpu);
	     
    casm_shuffle_reference_kernel(e.data, gridding.data, out_cpu.data, T, F);

    e = e.to_gpu();
    gridding = gridding.to_gpu();
    casm_shuffle_test_kernel<<<F, {32,24,1}>>> (e.data, gridding.data, out_gpu.data, T);
    CUDA_PEEK("casm_shuffle_test_kernel");

    assert_arrays_equal(out_cpu, out_gpu, "cpu", "gpu", {"t","f","p","ew","ns","reim"});
    cout << "test_casm_shuffle(T=" << T << ",F=" << F << "): pass" << endl;
}


// -------------------------------------------------------------------------------------------------
//
// fft_c2c (helper for "FFT1" kernel)
//
// FIXME implementing fft_c2c_state<2> could save two persistent registers and a few FMAs


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
    Array<float> out_cpu({S,N,2}, af_zero | af_rhost);
    Array<float> out_gpu({S,N,2}, af_random | af_gpu);
    
    for (int j = 0; j < N; j++) {
	for (int k = 0; k < N; k++) {
	    float theta = (2*M_PI/N) * ((j*k) % N);
	    float cth = cosf(theta);
	    float sth = sinf(theta);
	    
	    for (int s = 0; s < S; s++) {
		float xre = in.at({s,k,0});
		float xim = in.at({s,k,1});
		
		out_cpu.at({s,j,0}) += (cth*xre - sth*xim);
		out_cpu.at({s,j,1}) += (sth*xre + cth*xim);
	    }
	}
    }

    in = in.to_gpu();
    fft_c2c_test_kernel<R> <<<1,32>>> (in.data, out_gpu.data);
    CUDA_PEEK("fft_c2c_test_kernel");

    assert_arrays_equal(out_cpu, out_gpu, "cpu", "gpu", {"s","i","reim"}, 1.0e-3);
    cout << "test_casm_fft_c2c: pass" << endl;
}


// -------------------------------------------------------------------------------------------------
//
// FFT1
//
// Implements a zero-padded c2c FFT with 64 inputs and 128 outputs
//
// Input register assignment:
//   r1 r0 <-> x5 ReIm
//   t4 t3 t2 t1 t0 <-> x4 x3 x2 x1 x0
//
// Outputs are written to shared memory:
//   float G[W][2][128]   strides (257,128,1)   W=warps per threadblock
//
// NOTE: assumes that threads are a {32,W,1} grid (not a {32*W,1,1} grid).


template<bool Debug>
struct fft1_state
{
    fft_c2c_state<6> next_fft;
    float cre, cim;   // "twiddle" factor exp(2*pi*i t / 128)
    int sbase;        // shared memory offset

    __device__ fft1_state()
    {
	if constexpr (Debug) {
	    assert(blockDim.x == 32);
	    assert(blockDim.z == 1);
	}
	
	float x = 0.04908738521234052f * threadIdx.x;   // constant is (2pi)/128
	sincosf(x, &cim, &cre);                         // note ordering (im,re)

	// Shared memory writes will use register assignment (see below):
	//   t4 t3 t2 t1 t0 <-> y1 y2 y3 y4 y0
	//
	// The 'sbase' offset assumes y5=y6=ReIm=0
	
	sbase = (threadIdx.y) * 257;              // warp id
	sbase += (threadIdx.x & 1);               // t0 <-> y0
	sbase += __brev(threadIdx.x >> 1) >> 27;  // t4 t3 t2 t1 <-> y1 y2 y3 y4
	
	check_bank_conflict_free<Debug> (sbase);
    }
    
    __device__ void apply(float x0_re, float x0_im, float x1_re, float x1_im, float *sp)
    {
	// y0 = x0 * exp(2*pi*i t / 128) = c*x0
	// y1 = x1 * exp(2*pi*i (t+32) / 128) = i*c*x0
	//   where t = 0, ..., 31
	
	float y0_re = cre*x0_re - cim*x0_im;
	float y0_im = cim*x0_re + cre*x0_im;

	float y1_re = -cim*x1_re - cre*x1_im;
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
};


// Call with {W,32} threads.
// Input array has shape (W,64,2).
// Output array has shape (W,128,2).

template<int W>
__global__ void fft1_test_kernel(const float *in, float *out)
{
    __shared__ float shmem[W*257];

    // Input register assignment:
    //   r1 r0 <-> x5 ReIm
    //   t4 t3 t2 t1 t0 <-> x4 x3 x2 x1 x0

    int w = threadIdx.y;  // warp id
    float x0_re = in[128*w + 2*threadIdx.x];
    float x0_im = in[128*w + 2*threadIdx.x + 1];
    float x1_re = in[128*w + 2*threadIdx.x + 64];
    float x1_im = in[128*w + 2*threadIdx.x + 65];

    fft1_state<true> fft1;  // Debug=true
    fft1.apply(x0_re, x0_im, x1_re, x1_im, shmem);

    for (int reim = 0; reim < 2; reim++)
	for (int y = threadIdx.x; y < 128; y += 32)
	    out[256*w + 2*y + reim] = shmem[257*w + 128*reim + y];
}


void test_casm_fft1()
{
    static constexpr int W = 24;

    Array<float> in({W,64,2}, af_random | af_rhost);
    Array<float> out_cpu({W,128,2}, af_zero | af_rhost);
    Array<float> out_gpu({W,128,2}, af_random | af_gpu);

    for (int j = 0; j < 128; j++) {
	for (int k = 0; k < 64; k++) {
	    float theta = (2*M_PI/128) * ((j*k) % 128);
	    float cth = cosf(theta);
	    float sth = sinf(theta);
	    
	    for (int w = 0; w < W; w++) {
		out_cpu.at({w,j,0}) += (cth * in.at({w,k,0})) - (sth * in.at({w,k,1}));
		out_cpu.at({w,j,1}) += (sth * in.at({w,k,0})) + (cth * in.at({w,k,1}));
	    }
	}
    }

    in = in.to_gpu();
    fft1_test_kernel<W> <<<1,{32,W,1}>>> (in.data, out_gpu.data);
    CUDA_PEEK("fft1_test_kernel");
    
    assert_arrays_equal(out_cpu, out_gpu, "cpu", "gpu", {"w","i","reim"}, 1.0e-3);
    cout << "test_casm_fft1: pass" << endl;
}


// -------------------------------------------------------------------------------------------------
//
// FFT2
//
// There are 24 east-west beams 0 <= b < 24 and 6 east-west feeds 0 <= f < 6.
//
// Alternate parameterization: (bouter,binner) and (fouter,finner) where:
//
//   b = bouter ? (12+binner) : (11-binner)   0 <= bouter < 2    0 <= binner < 12
//   f = fouter ? (3+finner)  : (2-finner)    0 <= fouter < 2    0 <= finner < 3
//
// This parameterization is defined so that if we flip 'bouter' at fixed 'binner',
// the beam location goes to its negative, and likewise for feeds.
//
// We sometimes write 'binner' as:
//
//   binner = 4*b2 + 2*b1 + b0     0 <= b2 < 3    0 <= b1 < 2    0 <= b0 < 2
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
//   r1 r0 <-> (bouter) (b1)
//   t4 t3 t2 t1 t0 <-> (ns4) (ns3) (ns2) (ns1) (b0)
//   24 warps <-> (b2) (ns6) (ns5) (ns0)
//
// The G-array shared memory layout is (where "f" is an EW feed):
//
//   float G[6][2][128];  // (f,reim,ns), strides (257,128,1)
//
// We read from the G-array in register assignment:
//
//   t4 t3 t2 t1 t0 <-> (ns4) (ns3) (ns2) (ns1) (fouter)
//
// This is bank conflict free, since flipping 'fouter' always produces
// an odd change in f, and the f-stride is odd (257).
//
// The I-array shared memory layout is (where "b" indexes an EW beam):
//
//   float I[24][128];   // (b,ns), strides (133,1)
//
// We write to the I-array in the bank conflict free register assignment:
//
//   t4 t3 t2 t1 t0 <-> (ns4) (ns3) (ns2) (ns1) (b0)


template<bool Debug>
struct fft2_state
{
    // Note: we use 12 persistent registers/thread to store beamforming
    // phases, but the number of distinct phases is 24/warp or 72/block.
    // Maybe better to distribute registers as needed with __shfl_sync()?
     
    float I[2][2];      // beams are indexed by (bouter, b1)
    float pcos[2][3];   // beamforming phases are indexed by (b1, finner)
    float psin[2][3];   // beamforming phases are indexed by (b1, finner)
    
    int soff_g;     // base shared memory offset in G-array.
    int soff_i0;    // base shared memory offset in I-array, bouter=0
    int soff_i1;    // base shared memory offset in I-array, bouter=1


    // FIXME is alpha[3] the best interface here?
    __device__ fft2_state(float alpha[3])
    {
	if constexpr (Debug) {
	    assert(blockDim.x == 32);
	    assert(blockDim.y == 24);
	    assert(blockDim.z == 1);
	}
	
	I[0][0] = I[0][1] = I[1][0] = I[1][1] = 0.0f;

	// Each warp maps to an (b2, ns56, ns0) triple.
	uint ns0 = (threadIdx.y & 1);        // 0 <= ns0 < 2
	uint ns56 = (threadIdx.y >> 1) & 3;  // 0 <= ns56 < 4
	uint b2 = (threadIdx.y >> 3);        // 0 <= b2 < 3

	// Each thread maps to a (ns14, b0) pair.
	uint b0 = (threadIdx.x & 1);
	uint ns14 = (threadIdx.x >> 1);

	// Beamforming phases are indexed by (b1, finner)
	#pragma unroll
	for (uint b1 = 0; b1 < 2; b1++) {
	    // Beamforming phase is exp(i*t*alpha[finner]) where t = 1+2*binner
	    float t = (b2 << 3) | (b1 << 2) | (b0 << 1) | 1;
	    
	    #pragma unroll
	    for (uint finner = 0; finner < 3; finner++)
		sincosf(t * alpha[finner], &psin[b1][finner], &pcos[b1][finner]);
	}

	// Shared memory offset for reading G-array:
	//
	//   float G[6][2][128];  // (f,reim,ns), strides (257,128,1)
	//
	// When we read from the G-array, we read it as:
	//
	//   t4 t3 t2 t1 t0 <-> (ns4) (ns3) (ns2) (ns1) (fouter)
	//
	// 'soff_g' is the offset assuming finner=reim=0.
	
	uint fouter = threadIdx.x & 1;
	uint ns = (ns56 << 5) | (ns14 << 1) | ns0;
	soff_g = (fouter ? (2*257) : (3*257)) + ns;

	// Shared memory offset for writing I-array:
	//
	//   float I[24][128];   // (b,ns), strides (133,1)
	//
	// When we write to the I-array, we write as
	//
	//   t4 t3 t2 t1 t0 <-> (ns4) (ns3) (ns2) (ns1) (b0)
	//
	// 'soff_i{0,1}' is the offset with bouter={0,1} and b1=0.

	uint binner = (b2 << 2) | b0;
	soff_i0 = 133*(12+binner) + ns;  // offset for bouter=0
	soff_i1 = 133*(11-binner) + ns;  // offset for binner=1

	check_bank_conflict_free<Debug> (soff_i0);
	check_bank_conflict_free<Debug> (soff_i1);
    }

    
    // Accumulates one (time,pol) into I-registers.
    __device__ void apply(const float *sp)
    {
	// Beamformed electric fields are accumulated here.
	float Fre[2][2];   // (bouter, b1)
	float Fim[2][2];   // (bouter, b1)

	Fre[0][0] = Fre[0][1] = Fre[1][0] = Fre[1][1] = 0.0f;
	Fim[0][0] = Fim[0][1] = Fim[1][0] = Fim[1][1] = 0.0f;

	// finner-stride in the G-array
	int fouter = threadIdx.x & 1;
	int ds = fouter ? (-257) : 257;
	
        #pragma unroll
	for (int finner = 0; finner < 3; finner++) {
	    int s = soff_g + finner*ds;
	    check_bank_conflict_free<Debug> (s);
	    
	    float tre = sp[s];
	    float tim = sp[s + 128];

	    // FIXME can be improved.
	    // u{0,1} index is fouter.
	    float u0_re = __shfl_sync(0xffffffff, tre, threadIdx.x & ~1);
	    float u1_re = __shfl_sync(0xffffffff, tre, threadIdx.x | 1);
	    float u0_im = __shfl_sync(0xffffffff, tim, threadIdx.x & ~1);
	    float u1_im = __shfl_sync(0xffffffff, tim, threadIdx.x | 1);

	    #pragma unroll
	    for (int b1 = 0; b1 < 2; b1++) {
		// FIXME can be sped up with FFT-style trick.

		// F[0][b1] += (phase) (u0)
		// F[0][b1] += (phase^*) (u1)
		// F[1][b1] += (phase^*) (u0)
		// F[1][b1] += (phase) (u1)
		
		// FIXME I don't think zma() will be called here, in the final kernel.
		zma(Fre[0][b1], Fim[0][b1], pcos[b1][finner],  psin[b1][finner], u0_re, u0_im);
		zma(Fre[0][b1], Fim[0][b1], pcos[b1][finner], -psin[b1][finner], u1_re, u1_im);  // note (-psin)
		zma(Fre[1][b1], Fim[1][b1], pcos[b1][finner], -psin[b1][finner], u0_re, u0_im);  // note (-psin)
		zma(Fre[1][b1], Fim[1][b1], pcos[b1][finner],  psin[b1][finner], u1_re, u1_im);
	    }
	}

	#pragma unroll
	for (int bouter = 0; bouter < 2; bouter++) {
	    #pragma unroll
	    for (int b1 = 0; b1 < 2; b1++)
		I[bouter][b1] += (Fre[bouter][b1] * Fre[bouter][b1]) + (Fim[bouter][b1] * Fim[bouter][b1]);
	}
    }

    // Writes I[] register to shared memory and zeroes the registers.
    __device__ void write_and_reset(float *sp)
    {
	// Beams are indexed by (bouter, b1).
	//   float I[24][128];   // (b,ns), strides (133,1)

	sp[soff_i0] = I[0][0];
	sp[soff_i1] = I[1][0];
	
	sp[soff_i0 + 2*133] = I[0][1];
	sp[soff_i1 - 2*133] = I[1][1];

	I[0][0] = I[0][1] = I[1][0] = I[1][1] = 0.0f;
    }
};


// float G[TP][6][128][2];
// float I[24][128];
// Launch with {32,24,1} threads.

__global__ void fft2_test_kernel(const float *gp, float *ip, int TP, const float *alpha)
{
    __shared__ float shmem_g[6*257];   // size (6,2,128), strides (257,128,1)
    __shared__ float shmem_i[24*133];  // size (24,128), strides (133,1)

    assert(blockDim.x == 32);
    assert(blockDim.y == 24);
    assert(blockDim.z == 1);

    float a[3];
    a[0] = alpha[0];
    a[1] = alpha[1];
    a[2] = alpha[2];
    
    fft2_state<true> fft2(a);  // Debug=true
    
    // Set up G-array copy (global) -> (shared)
    int gns = ((threadIdx.y & 3) << 5) + threadIdx.x;
    int gew = (threadIdx.y >> 2);
    int gsh = (257*gew) + gns;   // array offset in shmem_g[] array
    float2 *gp2 = (float2 *)(gp) + 128*gew + gns;  // per-(warp+thread) offsets applied
    
    for (int tp = 0; tp < TP; tp++) {
	// G-array copy (global) -> (shared)
	float2 g = *gp2;
	shmem_g[gsh] = g.x;      // real part
	shmem_g[gsh+128] = g.y;  // imag part
	gp2 += 6*128;

	__syncthreads();

	fft2.apply(shmem_g);

	__syncthreads();
    }

    fft2.write_and_reset(shmem_i);
    __syncthreads();
	
    // Set up I-array copy (shared) -> (global)
    int goff = 128*threadIdx.y + threadIdx.x;  // array offset in ip[] global array
    int soff = 133*threadIdx.y + threadIdx.x;  // array offset in shmem_i[] shared array
    
    for (int j = 0; j < 4; j++)
	ip[goff + 32*j] = shmem_i[soff + 32*j];
}


// float G[TP][6][128][2];
// float I[24][128];

void fft2_reference_kernel(const float *gp, float *ip, int TP, const float *alpha)
{
    // beamforming phase is exp(i * bloc[b] * floc[f])
    float bloc[24];
    float floc[6];

    for (int binner = 0; binner < 12; binner++) {
	bloc[12+binner] = 1 + 2*binner;
	bloc[11-binner] = -(1 + 2*binner);
    }

    for (int finner = 0; finner < 3; finner++) {
	floc[3+finner] = alpha[finner];
	floc[2-finner] = -alpha[finner];
    }
    
    for (int ns = 0; ns < 128; ns++) {
	for (int b = 0; b < 24; b++) {
	    float I = 0.0f;
	    
	    for (int tp = 0; tp < TP; tp++) {
		float zre = 0.0f;
		float zim = 0.0f;

		for (int f = 0; f < 6; f++) {
		    float xre = cosf(bloc[b] * floc[f]);
		    float xim = sinf(bloc[b] * floc[f]);
		    float yre = gp[6*256*tp + 256*f + 2*ns];
		    float yim = gp[6*256*tp + 256*f + 2*ns + 1];
		    
		    zre += xre*yre - xim*yim;
		    zim += xre*yim + xim*yre;
		}

		I += (zre*zre + zim*zim);
	    }
	    
	    ip[128*b + ns] = I;
	}
    }
}


void test_casm_fft2()
{
    int TP = 4;
    Array<float> g({TP,6,128,2}, af_rhost | af_random);
    Array<float> i_cpu({24,128}, af_rhost | af_random);
    Array<float> i_gpu({24,128}, af_gpu | af_random);
    Array<float> alpha({3}, af_rhost | af_random);
    
    fft2_reference_kernel(g.data, i_cpu.data, TP, alpha.data);
    
    g = g.to_gpu();
    alpha = alpha.to_gpu();
    
    fft2_test_kernel<<< 1, {32,24,1} >>> (g.data, i_gpu.data, TP, alpha.data);
    CUDA_PEEK("fft2_test_kernel");

    assert_arrays_equal(i_cpu, i_gpu, "cpu", "gpu", {"b","ns"}, 1.0e-4);
    cout << "test_casm_fft2: pass" << endl;
}


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
	check_bank_conflict_free<Debug> (s, 2);   // at most 2:1 bank conflict
	
	float dx = __shfl_sync(0xffffffff, dx_g, src_lane);
	float dy = __shfl_sync(0xffffffff, dy_g, src_lane);
	
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
    test_casm_shuffle();
    test_casm_fft_c2c();
    test_casm_fft1();
    test_casm_fft2();
    test_casm_interpolation();
}


}  // namespace pirate
