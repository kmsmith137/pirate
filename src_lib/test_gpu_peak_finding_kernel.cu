#include "../include/pirate/PeakFindingKernel.hpp"
#include "../include/pirate/cuda_kernels/peak_finding.hpp"
#include "../include/pirate/inlines.hpp"
#include "../include/pirate/utils.hpp"

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


// -------------------------------------------------------------------------------------------------
//
// Test 'struct pf_ringbuf'.


// Launch with one warp!
// 'pstate' has length (O*I).
// 'data' has shape (O, 32*ninner).

template<int I, int O>
__global__ void pf_ringbuf_kernel(int *data, int *pstate, int ninner, int data_stride)
{
    int x[O];

    pf_ringbuf<int,I,O> rb;
    rb.load(pstate);

    for (int n = 0; n < ninner; n++) {
	#pragma unroll
	for (int r = 0; r < O; r++)
	    x[r] = data[r*data_stride + 32*n + threadIdx.x];
	
	rb.template multi_advance<0,O> (x);
	
	#pragma unroll
	for (int r = 0; r < O; r++)
	    data[r*data_stride + 32*n + threadIdx.x] = x[r];
    }

    rb.save(pstate);
}


// Function type for pf_ringbuf_kernel<I,O>
using rb_kernel_t = void (*)(int *, int *, int, int);


// Ugh
template<int I>
static rb_kernel_t get_rb_kernel1(int O)
{
    if (O == 1)
	return pf_ringbuf_kernel<I,1>;
    if (O == 2)
	return pf_ringbuf_kernel<I,2>;
    if (O == 3)
	return pf_ringbuf_kernel<I,3>;
    if (O == 4)
	return pf_ringbuf_kernel<I,4>;
    if (O == 5)
	return pf_ringbuf_kernel<I,5>;
    if (O == 6)
	return pf_ringbuf_kernel<I,6>;
    if (O == 7)
	return pf_ringbuf_kernel<I,7>;
    if (O == 8)
	return pf_ringbuf_kernel<I,8>;
    if (O == 9)
	return pf_ringbuf_kernel<I,9>;
    
    throw runtime_error("get_rb_kernel: invalid O");
}


// Ugh
static rb_kernel_t get_rb_kernel(int I, int O)
{
    if (I == 1)
	return get_rb_kernel1<1> (O);
    if (I == 2)
	return get_rb_kernel1<2> (O);
    if (I == 4)
	return get_rb_kernel1<4> (O);
    if (I == 8)
	return get_rb_kernel1<8> (O);
    if (I == 16)
	return get_rb_kernel1<16> (O);
    
    throw runtime_error("get_rb_kernel: invalid I");
}


void test_gpu_pf_ringbuf()
{
    int I = (1 << rand_int(0,5));
    int O = rand_int(1, 10);
    int ninner = rand_int(1, 10);
    int nouter = rand_int(1, 10);
    rb_kernel_t kernel = get_rb_kernel(I,O);
    
    cout << "test_gpu_ringbuf: I=" << I << ", O=" << O << ", ninner=" << ninner << ", nouter=" << nouter << endl;

    int T = 32*nouter*ninner;
    Array<int> hdata({O,T+I}, af_random | af_rhost);
    
    Array<int> pstate = hdata.slice(1,0,I);   // shape (O,I)
    pstate = pstate.clone().to_gpu();

    Array<int> gdata = hdata.slice(1,I,T+I);  // shape (O,T)
    gdata = gdata.to_gpu();

    for (int n = 0; n < nouter; n++)
	kernel<<<1,32>>> (gdata.data + 32*ninner*n, pstate.data, ninner, gdata.strides[0]);

    assert_arrays_equal(
	hdata.slice(1,0,T),  // shape (O,T)
	gdata,
	"host", "gpu",
	{"r","t"}
    );
}


// -------------------------------------------------------------------------------------------------
//
// Test 'struct pf_core'.


// Helper for pf_core_kernel().
//
// input array shape
//   = (Dt, S)
//   = (Dt, Souter * ST * SS)
//
// output array shapes
//   = (P, S)
//   = (P, Souter * ST * SS)
//
// The J0 argument is the initial "outer" block 0 <= J0 < Souter.


template<class Core, typename T32, int J0=0>
__device__ inline void pf_core_step(Core &core, const T32 *in_th, T32 *out_th, T32 *ssq_th, int out_pstride32)
{
    static_assert(sizeof(T32) == 4);
    
    if constexpr (J0 < Core::Souter) {
	constexpr int P = Core::P;
	constexpr int Dt = Core::Dt;
	constexpr int ST = Core::ST;
	constexpr int Sout = Core::Souter;

	T32 x[Dt];
	
	#pragma unroll
	for (int d = 0; d < Dt; d++)
	    x[d] = in_th[d * Sout*ST];
	
	core.template advance<J0> (x);

	#pragma unroll
	for (int p = 0; p < P; p++) {
	    out_th[p * out_pstride32] = core.pf_out[p];
	    ssq_th[p * out_pstride32] = core.pf_ssq[p];
	}

	// Advance to next J0
	pf_core_step<Core,T32,J0+1> (core, in_th + ST, out_th + ST, ssq_th + ST, out_pstride32);
    }
}


// input array shape = (Tin,S)
// output array shape = (P,Tout,S)   [ for both out and ssq ]
// Launch with one warp

template<typename T32, int Dt, int E, int S>
__global__ void pf_core_kernel(void *in_, void *out_, void *ssq_, void *pstate_, int nt_out, int out_pstride32, int pstate_nbytes)
{
    using Core = pf_core<T32, Dt, E, S>;
    Core core;
    
    constexpr int Souter = Core::Souter;
    constexpr int ST = Core::ST;
	
    T32 *in = reinterpret_cast<T32 *> (in_);
    T32 *out = reinterpret_cast<T32 *> (out_);
    T32 *ssq = reinterpret_cast<T32 *> (ssq_);
    T32 *pstate = reinterpret_cast<T32 *> (pstate_);
    
    assert(pstate_nbytes >= Core::pstate_nbytes_per_warp);
    core.load_pstate(pstate);

    // Apply per-thread offsets to 'in', 'out', 'ssq'.
    // Write laneId = t*ST + s, where 0 <= t < (32/ST) and 0 <= s < ST.
    int s = (threadIdx.x & (ST-1));
    int tST = (threadIdx.x - s);   // (t * ST)
    T32 *in_th = in + (Dt * Souter * tST) + s;
    T32 *out_th = out + (Souter * tST) + s;
    T32 *ssq_th = ssq + (Souter * tST) + s;
    
    for (int t = 0; t < nt_out; t += Core::Tout) {
	pf_core_step(core, in_th, out_th, ssq_th, out_pstride32);
	in_th += Core::Tin * Souter * ST;
	out_th += Core::Tout * Souter * ST;
	ssq_th += Core::Tout * Souter * ST;
    }

    core.save_pstate(pstate);
}

// Function type for pf_core_kernel<T32,Dt,E,S>
using pf_core_kernel_t = void (*)(void *, void *, void *, void *, int, int, int);

// In the next few functions, we build up:
//   get_pf_core_kernel(Dtype dt, int Dt, int E, int S)
//
// which supports values of (Dt,E,S) satisfiying the following constraints:
//   1 <= E <= Dt <= 16
//   Dt*L <= S <= min(4*Dt,16) * L

template<typename T32, int Dt, int E>
static pf_core_kernel_t get_pf_core_kernel3(int S)
{
    constexpr int L = ksgpu::dtype_ops<T32>::simd_lanes;
    
    if (S == Dt*L)
	return pf_core_kernel<T32, Dt, E, Dt*L>;
    
    if constexpr (Dt <= 8)
	if (S == 2*Dt*L)
	    return pf_core_kernel<T32, Dt, E, 2*Dt*L>;

    if constexpr (Dt <= 4)
	if (S == 4*Dt*L)
	    return pf_core_kernel<T32, Dt, E, 4*Dt*L>;

    throw runtime_error("get_pf_core_kernel(): invalid (Dt,S)");
}

template<typename T32, int Dt>
static pf_core_kernel_t get_pf_core_kernel2(int E, int S)
{
    if (E == 1)
	return get_pf_core_kernel3<T32,Dt,1> (S);

    if constexpr (Dt >= 2)
	if (E == 2)
	  return get_pf_core_kernel3<T32,Dt,2> (S);
    
    if constexpr (Dt >= 4)
	if (E == 4)
	    return get_pf_core_kernel3<T32,Dt,4> (S);

    if constexpr (Dt >= 8)
	if (E == 8)
	    return get_pf_core_kernel3<T32,Dt,8> (S);

    if constexpr (Dt >= 16)
	if (E == 16)
	    return get_pf_core_kernel3<T32,Dt,16> (S);

    throw runtime_error("get_pf_core_kernel(): invalid (Dt,E)");
}

template<typename T32>
static pf_core_kernel_t get_pf_core_kernel1(int Dt, int E, int S)
{
    static_assert(sizeof(T32) == 4);

    if (Dt == 1)
	return get_pf_core_kernel2<T32,1> (E,S);
    if (Dt == 2)
	return get_pf_core_kernel2<T32,2> (E,S);
    if (Dt == 4)
	return get_pf_core_kernel2<T32,4> (E,S);
    if (Dt == 8)
	return get_pf_core_kernel2<T32,8> (E,S);
    if (Dt == 16)
	return get_pf_core_kernel2<T32,16> (E,S);
    throw runtime_error("get_pf_core_kernel(): invalid Dt");
}
				  
static pf_core_kernel_t get_pf_core_kernel(const Dtype &dtype, int Dt, int E, int S)
{
    if (dtype == Dtype::native<float>())
	return get_pf_core_kernel1<float> (Dt, E, S);
#if 0
    if (dtype == Dtype::native<__half>())
	return get_pf_core_kernel1<__half2> (Dt, E, S);
#endif
    throw runtime_error("get_pf_core_kernel(): invalue dtype");
}

 
// reference pf_core code starts here.

// Helper for _reference_pf_core().
inline void _update_pf(float pf, int d, float &out, float &ssq)
{
    out = d ? max(out,pf) : pf;
    ssq = d ? (ssq+pf*pf) : (pf*pf);
}


// _reference_pf_core(): processes 3 kernels (boxcar2, gaussian3, gaussian4)
//   x.shape = (Tout*Dt,S)
//   out.shape = (3,Tout,S)
//   ssq.shape = (3,Tout,S)

static void _reference_pf_core(Array<float> &x, Array<float> &out, Array<float> &ssq, long Tout, long Dt, long S, long P, long p0)
{
    const float a = constants::pf_a;
    const float b = constants::pf_b;

    xassert(Dt >= 2);
    
    long Tin = Tout * Dt;
    xassert_shape_eq(x, ({Tin,S}));
    xassert_shape_eq(out, ({P,Tout,S}));
    xassert_shape_eq(ssq, ({P,Tout,S}));
    xassert_le(p0+3, P);
    
    xassert(x.is_fully_contiguous());
    xassert(out.is_fully_contiguous());
    xassert(ssq.is_fully_contiguous());

    // prepad by (Dt+1).
    Array<float> xpad({Tin+Dt+1,S}, af_zero | af_rhost);
    xpad.slice(0,Dt+1,Tin+Dt+1).fill(x);

    for (long tout = 0; tout < Tout; tout++) {
	// xp = array of shape (Dt+3,S).
	// outp = array of shape (3,S), stride Tout*S
	// ssqp = array of shape (3,S), stride Tout*S
	
	float *xp = xpad.data + (tout*Dt) * S;
	float *outp = out.data + tout*S;
	float *ssqp = ssq.data + tout*S;

	for (long d = 0; d < Dt; d++) {
	    for (long s = 0; s < S; s++) {
		float x0 = xp[d*S + s];
		float x1 = xp[(d+1)*S + s];
		float x2 = xp[(d+2)*S + s];
		float x3 = xp[(d+3)*S + s];
		    
		float b2 = x1 + x2;
		float g3 = a*x0 + x1 + a*x2;
		float g4 = b*x0 + x1 + x2 + b*x3;
		
		_update_pf(b2, d, outp[(p0)*Tout*S + s], ssqp[(p0)*Tout*S + s]);
		_update_pf(g3, d, outp[(p0+1)*Tout*S + s], ssqp[(p0+1)*Tout*S + s]);
		_update_pf(g4, d, outp[(p0+2)*Tout*S + s], ssqp[(p0+2)*Tout*S + s]);
	    }
	}
    }
}

// _reference_pf_core(): processes 3 kernels (boxcar2, gaussian3, gaussian4)
//   x.shape = (Tout*Dt,S)
//   out.shape = (P,Tout,S)
//   ssq.shape = (P,Tout,S)

static void reference_pf_core(Array<float> &x, Array<float> &out, Array<float> &ssq, long Tout, long Dt, long E, long S)
{
    xassert(E >= 1);
    xassert(Dt >= E);
    xassert(is_power_of_two(E));
    xassert(is_power_of_two(Dt));
    
    long Tin = Tout * Dt;
    long P = 3*integer_log2(E) + 1;
    
    xassert_shape_eq(x, ({Tin,S}));
    xassert_shape_eq(out, ({P,Tout,S}));
    xassert_shape_eq(ssq, ({P,Tout,S}));
    
    xassert(x.is_fully_contiguous());
    xassert(out.is_fully_contiguous());
    xassert(ssq.is_fully_contiguous());

    for (long s = 0; s < S; s++)
	out.data[s] = ssq.data[s] = 0.0f;
    
    for (long tout = 1; tout < Tout; tout++) {
	// xp = array of shape (Dt,S).
	// outp = ssqp = array of shape (S).
	float *xp = x.data + (tout-1)*Dt*S;
	float *outp = out.data + tout*S;
	float *ssqp = ssq.data + tout*S;
	    
	for (long d = 0; d < Dt; d++)
	    for (long s = 0; s < S; s++)
		_update_pf(xp[d*S+s], d, outp[s], ssqp[s]);
    }

    if (E == 1)
	return;

    Array<float> x_ds = x;
    long p0 = 1;

    for (;;) {
	_reference_pf_core(x_ds, out, ssq, Tout, Dt, S, P, p0);

	if (E == 2)
	    return;
	
	// Downsample by 2 in time. The function name "reference_downsample_freq()"
	// is a misnomer here, but it does the right thing!

	Array<float> xnew = Array<float> ({x_ds.shape[0]/2, S}, af_uhost | af_zero);
	reference_downsample_freq(x_ds, xnew, false);  // normalize=false
	x_ds = xnew;
	
	Dt /= 2;
	E /= 2;
	p0 += 3;
    }
}


void test_gpu_pf_core()
{
    Dtype dtype = Dtype::native<float>();  // FIXME generalize
    long L = 32 / dtype.nbits;             // simd lanes
    
    long lgE = rand_int(0,5);
    long lgD = rand_int(lgE,5);
    long lgSL = rand_int(lgD, min(lgD+3,5L));   // log2(S/L)
    
    long Dt = 1 << lgD;
    long E = 1 << lgE;
    long S = L << lgSL;
    long P = 3*lgE + 1;
    
    long Tk_in = 32*L * rand_int(1,10);  // input time samples per kernel
    long Tk_out = Tk_in / Dt;            // output time samples per kernel
    
    long Nk = rand_int(1,10);   // number of kernels    
    long Tin = Tk_in * Nk;      // total input time samples
    long Tout = Tk_out * Nk;    // total output time samples

    cout << "test_gpu_pf_core: dtype=" << dtype << ", Dt=" << Dt << ", E=" << E
	 << ", S=" << S << ", Tk_in=" << Tk_in << ", Nk=" << Nk << endl;

    Array<float> x({Tin,S}, af_random | af_rhost);
    Array<float> out({P,Tout,S}, af_random | af_rhost);
    Array<float> ssq({P,Tout,S}, af_random | af_rhost);

    reference_pf_core(x, out, ssq, Tout, Dt, E, S);

    Array<void> gx = x.convert(dtype).to_gpu();
    Array<void> gout(dtype, {P,Tout,S}, af_zero | af_gpu);
    Array<void> gssq(dtype, {P,Tout,S}, af_zero | af_gpu);

    long pstate_nbytes = (Dt+5) * S * (dtype.nbits/8);  // upper bound
    Array<char> pstate({pstate_nbytes}, af_zero | af_gpu);

    pf_core_kernel_t kernel = get_pf_core_kernel(dtype, Dt, E, S);
    
    for (long k = 0; k < Nk; k++) {
	long out_pstride32 = (Tout * S * dtype.nbits) / 32;
	char *inp = reinterpret_cast<char *> (gx.data) + k * Tk_in * S * (dtype.nbits / 8);
	char *outp = reinterpret_cast<char *> (gout.data) + k * Tk_out * S * (dtype.nbits / 8);
	char *ssqp = reinterpret_cast<char *> (gssq.data) + k * Tk_out * S * (dtype.nbits / 8);

	kernel<<<1,32>>> (inp, outp, ssqp, pstate.data, Tk_out, out_pstride32, pstate_nbytes);
	CUDA_PEEK("pf_core_test_kernel");
    }

    assert_arrays_equal(out, gout, "hout", "gout", {"p","t","s"});
    assert_arrays_equal(ssq, gssq, "hssq", "gssq", {"p","t","s"});
}


// -------------------------------------------------------------------------------------------------


// Tiny helper for test_full_pf_kernel()
struct pf_accumulator
{
    float rmax = -1.0e20;
    float rssq = 0.0;

    inline void update(float w, float x)
    {
	rmax = max(rmax, w*x);
	rssq += square(w*x);
    }
};


// Nk = number of kernel launches
// Tk_out = number of output times per kernel launch
static void test_full_pf_kernel(const pf_kernel &k, int B, int Mout, int Tk_out, int Nk)
{
    int M = k.M;
    int E = k.E;
    int P = k.P;
    int W = k.W;
    int RW = k.RW;
    int Dout = k.Dout;
    int Dcore = k.Dcore;
    int Tout = Tk_out * Nk;
    int Min = Mout * M;
    int Tin = Tout * Dout;
    int Tk_in = Tk_out * Dout;

    cout << "test_full_pf_kernel: M=" << M << ", E=" << E << ", Dout=" << Dout
	 << ", Dcore=" << Dcore << ", W=" << k.W << ", B=" << B << ", Mout=" << Mout
	 << ", Tk_out=" << Tk_out << ", Nk=" << Nk << endl;
    
    xassert_divisible(32, Dcore);
    xassert_divisible(Tout, 32/Dcore);

    Array<float> out_max({B,P,Mout,Tout}, af_rhost | af_zero);
    Array<float> out_ssq({B,P,Mout,Tout}, af_rhost | af_zero);
    Array<float> in({B,Min,Tin}, af_rhost | af_random);
    Array<float> wt({B,P,Min}, af_rhost | af_zero);

    // GPU kernel assumes weights are positive.
    for (long i = 0; i < wt.size; i++)
	wt.data[i] = rand_uniform(1.0, 2.0);
    
    // This loop just does p=0.
    for (int b = 0; b < B; b++) {
	for (int mout = 0; mout < Mout; mout++) {
	    for (int tout = 0; tout < Tout; tout++) {
		pf_accumulator acc0;

		for (int m = mout*M; m < (mout+1)*M; m++) {
		    float w = wt.at({b,0,m});   // p=0
		    
		    for (int t = tout*Dout; t < (tout+1)*Dout; t++) {
			float x = in.at({b,m,t});
			acc0.update(w, x);
		    }

		    out_max.at({b,0,mout,tout}) = acc0.rmax;  // p=0
		    out_ssq.at({b,0,mout,tout}) = acc0.rssq;  // p=0
		}
	    }
	}
    }

    int isamp = 0;
    Array<float> in_ds = in.clone();

    for (int ids = 0; ids < integer_log2(E); ids++) {
	
	// At top of loop, 'in_ds' is a downsampled version of 'in'.
	// The downsampling level is 2**ids, and the sampling rate is 2**(isamp).
	//
	// To write this out precisely, 'in_ds' has shape (B, Min, Tin/2**isamp).
	// Elements at time 0 <= tds < (Tin/2**isamp) are obtained by summing 'in' over
	//   ((tds+1) * 2**isamp) - 2**ids <= t < (tds+1) * 2**isamp
	//
	// The value of 'ids' increases by 1 in every iteration of the loop,
	// but the value of isamp "saturates" at log2(Dcore).
	//
	// The output arrays have been filled for 0 <= p < (3*ids+1).
	// In this iteration of the loop, we'll fill (3*ids+1) <= p < (3*ids+4).

	long Tds = xdiv(Tin, pow2(isamp));
	xassert_shape_eq(in_ds, ({B,Min,Tds}));
	xassert_eq(isamp, min(ids, integer_log2(Dcore)));
	xassert(in_ds.is_fully_contiguous());  // assumed in downsampling logic

	// For computing profiles.
	long DD = xdiv(Tds, Tout);  // Downsampling factor between 'in_ds' and 'out_{max,ssq}' arrays.
	int p0 = 3*ids+1;           // Base profile index

	// Time offset in 'in_ds' array corresponding to 2**ids samples (used for "prepadding" below)
	long dt = pow2(ids-isamp);
	
	// Compute 3 profiles.
	for (int b = 0; b < B; b++) {
	    for (int mout = 0; mout < Mout; mout++) {
		for (int tout = 0; tout < Tout; tout++) {
		    pf_accumulator acc0;
		    pf_accumulator acc1;
		    pf_accumulator acc2;
		    
		    for (int m = mout*M; m < (mout+1)*M; m++) {
			float w0 = wt.at({b,p0,m});
			float w1 = wt.at({b,p0+1,m});
			float w2 = wt.at({b,p0+2,m});
			float *p = &in_ds.at({b,m,0});
			
			for (int t = tout*DD; t < (tout+1)*DD; t++) {
			    // "Prepadding"
			    float x0 = (t >= 3*dt) ? p[t-3*dt] : 0.0f;
			    float x1 = (t >= 2*dt) ? p[t-2*dt] : 0.0f;
			    float x2 = (t >= dt) ? p[t-dt] : 0.0f;
			    float x3 = p[t];

			    acc0.update(w0, x2 + x3);
			    acc1.update(w1, 0.5f*x1 + x2 + 0.5f*x3);
			    acc2.update(w2, 0.5f*x0 + x1 + x2 + 0.5f*x3);
			}
		    }

		    out_max.at({b,p0,mout,tout}) = acc0.rmax;
		    out_max.at({b,p0+1,mout,tout}) = acc1.rmax;
		    out_max.at({b,p0+2,mout,tout}) = acc2.rmax;
		    
		    out_ssq.at({b,p0,mout,tout}) = acc0.rssq;
		    out_ssq.at({b,p0+1,mout,tout}) = acc1.rssq;
		    out_ssq.at({b,p0+2,mout,tout}) = acc2.rssq;
		}
	    }
	}
    
	// Downsample, to go to next iteration of loop.
	
	if (isamp < integer_log2(Dcore)) {
	    // Case 1: straightforward downsampling (ids, isamp both increase by 1)
	    long Tds2 = xdiv(Tds,2);
	    
	    Array<float> in_ds2({B,Min,Tds2}, af_uhost);
	    float *dst = in_ds2.data;	    
	    float *src = in_ds.data;
	    
	    for (long i = 0; i < B*Min*Tds2; i++)
		dst[i] = src[2*i] + src[2*i+1];
	    
	    in_ds = in_ds2;
	    isamp++;
	}
	else {
	    // Case 2: downsampling without decreasing array size	    
	    Array<float> in_ds2({B,Min,Tds}, af_uhost);
	    float *dst = in_ds2.data;
	    float *src = in_ds.data;

	    for (long i = 0; i < B*Min; i++) {
		for (long t = 0; t < Tds; t++) {
		    float x = (t >= dt) ? src[i*Tds+(t-dt)] : 0.0f;
		    dst[i*Tds+t] = src[i*Tds+t] + x;
		}
	    }

	    in_ds = in_ds2;	    
	    // note that 'isamp' is not incremented here
	}
    }
    

    // At this point, we have computed the host arrays out_{max,ssq}.
    // Now let's work on the GPU arrays gout_{max,ssq}.

    Array<float> gpu_out_max({B,P,Mout,Tk_out}, af_gpu | af_zero | af_guard);
    Array<float> gpu_out_ssq({B,P,Mout,Tk_out}, af_gpu | af_zero | af_guard);
    Array<float> gpu_pstate({B,Mout,RW}, af_gpu | af_zero | af_guard);
    Array<float> gpu_wt = wt.to_gpu();   // shape (B,P,Min)
			    
    for (int ik = 0; ik < Nk; ik++) {
	cout << "    kernel " << ik << "/" << Nk << endl;
	
	// Slice (B,Min,Tin) -> (B,Min,Tk_in)
	Array<float> gpu_in = in.slice(2, ik * Tk_in, (ik+1) * Tk_in);
	gpu_in = gpu_in.to_gpu();

	uint Bx = (Mout+W-1) / W;
	dim3 nblocks = {Bx, uint(B), 1};
	
	k.full_kernel <<< nblocks, 32*W >>>
	    (gpu_out_max.data, gpu_out_ssq.data,
	     gpu_pstate.data, gpu_in.data,
	     gpu_wt.data, Mout, Tk_out);

	CUDA_PEEK("pf kernel launch");

	// Slice (B,P,Mout,Tout) -> (B,P,Mout,Tk_out)
	Array<float> host_out_max = out_max.slice(3, ik * Tk_out, (ik+1) * Tk_out);
	Array<float> host_out_ssq = out_ssq.slice(3, ik * Tk_out, (ik+1) * Tk_out);
	
	assert_arrays_equal(host_out_max, gpu_out_max, "host_max", "gpu_max", {"b","p","mout","tout"});
	assert_arrays_equal(host_out_ssq, gpu_out_ssq, "host_ssq", "gpu_ssq", {"b","p","mout","tout"});
    }
}


static void test_reduce_only_kernel(const pf_kernel &k, int B, int Mout, int Tout)
{
    int M = k.M;
    int E = k.E;
    int P = k.P;
    int W = k.W;
    int Dout = k.Dout;
    int Dcore = k.Dcore;
    int Dt = xdiv(Dout, Dcore);

    cout << "test_reduce_only_kernel: M=" << M << ", E=" << E << ", Dout=" << Dout
	 << ", Dcore=" << Dcore << ", W=" << k.W << ", B=" << B << ", Mout=" << Mout
	 << ", Tout=" << Tout << endl;
    
    xassert_divisible(32, Dcore);
    xassert_divisible(Tout, 32/Dcore);

    Array<float> out_max({B,P,Mout,Tout}, af_rhost | af_zero);
    Array<float> out_ssq({B,P,Mout,Tout}, af_rhost | af_zero);
    Array<float> in_max({B,P,Mout*M,Tout*Dt}, af_rhost | af_random);
    Array<float> in_ssq({B,P,Mout*M,Tout*Dt}, af_rhost | af_random);
    Array<float> wt({B,P,Mout*M}, af_rhost | af_random);

    for (int b = 0; b < B; b++) {
	for (int p = 0; p < P; p++) {
	    for (int mout = 0; mout < Mout; mout++) {
		for (int tout = 0; tout < Tout; tout++) {
		    float rmax = -1.0e20;
		    float rssq = 0.0;

		    for (int m = mout*M; m < (mout+1)*M; m++) {
			float w = wt.at({b,p,m});
			for (int t = tout*Dt; t < (tout+1)*Dt; t++) {
			    rmax = max(rmax, w * in_max.at({b,p,m,t}));
			    rssq += w * w * in_ssq.at({b,p,m,t});
			}
		    }

		    out_max.at({b,p,mout,tout}) = rmax;
		    out_ssq.at({b,p,mout,tout}) = rssq;
		}
	    }
	}
    }

    Array<float> gpu_out_max({B,P,Mout,Tout}, af_gpu | af_zero | af_guard);
    Array<float> gpu_out_ssq({B,P,Mout,Tout}, af_gpu | af_zero | af_guard);
    Array<float> gpu_in_max = in_max.to_gpu();
    Array<float> gpu_in_ssq = in_ssq.to_gpu();
    Array<float> gpu_wt = wt.to_gpu();

    uint Bx = (Mout+W-1) / W;
    dim3 nblocks = {Bx, uint(B), 1};
   
    k.reduce_only_kernel <<< nblocks, 32*W >>>
	(gpu_out_max.data, gpu_out_ssq.data,
	 gpu_in_max.data, gpu_in_ssq.data,
	 gpu_wt.data, Mout, Tout);

    CUDA_PEEK("pf reduce-only kernel launch");

    assert_arrays_equal(out_max, gpu_out_max, "host_max", "gpu_max", {"b","p","mout","tout"});
    assert_arrays_equal(out_ssq, gpu_out_ssq, "host_ssq", "gpu_ssq", {"b","p","mout","tout"});
}



void test_gpu_peak_finding_kernel()
{
    vector<pf_kernel> all_kernels = pf_kernel::enumerate();

    for (int i = 0; i < 5; i++) {
	pf_kernel k = ksgpu::rand_element(all_kernels);

	long T = 32 / k.Dcore;
	auto v = ksgpu::random_integers_with_bounded_product(5, 10000 / (k.M * T));
	
	int B = v[0];
	int Tout = v[1] * v[2] * T;
        int Mout = v[3] * v[4];
	int Nk = 5;

	// Debug
	// B = 1;
	// Tout = 32;
	// Mout = 1;
	
	test_reduce_only_kernel(k, B, Mout, Tout);
	test_full_pf_kernel(k, B, Mout, Tout, Nk);
    }
}


}  // namespace pirate
