#include "../include/pirate/cuda_kernels/peak_finding.hpp"
#include "../include/pirate/inlines.hpp"
#include "../include/pirate/utils.hpp"

#include <ksgpu/Array.hpp>
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
	pf_core_step<Core,J0+1> (core, in_th + ST, out_th + ST, ssq_th + ST);
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

    int sth = (threadIdx.x / ST) * (Souter * ST) + (threadIdx.x % ST);
    T32 *in_th = in + sth;
    T32 *out_th = out + sth;
    T32 *ssq_th = ssq + sth;
    
    for (int t = 0; t < nt; t += Core::Tout) {
	pf_core_step(core, in_th, out_th, ssq_th, out_pstride32);
	in_th += Core::Dt * Souter * ST;
	out_th += Souter * ST;
	ssq_th += Souter * ST;
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
//   Dt*W <= S <= min(4*Dt,16) * W

template<typename T32, int Dt, int E>
static pf_core_kernel_t get_pf_core_kernel3(int S)
{
    constexpr int W = ksgpu::dtype_ops::simd_width<T32> ();
    
    if (S == Dt*W)
	return pf_core_kernel<T32, Dt, E, Dt*W>;
#if 0
    if constexpr (Dt <= 8)
	if (S == 2*Dt*W)
	    return pf_core_kernel<T32, Dt, E, 2*Dt*W>;

    if constexpr (Dt <= 4)
	if (S == 4*Dt*W)
	    return pf_core_kernel<T32, Dt, E, 4*Dt*W>;
#endif
    throw runtime_error("get_pf_core_kernel(): invalid (Dt,S)");
}

template<typename T32, int Dt>
static pf_core_kernel_t get_pf_core_kernel2(int E, int S)
{
    if (E == 1)
	return get_pf_core_kernel3<T32,Dt,1> (S);
#if 0
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
#endif
    throw runtime_error("get_pf_core_kernel(): invalid (Dt,E)");
}

template<typename T32>
static pf_core_kernel_t get_pf_core_kernel1(int Dt, int E, int S)
{
    static_assert(sizeof(T32) == 4);

    if (Dt == 1)
	return get_pf_core_kernel2<T32,1> (E,S);
#if 0
    if (Dt == 2)
	return get_pf_core_kernel2<T32,2> (E,S);
    if (Dt == 4)
	return get_pf_core_kernel2<T32,4> (E,S);
    if (Dt == 8)
	return get_pf_core_kernel2<T32,8> (E,S);
    if (Dt == 16)
	return get_pf_core_kernel2<T32,16> (E,S);
#endif
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
    xpad.slice(0,Dt+1,Dt+1+S).fill(x);

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
    long P = 3*integer_log2(Dt) + 1;
    
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
    long W = 32 / dtype.nbits;             // simd width
    
    long lgE = 0; // rand_int(0,5);
    long lgD = 0; // rand_int(lgE,5);
    long lgSW = 0; // rand_int(lgD, min(lgD+3,5L));   // log2(S/W)
    
    long Dt = 1 << lgD;
    long E = 1 << lgE;
    long S = W << lgSW;
    long P = 3*lgE + 1;
    
    long Tk_in = 32*W * rand_int(1,10);  // input time samples per kernel
    long Tk_out = Tk_in / Dt;            // output time samples per kernel
    
    long Nk = rand_int(1,10);   // number of kernels    
    long Tin = Tk_in * Nk;      // total input time samples
    long Tout = Tk_out * Nk;    // total input time samples

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
	char *inp = reinterpret_cast<char *> (gx.data) + k * Tk_in * (dtype.nbits / 8);
	char *outp = reinterpret_cast<char *> (gout.data) + k * Tk_out * (dtype.nbits / 8);
	char *ssqp = reinterpret_cast<char *> (gssq.data) + k * Tk_out * (dtype.nbits / 8);
	
	kernel(inp, outp, ssqp, pstate, nt_out, out_pstride32, pstate_nbytes);
    }

    assert_arrays_equal(out, gout, "hout", "gout", {"p","t","s"});
    assert_arrays_equal(ssq, gssq, "hssq", "gssq", {"p","t","s"});
}


}  // namespace pirate
