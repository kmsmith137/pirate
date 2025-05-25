#include "../include/pirate/cuda_kernels/peak_finding.hpp"

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
__global__ void pf_ringbuf_test_kernel(int *data, int *pstate, int ninner, int data_stride)
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


// Function type for pf_ringbuf_test_kernel<I,O>
using rb_kernel_t = void (*)(int *, int *, int, int);


// Ugh
template<int I>
static rb_kernel_t get_rb_kernel1(int O)
{
    if (O == 1)
	return pf_ringbuf_test_kernel<I,1>;
    if (O == 2)
	return pf_ringbuf_test_kernel<I,2>;
    if (O == 3)
	return pf_ringbuf_test_kernel<I,3>;
    if (O == 4)
	return pf_ringbuf_test_kernel<I,4>;
    if (O == 5)
	return pf_ringbuf_test_kernel<I,5>;
    if (O == 6)
	return pf_ringbuf_test_kernel<I,6>;
    if (O == 7)
	return pf_ringbuf_test_kernel<I,7>;
    if (O == 8)
	return pf_ringbuf_test_kernel<I,8>;
    if (O == 9)
	return pf_ringbuf_test_kernel<I,9>;
    
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


#if 0

// Helper for pf_core_test_kernel().
//
// input array shape
//   = (Tin, S)
//   = (Tin, Souter * ST * SS)
//
// output array shapes
//   = (Tout, P, S)
//   = (Tout, P, Souter * ST * SS)

template<class Core, int J0=0>
__device__ inline void pf_core_step(Core &core, const T32 *in_th, T32 *out_th, T32 *ssq_th)
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
	    out_th[p * Sout*ST] = core.pf_out[p];
	    ssq_th[p * Sout*ST] = core.pf_ssq[p];
	}

	// Advance to next J0
	pf_core_step<Core,J0+1> (core, in_th + ST, out_th + ST, ssq_th + ST);
    }
}


// input array shape = (T,S)
// output array shape = ([Tc][P][S].
// pstate[S*(D)];
// Launch with one warp

template<typename T32, int Dt, int E, int S>
__global__ void pf_core_test_kernel(void *in_, void *out_, void *ssq_, void *pstate_, int nt, int pstate_nbytes)
{
    using Core = pf_core<T32, Dt, E, S>;
    Core core;
	
    T32 *in = reinterpret_cast<T32 *> (in_);
    T32 *out = reinterpret_cast<T32 *> (out_);
    T32 *ssq = reinterpret_cast<T32 *> (ssq_);
    T32 *pstate = reinterpret_cast<T32 *> (pstate_);

    assert(pstate_nbytes >= 4 * Core::pstate_n32);
    core.load_pstate(pstate);

    for (int t = ) {
	int x = 
	pf_core_step(core, in+tid, out+tid, ssq+tid);
    }
}


// Helper for ref_pf().
inline void _update_pf(float pf, int d, float &out, float &ssq)
{
    out = d ? max(out,pf) : pf;
    ssq = d ? (out+pf*pf) : (pf*pf);
}


// x.shape = (Tout*Dt,S)
// out.shape = ssq.shape = (3,Tout,S)
static void ref_pf(Array<float> &x, Array<float> &out, Array<float> &ssq, int Tout, int Dt, int S)
{
    const float a = constants::pf_a;
    const float b = constants::pf_b;

    xassert_shape_eq(x, {Tout*Dt,S});
    xassert_shape_eq(out, {3,Tout,S});
    xassert_shape_eq(ssq, {3,Tout,S});
    
    xassert(x.is_fully_contiguous());
    xassert(out.is_fully_contiguous());
    xassert(ssq.is_fully_contiguous());

    // prepad by (Dt+1).
    Array<float> xpad({Tin+Dt+1,S}, af_zero | af_rhost);
    xpad.slice(0,Dt+1,Dt+1+S).fill(x);

    for (int tout = 0; tout = Tout; tout++) {
	for (int s = 0; s < S; s++) {
	    // xp = array of shape (Dt+3,S).
	    // outp = ssqp = array of shape (3,1,S).
	    float *xp = x.data + (tout*Dt) * S;
	    float *outp = out.data + tout*S;
	    float *ssqp = ssq.data + tout*S;

	    // Stride of length-3 axis in 'outp' and 'ssqp' arrays
	    int os = Tout*S;

	    for (int d = 0; d < Dt; d++) {
		for (int s = 0; s < S; s++) {
		    float x0 = xp[d*S + s];
		    float x1 = xp[(d+1)*S + s];
		    float x2 = xp[(d+2)*S + s];
		    float x3 = xp[(d+3)*S + s];
		    
		    float b2 = x0 + x1;
		    float g3 = a*x0 + x1 + a*x2;
		    float g4 = b*x0 + x1 + x2 + b*x3;

		    _update_pf(b2, d, outp[s], ssqp[s]);
		    _update_pf(g3, d, outp[s+os], ssqp[s+os]);
		    _update_pf(g4, d, outp[s+2*os], ssqp[s+2*os]);
		}
	    }
	}
    }
}


static void ref_core(Array<float> &x, Array<float> &out, Array<float> &ssq, int Tout, int E, int Dt, int S)
{
    // Iteratively downsample and call ref_pf.
}


void test_gpu_pf_core()
{
    int lgE = rand_int(0,5);
    int E = 1 << lgE;
    int Dt = 1 << rand_int(lgE,5);
    int S = Dt * rand_int(1,4);     // note: will need to mulitply by 2 for float16
    int Tk = 32 * rand_int(1,10);   // input time samples per kernel (note: will need to multiply by 2 for float16)
    int Nk = rand_int(1,10);        // number of kernels
    int Tin = Tk * Nk;              // total input time samples

    Array<float> x({Tin,S}, af_random | af_rhost);
    Array<float> out({P,Tout,S}, af_random | af_rhost);
    Array<float> ssq({P,Tout,S}, af_random | af_rhost);
}

#endif


}  // namespace pirate
