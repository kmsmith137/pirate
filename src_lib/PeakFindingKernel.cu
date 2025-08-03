#include "../include/pirate/PeakFindingKernel.hpp"
#include "../include/pirate/inlines.hpp"
#include "../include/pirate/utils.hpp"

#include <mutex>
#include <ksgpu/Array.hpp>
#include <ksgpu/cuda_utils.hpp>

using namespace std;
using namespace ksgpu;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// -------------------------------------------------------------------------------------------------


void PeakFindingKernelParams::validate() const
{
    if ((dtype != Dtype::native<float>()) && (dtype != Dtype::native<__half>()))
	throw runtime_error("LaggedDownsamplingKernelParams: unsupported dtype: " + dtype.str());
    
    // Check that everything is initialized.
    xassert(dm_downsampling_factor > 0);
    xassert(time_downsampling_factor > 0);
    xassert(max_kernel_width > 0);
    xassert(beams_per_batch > 0);
    xassert(total_beams > 0);
    xassert(ndm_in > 0);
    xassert(nt_in > 0);

    xassert(is_power_of_two(max_kernel_width));
    xassert(is_power_of_two(time_downsampling_factor));
    xassert_divisible(total_beams, beams_per_batch);
    xassert_divisible(ndm_in, dm_downsampling_factor);
    xassert_divisible(nt_in, time_downsampling_factor);

    long ndm_out = xdiv(ndm_in, dm_downsampling_factor);
    long nt_out = xdiv(nt_in, time_downsampling_factor);
    long nbits = dtype.nbits;
    
    // Currently assumed in GPU kernel.
    // FIXME add these asserts to the code generator (+ an assert on DCore)
    xassert(max_kernel_width <= 32);
    xassert(time_downsampling_factor <= 32);
    xassert_divisible(nt_in, xdiv(1024,nbits));
    xassert_divisible(nt_out, xdiv(32,nbits));   // tout is fastest axis in out_{max,ssq}
    xassert_divisible(ndm_in, xdiv(32,nbits));   // dm_in is fastest axis in 'wt'
}

PeakFindingKernel::PeakFindingKernel(const PeakFindingKernelParams &params_, long Dcore_) :
    params(params_), Dcore(Dcore_)
{
    params.validate();

    if (Dcore == 0)
	Dcore = params.time_downsampling_factor;

    xassert_divisible(params.time_downsampling_factor, Dcore);

    this->nbatches = xdiv(params.total_beams, params.beams_per_batch);
    this->ndm_out = xdiv(params.ndm_in, params.dm_downsampling_factor);
    this->nt_out = xdiv(params.nt_in, params.time_downsampling_factor);
    this->nprofiles = 3 * integer_log2(params.max_kernel_width) + 1;
}


void PeakFindingKernel::_check_args(const Array<void> &out_max, const Array<void> &out_ssq, const Array<void> &in, const Array<void> &wt, Dtype expected_dtype, long ibatch)
{
    int B = params.beams_per_batch;
    int Min = params.ndm_in;
    int Tin = params.nt_in;    
    int Mout = this->ndm_out;
    int Tout = this->nt_out;
    int P = this->nprofiles;
    
    xassert((ibatch >= 0) && (ibatch < this->nbatches));
    
    xassert_shape_eq(out_max, ({B,P,Mout,Tout}));
    xassert_shape_eq(out_ssq, ({B,P,Mout,Tout}));
    xassert_shape_eq(in, ({B,Min,Tin}));
    xassert_shape_eq(wt, ({B,P,Min}));

    xassert(out_max.is_fully_contiguous());
    xassert(out_ssq.is_fully_contiguous());
    xassert(in.is_fully_contiguous());
    xassert(wt.is_fully_contiguous());

    xassert(out_max.dtype == expected_dtype);
    xassert(out_ssq.dtype == expected_dtype);
    xassert(in.dtype == expected_dtype);
    xassert(wt.dtype == expected_dtype);
}


// -------------------------------------------------------------------------------------------------
//
// ReferencePeakFindingKernel


ReferencePeakFindingKernel::ReferencePeakFindingKernel(const PeakFindingKernelParams &params_, long Dcore_) :
    PeakFindingKernel(params_, Dcore_)  // base class constructor calls params.validate()
{
    long E = params.max_kernel_width;
    long W = (E > 1) ? (2*E) : 1;           // width of widest kernel
    long D = (E > 1) ? min(Dcore,E/2) : 1;  // downsampling factor of widest kernel
    
    this->pstate_nt = (W - D);
    this->pstate = Array<float> ({params.total_beams, params.ndm_in, pstate_nt}, af_uhost | af_zero);
}


// Tiny helper for ReferencePeakFindingKernel::apply().
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


void ReferencePeakFindingKernel::apply(Array<void> &out_max_, Array<void> &out_ssq_, const Array<void> &in_, const Array<void> &wt_, long ibatch)
{
    constexpr float one_over_sqrt2 = 0.7071067811865476f;
    
    int B = params.beams_per_batch;
    int M = params.dm_downsampling_factor;
    int E = params.max_kernel_width;
    int Dout = params.time_downsampling_factor;
    int Min = params.ndm_in;
    int Tin = params.nt_in;    
    int Mout = this->ndm_out;
    int Tout = this->nt_out;

    // The reference kernel uses float32, regardless of what dtype is specified.
    Array<float> out_max = out_max_.template cast<float> ("ReferencePeakFindingKernel::apply(): 'out_max' array");
    Array<float> out_ssq = out_ssq_.template cast<float> ("ReferencePeakFindingKernel::apply(): 'out_ssq' array");
    Array<float> in = in_.template cast<float> ("ReferencePeakFindingKernel::apply(): 'in' array");
    Array<float> wt = wt_.template cast<float> ("ReferencePeakFindingKernel::apply(): 'wt' array");

    _check_args(out_max, out_ssq, in, wt, Dtype::native<float>(), ibatch);
    
    xassert(out_max.on_host());
    xassert(out_ssq.on_host());
    xassert(in.on_host());
    xassert(wt.on_host());
    
    // This loop just does p=0.
    for (int b = 0; b < B; b++) {
	for (int mout = 0; mout < Mout; mout++) {
	    for (int tout = 0; tout < Tout; tout++) {
		pf_accumulator acc0;

		for (int m = mout*M; m < (mout+1)*M; m++) {
		    float w = wt.at({b,0,m});   // p=0
		    
		    for (int t = tout*Dout; t < (tout+1)*Dout; t++) {
			float x = in.at({b,m,t});  // decided to use 'in' here, not 'in_x'.
			acc0.update(w, x);
		    }

		    out_max.at({b,0,mout,tout}) = acc0.rmax;  // p=0
		    out_ssq.at({b,0,mout,tout}) = acc0.rssq;  // p=0
		}
	    }
	}
    }

    if (E == 1)
	return;
    
    // Throughout this kernel, 'in_x' denotes a version of 'in' which has been:
    //   - downsampled by 2**ids elements
    //   - sampled at sampling rate 2**isamp
    //   - prepadded by 'Tpp' elements
    //
    // To write this out explicitly, 'in_x' has shape (B, Min, Tin/2**isamp + Tpp).
    // Given innermost index 0 <= ix < (Tin/2**isamp + Tpp), we write tx = (ix-Tpp).
    // Array elements at this index are obtained by summing 'in' over:
    //
    //   ((tx+1) * 2**isamp) - 2**ids <= tin < ((tx+1) * 2**isamp)
    //
    // where negative values of 'tin' are saved in ReferencePeakFindingKernel::pstate.
    
    int ids = 0;
    int isamp = 0;
    int Tpp = pstate_nt;

    Array<float> ps = pstate.slice(0, ibatch*B, (ibatch+1)*B);
    Array<float> in_x({B,Min,Tin+Tpp}, af_uhost);

    in_x.slice(2,Tpp,Tin+Tpp).fill(in);
    in_x.slice(2,0,Tpp).fill(ps);
    ps.fill(in_x.slice(2,Tin,Tin+Tpp));   // update pstate, for next call to apply().

    for (;;) {
	
	// The value of 'ids' increases by 1 in every iteration of the loop,
	// but the value of isamp "saturates" at log2(Dcore).
	//
	// The output arrays have been filled for 0 <= p < (3*ids+1).
	// In this iteration of the loop, we'll fill (3*ids+1) <= p < (3*ids+4).

	long Tds = xdiv(Tin, pow2(isamp));
	xassert_shape_eq(in_x, ({B,Min,Tds+Tpp}));
	xassert_eq(isamp, min(ids, integer_log2(Dcore)));
	xassert(in_x.is_fully_contiguous());  // assumed in downsampling logic

	// For computing profiles.
	long DD = xdiv(Tds, Tout);  // Downsampling factor between 'in_ds' and 'out_{max,ssq}' arrays.
	int p0 = 3*ids+1;           // Base profile index

	// Time offset in 'in_ds' array corresponding to 2**ids samples (used for "prepadding" below)
	long dt = pow2(ids-isamp);
	xassert(Tpp >= 3*dt);
 
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
			float *p = &in_x.at({b,m,0});
			
			for (int t = tout*DD; t < (tout+1)*DD; t++) {
			    float x0 = p[t+Tpp-3*dt];
			    float x1 = p[t+Tpp-2*dt];
			    float x2 = p[t+Tpp-dt];
			    float x3 = p[t+Tpp];

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

	if (E <= 2*pow2(ids))
	    return;

	// Downsample, for next iteration of the loop.

	// Define constants (s,t0,dt) so that downsampled array at time t
	// is obtained from input array at times (s*t + t0), (s*t + t0 + dt).
	// Note: t0 is a "logical" time index, that does not incorporate prepadding.
	
	int s = (isamp < integer_log2(Dcore)) ? 2 : 1;
	// int dt = pow2(ids - isamp);   // already defined above
	int t0 = (s-1) - dt;
	xassert(t0 <= 0);
	xassert(t0 >= -Tpp);

	long Tds2 = xdiv(Tds,s);
	long Tpp2 = (Tpp+t0)/s;  // round down
	xassert(Tpp2 >= 0);

	Array<float> in_x2({B,Min,Tds2+Tpp2}, af_uhost);
	
	// t1 = version of t0 which accounts for prepadding
	int t1 = (t0 - Tpp2*s) + Tpp;
	int tlast = s*(Tds2+Tpp2-1) + t1 + dt;  // only used for range check
	
	// Range check for loop below.
	xassert(t1 >= 0);
	xassert(tlast < Tds+Tpp);
	
	for (long b = 0; b < B; b++) {
	    for (long m = 0; m < Min; m++) {
		float *src = &in_x.at({b,m,0});
		float *dst = &in_x2.at({b,m,0});

		for (long i = 0; i < (Tds2+Tpp2); i++)
		    dst[i] = one_over_sqrt2 * (src[s*i+t1] + src[s*i+t1+dt]);
	    }
	}

	in_x = in_x2;
	Tpp = Tpp2;
	isamp += integer_log2(s);
	ids++;
    }
}


// -------------------------------------------------------------------------------------------------
//
// GpuPeakFindingKernel


GpuPeakFindingKernel::GpuPeakFindingKernel(const PeakFindingKernelParams &params_) :
    PeakFindingKernel(params_, 0)
{
    long M = params.dm_downsampling_factor;
    long E = params.max_kernel_width;
    long Dout = params.time_downsampling_factor;
    
    this->cuda_kernel = pf_kernel::get(params.dtype, M, E, Dout);
    this->Dcore = cuda_kernel.Dcore;
    
    xassert(cuda_kernel.P == nprofiles);

    // Number of elements (not bytes) per beam
    long n = 2 * nprofiles * ndm_out * nt_out;   // (out_max, out_ssq)
    n += params.ndm_in * params.nt_in;           // in
    n += nprofiles * params.ndm_in;              // wt
    
    this->bw_per_launch.nbytes_gmem = params.beams_per_batch * n * xdiv(params.dtype.nbits,8);
    this->bw_per_launch.kernel_launches = 1;
}


void GpuPeakFindingKernel::allocate()
{
    if (is_allocated)
	throw runtime_error("GpuPeakFindingKernel: double call to allocate()");

    // Note 'af_zero' flag here.
    long ninner = cuda_kernel.P32 * xdiv(32, params.dtype.nbits);
    std::initializer_list<long> shape = { params.total_beams, ndm_out, ninner };
    this->persistent_state = Array<void> (params.dtype, shape, af_zero | af_gpu);
    this->is_allocated = true;
}


void GpuPeakFindingKernel::launch(Array<void> &out_max, Array<void> &out_ssq, const Array<void> &in, const Array<void> &wt, long ibatch, cudaStream_t stream)
{
    _check_args(out_max, out_ssq, in, wt, params.dtype, ibatch);

    xassert(this->is_allocated);    
    xassert(out_max.on_gpu());
    xassert(out_ssq.on_gpu());
    xassert(in.on_gpu());
    xassert(wt.on_gpu());

    int W = cuda_kernel.W;
    uint Bx = (ndm_out + W - 1) / W;
    dim3 nblocks = {Bx, uint(params.beams_per_batch), 1};
	
    char *pstate = (char *) persistent_state.data;
    pstate += ibatch * params.beams_per_batch * ndm_out * cuda_kernel.P32 * 4;

    long Tout32 = xdiv(nt_out * params.dtype.nbits, 32);

    cuda_kernel.full_kernel <<< nblocks, 32*W, 0, stream >>>
	(out_max.data, out_ssq.data, pstate, in.data, wt.data, ndm_out, Tout32);

    CUDA_PEEK("pf kernel launch");    
}


// -------------------------------------------------------------------------------------------------
//
// Kernel registry


static mutex pf_kernel_lock;
pf_kernel *pf_kernel_registry = nullptr;


void pf_kernel::register_kernel()
{
    // Just check that all members have been initialized.
    // (In the future, I may add more argument checking here.)
#if 0
    cout << "register_pf_kernel: M=" << M << ", E=" << E << ", Dout=" << Dout
	 << ", Dcore=" << Dcore << ", W=" << W << ", P=" << P << ", P32=" << P32 << endl;
#endif
    
    xassert(M > 0);
    xassert(E > 0);
    xassert(Dout > 0);
    xassert(Dcore > 0);
    xassert(W > 0);
    xassert(P > 0);
    xassert((E == 1) || (P32 > 0));
    xassert(full_kernel != nullptr);
    xassert(reduce_only_kernel != nullptr);
    xassert(next == nullptr);
    
    unique_lock<mutex> lk(pf_kernel_lock);

    // Check whether this (M,E,Dout) triple has been registered before.
    for (pf_kernel *k = pf_kernel_registry; k != nullptr; k = k->next) {
	if ((k->dtype != dtype) || (k->M != M) || (k->E != E) || (k->Dout != Dout))
	    continue;

	if (k->debug && !this->debug) {
	    cout << "Note: debug and non-debug pf kernels were registered; debug kernel takes priority" << endl;
	    return;
	}

	if (!k->debug && this->debug) {
	    cout << "Note: debug and non-debug pf kernels were registered; debug kernel takes priority" << endl;
	    pf_kernel *save = k->next;
	    *k = *this;
	    k->next = save;
	    return;
	}

	stringstream ss;
	ss << "pf_kernel::register() called twice with (M,E,Dout)=" << "(" << M << "," << E << "," << Dout << ")";
	throw runtime_error(ss.str());
    }
    
    // Memory leak is okay for registry.
    pf_kernel *pf_copy = new pf_kernel(*this);
    pf_copy->next = pf_kernel_registry;  // assign with lock held
    pf_kernel_registry = pf_copy;
}


// Static member function
pf_kernel pf_kernel::get(Dtype dtype, int M, int E, int Dout)
{
    unique_lock<mutex> lk(pf_kernel_lock);
    
    for (pf_kernel *k = pf_kernel_registry; k != nullptr; k = k->next) {
	if ((k->dtype == dtype) && (k->M == M) && (k->E == E) && (k->Dout == Dout)) {
	    pf_kernel ret = *k;
	    ret.next = nullptr;
	    return ret;
	}
    }

    stringstream ss;
    ss << "pf_kernel::get(): no kernel found for dtype=" << dtype
       << " and (M,E,Dout)=(" << M << "," << E << "," << Dout << ")";
    
    throw runtime_error(ss.str());
}


// Static member function
vector<pf_kernel> pf_kernel::enumerate()
{
    vector<pf_kernel> ret;
    unique_lock<mutex> lk(pf_kernel_lock);

    for (pf_kernel *k = pf_kernel_registry; k != nullptr; k = k->next)
	ret.push_back(*k);

    lk.unlock();

    for (pf_kernel &k: ret)
	k.next = nullptr;

    return ret;
}


}  // namespace pirate
