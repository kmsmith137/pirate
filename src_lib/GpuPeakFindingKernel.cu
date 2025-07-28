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
    // Check that everything is initialized.
    xassert(dm_downsampling_factor > 0);
    xassert(time_downsampling_factor > 0);
    xassert(max_kernel_width > 0);
    xassert(beams_per_batch > 0);
    xassert(total_beams > 0);
    xassert(ndm_in > 0);
    xassert(nt_in > 0);

    // xassert(is_power_of_two(dm_downsampling_factor));
    xassert(is_power_of_two(time_downsampling_factor));
    xassert(is_power_of_two(max_kernel_width));
    xassert_divisible(total_beams, beams_per_batch);
    xassert_divisible(ndm_in, dm_downsampling_factor);
    xassert_divisible(nt_in, time_downsampling_factor);

    // Currently assumed in GPU kernel.
    // FIXME incomplete -- I think there are more assumptions in the GPU kernel.
    xassert(max_kernel_width <= 32);
    
    // FIXME float16 coming soon
    if (dtype != Dtype::native<float>())
	throw runtime_error("LaggedDownsamplingKernelParams: unsupported dtype: " + dtype.str());    
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


void PeakFindingKernel::_check_args(const Array<void> &out_max, const Array<void> &out_ssq, const Array<void> &in, const Array<void> &wt, Dtype expected_dtype)
{
    int B = params.beams_per_batch;
    int Min = params.ndm_in;
    int Tin = params.nt_in;    
    int Mout = this->ndm_out;
    int Tout = this->nt_out;
    int P = this->nprofiles;
    
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
    PeakFindingKernel(params_, Dcore_) { }


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


void ReferencePeakFindingKernel::apply(Array<void> &out_max_, Array<void> &out_ssq_, const Array<void> &in_, const Array<void> &wt_)
{
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

    _check_args(out_max, out_ssq, in, wt, Dtype::native<float>());
    
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
    
    this->cuda_kernel = pf_kernel::get(M, E, Dout);
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
    std::initializer_list<long> shape = { params.total_beams, ndm_out, cuda_kernel.RW };
    this->persistent_state = Array<void> (params.dtype, shape, af_zero | af_gpu);
    this->is_allocated = true;
}


void GpuPeakFindingKernel::launch(Array<void> &out_max, Array<void> &out_ssq, const Array<void> &in, const Array<void> &wt, long ibatch, cudaStream_t stream)
{
    xassert(this->is_allocated);
    xassert((ibatch >= 0) && (ibatch < nbatches));
    
    _check_args(out_max, out_ssq, in, wt, params.dtype);

    xassert(out_max.on_gpu());
    xassert(out_ssq.on_gpu());
    xassert(in.on_gpu());
    xassert(wt.on_gpu());

    int W = cuda_kernel.W;
    uint Bx = (ndm_out + W - 1) / W;
    dim3 nblocks = {Bx, uint(params.beams_per_batch), 1};
    
    char *pstate = (char *) persistent_state.data;
    pstate += ibatch * params.beams_per_batch * ndm_out * cuda_kernel.RW * xdiv(params.dtype.nbits,8);
    
    cuda_kernel.full_kernel <<< nblocks, 32*W, 0, stream >>>
	(out_max.data, out_ssq.data, pstate, in.data, wt.data, ndm_out, nt_out);

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
	 << ", Dcore=" << Dcore << ", W=" << W << ", P=" << P << ", RW=" << RW << endl;
#endif
    
    xassert(M > 0);
    xassert(E > 0);
    xassert(Dout > 0);
    xassert(Dcore > 0);
    xassert(W > 0);
    xassert(P > 0);
    xassert((E == 1) || (RW > 0));
    xassert(full_kernel != nullptr);
    xassert(reduce_only_kernel != nullptr);
    xassert(next == nullptr);
    
    unique_lock<mutex> lk(pf_kernel_lock);

    // Check whether this (M,E,Dout) triple has been registered before.
    for (pf_kernel *k = pf_kernel_registry; k != nullptr; k = k->next) {
	if ((k->M != M) || (k->E != E) || (k->Dout != Dout))
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
pf_kernel pf_kernel::get(int M, int E, int Dout)
{
    unique_lock<mutex> lk(pf_kernel_lock);
    
    for (pf_kernel *k = pf_kernel_registry; k != nullptr; k = k->next) {
	if ((k->M == M) && (k->E == E) && (k->Dout == Dout)) {
	    pf_kernel ret = *k;
	    ret.next = nullptr;
	    return ret;
	}
    }

    stringstream ss;
    ss << "pf_kernel::get(): no kernel found for (M,E,Dout)="
       << "(" << M << "," << E << "," << Dout << ")";
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
