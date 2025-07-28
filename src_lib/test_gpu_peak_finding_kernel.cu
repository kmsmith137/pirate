#include "../include/pirate/PeakFindingKernel.hpp"
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


// -------------------------------------------------------------------------------------------------


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


// -------------------------------------------------------------------------------------------------


static void test_pf_kernel2(const PeakFindingKernelParams &params, long niter_gpu)
{
    cout << "\ntest_pf_kernel2: start\n"
	 << "   M = dm_downsampling_factor = " << params.dm_downsampling_factor << "\n"
	 << "   E = max_kernel_width = " << params.max_kernel_width << "\n"
	 << "   Dout = time_downsampling_factor = " << params.time_downsampling_factor << "\n"
	 << "   beams_per_batch = " << params.beams_per_batch << "\n"
	 << "   total_beams = " << params.total_beams << "\n"
	 << "   ndm_in = " << params.ndm_in << "\n"
	 << "   nt_in = " << params.nt_in << "\n"
	 << "   niter_gpu = " << niter_gpu
	 << endl;

    params.validate();
    
    PeakFindingKernelParams gpu_params = params;
    gpu_params.nt_in = xdiv(params.nt_in, niter_gpu);

    GpuPeakFindingKernel gpu_kernel(gpu_params);
    gpu_kernel.allocate();
    
    ReferencePeakFindingKernel ref_kernel(params, gpu_kernel.Dcore);

    // Allocate arrays.
    
    long B = params.total_beams;
    long P = gpu_kernel.nprofiles;
    long Min = params.ndm_in;
    long Tin = params.nt_in;
    long Mout = xdiv(Min, params.dm_downsampling_factor);
    long Tout = xdiv(Tin, params.time_downsampling_factor);
    
    Array<float> wt({B,P,Min}, af_rhost | af_zero);
    Array<float> in({B,Min,Tin}, af_rhost | af_random);
    Array<float> host_max({B,P,Mout,Tout}, af_rhost | af_zero);
    Array<float> host_ssq({B,P,Mout,Tout}, af_rhost | af_zero);
    Array<float> gpu_max({B,P,Mout,Tout}, af_rhost | af_zero);
    Array<float> gpu_ssq({B,P,Mout,Tout}, af_rhost | af_zero);

    // Weights must be positive.
    for (long i = 0; i < wt.size; i++)
	wt.data[i] = rand_uniform(1.0, 2.0);
    
    // Reference kernel.

    for (long b = 0; b < ref_kernel.nbatches; b++) {
	long ib0 = (b) * params.beams_per_batch;
	long ib1 = (b+1) * params.beams_per_batch;

	ref_kernel.apply(
	    host_max.slice(0,ib0,ib1),
	    host_ssq.slice(0,ib0,ib1),
	    in.slice(0,ib0,ib1),
	    wt.slice(0,ib0,ib1)
	);
    }

    // GPU kernel.

    long Tin_g = gpu_kernel.params.nt_in;
    long Tout_g = gpu_kernel.nt_out;
    
    Array<float> wt_g = wt.to_gpu();
    Array<float> in_g({B,Min,Tin_g}, af_gpu);
    Array<float> max_g({B,P,Mout,Tout_g}, af_gpu);
    Array<float> ssq_g({B,P,Mout,Tout_g}, af_gpu);
    
    for (long i = 0; i < niter_gpu; i++) {
	long it0 = (i) * Tin_g;
	long it1 = (i+1) * Tin_g;

	in_g.fill(in.slice(2,it0,it1));
		  
	for (long b = 0; b < gpu_kernel.nbatches; b++) {
	    long ib0 = (b) * params.beams_per_batch;
	    long ib1 = (b+1) * params.beams_per_batch;

	    gpu_kernel.launch(
	        max_g.slice(0,ib0,ib1),
	        ssq_g.slice(0,ib0,ib1),
	        in_g.slice(0,ib0,ib1),
	        wt_g.slice(0,ib0,ib1),
		b, NULL
	    );
	}

	it0 = (i) * Tout_g;
	it1 = (i+1) * Tout_g;
	gpu_max.slice(3,it0,it1).fill(max_g);
	gpu_ssq.slice(3,it0,it1).fill(ssq_g);
    }

    // Compare results.
    
    assert_arrays_equal(host_max, gpu_max, "host_max", "gpu_max", {"b","p","mout","tout"});
    assert_arrays_equal(host_ssq, gpu_ssq, "host_ssq", "gpu_ssq", {"b","p","mout","tout"});

    cout << "test_pf_kernel2: pass" << endl;
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

	PeakFindingKernelParams params;
	params.dtype = Dtype::native<float> ();
	params.dm_downsampling_factor = k.M;
	params.time_downsampling_factor = k.Dout;
	params.max_kernel_width = k.E;
	params.beams_per_batch = B;
	params.total_beams = B;
	params.ndm_in = Mout * k.M;
	params.nt_in = Nk * k.Dout * Tout;

	test_pf_kernel2(params, Nk);
    }
}


}  // namespace pirate
