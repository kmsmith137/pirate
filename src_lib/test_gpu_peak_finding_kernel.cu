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


static void test_reduce_only_kernel(const PeakFindingKernelParams &params)
{
    cout << "\ntest_reduce_only_kernel: start\n"
	 << "   dtype = " << params.dtype.str() << "\n"
	 << "   M = dm_downsampling_factor = " << params.dm_downsampling_factor << "\n"
	 << "   E = max_kernel_width = " << params.max_kernel_width << "\n"
	 << "   Dout = time_downsampling_factor = " << params.time_downsampling_factor << "\n"
	 << "   beams_per_batch = " << params.beams_per_batch << "\n"
	 << "   ndm_in = " << params.ndm_in << "\n"
	 << "   nt_in = " << params.nt_in << "\n";

    GpuPeakFindingKernel gpu_kernel(params);
    pf_kernel k = gpu_kernel.cuda_kernel;
    
    int M = k.M;
    int W = k.W;    
    int P = k.P;
    int B = params.beams_per_batch;  // note: params.total_beams is not used in this test
    int Dout = k.Dout;
    int Dcore = k.Dcore;
    int Dt = xdiv(Dout, Dcore);
    int Min = params.ndm_in;
    int Tin = params.nt_in;
    int Mout = xdiv(Min, k.M);
    int Tout = xdiv(Tin, k.Dout);
    int Tout32 = xdiv(Tout * k.dtype.nbits, 32);
    
    xassert_divisible(32, Dcore);

    Array<float> out_max({B,P,Mout,Tout}, af_rhost | af_zero);
    Array<float> out_ssq({B,P,Mout,Tout}, af_rhost | af_zero);
    Array<float> in_max({B,P,Mout*M,Tout*Dt}, af_rhost | af_random);
    Array<float> in_ssq({B,P,Mout*M,Tout*Dt}, af_rhost | af_random);
    Array<float> wt({B,P,Mout*M}, af_rhost | af_zero);

    // Weights must be positive.
    for (long i = 0; i < wt.size; i++)
	wt.data[i] = rand_uniform(0.5, 1.0);

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

    Array<void> gpu_out_max(k.dtype, {B,P,Mout,Tout}, af_gpu | af_zero | af_guard);
    Array<void> gpu_out_ssq(k.dtype, {B,P,Mout,Tout}, af_gpu | af_zero | af_guard);
    Array<void> gpu_in_max = in_max.convert(k.dtype).to_gpu();
    Array<void> gpu_in_ssq = in_ssq.convert(k.dtype).to_gpu();
    Array<void> gpu_wt = wt.convert(k.dtype).to_gpu();

    uint Bx = (Mout+W-1) / W;
    dim3 nblocks = {Bx, uint(B), 1};
   
    k.reduce_only_kernel <<< nblocks, 32*W >>>
	(gpu_out_max.data, gpu_out_ssq.data,
	 gpu_in_max.data, gpu_in_ssq.data,
	 gpu_wt.data, Mout, Tout32);

    CUDA_PEEK("pf reduce-only kernel launch");

    int nds = M * xdiv(Dout,Dcore);  // total amount of downsampling in reduce-only kernel
    double epsabs = 5.0 * sqrt(nds) * k.dtype.precision();  // appropriate epsabs for ssq comparison
    
    assert_arrays_equal(out_max, gpu_out_max, "host_max", "gpu_max", {"b","p","mout","tout"});
    assert_arrays_equal(out_ssq, gpu_out_ssq, "host_ssq", "gpu_ssq", {"b","p","mout","tout"}, epsabs);
}


// -------------------------------------------------------------------------------------------------


static void test_pf_kernel(const PeakFindingKernelParams &params, long niter_gpu, long niter_cpu)
{
    cout << "\ntest_pf_kernel: start\n"
	 << "   dtype = " << params.dtype.str() << "\n"
	 << "   M = dm_downsampling_factor = " << params.dm_downsampling_factor << "\n"
	 << "   E = max_kernel_width = " << params.max_kernel_width << "\n"
	 << "   Dout = time_downsampling_factor = " << params.time_downsampling_factor << "\n"
	 << "   beams_per_batch = " << params.beams_per_batch << "\n"
	 << "   total_beams = " << params.total_beams << "\n"
	 << "   ndm_in = " << params.ndm_in << "\n"
	 << "   nt_in = " << params.nt_in << "\n"
	 << "   niter_gpu = " << niter_gpu << "\n"
	 << "   niter_cpu = " << niter_cpu
	 << endl;

    params.validate();
    
    PeakFindingKernelParams gpu_params = params;
    gpu_params.nt_in = xdiv(params.nt_in, niter_gpu);

    GpuPeakFindingKernel gpu_kernel(gpu_params);
    gpu_kernel.allocate();

    PeakFindingKernelParams ref_params = params;
    ref_params.nt_in = xdiv(params.nt_in, niter_cpu);
	
    ReferencePeakFindingKernel ref_kernel(ref_params, gpu_kernel.Dcore);

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
    Array<void> gpu_max(params.dtype, {B,P,Mout,Tout}, af_rhost | af_zero);
    Array<void> gpu_ssq(params.dtype, {B,P,Mout,Tout}, af_rhost | af_zero);

    // Weights must be positive.
    for (long i = 0; i < wt.size; i++)
	wt.data[i] = rand_uniform(0.5, 1.0);
    
    // Reference kernel.
    
    long Tin_c = xdiv(Tin, niter_cpu);
    long Tout_c = xdiv(Tout, niter_cpu);
    
    Array<float> in_c({B,Min,Tin_c}, af_uhost);
    Array<float> max_c({B,P,Mout,Tout_c}, af_uhost);
    Array<float> ssq_c({B,P,Mout,Tout_c}, af_uhost);
    
    for (long i = 0; i < niter_cpu; i++) {
	long it0 = (i) * Tin_c;
	long it1 = (i+1) * Tin_c;

	Array<float> in_tmp = in.slice(2,it0,it1);
	in_c.fill(in_tmp);
	
	for (long b = 0; b < gpu_kernel.nbatches; b++) {
	    long ib0 = (b) * params.beams_per_batch;
	    long ib1 = (b+1) * params.beams_per_batch;

	    ref_kernel.apply(
	        max_c.slice(0,ib0,ib1),
		ssq_c.slice(0,ib0,ib1),
		in_c.slice(0,ib0,ib1),
		wt.slice(0,ib0,ib1),
		b
	   );
	}

	it0 = (i) * Tout_c;
	it1 = (i+1) * Tout_c;
	host_max.slice(3,it0,it1).fill(max_c);
	host_ssq.slice(3,it0,it1).fill(ssq_c);
    }

    // GPU kernel.

    long Tin_g = gpu_kernel.params.nt_in;
    long Tout_g = gpu_kernel.nt_out;
    
    Array<void> wt_g = wt.convert(params.dtype).to_gpu();
    Array<void> in_g(params.dtype, {B,Min,Tin_g}, af_gpu);
    Array<void> max_g(params.dtype, {B,P,Mout,Tout_g}, af_gpu);
    Array<void> ssq_g(params.dtype, {B,P,Mout,Tout_g}, af_gpu);
    
    for (long i = 0; i < niter_gpu; i++) {
	long it0 = (i) * Tin_g;
	long it1 = (i+1) * Tin_g;

	Array<void> in_tmp = in.slice(2,it0,it1).convert(params.dtype);
	in_g.fill(in_tmp);
	
	for (long b = 0; b < gpu_kernel.nbatches; b++) {
	    long ib0 = (b) * params.beams_per_batch;
	    long ib1 = (b+1) * params.beams_per_batch;

	    Array<void> max_tmp = max_g.slice(0,ib0, ib1);
	    Array<void> ssq_tmp = ssq_g.slice(0,ib0, ib1);
	    
	    gpu_kernel.launch(
		max_tmp,
		ssq_tmp,
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
    
    int nds = params.dm_downsampling_factor * params.time_downsampling_factor;
    double epsabs = 5.0 * sqrt(nds) * params.dtype.precision();  // appropriate epsabs for ssq comparison
	
    assert_arrays_equal(host_max, gpu_max, "host_max", "gpu_max", {"b","p","mout","tout"});
    assert_arrays_equal(host_ssq, gpu_ssq, "host_ssq", "gpu_ssq", {"b","p","mout","tout"}, epsabs);
    cout << endl;
}


void test_gpu_peak_finding_kernel(bool reduce_only)
{
    vector<pf_kernel> all_kernels = pf_kernel::enumerate();

    // We include this extra factor of 5, to guarantee that 'python -m pirate_frb test'
    // runs every kernel a few times. Currently there are 66 precompiled kernels, and
    // test_gpu_peak_finding_kernel() gets called 100 times.
    
    for (int i = 0; i < 5; i++) {
	// Choose a random precompiled kernel.
	pf_kernel k = ksgpu::rand_element(all_kernels);
	int nbits = k.dtype.nbits;
	
	// Note that by choosing 'niter_cpu' and 'niter_gpu' independently, this test
	// checks the incremental logic in both the reference kernel and the GPU kernel.
	
	int niter_cpu = rand_int(1,4);
	int niter_gpu = rand_int(1,4);

	// Determine smallest values of 'ndm_in' and 'nt_in' which are compatible with
	// divisibility constraints in PeakFindingParams::validate(). If these constraints
	// are changed, then logic here should be updated as well.

	int nn = std::lcm(niter_cpu, niter_gpu);
	int ndm_in_divisor = std::lcm(k.M, xdiv(32,nbits));
	int nt_in_divisor = nn * std::lcm(xdiv(1024,nbits), k.Dout * xdiv(32,nbits));
	
	auto v = ksgpu::random_integers_with_bounded_product(6, 1000000 / (ndm_in_divisor * nt_in_divisor));

	PeakFindingKernelParams params;
	params.dtype = k.dtype;
	params.time_downsampling_factor = k.Dout;
	params.dm_downsampling_factor = k.M;
	params.max_kernel_width = k.E;
	params.beams_per_batch = v[0];
	params.total_beams = v[0] * v[1];
	params.ndm_in = v[2] * v[3] * ndm_in_divisor;
	params.nt_in = v[4] * v[5] * nt_in_divisor;

	// Debug
	// niter_gpu = niter_cpu = 1;
	// params.beams_per_batch = params.total_beams = 1;
	// params.ndm_in = 2 * k.M;
	// params.nt_in = 64 * std::lcm(niter_gpu, niter_cpu);
	    
	params.validate();

	if (reduce_only)
	    test_reduce_only_kernel(params);
	else
	    test_pf_kernel(params, niter_gpu, niter_cpu);
    }
}


}  // namespace pirate
