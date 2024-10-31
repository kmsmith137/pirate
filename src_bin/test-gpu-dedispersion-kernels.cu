#include "../include/pirate/internals/GpuDedispersionKernel.hpp"
#include "../include/pirate/internals/ReferenceDedispersionKernel.hpp"
#include "../include/pirate/internals/inlines.hpp"    // pow2()
#include "../include/pirate/internals/utils.hpp"      // integer_log2()
#include "../include/pirate/constants.hpp"            // constants::bytes_per_gpu_cache_line

#include <gputils/Array.hpp>
#include <gputils/cuda_utils.hpp>
#include <gputils/rand_utils.hpp>    // rand_int()
#include <gputils/test_utils.hpp>    // assert_arrays_equal()

using namespace std;
using namespace pirate;
using namespace gputils;


struct TestInstance
{
    GpuDedispersionKernel::Params params;

    bool in_place = true;
    long nchunks = 0;

    // Strides for input/output arrays.
    // Reminder: arrays have shape (params.beams_per_kernel_launch, params.nambient, pow2(params.rank), params.ntime).
    vector<ssize_t> gpu_istrides;
    vector<ssize_t> gpu_ostrides;
    vector<ssize_t> cpu_istrides;
    vector<ssize_t> cpu_ostrides;

    void randomize()
    {
	const long max_nelts = 100 * 1000 * 1000;
	
	bool is_float32 = (rand_uniform() < 0.5);
	params.dtype = is_float32 ? "float32" : "float16";
	params.apply_input_residual_lags = (rand_uniform() < 0.5);
	params.input_is_downsampled_tree = (rand_uniform() < 0.5);
	this->in_place = (rand_uniform() < 0.5);
	
	params.nelts_per_segment = is_float32 ? 32 : 64;
	params.rank = rand_int(1, 9);

	long nchan = pow2(params.rank);
	params.ntime = rand_int(1, 2*nchan + 2*params.nelts_per_segment);
	params.ntime = align_up(params.ntime, params.nelts_per_segment);

	long cmax = (10*nchan + 10*params.ntime) / params.ntime;
	nchunks = rand_int(1, cmax+1);
	
	// nambient, (total_beams/beams_per_kernel_launch), beams_per_kernel_launch
	long pmax = max_nelts / (pow2(params.rank) * params.ntime * nchunks);
	pmax = max(pmax, 4L);
	pmax = min(pmax, 42L);

	auto v = gputils::random_integers_with_bounded_product(4, pmax);
	params.nambient = round_up_to_power_of_two(v[0]);
	params.total_beams = v[1] * v[2];
	params.beams_per_kernel_launch = v[2];

	// Strides
	vector<ssize_t> shape = { params.beams_per_kernel_launch, params.nambient, pow2(params.rank), params.ntime };
	gpu_istrides = gputils::make_random_strides(shape, 1, params.nelts_per_segment);
	cpu_istrides = gputils::make_random_strides(shape, 1, params.nelts_per_segment);
	gpu_ostrides = in_place ? gpu_istrides : gputils::make_random_strides(shape, 1, params.nelts_per_segment);
	cpu_ostrides = in_place ? cpu_istrides : gputils::make_random_strides(shape, 1, params.nelts_per_segment);
    }
};


static void run_test(const TestInstance &tp)
{
    const GpuDedispersionKernel::Params &p = tp.params;
    
    // Placeholders for future expansion.
    assert(!p.input_is_ringbuf);
    assert(!p.output_is_ringbuf);
    
    cout << "Test GpuDedispersionKernel\n"
	 << "    dtype = " << p.dtype << "\n"
	 << "    rank = " << p.rank << "\n"
	 << "    nambient = " << p.nambient << "\n"
	 << "    total_beams = " << p.total_beams << "\n"
	 << "    beams_per_kernel_launch = " << p.beams_per_kernel_launch << "\n"
	 << "    ntime = " << p.ntime << "\n"
	 << "    nchunks = " << tp.nchunks << "\n"
	 << "    input_is_ringbuf = " << (p.input_is_ringbuf ? "true" : "false")  << "\n"
	 << "    output_is_ringbuf = " << (p.output_is_ringbuf ? "true" : "false")  << "\n"
	 << "    apply_input_residual_lags = " << (p.apply_input_residual_lags ? "true" : "false")  << "\n"
	 << "    input_is_downsampled_tree = " << (p.input_is_downsampled_tree ? "true" : "false")  << "\n"
	 << "    nelts_per_segment = " << p.nelts_per_segment << "\n"
	 << "    in_place = " << (tp.in_place ? "true" : "false") << "\n"
	 << "    gpu_istrides = " << gputils::tuple_str(tp.gpu_istrides) << "\n"
	 << "    gpu_ostrides = " << gputils::tuple_str(tp.gpu_ostrides) << "\n"
	 << "    cpu_istrides = " << gputils::tuple_str(tp.cpu_istrides) << "\n"
	 << "    cpu_ostrides = " << gputils::tuple_str(tp.cpu_ostrides)
	 << endl;

    // FIXME unify ReferenceDedispersionKernel::Params and GpuDedispersionKernel::Params
    ReferenceDedispersionKernel::Params ref_params;
    ref_params.rank = p.rank;
    ref_params.ntime = p.ntime;
    ref_params.nambient = p.nambient;
    ref_params.nbeams = p.total_beams;
    ref_params.apply_input_residual_lags = p.apply_input_residual_lags;
    ref_params.is_downsampled_tree = p.input_is_downsampled_tree;
    ref_params.nelts_per_segment = p.nelts_per_segment;

    shared_ptr<ReferenceDedispersionKernel> ref_kernel = make_shared<ReferenceDedispersionKernel> (ref_params);
    shared_ptr<GpuDedispersionKernel> gpu_kernel = GpuDedispersionKernel::make(p);

    bool is_float32 = p.is_float32();
    vector<ssize_t> big_shape = { p.total_beams, p.nambient, pow2(p.rank), tp.nchunks * p.ntime };
    vector<ssize_t> medium_shape = { p.total_beams, p.nambient, pow2(p.rank), p.ntime };
    vector<ssize_t> small_shape = { p.beams_per_kernel_launch, p.nambient, pow2(p.rank), p.ntime };

    Array<float> cpu_in_big(big_shape, af_rhost | af_random);  // contiguous
    Array<float> cpu_out_big(big_shape, af_uhost | af_zero);   // contiguous
    Array<float> cpu_in_small(medium_shape, af_uhost | af_zero);  // contiguous for now (FIXME)
    // Array<float> cpu_out_small = tp.in_place ? cpu_in_small : Array<float> (small_shape, tp.cpu_ostrides, af_uhost | af_zero);
    
    UntypedArray gpu_in_big;
    UntypedArray gpu_out_big;
    UntypedArray gpu_in_small;
    UntypedArray gpu_out_small;

    if (is_float32) {
	gpu_in_big.data_float32 = cpu_in_big.to_gpu();
	gpu_out_big.data_float32 = Array<float> (big_shape, af_gpu | af_zero);
	gpu_in_small.data_float32 = Array<float> (small_shape, tp.gpu_istrides, af_gpu | af_zero);
	gpu_out_small.data_float32 = tp.in_place ? gpu_in_small.data_float32 : Array<float> (small_shape, tp.gpu_ostrides, af_gpu | af_zero);
    }
    else {
	// FIXME confirm that cudaMemcpy() is synchronous.
	Array<__half> t = cpu_in_big.convert_dtype<__half> ();
	gpu_in_big.data_float16 = t.to_gpu();
	gpu_out_big.data_float16 = Array<__half> (big_shape, af_gpu | af_zero);
	gpu_in_small.data_float16 = Array<__half> (small_shape, tp.gpu_istrides, af_gpu | af_zero);
	gpu_out_small.data_float16 = tp.in_place ? gpu_in_small.data_float16 : Array<__half> (small_shape, tp.gpu_ostrides, af_gpu | af_zero);
    }
    
    for (long ichunk = 0; ichunk < tp.nchunks; ichunk++) {
	Array<float> s;

	// Reference dedispersion.
	// FIXME currently in-place, and processing all beams

	s = cpu_in_big.slice(3, ichunk * p.ntime, (ichunk+1) * p.ntime);
	cpu_in_small.fill(s);
	
	ref_kernel->apply(cpu_in_small);
	
	s = cpu_out_big.slice(3, ichunk * p.ntime, (ichunk+1) * p.ntime);
	s.fill(cpu_in_small);

	// GPU dedispersion.
	for (int b = 0; b < p.total_beams; b += p.beams_per_kernel_launch) {
	    UntypedArray t;
	    
	    t = gpu_in_big.slice(0, b, b + p.beams_per_kernel_launch);
	    t = t.slice(3, ichunk * p.ntime, (ichunk+1) * p.ntime);
	    gpu_in_small.fill(t);

	    gpu_kernel->launch(gpu_in_small, gpu_out_small, ichunk, b);

	    t = gpu_out_big.slice(0, b, b + p.beams_per_kernel_launch);
	    t = t.slice(3, ichunk * p.ntime, (ichunk+1) * p.ntime);
	    t.fill(gpu_out_small);
	}
    }

    if (is_float32)
	gpu_out_big.data_float32 = gpu_out_big.data_float32.to_host();
    else {
	gpu_out_big.data_float16 = gpu_out_big.data_float16.to_host();
	gpu_out_big.data_float32 = gpu_out_big.data_float16.convert_dtype<float> ();
    }
    
    // FIXME revisit epsilon if we change the normalization of the dedispersion transform.
    double epsrel = is_float32 ? 1.0e-6 : 0.003;
    double epsabs = epsrel * pow(1.414, p.rank);
    gputils::assert_arrays_equal(cpu_out_big, gpu_out_big.data_float32, "cpu", "gpu", {"beam","amb","dmbr","time"}, epsabs, epsrel);
    cout << endl;
}


// -------------------------------------------------------------------------------------------------


int main(int argc, char **argv)
{
    const int niter = 200;
    
    for (int i = 0; i < niter; i++) {
	cout << "Iteration " << i << "/" << niter << "\n\n";
	TestInstance ti;
	ti.randomize();
	run_test(ti);
    }

    cout << "test-gpu-dedispersion-kernels: pass" << endl;
    return 0;
}

