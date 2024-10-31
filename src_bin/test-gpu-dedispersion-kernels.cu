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
    vector<ssize_t> istrides;
    vector<ssize_t> ostrides;

    void randomize()
    {
	const long max_nelts = 30 * 1000 * 1000;
	
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
	params.nambient = v[0];
	params.total_beams = v[1] * v[2];
	params.beams_per_kernel_launch = v[2];

	// Strides
	vector<ssize_t> shape = { params.beams_per_kernel_launch, params.nambient, pow2(params.rank), params.ntime };
	istrides = gputils::make_random_strides(shape, 1, params.nelts_per_segment);
	ostrides = gputils::make_random_strides(shape, 1, params.nelts_per_segment);
	ostrides = in_place ? istrides : ostrides;
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
	 << "    input_is_ringbuf = " << (p.input_is_ringbuf ? "true" : "false")  << "\n"
	 << "    output_is_ringbuf = " << (p.output_is_ringbuf ? "true" : "false")  << "\n"
	 << "    apply_input_residual_lags = " << (p.apply_input_residual_lags ? "true" : "false")  << "\n"
	 << "    input_is_downsampled_tree = " << (p.input_is_downsampled_tree ? "true" : "false")  << "\n"
	 << "    nelts_per_segment = " << p.nelts_per_segment << "\n"
	 << "    in_place = " << (tp.in_place ? "true" : "false") << "\n"
	 << "    istrides = " << gputils::tuple_str(tp.istrides) << "\n"
	 << "    ostrides = " << gputils::tuple_str(tp.ostrides)
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
    vector<shape> big_shape = { p.total_beams, p.nambient, pow2(p.rank), p.ntime };
    vector<shape> small_shape = { p.beams_per_kernel_launch, p.nambient, pow2(p.rank), p.ntime };

    Array<float> chunk0(big_shape, af_rhost | af_zero);      // no strides, in order to call randomize().
    Array<float> ref_chunk(big_shape, af_rhost | af_zero);   // FIXME reference kernel uses big_shape, in-place

    UntypedArray gpu_input_chunk;
    UntypedArray gpu_output_chunk;

    if (is_float32) {
	gpu_input_chunk.data_float32 = Array<float> (big_shape, tp.istride, af_gpu | af_zero);
	gpu_output_chunk.data_float32 = tp.in_place ? gpu_input_chunk.data_float32 : Array<float> (small_shape, tp.istride, af_gpu | af_zero);
    }
    else {
	gpu_input_chunk.data_float16 = Array<__half> (big_shape, tp.istride, af_gpu | af_zero);
	gpu_output_chunk.data_float16 = tp.in_place ? gpu_input_chunk.data_float16 : Array<__half> (small_shape, tp.istride, af_gpu | af_zero);
    }
    
    for (ichunk = 0; ichunk < nchunks; ichunk++) {
	// cout << "   chunk " << ichunk << "/" << nchunks << endl;

#if 1
	// Simulate chunk0.
	// Random chunk gives strongest test.
	gputils::randomize(chunk0.data, chunk0.size);
#else
	// One-hot chunk is sometimes useful for debugging.
	// (Note that if nchunks > 0, then the one-hot chunk will be repeated multiple times.)
	memset(chunk0.data, 0, chunk0.size * sizeof(float));
	int ibeam = rand_int(0, p.total_beams);   // FIXME I think "chunk0" will become a small chunk.
	int iamb = rand_int(0, p.nambient);
	int irow = rand_int(0, pow2(p.rank));
	int it = rand_int(0, p.ntime);
	// ibeam=0; iamb=0; irow=0; it=9; // Uncomment if you want a non-random one-hot test
	cout << "   one-hot chunk: ibeam=" << ibeam << "; iamb=" << iamb << "; irow=" << irow << "; it=" << it << ";" << endl;
	chunk0.at({ibeam,iamb,irow,it}) = 1.0;
#endif

	// Reference dedispersion.
	// FIXME no strides on ref_chunk for now.
	ref_chunk.fill(chunk0);
	ref_kernel->apply(ref_chunk);

	// Incrementally copy chunk to GPU and dedisperse.
	for (int b = 0; b < p.total_beams; b += p.beams_per_kernel_launch) {
	    
	}
    }
}


#if 0


// FIXME delete after de-templating.
template<typename T> struct _is_float32 { };
template<> struct _is_float32<float>   { static constexpr bool value = true; };
template<> struct _is_float32<__half>  { static constexpr bool value = false; };


template<typename T>
struct TestInstance
{
    int rank = 0;
    int ntime = 0;
    int nambient = 1;
    int nbeams = 1;
    int nchunks = 1;
    long row_stride = 0;
    long ambient_stride = 0;
    long beam_stride = 0;
    bool apply_input_residual_lags = false;
    bool is_downsampled_tree = false;
    

    int rand_n(long nmax)
    {
	nmax = min(nmax, 10L);
	nmax = max(nmax, 1L);
	return rand_int(1, nmax+1);
    }

    long rand_stride(long smin)
    {
	int n = max(0L, rand_int(-10,10));
	return smin + 64 * n;  // FIXME 64 -> (128 / sizeof(T))
    }
    
    void randomize()
    {
	const long max_nelts = 30 * 1000 * 1000;

	rank = rand_int(1, 9);
	nchunks = rand_int(1, 10);
	nambient = pow2(rand_int(0,4));
	apply_input_residual_lags = (rand_uniform() < 0.66) ? true : false;
	is_downsampled_tree = (rand_uniform() < 0.5) ? true : false;

	long nelts = pow2(rank) * nchunks * nambient;
	ntime = 64 * rand_n(max_nelts / (64 * nelts));
	nelts *= ntime;
	
	nbeams = rand_n(max_nelts / nelts);
	nelts *= nbeams;
	
	row_stride = rand_stride(ntime);
	ambient_stride = rand_stride(row_stride * pow2(rank));
	beam_stride = rand_stride(ambient_stride * nambient);
    }
    
    
    void run(bool noisy)
    {
	// No real argument checking, but check that everything was initialized.
	assert(rank > 0);
	assert(ntime > 0);
	assert(nambient > 0);
	assert(nbeams > 0);
	assert(nchunks > 0);
	assert(row_stride > 0);
	assert(ambient_stride > 0);
	assert(beam_stride > 0);
	
	if (noisy) {
	    long min_row_stride = ntime;
	    long min_ambient_stride = row_stride * pow2(rank);
	    long min_beam_stride = ambient_stride * nambient;
	    
	    cout << "Test GpuDedispersionKernel\n"
		 << "    dtype = " << gputils::type_name<T>() << "\n"
		 << "    rank = " << rank << "\n"
		 << "    ntime = " << ntime << "\n"
		 << "    nambient = " << nambient << "\n"
		 << "    nbeams = " << nbeams << "\n"
		 << "    nchunks = " << nchunks << "\n"
		 << "    row_stride = " << row_stride << " (minimum: " << min_row_stride << ")\n"
		 << "    ambient_stride = " << ambient_stride << " (minimum: " << min_ambient_stride << ")\n"
		 << "    beam_stride = " << beam_stride << " (minimum: " << min_beam_stride << ")\n"
		 << "    apply_input_residual_lags = " << (apply_input_residual_lags ? "true\n" : "false\n")
		 << "    is_downsampled_tree = " << (is_downsampled_tree ? "true" : "false")
		 << endl;
	}

	ReferenceDedispersionKernel::Params ref_params;
	ref_params.rank = rank;
	ref_params.ntime = ntime;
	ref_params.nambient = nambient;
	ref_params.nbeams = nbeams;
	ref_params.apply_input_residual_lags = apply_input_residual_lags;
	ref_params.is_downsampled_tree = is_downsampled_tree;
	ref_params.nelts_per_segment = xdiv(constants::bytes_per_gpu_cache_line, sizeof(T));  // matches DedispersionConfig::get_nelts_per_segment()

	constexpr bool is_float32 = _is_float32<T>::value;
	typename GpuDedispersionKernel::Params gpu_params;
	gpu_params.dtype = is_float32 ? "float32" : "float16";
	gpu_params.rank = rank;
	gpu_params.nambient = nambient;
	gpu_params.total_beams = nbeams;  // FIXME process in batches
	gpu_params.beams_per_kernel_launch = nbeams;
	gpu_params.ntime = ntime;
	gpu_params.apply_input_residual_lags = apply_input_residual_lags;
	gpu_params.input_is_downsampled_tree = is_downsampled_tree;
	gpu_params.nelts_per_segment = is_float32 ? 32 : 64;

	ReferenceDedispersionKernel ref_kernel(ref_params);
	shared_ptr<GpuDedispersionKernel> gpu_kernel = GpuDedispersionKernel::make(gpu_params);

	Array<T> gpu_iobuf({ nbeams, nambient, pow2(rank), ntime },         // shape
			   { beam_stride, ambient_stride, row_stride, 1 },  // strides
			   af_gpu | af_zero);

	UntypedArray gpu_ubuf;
	if constexpr (is_float32)
	    gpu_ubuf.data_float32 = gpu_iobuf;
	else
	    gpu_ubuf.data_float16 = gpu_iobuf;
	
	for (int ichunk = 0; ichunk < nchunks; ichunk++) {
#if 1
	    // Random chunk gives strongest test.
	    Array<float> ref_chunk({nbeams, nambient, pow2(rank), ntime},
				   { beam_stride, ambient_stride, row_stride, 1 },  // strides
				   af_rhost | af_random);
#else
	    // One-hot chunk is sometimes useful for debugging.
	    // (Note that if nchunks > 0, then the one-hot chunk will be repeated multiple times.)
	    Array<float> ref_chunk({nbeams, nambient, pow2(rank), ntime},
				   { beam_stride, ambient_stride, row_stride, 1 },  // strides
				   af_rhost | af_zero);
	    
	    cout << "   ichunk=" << ichunk << endl;
	    int ibeam = rand_int(0, nbeams);
	    int iamb = rand_int(0, nambient);
	    int irow = rand_int(0, pow2(rank));
	    int it = rand_int(0, ntime);
	    // ibeam=0; iamb=0; irow=0; it=9; // Uncomment if you want a non-random one-hot test
	    cout << "   one-hot chunk: ibeam=" << ibeam << "; iamb=" << iamb << "; irow=" << irow << "; it=" << it << ";" << endl;
	    ref_chunk.at({ibeam,iamb,irow,it}) = 1.0;
#endif

	    // Copy array to GPU before doing reference dedispersion, since reference dedispersion modifies array in-place.
	    gpu_iobuf.fill(ref_chunk.convert_dtype<T>());
	    gpu_kernel->launch(gpu_ubuf, gpu_ubuf, ichunk, 0);
	    CUDA_CALL(cudaDeviceSynchronize());
	    Array<float> gpu_output = gpu_iobuf.to_host().template convert_dtype<float> ();
	    
	    ref_kernel.apply(ref_chunk);

#if 0
	    // Sometimes useful for debugging
	    cout << "Printing reference output from chunk " << ichunk << endl;
	    print_array(ref_chunk, {"beam","amb","dmbr","time"});
	    cout << "Printing gpu output from chunk " << ichunk << endl;
	    print_array(gpu_output, {"beam","amb","dmbr","time"});
	    cout << "Printing gpu rstate from chunk " << ichunk << endl;
	    print_array(gpu_rstate.to_host().convert_dtype<float>(), {"beam","amb","ix"});
#endif

	    // FIXME revisit epsilon if we change the normalization of the dedispersion transform.
	    double epsrel = (sizeof(T)==4) ? 1.0e-6 : 0.003;   // float32 vs float16
	    double epsabs = epsrel * pow(1.414, rank);
	    assert_arrays_equal(ref_chunk, gpu_output, "ref", "gpu", {"beam","amb","dmbr","time"}, epsabs, epsrel);
	}

	if (noisy)
	    cout << endl;
    }
};

#endif


// -------------------------------------------------------------------------------------------------


int main(int argc, char **argv)
{
    const int niter = 500;

#if 0
    for (int i = 0; i < niter; i++) {
	cout << "Iteration " << i << "/" << niter << "\n\n";
	
	using T = __half;  // float or __half
	TestInstance<T> t;
	t.rank = 7;
	t.ntime = 192;
        t.nambient = 4;
	t.nbeams = 2; 
	t.nchunks = 9;
	t.row_stride = t.ntime + 64;
	t.ambient_stride = t.row_stride * pow2(t.rank) + 64*3;
	t.beam_stride = t.ambient_stride * t.nambient + 64*11;
	t.apply_input_residual_lags = true;
	t.run(noisy);
    }
    return 0;
#endif
    
    for (int i = 0; i < niter; i++) {
	cout << "Iteration " << i << "/" << niter << "\n\n";

	TestInstance ti;
	ti.randomize();
	run_test(ti);
    }

    cout << "test-gpu-dedispersion-kernels: pass" << endl;
    return 0;
}

