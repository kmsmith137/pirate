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


// -------------------------------------------------------------------------------------------------


int main(int argc, char **argv)
{
    // FIXME switch to 'false' when no longer actively developing
    const bool noisy = true;
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
	
	TestInstance<__half> th;
	th.randomize();
	th.run(noisy);
	
	TestInstance<float> tf;
	tf.randomize();
	tf.run(noisy);
    }

    cout << "test-gpu-dedispersion-kernels: pass" << endl;
    return 0;
}

