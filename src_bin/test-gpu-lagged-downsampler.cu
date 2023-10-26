#include "../include/pirate/internals/ReferenceLaggedDownsampler.hpp"
#include "../include/pirate/internals/GpuLaggedDownsamplingKernel.hpp"
#include "../include/pirate/internals/inlines.hpp"  // pow2()
#include "../include/pirate/constants.hpp"

#include <gputils/Array.hpp>
#include <gputils/cuda_utils.hpp>
#include <gputils/rand_utils.hpp>    // rand_int()
#include <gputils/test_utils.hpp>    // assert_arrays_equal()

using namespace std;
using namespace pirate;
using namespace gputils;


// -------------------------------------------------------------------------------------------------


template<typename T>
static long rand_stride(long min_stride)
{
    // FIXME allow negative strides
    long n = (rand_uniform() < 0.5) ? 0 : rand_int(1,4);
    return min_stride + n * (128/sizeof(T));
}


template<typename T>
struct TestInstance
{
    int small_input_rank = 0;
    int large_input_rank = 0;
    int num_downsampling_levels = 0;

    long nbeams = 0;
    long nchunks = 0;
    long nt_chunk = 0;
    
    long bstride_in = 0;
    long bstride_out = 0;
    

    long min_bstride_in() const
    {
	return pow2(large_input_rank) * nt_chunk;
    }
    
    long min_bstride_out() const
    {
	long nt_out = 0;
	for (int d = 1; d <= num_downsampling_levels; d++)
	    nt_out += xdiv(nt_chunk, pow2(d));
	return pow2(large_input_rank-1) * nt_out;
    }
    
    void randomize()
    {
	small_input_rank = rand_int(2, 9);  // GpuLaggedDownsamplingKernel needs small_input_rank >= 2
	large_input_rank = rand_int(small_input_rank, 13);
	num_downsampling_levels = rand_int(1, constants::max_downsampling_level);  // no +1 here
	nchunks = rand_int(2, 11);

	// Bytes per time sample per beam.
	long nb0 = pow2(large_input_rank) * sizeof(T);

	long nt_divisor = xdiv(128,sizeof(T)) * pow2(num_downsampling_levels+1);
	long imax = min(10L, (long)(2.0e9 / nb0 / nt_divisor));
	nt_chunk = nt_divisor * rand_int(1, imax+1);

	long bmax = min(10L, (long)(2.1e9 / nb0 / nt_chunk));
	nbeams = rand_int(1, bmax+1);
	
	bstride_in = rand_stride<T> (min_bstride_in());
	bstride_out = rand_stride<T> (min_bstride_out());
    }
    


    // ---------------------------------------------------------------------------------------------
    //
    // Helpers for input/output array allocation.
    

    // alloc_input() returns array of shape (nbeams, 2^large_input_rank, nt_chunk).
    // The beam axis may have a non-contiguous stride -- see below.    
    template<typename T2> 
    Array<T2> alloc_input(bool use_bstride_in, int aflags)
    {
	long nr = pow2(large_input_rank);
	long bstride = use_bstride_in ? bstride_in : min_bstride_in();
	return Array<T2> ({nbeams,nr,nt_chunk}, {bstride,nt_chunk,1}, aflags);
    }


    // Return type from alloc_output().
    template<typename T2> struct OutputArrays
    {
	// big_arr has shape (nbeams, min_bstride_out).
	// The beam axis may have a non-contiguous stride.
	Array<T2> big_arr;

	// small_arrs[ids] has shape (nbeams, 2^(large_input_rank-1), nt_chunk/2^(ids+1)).
	// The beam axis will usually have a non-contiguous stride.
	vector<Array<T2>> small_arrs;
    };


    template<typename T2>
    OutputArrays<T2> alloc_output(bool use_bstride_out, int aflags)
    {
	long bstride_min = min_bstride_out();
	long bstride = use_bstride_out ? bstride_out : bstride_min;
	long nr = pow2(large_input_rank - 1);
	long nt_cumul = 0;

	OutputArrays<T2> ret;
	ret.big_arr = Array<T2>({nbeams,bstride_min}, {bstride,1}, aflags);
	ret.small_arrs.resize(num_downsampling_levels);
	
	for (int i = 0; i < num_downsampling_levels; i++) {
	    long nt_ds = xdiv(nt_chunk, pow2(i+1));
	    Array<T2> a = ret.big_arr.slice(1, nr * nt_cumul, nr * (nt_cumul + nt_ds));
	    ret.small_arrs[i] = a.reshape_ref({ nbeams, nr, nt_ds });
	    nt_cumul += nt_ds;
	}

	assert(nr * nt_cumul == bstride_min);
	return ret;
    }


    // ---------------------------------------------------------------------------------------------
    

    void run(bool noisy)
    {
	if (noisy) {
	    cout << "Test GpuLaggedDownsamplingKernel<" << type_name<T>() << ">\n"
		 << "    small_input_rank = " << small_input_rank << "\n"
		 << "    large_input_rank = " << large_input_rank << "\n"
		 << "    num_downsampling_levels = " << num_downsampling_levels << "\n"
		 << "    nbeams = " << nbeams << "\n"
		 << "    nchunks = " << nchunks << "\n"
		 << "    nt_chunk = " << nt_chunk << "\n"
		 << "    bstride_in = " << bstride_in << " (min: " << min_bstride_in() << ")\n"
		 << "    bstride_out = " << bstride_out << " (min: " << min_bstride_out() << ")\n";
	}

	assert(small_input_rank > 0);
	assert(large_input_rank >= small_input_rank);
	assert(large_input_rank <= constants::max_tree_rank);
	assert(num_downsampling_levels > 0);
	assert(num_downsampling_levels <= constants::max_downsampling_level-1);
	assert(nbeams > 0);
	assert(nchunks > 0);
	assert(nt_chunk > 0);
	assert((nt_chunk * sizeof(T)) % (pow2(num_downsampling_levels) * constants::bytes_per_gpu_cache_line) == 0);
	assert(bstride_in >= min_bstride_in());
	assert(bstride_out >= min_bstride_out());

	ReferenceLaggedDownsampler::Params ref_params;
	ref_params.small_input_rank = small_input_rank;
	ref_params.large_input_rank = large_input_rank;
	ref_params.num_downsampling_levels = num_downsampling_levels;
	ref_params.nbeams = nbeams;
	ref_params.ntime = nt_chunk;

	auto ref_kernel = make_shared<ReferenceLaggedDownsampler> (ref_params);
	auto gpu_kernel = GpuLaggedDownsamplingKernel<T>::make(small_input_rank, large_input_rank, num_downsampling_levels);

	if (noisy) {
	    cout << "    GPU kernel params\n";
	    gpu_kernel->print(cout, 8);
	    cout << flush;
	}

	Array<T> gpu_in = alloc_input<T> (true, af_gpu | af_zero);       // use_bstride_in = true
	Array<T> gpu_state({ nbeams, gpu_kernel->params.state_nelts_per_beam }, af_gpu | af_zero);  // af_zero is important here

	OutputArrays<T> gpu_out = alloc_output<T> (true, af_gpu | af_zero);            // use_bstride_out = true
	OutputArrays<float> cpu_out = alloc_output<float> (false, af_uhost | af_zero); // use_bstride_out = false

	for (int ichunk = 0; ichunk < nchunks; ichunk++) {
#if 1
	    // Random chunk gives strongest test.
	    Array<float> cpu_in = alloc_input<float> (false, af_rhost | af_random);  // use_bstride_in = false
#else
	    // One-hot chunk is sometimes useful for debugging.
	    Array<float> cpu_in = alloc_input<float> (false, af_rhost | af_zero);    // use_bstride_in = false
	    long ibeam = rand_int(0, nbeams);
	    long irow = rand_int(0, pow2(large_input_rank));
	    long it = rand_int(0, nt_chunk);
	    cout << "   one-hot chunk: ibeam=" << ibeam << "; irow=" << irow << "; it=" << it << ";" << endl;
	    cpu_in.at({ibeam,irow,it}) = 1.0;
#endif

	    ref_kernel->apply(cpu_in, cpu_out.small_arrs);
	    
	    gpu_in.fill(cpu_in.convert_dtype<T> ());	    
	    gpu_kernel->launch(gpu_in, gpu_out.small_arrs, gpu_state, ichunk * nt_chunk);
	    CUDA_CALL(cudaDeviceSynchronize());

	    for (int ids = 0; ids < num_downsampling_levels; ids++) {
		// Note: if the definition of the LaggedDownsampler changes to include factors of 0.5,
		// then the value of 'rms' may ned to change.
		
		double eps = (sizeof(T)==4) ? 3.0e-7 : 3.0e-3;   // float32 vs float16
		double rms = sqrt(4 << ids);                     // rms of output array
		double epsabs = eps * rms * sqrt(ids+2);
		
		if (noisy)
		    cout << "ichunk=" << ichunk << ", ids=" << ids << ", epsabs=" << epsabs << endl;
		
		Array<float> from_gpu = gpu_out.small_arrs[ids].to_host().template convert_dtype<float> ();
		assert_arrays_equal(cpu_out.small_arrs[ids], from_gpu, "ref", "gpu", {"beam","freq","time"}, epsabs, 0.0);  // epsrel=0
	    }
	}
    }
};


// -------------------------------------------------------------------------------------------------


int main(int argc, char **argv)
{
    // FIXME switch to 'false' when no longer actively developing
    const bool noisy = true;
    const int ntests = 50;

#if 0
    // Uncomment to enable specific test
    TestInstance<__half> t;
    t.small_input_rank = 8;
    t.large_input_rank = 9;
    t.num_downsampling_levels = 5;
    t.nbeams = 4;
    t.nchunks = 5;
    t.nt_chunk = 12288;
    t.bstride_in = t.min_bstride_in() + 64;
    t.bstride_out = t.min_bstride_out() + 128;
    t.run(noisy);
    return 0;
#endif
    
    for (int i = 0; i < ntests; i++) {
	cout << "\nTest " << i << "/" << ntests << "\n\n";
	
	TestInstance<float> t32;
	t32.randomize();
	t32.run(noisy);

	cout << endl;
	
	TestInstance<__half> t16;
	t16.randomize();
	t16.run(noisy);
    }

    cout << "test-gpu-lagged-downsampler: pass" << endl;
    return 0;
}

