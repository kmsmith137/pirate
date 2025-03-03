#include "../include/pirate/internals/ReferenceLaggedDownsamplingKernel.hpp"
#include "../include/pirate/internals/GpuLaggedDownsamplingKernel.hpp"
#include "../include/pirate/internals/inlines.hpp"  // pow2()
#include "../include/pirate/constants.hpp"

#include <ksgpu/Array.hpp>
#include <ksgpu/cuda_utils.hpp>
#include <ksgpu/rand_utils.hpp>    // rand_int()
#include <ksgpu/test_utils.hpp>    // assert_arrays_equal()

using namespace std;
using namespace pirate;
using namespace ksgpu;


// -------------------------------------------------------------------------------------------------


static long rand_stride(Dtype dtype, long min_stride)
{
    long n = (rand_uniform() < 0.5) ? 0 : rand_int(1,4);
    return min_stride + n * xdiv(1024, dtype.nbits);
}


struct TestInstance
{
    GpuLaggedDownsamplingKernel::Params params;

    long nbeams = 0;
    long nchunks = 0;
    long nt_chunk = 0;
    
    long bstride_in = 0;
    long bstride_out = 0;
    

    long min_bstride_in() const
    {
	return pow2(params.large_input_rank) * nt_chunk;
    }
    
    long min_bstride_out() const
    {
	long nt_out = 0;
	for (int d = 1; d <= params.num_downsampling_levels; d++)
	    nt_out += xdiv(nt_chunk, pow2(d));
	return pow2(params.large_input_rank-1) * nt_out;
    }
    
    void randomize()
    {
	params.dtype = (rand_uniform() < 0.5) ? Dtype::native<float>() : Dtype::native<__half>();
	params.small_input_rank = rand_int(2,9);  // GpuLaggedDownsamplingKernel needs small_input_rank >= 2
	params.large_input_rank = params.small_input_rank + rand_int(0,4);
	params.num_downsampling_levels = rand_int(1, constants::max_downsampling_level);  // no +1 here
	
	this->nbeams = rand_int(1,5);

	long nt_divisor = xdiv(1024, params.dtype.nbits) * pow2(params.num_downsampling_levels+1);
	long p = nbeams * pow2(params.large_input_rank) * nt_divisor;
	long q = (10*1000*1000) / p;
	q = max(q, 4L);

	auto v = ksgpu::random_integers_with_bounded_product(2,q);
	this->nt_chunk = nt_divisor * v[0];
	this->nchunks = v[1];
	
	this->bstride_in = rand_stride(params.dtype, min_bstride_in());
	this->bstride_out = rand_stride(params.dtype, min_bstride_out());
    }


    // ---------------------------------------------------------------------------------------------
    //
    // Helpers for input/output array allocation.
    

    // alloc_input() returns array of shape (nbeams, 2^large_input_rank, nt_chunk).
    // The beam axis may have a non-contiguous stride -- see below.    

    template<typename T>
    Array<T> alloc_input(Dtype dtype, bool use_bstride_in, int aflags)
    {
	long nr = pow2(params.large_input_rank);
	long bstride = use_bstride_in ? bstride_in : min_bstride_in();
	return Array<T> (dtype, {nbeams,nr,nt_chunk}, {bstride,nt_chunk,1}, aflags);
    }


    // Return type from alloc_output().
    template<typename T>
    struct OutputArrays
    {
	// big_arr has shape (nbeams, min_bstride_out).
	// The beam axis may have a non-contiguous stride.
	Array<T> big_arr;

	// small_arrs[ids] has shape (nbeams, 2^(large_input_rank-1), nt_chunk/2^(ids+1)).
	// The beam axis will usually have a non-contiguous stride.
	vector<Array<T>> small_arrs;
    };


    template<typename T>
    OutputArrays<T> alloc_output(Dtype dtype, bool use_bstride_out, int aflags)
    {
	long bstride_min = min_bstride_out();
	long bstride = use_bstride_out ? bstride_out : bstride_min;
	// std::initializer_list<long> strides = { bstride, 1 };
	
	long nr = pow2(params.large_input_rank - 1);
	long nt_cumul = 0;
	
	OutputArrays<T> ret;
	ret.big_arr = Array<T> (dtype, {nbeams,bstride_min}, {bstride,1}, aflags);
	ret.small_arrs.resize(params.num_downsampling_levels);
	
	for (int i = 0; i < params.num_downsampling_levels; i++) {
	    long nt_ds = xdiv(nt_chunk, pow2(i+1));
	    Array<T> a = ret.big_arr.slice(1, nr * nt_cumul, nr * (nt_cumul + nt_ds));
	    ret.small_arrs[i] = a.reshape({ nbeams, nr, nt_ds });
	    nt_cumul += nt_ds;
	}

	assert(nr * nt_cumul == bstride_min);
	return ret;
    }


    // ---------------------------------------------------------------------------------------------
    

    void run()
    {
	cout << "Test GpuLaggedDownsamplingKernel\n"
	     << "    dtype = " << params.dtype << "\n"
	     << "    small_input_rank = " << params.small_input_rank << "\n"
	     << "    large_input_rank = " << params.large_input_rank << "\n"
	     << "    num_downsampling_levels = " << params.num_downsampling_levels << "\n"
	     << "    nbeams = " << nbeams << "\n"
	     << "    nchunks = " << nchunks << "\n"
	     << "    nt_chunk = " << nt_chunk << "\n"
	     << "    bstride_in = " << bstride_in << " (min: " << min_bstride_in() << ")\n"
	     << "    bstride_out = " << bstride_out << " (min: " << min_bstride_out() << ")\n";
	
	assert(params.small_input_rank > 0);
	assert(params.large_input_rank >= params.small_input_rank);
	assert(params.large_input_rank <= constants::max_tree_rank);
	assert(params.num_downsampling_levels > 0);
	assert(params.num_downsampling_levels <= constants::max_downsampling_level-1);
	assert(nbeams > 0);
	assert(nchunks > 0);
	assert(nt_chunk > 0);
	// assert((nt_chunk * sizeof(T)) % (pow2(num_downsampling_levels) * constants::bytes_per_gpu_cache_line) == 0);
	assert(bstride_in >= min_bstride_in());
	assert(bstride_out >= min_bstride_out());

	ReferenceLaggedDownsamplingKernel::Params ref_params;
	ref_params.small_input_rank = params.small_input_rank;
	ref_params.large_input_rank = params.large_input_rank;
	ref_params.num_downsampling_levels = params.num_downsampling_levels;
	ref_params.nbeams = nbeams;
	ref_params.ntime = nt_chunk;
	
	auto ref_kernel = make_shared<ReferenceLaggedDownsamplingKernel> (ref_params);
	auto gpu_kernel = GpuLaggedDownsamplingKernel::make(this->params);

	cout << "    GPU kernel params\n";
	gpu_kernel->print(cout, 8);
	cout << flush;

	Array<void> gpu_in = alloc_input<void> (params.dtype, true, af_gpu | af_zero);                          // use_bstride_in = true
	OutputArrays<void> gpu_out = alloc_output<void> (params.dtype, true, af_gpu | af_zero);                 // use_bstride_out = true
	OutputArrays<float> cpu_out = alloc_output<float> (Dtype::native<float>(), false, af_uhost | af_zero);  // use_bstride_out = false
	Array<void> gpu_pstate(params.dtype, { nbeams, gpu_kernel->state_nelts_per_beam }, af_gpu | af_zero);   // af_zero is important here

	for (int ichunk = 0; ichunk < nchunks; ichunk++) {
#if 1
	    // Random chunk gives strongest test.
	    Array<float> cpu_in = alloc_input<float> (Dtype::native<float>(), false, af_rhost | af_random);  // use_bstride_in = true
#else
	    // One-hot chunk is sometimes useful for debugging.
	    Array<float> cpu_in = alloc_input<float> (Dtype::native<float>(), false, af_rhost | af_zero);  // use_bstride_in = true
	    long ibeam = rand_int(0, nbeams);
	    long irow = rand_int(0, pow2(params.large_input_rank));
	    long it = rand_int(0, nt_chunk);
	    cout << "   one-hot chunk: ibeam=" << ibeam << "; irow=" << irow << "; it=" << it << ";" << endl;
	    cpu_in.at({ibeam,irow,it}) = 1.0;
#endif
	    
	    ref_kernel->apply(cpu_in, cpu_out.small_arrs);
	    
	    // Copy cpu_in -> gpu_in
	    array_fill(gpu_in, cpu_in.convert(params.dtype));

	    gpu_kernel->launch(gpu_in, gpu_out.small_arrs, gpu_pstate, ichunk * nt_chunk);
	    CUDA_CALL(cudaDeviceSynchronize());

	    for (int ids = 0; ids < params.num_downsampling_levels; ids++) {
		// Note: if the definition of the LaggedDownsampler changes to include factors of 0.5,
		// then the value of 'rms' may ned to change.
		
		double rms = sqrt(4 << ids);    // rms of output array
		double epsabs = 3 * params.dtype.precision() * rms * sqrt(ids+2);
		
		// cout << "ichunk=" << ichunk << ", ids=" << ids << ", epsabs=" << epsabs << endl;
		assert_arrays_equal(cpu_out.small_arrs.at(ids), gpu_out.small_arrs.at(ids),
				    "ref", "gpu", {"beam","freq","time"}, epsabs, 0.0);  // epsrel=0
	    }
	}
    }
};


// -------------------------------------------------------------------------------------------------


int main(int argc, char **argv)
{
#if 0
    // Uncomment to enable specific test
    TestInstance t;
    t.params.dtype = "float32";
    t.params.small_input_rank = 8;
    t.params.large_input_rank = 9;
    t.params.num_downsampling_levels = 5;
    t.nbeams = 4;
    t.nchunks = 5;
    t.nt_chunk = 12288;
    t.bstride_in = t.min_bstride_in() + 64;
    t.bstride_out = t.min_bstride_out() + 128;
    t.run();
    return 0;
#endif

    const int ntests = 50;
    
    for (int i = 0; i < ntests; i++) {
	cout << "\nTest " << i << "/" << ntests << "\n\n";

	TestInstance ti;
	ti.randomize();
	ti.run();
    }

    cout << "test-gpu-lagged-downsampler: pass" << endl;
    return 0;
}

