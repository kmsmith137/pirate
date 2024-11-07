#include "../include/pirate/internals/ReferenceLaggedDownsamplingKernel.hpp"
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


static long rand_stride(long min_stride, bool is_float32)
{
    // FIXME allow negative strides
    long n = (rand_uniform() < 0.5) ? 0 : rand_int(1,4);
    return min_stride + n * (is_float32 ? 32 : 64);
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
	params.dtype = (rand_uniform() < 0.5) ? "float32" : "float16";
	params.small_input_rank = rand_int(2,9);  // GpuLaggedDownsamplingKernel needs small_input_rank >= 2
	params.large_input_rank = params.small_input_rank + rand_int(0,4);
	params.num_downsampling_levels = rand_int(1, constants::max_downsampling_level);  // no +1 here
	
	this->nbeams = rand_int(1,5);

	long nt_divisor = (params.is_float32() ? 32 : 64) * pow2(params.num_downsampling_levels+1);
	long p = nbeams * pow2(params.large_input_rank) * nt_divisor;
	long q = (10*1000*1000) / p;
	q = max(q, 4L);

	auto v = gputils::random_integers_with_bounded_product(2,q);
	this->nt_chunk = nt_divisor * v[0];
	this->nchunks = v[1];
	
	this->bstride_in = rand_stride(min_bstride_in(), params.is_float32());
	this->bstride_out = rand_stride(min_bstride_out(), params.is_float32());
    }


    // ---------------------------------------------------------------------------------------------
    //
    // Helpers for input/output array allocation.
    

    // alloc_input() returns array of shape (nbeams, 2^large_input_rank, nt_chunk).
    // The beam axis may have a non-contiguous stride -- see below.    
	
    UntypedArray alloc_input(bool use_bstride_in, int aflags, bool is_float32)
    {
	long nr = pow2(params.large_input_rank);
	long bstride = use_bstride_in ? bstride_in : min_bstride_in();
	
	UntypedArray ret;
	ret.allocate({nbeams,nr,nt_chunk}, {bstride,nt_chunk,1}, aflags, is_float32);
	return ret;
    }


    // Return type from alloc_output().
    struct OutputArrays
    {
	// big_arr has shape (nbeams, min_bstride_out).
	// The beam axis may have a non-contiguous stride.
	UntypedArray big_arr;

	// small_arrs[ids] has shape (nbeams, 2^(large_input_rank-1), nt_chunk/2^(ids+1)).
	// The beam axis will usually have a non-contiguous stride.
	vector<UntypedArray> small_arrs;

	vector<Array<float>> get_float32()
	{
	    int n = small_arrs.size();
	    vector<Array<float>> ret(n);
	    for (int i = 0; i < n; i++)
		ret[i] = uarr_get<float> (small_arrs[i], "out");
	    return ret;
	}
    };


    OutputArrays alloc_output(bool use_bstride_out, int aflags, bool is_float32)
    {
	long bstride_min = min_bstride_out();
	long bstride = use_bstride_out ? bstride_out : bstride_min;
	// std::initializer_list strides = { bstride, 1 };
	
	long nr = pow2(params.large_input_rank - 1);
	long nt_cumul = 0;
	
	OutputArrays ret;
	ret.big_arr.allocate({nbeams,bstride_min}, {bstride,1}, aflags, is_float32);
	ret.small_arrs.resize(params.num_downsampling_levels);
	
	for (int i = 0; i < params.num_downsampling_levels; i++) {
	    long nt_ds = xdiv(nt_chunk, pow2(i+1));
	    UntypedArray a = ret.big_arr.slice(1, nr * nt_cumul, nr * (nt_cumul + nt_ds));

	    if (is_float32)
		ret.small_arrs[i].data_float32 = a.data_float32.reshape_ref({ nbeams, nr, nt_ds });
	    else
		ret.small_arrs[i].data_float16 = a.data_float16.reshape_ref({ nbeams, nr, nt_ds });
	    
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

	bool is_float32 = params.is_float32();
	
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

	UntypedArray gpu_state;
	gpu_state.allocate({ nbeams, gpu_kernel->state_nelts_per_beam }, af_gpu | af_zero, is_float32);  // af_zero is important here

	UntypedArray gpu_in = alloc_input(true, af_gpu | af_zero, is_float32);    // use_bstride_in = true
	OutputArrays gpu_out = alloc_output(true, af_gpu | af_zero, is_float32);  // use_bstride_out = true
	
	OutputArrays cpu_out_untyped = alloc_output(false, af_uhost | af_zero, true);     // (use_bstride_out, is_float32) = (false, true)
	vector<Array<float>> cpu_out = cpu_out_untyped.get_float32();
	
	for (int ichunk = 0; ichunk < nchunks; ichunk++) {
#if 1
	    // Random chunk gives strongest test.
	    UntypedArray cpu_in_untyped = alloc_input(false, af_rhost | af_random, true);  // (use_bstride_out, is_float32) = (false, true)
	    Array<float> cpu_in = uarr_get<float> (cpu_in_untyped, "cpu_in");
#else
	    // One-hot chunk is sometimes useful for debugging.
	    UntypedArray cpu_in = alloc_input(false, af_rhost | af_zero, true);
	    long ibeam = rand_int(0, nbeams);
	    long irow = rand_int(0, pow2(params.large_input_rank));
	    long it = rand_int(0, nt_chunk);
	    cout << "   one-hot chunk: ibeam=" << ibeam << "; irow=" << irow << "; it=" << it << ";" << endl;
	    cpu_in.data_float32.at({ibeam,irow,it}) = 1.0;
#endif
	    
	    ref_kernel->apply(cpu_in, cpu_out);;

	    // Copy cpu_in -> gpu_in
	    if (is_float32)
		gpu_in.data_float32.fill(cpu_in);
	    else {
		Array<__half> g2h = cpu_in.convert_dtype<__half> ();
		gpu_in.data_float16.fill(g2h);
	    }

	    gpu_kernel->launch(gpu_in, gpu_out.small_arrs, gpu_state, ichunk * nt_chunk);
	    CUDA_CALL(cudaDeviceSynchronize());

	    for (int ids = 0; ids < params.num_downsampling_levels; ids++) {
		// Note: if the definition of the LaggedDownsampler changes to include factors of 0.5,
		// then the value of 'rms' may ned to change.
		
		double eps = is_float32 ? 3.0e-7 : 3.0e-3;       // float32 vs float16
		double rms = sqrt(4 << ids);                     // rms of output array
		double epsabs = eps * rms * sqrt(ids+2);
		
		// cout << "ichunk=" << ichunk << ", ids=" << ids << ", epsabs=" << epsabs << endl;

		if (is_float32) {
		    Array<float> a = gpu_out.small_arrs[ids].data_float32.to_host();
		    assert_arrays_equal(cpu_out[ids], a, "ref", "gpu", {"beam","freq","time"}, epsabs, 0.0);  // epsrel=0
		}
		else {
		    Array<__half> a = gpu_out.small_arrs[ids].data_float16.to_host();
		    Array<float> b = a.template convert_dtype<float>();
		    assert_arrays_equal(cpu_out[ids], b, "ref", "gpu", {"beam","freq","time"}, epsabs, 0.0);  // epsrel=0
		}
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

