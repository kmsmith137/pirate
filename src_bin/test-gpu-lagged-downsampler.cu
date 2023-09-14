#include "../include/pirate/internals/GpuLaggedDownsamplingKernel.hpp"
#include "../include/pirate/internals/inlines.hpp"  // pow2()

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

    int nbeams = 0;
    int nchunks = 0;
    int nt_chunk = 0;
    
    int bstride_in = 0;
    int bstride_out = 0;
    

    Array<T> alloc_input(bool use_bstride_in, int aflags)
    {
	long nr = pow2(large_input_rank);
	long bstride = use_bstride_in ? bstride_in : min_bstride_in();
	return Array<T> ({nbeams,nr,nt_chunk}, {bstride,nt_chunk,1}, aflags);
    }

    
    vector<Array<T>> alloc_output(bool use_bstride_out, int aflags)
    {
	long bstride_min = min_bstride_out();
	long bstride = use_bstride_out ? bstride_out : bstride_min;
	long nr = pow2(large_input_rank - 1);

	Array<T> big_arr({nbeams,bstride_min}, {bstride,1}, aflags);
	vector<Array<T>> ret;
	long nt_cumul = 0;
	
	for (int i = 0; i < num_downsampling_levels; i++) {
	    long nt_ds = xdiv(nt_chunk, pow2(i+1));
	    Array<T> a = big_arr.slice(1, nr * nt_cumul, nr * (nt_cumul + nt_ds));
	    Array<T> b = a.reshape({ nbeams, nr, nt_ds });
	    
	    ret.push_back(b);
	    nt_cumul += nt_ds;
	}

	assert(nr * nt_cumul == bstride_min);
	return ret;
    }
    

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
	small_input_rank = rand_int(1, 9);
	large_input_rank = rand_int(small_input_rank, 13);
	num_downsampling_levels = rand_int(1, constants::max_downsampling_level + 1);
	nchunks = rand_int(2, 11);

	// Bytes per time sample per beam.
	int nb0 = pow2(large_input_rank) * sizeof(T);

	int nt_divisor = xdiv(128, sizeof(T)) * pow2(num_downsampling_levels+1);
	int imax = min(10, 2.0e9 / nb0 / nt_divisor);
	nt_chunk = nt_divisor * rand_int(1, imax+1);

	int bmax = min(10, 2.1e9 / nb0 / nt_chunk);
	nbeams = rand_int(1, bmax+1);
	
	bstride_in = rand_stride<T> (min_bstride_in());
	bstride_out = rand_stride<T> (min_bstride_out());
    }
    

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

	assert(small_input_rank >= );

	ReferenceLaggedDownsampler::Params ref_params;
	ref_params.small_input_rank = small_input_rank;
	ref_params.large_input_rank = large_input_rank;
	ref_params.num_downsampling_levels = num_downsampling_levels;
	ref_params.nbeams = nbeams;
	ref_params.ntime = nt_chunk;

	auto ref_kernel = make_shared<ReferenceLaggedDownsampler> (ref_params);
	auto gpu_kernel = GpuLaggedDownsamplingKernel<T>::make(small_input_rank, large_input_rank, num_downsampling_kernels);

	if (noisy) {
	    cout << "    GPU kernel params\n";
	    gpu_kernel->print(cout, 8);
	    cout << flush;
	}
	
	Array<T> gpu_in = alloc_input(true, af_gpu | af_zero);
	Array<T> gpu_state({ params.nbeams, gpu_kernel.params.state_nelts_per_beam }, af_gpu | af_zero);

	// Next task is to allocate the 'cpu_out' and 'gpu_out' arrays.
	
	vector<Array<float>> cpu_out;
	vector<Array<T>> gpu_out;
	
	int nr = pow2(large_input_rank-1);
	long nt_cumul = 0;

	Array<T> gpu_out0({ nbeams, min_bstride_out() },  // shape
			  { bstride_out, 1 },             // strides
			  af_gpu | af_zero);              // aflags
	
	for (int i = 0; i < num_downsampling_levels; i++) {
	    long nt_ds = xdiv(params.nt_chunk, pow2(i+1));
	    
	    Array<float> cpu_out1({ nbeams, nr, nt_ds }, af_rhost | af_zero);
	    cpu_out.push_back(cpu_out1);
	    
	    Array<T> gpu_out1 = gpu_out0.slice(1, nr * nt_cumul, nr * (nt_cumul + nt_ds));
	    Array<T> gpu_out2 = gpu_out1.reshape_ref({ nbeams, nr, nt_ds });
	    gpu_out.push_back(gpu_out2);
	    
	    nt_cumul += nt_ds;
	}
	
	for (int ichunk = 0; ichunk < nchunks; ichunk++) {
#if 1
	    // Random chunk gives strongest test.
	    Array<float> cpu_in = alloc_input(false, af_rhost | af_random);
#else
	    // One-hot chunk is sometimes useful for debugging.
	    Array<float> cpu_in = alloc_input(false, af_rhost | af_zero);
	    int ibeam = rand_int(0, nbeams);
	    int irow = rand_int(0, pow2(large_input_rank));
	    int it = rand_int(0, nt_chunk);
	    cout << "   one-hot chunk: ibeam=" << ibeam << "; irow=" << irow << "; it=" << it << ";" << endl;
	    ref_chunk.at({ibeam,irow,it}) = 1.0;
#endif

	    ref_kernel->apply(cpu_in, cpu_out);

	    gpu_in.fill(cpu_in.convert_dtype<T> ());
	    gpu_kernel->launch(gpu_in, gpu_out, gpu_state, ichunk * nt_chunk);
	    CUDA_CALL(cudaDeviceSynchronize());

	    for (int ids = 0; ids < num_downsampling_levels; ids++) {
		Array<float> from_gpu = gpu_out[ids].to_host().convert_dtype<float> ();
		double epsabs = 0.003 * pow(1.414, rank);
		assert_arrays_equal(ref_chunk, gpu_output, "ref", "gpu", {"beam","amb","dmbr","time"}, epsabs, 0.003);		
	    }
	}
    }
}


// -------------------------------------------------------------------------------------------------


int main(int argc, char **argv)
{
    // FIXME switch to 'false' when no longer actively developing
    const bool noisy = true;

#if 0
    // Uncomment to enable specific test
    for (int i = 0; i < 100; i++) {
	TestInstance t;
	t.small_input_rank = 2;
	t.large_input_rank = 3;
	t.num_downsampling_levels = 3;
	t.nbeams = 2;
	t.nchunks = 2;
	t.nt_chunk = 320;
	t.bstride_in = pow2(t.large_input_rank) * t.nt_chunk + 32;
	t.bstride_out = pow2(t.large_input_rank-1) * t.nt_chunk + 64;
	t.run(noisy);
    }
    return 0;
#endif
    
    for (int i = 0; i < 100; i++) {
	TestInstance t;
	t.randomize();
	t.run(noisy);
    }

    cout << "test-gpu-lagged-downsampler: pass" << endl;
    return 0;
}

