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
//
// ReferenceLaggedDownsampler
//
// FIXME move to ReferenceDedisperser.cu?
// FIXME make float16/float32 versions?
// FIXME integrate with ReferenceDedisperser?


struct ReferenceLaggedDownsampler
{
    struct Params
    {
	int small_input_rank = 0;
	int large_input_rank = 0;
	int num_downsampling_levels = 0;
	int nbeams = 0;
	long ntime = 0;
    };

    const Params params;

    std::shared_ptr<ReferenceLagbuf> lagbuf_small;
    std::shared_ptr<ReferenceLagbuf> lagbuf_large;
    std::shared_ptr<ReferenceLaggedDownsampler> next;

    
    ReferenceLaggedDownsampler(const Params &params_)
	: params(params_)
    {
	assert(params.small_input_rank >= 2);
	assert(params.small_input_rank <= 8);
	assert(params.large_input_rank >= params.small_input_rank);
	assert(params.large_input_rank <= constants::max_tree_rank);
	assert(params.num_downsampling_levels > 0);
	assert(params.num_downsampling_levels <= constants::max_downsampling_level);
	assert(params.nbeams > 0);
	assert(params.ntime > 0);
	assert((params.ntime % pow2(params.num_downsampling_levels)) == 0);

	int r = params.large_input_rank;
	int s = params.small_input_rank;
	
	vector<int> small_lags(pow2(r));
	vector<int> large_lags(pow2(r-1));
	
	for (int i = 0; i < pow2(r); i++)
	    small_lags[i] = (i & 1) ? 0 : 1;

	for (int i = 0; i < pow2(r-s-1); i++)
	    for (int j = 0; j < pow2(s-1); j++)
		large_lags[i*pow2(s-1)+j] = pow2(s-1)-j-1;

	this->lagbuf_small = make_shared<ReferenceLagbuf> (small_lags, params.ntime/2);
	this->lagbuf_large = make_shared<ReferenceLagbuf> (large_lags, params.ntime/2);

	if (params.num_downsampling_levels == 1)
	    return;
	
	Params next_params;
	next_params.small_input_rank = params.small_input_rank;
	next_params.large_input_rank = params.large_input_rank;
	next_params.num_downsampling_levels = params.num_downsampling_levels - 1;
	next_params.ntime = xdiv(params.ntime, 2);
	next_params.nbeams = nbeams;

	this->next = make_shared<ReferenceLaggedDownsampler> (next_params);
    }

    void apply(const Array<float> &in, vector<Array<float>> &out)
    {
	assert(out.size() == num_downsampling_levels);
	this->apply(in, &out[0]);
    }

    void apply(const Array<float> &in, Array<float> *outp)
    {
	int r = params.large_input_rank;
	int nbeams = params.nbeams;
	long ntime = params.ntime;

	assert(in.shape_equals({ nbeams, pow2(r), ntime }));
	assert(outp[0].shape_equals({ nbeams, pow2(r-1), xdiv(ntime,2) }));

	// Input/output arrays, reshaped to 2-d.
	Array<float> in_2d = in.reshape_ref({ nbeams * pow2(r), ntime });
	Array<float> out_2d = outp[0].reshape_ref({ nbeams * pow2(r-1), ntime/2 });

	// Reshaped time-downsampled input array: (nbeams * 2^r, ntime/2)
	Array<float> in_ds({ nbeams * pow2(r), ntime/2 }, af_uhost | af_zero);
	reference_downsample_time(in_2d, in_ds, false);  // normalize=false, i.e. sum with no factor 0.5

	// Apply "small" lags (one-sample lags in even channels), before frequency downsampling.
	Array<float> in_ds2 = in_ds.clone();   // copy since we'll need 'in_ds' later.
	lagbuf_small->apply_lags(in_ds2);

	// Downsample in frequency, and apply "large" lags.
	reference_downsample_freq(in_ds2, out_2d, false);   // normalize=false, i.e. sum with no factor 0.5
	lagbuf_large->apply_lags(out_2d);

	if (params.num_downsampling_levels == 1)
	    return;
	
	// Recurse to next downsampling level.
	in_ds = in_ds.reshape_ref({ nbeams, pow2(r), ntime/2 });
	next->apply(in_ds, outp+1);
    }
};


// -------------------------------------------------------------------------------------------------


template<typename T>
static long rand_stride(long min_stride)
{
    // FIXME allow negative strides
    long n = (rand_uniform() < 0.5) ? 0 : rand_int(1,100);
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
	small_input_rank = rand_int(2, 9);
	large_input_rank = small_input_rank + rand_int(0, 9);
	num_downsampling_levels = rand_int(1, constants::max_downsampling_level + 1);
	nchunks = rand_int(2, 11);

	// Bytes per time sample per beam.
	int nb0 = pow2(large_input_rank) * sizeof(T);

	int nt_divisor = xdiv(128, sizeof(T)) * pow2(num_downsampling_levels+1);
	int imax = min(10, 2.0e9 / nb0 / nt_divisor);
	nt_chunk = nt_divisor * rand_int(1, imax+1);

	int bmax = min(10, 2.1e9 / nb0 / nt_chunk);
	nbeams = rand_int(1, bmax+1);
	
	bstride_in = rand_stride(min_bstride_in());
	bstride_out = rand_stride(min_bstride_out());
    }


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
	long nt_cumul = 0;

	Array<T> big_arr({nbeams,bstride_min}, {bstride,1}, aflags);
	vector<Array<T>> ret;
	
	for (int i = 0; i < num_downsampling_levels; i++) {
	    long nt_ds = xdiv(nt_chunk, pow2(i+1));
	    Array<T> a = big_arr.slice(1, nr * nt_cumul, nr * (nt_cumul + nt_ds));
	    Array<T> b = a.reshape({ nbeams, nr, nt_ds });
	    
	    ret.push_back(b);
	    nt_cumul += nt_ds;
	}

	return ret;
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
	t.small_input_rank = 8;
	t.nchunks = 2;
	t.nbeams = 2;
	t.ntime = 320;
	t.run(noisy);
    }
    return 0;
#endif
    
    for (int i = 0; i < 100; i++) {
	TestInstance t;
	t.randomize();
	t.run(noisy);
    }

    cout << "test-gpu-dedispersion-kernels: pass" << endl;
    return 0;
}

