#include "../include/pirate/internals/LaggedDownsamplingKernel.hpp"
#include "../include/pirate/internals/inlines.hpp"  // pow2()
#include "../include/pirate/constants.hpp"

#include <ksgpu/Array.hpp>
#include <ksgpu/cuda_utils.hpp>
#include <ksgpu/rand_utils.hpp>    // rand_int()
#include <ksgpu/xassert.hpp>

using namespace std;
using namespace pirate;
using namespace ksgpu;


// -------------------------------------------------------------------------------------------------


struct TestInstance
{
    LaggedDownsamplingKernelParams params;

    long nchunks = 0;
    long bstride_pad_in = 0;
    long bstride_pad_out = 0;

    TestInstance() { }
    
    void print(ostream &os=cout, int indent=0) const
    {
	params.print(os, indent);
	print_kv("nchunks", this->nchunks, os, indent);
	print_kv("bstride_pad_in", this->bstride_pad_in, os, indent);
	print_kv("bstride_pad_out", this->bstride_pad_out, os, indent);
    }

    void validate() const
    {
	params.validate();
	xassert(nchunks > 0);
    }
    
    static TestInstance make_random()
    {
	TestInstance ti;
	ti.params.dtype = (rand_uniform() < 0.5) ? Dtype::native<float>() : Dtype::native<__half>();
	ti.params.small_input_rank = rand_int(2,9);  // GpuLaggedDownsamplingKernel needs small_input_rank >= 2
	ti.params.large_input_rank = ti.params.small_input_rank + rand_int(0,4);
	ti.params.num_downsampling_levels = rand_int(1, constants::max_downsampling_level);  // no +1 here

	auto v = ksgpu::random_integers_with_bounded_product(2,10);
	ti.params.total_beams = v[0] * v[1];
	ti.params.beams_per_batch = v[1];
	
	long nt_divisor = xdiv(1024, ti.params.dtype.nbits) * pow2(ti.params.num_downsampling_levels+1);
	long p = ti.params.total_beams * pow2(ti.params.large_input_rank) * nt_divisor;
	long q = (10*1000*1000) / p;
	q = max(q, 4L);

	auto w = ksgpu::random_integers_with_bounded_product(2,q);
	ti.params.ntime = nt_divisor * w[0];
	ti.nchunks = w[1];

	long n = xdiv(1024, ti.params.dtype.nbits);
	ti.bstride_pad_in = max(0L, rand_int(-2,4)) * n;
	ti.bstride_pad_out = max(0L, rand_int(-2,4)) * n;
	
	return ti;
    }
};


void test_gpu_lagged_downsampling_kernel(const TestInstance &ti)
{
    using Params = LaggedDownsamplingKernelParams;
    using Outbuf = LaggedDownsamplingKernelOutbuf;
    
    ti.params.validate();

    Params p = ti.params;
    Params ref_params = p;
    ref_params.dtype = Dtype::native<float> ();
    
    auto ref_kernel = make_shared<ReferenceLaggedDownsamplingKernel2> (ref_params);
    auto gpu_kernel = GpuLaggedDownsamplingKernel2::make(p);
    long nbatches = xdiv(p.total_beams, p.beams_per_batch);

    // Input arrays have shape (beams_per_batch, 2^(large_input_rank), ntime).
    vector<long> in_shape = { p.beams_per_batch, pow2(p.large_input_rank), p.ntime };
    vector<long> in_strides = { in_shape[1]*in_shape[2] + ti.bstride_pad_in, in_shape[2], 1 };
    Array<void> gpu_in(p.dtype, in_shape, in_strides, af_gpu);

    // Output arrays.
    Outbuf gpu_out(p);
    Outbuf cpu_out(ref_params);

    gpu_out.allocate(gpu_out.min_beam_stride + ti.bstride_pad_out, af_gpu);
    cpu_out.allocate(af_uhost);  // default bstride
    
    // Persistent state (for GPU kernel). The af_zero flag is important here.
    Array<void> gpu_pstate(p.dtype, { p.total_beams, gpu_kernel->state_nelts_per_beam }, af_gpu | af_zero);
    
    for (long ichunk = 0; ichunk < ti.nchunks; ichunk++) {
	for (long ibatch = 0; ibatch < nbatches; ibatch++) {
	    Array<float> cpu_in = Array<float> (in_shape, af_rhost | af_random);
	    ref_kernel->apply(cpu_in, cpu_out, ibatch);
	    
	    // Copy cpu_in -> gpu_in
	    array_fill(gpu_in, cpu_in.convert(p.dtype));

	    Array<void> s = gpu_pstate.slice(0, ibatch * p.beams_per_batch, (ibatch+1) * p.beams_per_batch);
	    gpu_kernel->launch(gpu_in, gpu_out, s, ichunk * p.ntime, nullptr);

	    for (int ids = 0; ids < p.num_downsampling_levels; ids++) {
		// Note: if the definition of the LaggedDownsampler changes to include factors of 0.5,
		// then the value of 'rms' may ned to change.
		
		double rms = sqrt(4 << ids);    // rms of output array
		double epsabs = 3 * p.dtype.precision() * rms * sqrt(ids+2);
		
		// cout << "ichunk=" << ichunk << ", ids=" << ids << ", epsabs=" << epsabs << endl;
		assert_arrays_equal(cpu_out.small_arrs.at(ids), gpu_out.small_arrs.at(ids),
				    "ref", "gpu", {"beam","freq","time"}, epsabs, 0.0);  // epsrel=0
	    }
	}
    }
}


// -------------------------------------------------------------------------------------------------


int main(int argc, char **argv)
{
    const int ntests = 50;
    
    for (int i = 0; i < ntests; i++) {
	cout << "\ntest_gpu_lagged_downsampling_kernel " << i << "/" << ntests << "\n";
	TestInstance ti = TestInstance::make_random();
	ti.print(cout, 4);
	test_gpu_lagged_downsampling_kernel(ti);
    }

    cout << "\ntest_gpu_lagged_downsampling_kernel: pass\n";
    return 0;
}

