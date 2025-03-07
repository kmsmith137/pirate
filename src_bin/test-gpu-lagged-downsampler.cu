#include "../include/pirate/internals/LaggedDownsamplingKernel.hpp"
#include "../include/pirate/internals/DedispersionBuffers.hpp"
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

    TestInstance() { }
    
    void print(ostream &os=cout, int indent=0) const
    {
	params.print(os, indent);
	print_kv("nchunks", this->nchunks, os, indent);
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
	
	return ti;
    }
};


static DedispersionBuffer make_buffer(const LaggedDownsamplingKernelParams &lds_params, int aflags)
{
    DedispersionBufferParams dd_params;
    dd_params.dtype = lds_params.dtype;
    dd_params.beams_per_batch = lds_params.beams_per_batch;
    dd_params.nbuf = lds_params.num_downsampling_levels;

    for (long ids = 0; ids < lds_params.num_downsampling_levels; ids++) {
	long rk = lds_params.large_input_rank - (ids ? 1 : 0);
	long nt = xdiv(lds_params.ntime, pow2(ids));
	dd_params.buf_rank.push_back(rk);
	dd_params.buf_ntime.push_back(nt);
    }

    dd_params.validate();
    
    DedispersionBuffer buf(dd_params);
    buf.allocate(aflags);
    return buf;
}


void test_gpu_lagged_downsampling_kernel(const TestInstance &ti)
{
    ti.params.validate();
    
    LaggedDownsamplingKernelParams p = ti.params;
    long nbatches = xdiv(p.total_beams, p.beams_per_batch);
    long nb = p.beams_per_batch;
    long rk = p.large_input_rank;
    long nt = p.ntime;
    
    LaggedDownsamplingKernelParams ref_params = p;
    ref_params.dtype = Dtype::native<float> ();
    
    auto ref_kernel = make_shared<ReferenceLaggedDownsamplingKernel> (ref_params);
    auto gpu_kernel = GpuLaggedDownsamplingKernel::make(p);
    gpu_kernel->allocate();
    
    DedispersionBuffer gpu_buf = make_buffer(p, af_gpu);
    DedispersionBuffer cpu_buf = make_buffer(ref_params, af_uhost);
    
    for (long ichunk = 0; ichunk < ti.nchunks; ichunk++) {
	for (long ibatch = 0; ibatch < nbatches; ibatch++) {
	    // Copy (random chunk) -> (cpu_in) -> (gpu_in).
	    Array<float> r({nb, pow2(rk), nt}, af_rhost | af_random);
	    array_fill(cpu_buf.bufs[0], r);
	    array_fill(gpu_buf.bufs[0], r.convert(p.dtype));

	    // Run GPU and reference kernels.
	    ref_kernel->apply(cpu_buf, ibatch);
	    gpu_kernel->launch(gpu_buf, ibatch, ichunk, nullptr);

	    // Compare results.
	    for (int ids = 1; ids < p.num_downsampling_levels; ids++) {
		// Note: if the definition of the LaggedDownsampler changes to include
		// factors of 0.5, then the value of 'rms' may need to change.
		
		double rms = sqrt(2 << ids);    // rms of output array
		double epsabs = 3 * p.dtype.precision() * rms * sqrt(ids+1);
		
		assert_arrays_equal(cpu_buf.bufs.at(ids),
				    gpu_buf.bufs.at(ids),
				    "ref", "gpu", {"beam","freq","time"},
				    epsabs, 0.0);  // epsrel=0
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

