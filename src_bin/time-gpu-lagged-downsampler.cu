#include <iostream>
#include <ksgpu/Array.hpp>
#include <ksgpu/CudaStreamPool.hpp>

#include "../include/pirate/internals/inlines.hpp"  // pow2()
#include "../include/pirate/internals/LaggedDownsamplingKernel.hpp"

using namespace std;
using namespace ksgpu;
using namespace pirate;


static void time_gpu_dedispersion_kernel(const LaggedDownsamplingKernelParams &params)
{
    using Outbuf = LaggedDownsamplingKernelOutbuf;
    
    // Use one cuda stream per batch of beams.
    long nb_tot = params.total_beams;
    long nb_batch = params.beams_per_batch;
    long nstreams = xdiv(nb_tot, nb_batch);
    long ncallbacks = 10;
    long niter = 20;
    
    long ST = xdiv(params.dtype.nbits, 8);  // sizeof(T)
    shared_ptr<GpuLaggedDownsamplingKernel> kernel = GpuLaggedDownsamplingKernel::make(params);
    kernel->allocate();
    
    Array<void> in(params.dtype, { nb_tot, pow2(params.large_input_rank), params.ntime }, af_gpu | af_zero);
    vector<Outbuf> outbufs;
    
    for (long s = 0; s < nstreams; s++)
	outbufs.push_back(Outbuf(params));
    for (long s = 0; s < nstreams; s++)
	outbufs[s].allocate(af_gpu);
    
    long out_nelts_per_stream = outbufs[0].big_arr.size;
    long in_nelts_per_stream = nb_batch * pow2(params.large_input_rank) * params.ntime;
    long pstate_nelts_per_stream = nb_batch * kernel->state_nelts_per_beam;
    long footprint_nelts_per_stream = out_nelts_per_stream + in_nelts_per_stream + pstate_nelts_per_stream;
    long footprint_nbytes = nstreams * footprint_nelts_per_stream * ST;

    // All global memory bandwidths are in GB
    double gmem_in = 1.0e-9 * niter * in_nelts_per_stream * ST;
    double gmem_out = 1.0e-9 * niter * out_nelts_per_stream * ST;
    double gmem_pstate = 2.0e-9 * niter * pstate_nelts_per_stream * ST;  // note factor 2 here
    double gmem_tot = gmem_in + gmem_out + gmem_pstate;
    double pstate_overhead_percentage = 100. * gmem_pstate / gmem_tot;
    
    stringstream kernel_name;
    kernel_name << "lagged_downsample("
		<< "dtype=" << params.dtype
		<< ", small_input_rank=" << params.small_input_rank
		<< ", large_input_rank=" << params.large_input_rank
		<< ", num_downsampling_levels=" << params.num_downsampling_levels
		<< ", pstate_overhead = " << pstate_overhead_percentage << "%"
		<< ")";

    cout << "\n" << kernel_name.str();
    kernel->print(cout, 4);  // indent=4

    cout << "    niter = " << niter << endl
	 << "    gpu memory footprint = " << ksgpu::nbytes_to_str(footprint_nbytes) << endl;

    auto callback = [&](const CudaStreamPool &pool, cudaStream_t stream, int istream)
        {
	    Array<void> in_s = in.slice(0, istream * nb_batch, (istream+1) * nb_batch);
	    Outbuf &outbuf_s = outbufs.at(istream);

	    for (int i = 0; i < niter; i++)
		kernel->launch(in_s, outbuf_s, istream, i, stream);
	};
    
    CudaStreamPool pool(callback, ncallbacks, nstreams, kernel_name.str());
    pool.monitor_throughput("global memory (GB/s)", gmem_tot);
    pool.run();
}


int main(int argc, char **argv)
{
    for (int num_downsampling_levels: {1,3,5}) {
	for (Dtype dtype: { Dtype::native<float>(), Dtype::native<__half>() }) {
	    LaggedDownsamplingKernelParams params;
	    params.dtype = dtype;
	    params.small_input_rank = 8;
	    params.large_input_rank = 16;
	    params.num_downsampling_levels = num_downsampling_levels;
	    params.total_beams = 4;
	    params.beams_per_batch = 4;
	    params.ntime = 2048;
    
	    time_gpu_dedispersion_kernel(params);
	}
    }
    
    return 0;
}
