#include <iostream>
#include <ksgpu/Array.hpp>
#include <ksgpu/CudaStreamPool.hpp>

#include "../include/pirate/timing.hpp"
#include "../include/pirate/inlines.hpp"  // pow2()
#include "../include/pirate/DedispersionBuffer.hpp"
#include "../include/pirate/LaggedDownsamplingKernel.hpp"

using namespace std;
using namespace ksgpu;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// FIXME this function is cut-and-paste from test-gpu-lagged-downsampler.cu.
// Move to src_lib/ somewhere?
static DedispersionBuffer make_buffer(const LaggedDownsamplingKernelParams &lds_params, int aflags)
{
    DedispersionBufferParams dd_params;
    dd_params.dtype = lds_params.dtype;
    dd_params.beams_per_batch = lds_params.beams_per_batch;
    dd_params.nbuf = lds_params.num_downsampling_levels;

    for (long ids = 0; ids < lds_params.num_downsampling_levels; ids++) {
	long rk = lds_params.input_total_rank - (ids ? 1 : 0);
	long nt = xdiv(lds_params.ntime, pow2(ids));
	dd_params.buf_rank.push_back(rk);
	dd_params.buf_ntime.push_back(nt);
    }

    dd_params.validate();
    
    DedispersionBuffer buf(dd_params);
    buf.allocate(aflags);
    return buf;
}


static void time_gpu_lagged_downsampling_kernel(const LaggedDownsamplingKernelParams &params)
{
    // Use one cuda stream per batch of beams.
    long nb_tot = params.total_beams;
    long nb_batch = params.beams_per_batch;
    long nstreams = xdiv(nb_tot, nb_batch);
    long ncallbacks = 10;
    long niter = 20;
    
    long ST = xdiv(params.dtype.nbits, 8);  // sizeof(T)
    shared_ptr<GpuLaggedDownsamplingKernel> kernel = GpuLaggedDownsamplingKernel::make(params);
    kernel->allocate();
    
    vector<DedispersionBuffer> bufs;
    for (long s = 0; s < nstreams; s++)
	bufs.push_back(make_buffer(params, af_zero | af_gpu));
    
    long buf_nelts_per_stream = bufs[0].ref.size;
    long pstate_nelts_per_stream = nb_batch * kernel->state_nelts_per_beam;
    long footprint_nelts_per_stream = buf_nelts_per_stream + pstate_nelts_per_stream;
    long footprint_nbytes = nstreams * footprint_nelts_per_stream * ST;

    // All global memory bandwidths are in GB
    double gmem_buf = 1.0e-9 * niter * buf_nelts_per_stream * ST;
    double gmem_pstate = 2.0e-9 * niter * pstate_nelts_per_stream * ST;  // note factor 2 here
    double gmem_tot = gmem_buf + gmem_pstate;
    double pstate_overhead_percentage = 100. * gmem_pstate / gmem_tot;
    
    stringstream kernel_name;
    kernel_name << "gpu_lagged_downsample("
		<< "dtype=" << params.dtype
		<< ", input_total_rank=" << params.input_total_rank
		<< ", output_dd_rank=" << params.output_dd_rank
		<< ", num_downsampling_levels=" << params.num_downsampling_levels
		<< ", pstate_overhead = " << pstate_overhead_percentage << "%"
		<< ")";

    cout << "\n" << kernel_name.str() << "\n";
    kernel->print(cout, 4);  // indent=4

    cout << "    niter = " << niter << endl
	 << "    gpu memory footprint = " << ksgpu::nbytes_to_str(footprint_nbytes) << endl;

    auto callback = [&](const CudaStreamPool &pool, cudaStream_t stream, int istream)
        {
	    DedispersionBuffer &buf = bufs.at(istream);
	    for (int i = 0; i < niter; i++)
		kernel->launch(buf, istream, i, stream);
	};
    
    CudaStreamPool pool(callback, ncallbacks, nstreams, kernel_name.str());
    pool.monitor_throughput("global memory (GB/s)", gmem_tot);
    pool.run();
}


void time_gpu_lagged_downsampling_kernels()
{
    for (int num_downsampling_levels: {2,6}) {
	for (Dtype dtype: { Dtype::native<float>(), Dtype::native<__half>() }) {
	    LaggedDownsamplingKernelParams params;
	    params.dtype = dtype;
	    params.input_total_rank = 16;
	    params.output_dd_rank = 7;
	    params.num_downsampling_levels = num_downsampling_levels;
	    params.total_beams = 4;
	    params.beams_per_batch = 4;
	    params.ntime = 2048;
    
	    time_gpu_lagged_downsampling_kernel(params);
	}
    }
}


}  // namespace pirate
