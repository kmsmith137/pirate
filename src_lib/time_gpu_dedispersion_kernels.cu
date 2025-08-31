#include <iostream>
#include <ksgpu/Array.hpp>
#include <ksgpu/CudaStreamPool.hpp>

#include "../include/pirate/timing.hpp"
#include "../include/pirate/inlines.hpp"  // pow2()
#include "../include/pirate/DedispersionKernel.hpp"

using namespace std;
using namespace ksgpu;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


static void time_gpu_dedispersion_kernel(Dtype dtype, int dd_rank, bool apply_input_residual_lags)
{
    long nstreams = 1;
    long ncallbacks = 10;
    long amb_rank = 8;
    long nbeams = pow2(10-dd_rank);
    long ntime = 2048;
    long niter = 5;

    DedispersionKernelParams params;
    params.dtype = dtype;
    params.dd_rank = dd_rank;
    params.amb_rank = amb_rank;
    params.total_beams = nbeams;
    params.beams_per_batch = nbeams;
    params.ntime = ntime;
    params.apply_input_residual_lags = apply_input_residual_lags;
    params.input_is_downsampled_tree = false;  // shouldn't affect timing
    params.nelts_per_segment = xdiv(1024, dtype.nbits);

    vector<shared_ptr<NewGpuDedispersionKernel>> kernels(nstreams);
    vector<int> itime(nstreams, 0);

    for (int i = 0; i < nstreams; i++) {
	kernels[i] = make_shared<NewGpuDedispersionKernel> (params);
	kernels[i]->allocate();
    }

    Array<void> buf(dtype, { nstreams, nbeams, pow2(amb_rank), pow2(dd_rank), ntime }, af_gpu | af_zero);

    long elt_size = xdiv(dtype.nbits, 8);
    long iobuf_bytes_per_stream = nbeams * pow2(amb_rank + dd_rank) * ntime * elt_size;
    long rstate_bytes_per_stream = nbeams * pow2(amb_rank) * kernels[0]->registry_value.pstate32_per_small_tree * 4;
    double gmem_gb = 2.0e-9 * niter * (iobuf_bytes_per_stream + rstate_bytes_per_stream);  // factor 2 from read+write
    
    auto callback = [&](const CudaStreamPool &pool, cudaStream_t stream, int istream)
        {
	    Array<void> x = buf.slice(0, istream);
	    for (int i = 0; i < niter; i++) {
		kernels[istream]->launch(x, x, 0, itime[istream], stream);
		itime[istream]++;
	    }
	};
    
    stringstream kernel_name;
    kernel_name << "gpu_dedisperse(dtype=" << dtype << ", dd_rank=" << dd_rank
		<< ", apply_input_residual_lags=" << (apply_input_residual_lags ? "true" : "false");
    
    CudaStreamPool pool(callback, ncallbacks, nstreams, kernel_name.str());
    pool.monitor_throughput("global memory (GB/s)", gmem_gb);
    pool.run();
}


void time_gpu_dedispersion_kernels()
{
    for (int dd_rank = 1; dd_rank <= 8; dd_rank++)
	for (bool apply_input_residual_lags: { false, true })
	    for (Dtype dtype: { Dtype::native<float>(), Dtype::native<__half>() })
		time_gpu_dedispersion_kernel(dtype, dd_rank, apply_input_residual_lags);
}


}  // namespace pirate
