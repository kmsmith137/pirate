#include <iostream>
#include <ksgpu/Array.hpp>
#include <ksgpu/CudaStreamPool.hpp>

#include "../include/pirate/internals/inlines.hpp"  // pow2()
#include "../include/pirate/internals/DedispersionKernel.hpp"

using namespace std;
using namespace ksgpu;
using namespace pirate;


static void time_gpu_dedispersion_kernel(Dtype dtype, int rank, bool apply_input_residual_lags)
{
    long nstreams = 1;
    long ncallbacks = 10;
    long nambient = 256;
    long nbeams = pow2(10-rank);
    long ntime = 2048;
    long niter = 5;

    DedispersionKernelParams params;
    params.dtype = dtype;
    params.rank = rank;
    params.nambient = nambient;
    params.total_beams = nbeams;
    params.beams_per_batch = nbeams;
    params.ntime = ntime;
    params.apply_input_residual_lags = apply_input_residual_lags;
    params.input_is_downsampled_tree = false;  // shouldn't affect timing
    params.nelts_per_segment = xdiv(1024, dtype.nbits);

    vector<shared_ptr<GpuDedispersionKernel>> kernels(nstreams);
    vector<int> itime(nstreams, 0);

    for (int i = 0; i < nstreams; i++)
	kernels[i] = GpuDedispersionKernel::make(params);

    Array<void> buf(dtype, { nstreams, nbeams, nambient, pow2(rank), ntime }, af_gpu | af_zero);

    long elt_size = xdiv(dtype.nbits, 8);
    long iobuf_bytes_per_stream = nbeams * nambient * pow2(rank) * ntime * elt_size;
    long rstate_bytes_per_stream = nbeams * kernels[0]->state_nelts_per_beam * elt_size;
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
    kernel_name << "dedisperse(dtype=" << dtype << ", rank=" << rank
		<< ", apply_input_residual_lags=" << (apply_input_residual_lags ? "true" : "false");
    
    CudaStreamPool pool(callback, ncallbacks, nstreams, kernel_name.str());
    pool.monitor_throughput("global memory (GB/s)", gmem_gb);
    pool.run();
}


int main(int argc, char **argv)
{
    for (int rank = 1; rank <= 8; rank++)
	for (bool apply_input_residual_lags: { false, true })
	    for (Dtype dtype: { Dtype::native<float>(), Dtype::native<__half>() })
		time_gpu_dedispersion_kernel(dtype, rank, apply_input_residual_lags);
    
    return 0;
}
