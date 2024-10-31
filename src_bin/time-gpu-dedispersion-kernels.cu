#include <iostream>
#include <gputils/Array.hpp>
#include <gputils/CudaStreamPool.hpp>

#include "../include/pirate/internals/inlines.hpp"  // pow2()
#include "../include/pirate/internals/GpuDedispersionKernel.hpp"

using namespace std;
using namespace gputils;
using namespace pirate;


static void time_gpu_dedispersion_kernel(const string &dtype, int rank, bool apply_input_residual_lags)
{
    long nstreams = 1;
    long ncallbacks = 10;
    long nambient = 256;
    long nbeams = pow2(10-rank);
    long ntime = 2048;
    long niter = 5;
    
    GpuDedispersionKernel::Params params;
    params.dtype = dtype;
    params.rank = rank;
    params.nambient = nambient;
    params.total_beams = nbeams;
    params.beams_per_kernel_launch = nbeams;
    params.ntime = ntime;
    params.apply_input_residual_lags = apply_input_residual_lags;
    params.input_is_downsampled_tree = false;  // shouldn't affect timing

    bool is_float32 = params.is_float32();
    params.nelts_per_segment = is_float32 ? 32 : 64;

    vector<shared_ptr<GpuDedispersionKernel>> kernels(nstreams);
    vector<UntypedArray> ubufs(nstreams);
    vector<int> itime(nstreams, 0);

    for (int i = 0; i < nstreams; i++)
	kernels[i] = GpuDedispersionKernel::make(params);

    vector<ssize_t> shape = { nstreams, nbeams, nambient, pow2(rank), ntime };
    
    if (is_float32) {
	Array<float> x(shape, af_gpu | af_zero);
	for (int i = 0; i < nstreams; i++)
	    ubufs[i].data_float32 = x.slice(0,i);
    }
    else {
	Array<__half> x(shape, af_gpu | af_zero);
	for (int i = 0; i < nstreams; i++)
	    ubufs[i].data_float16 = x.slice(0,i);
    }

    long elt_size = is_float32 ? 4 : 2;
    long iobuf_bytes_per_stream = nbeams * nambient * pow2(rank) * ntime * elt_size;
    long rstate_bytes_per_stream = nbeams * kernels[0]->state_nelts_per_beam * elt_size;
    double gmem_gb = 2.0e-9 * niter * (iobuf_bytes_per_stream + rstate_bytes_per_stream);  // factor 2 from read+write
    
    auto callback = [&](const CudaStreamPool &pool, cudaStream_t stream, int istream)
        {
	    for (int i = 0; i < niter; i++) {
		kernels[istream]->launch(ubufs[istream], ubufs[istream], itime[istream], 0, stream);
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
    for (int rank = 1; rank <= 8; rank++) {
	for (bool apply_input_residual_lags: { false, true }) {
	    for (string dtype: { "float32", "float16" }) {
		time_gpu_dedispersion_kernel(dtype, rank, apply_input_residual_lags);
		time_gpu_dedispersion_kernel(dtype, rank, apply_input_residual_lags);
	    }
	}
    }
    
    return 0;
}
