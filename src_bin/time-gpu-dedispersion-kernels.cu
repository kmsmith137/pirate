#include <iostream>
#include <gputils/Array.hpp>
#include <gputils/CudaStreamPool.hpp>

#include "../include/pirate/internals/inlines.hpp"  // pow2()
#include "../include/pirate/internals/GpuDedispersionKernel.hpp"

using namespace std;
using namespace gputils;
using namespace pirate;


template<typename T>
static void time_gpu_dedispersion_kernel(int rank, typename GpuDedispersionKernel<T>::RLagType rlag_type)
{
    shared_ptr<GpuDedispersionKernel<T>> kernel = GpuDedispersionKernel<T>::make(rank, rlag_type);
    
    long nstreams = 1;
    long ncallbacks = 10;
    long nambient = 256;
    long nbeams = pow2(12-rank) / sizeof(T);
    long ntime = 2048;
    long niter = 20;

    Array<T> iobuf({nstreams, nbeams, nambient, pow2(rank), ntime}, af_zero | af_gpu);
    Array<T> rstate({nstreams, nbeams, nambient, kernel->params.state_nelts_per_small_tree}, af_zero | af_gpu);
    
    long iobuf_bytes_per_stream = nbeams * nambient * pow2(rank) * ntime * sizeof(T);
    long rstate_bytes_per_stream = nbeams * nambient * kernel->params.state_nelts_per_small_tree * sizeof(T);
    double gmem_gb = 2.0e-9 * niter * (iobuf_bytes_per_stream + rstate_bytes_per_stream);  // factor 2 from read+write
    
    auto callback = [&](const CudaStreamPool &pool, cudaStream_t stream, int istream)
        {
	    Array<T> iobuf_s = iobuf.slice(0, istream);
	    Array<T> rstate_s = rstate.slice(0, istream);

	    for (int i = 0; i < niter; i++)
		kernel->launch(iobuf_s, rstate_s);
	};
    
    stringstream kernel_name;
    kernel_name << "dedisperse(" << gputils::type_name<T>() << ", rank=" << rank
		<< ", " << GpuDedispersionKernel<T>::rlag_str(rlag_type) << ")";
    
    CudaStreamPool pool(callback, ncallbacks, nstreams, kernel_name.str());
    pool.monitor_throughput("global memory (GB/s)", gmem_gb);
    pool.run();
}


int main(int argc, char **argv)
{
    for (int rank = 1; rank <= 8; rank++) {
	time_gpu_dedispersion_kernel<float> (rank, GpuDedispersionKernel<float>::RLagNone);
	time_gpu_dedispersion_kernel<float> (rank, GpuDedispersionKernel<float>::RLagInput);
	time_gpu_dedispersion_kernel<__half> (rank, GpuDedispersionKernel<__half>::RLagNone);
	time_gpu_dedispersion_kernel<__half> (rank, GpuDedispersionKernel<__half>::RLagInput);
    }
    
    return 0;
}
