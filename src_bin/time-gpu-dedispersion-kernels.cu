#include <iostream>
#include <gputils/Array.hpp>
#include <gputils/CudaStreamPool.hpp>

#include "../include/pirate/internals/inlines.hpp"  // pow2()
#include "../include/pirate/internals/GpuDedispersionKernel.hpp"

using namespace std;
using namespace gputils;
using namespace pirate;


// FIXME delete after de-templating.
template<typename T> struct _is_float32 { };
template<> struct _is_float32<float>   { static constexpr bool value = true; };
template<> struct _is_float32<__half>  { static constexpr bool value = false; };


template<typename T>
static void time_gpu_dedispersion_kernel(int rank, bool apply_input_residual_lags)
{
    long nstreams = 1;
    long ncallbacks = 10;
    long nambient = 256;
    long nbeams = pow2(12-rank) / sizeof(T);
    long ntime = 2048;
    long niter = 20;
    
    constexpr bool is_float32 = _is_float32<T>::value;
    typename GpuDedispersionKernel::Params params;
    params.dtype = is_float32 ? "float32" : "float16";
    params.rank = rank;
    params.nambient = nambient;
    params.total_beams = nbeams;
    params.apply_input_residual_lags = apply_input_residual_lags;
    params.input_is_downsampled_tree = false;  // shouldn't affect timing
    params.nelts_per_segment = is_float32 ? 32 : 64;

    vector<int> itime(nstreams, 0);
    vector<shared_ptr<GpuDedispersionKernel>> kernels(nstreams);
    Array<T> iobuf({nstreams, nbeams, nambient, pow2(rank), ntime}, af_zero | af_gpu);

    for (int i = 0; i < nstreams; i++)
	kernels[i] = GpuDedispersionKernel::make(params);
    
    long iobuf_bytes_per_stream = nbeams * nambient * pow2(rank) * ntime * sizeof(T);
    long rstate_bytes_per_stream = nbeams * kernels[0]->state_nelts_per_beam * sizeof(T);
    double gmem_gb = 2.0e-9 * niter * (iobuf_bytes_per_stream + rstate_bytes_per_stream);  // factor 2 from read+write
    
    auto callback = [&](const CudaStreamPool &pool, cudaStream_t stream, int istream)
        {
	    Array<T> iobuf_s = iobuf.slice(0, istream);
    
	    UntypedArray ubuf;
	    if constexpr (is_float32)
		ubuf.data_float32 = iobuf_s;
	    else
		ubuf.data_float16 = iobuf_s;

	    for (int i = 0; i < niter; i++) {
		kernels[istream]->launch(ubuf, ubuf, itime[istream], 0, stream);
		itime[istream]++;
	    }
	};
    
    stringstream kernel_name;
    kernel_name << "dedisperse(" << gputils::type_name<T>() << ", rank=" << rank
		<< ", apply_input_residual_lags=" << (apply_input_residual_lags ? "true" : "false");
    
    CudaStreamPool pool(callback, ncallbacks, nstreams, kernel_name.str());
    pool.monitor_throughput("global memory (GB/s)", gmem_gb);
    pool.run();
}


int main(int argc, char **argv)
{
    for (int rank = 1; rank <= 8; rank++) {
	for (bool apply_input_residual_lags: { false, true }) {
	    time_gpu_dedispersion_kernel<float> (rank, apply_input_residual_lags);
	    time_gpu_dedispersion_kernel<__half> (rank, apply_input_residual_lags);
	}
    }
    
    return 0;
}
