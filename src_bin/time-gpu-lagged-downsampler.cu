#include <iostream>
#include <gputils/Array.hpp>
#include <gputils/CudaStreamPool.hpp>

#include "../include/pirate/internals/inlines.hpp"  // pow2()
#include "../include/pirate/internals/GpuLaggedDownsamplingKernel.hpp"

using namespace std;
using namespace gputils;
using namespace pirate;


// T = float or __half (not __half2)
template<typename T>
static void time_gpu_dedispersion_kernel(int small_input_rank, int large_input_rank, int num_downsampling_levels)
{
    long nstreams = 1;
    long ncallbacks = 10;
    long nt_chunk = 2048;
    long nbeams = 4;
    long niter = 20;

    typename GpuLaggedDownsamplingKernel<T>::Params gpu_params;
    gpu_params.small_input_rank = small_input_rank;
    gpu_params.large_input_rank = large_input_rank;
    gpu_params.num_downsampling_levels = num_downsampling_levels;

    shared_ptr<GpuLaggedDownsamplingKernel<T>> kernel = make_shared<GpuLaggedDownsamplingKernel<T>> (gpu_params);

    long ninner = 0;
    for (int ids = 0; ids < num_downsampling_levels; ids++)
	ninner += pow2(large_input_rank-1) * xdiv(nt_chunk, pow2(ids+1));
    
    long out_nelts_per_stream = nbeams * ninner;
    long in_nelts_per_stream = nbeams * pow2(large_input_rank) * nt_chunk;
    long pstate_nelts_per_stream = nbeams * kernel->state_nelts_per_beam;
    long footprint_nelts_per_stream = out_nelts_per_stream + in_nelts_per_stream + pstate_nelts_per_stream;
    long footprint_nbytes = nstreams * footprint_nelts_per_stream * sizeof(T);

    // All global memory bandwidths are in GB
    double gmem_in = 1.0e-9 * niter * in_nelts_per_stream * sizeof(T);
    double gmem_out = 1.0e-9 * niter * out_nelts_per_stream * sizeof(T);
    double gmem_pstate = 2.0e-9 * niter * pstate_nelts_per_stream * sizeof(T);  // note factor 2 here
    double gmem_tot = gmem_in + gmem_out + gmem_pstate;
    double pstate_overhead_percentage = 100. * gmem_pstate / gmem_tot;
    
    stringstream kernel_name;
    kernel_name << "lagged_downsample(" << gputils::type_name<T>()
		<< ", small_input_rank=" << small_input_rank
		<< ", large_input_rank=" << large_input_rank
		<< ", num_downsampling_levels=" << num_downsampling_levels
		<< ", pstate_overhead = " << pstate_overhead_percentage << "%"
		<< ")\n";

    cout << kernel_name.str();
    kernel->print(cout, 4);  // indent=4

    cout << "    nstreams = " << nstreams << endl
	 << "    nt_chunk = " << nt_chunk << endl
	 << "    nbeams = " << nbeams << endl
	 << "    niter = " << niter << endl
	 << "    gpu memory footprint = " << gputils::nbytes_to_str(footprint_nbytes) << endl;

    gputils::Array<T> in({ nstreams, nbeams, pow2(large_input_rank), nt_chunk }, af_gpu | af_zero);
    gputils::Array<T> pstate({ nstreams, nbeams, kernel->state_nelts_per_beam }, af_gpu | af_zero);
    gputils::Array<T> out_flattened({ nstreams, nbeams, ninner }, af_gpu | af_zero);

    auto callback = [&](const CudaStreamPool &pool, cudaStream_t stream, int istream)
        {
	    Array<T> in_s = in.slice(0, istream);
	    Array<T> pstate_s = pstate.slice(0, istream);
	    Array<T> outf_s = out_flattened.slice(0, istream);

	    vector<Array<T>> out(num_downsampling_levels);
	    long nr = 1 << (large_input_rank-1);
	    long nt_cumul = 0;
	    
	    for (int ids = 0; ids < num_downsampling_levels; ids++) {
		long nt_ds = nt_chunk >> (ids+1);
		Array<T> a = outf_s.slice(1, nr * nt_cumul, nr * (nt_cumul + nt_ds));
		out[ids] = a.reshape_ref({ nbeams, nr, nt_ds });
		nt_cumul += nt_ds;
	    }

	    for (int i = 0; i < niter; i++)
		kernel->launch(in_s, out, pstate_s, i * nt_chunk, stream);
	};
    
    CudaStreamPool pool(callback, ncallbacks, nstreams, kernel_name.str());
    pool.monitor_throughput("global memory (GB/s)", gmem_tot);
    pool.run();
}


int main(int argc, char **argv)
{
    time_gpu_dedispersion_kernel<float> (8, 16, 1);
    time_gpu_dedispersion_kernel<float> (8, 16, 3);
    time_gpu_dedispersion_kernel<float> (8, 16, 5);
    
    return 0;
}
