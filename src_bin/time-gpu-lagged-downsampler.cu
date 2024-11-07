#include <iostream>
#include <gputils/Array.hpp>
#include <gputils/CudaStreamPool.hpp>

#include "../include/pirate/internals/inlines.hpp"  // pow2()
#include "../include/pirate/internals/GpuLaggedDownsamplingKernel.hpp"

using namespace std;
using namespace gputils;
using namespace pirate;


static void time_gpu_dedispersion_kernel(const GpuLaggedDownsamplingKernel::Params &params)
{
    long nstreams = 1;
    long ncallbacks = 10;
    long nt_chunk = 2048;
    long nbeams = 4;
    long niter = 20;
    
    bool is_float32 = params.is_float32();
    long ST = is_float32 ? 4 : 2;  // sizeof(T)

    shared_ptr<GpuLaggedDownsamplingKernel> kernel = GpuLaggedDownsamplingKernel::make(params);

    long ninner = 0;
    for (int ids = 0; ids < params.num_downsampling_levels; ids++)
	ninner += pow2(params.large_input_rank-1) * xdiv(nt_chunk, pow2(ids+1));
    
    long out_nelts_per_stream = nbeams * ninner;
    long in_nelts_per_stream = nbeams * pow2(params.large_input_rank) * nt_chunk;
    long pstate_nelts_per_stream = nbeams * kernel->state_nelts_per_beam;
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
		<< ")\n";

    cout << kernel_name.str();
    kernel->print(cout, 4);  // indent=4

    cout << "    nstreams = " << nstreams << endl
	 << "    nt_chunk = " << nt_chunk << endl
	 << "    nbeams = " << nbeams << endl
	 << "    niter = " << niter << endl
	 << "    gpu memory footprint = " << gputils::nbytes_to_str(footprint_nbytes) << endl;

    UntypedArray in;
    UntypedArray pstate;
    UntypedArray out_flattened;

    in.allocate({ nstreams, nbeams, pow2(params.large_input_rank), nt_chunk }, af_gpu | af_zero, is_float32);
    pstate.allocate({ nstreams, nbeams, kernel->state_nelts_per_beam }, af_gpu | af_zero, is_float32);
    out_flattened.allocate({ nstreams, nbeams, ninner }, af_gpu | af_zero, is_float32);

    auto callback = [&](const CudaStreamPool &pool, cudaStream_t stream, int istream)
        {
	    UntypedArray in_s = in.slice(0, istream);
	    UntypedArray pstate_s = pstate.slice(0, istream);
	    UntypedArray outf_s = out_flattened.slice(0, istream);

	    vector<UntypedArray> out(params.num_downsampling_levels);
	    long nr = 1 << (params.large_input_rank-1);
	    long nt_cumul = 0;
	    
	    for (int ids = 0; ids < params.num_downsampling_levels; ids++) {
		long nt_ds = nt_chunk >> (ids+1);
		UntypedArray a = outf_s.slice(1, nr * nt_cumul, nr * (nt_cumul + nt_ds));
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
    for (int num_downsampling_levels: {1,3,5}) {
	GpuLaggedDownsamplingKernel::Params params;
	params.dtype = "float32";  // FIXME should time float16
	params.small_input_rank = 8;
	params.large_input_rank = 16;
	params.num_downsampling_levels = num_downsampling_levels;
	time_gpu_dedispersion_kernel(params);
    }
    
    return 0;
}
