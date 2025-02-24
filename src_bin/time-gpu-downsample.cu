#include <cassert>
#include <iostream>
#include <ksgpu/Array.hpp>
#include <ksgpu/CudaStreamPool.hpp>
#include <ksgpu/string_utils.hpp>

#include "../include/pirate/internals/gpu_downsample.hpp"

using namespace std;
using namespace ksgpu;
using namespace pirate;


int main(int argc, char **argv)
{
    if (argc != 3) {
	cerr << "usage: time-downsample <Df> <Dt>" << endl;
	exit(1);
    }

    int Df = from_str<int> (argv[1]);
    int Dt = from_str<int> (argv[2]);
    bool transpose_output = true;  // FIXME
    
    assert((Df >= 1) && ((2048 % Df) == 0));
    assert((Dt >= 1) && ((2048 % Dt) == 0));
    
    ssize_t nbeams = 64;
    ssize_t nfreq_src = 2048;
    ssize_t ntime_src = 2048;
    ssize_t niter = 100;
    ssize_t num_callbacks = 20;
    ssize_t nstreams = 2;
    
    ssize_t nfreq_dst = nfreq_src / Df;
    ssize_t ntime_dst = ntime_src / Dt;
    double gmem_gb = 8. * nbeams * niter * double(nfreq_src * ntime_src + nfreq_dst * ntime_dst) / pow(2,30.);
    
    Array<float> src_si({nstreams, nbeams, nfreq_src, ntime_src}, af_gpu | af_zero);
    Array<float> src_sw({nstreams, nbeams, nfreq_src, ntime_src}, af_gpu | af_zero);
    Array<float> dst_si({nstreams, nbeams, ntime_dst, nfreq_dst}, af_gpu | af_zero);
    Array<float> dst_sw({nstreams, nbeams, ntime_dst, nfreq_dst}, af_gpu | af_zero);

    auto callback = [&](const CudaStreamPool &pool, cudaStream_t stream, int istream)
        {
	    Array<float> src_i = src_si.slice(0, istream);
	    Array<float> src_w = src_sw.slice(0, istream);
	    Array<float> dst_i = dst_si.slice(0, istream);
	    Array<float> dst_w = dst_sw.slice(0, istream);
	    
	    for (int i = 0; i < niter; i++)
		launch_downsample(dst_i, dst_w, src_i, src_w, Df, Dt, transpose_output, stream);
	};

    stringstream sp_name;
    sp_name << "downsample(Df=" << Df << ",Dt=" << Dt << ",transpose_output=" << transpose_output << ")";
    
    CudaStreamPool pool(callback, num_callbacks, nstreams, sp_name.str());
    pool.monitor_throughput("global memory (GB/s)", gmem_gb);
    pool.run();
    
    return 0;
}
