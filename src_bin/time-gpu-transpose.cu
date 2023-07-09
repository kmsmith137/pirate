#include <iostream>
#include <gputils/Array.hpp>
#include <gputils/CudaStreamPool.hpp>

#include "../include/pirate/internals/gpu_transpose.hpp"

using namespace std;
using namespace gputils;
using namespace pirate;


int main(int argc, char **argv)
{
    int nx = 2048;
    int ny = 2048;
    int nz = 64;
    int niter = 100;
    int num_callbacks = 20;
    int nstreams = 4;
    double gmem_gb = nx * ny * nz * double(niter) * 8. / pow(2,30.);
    
    vector<Array<float>> src(nstreams);
    vector<Array<float>> dst(nstreams);

    for (int istream = 0; istream < nstreams; istream++) {
	src[istream] = Array<float> ({nz,ny,nx}, af_gpu | af_zero);
	dst[istream] = Array<float> ({nz,nx,ny}, af_gpu | af_zero);
    }

    auto callback = [&](const CudaStreamPool &pool, cudaStream_t stream, int istream)
        {
	    for (int i = 0; i < niter; i++)
		launch_transpose(dst[istream], src[istream], stream);
	};

    CudaStreamPool pool(callback, num_callbacks, nstreams, "transpose");
    pool.monitor_throughput("global memory (GB/s)", 8. * nx*ny*nz * double(niter) / pow(2,30.));
    pool.run();
    
    return 0;
}
