#include <iostream>
#include <ksgpu.hpp>

#include "../../include/pirate/loose_ends/reduce2.hpp"
#include "../../include/pirate/loose_ends/tests.hpp"

using namespace std;
using namespace ksgpu;

namespace pirate {
#if 0
}  // editor auto-indent
#endif



__global__ void reduce2_kernel(float *dst, const float *num, const float *den)
{
    extern __shared__ float shmem[];
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    dst[s] = reduce2(num[s], den[s], shmem);
}


static void test_gpu_reduce2(int nblocks, int nwarps)
{
    cout << "test_gpu_reduce2(nblocks=" << nblocks << ", nwarps=" << nwarps << ")" << endl;
    
    int nthreads = nwarps * 32;
    Array<float> num_cpu({nblocks,nthreads}, af_rhost | af_random);
    Array<float> den_cpu({nblocks,nthreads}, af_rhost | af_random);
    Array<float> res_cpu({nblocks,nthreads}, af_rhost);
        
    for (int i = 0; i < nblocks; i++) {
        float *np = num_cpu.data + i*nthreads;
        float *dp = den_cpu.data + i*nthreads;
        float *rp = res_cpu.data + i*nthreads;
        
        bool zero = (rand_uniform() < 0.05);
        float nsum = 0.0;
        float dsum = 0.0;
        
        for (int j = 0; j < nthreads; j++) {
            dp[j] = zero ? 0.0 : fabs(dp[j]);
            nsum += np[j];
            dsum += dp[j];
        }

        float r = (dsum > 0.0) ? (nsum/dsum) : nsum;
        for (int j = 0; j < nthreads; j++)
            rp[j] = r;
    }

    Array<float> num_gpu = num_cpu.to_gpu();
    Array<float> den_gpu = den_cpu.to_gpu();
    Array<float> res_gpu({nblocks,nthreads}, af_gpu);

    int shmem_nbytes = 8 * nwarps;
    reduce2_kernel <<<nblocks, nthreads, shmem_nbytes>>> (res_gpu.data, num_gpu.data, den_gpu.data);
    CUDA_PEEK("reduce2_kernel");
    CUDA_CALL(cudaDeviceSynchronize());

    assert_arrays_equal(res_cpu, res_gpu, "reduce2 (cpu)", "reduce2 (gpu)", {"block","thread"});
    cout << "test_reduce2: pass" << endl;
}
                  

void test_gpu_reduce2()
{
    int nwarps = rand_int(1,33);
    int nblocks = rand_int(1,10);
    
    test_gpu_reduce2(nblocks, nwarps);
}


}  // namespace pirate
