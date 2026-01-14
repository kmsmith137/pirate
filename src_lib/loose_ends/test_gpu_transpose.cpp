#include <iostream>
#include <ksgpu/Array.hpp>
#include <ksgpu/cuda_utils.hpp>
#include <ksgpu/test_utils.hpp>

#include "../../include/pirate/loose_ends/gpu_transpose.hpp"
#include "../../include/pirate/loose_ends/tests.hpp"

using namespace std;
using namespace ksgpu;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


static void test_gpu_transpose(int nx, int ny, int nz, int src_ystride, int src_zstride, int dst_xstride, int dst_zstride)
{
    cout << "test_gpu_transpose: (nx, ny, nz, src_ystride, src_zstride, dst_xstride, dst_zstride) = "
         << nx << ", " << ny << ", " << nz << ", " << src_ystride << ", " << src_zstride
         << ", " << dst_xstride << ", " << dst_zstride << ")"
         << endl;
    
    Array<float> src_cpu({nz,ny,nx}, af_rhost | af_random);
    Array<float> dst_cpu({nz,nx,ny}, af_rhost);

    for (auto ix = dst_cpu.ix_start(); dst_cpu.ix_valid(ix); dst_cpu.ix_next(ix))
        dst_cpu.at(ix) = src_cpu.at({ix[0],ix[2],ix[1]});
    
    Array<float> src_gpu({nz,ny,nx}, {src_zstride,src_ystride,1}, af_gpu);
    Array<float> dst_gpu({nz,nx,ny}, {dst_zstride,dst_xstride,1}, af_gpu);

    src_gpu.fill(src_cpu);
    launch_transpose(dst_gpu, src_gpu);
    CUDA_PEEK("launch_transpose");
    CUDA_CALL(cudaDeviceSynchronize());

    assert_arrays_equal(dst_cpu, dst_gpu, "cpu", "gpu", {"z","y","x"});
}


void test_gpu_transpose()
{
    long nx = 32 * rand_int(1, 10);
    long ny = 32 * rand_int(1, 10);
    long nz = rand_int(1, 10);
        
    auto src_strides = make_random_strides({nz,ny,nx}, 1, 4);  // ncontig=1, nalign=4
    auto dst_strides = make_random_strides({nz,nx,ny}, 1, 4);  // ncontig=1, nalign=4
    
    test_gpu_transpose(nx, ny, nz, src_strides[1], src_strides[0], dst_strides[1], dst_strides[0]);
}


}  // namespace pirate
