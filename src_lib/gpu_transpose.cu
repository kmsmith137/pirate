#include <gputils/Array.hpp>

#include "../include/pirate/internals/gpu_transpose.hpp"
#include "../include/pirate/gpu/TransposeKernel.hpp"

using namespace gputils;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


void launch_transpose(Array<float> &dst, const Array<float> &src, cudaStream_t stream)
{
    assert(dst.ndim == 3);
    assert(src.ndim == 3);

    unsigned int nz = src.shape[0];
    unsigned int ny = src.shape[1];
    unsigned int nx = src.shape[2];

    assert(dst.shape_equals({nz,nx,ny}));
    assert((nx % 32) == 0);
    assert((ny % 32) == 0);
    
    assert(dst.strides[2] == 1);
    assert(src.strides[2] == 1);
    assert((src.strides[0] % 4) == 0);
    assert((src.strides[1] % 4) == 0);
    assert((dst.strides[0] % 4) == 0);
    assert((dst.strides[1] % 4) == 0);

    dim3 nblocks;
    nblocks.x = nx / 32;
    nblocks.y = ny / 32;
    nblocks.z = nz;

    transpose_kernel<<< nblocks, 32*TransposeKernel::warps_per_block, 0, stream >>>
	(dst.data, src.data, src.strides[1], src.strides[0], dst.strides[1], dst.strides[0]);
}


}  // namespace pirate
