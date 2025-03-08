#ifndef _PIRATE_LOOSE_ENDS_GPU_TRANSPOSE_KERNEL_HPP
#define _PIRATE_LOOSE_ENDS_GPU_TRANSPOSE_KERNEL_HPP

namespace pirate {
#if 0
}  // namespace pirate
#endif


// Note: since we use float4 loads/stores, strides must be multiples of 4.

struct TransposeKernel
{
    static constexpr int warps_per_block = 8;
    static constexpr int blocks_per_sm = 4;

    static __device__ void kernel_body(float *dst, const float *src, int src_ystride, int src_zstride, int dst_xstride, int dst_zstride)
    {
	__shared__ float shmem[32][33];
	
	int sx = 4 * (threadIdx.x & 0x07);
	int sy = threadIdx.x >> 3;

	src += sx + (32 * blockIdx.x);
	src += (sy + 32*blockIdx.y) * src_ystride;
	src += blockIdx.z * src_zstride;
	
	float4 t = *((float4 *) src);

	shmem[sy][sx] = t.x;
	shmem[sy][sx+1] = t.y;
	shmem[sy][sx+2] = t.z;
	shmem[sy][sx+3] = t.w;

	__syncthreads();
	
	int dx = sy;
	int dy = sx;

	dst += dy + (32 * blockIdx.y);
	dst += (dx + 32*blockIdx.x) * dst_xstride;
	dst += blockIdx.z * dst_zstride;
	
	t.x = shmem[dy][dx];
	t.y = shmem[dy+1][dx];
	t.z = shmem[dy+2][dx];
	t.w = shmem[dy+3][dx];

	*((float4 *) dst) = t;
    }
};


// FIXME should some strides be long?
__global__ void __launch_bounds__(32 * TransposeKernel::warps_per_block, TransposeKernel::blocks_per_sm)
transpose_kernel(float *dst, const float *src, int src_ystride, int src_zstride, int dst_xstride, int dst_zstride)
{
    TransposeKernel::kernel_body(dst, src, src_ystride, src_zstride, dst_xstride, dst_zstride);
}


} // namespace pirate

#endif // _PIRATE_LOOSE_ENDS_GPU_TRANSPOSE_KERNEL_HPP
