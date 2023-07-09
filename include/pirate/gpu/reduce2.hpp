#ifndef _PIRATE_GPU_REDUCE2
#define _PIRATE_GPU_REDUCE2

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// 'scratch' should point to shared memory with 2*nwarps elements (8*nwarps bytes)
__device__ double reduce2(float num, float den, float *scratch)
{
    const int warpId = threadIdx.x >> 5;
    const int laneId = threadIdx.x & 0x1f;
    const int nwarps = blockDim.x >> 5;

    int odd = threadIdx.x & 0x01;
    float x = odd ? den : num;
    float y = odd ? num : den;

    // Reminder: there is no __reduce_sync() for floating-point types.
    x += __shfl_xor_sync(0xffffffff, y, 0x01);  // Note y here
    x += __shfl_xor_sync(0xffffffff, x, 0x02);  // Note x here and afterwards
    x += __shfl_xor_sync(0xffffffff, x, 0x04);
    x += __shfl_xor_sync(0xffffffff, x, 0x08);
    x += __shfl_xor_sync(0xffffffff, x, 0x10);

    if (laneId <= 1)
	scratch[2*warpId + laneId] = x;

    __syncthreads();

    if (warpId == 0) {
	x = 0.0;
	if (laneId < 2*nwarps)
	    x = scratch[laneId];
	if (laneId+32 < 2*nwarps)
	    x += scratch[laneId+32];
	
	if (nwarps > 1)
	    x += __shfl_xor_sync(0xffffffff, x, 0x02);
	if (nwarps > 2)
	    x += __shfl_xor_sync(0xffffffff, x, 0x04);
	if (nwarps > 4)
	    x += __shfl_xor_sync(0xffffffff, x, 0x08);
	if (nwarps > 8)
	    x += __shfl_xor_sync(0xffffffff, x, 0x10);

	y = __shfl_xor_sync(0xffffffff, x, 0x01);
	y = (y > 0) ? y : 1;
	
	if (laneId == 0)
	    scratch[0] = (x/y);
    }

    __syncthreads();

    return scratch[0];
}


}  // namespace pirate

#endif // _PIRATE_GPU_REDUCE2
