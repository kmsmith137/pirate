#ifndef _PIRATE_LOOSE_ENDS_DOWNSAMPLE_KERNEL_HPP
#define _PIRATE_LOOSE_ENDS_DOWNSAMPLE_KERNEL_HPP

namespace pirate {
#if 0
}  // namespace pirate
#endif


// Notes:
//
//   - Currently the DownsampleKernel outputs data in transposed order!
//     Later, I may introduce two versions of the kernel, with/without transpose.
//
//     (It happens to be the case that for the current CHIME RFI transform chain,
//     the non-transposed version is not needed.)
//
//   - We currently assume that the weights/intensity arrays have the same strides.
//
//   - Since we use float4 loads/stores, strides must be multiples of 4.
//
//   - FIXME implementation for (Dt >= 4) involves loads which are not "cache friendly".
//     Timing shows that kernel is a little slow (~30% slower than memory bandwidth limited),
//     but only for Dt=8. (This value does not arise in the current CHIME RFI chain.)


struct DownsampleKernel
{
    static constexpr int warps_per_block = 8;
    static constexpr int blocks_per_sm = 4;

    
    template<int D>
    static __device__ void downsample_and_write(float4 a, float *p)
    {
        static_assert((D==1) || (D==2) || (D==4));
        
        if constexpr (D == 1) {
            p[0] = a.x;
            p[1] = a.y;
            p[2] = a.z;
            p[3] = a.w;
        }
        
        if constexpr (D == 2) {
            p[0] = a.x + a.y;
            p[1] = a.z + a.w;
        }
        
        if constexpr (D == 4)
            p[0] = a.x + a.y + a.z + a.w;
    }

    
    // First half of kernel: read from global memory, downsample, write to shared memory.
    
    template<int D>
    static __device__ void kernel_step1(const float *src_i, const float *src_w, int Df, int Dt, int src_fstride, int src_bstride, float shmem[2][32][33])
    {
        // Indices in downsampled array.
        int tdst = (threadIdx.x & (8*D-1)) * (4/D);  // tsrc = Dt*tdst
        int fdst = (threadIdx.x / (8*D));            // fsrc = Df*fdst
        
        int src_offset = (32*blockIdx.x) * Dt;
        src_offset += (32*blockIdx.y) * Df * src_fstride;
        src_offset += (blockIdx.z) * src_bstride;

        src_i += src_offset;
        src_w += src_offset;

        for (int f0 = fdst; f0 < 32; f0 += (32/D)) {
            float4 wisum = make_float4(0., 0., 0., 0.);
            float4 wsum = make_float4(0., 0., 0., 0.);
        
            for (int f1 = f0*Df; f1 < (f0+1)*Df; f1++) {
                for (int t = Dt*tdst; t < Dt*(tdst+1); t += 4) {
                    float4 isamples = *((float4 *) (src_i + f1*src_fstride + t));
                    float4 wsamples = *((float4 *) (src_w + f1*src_fstride + t));

                    wisum.x += wsamples.x * isamples.x;
                    wisum.y += wsamples.y * isamples.y;
                    wisum.z += wsamples.z * isamples.z;
                    wisum.w += wsamples.w * isamples.w;
                
                    wsum.x += wsamples.x;
                    wsum.y += wsamples.y;
                    wsum.z += wsamples.z;
                    wsum.w += wsamples.w;
                }
            }
            
            downsample_and_write<D> (wisum, &shmem[0][f0][tdst]);
            downsample_and_write<D> (wsum, &shmem[1][f0][tdst]);
        }
    }


    // Second half of kernel: read from shared memory, write to global memory.
    static __device__ void kernel_step2(float *dst_i, float *dst_w, int dst_tstride, int dst_bstride, float shmem[2][32][33])
    {   
        int tdst = threadIdx.x >> 3;
        int fdst = 4 * (threadIdx.x & 0x07);

        int dst_offset = (32*blockIdx.x + tdst) * dst_tstride;
        dst_offset += (32*blockIdx.y + fdst);
        dst_offset += (blockIdx.z) * dst_bstride;

        float4 wisum;
        wisum.x = shmem[0][fdst][tdst];
        wisum.y = shmem[0][fdst+1][tdst];
        wisum.z = shmem[0][fdst+2][tdst];
        wisum.w = shmem[0][fdst+3][tdst];
        
        float4 wsum;
        wsum.x = shmem[1][fdst][tdst];
        wsum.y = shmem[1][fdst+1][tdst];
        wsum.z = shmem[1][fdst+2][tdst];
        wsum.w = shmem[1][fdst+3][tdst];

        float4 den;
        den.x = (wsum.x > 0.0) ? wsum.x : 1.0;
        den.y = (wsum.y > 0.0) ? wsum.y : 1.0;
        den.z = (wsum.z > 0.0) ? wsum.z : 1.0;
        den.w = (wsum.w > 0.0) ? wsum.w : 1.0;

        float4 iout;
        iout.x = wisum.x / den.x;
        iout.y = wisum.y / den.y;
        iout.z = wisum.z / den.z;
        iout.w = wisum.w / den.w;
        
        *((float4 *) (dst_i + dst_offset)) = iout;
        *((float4 *) (dst_w + dst_offset)) = wsum;
    }
};


// FIXME should some strides be long?
template<int D>
__global__ void __launch_bounds__(32 * DownsampleKernel::warps_per_block, DownsampleKernel::blocks_per_sm)
downsample_kernel(float *dst_i, float *dst_w, const float *src_i, const float *src_w, int Df, int Dt, int src_fstride, int src_bstride, int dst_tstride, int dst_bstride)
{
    __shared__ float shmem[2][32][33];

    // First half of kernel: read from global memory, downsample, write to shared memory.
    DownsampleKernel::kernel_step1<D> (src_i, src_w, Df, Dt, src_fstride, src_bstride, shmem);

    __syncthreads();

    // Second half of kernel: read from shared memory, write to global memory.
    DownsampleKernel::kernel_step2(dst_i, dst_w, dst_tstride, dst_bstride, shmem);
}


} // namespace pirate

#endif // _PIRATE_LOOSE_ENDS_DOWNSAMPLE_KERNEL_HPP
