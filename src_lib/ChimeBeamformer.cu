#include "../include/pirate/ChimeBeamformer.hpp"

#include <algorithm>   // std::max
#include <cuda_fp16.h>
#include <cufftdx.hpp>

#include <iostream>

#include <ksgpu/xassert.hpp>
#include <ksgpu/cuda_utils.hpp>   // CUDA_PEEK
#include <ksgpu/KernelTimer.hpp>

using namespace std;
using namespace ksgpu;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// -------------------------------------------------------------------------------------------------
//
// _load_half4(), _store_half4(): load/store (4xfp16) with a single 64-bit instruction.
// The pointer 'p' must be 64-bit aligned.


__device__ __forceinline__ __half2 __uint_as_half2(unsigned int x)
{
    // According to Claude Code, memcpy() for type punning is well-defined C++,
    // and nvcc will optimize it away to a no-op (same register, reinterpreted).
    __half2 h;
    memcpy(&h, &x, sizeof(h));
    return h;
}

__device__ __forceinline__ unsigned int __half2_as_uint(__half2 h)
{
    unsigned int x;
    memcpy(&x, &h, sizeof(x));
    return x;
}

__device__ __forceinline__ void _load_half4(const __half2 *p, __half2 &x, __half2 &y)
{
    // According to Claude Code, the reinterpret_cast through (uint2 *) guarantees a single
    // 64-bit load/store instruction (LDG.64 or LDS.64) rather than two 32-bit transactions.
    uint2 tmp = *reinterpret_cast<const uint2 *>(p);
    x = __uint_as_half2(tmp.x);
    y = __uint_as_half2(tmp.y);
}

__device__ __forceinline__ void _store_half4(__half2 *p, __half2 x, __half2 y)
{
    uint2 tmp;
    tmp.x = __half2_as_uint(x);
    tmp.y = __half2_as_uint(y);
    *reinterpret_cast<uint2 *>(p) = tmp;
}


// -------------------------------------------------------------------------------------------------


// Helper for _fft128_sq() and chime_frb_upchan().
__device__ inline float _asymmetric_reduce(float x, float y, uint lane)
{
    // Input:
    //  register <-> i
    //  lane <-> j
    //
    // Output:
    //  lane <-> i,   summed over j

    float t = (threadIdx.x & lane) ? x : y;
    t = __shfl_sync(~0U, t, threadIdx.x ^ lane);
    t += (threadIdx.x & lane) ? y : x;
    return t;
}



// FFT, square, sum over f0+f1
//
// Input: float16+16 x
//   register <-> t6 t5
//   lane <-> t4 t3 t2 t1 t0
//
// Output: float32 I
//   lane <-> f4 f3 f2 f6 f5

__device__ inline float _fft128_sq(__half2 x0, __half2 x1, __half2 x2, __half2 x3, char *smem)
{
    // The #ifdef weirdness is needed to avoid nvcc errors during the "host" pass.
    using FFT = decltype(
          cufftdx::Size<128>()
          + cufftdx::Precision<float>()       // or __half
          + cufftdx::Type<cufftdx::fft_type::c2c>()
          + cufftdx::Direction<cufftdx::fft_direction::forward>()
          + cufftdx::ElementsPerThread<4>()
          + cufftdx::FFTsPerBlock<16>()
#ifdef __CUDA_ARCH__
          + cufftdx::SM<__CUDA_ARCH__>()
#else
          + cufftdx::SM<890>()
#endif
          + cufftdx::Block()
      );

    using complex_t = typename FFT::value_type;
    static_assert(sizeof(complex_t) == 8);
    static_assert(FFT::storage_size == 4);

    complex_t data[4];
    data[0] = complex_t(__half2float(x0.x), __half2float(x0.y));
    data[1] = complex_t(__half2float(x1.x), __half2float(x1.y));
    data[2] = complex_t(__half2float(x2.x), __half2float(x2.y));
    data[3] = complex_t(__half2float(x3.x), __half2float(x3.y));
    
    // FFT and absolute square.
    // After these steps, 't' has register assignment
    //   register <-> f6 f5
    //   lane <-> f4 f3 f2 f1 f0

    FFT().execute(data, smem);    

    float t0 = data[0].x * data[0].x + data[0].y * data[0].y;
    float t1 = data[1].x * data[1].x + data[1].y * data[1].y;
    float t2 = data[2].x * data[2].x + data[2].y * data[2].y;
    float t3 = data[3].x * data[3].x + data[3].y * data[3].y;

    // Call _asymmetric_reduce(), to sum over f0/f1, while "shuffling" f6/r5.
    // After these steps, 'u' has register assignment:
    //   lane <-> f4 f3 f2 f6 f5
    
    float u0 = _asymmetric_reduce(t0, t1, 0x1);   // sum over f0
    float u1 = _asymmetric_reduce(t2, t3, 0x1);   // sum over f0
    float u = _asymmetric_reduce(u0, u1, 0x2);    // sum over f1

    return u;
}


// This is the "upchannelization" part of the CHIME FRB beamforming kernel, which
// runs after the "beamforming" part.
//
// 'data': shape=(T,F,2,1024), dtype=float16+16, axes (time,freq,pol,beam)
// 'results_array': shape=(1024,F,T/384,16), axes (beam,cfreq,time,ufreq)
//
// Each threadblock processes (32 beams, one coarse freq, 768 times).
// Thus, gridDim = (32,F,T/768).
//
// Kernel is launched with {32,16} threads per block.


__global__ void __launch_bounds__(512,2)
chime_frb_upchan(const __half2 *data, float *results_array)
{
    // Shared memory
    // __half2 E[128][34];  // 17 KB
    //   + cuFFT            // 16 KB

    extern __shared__ __half2 smem_e[];
    char *smem_fft = reinterpret_cast<char *>(smem_e + 128*34);

    // Block indices.
    // Note that (T,F) are passed implicitly throgh grid dims.
    uint b32 = blockIdx.x;
    uint f = blockIdx.y;
    ulong t768 = blockIdx.z;  // note ulong
    ulong F = gridDim.y;      // note ulong
    ulong T768 = gridDim.z;   // note ulong
    
    // Absorb block indices into 'data'.
    // data: shape=(T,F,2,1024), axes (time,freq,pol,beam)
    data += (t768 * 768UL * F + f) * 2048UL;
    data += (b32 * 32);

    // Absorb block indices into 'results_array'.
    // results_array: shape=(1024,F,T/384,16), axes (beam,cfreq,time,ufreq)
    results_array += (b32 * 32 * F + f) * (T768 * 32);
    results_array += 32 * t768;

    // 'u' will accumulate intensity values for 32 beams, 16 coarse freqs, 2 times.
    // It's convenient to use the following weird register assignment:
    //   register <-> t384
    //   lane <-> f4 f3 b0 f6 f5
    //   warp <-> b4 b3 b2 b1

    float u0 = 0.0f;
    float u1 = 0.0f;

    // Outer loop over 6 blocks of 128 time samples (i.e. 768 time samples total)
    for (uint t128 = 0; t128 < 6; t128++) {
        for (uint pol = 0; pol < 2; pol++) {
            
            // Read data from global memory
            //   register <-> t6 t5 b0
            //   lane <-> t0 b4 b3 b2 b1
            //   warp <-> t4 t3 t2 t1
            //
            // Use 64-bit load instructions! This is the most important part of the kernel.
            
            uint bhi = (threadIdx.x & 0xf);                      // index bits (b4 b3 b2 b1)
            uint tlo = (threadIdx.x >> 4) | (threadIdx.y << 1);  // index bits (t4 t3 t2 t1 t0)

            // 'data' is a shape (768,2,32) array with strides (2048*F, 1024, 1)
            // (After "absorbing" block indices, see above.)
            
            const __half2 *dp = data;
            dp += ulong(t128 << 18) * F;  // (128 * t128) * (2048 * F)
            dp += (1024*pol + 2*bhi);
            
            __half2 e0, e1, e2, e3, e4, e5, e6, e7;
            
            _load_half4(dp, e0, e1);
            _load_half4(dp + 32*2048*F, e2, e3);
            _load_half4(dp + 64*2048*F, e4, e5);
            _load_half4(dp + 96*2048*F, e6, e7);
            
            // Write to shared memory, using 64-bit bank conflict free stores.
            // __half2 E[128][34]

            uint s = (34 * tlo) + (2 * bhi);
            
            _store_half4(smem_e + s, e0, e1);
            _store_half4(smem_e + s + 32*34, e2, e3);
            _store_half4(smem_e + s + 64*34, e4, e5);
            _store_half4(smem_e + s + 96*34, e6, e7);
            
            __syncthreads();

            // Read from shared memory, using 64-bit bank conflict free loads.
            //   register <-> t6 t5 b0
            //   lane <-> t4 t3 t2 t1 t0
            //   warp <-> b4 b3 b2 b1

            bhi = threadIdx.y;
            tlo = threadIdx.x;
            s = (34 * tlo) + (2 * bhi);
            
            _load_half4(smem_e + s, e0, e1);
            _load_half4(smem_e + s + 32*34, e2, e3);
            _load_half4(smem_e + s + 64*34, e4, e5);
            _load_half4(smem_e + s + 96*34, e6, e7);

            // FFT, square, sum over f0, f1, f2.
            // After these steps, we have:
            //   lane <-> f4 f3 b0 f6 f5
            //   warp <-> b4 b3 b2 b1
            
            float t0 = _fft128_sq(e0, e2, e4, e6, smem_fft);  // sum over f0, f1
            float t1 = _fft128_sq(e1, e3, e5, e7, smem_fft);  // sum over f0, f1
            float t = _asymmetric_reduce(t0, t1, 0x4);        // sum over f2

            // Accumulate into (u0, u1)
            // Reminder: u register layout is
            //   register <-> t384
            //   lane <-> f4 f3 b0 f6 f5
            //   warp <-> b4 b3 b2 b1
            
            u0 = (t128 < 3) ? (u0+t) : (u0);
            u1 = (t128 < 3) ? (u1) : (u1+t);
        }
    }

    // Warp shuffle u->v, obtaining:
    //   register <-> b0
    //   lane <-> f4 f3 t384 f6 f5
    //   warp <-> b4 b3 b2 b1

    float u = (threadIdx.x & 0x4) ? u0 : u1;
    u = __shfl_sync(~0u, u, threadIdx.x ^ 0x4);
    float v0 = (threadIdx.x & 0x4) ? u : u0;
    float v1 = (threadIdx.x & 0x4) ? u1 : u;

    // Permute lanes, obtaining:
    //   register <-> b0
    //   lane <-> t384 f6 f5 f4 f3
    //   warp <-> b1 b2 b3 b4

    uint lane = (threadIdx.x & 0x1c) >> 2;  // t384 f6 f5
    lane |= (threadIdx.x & 0x3) << 3;       // f4 f3

    v0 = __shfl_sync(~0u, v0, lane);
    v1 = __shfl_sync(~0u, v1, lane);

    // 'results_array' is a shape (32,2,16) array, strides (32*F*T768,16,1), axes (beam,t384,ufreq).
    // (After "absorbing" block indices, see above.)

    ulong bstride = 32UL * ulong(F) * ulong(T768);
    ulong roff = threadIdx.y * (2 * bstride) + (threadIdx.x);
    
    results_array[roff] = v0;
    results_array[roff + bstride] = v1;
}

// 'data': shape=(T,F,2,1024,2), axes (time,freq,pol,beam,ReIm)
// 'results_array': shape=(1024,F,T/384,16), axes (beam,cfreq,time,ufreq)
void launch_chime_frb_upchan(const __half *data, float *results_array, long T, long F, cudaStream_t stream)
{
    xassert(T > 0);
    xassert(F > 0);
    xassert_divisible(T, 768);

    long T768 = T / 768;

    // Shared memory: __half2 E[128][34] + cuFFTDx workspace.
    // Query cuFFTDx shared memory for each target SM and take the max.
    using FFTdesc = decltype(
          cufftdx::Size<128>()
          + cufftdx::Precision<float>()
          + cufftdx::Type<cufftdx::fft_type::c2c>()
          + cufftdx::Direction<cufftdx::fft_direction::forward>()
          + cufftdx::ElementsPerThread<4>()
          + cufftdx::FFTsPerBlock<16>()
          + cufftdx::Block()
      );

    constexpr unsigned int smem_fft = std::max({
        decltype(FFTdesc() + cufftdx::SM<800>())::shared_memory_size,
        decltype(FFTdesc() + cufftdx::SM<860>())::shared_memory_size,
        decltype(FFTdesc() + cufftdx::SM<890>())::shared_memory_size
    });

    unsigned int smem_nbytes = (128 * 34 * sizeof(__half2)) + smem_fft;
    
    chime_frb_upchan<<< {32,(uint)F,(uint)T768}, {32,16}, smem_nbytes, stream >>>
        (reinterpret_cast<__const __half2 *> (data), results_array);

    CUDA_PEEK("chime_frb_upchan");
}


// 'data': shape=(T,F,2,1024,2), axes (time,freq,pol,beam,ReIm)
// 'results_array': shape=(1024,F,T/384,16), axes (beam,cfreq,time,ufreq)
void launch_chime_frb_upchan(const Array<__half> &data, Array<float> &results_array, cudaStream_t stream)
{
    // data: shape=(T,F,2,1024,2), axes (time,freq,pol,beam,ReIm)
    xassert(data.ndim == 5);
    long T = data.shape[0];
    long F = data.shape[1];
    xassert_eq(data.shape[2], 2);
    xassert_eq(data.shape[3], 1024);
    xassert_eq(data.shape[4], 2);
    xassert(data.on_gpu());
    xassert(data.is_fully_contiguous());

    // results_array: shape=(1024,F,T/384,16), axes (beam,cfreq,time,ufreq)
    xassert_divisible(T, 768);
    xassert_shape_eq(results_array, ({1024, F, T/384, 16}));
    xassert(results_array.on_gpu());
    xassert(results_array.is_fully_contiguous());

    launch_chime_frb_upchan(data.data, results_array.data, T, F, stream);
}


void time_chime_frb_upchan()
{
    long T = 49152;
    long F = 16;
    long niterations = 1000;
    long nstreams = 1;

    // Global memory: read data + write results_array.
    // data: shape=(T,F,2,1024,2), dtype=__half (2 bytes)
    // results_array: shape=(1024,F,T/384,16), dtype=float (4 bytes)
    double gmem_gb = (double(T) * F * 2 * 1024 * 2 * sizeof(__half)
                      + 1024.0 * F * (T/384) * 16 * sizeof(float)) / pow(2,30.);

    Array<__half> data({T, F, 2, 1024, 2}, af_gpu | af_zero);
    Array<float> results_array({1024, F, T/384, 16}, af_gpu | af_zero);

    KernelTimer kt(niterations, nstreams);

    while (kt.next()) {
        launch_chime_frb_upchan(data, results_array, kt.stream);

        if (kt.warmed_up && (kt.curr_iteration % 50 == 49)) {
            double gb_per_sec = gmem_gb / kt.dt;
            cout << "chime_frb_upchan: " << gb_per_sec << " GB/s (iteration " << kt.curr_iteration << ")" << endl;
        }
    }
}

} // namespace pirate

