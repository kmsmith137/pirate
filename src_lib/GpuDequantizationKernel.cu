#include "../include/pirate/GpuDequantizationKernel.hpp"

#include <iostream>
#include <cuda_fp16.h>
#include <ksgpu/xassert.hpp>
#include <ksgpu/cuda_utils.hpp>
#include <ksgpu/device_fp16.hpp>    // half8_store()
#include <ksgpu/rand_utils.hpp>     // rand_uniform(), rand_int()
#include <ksgpu/test_utils.hpp>     // assert_arrays_equal()
#include <ksgpu/KernelTimer.hpp>

using namespace std;
using namespace ksgpu;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// -------------------------------------------------------------------------------------------------
//
// GPU Kernels
//
// Each warp processes 256 consecutive int4 elements for one (beam, freq, time_chunk).
// Uses 128-bit (float4/uint4) stores for improved memory bandwidth.


// Float32 kernel: 2 shuffles, 2 × 128-bit write transactions using float4
// Block dims: (32, 8, 1) - threadIdx.y runs over frequency for better occupancy
__global__ void gpu_dequantize_fp32_kernel(
    float4 *out,
    const uint32_t *in,
    long out_stride,   // stride between (beam,freq) rows in output (in float4 units)
    long in_stride,    // stride between (beam,freq) rows in input (in uint32 units)
    int nfreq)         // total number of frequencies (for bounds checking)
{
    // Grid mapping: blockIdx = (freq/8, time_chunk, beam), threadIdx.y = freq%8
    int freq = blockIdx.x * 8 + threadIdx.y;
    int time_chunk = blockIdx.y;
    int beam = blockIdx.z;
    int thread_id = threadIdx.x;  // 0-31
    
    // Bounds check: nfreq may not be a multiple of 8
    if (freq >= nfreq)
        return;
    
    // Pointers to this warp's data
    // Input: 32 uint32 values = 128 bytes = 256 int4 values
    // Output: 64 float4 values = 256 float32 values
    long bf_idx = long(beam) * nfreq + freq;
    const uint32_t *inp = in + bf_idx * in_stride + time_chunk * 32;
    float4 *outp = out + bf_idx * out_stride + time_chunk * 64;
    
    // Step 1: Coalesced read (32 threads × 4 bytes = 128 bytes)
    uint32_t packed = inp[thread_id];
    
    // Step 2: Only 2 shuffles needed for 128-bit stores!
    // Thread t writes elements [4t, 4t+3] and [128+4t, 128+4t+3]
    // Elements [4t, 4t+3] are 4 consecutive nibbles in thread t/2
    // Elements [128+4t, 128+4t+3] are 4 consecutive nibbles in thread 16+t/2
    int src_thread_lo = thread_id / 2;        // 0,0,1,1,2,2,...,15,15
    int src_thread_hi = 16 + thread_id / 2;   // 16,16,17,17,...,31,31
    int nibble_base = (thread_id % 2) * 4;    // 0,4,0,4,0,4,...
    
    uint32_t data_lo = __shfl_sync(~0u, packed, src_thread_lo);
    uint32_t data_hi = __shfl_sync(~0u, packed, src_thread_hi);
    
    // Extract 4 nibbles from data_lo, convert to float4
    int nib0 = (data_lo >> (nibble_base * 4)) & 0xF;
    int nib1 = (data_lo >> ((nibble_base + 1) * 4)) & 0xF;
    int nib2 = (data_lo >> ((nibble_base + 2) * 4)) & 0xF;
    int nib3 = (data_lo >> ((nibble_base + 3) * 4)) & 0xF;
    
    float4 out_lo;
    out_lo.x = (float)((nib0 >= 8) ? (nib0 - 16) : nib0);
    out_lo.y = (float)((nib1 >= 8) ? (nib1 - 16) : nib1);
    out_lo.z = (float)((nib2 >= 8) ? (nib2 - 16) : nib2);
    out_lo.w = (float)((nib3 >= 8) ? (nib3 - 16) : nib3);
    
    // Extract 4 nibbles from data_hi, convert to float4
    nib0 = (data_hi >> (nibble_base * 4)) & 0xF;
    nib1 = (data_hi >> ((nibble_base + 1) * 4)) & 0xF;
    nib2 = (data_hi >> ((nibble_base + 2) * 4)) & 0xF;
    nib3 = (data_hi >> ((nibble_base + 3) * 4)) & 0xF;
    
    float4 out_hi;
    out_hi.x = (float)((nib0 >= 8) ? (nib0 - 16) : nib0);
    out_hi.y = (float)((nib1 >= 8) ? (nib1 - 16) : nib1);
    out_hi.z = (float)((nib2 >= 8) ? (nib2 - 16) : nib2);
    out_hi.w = (float)((nib3 >= 8) ? (nib3 - 16) : nib3);
    
    // Step 3: 128-bit coalesced writes (2 transactions × 512 bytes = 1024 bytes = 256 floats)
    outp[thread_id] = out_lo;        // elements [4t, 4t+3]
    outp[32 + thread_id] = out_hi;   // elements [128+4t, 128+4t+3]
}


// Float16 kernel: 0 shuffles, 1 × 128-bit write transaction using uint4
// Thread t already has elements [8t, 8t+7] after the read - no shuffle needed!
// Block dims: (32, 8, 1) - threadIdx.y runs over frequency for better occupancy
__global__ void gpu_dequantize_fp16_kernel(
    uint4 *out,
    const uint32_t *in,
    long out_stride,   // stride between (beam,freq) rows in output (in uint4 units)
    long in_stride,    // stride between (beam,freq) rows in input (in uint32 units)
    int nfreq)         // total number of frequencies (for bounds checking)
{
    // Grid mapping: blockIdx = (freq/8, time_chunk, beam), threadIdx.y = freq%8
    int freq = blockIdx.x * 8 + threadIdx.y;
    int time_chunk = blockIdx.y;
    int beam = blockIdx.z;
    int thread_id = threadIdx.x;  // 0-31
    
    // Bounds check: nfreq may not be a multiple of 8
    if (freq >= nfreq)
        return;
    
    // Pointers to this warp's data
    // Input: 32 uint32 values = 128 bytes = 256 int4 values
    // Output: 32 uint4 values = 512 bytes = 256 float16 values
    long bf_idx = long(beam) * nfreq + freq;
    const uint32_t *inp = in + bf_idx * in_stride + time_chunk * 32;
    uint4 *outp = out + bf_idx * out_stride + time_chunk * 32;
    
    // Step 1: Coalesced read (32 threads × 4 bytes = 128 bytes)
    uint32_t packed = inp[thread_id];
    
    // Step 2: NO shuffles needed! Thread t already has elements [8t, 8t+7]
    // Extract all 8 nibbles and convert to signed values
    int n0 = (packed >> 0) & 0xF;   int n1 = (packed >> 4) & 0xF;
    int n2 = (packed >> 8) & 0xF;   int n3 = (packed >> 12) & 0xF;
    int n4 = (packed >> 16) & 0xF;  int n5 = (packed >> 20) & 0xF;
    int n6 = (packed >> 24) & 0xF;  int n7 = (packed >> 28) & 0xF;
    
    int s0 = (n0 >= 8) ? (n0 - 16) : n0;  int s1 = (n1 >= 8) ? (n1 - 16) : n1;
    int s2 = (n2 >= 8) ? (n2 - 16) : n2;  int s3 = (n3 >= 8) ? (n3 - 16) : n3;
    int s4 = (n4 >= 8) ? (n4 - 16) : n4;  int s5 = (n5 >= 8) ? (n5 - 16) : n5;
    int s6 = (n6 >= 8) ? (n6 - 16) : n6;  int s7 = (n7 >= 8) ? (n7 - 16) : n7;
    
    // Pack pairs into __half2
    __half2 h01 = __halves2half2(__int2half_rn(s0), __int2half_rn(s1));
    __half2 h23 = __halves2half2(__int2half_rn(s2), __int2half_rn(s3));
    __half2 h45 = __halves2half2(__int2half_rn(s4), __int2half_rn(s5));
    __half2 h67 = __halves2half2(__int2half_rn(s6), __int2half_rn(s7));
    
    // Step 3: Single 128-bit coalesced write (1 transaction × 512 bytes = 256 float16)
    half8_store(&outp[thread_id], h01, h23, h45, h67);
}


// -------------------------------------------------------------------------------------------------
//
// Constructor


GpuDequantizationKernel::GpuDequantizationKernel(Dtype dtype_, long nbeams_, long nfreq_, long ntime_)
    : dtype(dtype_), nbeams(nbeams_), nfreq(nfreq_), ntime(ntime_)
{
    Dtype fp16(df_float, 16);
    Dtype fp32(df_float, 32);
    
    xassert((dtype == fp16) || (dtype == fp32));
    xassert(nbeams > 0);
    xassert(nfreq > 0);
    xassert(ntime > 0);
    xassert((ntime % 256) == 0);  // Required for cache-line alignment
    
    // Bandwidth: read int4 (0.5 bytes per element) + write output dtype
    long bytes_in = nbeams * nfreq * ntime / 2;
    long bytes_out = nbeams * nfreq * ntime * (dtype.nbits / 8);
    bw_per_launch.nbytes_gmem = bytes_in + bytes_out;
    
    // Kernel config: each warp handles 256 time samples for one (beam, freq)
    // Block: (32, 8, 1) - threadIdx.y runs over frequency for better occupancy
    // Grid: (ceil(freq/8), time_chunk, beam) - freq in x since it can be large
    nthreads = dim3(32, 8, 1);
    nblocks = dim3((nfreq + 7) / 8, ntime / 256, nbeams);
}


// -------------------------------------------------------------------------------------------------
//
// apply_reference(): CPU reference implementation


void GpuDequantizationKernel::apply_reference(Array<float> &out, const Array<void> &in) const
{
    Dtype dt_int4 = Dtype::from_str("int4");
    
    // Validate input
    xassert(in.on_host());
    xassert(in.dtype == dt_int4);
    xassert_shape_eq(in, ({nbeams, nfreq, ntime}));
    xassert(in.is_fully_contiguous());
    
    // Validate output
    xassert(out.on_host());
    xassert_shape_eq(out, ({nbeams, nfreq, ntime}));
    xassert(out.is_fully_contiguous());
    
    const unsigned char *inp = (const unsigned char *) in.data;
    float *outp = out.data;
    
    long total_elems = nbeams * nfreq * ntime;
    
    for (long i = 0; i < total_elems; i++) {
        // Extract nibble (low nibble = even index, high nibble = odd index)
        int byte_idx = i / 2;
        int nibble_idx = i % 2;
        
        unsigned char byte = inp[byte_idx];
        int nibble = (nibble_idx == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
        
        // Convert to signed two's complement: range [-8, 7]
        int value = (nibble >= 8) ? (nibble - 16) : nibble;
        
        outp[i] = (float) value;
    }
}


// -------------------------------------------------------------------------------------------------
//
// launch(): dispatch to appropriate GPU kernel


void GpuDequantizationKernel::launch(Array<void> &out, const Array<void> &in, cudaStream_t stream) const
{
    Dtype dt_int4 = Dtype::from_str("int4");
    Dtype fp16(df_float, 16);
    Dtype fp32(df_float, 32);
    
    // Validate input
    xassert(in.on_gpu());
    xassert(in.dtype == dt_int4);
    xassert_shape_eq(in, ({nbeams, nfreq, ntime}));
    xassert(in.is_fully_contiguous());
    
    // Validate output
    xassert(out.on_gpu());
    xassert(out.dtype == dtype);
    xassert_shape_eq(out, ({nbeams, nfreq, ntime}));
    xassert(out.is_fully_contiguous());
    
    // Input stride: stride between (beam,freq) rows in uint32 units
    // Each row has ntime int4 elements = ntime/2 bytes = ntime/8 uint32 values
    long in_stride = ntime / 8;
    
    if (dtype == fp32) {
        // Output stride in float4 units (each float4 holds 4 float32 values)
        long out_stride = ntime / 4;
        
        gpu_dequantize_fp32_kernel <<< nblocks, nthreads, 0, stream >>> (
            reinterpret_cast<float4 *>(out.data),
            reinterpret_cast<const uint32_t *>(in.data),
            out_stride,
            in_stride,
            nfreq
        );
    }
    else if (dtype == fp16) {
        // Output stride in uint4 units (each uint4 holds 8 float16 values)
        long out_stride = ntime / 8;
        
        gpu_dequantize_fp16_kernel <<< nblocks, nthreads, 0, stream >>> (
            reinterpret_cast<uint4 *>(out.data),
            reinterpret_cast<const uint32_t *>(in.data),
            out_stride,
            in_stride,
            nfreq
        );
    }
    else {
        throw runtime_error("GpuDequantizationKernel::launch(): invalid dtype");
    }
    
    CUDA_PEEK("GpuDequantizationKernel::launch");
}


// -------------------------------------------------------------------------------------------------
//
// test(): static member function for unit testing


void GpuDequantizationKernel::test()
{
    // Random parameters
    Dtype dtype = (rand_uniform() < 0.5) ? Dtype(df_float,16) : Dtype(df_float,32);
    
    auto v = random_integers_with_bounded_product(3, 1000);
    long B = v[0];  // beams
    long F = v[1];  // freqs  
    long T = v[2] * 256;  // time (multiple of 256)
    
    cout << "GpuDequantizationKernel::test()\n"
         << "    dtype = " << dtype << "\n"
         << "    nbeams = " << B << "\n"
         << "    nfreq = " << F << "\n"
         << "    ntime = " << T << endl;
    
    // Create int4 input with random values
    Dtype dt_int4 = Dtype::from_str("int4");
    Array<void> hin(dt_int4, {B, F, T}, af_rhost);
    
    // Fill with random int4 values (0-15 per nibble, will be interpreted as signed)
    unsigned char *p = (unsigned char *) hin.data;
    long nbytes = B * F * T / 2;
    for (long i = 0; i < nbytes; i++)
        p[i] = rand_int(0, 256);  // random byte = 2 random int4 values
    
    // Reference output (always float32)
    Array<float> href({B, F, T}, af_rhost | af_zero);
    
    // GPU input/output
    Array<void> gin = hin.to_gpu();
    Array<void> gout(dtype, {B, F, T}, af_gpu | af_zero);
    
    // Run reference
    GpuDequantizationKernel kernel(dtype, B, F, T);
    kernel.apply_reference(href, hin);
    
    // Run GPU kernel
    kernel.launch(gout, gin, nullptr);
    
    // Compare
    assert_arrays_equal(href, gout, "href", "gout", {"b","f","t"});
}


// -------------------------------------------------------------------------------------------------
//
// time(): static member function for performance benchmarking


void GpuDequantizationKernel::time()
{
    Dtype fp16(df_float, 16);
    Dtype fp32(df_float, 32);
    Dtype dt_int4 = Dtype::from_str("int4");
    
    // Time both float32 and float16
    for (int pass = 0; pass < 2; pass++) {
        Dtype dtype = (pass == 0) ? fp32 : fp16;
        
        // Choose F so that output array is 4 GB
        // float32: B × F × T × 4 = 4 GB  →  F = 2^18 = 262144
        // float16: B × F × T × 2 = 4 GB  →  F = 2^19 = 524288
        long B = 4;
        long T = 1024;
        long F = (dtype == fp32) ? 262144 : 524288;
        
        GpuDequantizationKernel kernel(dtype, B, F, T);
        
        // Allocate arrays
        Array<void> gin(dt_int4, {B, F, T}, af_gpu | af_zero);
        Array<void> gout(dtype, {B, F, T}, af_gpu | af_zero);
        
        // Print header
        double output_gb = double(B) * F * T * (dtype.nbits / 8) / 1.0e9;
        cout << "\nGpuDequantizationKernel::time()\n"
             << "    dtype = " << dtype << "\n"
             << "    shape = (" << B << ", " << F << ", " << T << ")\n"
             << "    output size = " << output_gb << " GB\n"
             << "    bandwidth per launch = " << (kernel.bw_per_launch.nbytes_gmem / 1.0e9) << " GB\n"
             << endl;
        
        // Use KernelTimer with 500 iterations, 2 streams
        int niter = 500;
        int print_interval = 50;
        KernelTimer kt(niter, 2);  // 2 streams for latency hiding
        
        while (kt.next()) {
            kernel.launch(gout, gin, kt.stream);
            
            if (kt.warmed_up && ((kt.curr_iteration+1) % print_interval == 0)) {
                double bandwidth_gbps = kernel.bw_per_launch.nbytes_gmem / kt.dt / 1.0e9;
                cout << "    iter " << (kt.curr_iteration+1) << "/" << niter
                     << ": dt = " << (kt.dt * 1.0e3) << " ms"
                     << ", bandwidth = " << bandwidth_gbps << " GB/s" << endl;
            }
        }
        
        // Final result
        double bandwidth_gbps = kernel.bw_per_launch.nbytes_gmem / kt.dt / 1.0e9;
        cout << "\n    Final: bandwidth = " << bandwidth_gbps << " GB/s"
             << " (theoretical ~900 GB/s A100, ~2000 GB/s H100)\n";
    }
}


}  // namespace pirate

