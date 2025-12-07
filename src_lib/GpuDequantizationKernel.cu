#include "../include/pirate/GpuDequantizationKernel.hpp"

#include <iostream>
#include <cuda_fp16.h>
#include <ksgpu/xassert.hpp>
#include <ksgpu/cuda_utils.hpp>
#include <ksgpu/device_basics.hpp>  // FULL_MASK
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
// We use warp shuffles to transpose the data for coalesced writes.


// Float32 kernel: 8 shuffles, 8 coalesced write transactions
__global__ void gpu_dequantize_fp32_kernel(
    float *out,
    const uint32_t *in,
    long out_stride,   // stride between (beam,freq) rows in output (in float32 units)
    long in_stride)    // stride between (beam,freq) rows in input (in uint32 units)
{
    // Grid mapping: blockIdx = (time_chunk, freq, beam)
    int time_chunk = blockIdx.x;
    int freq = blockIdx.y;
    int beam = blockIdx.z;
    int thread_id = threadIdx.x;  // 0-31
    
    // Pointers to this warp's data
    // Input: 32 uint32 values = 128 bytes = 256 int4 values
    // Output: 256 float32 values
    long bf_idx = long(beam) * gridDim.y + freq;
    const uint32_t *inp = in + bf_idx * in_stride + time_chunk * 32;
    float *outp = out + bf_idx * out_stride + time_chunk * 256;
    
    // Step 1: Coalesced read (32 threads × 4 bytes = 128 bytes)
    uint32_t packed = inp[thread_id];
    
    // Step 2: Transpose using 8 shuffles
    // Thread t needs elements t, t+32, t+64, ..., t+224
    // Element (t + 32k) is in thread (t/8 + 4k), nibble position (t % 8)
    int base_src = thread_id / 8;      // source thread base: 0,0,0,0,0,0,0,0,1,1,...
    int nibble_pos = thread_id % 8;    // which nibble to extract: 0,1,2,3,4,5,6,7,0,1,...
    int shift = nibble_pos * 4;        // bit shift for nibble extraction
    
    float vals[8];
    
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        uint32_t shuffled = __shfl_sync(FULL_MASK, packed, base_src + 4*k);
        int nibble = (shuffled >> shift) & 0xF;
        int signed_val = (nibble >= 8) ? (nibble - 16) : nibble;
        vals[k] = (float) signed_val;
    }
    
    // Step 3: Coalesced writes (8 transactions × 32 floats = 256 floats)
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        outp[thread_id + 32*k] = vals[k];
    }
}


// Float16 kernel: 4 shuffles, 4 coalesced write transactions using __half2
__global__ void gpu_dequantize_fp16_kernel(
    __half2 *out,
    const uint32_t *in,
    long out_stride,   // stride between (beam,freq) rows in output (in __half2 units)
    long in_stride)    // stride between (beam,freq) rows in input (in uint32 units)
{
    // Grid mapping: blockIdx = (time_chunk, freq, beam)
    int time_chunk = blockIdx.x;
    int freq = blockIdx.y;
    int beam = blockIdx.z;
    int thread_id = threadIdx.x;  // 0-31
    
    // Pointers to this warp's data
    // Input: 32 uint32 values = 128 bytes = 256 int4 values
    // Output: 128 __half2 values = 256 float16 values
    long bf_idx = long(beam) * gridDim.y + freq;
    const uint32_t *inp = in + bf_idx * in_stride + time_chunk * 32;
    __half2 *outp = out + bf_idx * out_stride + time_chunk * 128;
    
    // Step 1: Coalesced read (32 threads × 4 bytes = 128 bytes)
    uint32_t packed = inp[thread_id];
    
    // Step 2: Transpose using 4 shuffles
    // Thread t writes __half2 containing elements (64k + 2t) and (64k + 2t + 1) for k=0,1,2,3
    // Element (64k + 2t) is in thread (8k + t/4), nibble position (2 * (t % 4))
    int base_src = thread_id / 4;        // source thread base: 0,0,0,0,1,1,1,1,...
    int pair_nibble = (thread_id % 4) * 2;  // nibble position of pair start: 0,2,4,6,0,2,4,6,...
    
    __half2 vals[4];
    
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        uint32_t shuffled = __shfl_sync(FULL_MASK, packed, base_src + 8*k);
        
        // Extract two consecutive nibbles
        int nib0 = (shuffled >> (pair_nibble * 4)) & 0xF;
        int nib1 = (shuffled >> ((pair_nibble + 1) * 4)) & 0xF;
        
        int val0 = (nib0 >= 8) ? (nib0 - 16) : nib0;
        int val1 = (nib1 >= 8) ? (nib1 - 16) : nib1;
        
        vals[k] = __halves2half2(__int2half_rn(val0), __int2half_rn(val1));
    }
    
    // Step 3: Coalesced writes (4 transactions × 32 __half2 = 128 __half2 = 256 float16)
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        outp[thread_id + 32*k] = vals[k];
    }
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
    nthreads = dim3(32, 1, 1);
    nblocks = dim3(ntime / 256, nfreq, nbeams);
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
        // Output stride in float32 units
        long out_stride = ntime;
        
        gpu_dequantize_fp32_kernel <<< nblocks, nthreads, 0, stream >>> (
            reinterpret_cast<float *>(out.data),
            reinterpret_cast<const uint32_t *>(in.data),
            out_stride,
            in_stride
        );
    }
    else if (dtype == fp16) {
        // Output stride in __half2 units (each __half2 holds 2 float16 values)
        long out_stride = ntime / 2;
        
        gpu_dequantize_fp16_kernel <<< nblocks, nthreads, 0, stream >>> (
            reinterpret_cast<__half2 *>(out.data),
            reinterpret_cast<const uint32_t *>(in.data),
            out_stride,
            in_stride
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
        
        // Use KernelTimer with 2 streams, 500 iterations
        int niter = 500;
        int print_interval = 50;
        KernelTimer kt(2);  // 2 streams for latency hiding
        
        for (int i = 0; i < niter; i++) {
            kernel.launch(gout, gin, kt.stream);
            
            if (kt.advance() && ((i+1) % print_interval == 0)) {
                double bandwidth_gbps = kernel.bw_per_launch.nbytes_gmem / kt.dt / 1.0e9;
                cout << "    iter " << (i+1) << "/" << niter
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

