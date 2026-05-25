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
// One (scale, offset) fp16 pair is shared by the 256 samples in the chunk; the
// pair is loaded once per warp as a single __half2.
//
// Perf optimization (e.g. wider scoff load patterns, scoff caching across
// time_chunks for the same (beam, freq)) is deferred -- this pass is for
// correctness only.


// Float32 kernel: 2 shuffles, 2 x 128-bit write transactions using float4
// Block dims: (32, 8, 1) - threadIdx.y runs over frequency for better occupancy
// scales_offsets is converted from fp16 to fp32 immediately, before any arithmetic.
__global__ void gpu_dequantize_fp32_kernel(
    float4 *out,
    const __half2 *scoff,   // packed (scale, offset) pairs
    const uint32_t *in,
    long out_stride,        // stride between (beam,freq) rows in output (in float4 units)
    long scoff_stride,      // stride between (beam,freq) rows in scoff (in __half2 / pair units)
    long in_stride,         // stride between (beam,freq) rows in input (in uint32 units)
    int nfreq)              // total number of frequencies (for bounds checking)
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

    // Load the (scale, offset) pair for this (beam, freq, time_chunk) and convert
    // fp16 -> fp32 BEFORE any arithmetic.
    __half2 sc_off = scoff[bf_idx * scoff_stride + time_chunk];
    float scale  = __half2float(__low2half(sc_off));
    float offset = __half2float(__high2half(sc_off));

    // Step 1: Coalesced read (32 threads x 4 bytes = 128 bytes)
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

    // Extract 4 nibbles from data_lo, convert to float4 with scale*x + offset
    int nib0 = (data_lo >> (nibble_base * 4)) & 0xF;
    int nib1 = (data_lo >> ((nibble_base + 1) * 4)) & 0xF;
    int nib2 = (data_lo >> ((nibble_base + 2) * 4)) & 0xF;
    int nib3 = (data_lo >> ((nibble_base + 3) * 4)) & 0xF;

    float4 out_lo;
    out_lo.x = scale * (float)((nib0 >= 8) ? (nib0 - 16) : nib0) + offset;
    out_lo.y = scale * (float)((nib1 >= 8) ? (nib1 - 16) : nib1) + offset;
    out_lo.z = scale * (float)((nib2 >= 8) ? (nib2 - 16) : nib2) + offset;
    out_lo.w = scale * (float)((nib3 >= 8) ? (nib3 - 16) : nib3) + offset;

    // Extract 4 nibbles from data_hi, convert to float4 with scale*x + offset
    nib0 = (data_hi >> (nibble_base * 4)) & 0xF;
    nib1 = (data_hi >> ((nibble_base + 1) * 4)) & 0xF;
    nib2 = (data_hi >> ((nibble_base + 2) * 4)) & 0xF;
    nib3 = (data_hi >> ((nibble_base + 3) * 4)) & 0xF;

    float4 out_hi;
    out_hi.x = scale * (float)((nib0 >= 8) ? (nib0 - 16) : nib0) + offset;
    out_hi.y = scale * (float)((nib1 >= 8) ? (nib1 - 16) : nib1) + offset;
    out_hi.z = scale * (float)((nib2 >= 8) ? (nib2 - 16) : nib2) + offset;
    out_hi.w = scale * (float)((nib3 >= 8) ? (nib3 - 16) : nib3) + offset;

    // Step 3: 128-bit coalesced writes (2 transactions x 512 bytes = 1024 bytes = 256 floats)
    outp[thread_id] = out_lo;        // elements [4t, 4t+3]
    outp[32 + thread_id] = out_hi;   // elements [128+4t, 128+4t+3]
}


// Float16 kernel: 0 shuffles, 1 x 128-bit write transaction using uint4
// Thread t already has elements [8t, 8t+7] after the read - no shuffle needed!
// Block dims: (32, 8, 1) - threadIdx.y runs over frequency for better occupancy
// scales_offsets is kept in fp16 throughout (matches output dtype).
__global__ void gpu_dequantize_fp16_kernel(
    uint4 *out,
    const __half2 *scoff,   // packed (scale, offset) pairs
    const uint32_t *in,
    long out_stride,        // stride between (beam,freq) rows in output (in uint4 units)
    long scoff_stride,      // stride between (beam,freq) rows in scoff (in __half2 / pair units)
    long in_stride,         // stride between (beam,freq) rows in input (in uint32 units)
    int nfreq)              // total number of frequencies (for bounds checking)
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

    // Load the (scale, offset) pair for this (beam, freq, time_chunk). Keep in fp16
    // and broadcast each into a __half2 for fp16 fused-multiply-add.
    __half2 sc_off = scoff[bf_idx * scoff_stride + time_chunk];
    __half2 scale2  = __half2half2(__low2half(sc_off));   // (scale, scale)
    __half2 offset2 = __half2half2(__high2half(sc_off));  // (offset, offset)

    // Step 1: Coalesced read (32 threads x 4 bytes = 128 bytes)
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

    // Apply affine transform in fp16: h = scale * h + offset
    h01 = __hfma2(h01, scale2, offset2);
    h23 = __hfma2(h23, scale2, offset2);
    h45 = __hfma2(h45, scale2, offset2);
    h67 = __hfma2(h67, scale2, offset2);

    // Step 3: Single 128-bit coalesced write (1 transaction x 512 bytes = 256 float16)
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

    // Bandwidth: read int4 data (0.5 bytes per element) + read scales_offsets
    // (4 bytes per minichunk, i.e. per 256 time samples) + write output dtype.
    long nbytes_raw_in   = nbeams * nfreq * ntime / 2;
    long nbytes_scoff_in = nbeams * nfreq * (ntime / 256) * 2 * sizeof(__half);
    long nbytes_in       = nbytes_raw_in + nbytes_scoff_in;
    long nbytes_out      = nbeams * nfreq * ntime * (dtype.nbits / 8);
    resource_tracker.add_kernel("dequantization", nbytes_in + nbytes_out);

    // Kernel config: each warp handles 256 time samples for one (beam, freq)
    // Block: (32, 8, 1) - threadIdx.y runs over frequency for better occupancy
    // Grid: (ceil(freq/8), time_chunk, beam) - freq in x since it can be large
    nthreads = dim3(32, 8, 1);
    nblocks = dim3((nfreq + 7) / 8, ntime / 256, nbeams);
}


// -------------------------------------------------------------------------------------------------
//
// apply_reference(): CPU reference implementation


void GpuDequantizationKernel::apply_reference(Array<float> &out,
                                              const Array<__half> &scales_offsets,
                                              const Array<void> &data) const
{
    Dtype dt_int4 = Dtype::from_str("int4");

    // Validate scales_offsets (dtype enforced by Array<__half>)
    xassert(scales_offsets.on_host());
    xassert_shape_eq(scales_offsets, ({nbeams, nfreq, ntime/256, 2}));
    xassert(scales_offsets.is_fully_contiguous());

    // Validate data
    xassert(data.on_host());
    xassert(data.dtype == dt_int4);
    xassert_shape_eq(data, ({nbeams, nfreq, ntime}));
    xassert(data.is_fully_contiguous());

    // Validate output
    xassert(out.on_host());
    xassert_shape_eq(out, ({nbeams, nfreq, ntime}));
    xassert(out.is_fully_contiguous());

    const unsigned char *datap = (const unsigned char *) data.data;
    const __half *scoffp = scales_offsets.data;
    float *outp = out.data;

    long nchunks = ntime / 256;

    for (long b = 0; b < nbeams; b++) {
        for (long f = 0; f < nfreq; f++) {
            for (long c = 0; c < nchunks; c++) {
                long pair_idx = (b * nfreq + f) * nchunks + c;
                // Convert fp16 -> fp32 immediately, before any arithmetic.
                float scale  = __half2float(scoffp[2 * pair_idx + 0]);
                float offset = __half2float(scoffp[2 * pair_idx + 1]);

                long base = ((b * nfreq + f) * nchunks + c) * 256;
                for (long it = 0; it < 256; it++) {
                    long i = base + it;

                    // Extract nibble (low nibble = even index, high nibble = odd index)
                    long byte_idx = i / 2;
                    int nibble_idx = i % 2;
                    unsigned char byte = datap[byte_idx];
                    int nibble = (nibble_idx == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);

                    // Convert to signed two's complement: range [-8, 7]
                    int value = (nibble >= 8) ? (nibble - 16) : nibble;

                    outp[i] = scale * (float) value + offset;
                }
            }
        }
    }
}


// -------------------------------------------------------------------------------------------------
//
// launch(): dispatch to appropriate GPU kernel


void GpuDequantizationKernel::launch(Array<void> &out,
                                     const Array<__half> &scales_offsets,
                                     const Array<void> &data,
                                     cudaStream_t stream) const
{
    Dtype dt_int4 = Dtype::from_str("int4");
    Dtype fp16(df_float, 16);
    Dtype fp32(df_float, 32);

    // Validate data
    xassert(data.on_gpu());
    xassert(data.dtype == dt_int4);
    xassert_shape_eq(data, ({nbeams, nfreq, ntime}));
    xassert(data.is_fully_contiguous());

    // Validate scales_offsets (dtype enforced by Array<__half>)
    xassert(scales_offsets.on_gpu());
    xassert_shape_eq(scales_offsets, ({nbeams, nfreq, ntime/256, 2}));
    xassert(scales_offsets.is_fully_contiguous());

    // Validate output
    xassert(out.on_gpu());
    xassert(out.dtype == dtype);
    xassert_shape_eq(out, ({nbeams, nfreq, ntime}));
    xassert(out.is_fully_contiguous());

    // Input stride: stride between (beam,freq) rows in uint32 units
    // Each row has ntime int4 elements = ntime/2 bytes = ntime/8 uint32 values
    long in_stride = ntime / 8;

    // scales_offsets stride: stride between (beam,freq) rows in __half2 / pair units.
    // Each row has (ntime/256) (scale, offset) pairs.
    long scoff_stride = ntime / 256;

    if (dtype == fp32) {
        // Output stride in float4 units (each float4 holds 4 float32 values)
        long out_stride = ntime / 4;

        gpu_dequantize_fp32_kernel <<< nblocks, nthreads, 0, stream >>> (
            reinterpret_cast<float4 *>(out.data),
            reinterpret_cast<const __half2 *>(scales_offsets.data),
            reinterpret_cast<const uint32_t *>(data.data),
            out_stride,
            scoff_stride,
            in_stride,
            nfreq
        );
    }
    else if (dtype == fp16) {
        // Output stride in uint4 units (each uint4 holds 8 float16 values)
        long out_stride = ntime / 8;

        gpu_dequantize_fp16_kernel <<< nblocks, nthreads, 0, stream >>> (
            reinterpret_cast<uint4 *>(out.data),
            reinterpret_cast<const __half2 *>(scales_offsets.data),
            reinterpret_cast<const uint32_t *>(data.data),
            out_stride,
            scoff_stride,
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
// convert_uint8_to_int4(): helper for pybind11 wrappers


Array<void> GpuDequantizationKernel::convert_uint8_to_int4(const Array<void> &in_uint8) const
{
    Dtype dtype_uint8(df_uint, 8);
    xassert_eq(in_uint8.dtype, dtype_uint8);
    xassert_shape_eq(in_uint8, ({nbeams, nfreq, ntime/2}));
    xassert(in_uint8.is_fully_contiguous());

    // Reinterpret uint8 (nbeams, nfreq, ntime/2) as int4 (nbeams, nfreq, ntime)
    Dtype dtype_int4(df_int, 4);
    Array<void> in_int4;
    in_int4.data = in_uint8.data;
    in_int4.ndim = 3;
    in_int4.shape[0] = nbeams;
    in_int4.shape[1] = nfreq;
    in_int4.shape[2] = ntime;
    in_int4.size = nbeams * nfreq * ntime;
    in_int4.strides[0] = nfreq * ntime;
    in_int4.strides[1] = ntime;
    in_int4.strides[2] = 1;
    in_int4.dtype = dtype_int4;
    in_int4.aflags = in_uint8.aflags;
    in_int4.base = in_uint8.base;
    in_int4.check_invariants("GpuDequantizationKernel::convert_uint8_to_int4");

    return in_int4;
}


// -------------------------------------------------------------------------------------------------
//
// test(): static member function for unit testing


void GpuDequantizationKernel::test_random()
{
    // Random parameters
    Dtype dtype = (rand_uniform() < 0.5) ? Dtype(df_float,16) : Dtype(df_float,32);

    auto v = random_integers_with_bounded_product(3, 1000);
    long B = v[0];        // beams
    long F = v[1];        // freqs
    long T = v[2] * 256;  // time (multiple of 256)

    cout << "GpuDequantizationKernel::test()\n"
         << "    dtype = " << dtype << "\n"
         << "    nbeams = " << B << "\n"
         << "    nfreq = " << F << "\n"
         << "    ntime = " << T << endl;

    // Create int4 data with random values
    Dtype dt_int4 = Dtype::from_str("int4");
    Array<void> h_data(dt_int4, {B, F, T}, af_rhost);

    // Fill with random int4 values (0-15 per nibble, will be interpreted as signed)
    unsigned char *dp = (unsigned char *) h_data.data;
    long nbytes = B * F * T / 2;
    for (long i = 0; i < nbytes; i++)
        dp[i] = rand_int(0, 256);  // random byte = 2 random int4 values

    // Create scales_offsets with random values: scales in [0, 1], offsets in [-1, 1].
    Array<__half> h_scoff({B, F, T/256, 2}, af_rhost);
    __half *sp = h_scoff.data;
    long npair = B * F * (T/256);
    for (long i = 0; i < npair; i++) {
        sp[2*i + 0] = __float2half(rand_uniform(0.0, 1.0));    // scale
        sp[2*i + 1] = __float2half(rand_uniform(-1.0, 1.0));   // offset
    }

    // Reference output (always float32)
    Array<float> href({B, F, T}, af_rhost | af_zero);

    // GPU inputs/output
    Array<void>   g_data  = h_data.to_gpu();
    Array<__half> g_scoff = h_scoff.to_gpu();
    Array<void>   gout(dtype, {B, F, T}, af_gpu | af_zero);

    // Run reference
    GpuDequantizationKernel kernel(dtype, B, F, T);
    kernel.apply_reference(href, h_scoff, h_data);

    // Run GPU kernel
    kernel.launch(gout, g_scoff, g_data, nullptr);

    // Compare
    assert_arrays_equal(href, gout, "href", "gout", {"b","f","t"});
}


// -------------------------------------------------------------------------------------------------
//
// time(): static member function for performance benchmarking


void GpuDequantizationKernel::time_selected()
{
    Dtype fp16(df_float, 16);
    Dtype fp32(df_float, 32);
    Dtype dt_int4 = Dtype::from_str("int4");

    // Time both float32 and float16
    for (int pass = 0; pass < 2; pass++) {
        Dtype dtype = (pass == 0) ? fp32 : fp16;

        // Choose F so that output array is 4 GB
        // float32: B x F x T x 4 = 4 GB  ->  F = 2^18 = 262144
        // float16: B x F x T x 2 = 4 GB  ->  F = 2^19 = 524288
        long B = 4;
        long T = 1024;
        long F = (dtype == fp32) ? 262144 : 524288;

        GpuDequantizationKernel kernel(dtype, B, F, T);

        // Allocate arrays. scales_offsets is initialized to zero, per spec.
        Array<void>   gin(dt_int4, {B, F, T}, af_gpu | af_zero);
        Array<__half> g_scoff({B, F, T/256, 2}, af_gpu | af_zero);
        Array<void>   gout(dtype, {B, F, T}, af_gpu | af_zero);

        // Print header
        double output_gb = double(B) * F * T * (dtype.nbits / 8) / 1.0e9;
        cout << "\nGpuDequantizationKernel::time()\n"
             << "    dtype = " << dtype << "\n"
             << "    shape = (" << B << ", " << F << ", " << T << ")\n"
             << "    output size = " << output_gb << " GB\n"
             << "    bandwidth per launch = " << (kernel.resource_tracker.get_gmem_bw() / 1.0e9) << " GB\n"
             << endl;

        // Use KernelTimer with 500 iterations, 2 streams
        int niter = 500;
        int print_interval = 50;
        KernelTimer kt(niter, 2);  // 2 streams for latency hiding

        while (kt.next()) {
            kernel.launch(gout, g_scoff, gin, kt.stream);

            if (kt.warmed_up && ((kt.curr_iteration+1) % print_interval == 0)) {
                double bandwidth_gbps = kernel.resource_tracker.get_gmem_bw()  / kt.dt / 1.0e9;
                cout << "    iter " << (kt.curr_iteration+1) << "/" << niter
                     << ": dt = " << (kt.dt * 1.0e3) << " ms"
                     << ", bandwidth = " << bandwidth_gbps << " GB/s" << endl;
            }
        }
    }
}


}  // namespace pirate

