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
// Threadblocking scheme (shared by both fp32 and fp16 kernels):
//
//   - All arrays are contiguous, so we flatten (beam, freq, time_chunk) into a
//     single "spectator" index 'spec' (nspec = nbeams * nfreq * (ntime/256))
//     and view the arrays as:
//
//        scales_offsets: (nspec, 2)   fp16   one (scale, offset) per spec
//        data:           (nspec, 256) int4   256 time samples per spec
//        out:            (nspec, 256) T      256 output samples per spec
//
//   - 1D grid, 1D blocks. blockDim.x = 256 (= 8 warps x 32 threads).
//     Each block handles 32 consecutive spectator indices.
//     gridDim.x = (nspec + 31) / 32.
//
//   - Warp w (0..7) of a block handles the 4 specs [32*b + 4*w, 32*b + 4*w + 4).
//     Within each spec, each of the 32 threads loads one uint32 (= 8 packed
//     int4 nibbles) and writes 8 output elements.
//
//   - Per-block global memory traffic:
//       scoff: 32 specs x 4 B (one __half2 each) = 128 B  = 1 cache line
//       data:  32 specs x 128 B (256 int4 each)  = 4096 B = 32 cache lines
//       out:   32 specs x 256 x sizeof(T)
//                 fp32 -> 32 KB = 256 cache lines
//                 fp16 -> 16 KB = 128 cache lines
//
// This pass focuses on the threadblocking scheme; the per-thread arithmetic is
// straightforward (1 uint32 load + 8 unrolled scalar stores). Perf tuning of
// the inner work -- wider stores, shuffle-based int4 -> fp conversion, etc.
// -- is deferred to a follow-up.


__launch_bounds__(256)
__global__ void gpu_dequantize_fp32_kernel(
    float *out,
    const __half2 *scoff,    // (nspec, 2) fp16 viewed as (nspec,) __half2
    const uint32_t *data,    // (nspec, 32) uint32 (each uint32 = 8 packed int4 nibbles)
    long nspec)
{
    int warp_id = threadIdx.x >> 5;   // 0..7
    int lane    = threadIdx.x & 31;   // 0..31

    long warp_first_spec = long(blockIdx.x) * 32 + warp_id * 4;

    // Length-4 loop over this warp's 4 spectator indices.
    for (int i = 0; i < 4; i++) {
        long spec = warp_first_spec + i;
        if (spec >= nspec)
            return;   // warp-uniform: all 32 lanes see the same spec/nspec

        // (scale, offset) for this spec. Convert fp16 -> fp32 BEFORE any
        // arithmetic, per GpuDequantizationKernel contract.
        __half2 sc_off = scoff[spec];
        float scale  = __half2float(__low2half(sc_off));
        float offset = __half2float(__high2half(sc_off));

        // Coalesced read: 32 threads x 4 B = 128 B = 1 cache line per spec.
        uint32_t packed = data[spec * 32 + lane];

        // Each thread writes 8 consecutive output elements (out[spec, lane*8 ..]).
        float *out_p = out + spec * 256 + lane * 8;
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            int nibble = (packed >> (k * 4)) & 0xF;
            int value  = (nibble >= 8) ? (nibble - 16) : nibble;
            out_p[k] = scale * (float) value + offset;
        }
    }
}


__launch_bounds__(256)
__global__ void gpu_dequantize_fp16_kernel(
    __half *out,
    const __half2 *scoff,
    const uint32_t *data,
    long nspec)
{
    int warp_id = threadIdx.x >> 5;
    int lane    = threadIdx.x & 31;

    long warp_first_spec = long(blockIdx.x) * 32 + warp_id * 4;

    for (int i = 0; i < 4; i++) {
        long spec = warp_first_spec + i;
        if (spec >= nspec)
            return;

        // Keep (scale, offset) in fp16 -- the fp16 kernel uses fp16 arithmetic
        // throughout, matching the output dtype.
        __half2 sc_off = scoff[spec];
        __half scale  = __low2half(sc_off);
        __half offset = __high2half(sc_off);

        uint32_t packed = data[spec * 32 + lane];

        __half *out_p = out + spec * 256 + lane * 8;
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            int nibble = (packed >> (k * 4)) & 0xF;
            int value  = (nibble >= 8) ? (nibble - 16) : nibble;
            __half h = __int2half_rn(value);
            out_p[k] = __hfma(scale, h, offset);
        }
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

    // Bandwidth: read int4 data (0.5 bytes per element) + read scales_offsets
    // (4 bytes per minichunk, i.e. per 256 time samples) + write output dtype.
    long nbytes_raw_in   = nbeams * nfreq * ntime / 2;
    long nbytes_scoff_in = nbeams * nfreq * (ntime / 256) * 2 * sizeof(__half);
    long nbytes_in       = nbytes_raw_in + nbytes_scoff_in;
    long nbytes_out      = nbeams * nfreq * ntime * (dtype.nbits / 8);
    resource_tracker.add_kernel("dequantization", nbytes_in + nbytes_out);

    // Threadblocking: 1D grid, 1D blocks of 256 threads (= 8 warps x 32 threads).
    // Each block processes 32 spectator indices, where
    //     nspec = nbeams * nfreq * (ntime / 256).
    // See the comment above the GPU kernels for the full scheme.
    long nspec = nbeams * nfreq * (ntime / 256);
    nthreads = dim3(256, 1, 1);
    nblocks  = dim3((nspec + 31) / 32, 1, 1);
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

    // Flatten (beam, freq, time_chunk) into a single 1D spectator index for
    // the GPU kernel. Since all arrays are contiguous, no strides are needed.
    long nspec = nbeams * nfreq * (ntime / 256);

    if (dtype == fp32) {
        gpu_dequantize_fp32_kernel <<< nblocks, nthreads, 0, stream >>> (
            reinterpret_cast<float *>(out.data),
            reinterpret_cast<const __half2 *>(scales_offsets.data),
            reinterpret_cast<const uint32_t *>(data.data),
            nspec
        );
    }
    else if (dtype == fp16) {
        gpu_dequantize_fp16_kernel <<< nblocks, nthreads, 0, stream >>> (
            reinterpret_cast<__half *>(out.data),
            reinterpret_cast<const __half2 *>(scales_offsets.data),
            reinterpret_cast<const uint32_t *>(data.data),
            nspec
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

