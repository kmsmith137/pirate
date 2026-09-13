#ifndef _PIRATE_CHIMEFRB_WT_UPSAMPLING_KERNEL_HPP
#define _PIRATE_CHIMEFRB_WT_UPSAMPLING_KERNEL_HPP

#include <cuda_runtime.h>
#include <ksgpu/Array.hpp>

namespace pirate {
namespace chimefrb {
#if 0
}}  // editor auto-indent
#endif


// GpuWtUpsamplingKernel: pushes a low-resolution weight mask back up to full resolution. A port
// of rf_kernels::weight_upsampler, which rf_pipelines::wi_sub_pipeline runs after its
// sub-pipeline: every full-resolution weight whose (Df x Dt) cell has a low-resolution weight
// w_lo <= w_cutoff is set to +0.0, and every other one is left bit-identical.
//
// The comparison is the old code's exactly: strict ('>' keeps, so w_lo == w_cutoff masks), in
// float32 against float(w_cutoff), and false for NaN, so a NaN w_lo masks its cell. Denormal
// weights are not supported: under --use_fast_math a denormal may compare as zero.
//
// The kernel writes only the zeros, and never reads the full-resolution weights, so its cost
// is one read of w_lores plus the masked cells. Nothing depends on where a chunk starts, so
// there is no chunking rule, and no shape needs to be a multiple of anything.
//
// See notes/chimefrb.md for the porting rules this class follows.

struct GpuWtUpsamplingKernel
{
    // Throws on Df < 1, Dt < 1, w_cutoff < 0 or NaN, and an unsupported warps_per_block.
    // (Df, Dt) = (1, 1) is allowed, and is not the identity: it zeroes w_hires wherever
    // w_lores <= w_cutoff.
    //
    // 'warps_per_block' is a performance knob, not a semantic one: it must not change the
    // result. Must be 4, 8, 16 or 32.
    //
    // The default is the largest, which measures fastest on an L40S whenever the kernel
    // actually stores anything: 10% over 4 warps on a contiguous run of masked channels, and
    // 14% with every cell masked. It is 5% SLOWER on a mask that stores nothing at all, where
    // the whole cost is the read and the launch. Run time_selected() on a new GPU before
    // assuming it still holds.
    GpuWtUpsamplingKernel(long Df, long Dt, double w_cutoff = 0.0, long warps_per_block = 32);

    const long Df;                 // frequency upsampling factor
    const long Dt;                 // time upsampling factor
    const double w_cutoff;         // a cell survives iff w_lo > float(w_cutoff)
    const long warps_per_block;    // 4, 8, 16 or 32

    // launch(): asynchronously launch the kernel, and return without synchronizing the
    // stream. Note: stream=NULL is allowed, but is not the default.
    //
    // Both arrays are float32, fully contiguous, and in GPU memory, and must not be the same
    // array. Note the order: the array that is modified comes first, as in the old code's
    // weight_upsampler::upsample(). At (Df, Dt) = (1, 1) the two shapes agree, so a swapped
    // call is not caught.
    //
    //   w_hires   shape (B, F_lo*Df, T_lo*Dt). MODIFIED IN PLACE: every weight in a masked
    //             cell is set to +0.0, and every other weight is left bit-identical. Never
    //             read.
    //
    //   w_lores   shape (B, F_lo, T_lo), any B, F_lo, T_lo >= 1. Read only.
    //
    //   stream    CUDA stream.
    void launch(ksgpu::Array<float> &w_hires,
                const ksgpu::Array<float> &w_lores,
                cudaStream_t stream) const;

    // Static timing function (called via 'python -m pirate_frb time --cfrb'). Times the
    // production configuration, (Df, Dt) = (16, 1) at 8 beams, for masks from nothing to
    // everything, at every supported warps_per_block.
    static void time_selected();
};


#ifdef __CUDACC__

// zero_cell(): zero the Df-by-Dt block of full-resolution weights behind one low-resolution
// cell (b, f_lo, t_lo). 'w_hires' is the whole (B, F_lo*Df, T_lo*Dt) contiguous array.
//
// This is a one-cell weight upsample, so it lives here: GpuWtUpsamplingKernel's kernel calls it
// for every cell that fails its test, and the clippers' final kernels (IntensityClipper.cu,
// StdDevClipper.cu) call it for every downsampled cell they mask.
//
// The stores are 32-bit and, for Dt > 1, strided when a warp calls this from 32 consecutive
// cells: its lanes are Dt*4 bytes apart, so one instruction touches up to 32 sectors rather
// than one cache line. Deliberate. Callers zero only cells that are being masked, and when a
// whole row is masked the warp's Dt stores together cover 32*Dt*4 contiguous bytes anyway.
// GpuWiDownsamplingKernel reads through exactly this pattern at 657 GB/s.
__device__ __forceinline__ void zero_cell(float *w_hires, long b, long f_lo, long t_lo,
                                          long F_lo, long T_lo, int Df, int Dt)
{
    // Apply per-cell pointer offset.
    //   before: shape (B, F_lo*Df, T_lo*Dt), contiguous
    //   after: shape (Df, Dt), strides (T_lo*Dt, 1)
    const long T = T_lo * Dt;
    float *wp = w_hires + ((b*F_lo + f_lo)*Df)*T + t_lo*Dt;

    for (int df = 0; df < Df; df++) {
        for (int dt = 0; dt < Dt; dt++)
            wp[dt] = 0.0f;
        wp += T;
    }
}

#endif  // __CUDACC__


}}  // namespace pirate::chimefrb

#endif  // _PIRATE_CHIMEFRB_WT_UPSAMPLING_KERNEL_HPP
