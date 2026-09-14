#ifndef _PIRATE_CHIMEFRB_RFI_MASK_PACKING_KERNEL_HPP
#define _PIRATE_CHIMEFRB_RFI_MASK_PACKING_KERNEL_HPP

#include <cstdint>
#include <cuda_runtime.h>
#include <ksgpu/Array.hpp>

namespace pirate {
namespace chimefrb {
#if 0
}}  // editor auto-indent
#endif


// RfiMaskPackingKernel: turns the weights an RFI chain leaves behind into the bit-packed
// RFI mask that a chimefrb data file carries, on the GPU.
//
//   rfi_mask[f, t/8] bit (t%8) = 1  if  weights[f,t] > 0,  else 0
//
// A SET bit means GOOD data, and the packing is LSB-FIRST: bit i of byte j is time sample
// 8*j+i. That is the convention every real file uses (AssembledChunk.hpp says the same thing
// from the reading side), and it is not a choice we get to make -- see below.
//
// This is the GPU version of rf_kernels::mask_counter_data, whose scalar reference
// slow_reference_mask_count() is the statement of the format:
//
//     for (int ibit = 0; ibit < 8; ibit++)
//         if (in[ifreq*istride + 8*ibyte + ibit] > 0.0f)
//             byte |= (1 << ibit);
//
// rf_pipelines::mask_counter_transform pointed that kernel's output straight at a live
// assembled_chunk's mask buffer, so it is literally the code that packed every rfi_mask in
// every file we read. The threshold is STRICTLY > 0, which makes NaN and -0.0 masked.
//
// Together with ChimeDequantizationKernel this pair brackets a ported chain: that one
// consumes a packed mask and produces (intensity, weights), this one consumes the weights
// and produces a packed mask. Neither is a GpuTransform, and neither appears in a chain.
//
// EXACT AGREEMENT WITH THE CPU, WITH ONE EXCEPTION. Packing is a comparison against zero, not
// arithmetic, so there is nothing to round and the two agree bit for bit -- including NaN,
// -0.0 and the infinities. The exception is a positive DENORMAL weight (below 1.18e-38):
// pirate builds with --use_fast_math, so the GPU comparison flushes it to zero and calls the
// sample bad, where the CPU calls it good. Chain weights are 0, 1, or averages of those, so
// this cannot arise in practice; the unit test excludes denormals for this reason and no
// other.
//
// PERFORMANCE. The kernel reads 32 bytes for every byte it writes, so it is read-bound, and
// measured on an L40S at (nfreq, nt) = (16384, 4096) it runs AT the read ceiling: 730 GB/s,
// against 729 GB/s for a kernel that reads the same array with float4 and does nothing else.
// There is no headroom to chase, which is why the loads are plain 32-bit -- a warp's 32 lanes
// already cover one full cache line per instruction. Packing a chunk at the production
// geometry (1024 x 1024) takes 4-6 us, dominated by launch overhead rather than memory.
//
// See notes/chimefrb.md for the porting rules this class follows.

struct RfiMaskPackingKernel
{
    // Throws unless nfreq and nt are positive and nt is a multiple of 1024.
    //
    // The 1024 is the kernel's tiling, not a property of the data: one warp packs 1024 time
    // samples into the 128 bytes that are exactly one cache line (see the .cu file). Every
    // chimefrb config we have runs the mask counter at nt_chunk = 1024, so it costs a caller
    // nothing.
    //
    // 'warps_per_block' is a performance knob, not a semantic one: it must not change the
    // result. Must be 4, 8, 16 or 32. Measured on an L40S it makes no difference worth
    // caring about (see the note above time_selected()), so the default just matches
    // ChimeDequantizationKernel's.
    RfiMaskPackingKernel(long nfreq, long nt, long warps_per_block = 32);

    const long nfreq;             // frequency channels (no downsampling: the mask has nfreq rows too)
    const long nt;                // time samples, a multiple of 1024
    const long warps_per_block;   // 4, 8, 16 or 32

    // launch(): asynchronously launch the kernel, and return without synchronizing the
    // stream. Note: stream=NULL is allowed, but is not the default.
    //
    // Both arrays are in GPU memory, and they must not be the same array.
    //
    //   rfi_mask    shape (nfreq, nt/8), uint8, FULLY CONTIGUOUS. Fully overwritten. The
    //               contiguity is what makes each row start on a 128-byte boundary, which is
    //               what lets a warp store its whole tile in one cache-line-aligned
    //               instruction. A chunk's mask buffer is contiguous, so this costs nothing.
    //
    //   weights     shape (nfreq, nt), float32. Read only. PARTIALLY CONTIGUOUS: the time
    //               stride must be 1, but the frequency stride is free (>= nt), so a caller
    //               can pack a column window of a wider pipeline block -- which is the point,
    //               since the chain runs on blocks and the mask is written per chunk.
    //
    //   stream      CUDA stream.
    void launch(ksgpu::Array<uint8_t> &rfi_mask,
                const ksgpu::Array<float> &weights,
                cudaStream_t stream) const;

    // Static timing function (called via 'python -m pirate_frb time --cfrb'). Times the
    // production CHIME geometry and two larger ones, at every supported warps_per_block, and
    // reports the bandwidth achieved against a kernel that only reads the input -- the
    // ceiling, since this kernel reads 32 bytes for every byte it writes.
    static void time_selected();
};


}}  // namespace pirate::chimefrb

#endif // _PIRATE_CHIMEFRB_RFI_MASK_PACKING_KERNEL_HPP
