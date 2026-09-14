#ifndef _PIRATE_CHIMEFRB_CHIME_DEQUANTIZATION_KERNEL_HPP
#define _PIRATE_CHIMEFRB_CHIME_DEQUANTIZATION_KERNEL_HPP

#include <cstdint>
#include <cuda_runtime.h>
#include <ksgpu/Array.hpp>

namespace pirate {
namespace chimefrb {
#if 0
}}  // editor auto-indent
#endif


// ChimeDequantizationKernel: turns one AssembledChunk's raw arrays into the (intensity,
// weights) pair the ported RFI chain runs on, on the GPU.
//
//   intensity[f,t] = (scales[ifc,itc] * scale) * data[f,t] + (offsets[ifc,itc] * scale)
//   weights[f,t]   = 0 where data is 0 or 255 (the saturation sentinels), else 1
//
// where ifc = f/nupfreq and itc = t/16, and where -- if 'apply_rfimask' is true -- both
// outputs are instead +0.0 wherever the file's RFI mask marks the sample bad. 'scale' is the
// CHIME L1 server's 'intensity_prescale' (1e-4 in production), applied the way its decode
// applied it: the two products are rounded to float32 first, then one fused multiply-add.
// With scale = 1 the products are exact and nothing changes.
//
// This is the GPU version of AssembledChunk::decode_intensity() and decode_weights(), whose
// header comment is the reference for the semantics (frequency ordering, what the RFI mask's
// polarity and resolution are, why apply_rfimask has no default). It agrees with them BIT
// FOR BIT: the intensity is one fused multiply-add on both sides, and masking is a select
// rather than a multiply, so a masked sample is +0.0 whatever it held. Note the class name:
// pirate::GpuDequantizationKernel is a different operation on a different data format.
//
// The kernel does NOT copy anything to the GPU. Its inputs are GPU arrays; getting a chunk
// there is the caller's job.
//
// See notes/chimefrb.md for the porting rules this class follows.

struct ChimeDequantizationKernel
{
    // Throws unless nfreq, nt, nfreq_coarse and nt_coarse are positive, nfreq is a multiple
    // of nfreq_coarse, and nt == 16*nt_coarse.
    //
    // The last of those is nt_per_packet == 16 -- the number of time samples sharing one
    // (scale, offset) pair -- which the kernel requires and every file we have satisfies.
    // AssembledChunk's parser accepts any value; only the kernels are specialized. Passing
    // nt and nt_coarse separately, rather than deriving one from the other, is what lets the
    // constructor say so when a caller's chunk has some other value.
    //
    // 'warps_per_block' is a performance knob, not a semantic one: it must not change the
    // result. Must be 4, 8, 16 or 32. See time_selected() before changing the default.
    ChimeDequantizationKernel(long nfreq, long nt, long nfreq_coarse, long nt_coarse,
                              long warps_per_block = 32);

    const long nfreq;             // fine frequency channels
    const long nt;                // time samples, == 16*nt_coarse
    const long nfreq_coarse;      // coarse channels: one (scale, offset) row each
    const long nt_coarse;         // (scale, offset) columns, == nt/16
    const long nupfreq;           // fine channels per coarse channel, == nfreq/nfreq_coarse
    const long warps_per_block;   // 4, 8, 16 or 32

    // launch(): asynchronously launch the kernel, and return without synchronizing the
    // stream. Note: stream=NULL is allowed, but is not the default.
    //
    // All six arrays are in GPU memory, and the outputs must not overlap the inputs.
    //
    //   intensity   shape (nfreq, nt), float32. Fully overwritten. PARTIALLY CONTIGUOUS: the
    //               time stride must be 1, but the frequency stride is free (>= nt), so a
    //               caller can write into a column slice of a larger block -- which is the
    //               point, since one chunk is a fraction of a pipeline block's time axis.
    //
    //   weights     shape (nfreq, nt), float32, same rules as 'intensity' and its own
    //               frequency stride. Fully overwritten. Must not be 'intensity'.
    //
    //   scales      shape (nfreq_coarse, nt_coarse), float32, fully contiguous. Read only.
    //   offsets     shape (nfreq_coarse, nt_coarse), float32, fully contiguous. Read only.
    //   data        shape (nfreq, nt), uint8, fully contiguous. Read only.
    //
    //   rfi_mask    shape (nrfifreq, nt/8), uint8, fully contiguous, bit-packed LSB-first
    //               with a SET bit meaning GOOD data (AssembledChunk.hpp). nrfifreq must
    //               divide nfreq; each mask row covers nfreq/nrfifreq fine channels. IGNORED
    //               when 'apply_rfimask' is false, and may then be an empty array.
    //
    //   stream      CUDA stream.
    //   scale       multiplies scales and offsets before the decode (see the class comment).
    //
    //   stream      CUDA stream.
    void launch(ksgpu::Array<float> &intensity,
                ksgpu::Array<float> &weights,
                const ksgpu::Array<float> &scales,
                const ksgpu::Array<float> &offsets,
                const ksgpu::Array<uint8_t> &data,
                const ksgpu::Array<uint8_t> &rfi_mask,
                bool apply_rfimask,
                float scale,
                cudaStream_t stream) const;

    // Static timing function (called via 'python -m pirate_frb time --cfrb'). Times the
    // production CHIME geometry with and without the RFI mask, at every supported
    // warps_per_block, and reports the bandwidth achieved against a cudaMemset of the two
    // output arrays -- the ceiling, since this kernel is store-bound.
    static void time_selected();
};


}}  // namespace pirate::chimefrb

#endif // _PIRATE_CHIMEFRB_CHIME_DEQUANTIZATION_KERNEL_HPP
