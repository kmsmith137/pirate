#ifndef _PIRATE_CHIMEFRB_BAD_CHANNEL_MASK_HPP
#define _PIRATE_CHIMEFRB_BAD_CHANNEL_MASK_HPP

#include <cstdint>
#include <cuda_runtime.h>
#include <ksgpu/Array.hpp>

namespace pirate {
namespace chimefrb {
#if 0
}}  // editor auto-indent
#endif


// GpuBadChannelMask: zeroes the weights of whole frequency channels. A port of
// rf_pipelines::badchannel_mask, which the old CHIME FRB search used at the start of its RFI
// chain to remove channels known in advance to be bad.
//
// The channels are given as a per-channel 'keep' array. The old transform takes a list of
// (freq_lo, freq_hi) MHz ranges instead, and turns them into channel indices with some fussy
// arithmetic. That conversion is in python -- the factory method
// GpuBadChannelMask.from_mask_ranges(), and badchannel_keep() in
// pirate_frb/chimefrb/ReferenceBadChannelMask.py -- and this class never sees a frequency.
//
// A masked channel is set to +0.0, whatever it held (the old code stores a literal 0, not a
// multiply). Every other weight is left bit-identical. The kernel writes only the zeros, so
// its cost is proportional to the number of masked channels.
//
// Nothing depends on time, so there is no chunking rule: T may be anything.
//
// See notes/chimefrb.md for the porting rules this class follows.

struct GpuBadChannelMask
{
    // 'keep' is a 1-d uint8 array IN HOST MEMORY, one entry per frequency channel: 0 means
    // "mask this channel", and any nonzero value means "keep it". Any stride. The constructor
    // copies it to the GPU, normalized to 0/1, so the caller may reuse or free its array
    // afterwards.
    //
    // Throws if 'keep' is not 1-d, is empty, or is not in host memory, and on an unsupported
    // warps_per_block.
    //
    // 'warps_per_block' is a performance knob, not a semantic one: it must not change the
    // result. Must be 4, 8, 16 or 32.
    //
    // The default is the smallest. A warp owns one channel, and masked channels usually come
    // in contiguous runs, so a large block piles a whole run's work onto a few SMs. On an
    // L40S, 4 measures fastest on a contiguous run and on an all-channel mask, and within 5%
    // of the fastest on the others, while 16 and 32 cost 20-25% on the contiguous run. Run
    // time_selected() on a new GPU before assuming it still holds.
    GpuBadChannelMask(const ksgpu::Array<uint8_t> &keep, long warps_per_block = 4);

    const long F;                       // number of frequency channels
    const long nmasked;                 // number of channels with keep == 0
    const ksgpu::Array<uint8_t> keep;   // shape (F,), IN GPU MEMORY, values 0 or 1
    const long warps_per_block;         // 4, 8, 16 or 32

    // launch(): asynchronously launch the kernel, and return without synchronizing the
    // stream. Note: stream=NULL is allowed, but is not the default.
    //
    //   weights   shape (B, F, T), float32, fully contiguous, in GPU memory. MODIFIED IN
    //             PLACE: every weight in a masked channel is set to +0.0, and every other
    //             weight is left bit-identical. B >= 1 and T >= 1 are otherwise arbitrary.
    //
    //   stream    CUDA stream.
    //
    // When nmasked == 0, launch() checks its argument and returns without launching.
    void launch(ksgpu::Array<float> &weights, cudaStream_t stream) const;

    // Static timing function (called via 'python -m pirate_frb time --cfrb').
    // Times the kernel at the production array size, for masks from one channel to all of
    // them, at every supported warps_per_block.
    static void time_selected();
};


}}  // namespace pirate::chimefrb

#endif  // _PIRATE_CHIMEFRB_BAD_CHANNEL_MASK_HPP
