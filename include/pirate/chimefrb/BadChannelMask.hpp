#ifndef _PIRATE_CHIMEFRB_BAD_CHANNEL_MASK_HPP
#define _PIRATE_CHIMEFRB_BAD_CHANNEL_MASK_HPP

#include <cstdint>
#include <utility>
#include <vector>
#include <cuda_runtime.h>
#include <ksgpu/Array.hpp>

#include "TransformBase.hpp"

namespace pirate {
namespace chimefrb {
#if 0
}}  // editor auto-indent
#endif


// GpuBadChannelMask: zeroes the weights of whole frequency channels. A port of
// rf_pipelines::badchannel_mask, which the old CHIME FRB search used at the start of its RFI
// chain to remove channels known in advance to be bad.
//
// As in the old transform, the channels are given as a list of (lo, hi) frequency ranges in
// MHz, together with the band (freq_lo, freq_hi) that the nfreq channels span, channel 0 at
// the top. The constructor converts the ranges to a per-channel 'keep' array with the old
// code's arithmetic -- including its quirks, which are stated in the docstring of the python
// transcription of that arithmetic, badchannel_keep() in
// pirate_frb/chimefrb/ReferenceBadChannelMask.py. The unit test checks the two agree.
//
// A masked channel is set to +0.0, whatever it held (the old code stores a literal 0, not a
// multiply). Every other weight is left bit-identical. The kernel writes only the zeros, so
// its cost is proportional to the number of masked channels.
//
// Nothing depends on time, so ntime may be anything. The intensity is never touched: it is
// a launch() argument only because every chimefrb transform takes the same four
// (see GpuTransformBase in TransformBase.hpp).
//
// See notes/chimefrb.md for the porting rules this class follows.

struct GpuBadChannelMask : public GpuTransformBase
{
    // (nbeams, nfreq, ntime) is the shape of the arrays launch() will be given.
    //
    // 'mask_ranges' are (lo, hi) pairs in MHz, each with lo < hi, in any order; overlaps are
    // fine. 'freq_range' is (lo, hi) of the band, 400 and 800 for CHIME.
    //
    // Throws on nbeams, nfreq or ntime < 1; on a range with lo >= hi, or one that lies
    // entirely outside the band or strictly covers it (the old code refuses both; to mask
    // every channel, pass the band itself); on freq_range with lo >= hi; and on an
    // unsupported warps_per_block.
    //
    // 'warps_per_block' is a performance knob, not a semantic one: it must not change the
    // result. Must be 4, 8, 16 or 32.
    //
    // The default is the smallest. A warp owns one channel, and masked channels usually come
    // in contiguous runs, so a large block piles a whole run's work onto a few SMs. On an
    // L40S, 4 measures fastest on a contiguous run and on an all-channel mask, and within 5%
    // of the fastest on the others, while 16 and 32 cost 20-25% on the contiguous run. Run
    // time_selected() on a new GPU before assuming it still holds.
    GpuBadChannelMask(long nbeams, long nfreq, long ntime,
                      const std::vector<std::pair<double,double>> &mask_ranges,
                      std::pair<double,double> freq_range,
                      long warps_per_block = 4);

    // Inherited from GpuTransformBase: nbeams, nfreq, ntime (the array shape), scratch_nelts
    // (always 0: the kernel needs no scratch), and launch().
    const std::vector<std::pair<double,double>> mask_ranges;   // MHz, as given
    const std::pair<double,double> freq_range;                 // (lo, hi) MHz of the band
    const long warps_per_block;                                // 4, 8, 16 or 32
    const long nmasked;                                        // number of channels with keep == 0
    const ksgpu::Array<uint8_t> keep;                          // shape (nfreq,), IN GPU MEMORY, 0 or 1

    // launch_checked(): asynchronously launch the kernel, and return without synchronizing
    // the stream. Called by GpuTransformBase::launch(), which checks the arguments first.
    //
    //   intensity  shape (nbeams, nfreq, ntime), float32, fully contiguous, in GPU memory.
    //              Never touched.
    //
    //   weights    same shape and layout. MODIFIED IN PLACE: every weight in a masked
    //              channel is set to +0.0, and every other weight is left bit-identical.
    //
    //   scratch    Unused: scratch_nelts == 0.
    //
    //   stream     CUDA stream.
    //
    // When nmasked == 0, returns without launching.
    void launch_checked(ksgpu::Array<float> &intensity, ksgpu::Array<float> &weights,
                        ksgpu::Array<float> &scratch, cudaStream_t stream) const override;

    // Static timing function (called via 'python -m pirate_frb time --cfrb').
    // Times the kernel at the production array size, for masks from one channel to all of
    // them, at every supported warps_per_block.
    static void time_selected();

private:
    // The public constructor validates its arguments, converts the ranges to a host 'keep'
    // array, and delegates here, so that 'nmasked' and 'keep' can both be initialized from
    // one conversion.
    GpuBadChannelMask(long nbeams, long nfreq, long ntime,
                      const std::vector<std::pair<double,double>> &mask_ranges,
                      std::pair<double,double> freq_range, long warps_per_block,
                      const ksgpu::Array<uint8_t> &host_keep);
};


}}  // namespace pirate::chimefrb

#endif  // _PIRATE_CHIMEFRB_BAD_CHANNEL_MASK_HPP
