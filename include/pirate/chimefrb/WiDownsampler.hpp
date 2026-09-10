#ifndef _PIRATE_CHIMEFRB_WI_DOWNSAMPLER_HPP
#define _PIRATE_CHIMEFRB_WI_DOWNSAMPLER_HPP

#include <cuda_runtime.h>
#include <ksgpu/Array.hpp>

namespace pirate {
namespace chimefrb {
#if 0
}}  // editor auto-indent
#endif


// GpuWiDownsampler: reduces an (intensity, weights) pair by a factor Df in frequency
// and Dt in time, using the normalization of the old CHIME FRB search:
//
//    out_w = sum of the cell's weights          <--- SUM, not mean
//    out_i = (sum of w*i) / out_w,  or 0 where out_w <= 0
//
// The "sum, not mean" is worth flagging, because the old code has both conventions in
// it: rf_kernels::wi_downsampler (which this class ports) sums, and the python helper
// rf_pipelines.utils.wi_downsample() takes the mean, so the two differ by a factor
// (Df*Dt). The clippers downstream consume the summed convention.
//
// One thing the old kernel does not have: 'transpose' writes the output with frequency
// as the fastest-varying axis. The clippers want the downsampled data in both layouts
// -- a frequency-axis statistic wants a contiguous frequency column, and applying the
// resulting mask to full-resolution weights wants contiguous time -- so a caller may
// run this kernel twice on the same source arrays, once each way.
//
//
// One deliberate divergence from rf_kernels, in a don't-care slot. Where a cell's
// weights sum to zero the downsampled intensity is undefined (every consumer multiplies
// it by that zero weight), and the two codes fill it differently: rf_kernels' general
// path writes 0, but it short-circuits (Df,Dt)=(1,1) to a memcpy, which passes the
// intensity through untouched. This class writes 0 in every case -- one rule instead of
// two, and a consumer that forgets to check the weight sees zeros rather than stale
// intensity. misc/chimefrb/rfi_wi_downsample/ is the spot test that measures this.
//
// (Df, Dt) and the array shapes are runtime, so there is no table of supported
// configurations: the constructor just checks the divisibility rules in launch().
//
// See notes/chimefrb.md for the porting rules this class follows.

struct GpuWiDownsampler
{
    // Throws on Df < 1, Dt < 1, or an unsupported warps_per_block. Also throws on
    // (Df, Dt, transpose) = (1, 1, false), which is the identity: a caller should use
    // its input array directly rather than pay for a full copy.
    //
    // 'warps_per_block' is a performance knob, not a semantic one. The kernel always
    // works on a 32-by-32 tile of output cells; this says how many warps share a tile,
    // so each thread handles (32 / warps_per_block) cells. Must be 4, 8, 16 or 32, and
    // must not change the result.
    //
    // The default is the largest value because that is what measures fastest on an
    // L40S, at every configuration the old RFI chain uses -- by 1% where the block
    // count is already large, and by 5% on the pure transpose. Run time_selected() on
    // a new GPU before assuming it still holds.
    GpuWiDownsampler(long Df, long Dt, bool transpose, long warps_per_block = 32);

    const long Df;                // frequency downsampling factor
    const long Dt;                // time downsampling factor
    const bool transpose;         // if true, output axes are (beam, time, freq)
    const long warps_per_block;   // 4, 8, 16 or 32

    // launch(): asynchronously launch the kernel, and return without synchronizing the
    // stream. Note: stream=NULL is allowed, but is not the default.
    //
    // Shapes, with B = beams, F = input frequency channels, T = input time samples, and
    // (F_ds, T_ds) = (F/Df, T/Dt). All four arrays are float32, fully contiguous, and in
    // GPU memory. The output arrays must not alias the input arrays -- in particular,
    // (Df, Dt) = (1, 1) with transpose=true is a pure transpose, and does NOT work in
    // place.
    //
    //   out_i    shape (B, F_ds, T_ds) if !transpose, else (B, T_ds, F_ds).
    //            Downsampled intensity (see the class comment). Fully overwritten.
    //
    //   out_w    same shape as out_i. Downsampled weights. Fully overwritten.
    //
    //   in_i     shape (B, F, T). Intensity. Read only.
    //
    //   in_w     shape (B, F, T). Weights, which must be >= 0. This is not checked in
    //            the inner loop: negative weights would break the "a sum of weights is
    //            zero iff every weight is zero" property that out_i's guarded divide
    //            relies on.  Read only.
    //
    //   stream   CUDA stream.
    //
    // B, F and T are taken from the array shapes, and must satisfy
    // F % (32*Df) == 0 and T % (32*Dt) == 0 (the 32 is the output tile size).
    void launch(ksgpu::Array<float> &out_i,
                ksgpu::Array<float> &out_w,
                const ksgpu::Array<float> &in_i,
                const ksgpu::Array<float> &in_w,
                cudaStream_t stream) const;

    // Static timing function (called via 'python -m pirate_frb time --cfrb').
    // Times the (Df, Dt, transpose) configurations used by the old search's production
    // RFI chain, at every supported warps_per_block, and reports the memory bandwidth
    // achieved. Every byte of every array is touched exactly once, so the reported
    // bandwidth is directly comparable to the GPU's peak.
    static void time_selected();
};


}}  // namespace pirate::chimefrb

#endif  // _PIRATE_CHIMEFRB_WI_DOWNSAMPLER_HPP
