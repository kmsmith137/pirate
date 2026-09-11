#ifndef _PIRATE_CHIMEFRB_INTENSITY_CLIPPER_HPP
#define _PIRATE_CHIMEFRB_INTENSITY_CLIPPER_HPP

#include <cuda_runtime.h>
#include <ksgpu/Array.hpp>

#include "ClipperAxis.hpp"

namespace pirate {
namespace chimefrb {
#if 0
}}  // editor auto-indent
#endif


// GpuIntensityClipper: zeroes the weights of samples that sit more than 'sigma' standard
// deviations from a weighted mean. A port of rf_kernels::intensity_clipper, which is the
// old CHIME FRB search's principal RFI flagger (48 of its production chain's 120 nodes).
//
// Three steps, of which only the last is new code:
//
//   1. GpuWiDownsampler(Df, Dt, false), skipped when (Df,Dt) == (1,1) because it is then
//      the identity.
//   2. GpuWiDownsampler(1, 1, true) on that result when axis == FREQ, since the statistic
//      wants a contiguous frequency column. Then GpuWrms on the resulting array, viewed
//      as (R, L): the three axes differ only in how the caller reshapes.
//   3. intensity_clip: mask every downsampled cell with |I_ds - mean| >= sigma*sqrt(var),
//      and zero all Df*Dt full-resolution weights of a masked cell.
//
// Three things are worth knowing before calling it:
//
//   - 'sigma' and 'iter_sigma' are DIFFERENT NUMBERS. sigma is the final clip; iter_sigma
//     is used inside the statistic's refinements. In the production chain they are 5 and
//     3. Confusing them is the easiest mistake available in this transform, and it
//     produces a plausible-looking result.
//
//   - A row whose variance was rejected (var == 0) gets thresh == 0, and the strict '<'
//     in wrms_survives() then masks EVERY sample in the row, including samples exactly at
//     the mean. That is intentional in the original: no usable statistic means no usable
//     data.
//
//   - CHUNKING. The old code applies this transform to one 'nt_chunk' block of the stream
//     at a time, and the result does depend on where those boundaries fall. This class
//     requires the array to hold exactly ONE chunk, which is all the production RFI chain
//     needs (every intensity_clipper in it has nt_chunk = 4096, and so does the chain).
//     Processing T = N*nt_chunk samples in one call would be a useful generalization and
//     is deliberately left for later: AXIS_TIME and AXIS_FREQ are nearly free (for TIME,
//     (B,F,T) reshapes to (B,F,N,nt_chunk), which is contiguous; for FREQ the boundaries
//     are irrelevant, since each time column is its own statistic), but AXIS_NONE needs
//     real work, because its sub-planes are strided in frequency and GpuWrms cannot view
//     them as rows. ReferenceIntensityClipper already implements the general case, so
//     the semantics are pinned down and tested.
//
// Nothing here is stateful ACROSS chunks: no ring buffers, no lag buffers, no carry-over
// of any kind. See notes/chimefrb.md for the porting rules this class follows.

struct GpuIntensityClipper
{
    // (B, F, nt_chunk) is the full-resolution array shape: beams, frequency channels, and
    // time samples. The old code calls the last two (nfreq, nt_chunk) and has no beam
    // axis. All three are constructor arguments, so the array geometry is fully fixed at
    // construction: F and nt_chunk because they fix the statistic's row length, which
    // GpuWrms needs at construction, and B because it fixes the row count and the scratch
    // size.
    //
    // Throws on: F % (32*Df) != 0 or nt_chunk % (32*Dt) != 0 (the 32 is GpuWiDownsampler's
    // output tile size, and intensity_clip's warp width); B < 1; Df < 1; Dt < 1;
    // niter < 1; sigma < 0; iter_sigma < 0; an unsupported warps_per_block.
    //
    // 'sigma' is the FINAL clip threshold, in units of the row's rms.
    // 'iter_sigma' is the threshold used BY THE STATISTIC'S REFINEMENTS, in the same
    // units. Ignored when niter == 1. Unlike rf_kernels::intensity_clipper, 0 has no
    // special meaning here (there it means "use sigma"): GpuWrms already gives 0 a
    // meaning of its own, and one value meaning two things in adjacent classes is worse
    // than making the caller say what it wants.
    // 'niter' counts TOTAL passes of the statistic, so niter == 1 means no refinement.
    // 'two_pass' selects the stabler two-pass first pass of the statistic.
    //
    // 'warps_per_block' is a performance knob for the intensity_clip kernel, not a
    // semantic one: it must not change the result. Must be 4, 8, 16 or 32. It does not
    // affect the downsampler or GpuWrms, which use their own defaults.
    //
    // The default is 16, which is the MIDDLE of the menu -- GpuWiDownsampler wants 32 and
    // GpuWrms wants the smallest block on its own menu, and this kernel wants neither.
    // 16 measures best or within 0.2% of best at every configuration the RFI chain uses,
    // while 32 costs up to 7% of the whole transform. Three kernels in one port, three
    // different answers: run time_selected() on a new GPU rather than assuming any of
    // them carries over.
    GpuIntensityClipper(long B, long F, long nt_chunk, ClipperAxis axis, double sigma,
                        long Df, long Dt, long niter, double iter_sigma, bool two_pass,
                        long warps_per_block = 16);

    const long B, F, nt_chunk;     // full-resolution array shape, (B, F, nt_chunk)
    const ClipperAxis axis;
    const double sigma;            // FINAL clip threshold, in rms units
    const long Df, Dt;             // downsampling factors
    const long niter;              // TOTAL passes; 1 means no refinement
    const double iter_sigma;       // REFINEMENT threshold; ignored when niter == 1
    const bool two_pass;
    const long warps_per_block;    // 4, 8, 16 or 32

    // Derived geometry. Exposed because the unit test reconstructs the internal pipeline
    // from public pieces and compares against it -- which is how rf_kernels' own test
    // checks this transform (it runs the production weighted_mean_rms and references only
    // the final clip). rf_kernels::intensity_clipper exposes nfreq_ds and nt_ds for the
    // same reason.
    const long F_ds, T_ds;         // F/Df, nt_chunk/Dt
    const long wrms_L;             // samples per statistic row
    const long wrms_R;             // statistic rows: B*F_ds (TIME), B*T_ds (FREQ), B (NONE)

    // Number of float32 scratch elements launch() needs. Covers the downsampled and
    // transposed arrays, the (mean, var) outputs, and GpuWrms' own scratch. Never zero:
    // (mean, var) always live here. A chain should allocate the max over its transforms
    // once and share one array.
    const long scratch_nelts;

    // launch(): asynchronously launch the kernels, and return without synchronizing the
    // stream. Note: stream=NULL is allowed, but is not the default.
    //
    // All arrays are float32, fully contiguous, and in GPU memory.
    //
    //   intensity  shape (B, F, nt_chunk). Read only, never modified.
    //
    //   weights    shape (B, F, nt_chunk). MODIFIED IN PLACE: zeroed where the clip
    //              fires, and left bit-identical everywhere else. Must be >= 0 on entry,
    //              which is not checked (see GpuWrms::launch()).
    //
    //   scratch    shape (scratch_nelts,). Contents on entry are ignored and on exit are
    //              garbage.
    //
    //   stream     CUDA stream.
    //
    // Note the argument order: the read-only array comes first, matching
    // rf_kernels::intensity_clipper::clip(), rather than pirate's usual outputs-first
    // convention -- 'weights' is an in-out argument, not an output.
    void launch(const ksgpu::Array<float> &intensity,
                ksgpu::Array<float> &weights,
                ksgpu::Array<float> &scratch,
                cudaStream_t stream) const;

    // Static timing function (called via 'python -m pirate_frb time --cfrb').
    // Times the four configurations the old search's production RFI chain uses, at every
    // supported warps_per_block, and reports bandwidth against the traffic each one is
    // predicted to move.
    static void time_selected();
};


}}  // namespace pirate::chimefrb

#endif  // _PIRATE_CHIMEFRB_INTENSITY_CLIPPER_HPP
