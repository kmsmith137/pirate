#ifndef _PIRATE_CHIMEFRB_STD_DEV_CLIPPER_HPP
#define _PIRATE_CHIMEFRB_STD_DEV_CLIPPER_HPP

#include <cuda_runtime.h>
#include <ksgpu/Array.hpp>

#include "ClipperBase.hpp"

namespace pirate {
namespace chimefrb {
#if 0
}}  // editor auto-indent
#endif


// GpuStdDevClipper: zeroes whole channels (AXIS_TIME) or whole time samples (AXIS_FREQ)
// whose variance is an outlier among its peers. A port of rf_kernels::std_dev_clipper, the
// most numerous transform in the old CHIME FRB search's RFI chain (60 of its 120 nodes).
// Where GpuIntensityClipper catches samples that are too bright, this catches rows whose
// NOISE LEVEL is wrong.
//
// Four steps, the first two inherited from GpuClipperBase:
//
//   1. Downsample by (Df, Dt), and transpose if axis == FREQ.
//   2. One variance per row with GpuWrms at niter = 1 -- variances, not standard
//      deviations, despite the name. A variance of zero means "no usable statistic".
//   3. Per beam, over its rows' variances: the mean vbar and standard deviation s of the
//      nonzero ones (s divides by n, not n-1), computed before anything is clipped; then
//      zero every variance with |v - vbar| >= sigma*s.
//   4. Zero all full-resolution weights of every row whose variance is now zero.
//
// Three things are worth knowing before calling it:
//
//   - If at most ONE row of a beam has a usable variance, the whole beam's chunk is
//     zeroed -- including, when exactly one channel has data, that channel. This is the
//     old code's behaviour, reproduced deliberately.
//
//   - If every usable variance in a beam is exactly equal, the outcome is decided by
//     float32 roundoff: exact arithmetic clips every row, while a rounded mean usually
//     clips none, and this class and the old code round differently. Real data cannot
//     produce it (it needs bit-identical variances); it is documented rather than fixed.
//
//   - CHUNKING: the array must hold exactly ONE nt_chunk, as for every GpuClipperBase;
//     see the chunking note in ClipperBase.hpp. ReferenceStdDevClipper implements
//     T = N*nt_chunk.
//
// AXIS_NONE is not supported: the old code does not implement it, and the chain does not
// use it. Nothing here is stateful across chunks.

struct GpuStdDevClipper : public GpuClipperBase
{
    // (B, F, nt_chunk) is the full-resolution array shape, fixed at construction.
    //
    // Throws on axis == NONE, sigma < 0, or an unsupported warps_per_block, and on
    // everything GpuClipperBase checks: F % (32*Df) != 0 or nt_chunk % (32*Dt) != 0;
    // B < 1; Df < 1; Dt < 1.
    //
    // 'sigma' is step 3's threshold, in units of the standard deviation OF THE VARIANCES.
    // Unlike rf_kernels, which requires sigma >= 1, any sigma >= 0 is accepted; note that
    // step 3 cannot clip anything if sigma >= sqrt(n-1), for n usable rows (Samuelson's
    // inequality bounds every |v - vbar| by s*sqrt(n-1)).
    // 'two_pass' selects the stabler two-pass form of the step-2 variance.
    //
    // 'warps_per_block' is a performance knob for step 4's kernel only: it must not change
    // the result. Must be 4, 8, 16 or 32. Measured on an L40S at both production
    // configurations, 4, 8 and 16 are within 0.1% of each other and 32 is about 0.5%
    // slower, so the default is 16. Do not assume a setting carries over from another
    // chimefrb kernel: the ones built so far have each wanted a different one.
    GpuStdDevClipper(long B, long F, long nt_chunk, ClipperAxis axis, double sigma,
                     long Df, long Dt, bool two_pass, long warps_per_block = 16);

    const double sigma;            // step-3 threshold, in units of sd(variances)
    const long warps_per_block;    // 4, 8, 16 or 32; step 4 only

    // Inherited from GpuClipperBase: B, F, nt_chunk, axis, Df, Dt, two_pass; the derived
    // geometry F_ds, T_ds, wrms_L, wrms_R (the rows per beam are wrms_R / B); scratch_nelts;
    // and niter and iter_sigma, which are always 1 and 0 here.

    // launch(): asynchronously launch the kernels, and return without synchronizing the
    // stream. Note: stream=NULL is allowed, but is not the default.
    //
    // All arrays are float32, fully contiguous, and in GPU memory.
    //
    //   intensity  shape (B, F, nt_chunk). Read only, never modified.
    //
    //   weights    shape (B, F, nt_chunk). MODIFIED IN PLACE: whole rows are zeroed where
    //              the clip fires, and every other weight is left bit-identical. Must be
    //              >= 0 on entry, which is not checked (see GpuWrms::launch()).
    //
    //   scratch    shape (scratch_nelts,). Contents on entry are ignored and on exit are
    //              garbage.
    //
    //   stream     CUDA stream.
    void launch(const ksgpu::Array<float> &intensity,
                ksgpu::Array<float> &weights,
                ksgpu::Array<float> &scratch,
                cudaStream_t stream) const;

    // Static timing function (called via 'python -m pirate_frb time --cfrb'). Times the two
    // configurations the old search's production RFI chain uses, at every supported
    // warps_per_block, against the traffic each is predicted to move; and reports what
    // steps 3-4 cost on top of steps 1-2.
    static void time_selected();
};


}}  // namespace pirate::chimefrb

#endif  // _PIRATE_CHIMEFRB_STD_DEV_CLIPPER_HPP
