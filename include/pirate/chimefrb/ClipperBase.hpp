#ifndef _PIRATE_CHIMEFRB_CLIPPER_BASE_HPP
#define _PIRATE_CHIMEFRB_CLIPPER_BASE_HPP

#include <cuda_runtime.h>
#include <ksgpu/Array.hpp>

#include "ClipperAxis.hpp"
#include "TransformBase.hpp"
#include "WeightUpsampler.hpp"   // zero_cell(), which both clippers' final kernels call

namespace pirate {
namespace chimefrb {
#if 0
}}  // editor auto-indent
#endif


// GpuClipperBase: what the chimefrb clippers have in common -- axis, (Df, Dt), the per-row
// wrms statistic, the chunking rule, and the scratch layout -- on top of GpuTransformBase
// (TransformBase.hpp), which supplies the array geometry and the checked launch(). Not a
// transform on its own: GpuIntensityClipper and GpuStdDevClipper derive from it, each
// adding its own threshold and its own final kernel(s).
//
// The statistic is two steps, shared by both clippers:
//
//   1. GpuWiDownsampler(Df, Dt, false), skipped when (Df,Dt) == (1,1) because it is then
//      the identity; then GpuWiDownsampler(1, 1, true) on that result when axis == FREQ,
//      since the statistic wants a contiguous frequency column. (Transposing the small
//      downsampled pair, rather than downsampling a second time with transpose=true, is
//      bit-identical and reads the full-resolution pair once instead of twice.)
//   2. GpuWrms on the resulting array viewed as (R, L). The three axes differ only in the
//      reshape: one row per downsampled frequency (TIME), per downsampled time sample
//      (FREQ), or per beam (NONE).
//
// CHUNKING. The old code applies each clipper to one 'nt_chunk' block of the stream at a
// time, and the result does depend on where those boundaries fall. This class requires
// ntime == nt_chunk -- the array holds exactly ONE chunk -- which is all the production RFI
// chain needs (every clipper in it has nt_chunk = 4096, and the chain runs at ntime = 4096).
// Both are constructor arguments, and kept as separate members even though they are equal,
// because nt_chunk is the semantic parameter that a saved configuration carries. Processing
// ntime = N*nt_chunk in one launch would be a useful generalization and is deliberately
// left for later: for AXIS_TIME the rows just regroup, but for AXIS_FREQ GpuStdDevClipper
// pools its variances per (beam, chunk), and AXIS_NONE has sub-planes strided in
// frequency that GpuWrms cannot view as rows. The numpy references
// (ReferenceIntensityClipper, ReferenceStdDevClipper) implement the general case, so the
// semantics are pinned down and tested.
//
// Not constructible on its own (the constructor is protected), and adds no virtual
// functions of its own: the virtual launch_checked() that the clippers override is
// GpuTransformBase's.

struct GpuClipperBase : public GpuTransformBase
{
    // Inherited from GpuTransformBase: name, nbeams, nfreq, ntime (the full-resolution
    // array shape), scratch_nelts, and launch().
    const long nt_chunk;               // samples per chunk; == ntime (see CHUNKING above)
    const ClipperAxis axis;
    const long Df, Dt;             // downsampling factors
    const long niter;              // TOTAL passes of the statistic (1 for GpuStdDevClipper)
    const double iter_sigma;       // the statistic's refinement threshold; unused at niter=1
    const bool two_pass;           // the stabler two-pass form of the statistic's first pass

    // Derived geometry. Public because the unit tests rebuild steps 1-2 from
    // GpuWiDownsampler and GpuWrms, and compare -- which is how rf_kernels' own tests check
    // the clippers.
    const long F_ds, T_ds;         // nfreq/Df, nt_chunk/Dt
    const long wrms_L;             // samples per statistic row
    const long wrms_R;             // statistic rows: nbeams*F_ds (TIME), nbeams*T_ds (FREQ), nbeams (NONE)

    // scratch_nelts (inherited) covers the downsampled and transposed arrays, the (mean, var)
    // outputs, and GpuWrms' own scratch. Never zero: (mean, var) always live there. A chain
    // should allocate the max over its transforms once and share one array.

protected:
    // 'name' prefixes every exception message, so that a caller sees "GpuStdDevClipper:
    // ..." rather than "GpuClipperBase: ...".
    //
    // Throws on: ntime != nt_chunk (see CHUNKING above); nfreq % (32*Df) != 0 or
    // nt_chunk % (32*Dt) != 0 (the 32 is GpuWiDownsampler's output tile size, and the clip
    // kernels' warp width); nt_chunk < 1; Df < 1; Dt < 1; niter < 1; iter_sigma < 0; and
    // whatever GpuTransformBase throws on (nbeams, nfreq or ntime < 1). The derived class
    // checks its own parameters.
    GpuClipperBase(const char *name, long nbeams, long nfreq, long ntime, long nt_chunk,
                   ClipperAxis axis, long Df, long Dt, long niter, double iter_sigma,
                   bool two_pass);

    // What steps 1-2 leave behind. 'mean' and 'var' are views into the caller's scratch;
    // 'cell_i' is the UNTRANSPOSED downsampled intensity, which the clip kernels read,
    // and IS 'intensity' when (Df,Dt) == (1,1).
    struct StatisticOutputs {
        ksgpu::Array<float> cell_i;    // (nbeams, F_ds, T_ds)
        ksgpu::Array<float> mean;      // (wrms_R,)
        ksgpu::Array<float> var;       // (wrms_R,)
    };

    // Launches steps 1-2 on 'stream', without synchronizing. Called from the clippers'
    // launch_checked(), so the arguments have already been checked by
    // GpuTransformBase::launch(), and 'scratch' holds exactly scratch_nelts elements.
    StatisticOutputs _launch_statistic(const ksgpu::Array<float> &intensity,
                                       const ksgpu::Array<float> &weights,
                                       ksgpu::Array<float> &scratch,
                                       cudaStream_t stream) const;
};


#ifdef __CUDACC__

// clipper_row(): the statistic row that downsampled cell (b, f_ds, t_ds) belongs to --
// the inverse of the (R, L) view described above. Must stay in sync with the reshape in
// GpuClipperBase::_launch_statistic().
__device__ __forceinline__ long clipper_row(int axis, long b, long f_ds, long t_ds,
                                            long F_ds, long T_ds)
{
    if (axis == int(ClipperAxis::TIME))
        return b*F_ds + f_ds;
    if (axis == int(ClipperAxis::FREQ))
        return b*T_ds + t_ds;
    return b;
}

#endif  // __CUDACC__


}}  // namespace pirate::chimefrb

#endif  // _PIRATE_CHIMEFRB_CLIPPER_BASE_HPP
