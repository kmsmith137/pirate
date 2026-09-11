#ifndef _PIRATE_CHIMEFRB_CLIPPER_BASE_HPP
#define _PIRATE_CHIMEFRB_CLIPPER_BASE_HPP

#include <string>
#include <cuda_runtime.h>
#include <ksgpu/Array.hpp>

#include "ClipperAxis.hpp"

namespace pirate {
namespace chimefrb {
#if 0
}}  // editor auto-indent
#endif


// GpuClipperBase: what the chimefrb clippers have in common -- array geometry, axis,
// (Df, Dt), the per-row wrms statistic, argument checking, the chunking rule, and the
// scratch layout. Not a transform on its own: GpuIntensityClipper and GpuStdDevClipper
// derive from it, each adding its own threshold and its own final kernel(s).
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
// the array to hold exactly ONE chunk, which is all the production RFI chain needs (every
// clipper in it has nt_chunk = 4096, and so does the chain). Processing T = N*nt_chunk
// samples in one call would be a useful generalization and is deliberately left for
// later. AXIS_TIME and AXIS_FREQ are nearly free: for TIME, (B,F,T) reshapes to
// (B,F,N,nt_chunk), which is contiguous; for FREQ the boundaries only change which rows
// a clipper pools together. AXIS_NONE needs real work, because its sub-planes are strided
// in frequency and GpuWrms cannot view them as rows. The numpy references
// (ReferenceIntensityClipper, ReferenceStdDevClipper) implement the general case, so the
// semantics are pinned down and tested.
//
// Not for polymorphic use. There are no virtual functions, and a clipper must never be
// deleted through a GpuClipperBase pointer. (The destructor is public only because
// pybind11 needs it in order to register the base class.)

struct GpuClipperBase
{
    const long B, F, nt_chunk;     // full-resolution array shape, (B, F, nt_chunk)
    const ClipperAxis axis;
    const long Df, Dt;             // downsampling factors
    const long niter;              // TOTAL passes of the statistic (1 for GpuStdDevClipper)
    const double iter_sigma;       // the statistic's refinement threshold; unused at niter=1
    const bool two_pass;           // the stabler two-pass form of the statistic's first pass

    // Derived geometry. Public because the unit tests rebuild steps 1-2 from
    // GpuWiDownsampler and GpuWrms, and compare -- which is how rf_kernels' own tests check
    // the clippers.
    const long F_ds, T_ds;         // F/Df, nt_chunk/Dt
    const long wrms_L;             // samples per statistic row
    const long wrms_R;             // statistic rows: B*F_ds (TIME), B*T_ds (FREQ), B (NONE)

    // Number of float32 scratch elements launch() needs. Covers the downsampled and
    // transposed arrays, the (mean, var) outputs, and GpuWrms' own scratch. Never zero:
    // (mean, var) always live here. A chain should allocate the max over its transforms
    // once and share one array.
    const long scratch_nelts;

protected:
    // 'name' prefixes every exception message, so that a caller sees "GpuStdDevClipper:
    // ..." rather than "GpuClipperBase: ...".
    //
    // Throws on: F % (32*Df) != 0 or nt_chunk % (32*Dt) != 0 (the 32 is GpuWiDownsampler's
    // output tile size, and the clip kernels' warp width); B < 1; Df < 1; Dt < 1;
    // niter < 1; iter_sigma < 0. The derived class checks its own parameters.
    GpuClipperBase(const char *name, long B, long F, long nt_chunk, ClipperAxis axis,
                   long Df, long Dt, long niter, double iter_sigma, bool two_pass);

    const std::string name;

    // What steps 1-2 leave behind. 'mean' and 'var' are views into the caller's scratch;
    // 'cell_i' is the UNTRANSPOSED downsampled intensity, which the clip kernels read,
    // and IS 'intensity' when (Df,Dt) == (1,1).
    struct StatisticOutputs {
        ksgpu::Array<float> cell_i;    // (B, F_ds, T_ds)
        ksgpu::Array<float> mean;      // (wrms_R,)
        ksgpu::Array<float> var;       // (wrms_R,)
    };

    // Checks the launch() arguments -- shapes, contiguity, location, aliasing, and the
    // T == nt_chunk rule with an exception that explains it -- then launches steps 1-2
    // on 'stream', without synchronizing.
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


// clipper_zero_cell(): zero the Df-by-Dt block of full-resolution weights behind one
// downsampled cell. 'weights' is the full (B, F_ds*Df, T_ds*Dt) contiguous array.
//
// The stores are 32-bit and, for Dt > 1, strided when a warp calls this from 32
// consecutive cells: its lanes are Dt*4 bytes apart, so one instruction touches up to 32
// sectors rather than one cache line. Deliberate. The clippers only zero cells that are
// being masked, and when a whole row is masked the warp's Dt stores together cover
// 32*Dt*4 contiguous bytes anyway. GpuWiDownsampler reads through exactly this pattern at
// 657 GB/s.
__device__ __forceinline__ void clipper_zero_cell(float *weights, long b, long f_ds,
                                                  long t_ds, long F_ds, long T_ds,
                                                  int Df, int Dt)
{
    // Apply per-cell pointer offset.
    //   before: shape (B, F_ds*Df, T_ds*Dt), contiguous
    //   after: shape (Df, Dt), strides (T_ds*Dt, 1)
    const long T = T_ds * Dt;
    float *wp = weights + ((b*F_ds + f_ds)*Df)*T + t_ds*Dt;

    for (int df = 0; df < Df; df++) {
        for (int dt = 0; dt < Dt; dt++)
            wp[dt] = 0.0f;
        wp += T;
    }
}

#endif  // __CUDACC__


}}  // namespace pirate::chimefrb

#endif  // _PIRATE_CHIMEFRB_CLIPPER_BASE_HPP
