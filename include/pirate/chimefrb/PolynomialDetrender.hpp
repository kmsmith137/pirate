#ifndef _PIRATE_CHIMEFRB_POLYNOMIAL_DETRENDER_HPP
#define _PIRATE_CHIMEFRB_POLYNOMIAL_DETRENDER_HPP

#include <cuda_runtime.h>
#include <ksgpu/Array.hpp>

namespace pirate {
namespace chimefrb {
#if 0
}}  // editor auto-indent
#endif


// GpuPolynomialDetrender: a port of rf_pipelines::polynomial_detrender along the TIME
// axis -- the only axis the old CHIME FRB search's production RFI chain ran it on -- with
// rf_kernels::polynomial_detrender as the kernel underneath.
//
// Per (beam, channel, chunk of nt_chunk samples), independently: fit a polynomial of
// degree 'polydeg' in time to the intensity by weighted least squares, and subtract it at
// every sample of the chunk, weighted or not. There is no regularization. Before the
// solve, a CONDITIONING GATE decides whether the row is fit at all: the normal matrix is
// Cholesky-factored pivot by pivot, in the Legendre basis on the old code's sample grid,
// and the row passes only if at every pivot the Schur complement exceeds 'epsilon' times
// the diagonal entry. A row that fails has ALL nt_chunk of its weights set to zero and its
// intensity left untouched. That is this transform's only effect on the weights.
//
// The gate is not a corner case. At the production setting (degree 4, epsilon 0.01,
// 1024-sample chunks) a channel whose weighted samples form one contiguous run shorter
// than about half the chunk fails it: a channel that drops out for the second half of a
// chunk is erased for the whole chunk. Isolated flagged samples, however many, never trip
// it, and an interior gap with data on both sides rarely does; the gate measures how
// collinear the basis functions are on the weighted samples, not how many there are.
//
// Weights are real-valued and nonnegative ({0,1} at full resolution, integer counts after
// 16x downsampling, in the production chain). A NaN weight fails the gate (row masked); a
// NaN intensity at a weighted sample makes the whole row NaN, as in the old code; a NaN
// intensity at a ZERO-weight sample contributes exactly zero (the old code multiplied, and
// 0*NaN would have made the row NaN with its weights left nonzero).
//
// What is reproduced is the ESTIMATOR and the gate DECISION, not the arithmetic: the solve
// is rescaled to unit diagonal first (the gate is invariant under that), sums run in a
// different order, and a row within float32 roundoff of the gate threshold may be decided
// either way. The numpy reference is pirate_frb.chimefrb.ReferencePolynomialDetrender,
// which also implements the old code's AXIS_FREQ variant; misc/chimefrb/polynomial_detrender/
// is the spot check against the old code itself.
//
// STATELESS: no tables and no scratch arrays (a whole fit lives in one warp's registers),
// so one instance may be used from any number of streams at once. This differs from
// GpuSplineDetrender and the clippers, which own per-launch scratch and want one instance
// per stream.
//
// IMPLEMENTATION: one warp per row; see the .cu file.
//
// See notes/chimefrb.md for the porting rules this class follows.

struct GpuPolynomialDetrender
{
    // Throws on polydeg outside 0..8, epsilon <= 0, nt_chunk not a positive multiple of
    // 64, or warps_per_block not one of 4, 8, 16.
    //
    // polydeg <= 8 follows the old code's own tests (float32 fails above that). The
    // multiple-of-64 rule is two samples per lane of a warp, which lets the kernel use
    // 64-bit loads (13% faster than 32-bit at the full-band shape); the old code's rule was
    // 8. 'warps_per_block' is a performance knob that must not change the result (the unit
    // test asserts bitwise agreement across settings); 32 is not offered because the fit's
    // registers would not fit a 1024-thread block. Measured on an L40S at the production
    // configuration (8 beams, 16384 channels, one 1024-sample chunk): 663, 665 and 672 GB/s
    // for 4, 8 and 16, so the default is 16.
    GpuPolynomialDetrender(long polydeg, double epsilon, long nt_chunk, long warps_per_block = 16);

    const long polydeg;          // degree; the fit has polydeg+1 coefficients
    const double epsilon;        // gate threshold (the production chain uses 0.01)
    const long nt_chunk;         // samples per independent fit; positive multiple of 64
    const long warps_per_block;  // 4, 8 or 16

    // launch(): asynchronously launch the kernel, and return without synchronizing the
    // stream. Note: stream=NULL is allowed, but is not the default.
    //
    //   intensity  shape (M, nfreq, T), float32, fully contiguous, on GPU, with T a positive
    //              multiple of nt_chunk. One fit per (beam, channel, chunk). The fitted
    //              polynomial is subtracted in place at every sample of a row that passes
    //              the gate; a row that fails is left untouched.
    //
    //   weights    same shape, dtype and layout, not aliased with 'intensity'. MODIFIED IN
    //              PLACE: all nt_chunk weights of a failing row are set to zero, and every
    //              other weight is left bit-identical. Must be >= 0 on entry, which is NOT
    //              checked (every producer in the chain guarantees it by construction; a
    //              negative weight would make the normal equations indefinite).
    void launch(ksgpu::Array<float> &intensity, ksgpu::Array<float> &weights,
                cudaStream_t stream) const;

    // Static timing function (called via 'python -m pirate_frb time --cfrb'). Times the
    // production configuration at the two channel counts the old chain runs it at, for
    // every warps_per_block, against the traffic an ideal implementation would move.
    static void time_selected();
};


}}  // namespace pirate::chimefrb

#endif  // _PIRATE_CHIMEFRB_POLYNOMIAL_DETRENDER_HPP
