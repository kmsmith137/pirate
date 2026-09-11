#ifndef _PIRATE_CHIMEFRB_SPLINE_DETRENDER_HPP
#define _PIRATE_CHIMEFRB_SPLINE_DETRENDER_HPP

#include <vector>
#include <cuda_runtime.h>
#include <ksgpu/Array.hpp>

namespace pirate {
namespace chimefrb {
#if 0
}}  // editor auto-indent
#endif


// GpuSplineDetrender: a port of rf_kernels::spline_detrender, the frequency-direction
// detrender of the old CHIME FRB search's RFI chain.
//
// Per (beam, time sample), independently: fit a piecewise-cubic spline in frequency to
// the intensity by weighted least squares, and subtract it from every channel. The
// spline has 'nbins' equal bins with a C^1 join at each interior bin edge (a cubic Hermite
// spline: value and slope at each of the nbins+1 edges, 2*(nbins+1) coefficients), and the
// fit is regularized by a penalty on the integrated squared slope,
//
//     chi^2 = sum_f w_f (d_f - b(f))^2  +  epsilon * (sum_f w_f) / nbins * sum_bins int_0^1 (db/dx)^2 dx,
//
// with x the position within a bin. Note the strength scales with the sample's total
// weight, so under heavy flagging the penalty weakens in step with the data term. The
// penalty's null space is the constant function, so a constant baseline is removed
// exactly at any epsilon. Weights are real-valued and nonnegative -- in the production
// chain they are {0,1} at full resolution and integer counts after 16x downsampling --
// and are read but never modified.
//
// The bin edges are those of the old code, at fractional channel positions
// b*nfreq/nbins rounded to the nearest channel, and the channel f datum sits at f + 1/2.
//
// What this class does NOT do, deliberately: there is no conditioning statistic, no
// mask expansion, and no per-sample gate other than "a sample with no weight anywhere
// is left untouched" -- exactly as in the old code. It is a reproduction of the
// estimator, not an improvement of it. Two departures from the old code's ARITHMETIC,
// which change nothing but roundoff and failure modes: the normal equations are
// equilibrated before the float32 Cholesky (the old code is not), and a channel with
// zero weight contributes exactly zero to the fit even if its intensity is NaN (the old
// code multiplies, and 0*NaN would poison the whole time sample).
//
// The numpy reference is pirate_frb.chimefrb.ReferenceSplineDetrender, transcribed
// from the old code's own reference implementation; misc/chimefrb/spline_detrender/ is
// the spot check against the old code itself.
//
// IMPLEMENTATION. The three cuda kernels are those of GpuDetrenderLps2d, shared through
// include/pirate/detrender_kernels.hpp and instantiated here with the Hermite basis in
// the per-channel tables, real-valued weights, and the slope penalty as the regulator
// table. The time window of that detrender is not used: this class runs the kernels at
// (n, W) = (0, 0), one independent fit per time sample. See the .cu file.
//
// THREAD SAFETY: an instance owns per-launch scratch arrays, so one instance must not be
// used concurrently from two streams. Construct one instance per stream.
//
// See notes/chimefrb.md for the porting rules this class follows.

struct GpuSplineDetrender
{
    // Throws on nbins < 1, nfreq < nbins (every bin must hold at least one channel),
    // epsilon <= 0, M < 1, or T not a positive multiple of 32.
    //
    // 'M' (beams) and 'T' (time samples per launch) are constructor arguments because the
    // scratch arrays are sized from them. T is a runtime kernel argument otherwise; the
    // multiple-of-32 rule keeps the solve kernel's grid free of a predicated tail. The
    // old code's nfreq >= 16*nbins was an AVX2 convenience and is not carried over.
    GpuSplineDetrender(long nfreq, long nbins, double epsilon, long M, long T);

    ~GpuSplineDetrender();

    const long nfreq;
    const long nbins;         // equal bins; the spline is C^1 across bin edges
    const double epsilon;     // regularization strength (see the class comment)
    const long M;             // spectator (beam) rows per launch
    const long T;             // time samples per launch; positive multiple of 32

    // Derived in the constructor.
    long N_phi;               // 2*(nbins+1): value and slope at each bin edge
    long nfrange;             // freq-ranges (an internal decomposition; see the .cu)
    long channels_per_range;  // freq-range width used, derived from (nfreq, T)
    long solve_threads;       // block size of the solve kernel

    // The bin edges, channel indices of length nbins+1, running from 0 to nfreq. Exposed
    // so that a test can check them against the reference rather than recompute them.
    std::vector<long> bin_edges() const;

    // launch(): asynchronously launch the kernels, and return without synchronizing the
    // stream. Note: stream=NULL is allowed, but is not the default.
    //
    //   intensity  shape (M, nfreq, T), float32, fully contiguous, on GPU. The fitted
    //              baseline is subtracted in place at EVERY channel, weighted or not. A
    //              time sample whose weights are all zero is left untouched.
    //
    //   weights    shape (M, nfreq, T), float32, fully contiguous, on GPU. Must be >= 0.
    //              NOT checked (that would cost a pass over the data, and every producer
    //              in the chain guarantees it by construction); negative weights would
    //              make the normal equations indefinite. Read only.
    void launch(ksgpu::Array<float> &intensity,
                const ksgpu::Array<float> &weights,
                cudaStream_t stream) const;

    // Static timing function (called via 'python -m pirate_frb time --cfrb'). Times the
    // two shapes the old search's production RFI chain uses this detrender at, and
    // reports the memory bandwidth achieved against the traffic a perfect implementation
    // would move.
    static void time_selected();

private:
    // Persistent device arrays, built once in the constructor: the per-channel basis
    // tables, the regulator table and the constant function's coefficients, and the
    // freq-range / zone descriptors. See the .cu file and detrender_kernels.hpp.
    ksgpu::Array<float> phi_tab;      // (nfreq, 4)
    ksgpu::Array<float> prod_tab;     // (nfreq, 12)
    ksgpu::Array<float> reg_tab;      // (N_phi, 4)
    ksgpu::Array<float> unit_coef;    // (N_phi,)
    ksgpu::Array<int> fr_desc;        // (nfrange, 4)
    ksgpu::Array<int> zone_desc;      // (1, 4)

    // Per-launch scratch. Sized in the constructor and reused, which is why an instance
    // is single-stream (see the class comment).
    mutable ksgpu::Array<float> gu;      // (M, nfrange, 14, T)
    mutable ksgpu::Array<float> acoef;   // (M, N_phi, T)
    mutable ksgpu::Array<float> rmin;    // (M, 1, T); computed by the shared kernel, unused here

    std::vector<long> _bin_edges;

    // The time-basis stencils (trivial at (n,W) = (0,0), but the kernel takes them by
    // value), held as an opaque blob so that this header does not need the device-code
    // header. Freed in the destructor.
    void *tb_blob = nullptr;
};


}}  // namespace pirate::chimefrb

#endif  // _PIRATE_CHIMEFRB_SPLINE_DETRENDER_HPP
