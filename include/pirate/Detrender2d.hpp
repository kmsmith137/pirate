#ifndef _PIRATE_DETRENDER_2D_HPP
#define _PIRATE_DETRENDER_2D_HPP

#include <tuple>
#include <vector>
#include <cuda_runtime.h>
#include <ksgpu/Array.hpp>

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// Detrender2d: the 2-d spline detrender, a regularized least-squares fit of a
// B-spline in frequency times a local polynomial in time. The algorithm is specified
// in notes/tree_dedispersion.tex, section "2-d detrending"; pirate_frb/detrending_spline
// is the pure-numpy reference that this kernel is validated against.
//
// Operates in place on a (data, mask) pair of shape (M, nfreq, nbuf). For each output
// sample t, the baseline over a window of 2W+1 time samples is modelled as
//
//     b(f,s) = sum_{jq} alpha_jq phi_j(f) p_q(s)
//
// with {phi_j} the B-spline basis of the caller's knot vector and {p_q} an orthonormal
// polynomial basis on the window, fitted by weighted least squares over the unmasked
// samples with a first-difference regulator eta*D_1 on the frequency coefficients, and
// evaluated back at the window centre. The fit is subtracted from the data.
//
// Zones -- interior knots of multiplicity n_phi+1 -- decouple the fit exactly, and mask
// expansion is per zone: a zone whose conditioning statistic r_min falls below eps has
// all of its channels dropped for that time sample. There is no per-channel expansion.
//
// The kernel writes only the middle T samples of each row, i.e. buffer samples
// [W, W+T). The 2W padding samples are read but not written, and the caller is
// responsible for the buffer shift between chunks. Since the caller owns the padding,
// detrend is a pure function of its arguments: no carried state, and chunks may be
// processed in any order.
//
// Where the expanded mask is false, the residual is written as zero rather than left
// untouched, matching SplineDetrender.detrend_chunk() in the numpy reference.
//
// n_phi is the ONLY compile-time parameter of the cuda kernel, so only the values listed
// in the constructor's error message exist. Everything else is runtime, including the
// time-polynomial degree n, the window half-width W and the chunk length T. T must be a
// positive multiple of 32, W at most 16, and n at most 3.
//
// FOOTGUN, inherited from the reference: no constant-offset subtraction is performed.
// The constant function is exactly in the span, so subtracting a per-zone offset would
// be mathematically inert, but it is what would protect float32 precision against a
// large DC level. Until that exists, feeding float32 data with a large offset relative
// to its structure loses mantissa bits for nothing. In the intended pipeline the 1-d
// time detrender runs first and leaves the data roughly zero-mean.
//
// THREAD SAFETY: an instance owns per-launch scratch arrays, so one instance must not
// be used concurrently from two streams. Construct one instance per stream.

struct Detrender2d
{
    // Throws unless n_phi is one of the compiled configurations, if T is not a positive
    // multiple of 32, if n is outside [0,3], if W is outside [0,16] or gives 2W+1 < n+1,
    // or if the knot vector is invalid. 'knots' is a non-decreasing list of channel indices with
    // multiplicity expressed by repetition; it must run from 0 to nfreq, with the first
    // and last values repeated exactly n_phi+1 times (clamped ends are what put the
    // constant function in the span) and no interior value repeated more than n_phi+1
    // times.
    Detrender2d(long nfreq, const std::vector<long> &knots, long M,
                long n_phi = 2, long n = 2, long W = 4, long T = 2048,
                double eta = 1.0e-3, double eps = 3.0e-5);

    ~Detrender2d();

    const long nfreq;
    const long M;        // number of spectator (beam) rows
    const long n_phi;    // spline degree in frequency
    const long n;        // degree of the time polynomial
    const long W;        // window half-width (the window is 2W+1 samples)
    const long T;        // output samples per row (chunk size)
    const long nbuf;     // buffer samples per row, = T + 2W
    const double eta;    // regularization strength (dimensionless)
    const double eps;    // mask-expansion threshold on r_min

    long N_phi;          // number of B-spline basis functions
    long nzone;          // number of zones
    long nfrange;        // number of freq-ranges (an internal decomposition; see the .cu)

    // launch(): asynchronously launch the kernels, and return without synchronizing
    // the stream. Note: stream=NULL is allowed, but is not the default.
    //
    //   data: shape (M, nfreq, nbuf), float32, fully contiguous, on GPU. Modified in
    //         place over buffer samples [W, W+T).
    //   mask: shape (M, nfreq, nbuf), uint8, fully contiguous, on GPU, {0,1}-valued on
    //         both input and output. Modified in place over the same range.
    //
    // The caller must treat the output mask as the authoritative one: it is the input
    // mask with the ill-conditioned zones removed, so it can only lose samples.
    void launch(ksgpu::Array<float> &data,
                ksgpu::Array<unsigned char> &mask,
                cudaStream_t stream) const;

    // The compiled n_phi values. n, W and T are not among them: all three are runtime.
    static std::vector<long> configs();

    // Static timing function (called via 'python -m pirate_frb time --dt2d').
    static void time_selected();

private:
    // Persistent device arrays, built once in the constructor: the per-channel basis
    // tables, and the freq-range / zone descriptors. See the .cu file.
    ksgpu::Array<float> phi_tab;      // (nfreq, phi_stride)
    ksgpu::Array<float> prod_tab;     // (nfreq, prod_stride)
    ksgpu::Array<int> fr_desc;        // (nfrange, 4)
    ksgpu::Array<int> zone_desc;      // (nzone, 4)

    // Per-launch scratch. Sized in the constructor and reused, which is why an instance
    // is single-stream (see the class comment).
    mutable ksgpu::Array<float> gu;      // (M, nfrange, ncomp, nbuf)
    mutable ksgpu::Array<float> acoef;   // (M, N_phi, T)
    mutable ksgpu::Array<float> rmin;    // (M, nzone, T)

    long phi_stride = 0;
    long prod_stride = 0;
    long nphi_zone_max = 0;
    long solve_threads = 0;

    // The time-basis stencils, held as an opaque blob so that this header does not need
    // the templated struct. Freed in the destructor.
    void *tb_blob = nullptr;
};


}  // namespace pirate

#endif // _PIRATE_DETRENDER_2D_HPP
