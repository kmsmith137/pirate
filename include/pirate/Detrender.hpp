#ifndef _PIRATE_DETRENDER_HPP
#define _PIRATE_DETRENDER_HPP

#include <tuple>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <ksgpu/Array.hpp>

namespace YAML { class Emitter; }      // #include <yaml-cpp/yaml.h>
namespace pirate { struct YamlFile; }  // #include <pirate/YamlFile.hpp>

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// The GPU detrenders. Each has a pure-numpy reference implementation, in the python
// package pirate_frb/detrending, that it is validated against:
//
//   GpuDetrenderLps1d   <->   pirate_frb.detrending.ReferenceDetrenderLps1d
//   GpuDetrenderLps2d   <->   pirate_frb.detrending.ReferenceDetrenderLps2d
//
// "Lps" is local polynomial subtraction, the algorithm specified in notes/detrending.tex,
// sections "Time detrending algorithm 1: local polynomial subtraction" (the 1-d detrender)
// and "2-d detrending". A third detrender, a fixed-lag Kalman filter ("Kf", algorithm 2 in
// the same notes), exists in python only; if a GPU version is ever written, this is where
// it goes.


// GpuDetrenderLps1d: the 1-d time detrender, a masked, adaptively centered moving local
// polynomial fit. The algorithm is specified in notes/detrending.tex, section
// "Time detrending algorithm 1: local polynomial subtraction"; pirate_frb/detrending/lps1d
// is the pure-numpy reference that this kernel is validated against.
//
// Operates in place on a (data, mask) pair, independently for each row (one row per
// (beam, freq) pair). For each output sample t, a degree-n polynomial is fit to the
// valid samples of the window [t-W, t+W] and evaluated back at t; the fit is
// subtracted from the data, and the sample is dropped ("mask expansion") if its
// window is too ill-conditioned to determine the fit.
//
// The kernel writes only the middle T samples of each row, i.e. buffer samples
// [W, W+T). The 2W padding samples are read but not written, and the caller is
// responsible for the buffer shift between chunks (each output sample is committed
// once and forever, so the detrender never revisits it).
//
// Where the expanded mask is false, the residual is written as zero rather than
// left untouched. This matches ReferenceDetrenderLps1d.detrend_chunk() in the numpy reference,
// and means that a consumer which forgets to check the mask sees zeros rather than
// raw un-detrended intensity.
//
// (n, W, T) are compile-time parameters of the cuda kernel, so only the combinations
// listed in the constructor's error message exist. The number of rows M is runtime.

struct GpuDetrenderLps1d
{
    // Throws unless (n, W, T) is one of the compiled configurations.
    GpuDetrenderLps1d(long n, long W, long T = 2048);

    // Mask-expansion threshold on the conditioning statistic rmin, and the NaN guard
    // used where rmin is below it. Neither is a regularizer: see the "Cholesky, and
    // the conditioning statistic" discussion in notes/detrending.tex.
    //
    // Note eps is inert at n=1, where rmin is {0,1}-valued and mask expansion reduces
    // to "the window holds at least 2 valid samples".
    static constexpr float eps = 1.0e-3f;
    static constexpr float mu = 1.0e-30f;

    const long n;      // polynomial degree
    const long W;      // window half-width (window is 2W+1 samples)
    const long T;      // output samples per row (chunk size)
    const long nbuf;   // buffer samples per row, = T + 2W

    // launch(): asynchronously launch kernel, and return without synchronizing stream.
    // Note: stream=NULL is allowed, but is not the default.
    //
    //   data: shape (M, nbuf), float32, fully contiguous, on GPU. Modified in place.
    //   mask: shape (M, nbuf), uint8, fully contiguous, on GPU. Modified in place,
    //         and {0,1}-valued on both input and output.
    //
    // The caller must treat the output mask as the authoritative one: it is the input
    // mask with the ill-conditioned windows removed, so it can only lose samples.
    void launch(ksgpu::Array<float> &data,
                ksgpu::Array<unsigned char> &mask,
                cudaStream_t stream) const;

    // The compiled (n, W, T) configurations, i.e. the arguments the constructor accepts.
    static std::vector<std::tuple<long,long,long>> configs();

    // Static timing function (called via 'python -m pirate_frb time --dtl1').
    // Times every compiled configuration.
    static void time_selected();
};


// DetrenderLps2dParams: the parameters of a 2-d spline detrender.
//
// A standalone struct rather than a member of GpuDetrenderLps2d, with no "Gpu" in its name,
// because it describes the ALGORITHM and not the implementation: instances are written by
// hand as yaml files, stored inside variance-map files, and passed around by pirate_frb.varmap
// as "the detrender". The class comment on GpuDetrenderLps2d (below) says what the fields mean.

struct DetrenderLps2dParams
{
    long nfreq = 0;

    // A non-decreasing list of channel indices, with multiplicity expressed by
    // repetition. It must run from 0 to nfreq, with the first and last values repeated
    // exactly n_phi+1 times (clamped ends are what put the constant function in the
    // span) and no interior value repeated more than n_phi+1 times.
    std::vector<long> knots;

    long M = 0;             // number of spectator (beam) rows
    long n_phi = 2;         // spline degree in frequency
    long n = 2;             // degree of the time polynomial
    long W = 4;             // window half-width (the window is 2W+1 samples)
    long T = 2048;          // output samples per row (chunk size)
    double eta = 1.0e-3;    // regularization strength (dimensionless)
    double eps = 3.0e-5;    // mask-expansion threshold on r_min

    // Throws unless n_phi is one of the compiled configurations (see
    // GpuDetrenderLps2d::configs()), if T
    // is not a positive multiple of 32, if n is outside [0,2], if W is outside [0,16]
    // or gives 2W+1 < n+1, if nfreq/M/eta/eps are non-positive, or if the knot vector
    // violates any of the rules above.
    void validate() const;

    // Yaml I/O. The yaml keys are spelled out rather than matching the member names
    // (M -> num_beams, n_phi -> spline_degree_freq, and so on); the mapping lives in
    // to_yaml()/from_yaml() and nowhere else. 'eta' and 'eps' are optional on read and
    // default to the values above; every other key is required. from_yaml() calls
    // validate(), so an invalid file fails at read rather than at construction.
    void to_yaml(YAML::Emitter &emitter, bool verbose = false) const;
    std::string to_yaml_string(bool verbose = false) const;

    static DetrenderLps2dParams from_yaml(const std::string &filename);
    static DetrenderLps2dParams from_yaml(const YamlFile &f);

    // Inverse of to_yaml_string(), for a parameter set that travels as a string rather
    // than a file (e.g. embedded in a variance-map file; see pirate_frb.varmap.asdf_io).
    static DetrenderLps2dParams from_yaml_string(const std::string &yaml_string);
};


// GpuDetrenderLps2d: the 2-d spline detrender, a regularized least-squares fit of a
// B-spline in frequency times a local polynomial in time. The algorithm is specified
// in notes/detrending.tex, section "2-d detrending"; pirate_frb/detrending/lps2d
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
// untouched, matching ReferenceDetrenderLps2d.detrend_chunk() in the numpy reference.
//
// n_phi is the ONLY compile-time parameter of the cuda kernel, so only the values listed
// in the constructor's error message exist. Everything else is runtime, including the
// time-polynomial degree n, the window half-width W and the chunk length T. T must be a
// positive multiple of 32, W at most 16, and n at most 2.
//
// Two things a caller should know about n and about reproducibility:
//
//   - The TIME polynomial degree is capped at n <= 2, matching the numpy reference. The
//     kernel's arrays are sized for n = 3, but running there would be a configuration no
//     test validates, so the constructor refuses it; extend the reference first. The
//     spline degree n_phi = 3 is a different matter -- fully supported and tested.
//   - Results are bit-reproducible run to run, and across chunkings AT A FIXED T, but NOT
//     across different T. The frequency summation is grouped into "freq-ranges" whose
//     width is derived from (nfreq, knots, T), so two instances with different T sum
//     frequency in different groupings and agree only to roundoff (~1e-6 relative). M is
//     deliberately not an input to that width, so the beam axis is always a spectator: one
//     row's output never depends on how many rows were processed alongside it.
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

struct GpuDetrenderLps2d
{
    explicit GpuDetrenderLps2d(const DetrenderLps2dParams &params);

    ~GpuDetrenderLps2d();

    const DetrenderLps2dParams params;

    // Derived in the constructor.
    const long nbuf;     // buffer samples per row, = T + 2W
    long N_phi;          // number of B-spline basis functions
    long nzone;          // number of zones
    long nfrange;        // number of freq-ranges (an internal decomposition; see the .cu)
    long channels_per_range;   // freq-range width used, derived from (nfreq, knots, T)

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

    // Static timing function (called via 'python -m pirate_frb time --dtl2').
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

#endif // _PIRATE_DETRENDER_HPP
