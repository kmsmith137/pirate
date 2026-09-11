#ifndef _PIRATE_CHIMEFRB_WRMS_HPP
#define _PIRATE_CHIMEFRB_WRMS_HPP

#include <cuda_runtime.h>
#include <ksgpu/Array.hpp>

namespace pirate {
namespace chimefrb {
#if 0
}}  // editor auto-indent
#endif


// The variance-validity cutoffs of the old CHIME FRB search's weighted_mean_rms: a row
// whose variance falls below (eps_2 * mean)^2, or below eps_3 * mean^2, is declared to
// have no usable statistic and its variance is set to exactly zero.
//
// These are FLOAT32 constants by definition, not by accident: simd_helpers hardcodes
// machine_epsilon<float>() = 1.19e-07, and the cutoffs are 100x and 1000x it. A float64
// reimplementation must still use these values, or it is not the same algorithm.
constexpr float wrms_eps_2 = 1.0e2f * 1.19e-07f;
constexpr float wrms_eps_3 = 1.0e3f * 1.19e-07f;


#ifdef __CUDACC__
// wrms_survives(): does a sample at distance 'd' from the mean survive a clip at
// 'thresh'?
//
// THIS PREDICATE IS SHARED ON PURPOSE, and the sharing is load-bearing. GpuWrms uses it
// to decide which samples a refinement keeps; the intensity_clipper's final clip uses its
// exact complement to decide which weights to zero. They must agree, including on the
// boundary -- note the strict '<', which means thresh = 0 kills everything, and which is
// how a rejected variance propagates through the remaining refinements.
//
// The reason to share one function rather than write the comparison twice: the final clip
// is tested against an independent reference, but the refinement's copy is not (the
// induction in test_wrms.py supplies the survivor set from our own clip, so a wrong
// comparison would cancel out). Sharing is what carries the tested one's guarantee over
// to the untested one.
__device__ __forceinline__ bool wrms_survives(float d, float thresh)
{
    return fabsf(d) < thresh;
}
#endif


// GpuWrms: the weighted mean and variance of each row of an (R, L) array, refined by
// iterated sigma clipping. A port of rf_kernels::weighted_mean_rms.
//
// This is the statistic both chimefrb clippers are built on. It does NOT apply any
// threshold to the weights -- that is a separate kernel -- and its only outputs are, per
// row, a mean and a variance.
//
// The three clipper axes all arrive here as row reductions of a contiguous 2-D array, so
// this class knows nothing about frequencies, times or axes: GpuWiDownsampler has already
// produced either a plain (B, F_ds, T_ds) array or a transposed (B, T_ds, F_ds) one, and
// the caller views it as (R, L). The whole-plane case is the same thing with one long row,
// since a contiguous (B, F_ds, T_ds) array IS a (B, F_ds*T_ds) array -- which is why L
// ranges from a few hundred to a few hundred thousand, and why there are two kernels.
//
// The algorithm, in full:
//
//   First pass, two_pass=true:   mean = sum(w I) / sum(w)
//                                var  = sum(w (I-mean)^2) / sum(w)
//                                var  = 0 unless var >= (eps_2 mean)^2
//
//   First pass, two_pass=false:  mean = sum(w I) / sum(w)
//                                var  = sum(w I^2) / sum(w) - mean^2
//                                var  = 0 unless var >= eps_3 mean^2
//
//   Refinements 2..niter, over the samples with |I - mean| < iter_sigma*sqrt(var):
//                                d    = sum(w (I-mean)) / sum(w)
//                                var  = sum(w (I-mean)^2) / sum(w) - d^2
//                                var  = 0 unless BOTH var >= (eps_2 mean)^2  [OLD mean]
//                                                and  var >= eps_3 d^2      [the INCREMENT]
//                                mean = mean + d
//
// where a weight sum of zero makes both outputs zero. Four things are easy to misread:
// 'niter' counts TOTAL passes, so niter=1 means no refinement at all; the refinements
// always use the single-pass variance form regardless of two_pass; the mean accumulates
// rather than being recomputed, which is what keeps the sums stable when it is far from
// zero; and a rejected variance stays rejected, because zero variance means zero
// threshold means no survivors.

struct GpuWrms
{
    // Throws on L < 1, niter < 1, iter_sigma < 0, or an unsupported threads_per_block.
    //
    // 'L' is a constructor argument rather than a launch argument because it decides
    // which kernel runs: a row that fits in shared memory is staged there once and
    // refined on-chip, so the input is read exactly once however many refinements run;
    // a longer row is re-read from global once per refinement. 'R' is a launch argument.
    //
    // 'iter_sigma' is the threshold used BY THE REFINEMENTS, in units of the current rms.
    // It is not the intensity_clipper's final-clip sigma, which is a different number
    // applied by a different kernel (in the production chain they are 3 and 5). Ignored
    // when niter == 1.
    //
    // 'threads_per_block' is a performance knob, not a semantic one: it must not change
    // the result. Must be 128, 256, 512 or 1024.
    //
    // The default is the SMALLEST value, which is the opposite of GpuWiDownsampler's
    // default and is not an oversight. Each step here ends in a block-wide reduction
    // whose cost grows with the warp count, while the work per block is fixed by L, so a
    // big block does more synchronizing and less summing: at (L, niter) = (1024, 9),
    // 1024 threads measures 9x slower than 128. See time_selected(); 128 is best or
    // within 2% of best at every configuration the RFI chain uses.
    GpuWrms(long L, long niter, double iter_sigma, bool two_pass,
            long threads_per_block = 128);

    const long L;                   // samples per row
    const long niter;               // TOTAL passes; 1 means no refinement
    const double iter_sigma;        // refinement threshold, in rms units
    const bool two_pass;            // use the stabler two-pass first pass
    const long threads_per_block;   // 128, 256, 512 or 1024

    // True if this L uses the shared-memory kernel. Exposed because the two paths
    // perform very differently, so a test or a timing run wants to say which it measured.
    bool is_shared_memory_path() const;

    // The largest L that uses the shared-memory kernel. One source of truth for the
    // threshold: a test that wants to draw L either side of it should ask, rather than
    // recompute it from the shared-memory budget and drift.
    static long max_shared_L();

    // Number of float32 scratch elements launch() needs for R rows. Zero on the
    // shared-memory path, in which case 'scratch' may be an empty array.
    long scratch_nelts(long R) const;

    // launch(): asynchronously launch the kernel(s), and return without synchronizing the
    // stream. Note: stream=NULL is allowed, but is not the default.
    //
    // All arrays are float32, fully contiguous, and in GPU memory. The outputs must not
    // alias the inputs.
    //
    //   mean     shape (R,). Weighted mean of each row. Fully overwritten. Zero for a
    //            row whose weights sum to zero.
    //
    //   var      shape (R,). Weighted variance of each row, or 0 for a row with no usable
    //            statistic (no weight, or a variance below the cutoffs above). Fully
    //            overwritten, and never negative.
    //
    //   in_i     shape (R, L). Intensity. Read only.
    //
    //   in_w     shape (R, L). Weights, which must be >= 0. NOT checked: that would cost
    //            a pass over the data, and every producer in the chain guarantees it by
    //            construction. Negative weights would break the "a sum of weights is zero
    //            iff every weight is zero" property the guarded divides rely on.
    //            Read only.
    //
    //   scratch  shape (scratch_nelts(R),), or empty when that is zero. Per-block partial
    //            sums; contents on entry are ignored and on exit are garbage.
    //
    //   stream   CUDA stream.
    //
    // R comes from in_i.shape[0]. There are no divisibility requirements on L or R.
    void launch(ksgpu::Array<float> &mean,
                ksgpu::Array<float> &var,
                const ksgpu::Array<float> &in_i,
                const ksgpu::Array<float> &in_w,
                ksgpu::Array<float> &scratch,
                cudaStream_t stream) const;

    // Static timing function (called via 'python -m pirate_frb time --cfrb').
    // Times the configurations the old search's production RFI chain uses, on both
    // paths, at every supported threads_per_block. Reports the memory bandwidth
    // achieved against the traffic a perfect implementation would move.
    static void time_selected();
};


}}  // namespace pirate::chimefrb

#endif  // _PIRATE_CHIMEFRB_WRMS_HPP
