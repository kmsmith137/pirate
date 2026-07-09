#ifndef _PIRATE_PF_VARIANCE_HPP
#define _PIRATE_PF_VARIANCE_HPP

#include <vector>
#include <memory>
#include <ksgpu/Array.hpp>

#include "SparseTile.hpp"
#include "DedispersionPlan.hpp"

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// C++ port of pirate_frb/slow_avar/PfVariance.py (the subset reachable from PfAvarApproximation;
// PfAvarExact and the test-only methods are not ported). See that file and
// notes/tree_dedispersion.tex for the math.


// Converts time series to variances, after convolving with the first P peak-finding kernels.
// See variance(). Mirrors python PfVarianceConvolver.

struct PfVarianceConvolver
{
    long Pmax = 0;                // = 3*log2(max_pf_width)+1
    long Tmax_last = 0;           // = length of the longest kernel = 2*max_pf_width
    std::vector<long> Tmax;       // (Pmax,) per-profile autocorr extent, non-decreasing
    std::vector<double> A;        // (Pmax, Tmax_last) row-major autocorr table; A[p*Tmax_last + k]

    PfVarianceConvolver();        // builds the kernels analytically and fills Tmax and A

    // out: (S, P) row-major. x: (S, nt) row-major (the only shape used: a singleton tile's data,
    // (1,S,nt), squeezed). d = min(nt, Tmax[P-1]). The python variance() is more general (arbitrary
    // spectator dims); this is the specialized 2-D form (fast: bare pointers, stack temporary).
    void variance(const double *x, long S, long nt, long P, double *out) const;
};


// Represents a variance array var[d, p] (delay 0 <= d < 2^rank, profile 0 <= p < P), stored as a
// small sum of terms each depending on d through only a few bits. Mirrors python PfVariance.

struct PfVariance
{
    long rank = 0;
    long P = 0;

    // A term: an (2^popcount(dbits), P) row-major array keyed by the bitmask 'dbits'.
    struct Term {
        long dbits;
        std::vector<double> arr;
    };
    std::vector<Term> terms;      // few entries; looked up by linear scan (python uses a dict)

    PfVariance() = default;
    PfVariance(long rank, long P);

    long get_all_dbits() const;                       // bitwise-OR of all term keys

    // Expand every term to (2^popcount(dbits), P) and return their sum. 'dbits' must be a superset
    // of every term's dbits. Used in production (the PfAvarApproximation final reduction).
    ksgpu::Array<double> unpack(long dbits) const;

    // Compute the variance of a singleton SparseTile and accumulate it into this object.
    void add_tile(const SparseTile &t, const PfVarianceConvolver &conv);

    // Accumulate (scale * src) into this object. If upper_half, accumulate src's upper delay-half
    // (fix the top delay bit to 1 and drop it). Requires src.rank == rank + (upper_half?1:0) and
    // src.P >= P (extra profiles in src are discarded).
    void add(const PfVariance &src, bool upper_half = false, double scale = 1.0);

    static PfVariance from_tile(const SparseTile &t, long P, const PfVarianceConvolver &conv);

    // Accumulate scale * src[row_off + i, 0:P] into term[dbits][i, 0:P] for i in [0, nrows).
    // src is row-major with row stride src_P (>= P). nrows must equal 2^popcount(dbits).
    void accumulate(long dbits, const double *src, long row_off, long nrows, long src_P, double scale);
};


// Approximate analytic peak-finding variances for a DedispersionPlan (all DedispersionTrees).
// Mirrors python PfAvarApproximation, but computes ONLY the tree_variance[] arrays (per_tff is
// dropped; per_tf is kept as a member). See the python docstring for the approximation.

class PfAvarApproximation
{
public:
    long nfreq = 0;
    long ntrees = 0;

    // Output: one array per tree, tree_variance[itree] has shape (N, 2^(r-L), P), where:
    //    r = tree rank = config.toplevel_tree_rank - delta_rank - (ipri>0 ? 1 : 0)
    //    2^L = tree.pf.wt_dm_downsampling
    //    N = frequency_subbands.N
    // Note that the shape can also be written as (N, tree.ndm_wt, P).
    std::vector<ksgpu::Array<double>> tree_variance;

    // Per-frequency-summed accumulators, kept as a member (per python request).
    // per_tf[itree][f] is a rank-(r-L) PfVariance, f in [0, 2^R).
    std::vector<std::vector<PfVariance>> per_tf;

    PfAvarApproximation(const std::shared_ptr<DedispersionPlan> &plan, const ksgpu::Array<double> &freq_variances);

private:
    PfVarianceConvolver convolver;     // shared full kernel bank; sliced per-tree by P

    // Per-tree scalars (length ntrees).
    std::vector<long> tree_r, tree_R, tree_L, tree_P, tree_ipri, tree_N, tree_klevel;
    std::vector<std::vector<long>> tree_n_to_flo, tree_n_to_fhi;

    std::vector<double> freq_variances_vec;   // (nfreq,)
    std::vector<double> channel_map;          // plan.config.make_channel_map(), length 2^toplevel_tree_rank+1

    long max_klevel = 0;
    std::vector<long> klevel_Pmax, klevel_Lmax;   // max P (or L) among trees at a given klevel

    void process_klevel(const SparseTileTriple &sarr, long k, long ifreq);
};


}  // namespace pirate

#endif  // _PIRATE_PF_VARIANCE_HPP
