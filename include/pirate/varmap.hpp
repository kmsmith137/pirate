#ifndef _PIRATE_VARMAP_HPP
#define _PIRATE_VARMAP_HPP

#include <vector>
#include <memory>
#include <ksgpu/Array.hpp>

#include "constants.hpp"
#include "inlines.hpp"
#include "DedispersionPlan.hpp"

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// C++ port of (a subset of) pirate_frb/slow_avar/SparseTile.py and
// pirate_frb/slow_avar/PfVariance.py. See those files and notes/variance_map.tex for the math.
// Only the subset reachable from PfAvarApproximation is ported (plus the test-only unpack()
// methods); PfAvarExact and the test-only python methods are not.
//
// Contents, in dependency order:
//
//    SparseTile, SparseTileTriple        (port of SparseTile.py)
//    PfVarianceConvolver, PfVariance     (port of PfVariance.py)
//    PfAvarApproximation                 (port of PfVariance.py)
//
// NOTE: the PfAvar* classes, together with the python TmpVmap* classes built from them, were a
// first pass at representing a variance map, written before we settled on the VarianceMap
// representation (notes/variance_map.tex, section "The variance map"). They are left unchanged
// for now, and are deliberately outside it. We may revisit them later. In particular, this file
// is NOT the C++ side of the python package pirate_frb/varmap/, which implements the current
// VarianceMap representation and has no C++ counterpart.


// -------------------------------------------------------------------------------------------------
//
// SparseTile, SparseTileTriple


// A SparseTile represents a subset of a (2^(r-k), 2^k, ntime) tree-dedispersion array
// (axes: coarse-freq, delay, time) in a compressed two-stage form. See the python docstring
// and unpack() for the unpacking semantics.
//
// Memory model (this is the key difference from the python numpy version): the data buffer is
// owned by a shared_ptr<double[]> 'base', and 'data' may point into the middle of it. This makes
// slice(), iterate_lower() and iterate_upper() zero-copy "views" that share 'base' (matching numpy
// view semantics), while iterate_aligned(), the both-halves iterate_singletons(), and gridding
// allocate a fresh buffer. ksgpu::Array is deliberately NOT used for the hot-path buffers (it is
// not optimized for frequent small allocations) -- only unpack() returns one, for testing.

struct SparseTile
{
    long r = 0;            // rank
    long k = 0;            // iteration index, 0 <= k <= r
    long f0 = 0;           // tile covers f-indices [f0, f0+nf)
    long nf = 0;
    long nt = 0;           // time indices outside [0, nt) are zero
    long dbits = 0;        // bitmask of selected delay bits, 0 <= dbits < 2^k
    long t0 = 0;           // constant time shift on unpack
    double scale = 1.0;    // scalar multiplier applied on unpack

    long tshifts[constants::max_tree_rank];   // length-k delay-bit time shifts (first k entries valid)

    // data: logical shape (nf, S, nt) row-major, S = 2^popcount(dbits). 'base' owns the allocation;
    // 'data' may point into it (zero-copy slice). data[i_f*S*nt + i_s*nt + i_t].
    std::shared_ptr<double[]> base;
    double *data = nullptr;

    SparseTile() = default;

    // Copying constructor (used by the pybind/test path). Allocates a fresh zero-filled buffer and
    // memcpy's 'src_data' (length nf*S*nt) into it, unless src_data==nullptr (caller fills .data).
    // 'src_tshifts' has length k (may be nullptr iff k==0).
    SparseTile(long r, long k, long f0, long nf, long nt, long dbits,
               const double *src_data, const long *src_tshifts, long t0 = 0, double scale = 1.0);

    long S() const { return 1L << popcount(dbits); }   // delay-axis length of 'data'

    void check_invariants() const;

    // Zero-copy sub-tile for f-index range [c0, c1) (must lie within [f0, f0+nf)).
    SparseTile slice(long c0, long c1) const;

    // Tile-level DD(k) operations (k -> k+1). See python SparseTile.
    static SparseTile iterate_aligned(const SparseTile &t);                          // allocates
    static SparseTile iterate_singletons(const SparseTile *lower, const SparseTile *upper);
    static SparseTile iterate_lower(const SparseTile &lower);                          // zero-copy
    static SparseTile iterate_upper(const SparseTile &upper);                          // zero-copy

    // Bit-index helpers (scalar versions of the python vectorized staticmethods).
    static long remap_d(long d, long dbits_in, long dbits_out);
    static long eval_tshifts(long d, long dbits, const long *tshifts);
    static void dd_tshifts(long k, long *out);    // out has length k+1

    // 'dbits' after iterating the f-range [f0, f0+nf) for 'kmax' steps from level 0. Note that
    // 'kmax' is a STEP COUNT, not a level index (elsewhere in this class, 'k' is the current
    // level). If the range spans several level-kmax tiles, whose dbits differ, the return value
    // is the union of their dbits; see the .cpp for the semantics and the derivation.
    static long predict_dbits(long kmax, long f0, long nf);

    // Test-only: densify to a (nf, 2^k, ntime) array. Not used in production.
    ksgpu::Array<double> unpack(long ntime) const;

    // Allocate an owning tile with a fresh zero-filled buffer of size nf*S*nt; caller fills .data.
    static SparseTile alloc(long r, long k, long f0, long nf, long nt, long dbits,
                            const long *tshifts, long t0, double scale);
};


// C++ port of (a subset of) pirate_frb/slow_avar/SparseTile.py's SparseTileTriple. Represents a
// (2^(r-k), 2^k, ntime) array over a contiguous f-range [f0, f0+nf) as 1..3 SparseTiles (the first
// and last f-index can carry a smaller sparsity pattern than the bulk).

struct SparseTileTriple
{
    long r = 0, k = 0, f0 = 0, nf = 0;
    SparseTile tiles[3];
    int ntiles = 0;

    SparseTileTriple() = default;

    // Canonical (c0, c1) tile boundaries for [f0, f0+nf); fills out_c0/out_c1 and n (1..3).
    static void tile_bounds(long f0, long nf, long out_c0[3], long out_c1[3], int &n);

    // Build a canonical triple by splitting a single tile into 1..3 sub-tiles (zero-copy slices).
    static SparseTileTriple from_tile(const SparseTile &t);

    // Gridding output for a one-hot (ifreq, t=0) input; see python make_tree_gridding_output.
    // 'cm' is the channel map (length cm_len = 2^rank + 1, strictly decreasing).
    static SparseTileTriple make_tree_gridding_output(const double *cm, long cm_len, long ifreq);

    // Singleton SparseTile for f-index f (zero-copy). Returns false if f is out of [f0, f0+nf)
    // (the python None case).
    bool get_singleton(long f, SparseTile &out) const;

    // Apply DD(k), returning a triple with k -> k+1.
    SparseTileTriple iterate() const;

    // Test-only: densify to a (2^(r-k), 2^k, ntime) array.
    ksgpu::Array<double> unpack(long ntime) const;
};


// -------------------------------------------------------------------------------------------------
//
// PfVarianceConvolver, PfVariance, PfAvarApproximation


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
    //    r = tree rank = config.toplevel_tree_rank - et_level - (ipri>0 ? 1 : 0)
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

#endif  // _PIRATE_VARMAP_HPP
