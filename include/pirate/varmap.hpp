#ifndef _PIRATE_VARMAP_HPP
#define _PIRATE_VARMAP_HPP

#include <vector>
#include <memory>
#include <cstdint>          // uint64_t (the sdbits key)
#include <unordered_map>
#include <ksgpu/Array.hpp>

#include "constants.hpp"
#include "inlines.hpp"
#include "DedispersionPlan.hpp"
#include "DedispersionTree.hpp"   // SdPlan holds one by value

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
//    SparseTile, SparseTileTriple        (port of slow_avar/SparseTile.py)
//    PfVarianceConvolver, PfVariance     (port of slow_avar/PfVariance.py)
//    PfAvarApproximation                 (port of slow_avar/PfVariance.py)
//    SdPlan, compute_detrender_free_*    (port of varmap/detrender_free.py, and of the two
//                                         free functions it needs from varmap/VarianceMap.py
//                                         and varmap/VarianceMultiMap.py)
//
// SO THE FILE HOLDS TWO LAYERS, and they are at different stages of the same story.
//
// The PfAvar* classes, together with the python TmpVmap* classes built from them, were a first
// pass at representing a variance map, written before we settled on the VarianceMap
// representation (notes/variance_map.tex, section "The variance map"). They are left unchanged
// for now, and are deliberately outside it. We may revisit them later.
//
// SdPlan and compute_detrender_free_varfine() / compute_detrender_free_varcoarse() are the
// current path: the analytic, detrender-free variance vector of every tree of a config, ported
// from pirate_frb/varmap/ for speed (roughly 15.9 seconds -> under a second at
// chord_sb2_et.yml). The python is the reference, and stays so -- pirate_frb.varmap keeps the
// python implementation, pirate_frb.fast_avar exposes this one, and a unit test asserts they
// agree. The lower layer is what the upper one is built on: SdPlan's tile pass is
// SparseTileTriple::iterate(), and its rows come from PfVarianceConvolver::variance().


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
    //
    // The footprint is clipped to [flo, fhi) BEFORE it is split into the canonical one/two/three
    // tiles. Clipping at the SOURCE (rather than slicing the result) is what lets a caller iterate
    // part of a channel's footprint without the rest of the channel's weight leaking in through
    // the aligned merges -- SdPlan's tile pass is the caller that needs this. The clip range must
    // meet the footprint, i.e. leave at least one f-index.
    //
    // The defaults are the unclipped case: flo=0 is already a no-op (f0 >= 0 always), and fhi < 0
    // means "no upper clip". The python spells that second default 'None'.
    static SparseTileTriple make_tree_gridding_output(const double *cm, long cm_len, long ifreq,
                                                      long flo = 0, long fhi = -1);

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


// -------------------------------------------------------------------------------------------------
//
// SdPlan and the detrender-free variance vectors
//
// C++ port of pirate_frb/varmap/detrender_free.py, restricted to what
// compute_detrender_free_varcoarse() needs. See that file's module docstring for the algorithm and
// notes/variance_map.tex for the math.
//
// WHAT IS NOT PORTED, and why the port is so much smaller than the python. The python SdPlan
// serves two callers: compute_detrender_free_base_map(), which builds the factored variance MAP,
// and compute_detrender_free_varfine(), which computes A v without ever forming the map. Only the
// second is ported, i.e. the python's init_sd_matrices=False mode. That drops class SdMatrix and
// its per-group SVD, the coarse-graining rank 'Lmat' and threshold 'epsilon' (which the python
// REJECTS in this mode rather than ignoring), and the sd_matrices half of four methods. It also
// drops the python's 'progress' reporting, its 'stats' dict, and its 'debug' cross-checks -- see
// SdPlan::plan_pass() in the .cpp for what those checked and where they still run.
//
// The python remains the reference implementation and is fully tested (varmap/tests.py,
// test_varfine()). The only test on this side asserts that the two agree; see
// pirate_frb/fast_avar/test_fast_avar.py.


// Everything except the lift, for the base tree of 'config'. Mirrors python class SdPlan.
//
// The constructor runs the three passes -- planning, sizing, tiles -- and leaves one accumulator,
// 'sd_vectors'. THE CONSTRUCTOR DOES NO LIFTING; lift_sd_vectors() is a separate, opt-in step, for
// the same reason as in python: a caller who wants only the per-group terms would otherwise be
// handed an O(nalpha) array to discard.

class SdPlan
{
public:
    // The sdbits key is (dbits << sbits_width) | sbits, where
    //
    //    sbits = bit n set iff subband n is served by this term
    //    dbits = the delay-bit mask, IN THE LEVEL-r LABELLING (see emit())
    //
    // The split is chosen so the key fits a uint64_t: N <= 42 subbands (notes/dedispersion.tex,
    // section "Subbanded dedispersion", at R <= constants::max_peak_finding_rank == 4) and
    // r <= 16 (constants::max_tree_rank), so 42 + 16 < 64. The constructor asserts both bounds up
    // front, because raising either would silently corrupt keys rather than fail.
    static constexpr int sbits_width = 42;
    static constexpr uint64_t sbits_mask = (uint64_t(1) << sbits_width) - 1;

    // One (input channel, subband set) group on the common path. Python's unstraddled_plan entry,
    // now the dataclass _UnstraddledEntry.
    //
    // 'ifreq' is a field and not merely the loop variable that produced it: the tile pass walks the
    // flat plan and needs it to rebuild the channel's gridding triple. The sbits half of 'sdbits'
    // may have several bits set -- ONE tile computation and ONE convolution serve every subband
    // that sees the same [lo, hi), which is where this algorithm's advantage over the
    // per-multiplet route of PfAvarApproximation comes from.
    struct UnstraddledEntry {
        long ifreq;
        long lo, hi;          // tree-freq range, already intersected with the subband
        uint64_t sdbits;
    };

    // A case-2 subband whose footprint straddles its own midpoint. Python's straddled_plan entry,
    // now the dataclass _StraddledEntry. Carries no (lo, hi) because (ifreq, n) already determines
    // them, via the same intersect() the tile pass calls. A straddling subband always gets a row to
    // itself, so sdbits' sbits is exactly (1 << n) -- which size_pass() asserts.
    struct StraddledEntry {
        long ifreq;
        long n;
        uint64_t sdbits;
    };

    // One group of the accumulator: every plan entry sharing one sdbits. Python's sd_vectors dict
    // entry. 'y' holds the summed rows in the FULL delay-bit basis.
    struct SdVector {
        uint64_t sdbits;
        long D_full;                // 1 << popcount(dbits), the delay-axis length
        std::vector<double> y;      // (D_full * P) row-major
    };

    // A channel's UNCLIPPED footprint [j0, j1) in tree-freq units, cached by the planning pass.
    // The tile pass needs it to recover a straddled entry's (lo, hi).
    struct Footprint {
        long j0, j1;
    };

    // Runs the three passes. Calls config.validate(), so an invalid config throws here rather than
    // deeper in.
    //
    // 'freq_variances' is the length-nfreq vector of input-channel variances. NOT required to be
    // positive: this is defined against VarianceMap.apply(), which does not require it either. The
    // length and the dtype are checked, nothing more. Unlike the python there is no all-ones
    // default, because every C++ caller supplies one.
    SdPlan(const DedispersionConfig &config, const ksgpu::Array<double> &freq_variances);

    // The FINE (ndm, M, P) vector that sd_vectors' per-group terms add up to, i.e. A v. It is
    // UNTRUNCATED -- no SVD is involved anywhere on this path -- which is why it is more accurate
    // than the expression that defines it. See python lift_sd_vectors().
    ksgpu::Array<double> lift_sd_vectors() const;

    // The base tree, i.e. tree (primary_tree_index, early_trigger_level) == (0, 0).
    //
    // NOTE itree0 IS NOT ALWAYS ZERO: early_trigger_level DESCENDS within a primary-tree family,
    // so the e == 0 tree is the LAST of its family. It is 0 for every shipped config -- which is
    // exactly what would make an assertion to that effect a trap.
    long itree0 = 0;
    DedispersionTree tree0;

    // Geometry, with ndm = 2^(r-R) and nalpha = ndm*M*P.
    long r = 0, R = 0, N = 0, M = 0, P = 0, nfreq = 0, ndm = 0, nalpha = 0;

    // Number of straddled plan entries. The straddle branch is RARE (6 entries of 16407 at
    // chime_sb2_et.yml, 1 of 645 at toy.yml) and omitting it moves the answer by only ~1.6e-3, so
    // a port that never took it would pass a loose agreement test. Exposed so the unit test can
    // assert the branch was exercised (notes/unit_tests.md item 8).
    long n_straddled = 0;

protected:
    // The two plans; see plan_pass().
    std::vector<UnstraddledEntry> unstraddled_plan;
    std::vector<StraddledEntry> straddled_plan;
    std::vector<Footprint> footprint;                  // (nfreq,)

    // The accumulator. THE VECTOR, NOT THE MAP, IS WHAT THE LIFT WALKS: it is in insertion order,
    // which is python's dict iteration order, so the two implementations accumulate in exactly the
    // same sequence. That is not needed for correctness, but it means any disagreement the unit
    // test finds comes from the arithmetic kernels (SparseTileTriple::iterate(),
    // PfVarianceConvolver::variance()) rather than from summation order -- which is worth a lot
    // when debugging a three-pass port.
    std::vector<SdVector> sd_vectors;
    std::unordered_map<uint64_t, long> sd_vector_index;   // sdbits -> index into sd_vectors

    // The length-N per-subband tables, all in TOPLEVEL TREE-FREQ units; see subband_geometry().
    std::vector<long> lev, c, I_lo, I_hi, I_mid, mbase;
    std::vector<char> case1;      // not vector<bool>: its proxy references buy nothing at N <= 42

    ksgpu::Array<double> cmap;                // (nfreq+1,) input-channel edges in tree-freq units
    std::vector<double> freq_variances_vec;   // (nfreq,)
    PfVarianceConvolver convolver;            // ONE shared instance for the whole run

    // Scratch, reused across plan entries so the tile pass allocates nothing per entry.
    std::vector<double> var_scratch;
    mutable std::vector<long> bit_scratch;

    void subband_geometry();
    void plan_pass();
    void size_pass();
    void tile_pass();

    void count(uint64_t sdbits);
    void emit(long ifreq, const SparseTile &tile, long klev, uint64_t sdbits);

    // This channel's footprint [j0, j1) intersected with subband n. Empty iff lo >= hi. Shared by
    // the planning pass and the tile pass so the two cannot drift.
    void intersect(long j0, long j1, long n, long &lo, long &hi) const;

    // The level-r dbits of the range [lo, hi), for any subband n of its entry. See python
    // predict_dbits_r() for why subband-LOCAL coordinates are the right ones on both branches.
    long predict_dbits_r(long lo, long hi, long n) const;

    // The set bit positions of 'bits', ascending, appended to 'out' (which is cleared first).
    //
    // DELIBERATELY KNOWS NOTHING ABOUT THE sdbits PACKING. A caller iterating a group's subbands
    // passes 'sdbits & sbits_mask' explicitly, so that the mask reads as the load-bearing step it
    // is: without it the delay bits would arrive as subband indices.
    static void iter_bits(uint64_t bits, std::vector<long> &out);
};


// Max-reduce a length-nalpha vector to the length-nbeta coarse groups at rank L, i.e.
// out[beta] = max over alpha in beta of y[alpha], with R <= L <= r. Mirrors python
// coarse_grain_vector(); see it and VarianceMap's module docstring for the index conventions.
//
// 'y' has length 2^(r-R) * M * P; the result has length 2^(r-L) * N * P.
//
// THE REDUCTION IS A MAX, not a mean: a stored variance has to dominate every output it covers.
// Being a max, it is exact and order-independent, so this cannot disagree with the python by a
// rounding difference the way the summations elsewhere in this file can.
extern ksgpu::Array<double> coarse_grain_vector(const DedispersionTree &tree,
                                                const double *y, long ylen, long L);


// Length-ntrees vector of (D, M, P) arrays from one FINE vector per PRIMARY tree, indexed by
// ITREE. Mirrors python expand_fine_vectors().
//
// 'per_primary' is one flat, length-nalpha array per primary tree, in gamma order.
//
// THE CHILD TREES COME FROM PROPOSITION 1 of the appendix "Variance maps of a config's trees are
// row-restrictions of one another" in notes/variance_map.tex: an early-trigger tree's map is a
// subset of its parent's ROWS, and row selection commutes with A @ v, so its result is the
// corresponding subset of the parent's. Nothing here assumes anything about the upstream chain, so
// unlike Proposition 2 this holds with a detrender too.
//
// THE PARENT IS THE LAST TREE OF ITS FAMILY, not the first: early_trigger_level DESCENDS within a
// family, so iparent is NOT itree - e. That trap is the reason this is one function rather than a
// loop each caller writes.
//
// NO TWO ENTRIES OF THE RESULT SHARE STORAGE, which callers rely on. A child's is a fresh copy; a
// parent's is a reshaped VIEW of the corresponding per_primary entry (Array is refcounted, so it
// outlives the caller's vector). So the caller must hand in per_primary entries that do not alias
// each other -- compute_detrender_free_varfine() copies rather than slices for exactly this
// reason, and the python says the same thing about ascontiguousarray().
extern std::vector<ksgpu::Array<double>>
expand_fine_vectors(const DedispersionConfig &config,
                    const std::vector<ksgpu::Array<double>> &per_primary);


// 'A v' at FINE granularity for EVERY tree of 'config'. No detrender, no map formed.
//
// Returns a length-ntrees vector of (2^(r-R), M, P) arrays INDEXED BY ITREE, each in its own tree's
// geometry. Mirrors python compute_detrender_free_varfine(); see it for why this is more accurate
// than the expression that defines it, and for the two propositions it rests on.
//
// THE NO-DETRENDER HYPOTHESIS IS LOAD-BEARING. The step from the base tree to the other PRIMARY
// trees is Proposition 2, which is FALSE with a Detrender2d in front (measured against the
// brute-force sweep: 4.9e-7 without one, 2.1 with). A future detrender path must not be routed
// through this function.
extern std::vector<ksgpu::Array<double>>
compute_detrender_free_varfine(const DedispersionConfig &config,
                               const ksgpu::Array<double> &freq_variances);


// 'A v' for every tree of 'config', max-reduced to the WEIGHTS array's granularity.
//
// Returns a length-ntrees vector of (ndm_wt, N, P) arrays INDEXED BY ITREE, i.e.
// compute_detrender_free_varfine() followed by a per-tree coarse_grain_vector(). This is the form
// production stores: the weights array resolves DM only to pf.wt_dm_downsampling and frequency only
// to SUBBANDS rather than multiplets, so one entry here is one variance per weights-array element.
//
// THE COARSE-GRAINING RANK IS THE TREE'S OWN, L = log2(pf.wt_dm_downsampling), and that is what
// makes the result line up: 2^(r-L) is exactly DedispersionTree::ndm_wt, checked below. Note L is a
// property of the PRIMARY tree while r is not, so L is constant within an early-trigger family and
// ndm_wt still varies across it.
//
// NOT THE SAME AS compute_detrender_free_base_map(config, L=...).apply(v) on the python side, which
// is sum_F max_alpha A[alpha,F] v_F and DOMINATES the max_alpha (A v)[alpha] computed here -- that
// one maxes the MAP, this one maxes the ANSWER.
extern std::vector<ksgpu::Array<double>>
compute_detrender_free_varcoarse(const DedispersionConfig &config,
                                 const ksgpu::Array<double> &freq_variances);


}  // namespace pirate

#endif  // _PIRATE_VARMAP_HPP
