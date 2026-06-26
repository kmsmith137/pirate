#ifndef _PIRATE_SPARSE_TILE_HPP
#define _PIRATE_SPARSE_TILE_HPP

#include <memory>
#include <ksgpu/Array.hpp>

#include "constants.hpp"
#include "inlines.hpp"

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// C++ port of (a subset of) pirate_frb/slow_avar/SparseTile.py. See that file and
// notes/tree_dedispersion.tex for the math. Only the methods reachable from
// PfAvarApproximation (plus the test-only unpack()) are ported here.
//
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


}  // namespace pirate

#endif  // _PIRATE_SPARSE_TILE_HPP
