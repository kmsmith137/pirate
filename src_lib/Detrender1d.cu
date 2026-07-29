#include "../include/pirate/Detrender1d.hpp"

#include <sstream>
#include <iostream>
#include <ksgpu/xassert.hpp>
#include <ksgpu/cuda_utils.hpp>
#include <ksgpu/KernelTimer.hpp>

using namespace std;
using namespace ksgpu;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// -------------------------------------------------------------------------------------------------
//
// Overview.
//
// Notation follows notes/tree_dedispersion.tex, section "Time detrending algorithm 1:
// local polynomial subtraction". The moment monoid, the van Herk block decomposition,
// and the per-window solve are all specified there; pirate_frb/detrending_1d is a
// line-by-line numpy implementation of the same algorithm, and is the oracle this
// kernel is tested against.
//
// Geometry (all compile-time). The window is 2W+1 samples and the scan blocks are
// B = 2W samples, so a window spans exactly two adjacent blocks for every alignment.
// The buffer is T + 2W samples = T/B + 1 blocks, and output j (0 <= j < T) has window
// [j, j+B] in buffer coordinates:
//
//    M(window of output j) = Suff_b[p] (+) Pref_{b+1}[p],   b = j/B,  p = j%B
//
// where Pref/Suff are inclusive prefix/suffix scans of the monoid within a block.
//
// Thread mapping. One warp per row; lane 'l' owns sub-block 'l' of a block, i.e. the
// L = B/32 consecutive samples [Ll, Ll+L). Writing p = Ll + i,
//
//    Suff_b[p]     = R_suf(i)  (+)  Q_{b,l}         R_suf(i) = local samples i..L-1 of block b
//    Pref_{b+1}[p] = P_{b+1,l} (+)  R_pre(i)        R_pre(i) = local samples 0..i of block b+1
//
// where Q_{b,l} is the exclusive suffix of block b's 32 sub-block aggregates (sub-blocks
// l+1..31) and P_{b+1,l} the exclusive prefix (sub-blocks 0..l-1). Neither depends on i,
// so their merge O_{b,l} = Q_{b,l} (+) P_{b+1,l} is computed once per (pair, lane) and
//
//    M(window) = R_suf(i) (+) O_{b,l} (+) R_pre(i)                                     (*)
//
// costs two merges per output. Sanity check on the sample counts:
// (L-i) + (B - L(l+1)) + Ll + (i+1) = B+1 = 2W+1 for all l, i.
//
// The payoff is that lane l produces its L outputs with no cross-lane traffic except a
// single shuffle for the output sample itself: all lane cooperation is confined to the
// once-per-block scan over sub-block aggregates. There is no shared memory and no
// __syncthreads() anywhere in the kernel.
//
// Three levels of moment accumulation, and why:
//
//   - Within a sub-block, moments are raw running sums about a fixed origin (struct Raw).
//     No centroid, no Pascal shift. The span is only L/W in normalized units, so
//     adaptive centering buys nothing there.
//   - A sub-block aggregate is converted to centroid-carrying form (struct Canon) by a
//     single Pascal shift, and the 32 aggregates are scanned with warp shuffles.
//   - Per output, the two raw partials of (*) are merged into O_{b,l} directly from their
//     own origins, by merge_raw_canon().
//
// It is NOT safe to save a merge by combining the two raw partials about a common origin
// first: they live in blocks B samples apart, so the common origin would be ~W away from
// either, and a window whose valid samples are a narrow cluster near one window edge
// (which is well conditioned and survives) would have its moments computed by cancelling
// O(N_v) quantities down to O(N_v (w/W)^(2n)). This is the same failure the tex describes
// for scans that do not carry their own centroid.
//
// Everything below is templated on (NDEG, W, T) and instantiated by the dispatch switch
// in Detrender1d::launch(). The moment monoid itself depends only on NDEG: the centroid
// is carried in units of W (see struct Canon), which is exact since W is a power of two,
// and that removes W from all the monoid arithmetic.


static constexpr int WARPS_PER_BLOCK = 4;
static constexpr unsigned int ALL_LANES = 0xffffffffU;

static constexpr float EPS = Detrender1d::eps;
static constexpr float MU = Detrender1d::mu;

// The Makefile compiles everything with --use_fast_math, which turns '/' into an
// unrefined MUFU.RCP (~2^-22 relative). That is not good enough in two places, so both
// use the correctly-rounded intrinsics, which --use_fast_math does not affect:
//
//   - the merge reciprocal, because S1 is FORCED to zero rather than computed, so an
//     error eta in the merged centroid leaves a spurious S1 ~ N*eta behind, which
//     competes with S2 ~ N (w/W)^2 when the valid samples are a cluster of width w;
//   - the final divide by det(G), because fhat is of order |d - kappa|, which the tex
//     allows to reach 1e3 sigma, and 2^-22 of that is 2.4e-4 sigma -- above the 1e-4
//     sigma the whole float32 design is budgeted for.
//
// Anywhere else (kappa, which needs no accuracy at all) plain '/' is fine.
static __device__ __forceinline__ float safe_rcp(float x)
{
    return (x > 0.0f) ? __frcp_rn(x) : 0.0f;
}


// -------------------------------------------------------------------------------------------------
//
// The moment monoid (notes/tree_dedispersion.tex, "The moment monoid"), for general
// degree NDEG. Everything here is degree-generic -- the change of origin is the only
// degree-dependent arithmetic, and pascal_shift() below handles any degree -- which is
// what makes adding a degree a matter of instantiating a template.


// Canon: a monoid element in canonical form, i.e. moments taken about the set's own
// mask-weighted centroid. S0 = N and S1 = 0 are implicit and not stored, so the element
// is 3n+2 floats, as in the tex. S[k] holds S_{k+2}.
//
// The centroid 'ct' is in units of W, and relative to the start of the buffer: absolute
// stream coordinates would destroy float32 precision in a long run. Since W is a power
// of two this is exactly the numpy reference's sample-unit centroid divided by W, and
// every operation below is the reference's with both sides scaled by W -- float rounding
// is invariant under scaling by a power of two, so the two agree bit for bit. Carrying
// the scaled version is what removes W from the monoid arithmetic entirely.
template<int NDEG>
struct Canon
{
    float N;                 // number of valid samples (= S0)
    float ct;                // mask-weighted centroid, in units of W
    float S[2*NDEG - 1];     // mask moments S_2 .. S_{2n} about ct
    float U[NDEG + 1];       // data moments U_0 .. U_n about ct
};


// Raw: moments about a FIXED origin carried by the caller, so R[1] is not zero.
// Used for partial sums within a sub-block, where accumulation is a plain running
// sum with compile-time coefficients.
template<int NDEG>
struct Raw
{
    float R[2*NDEG + 1];     // mask moments R_0 .. R_{2n} about the origin
    float V[NDEG + 1];       // data moments V_0 .. V_n
};


template<int NDEG>
static __device__ __forceinline__ void raw_zero(Raw<NDEG> &a)
{
    #pragma unroll
    for (int r = 0; r <= 2*NDEG; r++)
        a.R[r] = 0.0f;
    #pragma unroll
    for (int r = 0; r <= NDEG; r++)
        a.V[r] = 0.0f;
}


// raw_add() / raw_sub(): accumulate (or remove) one sample. 'm' is the mask (0 or 1),
// 'dz' the masked, offset-subtracted data m*(d-kappa), and 'y' the sample's coordinate
// (u - origin)/W, which is a compile-time constant at every call site.
template<int NDEG>
static __device__ __forceinline__ void raw_add(Raw<NDEG> &a, float m, float dz, float y)
{
    float yp = 1.0f;
    #pragma unroll
    for (int r = 0; r <= 2*NDEG; r++) {
        a.R[r] += m * yp;
        if (r <= NDEG)
            a.V[r] += dz * yp;
        yp *= y;
    }
}

template<int NDEG>
static __device__ __forceinline__ void raw_sub(Raw<NDEG> &a, float m, float dz, float y)
{
    float yp = 1.0f;
    #pragma unroll
    for (int r = 0; r <= 2*NDEG; r++) {
        a.R[r] -= m * yp;
        if (r <= NDEG)
            a.V[r] -= dz * yp;
        yp *= y;
    }
}


// pascal_shift(): the change of origin of eq. (dt_shift), in place,
//
//     v[r] <- sum_{j <= r} C(r,j) delta^(r-j) v[j],
//
// evaluated as K rounds of Horner rather than from the binomial coefficients directly:
//
//     for k = 1..K:  for r = K down to k:  v[r] += delta * v[r-1]
//
// which is K(K+1)/2 FMAs with no binomials and no powers of delta -- cheaper than the
// direct form as well as shorter.
//
// Writing the coefficients out is a trap here, and an expensive one. A constexpr C(r,j)
// helper does NOT constant-fold inside these loops: nvcc leaves a runtime integer
// division behind (~740 IABS/IMAD.HI pairs in the SASS), which cost 2.5x the instruction
// count and 6x the runtime at n=2. Hardcoding the coefficients per degree instead is
// exactly what this template exists to avoid. So: no binomials.
template<int K>
static __device__ __forceinline__ void pascal_shift(float (&v)[K+1], float d)
{
    #pragma unroll
    for (int k = 1; k <= K; k++) {
        #pragma unroll
        for (int r = K; r >= k; r--)
            v[r] += d * v[r-1];
    }
}


// canon_to_full(): expand a Canon's mask moments to the full vector S_0 .. S_{2n}, using
// the implicit S_0 = N and S_1 = 0.
template<int NDEG>
static __device__ __forceinline__ void canon_to_full(const Canon<NDEG> &a, float (&s)[2*NDEG+1])
{
    s[0] = a.N;
    s[1] = 0.0f;
    #pragma unroll
    for (int k = 0; k < 2*NDEG-1; k++)
        s[k+2] = a.S[k];
}


// merge_canon(): the monoid merge of two canonical elements, MomentSet.merge() verbatim.
//
// The centroid update is written as c + f*Delta rather than (N*c + N'*c')/N to avoid
// forming the large product. S1 vanishes by construction of the new centroid and is
// forced to zero rather than computed, which is what keeps G_01 = 0 and hence makes the
// pivots p_0 = G_00 and p_1 = G_11 exact in the solve.
//
// Empty-set rule (the NaN trap): the centroid is undefined when N = 0, and a scan
// performs many merges involving empty aggregates (any fully masked sub-block). Two
// things keep this branch-free and NaN-free: every element carries a FINITE nominal
// centroid (an empty aggregate or scan identity uses its own sub-block center), and the
// divide is guarded. Merging empty with non-empty then gives f = 1, db = 0, ct = b.ct,
// which is correct.
template<int NDEG>
static __device__ __forceinline__ Canon<NDEG> merge_canon(const Canon<NDEG> &a, const Canon<NDEG> &b)
{
    float N = a.N + b.N;
    float inv = safe_rcp(N);
    float Delta = b.ct - a.ct;
    float f = b.N * inv;

    float da = -f * Delta;              // a.ct - ct_new
    float db = Delta + da;              // b.ct - ct_new

    float sa[2*NDEG+1], sb[2*NDEG+1];
    canon_to_full<NDEG>(a, sa);
    canon_to_full<NDEG>(b, sb);
    pascal_shift<2*NDEG>(sa, da);
    pascal_shift<2*NDEG>(sb, db);

    float ua[NDEG+1], ub[NDEG+1];
    #pragma unroll
    for (int k = 0; k <= NDEG; k++) {
        ua[k] = a.U[k];
        ub[k] = b.U[k];
    }
    pascal_shift<NDEG>(ua, da);
    pascal_shift<NDEG>(ub, db);

    Canon<NDEG> r;
    r.N = N;
    r.ct = a.ct - da;
    #pragma unroll
    for (int k = 0; k < 2*NDEG-1; k++)
        r.S[k] = sa[k+2] + sb[k+2];    // sa[1]+sb[1] is the S_1 that is forced to zero
    #pragma unroll
    for (int k = 0; k <= NDEG; k++)
        r.U[k] = ua[k] + ub[k];
    return r;
}


// merge_raw_canon(): merge a raw element (moments about the fixed origin 'ot', in units
// of W) with a canonical one. Mathematically this is merge_canon() applied to the
// canonicalization of 'a', but the raw side's Pascal shift is taken directly from ot to
// the merged centroid rather than from ot to a's own centroid and then to the merged one.
// That saves a shift and a reciprocal, and is strictly more accurate: one origin hop, not
// two.
//
// The result is symmetric in its operands, so it does not matter whether the raw side is
// the earlier or the later sample range.
template<int NDEG>
static __device__ __forceinline__ Canon<NDEG> merge_raw_canon(const Raw<NDEG> &a, float ot,
                                                              const Canon<NDEG> &b)
{
    float N = a.R[0] + b.N;
    float inv = safe_rcp(N);
    float Delta = b.ct - ot;
    float s = (a.R[1] + b.N * Delta) * inv;    // s = ct_new - ot

    float da = -s;                      // ot - ct_new
    float db = Delta - s;               // b.ct - ct_new

    // The raw side is already a full moment vector (R[1] is nonzero in general); the
    // canonical side needs S_0 = N and S_1 = 0 filled in.
    float sa[2*NDEG+1], sb[2*NDEG+1];
    #pragma unroll
    for (int k = 0; k <= 2*NDEG; k++)
        sa[k] = a.R[k];
    canon_to_full<NDEG>(b, sb);
    pascal_shift<2*NDEG>(sa, da);
    pascal_shift<2*NDEG>(sb, db);

    float ua[NDEG+1], ub[NDEG+1];
    #pragma unroll
    for (int k = 0; k <= NDEG; k++) {
        ua[k] = a.V[k];
        ub[k] = b.U[k];
    }
    pascal_shift<NDEG>(ua, da);
    pascal_shift<NDEG>(ub, db);

    Canon<NDEG> r;
    r.N = N;
    r.ct = ot + s;
    #pragma unroll
    for (int k = 0; k < 2*NDEG-1; k++)
        r.S[k] = sa[k+2] + sb[k+2];
    #pragma unroll
    for (int k = 0; k <= NDEG; k++)
        r.U[k] = ua[k] + ub[k];
    return r;
}


// canonicalize(): convert a raw sub-block total (about origin 'ot') to canonical form, by
// shifting the origin to the set's centroid ct = ot + R1/R0. The origin travel is at most
// half a sub-block, L/2W, so the cancellation in the shift is bounded by (1 + L/2W)^(2n).
//
// An empty sub-block (R0 = 0) gets s = 0, hence ct = ot: a finite nominal centroid, as
// the empty-set rule of merge_canon() requires.
template<int NDEG>
static __device__ __forceinline__ Canon<NDEG> canonicalize(const Raw<NDEG> &a, float ot)
{
    float s = a.R[1] * safe_rcp(a.R[0]);
    float d = -s;

    float sv[2*NDEG+1];
    #pragma unroll
    for (int k = 0; k <= 2*NDEG; k++)
        sv[k] = a.R[k];
    pascal_shift<2*NDEG>(sv, d);

    float uv[NDEG+1];
    #pragma unroll
    for (int k = 0; k <= NDEG; k++)
        uv[k] = a.V[k];
    pascal_shift<NDEG>(uv, d);

    Canon<NDEG> r;
    r.N = a.R[0];
    r.ct = ot + s;
    #pragma unroll
    for (int k = 0; k < 2*NDEG-1; k++)
        r.S[k] = sv[k+2];
    #pragma unroll
    for (int k = 0; k <= NDEG; k++)
        r.U[k] = uv[k];
    return r;
}


// solve_and_test(): evaluate the fit at x0 = (t - c)/W, and decide whether to keep the
// sample. Returns fhat and sets 'ok' to (rmin >= eps), which the caller ANDs with the
// input mask.
//
// The normal equations are G a = U with G_{jl} = S_{j+l} and G_01 = 0, and the LDL^T
// pivots satisfy p_i/G_ii = 1 for i = 0,1 exactly, so rmin = min_{i>=2} p_i/G_ii. What
// that leaves is degree-dependent, and so is the cheapest way to compute it:
//
//   n=1: there is no i >= 2, so G is diagonal and rmin is 1 iff S_0 > 0 and S_2 > 0
//        (S_2 vanishes exactly when every valid sample sits at the centroid, i.e. when
//        N_v <= 1). Mask expansion is then a valid-sample count and eps is inert. The
//        solve is two scalar divides, combined here over det = S_0 S_2 so that only one
//        reciprocal is needed.
//   n=2: only i = 2 binds, and rmin = p_2/G_22 = det(G)/(G_00 G_11 G_22), so the masking
//        test becomes the division-free comparison det >= eps*N*A*C and the fit evaluates
//        through the adjugate -- one reciprocal, no square roots.
//
// Both forms are algebraically identical to LocalPolyFit.cholesky/solve, so they inherit
// its error bound: near the threshold det is a cancelling difference with relative error
// ~eps_mach/eps, and the absolute error in rmin itself is O(eps_mach). At n=1 there is no
// threshold to be near -- the equilibrated G is the identity.
//
// A degree n>=3 would need an unrolled Cholesky here rather than either form, since two
// ratios bind and min(p_2/G_22, p_3/G_33) cannot be recovered from det(G) alone. That is
// the one part of this file that a new degree cannot inherit.
template<int NDEG>
static __device__ __forceinline__ float solve_and_test(const Canon<NDEG> &ms, float x0, bool &ok)
{
    // MU is a NaN guard, not a tuning parameter: it is inert wherever the sample is kept,
    // and on a sample we drop, fhat is discarded by the caller's select.
    if constexpr (NDEG == 1) {
        float S0 = ms.N, S2 = ms.S[0];
        ok = (S0 > 0.0f) && (S2 > 0.0f);
        return __fdiv_rn(ms.U[0]*S2 + ms.U[1]*S0*x0, fmaxf(S0*S2, MU));
    }
    else {
        static_assert(NDEG == 2);
        float N = ms.N, A = ms.S[0], Bm = ms.S[1], C = ms.S[2];

        float adj00 = A*C - Bm*Bm;
        float adj01 = A*Bm;
        float adj02 = -A*A;
        float adj11 = N*C - A*A;
        float adj12 = -N*Bm;
        float adj22 = N*A;
        float det = N*adj00 + A*adj02;

        float v0 = adj00*ms.U[0] + adj01*ms.U[1] + adj02*ms.U[2];
        float v1 = adj01*ms.U[0] + adj11*ms.U[1] + adj12*ms.U[2];
        float v2 = adj02*ms.U[0] + adj12*ms.U[1] + adj22*ms.U[2];

        ok = (N > 0.0f) && (A > 0.0f) && (C > 0.0f) && (det >= EPS*N*A*C);
        return __fdiv_rn(v0 + x0*(v1 + x0*v2), fmaxf(det, MU));
    }
}


// -------------------------------------------------------------------------------------------------
//
// Warp-level scans over the 32 sub-block aggregates of one block.


template<int NDEG>
static __device__ __forceinline__ Canon<NDEG> canon_shfl(const Canon<NDEG> &x, int src_lane)
{
    Canon<NDEG> r;
    r.N  = __shfl_sync(ALL_LANES, x.N,  src_lane);
    r.ct = __shfl_sync(ALL_LANES, x.ct, src_lane);
    #pragma unroll
    for (int k = 0; k < 2*NDEG-1; k++)
        r.S[k] = __shfl_sync(ALL_LANES, x.S[k], src_lane);
    #pragma unroll
    for (int k = 0; k <= NDEG; k++)
        r.U[k] = __shfl_sync(ALL_LANES, x.U[k], src_lane);
    return r;
}


// canon_identity(): the monoid identity, with a finite nominal centroid (see the
// empty-set rule in merge_canon()).
template<int NDEG>
static __device__ __forceinline__ Canon<NDEG> canon_identity(float ct_nominal)
{
    Canon<NDEG> r;
    r.N = 0.0f;
    r.ct = ct_nominal;
    #pragma unroll
    for (int k = 0; k < 2*NDEG-1; k++)
        r.S[k] = 0.0f;
    #pragma unroll
    for (int k = 0; k <= NDEG; k++)
        r.U[k] = 0.0f;
    return r;
}


// warp_scans(): given one sub-block aggregate per lane (in sub-block order), return the
// EXCLUSIVE prefix and suffix scans over the 32 lanes.
//
// Hillis-Steele, 5 steps each. The reason the scan must be a tree at all is roundoff:
// each output then depends on its leaves through O(log 32) merges rather than O(32).
// Every step merges two disjoint, adjacent ranges, which is what merge_canon() requires.
template<int NDEG>
static __device__ __forceinline__ void warp_scans(const Canon<NDEG> &agg, int lane, float ot,
                                                  Canon<NDEG> &excl_pre, Canon<NDEG> &excl_suf)
{
    Canon<NDEG> pre = agg;
    Canon<NDEG> suf = agg;

    #pragma unroll
    for (int k = 1; k < 32; k *= 2) {
        Canon<NDEG> up = canon_shfl<NDEG>(pre, lane - k);
        if (lane >= k)
            pre = merge_canon<NDEG>(up, pre);

        Canon<NDEG> dn = canon_shfl<NDEG>(suf, lane + k);
        if (lane + k < 32)
            suf = merge_canon<NDEG>(suf, dn);
    }

    // Inclusive -> exclusive. The identity's nominal centroid is this lane's own
    // sub-block center, which is finite and otherwise arbitrary.
    excl_pre = canon_shfl<NDEG>(pre, lane - 1);
    excl_suf = canon_shfl<NDEG>(suf, lane + 1);
    if (lane == 0)
        excl_pre = canon_identity<NDEG>(ot);
    if (lane == 31)
        excl_suf = canon_identity<NDEG>(ot);
}


// -------------------------------------------------------------------------------------------------
//
// The kernel.


// Cfg: the derived geometry of one compiled configuration. Everything here follows from
// (NDEG, W, T); the static_asserts are the assumptions the kernel body relies on.
template<int NDEG, int W, int T>
struct Cfg
{
    static constexpr int B = 2*W;          // scan block length
    static constexpr int L = B / 32;       // samples per lane (one sub-block)
    static constexpr int NBUF = T + 2*W;   // buffer samples per row
    static constexpr int NPAIR = T / B;    // block pairs, i.e. loop trips
    static constexpr int NBLK = NPAIR + 1; // blocks per buffer

    static_assert((NDEG == 1) || (NDEG == 2));    // the degrees solve_and_test() supports
    static_assert(T % B == 0);                    // block lattice is chunk-invariant
    static_assert(NBLK * B == NBUF);
    static_assert((L == 8) || (L == 16));         // L mask bytes = one uint2 or uint4
    static_assert((W & (W-1)) == 0);              // power of two: see sb_y() below

    // sb_y(): the sub-block-local coordinate x = (u - origin)/W of local sample i, where
    // the origin is the sub-block center.
    //
    // Since W is a power of two, y = (2i-(L-1))/(2W) is exact, and so is every power up
    // to y^(2n): the numerator is an integer bounded by (L-1)^(2n) and the denominator is
    // a power of two, and both fit in a float32 mantissa at the sizes here. A sum of L
    // such terms is exactly representable too, so ALL the raw mask moments of a sub-block
    // -- and every running prefix and suffix of them, including the ones formed by
    // subtraction in the output loop -- are computed exactly. Rounding enters the mask
    // moments only at the Pascal shift, never in the accumulation.
    static __device__ __forceinline__ float sb_y(int i)
    {
        return (float(i) - 0.5f*float(L-1)) * (1.0f / float(W));
    }

    // Origin of this lane's sub-block of block beta, in units of W and relative to the
    // buffer start. Exact: the numerator is a multiple of 1/2 and W is a power of two.
    static __device__ __forceinline__ float origin(int beta, int lane)
    {
        return (float(B*beta + L*lane) + 0.5f*float(L-1)) * (1.0f / float(W));
    }
};


// One block of the row, as held by one lane: L consecutive samples plus the block's
// constant offset and the scan results.
//
// The offset kappa is per BLOCK, not per buffer as in Detrender.detrend_chunk(). Any
// value is mathematically exact -- the polynomial basis contains the constant function,
// so replacing d by d-kappa shifts fhat by exactly kappa and leaves the residual
// unchanged -- and the only requirement is |d - kappa| <~ 1e3 sigma during accumulation.
// A per-block kappa is if anything more local than a per-buffer one, and it lets the
// kernel read each row exactly once: a per-buffer kappa would need a second pass over
// the row, since the buffer does not fit in a warp's registers.
//
// The price is that the two blocks of a pair are in different offset frames and must be
// reconciled before their moments are merged. The conversion is exact and cheap: for
// moments about any common origin, U_r(kappa) = U_r(kappa') + (kappa' - kappa) S_r, and
// S_0 = N, S_1 = 0.
template<int NDEG, int W, int T>
struct BlockState
{
    float dz[Cfg<NDEG,W,T>::L];   // masked, offset-subtracted data m*(d - kappa)
    unsigned int mb;              // mask bits: bit i is sample i
    float kappa;
    float nv;                     // valid samples in the whole block (all 32 lanes), i.e. is kappa meaningful
    Canon<NDEG> pre;              // exclusive prefix of this block's sub-block aggregates
    Canon<NDEG> suf;              // exclusive suffix
};


static __device__ __forceinline__ float mask_bit(unsigned int mb, int i)
{
    return float((mb >> i) & 1U);
}


// load_block(): load block 'beta' of the row into 'bs', and do all the per-block work
// (constant offset, sub-block aggregate, warp scans).
//
// Pointer offsets: 'dptr'/'mptr' have already had the per-row offset applied, so they
// point at shape-(NBUF,) contiguous arrays. Adding (B*beta + L*lane) leaves a shape-(L,)
// contiguous array, which is this lane's sub-block.
//
// The mask load is one uint2 (L=8) or uint4 (L=16) per lane: 32L contiguous bytes per
// warp, i.e. whole cache lines per instruction. The data load is L/4 float4's per lane
// covering L consecutive floats, so a single instruction touches 4L cache lines and uses
// 512 of their bytes; the L/4 together cover them exactly once, so DRAM traffic is
// unaffected and only the L1 wavefront count goes up (measured at ~2% of the kernel).
// Making each instruction cache-line-perfect would need a coalesced load plus a warp
// transpose, which is not worth the complexity here.
template<int NDEG, int W, int T>
static __device__ __forceinline__ void load_block(BlockState<NDEG,W,T> &bs, const float *dptr,
                                                  const unsigned char *mptr, int beta, int lane)
{
    using C = Cfg<NDEG, W, T>;
    constexpr int L = C::L;

    long off = long(C::B)*beta + long(L)*lane;

    float d[L];
    #pragma unroll
    for (int k = 0; k < L/4; k++) {
        float4 q = *((const float4 *) (dptr + off + 4*k));
        d[4*k+0] = q.x;  d[4*k+1] = q.y;  d[4*k+2] = q.z;  d[4*k+3] = q.w;
    }

    unsigned int mw[L/4];
    if constexpr (L == 8) {
        uint2 mv = *((const uint2 *) (mptr + off));
        mw[0] = mv.x;  mw[1] = mv.y;
    }
    else {
        uint4 mv = *((const uint4 *) (mptr + off));
        mw[0] = mv.x;  mw[1] = mv.y;  mw[2] = mv.z;  mw[3] = mv.w;
    }

    unsigned int mb = 0;
    #pragma unroll
    for (int i = 0; i < L; i++)
        mb |= ((mw[i >> 2] >> (8*(i & 3))) & 0xffU) ? (1U << i) : 0U;
    bs.mb = mb;

    // Constant offset: the masked mean of this block. Precision is irrelevant here (any
    // value is exact), so a plain warp reduction over the B samples is fine.
    float sm = 0.0f, smd = 0.0f;
    #pragma unroll
    for (int i = 0; i < L; i++) {
        float m = mask_bit(mb, i);
        sm += m;
        smd += m * d[i];
    }
    #pragma unroll
    for (int k = 16; k > 0; k >>= 1) {
        sm += __shfl_xor_sync(ALL_LANES, sm, k);
        smd += __shfl_xor_sync(ALL_LANES, smd, k);
    }
    float kappa = (sm > 0.0f) ? (smd / sm) : 0.0f;
    bs.kappa = kappa;
    bs.nv = sm;

    Raw<NDEG> tot;
    raw_zero<NDEG>(tot);
    #pragma unroll
    for (int i = 0; i < L; i++) {
        float m = mask_bit(mb, i);
        float dz = m * (d[i] - kappa);
        bs.dz[i] = dz;
        raw_add<NDEG>(tot, m, dz, C::sb_y(i));
    }

    float ot = C::origin(beta, lane);
    warp_scans<NDEG>(canonicalize<NDEG>(tot, ot), lane, ot, bs.pre, bs.suf);
}


// detrend_1d_kernel(): one warp per row.
//
// __launch_bounds__ carries no min-blocks-per-SM hint on purpose. The kernel wants a lot
// of registers (mostly the 2 x L samples of block b and b+1 that each lane holds) -- 168
// at (n,W)=(2,256), 96 at (1,128) -- and asking for more occupancy makes it worse, not
// better: at n=2, forcing 128 registers spills 192 bytes and drops the kernel from 688 to
// 533 GB/s, and 96 registers to 334 GB/s. There is nothing to gain anyway, since both
// configurations run at or near the memory roof -- see time_selected().
//
// In-place overwrite is safe, and the ordering requirement is load-bearing: pair b writes
// buffer samples [B*b + W, B*b + W + B), which is the second half of block b and the first
// half of block b+1. Both blocks are already resident in warp registers when the writes
// happen, and neither is re-read from global memory afterwards (block b is discarded, and
// block b+1 rotates into the register slot for pair b+1). So block b+1 must be LOADED
// BEFORE pair b executes, which is what the loop below does. Do not reorder it.
template<int NDEG, int W, int T>
__global__ void __launch_bounds__(32*WARPS_PER_BLOCK)
detrend_1d_kernel(float *data, unsigned char *mask, int M)
{
    using C = Cfg<NDEG, W, T>;
    constexpr int L = C::L;
    constexpr int B = C::B;

    int lane = threadIdx.x;
    long row = long(blockIdx.x) * WARPS_PER_BLOCK + threadIdx.y;

    // 'row' depends only on threadIdx.y, so it is warp-uniform and whole warps exit
    // together. That is what makes the early return safe: the kernel has no
    // __syncthreads(), but it is full of __shfl_sync()'s that need all 32 lanes.
    if (row >= M)
        return;

    // Apply per-warp (= per-row) pointer offset.
    //   before: shape (M, NBUF), contiguous
    //   after: shape (NBUF,), contiguous
    float *dptr = data + row * long(C::NBUF);
    unsigned char *mptr = mask + row * long(C::NBUF);

    BlockState<NDEG,W,T> s0, s1;
    load_block<NDEG,W,T>(s0, dptr, mptr, 0, lane);

    for (int b = 0; b < C::NPAIR; b++) {
        load_block<NDEG,W,T>(s1, dptr, mptr, b+1, lane);

        // Everything in this pair is computed in ONE offset frame, normally block b's.
        //
        // The exception is load-bearing. A fully masked block has no meaningful kappa
        // (the guarded divide in load_block() returns 0), and adopting that as the frame
        // would add the whole system temperature back into the OTHER block's data
        // moments -- exactly the "kappa collapses to zero" failure the constant-offset
        // subtraction exists to prevent, and worth ~1e-3 sigma at an offset of 1e3 sigma.
        // So when block b is empty we use block b+1's frame instead. Block b's dz are
        // then all zero and all its samples are masked, so it needs no reframing, and it
        // contributes nothing to any moment.
        //
        // The mirror case needs no guard: if block b+1 is empty then dk is garbage, but
        // every use of dk below is multiplied by one of that block's mask-weighted
        // quantities (m, N, S_r), all of which are zero.
        float dk = (s0.nv > 0.0f) ? (s1.kappa - s0.kappa) : 0.0f;

        // Reframe block b+1's exclusive prefix, using U_r(kappa) = U_r(kappa') + dk*S_r
        // with S_0 = N and S_1 = 0 (so U[1] is unchanged).
        Canon<NDEG> pre = s1.pre;
        pre.U[0] += dk * pre.N;
        #pragma unroll
        for (int rr = 2; rr <= NDEG; rr++)
            pre.U[rr] += dk * pre.S[rr-2];
        Canon<NDEG> O = merge_canon<NDEG>(s0.suf, pre);

        // Sub-block origins, in units of W and relative to the buffer start.
        float o0 = C::origin(b, lane);
        float o1 = C::origin(b+1, lane);

        // R_suf starts as the whole sub-block of block b, and is walked down by
        // subtraction; R_pre starts empty and is walked up. See (*) at the top.
        Raw<NDEG> rsuf, rpre;
        raw_zero<NDEG>(rsuf);
        raw_zero<NDEG>(rpre);
        #pragma unroll
        for (int i = 0; i < L; i++)
            raw_add<NDEG>(rsuf, mask_bit(s0.mb, i), s0.dz[i], C::sb_y(i));

        // The output sample of local index i lives in sub-block (lane^16): in block b for
        // lane < 16, in block b+1 for lane >= 16. Either way the holder is the xor-16
        // partner, so the sender selects which of its two blocks to send by its OWN lane id.
        unsigned int obits = __shfl_xor_sync(ALL_LANES, (lane >= 16) ? s0.mb : s1.mb, 16);
        float dko = (lane >= 16) ? dk : 0.0f;   // output sample needs reframing iff it came from block b+1

        float rbuf[4];
        unsigned int ow[L/4];
        #pragma unroll
        for (int k = 0; k < L/4; k++)
            ow[k] = 0;

        #pragma unroll
        for (int i = 0; i < L; i++) {
            if (i > 0)
                raw_sub<NDEG>(rsuf, mask_bit(s0.mb, i-1), s0.dz[i-1], C::sb_y(i-1));

            float m1 = mask_bit(s1.mb, i);
            raw_add<NDEG>(rpre, m1, s1.dz[i] + dk*m1, C::sb_y(i));

            Canon<NDEG> ms = merge_raw_canon<NDEG>(rpre, o1, merge_raw_canon<NDEG>(rsuf, o0, O));

            // Buffer index of the output sample: window [q, q+B] with q = B*b + L*lane + i,
            // evaluated back at its center q + W. In units of W, to match ms.ct.
            float t = float(B*b + W + L*lane + i) * (1.0f / float(W));
            float x0 = t - ms.ct;

            bool ok;
            float fhat = solve_and_test<NDEG>(ms, x0, ok);

            float mo = mask_bit(obits, i);
            float dzo = __shfl_xor_sync(ALL_LANES, (lane >= 16) ? s0.dz[i] : s1.dz[i], 16);
            dzo += dko * mo;

            bool keep = ok && (mo > 0.0f);

            rbuf[i & 3] = keep ? (dzo - fhat) : 0.0f;
            ow[i >> 2] |= (keep ? 1U : 0U) << (8*(i & 3));

            if ((i & 3) == 3)
                *((float4 *) (dptr + B*b + W + L*lane + i-3)) = make_float4(rbuf[0], rbuf[1], rbuf[2], rbuf[3]);
        }

        if constexpr (L == 8)
            *((uint2 *) (mptr + B*b + W + L*lane)) = make_uint2(ow[0], ow[1]);
        else
            *((uint4 *) (mptr + B*b + W + L*lane)) = make_uint4(ow[0], ow[1], ow[2], ow[3]);

        s0 = s1;
    }
}


// -------------------------------------------------------------------------------------------------
//
// Host code.


// The compiled configurations. To add one: add a row here, and a line to the dispatch in
// launch(). Nothing else needs to change, provided Cfg's static_asserts are satisfied and
// solve_and_test() handles the degree.
struct Detrender1dConfig { long n, W, T; };

static constexpr Detrender1dConfig detrender_1d_configs[] = {
    { 1, 128, 2048 },
    { 2, 256, 2048 },
};


static string config_list_str()
{
    stringstream ss;
    for (const Detrender1dConfig &c: detrender_1d_configs)
        ss << ((&c == &detrender_1d_configs[0]) ? "" : ", ")
           << "(n=" << c.n << ", W=" << c.W << ", T=" << c.T << ")";
    return ss.str();
}


vector<tuple<long,long,long>> Detrender1d::configs()
{
    vector<tuple<long,long,long>> ret;
    for (const Detrender1dConfig &c: detrender_1d_configs)
        ret.push_back({ c.n, c.W, c.T });
    return ret;
}


// The constructor is where an unsupported configuration is rejected, so that a caller
// finds out at construction rather than at launch.
Detrender1d::Detrender1d(long n_, long W_, long T_) :
    n(n_), W(W_), T(T_), nbuf(T_ + 2*W_)
{
    for (const Detrender1dConfig &c: detrender_1d_configs)
        if ((c.n == n_) && (c.W == W_) && (c.T == T_))
            return;

    stringstream ss;
    ss << "Detrender1d: no kernel is compiled for (n=" << n_ << ", W=" << W_
       << ", T=" << T_ << "); available configurations are " << config_list_str();
    throw runtime_error(ss.str());
}


template<int NDEG, int W, int T>
static void _launch(float *data, unsigned char *mask, long M, cudaStream_t stream)
{
    long nblocks = (M + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    dim3 nthreads(32, WARPS_PER_BLOCK);

    detrend_1d_kernel<NDEG,W,T> <<< nblocks, nthreads, 0, stream >>> (data, mask, int(M));
    CUDA_PEEK("detrend_1d_kernel");
}


void Detrender1d::launch(Array<float> &data, Array<unsigned char> &mask, cudaStream_t stream) const
{
    xassert_eq(data.ndim, 2);
    xassert_eq(data.shape[1], nbuf);
    xassert_shape_eq(mask, ({data.shape[0], nbuf}));
    xassert(data.is_fully_contiguous());
    xassert(mask.is_fully_contiguous());
    xassert(data.on_gpu());
    xassert(mask.on_gpu());

    // The kernel loads/stores both arrays with 128-bit instructions, at offsets which are
    // multiples of 16 bytes from the base pointer. A cudaMalloc'ed array is 256-byte
    // aligned, but a python caller can pass a view whose base is not, and the failure mode
    // is a misaligned-address fault deep inside the kernel rather than here.
    xassert((reinterpret_cast<uintptr_t>(data.data) % 16) == 0);
    xassert((reinterpret_cast<uintptr_t>(mask.data) % 16) == 0);

    long M = data.shape[0];
    xassert_gt(M, 0);
    xassert_le(M, 0x7fffffffL);   // kernel takes 'int M'

    if ((n == 1) && (W == 128) && (T == 2048))
        _launch<1,128,2048> (data.data, mask.data, M, stream);
    else if ((n == 2) && (W == 256) && (T == 2048))
        _launch<2,256,2048> (data.data, mask.data, M, stream);
    else
        throw runtime_error("Detrender1d::launch: internal error, unhandled configuration");
}


void Detrender1d::time_selected()
{
    long M = 64*1024;

    for (const Detrender1dConfig &c: detrender_1d_configs) {
        Detrender1d det(c.n, c.W, c.T);

        // Global memory traffic: the whole buffer is read, and the T-sample output window
        // is written. Every byte is touched exactly once, so this is also the DRAM traffic
        // of an ideal implementation, and (time -> bandwidth) is the figure of merit: the
        // kernel is expected to be memory bound.
        double nbytes = double(M) * (double(det.nbuf) + double(det.T)) * 5.0;   // 4 bytes data + 1 byte mask

        // The kernel is branch-free, so the timing does not depend on the data. We use an
        // all-valid mask anyway, so that the "normal" path is what gets timed (a fully
        // masked buffer would take every divide down the guarded branch).
        Array<float> data({M, det.nbuf}, af_gpu | af_zero);
        Array<unsigned char> mask({M, det.nbuf}, af_gpu);
        CUDA_CALL(cudaMemset(mask.data, 1, M*det.nbuf));

        cout << "\nDetrender1d::time_selected()\n"
             << "    (n, W, T) = (" << det.n << ", " << det.W << ", " << det.T << "), M = " << M << "\n"
             << "    data = " << (double(M)*det.nbuf*4 / 1.0e9) << " GB, "
             << "mask = " << (double(M)*det.nbuf / 1.0e9) << " GB\n"
             << "    global memory traffic per launch = " << (nbytes / 1.0e9) << " GB\n"
             << endl;

        int niter = 50;
        int print_interval = 10;
        KernelTimer kt(niter, 1);

        while (kt.next()) {
            det.launch(data, mask, kt.stream);

            if (kt.warmed_up && ((kt.curr_iteration+1) % print_interval == 0)) {
                cout << "    iter " << (kt.curr_iteration+1) << "/" << niter
                     << ": dt = " << (kt.dt * 1.0e3) << " ms"
                     << ", bandwidth = " << (nbytes / kt.dt / 1.0e9) << " GB/s" << endl;
            }
        }
    }
}


}  // namespace pirate
