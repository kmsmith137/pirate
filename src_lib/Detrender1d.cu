#include "../include/pirate/Detrender1d.hpp"

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
// local polynomial subtraction". The moment
// monoid, the van Herk block decomposition, and the per-window solve are all specified
// there; pirate_frb/detrending_1d is a line-by-line numpy implementation of the same
// algorithm, and is the oracle this kernel is tested against.
//
// Geometry (all compile-time). The window is 2W+1 = 513 samples and the scan blocks are
// B = 2W = 512 samples, so a window spans exactly two adjacent blocks for every
// alignment. The buffer is T + 2W = 2560 samples = 5 blocks, and output j (0 <= j < 2048)
// has window [j, j+B] in buffer coordinates:
//
//    M(window of output j) = Suff_b[p] (+) Pref_{b+1}[p],   b = j/B,  p = j%B
//
// where Pref/Suff are inclusive prefix/suffix scans of the monoid within a block.
//
// Thread mapping. One warp per row; lane 'l' owns sub-block 'l' of a block, i.e. the
// L = B/32 = 16 consecutive samples [16l, 16l+16). Writing p = 16l + i,
//
//    Suff_b[p]     = R_suf(i)  (+)  Q_{b,l}          R_suf(i) = local samples i..15 of block b
//    Pref_{b+1}[p] = P_{b+1,l} (+)  R_pre(i)         R_pre(i) = local samples 0..i of block b+1
//
// where Q_{b,l} is the exclusive suffix of block b's 32 sub-block aggregates (sub-blocks
// l+1..31) and P_{b+1,l} the exclusive prefix (sub-blocks 0..l-1). Neither depends on i,
// so their merge O_{b,l} = Q_{b,l} (+) P_{b+1,l} is computed once per (pair, lane) and
//
//    M(window) = R_suf(i) (+) O_{b,l} (+) R_pre(i)                                     (*)
//
// costs two merges per output. Sanity check on the sample counts:
// (16-i) + (512 - 16(l+1)) + 16l + (i+1) = 513 for all l, i.
//
// The payoff is that lane l produces its 16 outputs with no cross-lane traffic except a
// single shuffle for the output sample itself: all lane cooperation is confined to the
// once-per-block scan over sub-block aggregates. There is no shared memory and no
// __syncthreads() anywhere in the kernel.
//
// Three levels of moment accumulation, and why:
//
//   - Within a sub-block, moments are raw running sums about a fixed origin (struct Raw).
//     No centroid, no Pascal shift, 8 FMAs per sample. The span is only 16/W = 1/16 in
//     normalized units, so adaptive centering buys nothing here.
//   - A sub-block aggregate is converted to centroid-carrying form (struct Canon) by a
//     single Pascal shift, and the 32 aggregates are scanned with warp shuffles.
//   - Per output, the two raw partials of (*) are merged into O_{b,l} directly from their
//     own origins, by merge_raw_canon().
//
// It is NOT safe to save a merge by combining the two raw partials about a common origin
// first: they live in blocks 512 samples apart, so the common origin would be ~W away from
// either, and a window whose valid samples are a narrow cluster near one window edge (which
// is well conditioned and survives -- 17 consecutive samples give rmin = 0.44) would have
// its moments computed by cancelling O(N_v) quantities down to O(N_v (17/W)^4). This is the
// same failure the tex describes for scans that do not carry their own centroid.


// Compile-time parameters. B, L and NPAIR are derived; the static_asserts below are
// the assumptions the kernel body actually relies on.
static constexpr int W = Detrender1d::W;         // window half-width
static constexpr int T = Detrender1d::T;         // outputs per row
static constexpr int NBUF = Detrender1d::nbuf;   // buffer samples per row
static constexpr int B = 2*W;                    // scan block length
static constexpr int L = B / 32;                 // samples per lane (one sub-block)
static constexpr int NPAIR = T / B;              // block pairs, i.e. loop trips
static constexpr int NBLK = NPAIR + 1;           // blocks per buffer

static constexpr float EPS = Detrender1d::eps;
static constexpr float MU = Detrender1d::mu;

static constexpr int WARPS_PER_BLOCK = 4;
static constexpr unsigned int ALL_LANES = 0xffffffffU;

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

static_assert(Detrender1d::n == 2);              // the solve below is hardwired to n=2
static_assert(T % B == 0);                       // block lattice is chunk-invariant
static_assert(NBLK * B == NBUF);
static_assert(L == 16);                          // 16 mask bytes = one uint4 load
static_assert((W & (W-1)) == 0);                 // power of two: see sb_y() below


// -------------------------------------------------------------------------------------------------
//
// The moment monoid (notes/tree_dedispersion.tex, "The moment monoid").


// Canon: a monoid element in canonical form, i.e. moments taken about the set's own
// mask-weighted centroid 'c'. S0 = N and S1 = 0 are implicit and not stored, which is
// the 3n+2 = 8 floats of the tex.
//
// 'c' is in buffer-relative sample units, matching MomentSet.py. Absolute stream
// coordinates would destroy float32 precision in a long run.
struct Canon
{
    float N;                    // number of valid samples (= S0)
    float c;                    // mask-weighted centroid
    float S2, S3, S4;           // mask moments about c
    float U0, U1, U2;           // data moments about c
};


// Raw: moments about a FIXED origin 'o' carried by the caller, so R1 is not zero.
// Used for partial sums within a sub-block, where accumulation is a plain running
// sum with compile-time coefficients.
struct Raw
{
    float R0, R1, R2, R3, R4;   // mask moments about o
    float V0, V1, V2;           // data moments about o
};


// sb_y(): the sub-block-local coordinate x = (u - o)/W of local sample i, where the
// origin o is the sub-block center (local index 7.5).
//
// Since W is a power of two, y = (2i-15)/512 is exact, and so is every power up to
// y^4 = (2i-15)^4 / 2^36 (numerator < 2^17). A sum of 16 such terms has numerator
// < 2^21, so ALL the raw mask moments of a sub-block -- and every running prefix and
// suffix of them, including the ones formed by subtraction in the output loop -- are
// computed exactly in float32. Rounding enters the mask moments only at the Pascal
// shift, never in the accumulation.
static __device__ __forceinline__ float sb_y(int i)
{
    return (float(i) - 7.5f) * (1.0f / float(W));
}


static __device__ __forceinline__ void raw_zero(Raw &a)
{
    a.R0 = a.R1 = a.R2 = a.R3 = a.R4 = 0.0f;
    a.V0 = a.V1 = a.V2 = 0.0f;
}


// raw_add() / raw_sub(): accumulate (or remove) one sample. 'm' is the mask (0 or 1),
// 'dz' the masked, offset-subtracted data m*(d-kappa), and 'i' the sub-block-local
// sample index, which is a compile-time constant at every call site (the output loop
// is fully unrolled), so the powers of y fold to immediates.
static __device__ __forceinline__ void raw_add(Raw &a, float m, float dz, int i)
{
    float y = sb_y(i);
    float y2 = y*y, y3 = y2*y, y4 = y2*y2;
    a.R0 += m;    a.R1 += m*y;    a.R2 += m*y2;   a.R3 += m*y3;   a.R4 += m*y4;
    a.V0 += dz;   a.V1 += dz*y;   a.V2 += dz*y2;
}

static __device__ __forceinline__ void raw_sub(Raw &a, float m, float dz, int i)
{
    float y = sb_y(i);
    float y2 = y*y, y3 = y2*y, y4 = y2*y2;
    a.R0 -= m;    a.R1 -= m*y;    a.R2 -= m*y2;   a.R3 -= m*y3;   a.R4 -= m*y4;
    a.V0 -= dz;   a.V1 -= dz*y;   a.V2 -= dz*y2;
}


// merge_canon(): the monoid merge of two canonical elements, MomentSet.merge() verbatim.
//
// The centroid update is written as c + f*Delta*W rather than (N*c + N'*c')/N to avoid
// forming the large product. S1 vanishes by construction of the new centroid and is
// forced to zero rather than computed, which is what keeps G_01 = 0 and hence makes the
// pivots p_0 = G_00 and p_1 = G_11 exact in the solve.
//
// Empty-set rule (the NaN trap): 'c' is undefined when N = 0, and a scan performs many
// merges involving empty aggregates (any fully masked sub-block). Two things keep this
// branch-free and NaN-free: every element carries a FINITE nominal c (an empty aggregate
// or scan identity uses its own sub-block center), and the divide is guarded. Merging
// empty with non-empty then gives f = 1, db = 0, c = b.c, which is correct.
static __device__ __forceinline__ Canon merge_canon(const Canon &a, const Canon &b)
{
    float N = a.N + b.N;
    float inv = safe_rcp(N);
    float Delta = (b.c - a.c) * (1.0f / float(W));
    float f = b.N * inv;

    float da = -f * Delta;              // (a.c - c_new)/W
    float db = Delta + da;              // (b.c - c_new)/W

    float da2 = da*da, da3 = da2*da, da4 = da2*da2;
    float db2 = db*db, db3 = db2*db, db4 = db2*db2;

    Canon r;
    r.N = N;
    r.c = a.c - float(W) * da;

    // Pascal shifts, using S0 = N and S1 = 0 on both sides.
    r.S2 = (a.S2 + da2*a.N) + (b.S2 + db2*b.N);
    r.S3 = (a.S3 + 3.0f*da*a.S2 + da3*a.N) + (b.S3 + 3.0f*db*b.S2 + db3*b.N);
    r.S4 = (a.S4 + 4.0f*da*a.S3 + 6.0f*da2*a.S2 + da4*a.N)
         + (b.S4 + 4.0f*db*b.S3 + 6.0f*db2*b.S2 + db4*b.N);

    r.U0 = a.U0 + b.U0;
    r.U1 = (a.U1 + da*a.U0) + (b.U1 + db*b.U0);
    r.U2 = (a.U2 + 2.0f*da*a.U1 + da2*a.U0) + (b.U2 + 2.0f*db*b.U1 + db2*b.U0);
    return r;
}


// merge_raw_canon(): merge a raw element (moments about the fixed origin 'o') with a
// canonical one. Mathematically this is merge_canon() applied to the canonicalization
// of 'a', but the raw side's Pascal shift is taken directly from o to the merged
// centroid rather than from o to a's own centroid and then to the merged one. That
// saves a shift and a reciprocal, and is strictly more accurate: one origin hop, not two.
//
// The result is symmetric in its operands, so it does not matter whether the raw side is
// the earlier or the later sample range.
static __device__ __forceinline__ Canon merge_raw_canon(const Raw &a, float o, const Canon &b)
{
    float N = a.R0 + b.N;
    float inv = safe_rcp(N);
    float Delta = (b.c - o) * (1.0f / float(W));
    float s = (a.R1 + b.N * Delta) * inv;    // s = (c_new - o)/W

    float da = -s;                      // (o - c_new)/W
    float db = Delta - s;               // (b.c - c_new)/W

    float da2 = da*da, da3 = da2*da, da4 = da2*da2;
    float db2 = db*db, db3 = db2*db, db4 = db2*db2;

    Canon r;
    r.N = N;
    r.c = o + float(W) * s;

    // Raw side: full Pascal shift, since R1 is nonzero in general.
    // Canonical side: S0 = N, S1 = 0, as in merge_canon().
    r.S2 = (a.R2 + 2.0f*da*a.R1 + da2*a.R0)
         + (b.S2 + db2*b.N);
    r.S3 = (a.R3 + 3.0f*da*a.R2 + 3.0f*da2*a.R1 + da3*a.R0)
         + (b.S3 + 3.0f*db*b.S2 + db3*b.N);
    r.S4 = (a.R4 + 4.0f*da*a.R3 + 6.0f*da2*a.R2 + 4.0f*da3*a.R1 + da4*a.R0)
         + (b.S4 + 4.0f*db*b.S3 + 6.0f*db2*b.S2 + db4*b.N);

    r.U0 = a.V0 + b.U0;
    r.U1 = (a.V1 + da*a.V0) + (b.U1 + db*b.U0);
    r.U2 = (a.V2 + 2.0f*da*a.V1 + da2*a.V0) + (b.U2 + 2.0f*db*b.U1 + db2*b.U0);
    return r;
}


// canonicalize(): convert a raw sub-block total (about origin o) to canonical form, by
// shifting the origin to the set's centroid c = o + W*(R1/R0). The origin travel is at
// most 8 samples = W/32, so the cancellation in the shift is bounded by (1 + 1/32)^4.
//
// An empty sub-block (R0 = 0) gets s = 0, hence c = o: a finite nominal centroid, as the
// empty-set rule of merge_canon() requires.
static __device__ __forceinline__ Canon canonicalize(const Raw &a, float o)
{
    float s = a.R1 * safe_rcp(a.R0);
    float d = -s;
    float d2 = d*d, d3 = d2*d, d4 = d2*d2;

    Canon r;
    r.N = a.R0;
    r.c = o + float(W) * s;
    r.S2 = a.R2 + 2.0f*d*a.R1 + d2*a.R0;
    r.S3 = a.R3 + 3.0f*d*a.R2 + 3.0f*d2*a.R1 + d3*a.R0;
    r.S4 = a.R4 + 4.0f*d*a.R3 + 6.0f*d2*a.R2 + 4.0f*d3*a.R1 + d4*a.R0;
    r.U0 = a.V0;
    r.U1 = a.V1 + d*a.V0;
    r.U2 = a.V2 + 2.0f*d*a.V1 + d2*a.V0;
    return r;
}


// -------------------------------------------------------------------------------------------------
//
// Warp-level scans over the 32 sub-block aggregates of one block.


static __device__ __forceinline__ Canon canon_shfl(const Canon &x, int src_lane)
{
    Canon r;
    r.N  = __shfl_sync(ALL_LANES, x.N,  src_lane);
    r.c  = __shfl_sync(ALL_LANES, x.c,  src_lane);
    r.S2 = __shfl_sync(ALL_LANES, x.S2, src_lane);
    r.S3 = __shfl_sync(ALL_LANES, x.S3, src_lane);
    r.S4 = __shfl_sync(ALL_LANES, x.S4, src_lane);
    r.U0 = __shfl_sync(ALL_LANES, x.U0, src_lane);
    r.U1 = __shfl_sync(ALL_LANES, x.U1, src_lane);
    r.U2 = __shfl_sync(ALL_LANES, x.U2, src_lane);
    return r;
}


// canon_identity(): the monoid identity, with a finite nominal centroid (see the
// empty-set rule in merge_canon()).
static __device__ __forceinline__ Canon canon_identity(float c_nominal)
{
    Canon r;
    r.N = 0.0f;
    r.c = c_nominal;
    r.S2 = r.S3 = r.S4 = 0.0f;
    r.U0 = r.U1 = r.U2 = 0.0f;
    return r;
}


// warp_scans(): given one sub-block aggregate per lane (in sub-block order), return the
// EXCLUSIVE prefix and suffix scans over the 32 lanes.
//
// Hillis-Steele, 5 steps each. The reason the scan must be a tree at all is roundoff:
// each output then depends on its leaves through O(log 32) merges rather than O(32).
// Every step merges two disjoint, adjacent ranges, which is what merge_canon() requires.
static __device__ __forceinline__ void warp_scans(const Canon &agg, int lane, float o,
                                                  Canon &excl_pre, Canon &excl_suf)
{
    Canon pre = agg;
    Canon suf = agg;

    #pragma unroll
    for (int k = 1; k < 32; k *= 2) {
        Canon up = canon_shfl(pre, lane - k);
        if (lane >= k)
            pre = merge_canon(up, pre);

        Canon dn = canon_shfl(suf, lane + k);
        if (lane + k < 32)
            suf = merge_canon(suf, dn);
    }

    // Inclusive -> exclusive. The identity's nominal centroid is this lane's own
    // sub-block center, which is finite and otherwise arbitrary.
    Canon id = canon_identity(o);
    excl_pre = canon_shfl(pre, lane - 1);
    excl_suf = canon_shfl(suf, lane + 1);
    if (lane == 0)
        excl_pre = id;
    if (lane == 31)
        excl_suf = id;
}


// -------------------------------------------------------------------------------------------------
//
// The kernel.


// One block of the row, as held by one lane: L consecutive samples plus the block's
// constant offset and the scan results.
//
// The offset kappa is per BLOCK, not per buffer as in Detrender.detrend_chunk(). Any
// value is mathematically exact -- the polynomial basis contains the constant function,
// so replacing d by d-kappa shifts fhat by exactly kappa and leaves the residual
// unchanged -- and the only requirement is |d - kappa| <~ 1e3 sigma during accumulation.
// A per-block kappa is if anything more local than a per-buffer one, and it lets the
// kernel read each row exactly once: a per-buffer kappa would need a second pass over
// the row, since 2560 samples do not fit in a warp's registers.
//
// The price is that the two blocks of a pair are in different offset frames and must be
// reconciled before their moments are merged. The conversion is exact and cheap: for
// moments about any common origin, U_r(kappa) = U_r(kappa') + (kappa' - kappa) S_r, and
// S_0 = N, S_1 = 0.
struct BlockState
{
    float dz[L];       // masked, offset-subtracted data m*(d - kappa)
    unsigned int mb;   // mask bits: bit i is sample i
    float kappa;
    float nv;          // valid samples in the whole block (all 32 lanes), i.e. is kappa meaningful
    Canon pre;         // exclusive prefix of this block's sub-block aggregates
    Canon suf;         // exclusive suffix
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
// The mask load is one uint4 per lane: 16 bytes each, 512 contiguous bytes per warp, i.e.
// 4 whole cache lines per instruction. The data load is 4 float4's per lane covering 16
// consecutive floats, so a single instruction touches 16 cache lines and uses 32 of the
// 128 bytes of each; the four together cover them exactly once, so DRAM traffic is
// unaffected and only the L1 wavefront count goes up (measured at ~2% of the kernel).
// Making each instruction cache-line-perfect would need a coalesced load plus a warp
// transpose, which is not worth the complexity here.
static __device__ __forceinline__ void load_block(BlockState &bs, const float *dptr,
                                                  const unsigned char *mptr, int beta, int lane)
{
    long off = long(B)*beta + long(L)*lane;

    float4 q0 = *((const float4 *) (dptr + off));
    float4 q1 = *((const float4 *) (dptr + off + 4));
    float4 q2 = *((const float4 *) (dptr + off + 8));
    float4 q3 = *((const float4 *) (dptr + off + 12));
    float d[L] = { q0.x, q0.y, q0.z, q0.w,  q1.x, q1.y, q1.z, q1.w,
                   q2.x, q2.y, q2.z, q2.w,  q3.x, q3.y, q3.z, q3.w };

    uint4 mv = *((const uint4 *) (mptr + off));
    unsigned int mw[4] = { mv.x, mv.y, mv.z, mv.w };

    unsigned int mb = 0;
    #pragma unroll
    for (int i = 0; i < L; i++)
        mb |= ((mw[i >> 2] >> (8*(i & 3))) & 0xffU) ? (1U << i) : 0U;
    bs.mb = mb;

    // Constant offset: the masked mean of this block. Precision is irrelevant here (any
    // value is exact), so a plain warp reduction over the 512 samples is fine.
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

    Raw tot;
    raw_zero(tot);
    #pragma unroll
    for (int i = 0; i < L; i++) {
        float m = mask_bit(mb, i);
        float dz = m * (d[i] - kappa);
        bs.dz[i] = dz;
        raw_add(tot, m, dz, i);
    }

    // Origin of this lane's sub-block, in buffer-relative sample units.
    float o = float(B*beta + L*lane) + 7.5f;
    warp_scans(canonicalize(tot, o), lane, o, bs.pre, bs.suf);
}


// detrend_1d_kernel(): one warp per row.
//
// __launch_bounds__ carries no min-blocks-per-SM hint on purpose. The kernel uses ~168
// registers (mostly the 2 x 16 samples of block b and b+1 that each lane holds), which is
// 12 warps/SM on an L40S. Asking for more occupancy makes it worse, not better: at 128
// registers it spills 192 bytes and runs at 533 GB/s, at 96 registers 392 bytes and
// 334 GB/s, against 688 GB/s here. There is nothing to gain anyway -- see time_selected().
//
// In-place overwrite is safe, and the ordering requirement is load-bearing: pair b writes
// buffer samples [B*b + W, B*b + W + B), which is the second half of block b and the first
// half of block b+1. Both blocks are already resident in warp registers when the writes
// happen, and neither is re-read from global memory afterwards (block b is discarded, and
// block b+1 rotates into the register slot for pair b+1). So block b+1 must be LOADED
// BEFORE pair b executes, which is what the loop below does. Do not reorder it.
__global__ void __launch_bounds__(32*WARPS_PER_BLOCK)
detrend_1d_kernel(float *data, unsigned char *mask, int M)
{
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
    float *dptr = data + row * long(NBUF);
    unsigned char *mptr = mask + row * long(NBUF);

    BlockState s0, s1;
    load_block(s0, dptr, mptr, 0, lane);

    for (int b = 0; b < NPAIR; b++) {
        load_block(s1, dptr, mptr, b+1, lane);

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
        // quantities (m, N, S2), all of which are zero.
        float dk = (s0.nv > 0.0f) ? (s1.kappa - s0.kappa) : 0.0f;

        Canon pre = s1.pre;
        pre.U0 += dk * pre.N;
        pre.U2 += dk * pre.S2;      // U1 is unchanged, since S1 = 0
        Canon O = merge_canon(s0.suf, pre);

        // Sub-block origins, in buffer-relative sample units.
        float o0 = float(B*b + L*lane) + 7.5f;
        float o1 = o0 + float(B);

        // R_suf starts as the whole sub-block of block b, and is walked down by
        // subtraction; R_pre starts empty and is walked up. See (*) at the top.
        Raw rsuf, rpre;
        raw_zero(rsuf);
        raw_zero(rpre);
        #pragma unroll
        for (int i = 0; i < L; i++)
            raw_add(rsuf, mask_bit(s0.mb, i), s0.dz[i], i);

        // The output sample of local index i lives in sub-block (lane^16): in block b for
        // lane < 16, in block b+1 for lane >= 16. Either way the holder is the xor-16
        // partner, so the sender selects which of its two blocks to send by its OWN lane id.
        unsigned int obits = __shfl_xor_sync(ALL_LANES, (lane >= 16) ? s0.mb : s1.mb, 16);
        float dko = (lane >= 16) ? dk : 0.0f;   // output sample needs reframing iff it came from block b+1

        float rbuf[4];
        unsigned int ow[4] = { 0, 0, 0, 0 };

        #pragma unroll
        for (int i = 0; i < L; i++) {
            if (i > 0)
                raw_sub(rsuf, mask_bit(s0.mb, i-1), s0.dz[i-1], i-1);

            float m1 = mask_bit(s1.mb, i);
            raw_add(rpre, m1, s1.dz[i] + dk*m1, i);

            Canon ms = merge_raw_canon(rpre, o1, merge_raw_canon(rsuf, o0, O));

            // Buffer index of the output sample: window [q, q+B] with q = B*b + 16*lane + i,
            // evaluated back at its center q + W.
            float t = float(B*b + W + L*lane + i);
            float x0 = (t - ms.c) * (1.0f / float(W));

            // Solve G a = U and evaluate at x0, where G is the Hankel matrix of the mask
            // moments with G_01 = 0:
            //
            //     G = [ N   0   A  ]                  fhat = w^T G^{-1} U,  w = (1, x0, x0^2)
            //         [ 0   A   Bm ]
            //         [ A   Bm  C  ]
            //
            // For n=2 the LDL^T pivots are p0 = N, p1 = A, p2 = C - A^2/N - Bm^2/A, so
            // p_i/G_ii is 1 for i = 0,1 and the conditioning statistic collapses to
            //
            //     rmin = p2/C = det(G) / (N*A*C)      (zero if any of N, A, C is <= 0)
            //
            // Two consequences, both used here: the masking test rmin >= eps becomes the
            // division-free comparison det >= eps*N*A*C, and evaluating the fit through the
            // adjugate needs one reciprocal and no square roots. This is algebraically
            // identical to LocalPolyFit.cholesky/solve, so it inherits the same error
            // bound: near the threshold det is a cancelling difference with relative error
            // ~eps_mach/eps, and the absolute error in rmin itself is O(eps_mach).
            float N = ms.N, A = ms.S2, Bm = ms.S3, C = ms.S4;

            float adj00 = A*C - Bm*Bm;
            float adj01 = A*Bm;
            float adj02 = -A*A;
            float adj11 = N*C - A*A;
            float adj12 = -N*Bm;
            float adj22 = N*A;
            float det = N*adj00 + A*adj02;

            float v0 = adj00*ms.U0 + adj01*ms.U1 + adj02*ms.U2;
            float v1 = adj01*ms.U0 + adj11*ms.U1 + adj12*ms.U2;
            float v2 = adj02*ms.U0 + adj12*ms.U1 + adj22*ms.U2;

            // MU is a NaN guard, not a tuning parameter: it is inert wherever the sample
            // is kept, and on a sample we drop, fhat is discarded by the select below.
            float fhat = __fdiv_rn(v0 + x0*(v1 + x0*v2), fmaxf(det, MU));

            float mo = mask_bit(obits, i);
            float dzo = __shfl_xor_sync(ALL_LANES, (lane >= 16) ? s0.dz[i] : s1.dz[i], 16);
            dzo += dko * mo;

            bool keep = (mo > 0.0f) && (N > 0.0f) && (A > 0.0f) && (C > 0.0f)
                        && (det >= EPS*N*A*C);

            rbuf[i & 3] = keep ? (dzo - fhat) : 0.0f;
            ow[i >> 2] |= (keep ? 1U : 0U) << (8*(i & 3));

            if ((i & 3) == 3)
                *((float4 *) (dptr + B*b + W + L*lane + i-3)) = make_float4(rbuf[0], rbuf[1], rbuf[2], rbuf[3]);
        }

        *((uint4 *) (mptr + B*b + W + L*lane)) = make_uint4(ow[0], ow[1], ow[2], ow[3]);

        s0 = s1;
    }
}


// -------------------------------------------------------------------------------------------------


void Detrender1d::launch(Array<float> &data, Array<unsigned char> &mask, cudaStream_t stream)
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

    long nblocks = (M + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    dim3 nthreads(32, WARPS_PER_BLOCK);

    detrend_1d_kernel <<< nblocks, nthreads, 0, stream >>> (data.data, mask.data, int(M));
    CUDA_PEEK("detrend_1d_kernel");
}


void Detrender1d::time_selected()
{
    long M = 64*1024;

    // Global memory traffic: the whole buffer is read, and the T-sample output window is
    // written. Every byte is touched exactly once, so this is also the DRAM traffic of an
    // ideal implementation, and (time -> bandwidth) is the figure of merit: the kernel is
    // expected to be memory bound.
    double nbytes = double(M) * (double(nbuf) + double(T)) * 5.0;   // 4 bytes data + 1 byte mask

    // The kernel is branch-free, so the timing does not depend on the data. We use an
    // all-valid mask anyway, so that the "normal" path is what gets timed (a fully masked
    // buffer would take every divide down the guarded branch).
    Array<float> data({M, nbuf}, af_gpu | af_zero);
    Array<unsigned char> mask({M, nbuf}, af_gpu);
    CUDA_CALL(cudaMemset(mask.data, 1, M*nbuf));

    cout << "\nDetrender1d::time_selected()\n"
         << "    (n, W, T) = (" << n << ", " << W << ", " << T << "), M = " << M << "\n"
         << "    data = " << (double(M)*nbuf*4 / 1.0e9) << " GB, "
         << "mask = " << (double(M)*nbuf / 1.0e9) << " GB\n"
         << "    global memory traffic per launch = " << (nbytes / 1.0e9) << " GB\n"
         << endl;

    int niter = 50;
    int print_interval = 10;
    KernelTimer kt(niter, 1);

    while (kt.next()) {
        Detrender1d::launch(data, mask, kt.stream);

        if (kt.warmed_up && ((kt.curr_iteration+1) % print_interval == 0)) {
            cout << "    iter " << (kt.curr_iteration+1) << "/" << niter
                 << ": dt = " << (kt.dt * 1.0e3) << " ms"
                 << ", bandwidth = " << (nbytes / kt.dt / 1.0e9) << " GB/s" << endl;
        }
    }
}


}  // namespace pirate
