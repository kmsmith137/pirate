#include "../include/pirate/Detrender2d.hpp"

#include <cmath>
#include <sstream>
#include <iostream>
#include <algorithm>
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
// Notation follows notes/tree_dedispersion.tex, section "2-d detrending"; the numpy
// reference is pirate_frb/detrending_spline, and the two are compared by
// test_gpu_kernel() in that package.
//
// Three kernels, in stream order. The split is forced by the shape of the problem: the
// fit at one time sample couples every channel of a zone, so the full-resolution data
// must be reduced to sufficient statistics, solved, and then swept again. There is no
// single-pass formulation.
//
//   1. accumulate: (data, mask) -> (G, U), the per-freq-range sufficient statistics
//      G_jl[t] = sum_f m[f,t] phi_j(f) phi_l(f),   U_j[t] = sum_f m[f,t] phi_j(f) d[f,t].
//   2. solve: window-average (G,U) over 2W+1 samples, assemble the regularized normal
//      equations, factor, solve, and commit the baseline coefficients at the window
//      centre. One thread per (beam, zone, output sample).
//   3. subtract: evaluate the baseline at full resolution, subtract, expand the mask.
//
// WHERE THE TIME GOES. Kernels 1 and 3 move 15 bytes of DRAM traffic per (beam, channel,
// output sample) and do 9 and 3 FMA respectively, i.e. 3.6 and 0.6 flop/byte against an
// L40S balance point of 106. They are pure bandwidth with a 30x margin, so the only
// things that matter in them are full cache lines and enough warps in flight. Kernel 2
// does ~2000 FMA per (beam, zone, output sample) but there are only M*nzone*T of those,
// which is ~1% of the chunk; it is written for clarity, not speed.
//
// FREQ-RANGES. A "freq-range" is a contiguous run of channels inside a single non-empty
// knot interval, so every channel in it shares a span index j0 and exactly n_phi+1 basis
// functions are nonzero on it. Kernel 1 therefore accumulates a fixed
// (n_phi+1)x(n_phi+1) symmetric block plus a length-(n_phi+1) vector -- 9 floats at
// n_phi=2 -- per (beam, freq-range, buffer sample), and kernel 2 sums the freq-ranges of
// a zone. The partition is a host-side function of (nfreq, knots, CHANNELS_PER_RANGE)
// and of nothing else; see "chunk invariance" below. Its width is an occupancy knob: it
// sets the block count in kernels 1 and 3.
//
// CHUNK INVARIANCE. Output sample t is computed by an identical sequence of floating
// point operations whatever T is, so results are bit-identical across chunkings (given
// consistent padding, which is the caller's contract) and across runs. This is a
// correctness property inherited from the reference, and it is not free: it requires
// that the freq-range partition not depend on T, M or the launch geometry, that the
// per-thread accumulation blocking be a compile-time constant, and that there be NO
// atomics anywhere in the reduction. Reducing (G,U) with atomicAdd would be slightly
// simpler and about as fast, and would give this up; do not do it without revisiting
// test_gpu_kernel()'s bit-identity assertions, which are the strongest structural tests
// we have.
//
// NAN SAFETY. Masked samples are allowed to hold anything, including NaN from a dropped
// packet. Every use of the data is a SELECT on the mask, never a multiply, because
// 0*nan = nan. This is checked by bit-identity under poisoning.


// COMPILE-TIME PARAMETERS, as template arguments throughout:
//
//   NPHI       spline degree in frequency          (n_phi in the reference and the tex)
//   n_deg  degree of the local time polynomial (n)
//
// These two must be compile-time: they size register arrays (acc[NCOMP], a[NPHI+1],
// Mc[NPAIR_T], Vc[n_deg+1]) and drive the #pragma unroll in kernel 1's inner loop.
//
// T (chunk length) and W (window half-width) are RUNTIME kernel arguments. Neither sizes
// a register array: T appears only as a stride and a bound, and W only as a loop bound
// and an offset into shared memory, once kernel 2's parity fold is written with k on the
// outside (see there). W is bounded by MAX_W only because the stencil struct is passed by
// value; T must be a positive multiple of 32.

static constexpr int PASS_THREADS = 256;

// Largest supported window half-width. It bounds nothing in the algorithm -- it exists
// only to give the by-value stencil struct a compile-time size.
static constexpr int MAX_W = 16;

// Largest supported time-polynomial degree, and the pair count it implies. Like MAX_W
// these bound nothing in the algorithm: they give the by-value stencil struct and kernel
// 2's moment registers a compile-time size.
static constexpr int MAX_NDEG = 3;
static constexpr int MAX_NPAIR_T = (MAX_NDEG+1)*(MAX_NDEG+2)/2;

// Target width of a freq-range, in channels. Occupancy knob, and NOT a correctness
// parameter -- but it must stay a compile-time constant, because the freq-range
// partition it induces is part of the summation order (see "chunk invariance" above).
static constexpr long CHANNELS_PER_RANGE = 512;

// The Makefile compiles with --use_fast_math, which turns '/' into an unrefined
// MUFU.RCP and 'sqrtf' into MUFU.RSQ, both ~2^-22 relative. That is twice float32
// machine epsilon, and the conditioning design is budgeted at eps_mach/(4 r_min), so
// kernel 2's factorization uses the correctly-rounded intrinsics instead. One
// reciprocal per row followed by multiplies costs ~1 ulp against 0.5 for a true divide,
// and is far cheaper than n_b divides. Kernels 1 and 3 are pure FMA and are unaffected.
static __device__ __forceinline__ float rn_rsqrt(float x)
{
    return __frcp_rn(__fsqrt_rn(x));
}


// Half-bandwidth of the assembled matrix, in the coefficient-major index order
// I = j*(n+1)+q. The max() is not cosmetic: the assembled matrix carries bands from two
// sources with different bandwidths in j -- the data block (n_phi) and the regulator
// (1, whatever n_phi is, since D_1 is a difference penalty on coefficient indices and
// knows nothing about the spline degree). At n_phi = 0 the regulator is the wider of
// the two and a formula using n_phi alone would under-allocate.
static __host__ __device__ constexpr int bandwidth(int n_phi, int n)
{
    return ((n_phi > 1) ? n_phi : 1) * (n + 1) + n;
}

// Packed index of the unordered pair (a,b) over 0..deg, enumerated (0,0),(0,1),...
static __host__ __device__ constexpr int pair_index(int deg, int a, int b)
{
    int lo = (a < b) ? a : b;
    int hi = (a < b) ? b : a;
    return lo*(deg+1) - (lo*(lo-1))/2 + (hi - lo);
}


// The time-axis stencils. Everything about the polynomial basis {p_q} is compile-time
// data (it depends only on n and W), so it is computed once on the host and passed to
// the kernel BY VALUE: cuda puts kernel parameters in the constant bank, so this is a
// __constant__ broadcast without the cudaMemcpyToSymbol dance.
//
// Folded by parity. On a symmetric window the orthonormal basis polynomials have
// definite parity, so every stencil is even or odd in s and the window is summed in half
// the multiplies. That also makes the even/odd structure EXACT rather than a
// cancellation of rounding errors, which matters because "n=1 reduces to n=0 for a
// window-constant mask" depends on the odd moments vanishing identically.
struct TimeStencils
{
    float gs[MAX_NPAIR_T][MAX_W+1];   // gs[p][k] = p_q(k) p_r(k), the s >= 0 half
    float us[MAX_NDEG+1][MAX_W+1];    // us[q][k] = p_q(k)
    float gpar[MAX_NPAIR_T];          // +1 if even in s, -1 if odd
    float upar[MAX_NDEG+1];
    float eval0[MAX_NDEG+1];          // p_q(0): the contraction that commits the baseline
};


// -------------------------------------------------------------------------------------------------
//
// Kernel 1: accumulate.
//
// gridDim = (ntile, nfrange, M), blockDim = PASS_THREADS. Thread i of tile b owns ONE
// buffer sample t = b*PASS_THREADS + i and loops over its freq-range's channels. A warp
// therefore reads 128 contiguous bytes of 'data' (one full cache line) and 32 contiguous
// bytes of 'mask' (one 32-byte sector), so nothing is wasted at DRAM granularity.
//
// The basis tables are broadcast loads -- every thread in the block reads the same
// address -- so the ~1 MB of tables is served from L1/L2, not DRAM. They are read as
// scalars rather than 128-bit vectors: at 9 broadcast loads per 9 FMA the LSU ceiling is
// tens of TB/s, i.e. 30x above DRAM, so vectorizing them would buy nothing. The tables
// are padded to a multiple of 4 floats per channel anyway, so a future 128-bit load
// needs no change to the host side.

template<int NPHI>
__global__ void __launch_bounds__(PASS_THREADS)
detrend_2d_accum_kernel(const float *data, const unsigned char *mask, float *gu,
                        const float *phi_tab, const float *prod_tab, const int *fr_desc,
                        int nfreq, int nfrange, int nbuf,
                        int phi_stride, int prod_stride)
{
    constexpr int NPAIR_F = (NPHI+1)*(NPHI+2)/2;
    constexpr int NCOMP   = NPAIR_F + (NPHI+1);

    const int t = blockIdx.x*PASS_THREADS + threadIdx.x;
    const int fr = blockIdx.y;
    const int m = blockIdx.z;

    // The buffer length need not be a multiple of PASS_THREADS (nbuf = T + 2W), so the
    // last tile is partially predicated off. There is no __syncthreads() in this kernel,
    // so an early return is safe.
    if (t >= nbuf)
        return;

    const int c_lo = fr_desc[4*fr + 0];
    const int c_hi = fr_desc[4*fr + 1];

    // Apply per-block and per-thread offsets to the full-resolution arrays.
    //   before: shape (M, nfreq, nbuf), contiguous
    //   after:  shape (nfreq,), stride nbuf  -- one element per channel
    data += (long(m)*nfreq)*nbuf + t;
    mask += (long(m)*nfreq)*nbuf + t;

    float acc[NCOMP];
    #pragma unroll
    for (int c = 0; c < NCOMP; c++)
        acc[c] = 0.0f;

    // Two-level accumulation, and this is accuracy rather than an optimization. The
    // reference reduces over frequency with an explicit binary tree because a zone can be
    // thousands of channels wide; a flat float32 sum over a 512-channel freq-range carries
    // ~23 eps of rounding, while the 32-then-16 split below brings it to ~10, within 3x of
    // a full tree, for 9 extra registers and 3% more adds.
    for (int f0 = c_lo; f0 < c_hi; f0 += 32) {
        const int f1 = min(f0 + 32, c_hi);

        float blk[NCOMP];
        #pragma unroll
        for (int c = 0; c < NCOMP; c++)
            blk[c] = 0.0f;

        #pragma unroll 4
        for (int f = f0; f < f1; f++) {
            const float dv = data[long(f)*nbuf];
            const bool mv = (mask[long(f)*nbuf] != 0);

            // SELECT, never multiply: dv may be NaN where mv is false.
            const float w  = mv ? 1.0f : 0.0f;
            const float wd = mv ? dv : 0.0f;

            #pragma unroll
            for (int p = 0; p < NPAIR_F; p++)
                blk[p] += w * __ldg(prod_tab + long(f)*prod_stride + p);
            #pragma unroll
            for (int a = 0; a <= NPHI; a++)
                blk[NPAIR_F+a] += wd * __ldg(phi_tab + long(f)*phi_stride + a);
        }

        #pragma unroll
        for (int c = 0; c < NCOMP; c++)
            acc[c] += blk[c];
    }

    // Layout (M, nfrange, NCOMP, nbuf), time fastest: each component is a coalesced
    // 32-bit store, one full cache line per warp.
    gu += ((long(m)*nfrange + fr)*NCOMP)*nbuf + t;
    #pragma unroll
    for (int c = 0; c < NCOMP; c++)
        gu[long(c)*nbuf] = acc[c];
}


// -------------------------------------------------------------------------------------------------
//
// Kernel 2: assemble and solve.
//
// gridDim = (T/S, nzone, M), blockDim = S threads. ONE THREAD PER (beam, zone, output
// sample): block (b,z,m) owns output samples [b*S, (b+1)*S) of zone z, hence buffer
// samples [b*S, (b+1)*S + 2W).
//
// WHY ONE THREAD PER SOLVE, WITH SHARED MEMORY. The per-solve working set is the banded
// matrix (N_blk, n_b+1) plus the right-hand side and the equilibration scales, 11*N_blk
// floats = 792 bytes at n_phi = n = 2. That is past registers, and registers are not an
// option anyway: N_blk is a runtime quantity (arbitrary knots, and zones within one
// launch may differ), so the loops cannot be unrolled and a local array would spill.
// Shared memory it is, with the THREAD INDEX FASTEST in every array, so that all lanes
// of a warp touch distinct banks on every access -- conflict-free with no padding.
//
// S is chosen by the host from the shared-memory budget and must divide T. The shared
// layout is sized by the LARGEST zone so that every block agrees on the offsets.
//
// Stage 1 reduces the zone's freq-ranges into shared Z, cooperatively. Doing it once per
// block rather than once per thread is worth 8x in global loads: each thread would
// otherwise read nfr*9*(2W+1) floats of its own.

template<int NPHI>
__global__ void __launch_bounds__(256)
detrend_2d_solve_kernel(const float *gu, float *acoef, float *rmin_out,
                        const int *zone_desc, const int *fr_desc,
                        int nfrange, int nzone, int N_phi, int nbuf, int nphi_zone_max,
                        int n_deg, int W, int T, float eta, float eps, TimeStencils tb)
{
    constexpr int NPAIR_F = (NPHI+1)*(NPHI+2)/2;
    constexpr int NCOMP   = NPAIR_F + (NPHI+1);
    const int NPAIR_T = (n_deg+1)*(n_deg+2)/2;
    const int NB      = bandwidth(NPHI, n_deg);
    const int NQ      = n_deg + 1;

    const int S = blockDim.x;
    const int tid = threadIdx.x;
    const int b = blockIdx.x;
    const int z = blockIdx.y;
    const int m = blockIdx.z;

    const int fr_lo   = zone_desc[4*z + 0];
    const int fr_hi   = zone_desc[4*z + 1];
    const int coef_lo = zone_desc[4*z + 2];
    const int nphi_z  = zone_desc[4*z + 3];

    const int nblk   = nphi_z * NQ;             // dimension of this zone's linear system
    const int ncompz = nphi_z * (NPHI+2);       // banded G (n_phi+1 bands) plus U, per coefficient
    const int nloc   = S + 2*W;                 // buffer samples this block needs

    // Shared layout. Offsets use the LARGEST zone, not this block's, so that all blocks
    // in the grid agree; a block whose zone is smaller simply under-uses its slice.
    extern __shared__ float sh[];
    const int nblk_max   = nphi_zone_max * NQ;
    const int ncompz_max = nphi_zone_max * (NPHI+2);
    float *A     = sh;                                  // A[(I*(NB+1)+B)*S + tid]
    float *u     = A + long(nblk_max)*(NB+1)*S;         // u[I*S + tid]
    float *rs    = u + long(nblk_max)*S;                // rs[I*S + tid]
    float *Z     = rs + long(nblk_max)*S;               // Z[e*nloc + i]
    float *Zlive = Z + long(ncompz_max)*nloc;           // Zlive[i]

    // ---- Stage 1: reduce this zone's freq-ranges into Z, one buffer sample per thread.
    //
    // Thread i owns buffer sample b*S+i EXCLUSIVELY, so the scatter-add needs no atomics
    // and no conflict resolution. nloc slightly exceeds S, so the loop is strided and the
    // first 2W threads take a second sample.
    for (int i = tid; i < nloc; i += S) {
        const int s = b*S + i;

        for (int e = 0; e < ncompz; e++)
            Z[e*nloc + i] = 0.0f;

        for (int fr = fr_lo; fr < fr_hi; fr++) {
            const int j0 = fr_desc[4*fr + 2];
            const int jb = j0 - NPHI - coef_lo;          // zone-local base coefficient
            const float *g = gu + ((long(m)*nfrange + fr)*NCOMP)*nbuf + s;

            // Enumerate the frequency pairs in the same order kernel 1 stored them.
            int p = 0;
            for (int a = 0; a <= NPHI; a++)
                for (int c = a; c <= NPHI; c++, p++)
                    Z[((jb+a)*(NPHI+1) + (c-a))*nloc + i] += g[long(p)*nbuf];
            for (int a = 0; a <= NPHI; a++)
                Z[(nphi_z*(NPHI+1) + jb + a)*nloc + i] += g[long(NPAIR_F+a)*nbuf];
        }

        // The rank test's per-offset bit. sum_j G_jj > 0 iff this zone holds at least one
        // unmasked channel at this buffer sample: one unmasked channel contributes at
        // least sum_j phi_j(f)^2 >= 1/(n_phi+1) to the diagonal sum, so there is no
        // underflow-to-zero hazard in float32.
        float live = 0.0f;
        for (int j = 0; j < nphi_z; j++)
            live += Z[(j*(NPHI+1))*nloc + i];
        Zlive[i] = live;
    }

    __syncthreads();

    // ---- Stage 2: one solve per thread, for output sample t = b*S + tid.
    const int t = b*S + tid;

    for (int k = 0; k < nblk*(NB+1); k++)
        A[k*S + tid] = 0.0f;

    // Window stencil + assembly, fused: the moment arrays M and V are never materialized.
    for (int j = 0; j < nphi_z; j++) {

        // --- the G entries of coefficient j: bands 0..n_phi, i.e. G_{j,j+bd}.
        for (int bd = 0; bd <= NPHI; bd++) {
            if (j + bd >= nphi_z)
                continue;                       // no coupling across the zone boundary

            const float *zz = Z + long((j*(NPHI+1) + bd))*nloc + tid;

            // Fold the window by parity, with k on the OUTSIDE.  The natural order is k
            // inner, with the folded halves precomputed into ev[W]/od[W] -- but those are
            // register arrays sized by W, and they are the only thing in the whole kernel
            // that would force W to be a compile-time constant.  This form needs two
            // scalars per k instead, is register-cheaper, and accumulates each Mc[p] over
            // k in exactly the same order, so it is bit-identical.
            float Mc[MAX_NPAIR_T];
            #pragma unroll
            for (int p = 0; p < NPAIR_T; p++)
                Mc[p] = tb.gs[p][0] * zz[W];

            for (int k = 1; k <= W; k++) {
                const float e = zz[W+k] + zz[W-k];
                const float o = zz[W+k] - zz[W-k];
                #pragma unroll
                for (int p = 0; p < NPAIR_T; p++)
                    Mc[p] += tb.gs[p][k] * ((tb.gpar[p] > 0.0f) ? e : o);
            }

            // Scatter into the banded matrix. Coefficient-major indexing I = j*(n+1)+q
            // is load-bearing: it is what makes the assembled matrix banded with
            // half-bandwidth NB. The other natural order, I = q*N_phi+j, has
            // half-bandwidth n*N_phi + n_phi, which grows with N_phi and is effectively
            // dense -- an O(N nb^2) factorization would silently become O(N^3).
            #pragma unroll
            for (int q = 0; q <= n_deg; q++) {
                #pragma unroll
                for (int r = 0; r <= n_deg; r++) {
                    if ((bd == 0) && (r < q))
                        continue;               // held by the (r,q) entry instead
                    const int I = j*NQ + q;
                    const int B = bd*NQ + (r - q);
                    A[(I*(NB+1) + B)*S + tid] += Mc[pair_index(n_deg, q, r)];
                }
            }
        }

        // --- the regulator, eta * D_1 kron Theta.
        //
        // D_1 is the per-zone first-difference penalty; its null space is the zone's
        // all-ones vector, which (because the basis is a partition of unity on each zone)
        // IS the constant function, so a baseline constant in frequency is removed
        // exactly at any eta. It is assembled per zone, never across a boundary: a
        // difference penalty spanning two zones would couple them with weight 1 and drop
        // the null space to a single global constant.
        //
        // Theta = I exactly, because the time basis is orthonormal -- so there is no
        // Kronecker contraction here, only the q == r diagonal. The kernel implements
        // ONLY the orthonormal basis; the reference's monomial option exists to
        // cross-check assembly and costs up to 5x in conditioning at n = 2.
        const float d0 = (nphi_z == 1) ? 0.0f : (((j == 0) || (j == nphi_z-1)) ? 1.0f : 2.0f);
        const float d1 = (j < nphi_z-1) ? -1.0f : 0.0f;
        #pragma unroll
        for (int q = 0; q <= n_deg; q++) {
            const int I = j*NQ + q;
            A[(I*(NB+1) + 0)*S + tid] += eta * d0;
            if (j+1 < nphi_z)
                A[(I*(NB+1) + NQ)*S + tid] += eta * d1;
        }

        // --- the right-hand side.
        {
            const float *zz = Z + long(nphi_z*(NPHI+1) + j)*nloc + tid;
            float Vc[MAX_NDEG+1];
            #pragma unroll
            for (int q = 0; q <= n_deg; q++)
                Vc[q] = tb.us[q][0] * zz[W];

            for (int k = 1; k <= W; k++) {
                const float e = zz[W+k] + zz[W-k];
                const float o = zz[W+k] - zz[W-k];
                #pragma unroll
                for (int q = 0; q <= n_deg; q++)
                    Vc[q] += tb.us[q][k] * ((tb.upar[q] > 0.0f) ? e : o);
            }
            #pragma unroll
            for (int q = 0; q <= n_deg; q++)
                u[(j*NQ + q)*S + tid] = Vc[q];
        }
    }

    // ---- Equilibrate to unit diagonal.
    //
    // Load-bearing twice over. It is what makes eps a scale-invariant threshold, so the
    // masking decision does not depend on the units of the data. Less obviously it is
    // also what makes the problem well conditioned at all: the raw matrix G + eta*D_1 has
    // condition number O(h/eta) -- linear in the widest knot interval and inverse in eta,
    // so order 1e5 at h = 3000, eta = 1e-3 -- and equilibration removes BOTH factors,
    // since a coefficient with no data has diagonal eta*(D_1)_jj and one with data has
    // diagonal O(h). Do not optimize it away, and do not threshold an un-equilibrated
    // pivot.
    for (int I = 0; I < nblk; I++) {
        const float d = A[(I*(NB+1))*S + tid];
        rs[I*S + tid] = (d > 0.0f) ? rn_rsqrt(d) : 1.0f;
    }
    for (int I = 0; I < nblk; I++) {
        const float rI = rs[I*S + tid];
        for (int B = 1; B <= NB; B++)
            if (I+B < nblk)
                A[(I*(NB+1) + B)*S + tid] *= rI * rs[(I+B)*S + tid];
    }
    for (int I = 0; I < nblk; I++) {
        // Set the diagonal to exactly 1 (or 0 for a row with no data) rather than letting
        // it come out 1 +- 2 ulp. r_min is then literally the smallest pivot.
        const float d = A[(I*(NB+1))*S + tid];
        A[(I*(NB+1))*S + tid] = (d > 0.0f) ? 1.0f : 0.0f;
    }

    // ---- Banded Cholesky, in place, recording the smallest pivot.
    //
    // The factor overwrites A in the "column band" convention LU[j][b] = L_{j+b,j}, which
    // is what makes the factorization in-place: at step j we read A[j][0..NB] and write
    // LU[j][0..NB] into the same slots, while every OTHER entry the step reads
    // (LU[j-c][*]) belongs to an earlier column and is already final.
    //
    // A non-positive pivot can only arise in a zone with no unmasked channels. The guard
    // is not paranoia: unguarded, a fully dead zone produces a negative pivot in every
    // measured case, and the resulting NaN does not stay in its zone -- the banded write
    // crosses into the next one, and 0/NaN = NaN. Zone decoupling is a property of the
    // matrix, not of the factorization loop. Such a zone is masked out wholesale anyway
    // (its r_min is 0), so the guard exists only to keep NaN and Inf away from the caller.
    float rmin = 1.0f;
    for (int j = 0; j < nblk; j++) {
        float acc = A[(j*(NB+1))*S + tid];
        const int cmax = min(NB, j);
        for (int c = 1; c <= cmax; c++) {
            const float v = A[((j-c)*(NB+1) + c)*S + tid];
            acc -= v*v;
        }

        const bool good = (acc > 0.0f);
        rmin = fminf(rmin, good ? acc : 0.0f);

        const float diag = good ? __fsqrt_rn(acc) : 1.0f;
        A[(j*(NB+1))*S + tid] = diag;
        const float rdiag = __frcp_rn(diag);

        const int bmax = min(NB, nblk-1-j);
        for (int bb = 1; bb <= bmax; bb++) {
            float acc2 = A[(j*(NB+1) + bb)*S + tid];
            const int cm = min(NB-bb, j);
            for (int c = 1; c <= cm; c++)
                acc2 -= A[((j-c)*(NB+1) + c)*S + tid] * A[((j-c)*(NB+1) + bb+c)*S + tid];
            A[(j*(NB+1) + bb)*S + tid] = good ? (acc2 * rdiag) : 0.0f;
        }
    }

    // ---- The rank test, which generalizes the 1-d dead-zone test.
    //
    // At n = 0 "at least one unmasked channel" is exactly "live >= 1". At n > 0 a
    // degree-n fit in time is singular unless the zone carries data at n+1 DISTINCT
    // window offsets, whatever the channel count at those offsets, because a nonzero
    // degree-n polynomial vanishing on every populated offset is a null direction of the
    // whole assembled matrix. Structural and exact, and not inferrable from a pivot.
    int live = 0;
    for (int k = 0; k <= 2*W; k++)
        live += (Zlive[tid + k] > 0.0f) ? 1 : 0;
    if (live < n_deg+1)
        rmin = 0.0f;

    // ---- Solve, unscale, commit.
    for (int I = 0; I < nblk; I++)
        u[I*S + tid] *= rs[I*S + tid];

    for (int j = 0; j < nblk; j++) {          // forward substitution, L y = b
        float acc = u[j*S + tid];
        const int cmax = min(NB, j);
        for (int c = 1; c <= cmax; c++)
            acc -= A[((j-c)*(NB+1) + c)*S + tid] * u[(j-c)*S + tid];
        u[j*S + tid] = acc * __frcp_rn(A[(j*(NB+1))*S + tid]);
    }
    for (int j = nblk-1; j >= 0; j--) {       // backward substitution, L^T x = y
        float acc = u[j*S + tid];
        const int bmax = min(NB, nblk-1-j);
        for (int bb = 1; bb <= bmax; bb++)
            acc -= A[(j*(NB+1) + bb)*S + tid] * u[(j+bb)*S + tid];
        u[j*S + tid] = acc * __frcp_rn(A[(j*(NB+1))*S + tid]);
    }
    for (int I = 0; I < nblk; I++)
        u[I*S + tid] *= rs[I*S + tid];

    // The committed baseline is sum_q alpha_jq p_q(0), NOT alpha_j0. Those coincide only
    // for monomials, where p_q(0) = delta_q0; with an orthonormal basis every even q
    // contributes. Getting it wrong is silent -- it still produces a plausible baseline,
    // just the wrong one.
    const bool bad = (rmin < eps);
    for (int j = 0; j < nphi_z; j++) {
        float a = 0.0f;
        #pragma unroll
        for (int q = 0; q <= n_deg; q++)
            a += u[(j*NQ + q)*S + tid] * tb.eval0[q];
        // A select, not a multiply: 'a' may be Inf or NaN in a zone that failed above.
        acoef[(long(m)*N_phi + coef_lo + j)*T + t] = bad ? 0.0f : a;
    }
    rmin_out[(long(m)*nzone + z)*T + t] = rmin;
}


// -------------------------------------------------------------------------------------------------
//
// Kernel 3: evaluate, subtract, expand the mask.
//
// Same block decomposition as kernel 1 -- the freq-range descriptors and basis tables are
// reused verbatim -- but over the OUTPUT region only, buffer samples [W, W+T).
//
// The n_phi+1 coefficients and the zone's r_min are loaded ONCE per thread, not once per
// channel: they depend on (j, t) and the freq-range fixes j. The channel loop is 3 FMA
// against 10 bytes of DRAM traffic.
//
// Mask expansion is whole-zone-only, and a freq-range lies in exactly one zone, so
// "expand" is the single '&& !bad' below. There is no connected-component chase through
// G's zero pattern; that is what eta > 0 bought.

template<int NPHI>
__global__ void __launch_bounds__(PASS_THREADS)
detrend_2d_subtract_kernel(float *data, unsigned char *mask,
                           const float *acoef, const float *rmin,
                           const float *phi_tab, const int *fr_desc,
                           int nfreq, int N_phi, int nzone, int nbuf,
                           int W, int T, int phi_stride, float eps)
{
    // TILED ON THE BUFFER INDEX, NOT THE OUTPUT INDEX, and this is worth 30% of the
    // kernel's DRAM traffic.
    //
    // The output region starts at buffer sample W, so tiling on t would put every warp's
    // base at (W + 32k) floats. DRAM granularity is a 32-byte sector, i.e. 8 floats, and
    // a channel row is nbuf floats long; with nbuf = 2056 = 8 (mod 32) each row already
    // starts on a sector boundary, so the only misalignment is the +W itself. At W = 4
    // that is 16 bytes -- half a sector -- and every warp's 128-byte data access then
    // straddles 5 sectors instead of 4 (+25%), while every 32-byte mask access straddles
    // 2 instead of 1 (+100%). Measured: 863 MB of DRAM reads against 614 MB of useful
    // traffic, and 62% of peak bandwidth against kernel 1's 92%.
    //
    // Tiling on the buffer index puts warp bases at multiples of 32 floats, which is
    // sector-aligned for every channel, and the ends are predicated off instead. The
    // cost is one extra block per (freq-range, beam) and a comparison.
    const int bi = blockIdx.x*PASS_THREADS + threadIdx.x;   // buffer sample
    const int fr = blockIdx.y;
    const int m = blockIdx.z;

    if ((bi < W) || (bi >= W + T))
        return;
    const int t = bi - W;                                   // output sample

    const int c_lo = fr_desc[4*fr + 0];
    const int c_hi = fr_desc[4*fr + 1];
    const int j0   = fr_desc[4*fr + 2];
    const int z    = fr_desc[4*fr + 3];

    float a[NPHI+1];
    #pragma unroll
    for (int aa = 0; aa <= NPHI; aa++)
        a[aa] = acoef[(long(m)*N_phi + j0 - NPHI + aa)*T + t];

    // Kernel 2 already zeroed the coefficients of a flagged zone, so the model is zero
    // there and the select below is belt and braces -- but the mask must be cleared here
    // regardless, and that is what makes the zone actually disappear downstream.
    const bool bad = (rmin[(long(m)*nzone + z)*T + t] < eps);

    // Apply per-block and per-thread offsets to the full-resolution arrays.
    //   before: shape (M, nfreq, nbuf), contiguous
    //   after:  shape (nfreq,), stride nbuf, based at output sample t
    data += (long(m)*nfreq)*nbuf + bi;
    mask += (long(m)*nfreq)*nbuf + bi;

    #pragma unroll 4
    for (int f = c_lo; f < c_hi; f++) {
        // Summed by an explicit loop in a fixed order, matching the reference: the
        // grouping is part of the bit-identity contract, not an implementation detail.
        float model = 0.0f;
        #pragma unroll
        for (int aa = 0; aa <= NPHI; aa++)
            model += __ldg(phi_tab + long(f)*phi_stride + aa) * a[aa];

        const float dv = data[long(f)*nbuf];
        const bool keep = (mask[long(f)*nbuf] != 0) && !bad;

        // SELECT, do not multiply. 'dv - model' IS computed at masked channels and may be
        // NaN there; the select discards it. The arithmetic form mask*(d-model) would
        // propagate the NaN, since 0*nan = nan.
        data[long(f)*nbuf] = keep ? (dv - model) : 0.0f;
        mask[long(f)*nbuf] = keep ? 1 : 0;
    }
}


// -------------------------------------------------------------------------------------------------
//
// Host code: knot vector, basis tables, freq-ranges, zones, time basis.


// The compiled configurations. To add one: add a row here, and a line to the dispatch in
// launch(). T is deliberately NOT part of a configuration -- it is a runtime kernel
// argument (see the glossary at the top), so a caller may pick any chunk length without
// a recompile, and test_gpu_kernel() uses a small one to keep its numpy oracle cheap.
struct Detrender2dConfig { long n_phi; };

static constexpr Detrender2dConfig detrender_2d_configs[] = {
    { 0 },
    { 1 },
    { 2 },
};


static string config_list_str()
{
    stringstream ss;
    for (const Detrender2dConfig &c: detrender_2d_configs)
        ss << ((&c == &detrender_2d_configs[0]) ? "" : ", ")
           << "(n_phi=" << c.n_phi << ")";
    return ss.str();
}


vector<long> Detrender2d::configs()
{
    vector<long> ret;
    for (const Detrender2dConfig &c: detrender_2d_configs)
        ret.push_back(c.n_phi);
    return ret;
}


// Evaluate the n_phi+1 nonzero B-splines at x, whose knot span is j0. This is the
// standard triangular form of the Cox-de Boor recursion (the NURBS book's Algorithm
// A2.2), which has no zero denominators at all provided the span is non-empty -- which
// is why there is no "drop the term with a vanishing denominator" special case here even
// though repeated knots are fully supported.
static void eval_basis(const vector<long> &knots, long n_phi, double x, long j0, double *out)
{
    vector<double> left(n_phi+1, 0.0), right(n_phi+1, 0.0);
    out[0] = 1.0;
    for (long p = 1; p <= n_phi; p++)
        out[p] = 0.0;

    for (long p = 1; p <= n_phi; p++) {
        left[p]  = x - double(knots[j0 + 1 - p]);
        right[p] = double(knots[j0 + p]) - x;
        double saved = 0.0;
        for (long r = 0; r < p; r++) {
            // The denominator straddles the span [knots[j0], knots[j0+1]) and is
            // therefore strictly positive.
            const double temp = out[r] / (right[r+1] + left[p-r]);
            out[r] = saved + right[r+1]*temp;
            saved = left[p-r]*temp;
        }
        out[p] = saved;
    }
}


// The orthonormal polynomial basis on the window, by modified Gram-Schmidt on the
// monomials s^q, s = -W..W.
//
// ORTHONORMAL, NOT MONOMIAL, and this is the one choice here that moves a threshold
// rather than a constant. For a window-constant mask the assembled matrix is exactly
// (G + eta D_1) kron Theta with Theta_qr = sum_s p_q p_r, and equilibration of a
// Kronecker product is Kronecker, so the pivots multiply: r_min(2d) = r_min(1d) *
// r_min(Theta). With raw monomials r_min(Theta) is 0.18 to 0.25 at n = 2, which would
// multiply the 1-d conditioning margin by that factor -- enough to push the worst
// adversarial mask below eps and expand zones whose fits are perfectly accurate. With
// p_q orthonormal, Theta = I and r_min(2d) = r_min(1d) exactly, and it costs nothing:
// the basis enters only through stencil coefficients that are precomputed either way.
//
// Gram-Schmidt on a symmetric grid preserves parity, because <s^i, s^j> = 0 whenever i+j
// is odd. moments.py's window folding depends on that being exact.
static void build_time_basis(long n, long W, vector<double> &P)
{
    const long nk = 2*W + 1;
    P.assign(nk*(n+1), 0.0);

    for (long k = 0; k < nk; k++) {
        const double s = double(k - W);
        double v = 1.0;
        for (long q = 0; q <= n; q++) {
            P[k*(n+1) + q] = v;
            v *= s;
        }
    }

    for (long q = 0; q <= n; q++) {
        for (long r = 0; r < q; r++) {
            double dot = 0.0;
            for (long k = 0; k < nk; k++)
                dot += P[k*(n+1) + r] * P[k*(n+1) + q];
            for (long k = 0; k < nk; k++)
                P[k*(n+1) + q] -= dot * P[k*(n+1) + r];
        }
        double nrm = 0.0;
        for (long k = 0; k < nk; k++)
            nrm += P[k*(n+1) + q] * P[k*(n+1) + q];
        nrm = sqrt(nrm);
        // Positive normalization fixes the sign convention (numpy's QR needs an explicit
        // sign(diag(R)) correction to get the same basis).
        for (long k = 0; k < nk; k++)
            P[k*(n+1) + q] /= nrm;
    }
}


static void fill_stencils(TimeStencils &tb, const vector<double> &P, int n_deg, int W)
{
    const int NPAIR_T = (n_deg+1)*(n_deg+2)/2;
    const int nq = n_deg + 1;

    double parity[MAX_NDEG+1];
    for (int q = 0; q <= n_deg; q++)
        parity[q] = (q % 2 == 0) ? 1.0 : -1.0;

    int p = 0;
    for (int q = 0; q <= n_deg; q++) {
        for (int r = q; r <= n_deg; r++, p++) {
            for (int k = 0; k <= W; k++)
                tb.gs[p][k] = float(P[(W+k)*nq + q] * P[(W+k)*nq + r]);
            tb.gpar[p] = float(parity[q] * parity[r]);
        }
    }
    xassert_eq(p, NPAIR_T);

    for (int q = 0; q <= n_deg; q++) {
        for (int k = 0; k <= W; k++)
            tb.us[q][k] = float(P[(W+k)*nq + q]);
        tb.upar[q] = float(parity[q]);
        tb.eval0[q] = float(P[W*nq + q]);
    }
}


Detrender2d::Detrender2d(long nfreq_, const vector<long> &knots_, long M_,
                         long n_phi_, long n_, long W_, long T_,
                         double eta_, double eps_) :
    nfreq(nfreq_), M(M_), n_phi(n_phi_), n(n_), W(W_), T(T_), nbuf(T_ + 2*W_),
    eta(eta_), eps(eps_)
{
    bool found = false;
    for (const Detrender2dConfig &c: detrender_2d_configs)
        if (c.n_phi == n_phi_)
            found = true;

    if (!found) {
        stringstream ss;
        ss << "Detrender2d: no kernel is compiled for n_phi=" << n_phi_
           << "; available configurations are " << config_list_str();
        throw runtime_error(ss.str());
    }

    // n is runtime, bounded only by the size of the by-value stencil struct.
    if ((n_ < 0) || (n_ > MAX_NDEG)) {
        stringstream ss;
        ss << "Detrender2d: n=" << n_ << " must be in [0, " << MAX_NDEG << "]";
        throw runtime_error(ss.str());
    }

    // W is runtime.  2W+1 >= n+1 is the algebraic minimum -- below it the time fit is
    // underdetermined before any masking -- and MAX_W exists only to give the by-value
    // stencil struct a compile-time size.
    if ((W < 0) || (W > MAX_W)) {
        stringstream ss;
        ss << "Detrender2d: W=" << W << " must be in [0, " << MAX_W << "]";
        throw runtime_error(ss.str());
    }
    if (2*W + 1 < n + 1) {
        stringstream ss;
        ss << "Detrender2d: a degree-" << n << " fit in time needs 2W+1 >= n+1, but W="
           << W << " gives a " << (2*W+1) << "-sample window";
        throw runtime_error(ss.str());
    }

    // T is runtime, but kernel 2's grid is T/solve_threads blocks and solve_threads is
    // chosen from {256,128,64,32}, so a multiple of 32 guarantees one of them divides T.
    // Requiring it keeps that kernel free of a predicated tail; every realistic chunk
    // length satisfies it.
    if ((T <= 0) || (T % 32 != 0)) {
        stringstream ss;
        ss << "Detrender2d: T=" << T << " must be a positive multiple of 32";
        throw runtime_error(ss.str());
    }

    if (nfreq < 1)
        throw runtime_error("Detrender2d: nfreq must be >= 1");
    if (M < 1)
        throw runtime_error("Detrender2d: M must be >= 1");
    if (eta <= 0.0)
        throw runtime_error("Detrender2d: eta must be > 0");
    if (eps <= 0.0)
        throw runtime_error("Detrender2d: eps must be > 0");

    // ---- Validate the knot vector.
    //
    // Strict, because the array comes from the caller. The end-multiplicity rule is the
    // one that is not merely stylistic: clamped ends are what put the constant function
    // in the span and make the basis a partition of unity on each zone, which is in turn
    // what makes the regulator's null space exactly the constants -- hence what makes a
    // constant baseline removable EXACTLY rather than shrunk. Reducing it does not
    // degrade gracefully, it destroys the property (at end multiplicity n_phi the best
    // fit to the constant 1 is off by 0.99).
    const vector<long> &kn = knots_;
    if (long(kn.size()) < n_phi + 2)
        throw runtime_error("Detrender2d: knot vector is too short");
    for (size_t i = 1; i < kn.size(); i++)
        if (kn[i] < kn[i-1])
            throw runtime_error("Detrender2d: knots must be non-decreasing");
    if ((kn.front() != 0) || (kn.back() != nfreq)) {
        stringstream ss;
        ss << "Detrender2d: knots must run from 0 to nfreq=" << nfreq
           << ", got [" << kn.front() << ", " << kn.back() << "]";
        throw runtime_error(ss.str());
    }
    for (int which = 0; which < 2; which++) {
        const long val = which ? nfreq : 0;
        long mult = 0;
        for (long v: kn)
            if (v == val)
                mult++;
        if (mult != n_phi + 1) {
            stringstream ss;
            ss << "Detrender2d: the " << (which ? "last" : "first") << " knot (" << val
               << ") has multiplicity " << mult << ", expected exactly n_phi+1 = "
               << (n_phi+1) << ". Clamped ends are what put the constant function in the"
               << " span and make the basis complete on the whole band.";
            throw runtime_error(ss.str());
        }
    }

    const long nk = long(kn.size());
    N_phi = nk - n_phi - 1;
    if (N_phi < 1)
        throw runtime_error("Detrender2d: N_phi = len(knots)-n_phi-1 must be >= 1");

    // Interior multiplicities, and the zone boundaries (an interior knot of multiplicity
    // exactly n_phi+1). No basis function straddles one -- phi_j has support
    // [k_j, k_{j+n_phi+1}), and if the boundary occupies knot indices i..i+n_phi then
    // j <= i-1 gives supp_hi <= v and j >= i gives supp_lo >= v -- so G and D_1 are
    // exactly block diagonal there and the fits on the two sides decouple.
    vector<long> bounds;
    for (long i = 0; i < nk; ) {
        long j = i;
        while ((j < nk) && (kn[j] == kn[i]))
            j++;
        const long mult = j - i;
        if ((kn[i] > 0) && (kn[i] < nfreq)) {
            if (mult > n_phi + 1) {
                stringstream ss;
                ss << "Detrender2d: interior knot " << kn[i] << " has multiplicity "
                   << mult << ", above n_phi+1 = " << (n_phi+1);
                throw runtime_error(ss.str());
            }
            if (mult == n_phi + 1)
                bounds.push_back(kn[i]);
        }
        i = j;
    }
    nzone = long(bounds.size()) + 1;

    // Span index of each channel: the largest j with knots[j] <= f, which lands on the
    // last knot of a repeated group and hence always on a NON-EMPTY span. That is what
    // the Cox-de Boor recursion needs. Channel f occupies [f, f+1) and its data sits at
    // f + 1/2, so no data point ever coincides with a knot.
    vector<long> j0(nfreq);
    {
        long j = 0;
        for (long f = 0; f < nfreq; f++) {
            while ((j+1 < nk) && (kn[j+1] <= f))
                j++;
            j0[f] = j;
        }
    }

    // The zone of phi_j is decided by its supp_lo = knots[j] alone (see above).
    vector<long> zone_of_coef(N_phi);
    for (long j = 0; j < N_phi; j++) {
        long z = 0;
        while ((z < long(bounds.size())) && (bounds[z] <= kn[j]))
            z++;
        zone_of_coef[j] = z;
    }

    // ---- Basis tables, built in float64 and cast, so that the working dtype affects the
    // arithmetic that uses the basis but not the basis itself.
    const long npair_f = (n_phi+1)*(n_phi+2)/2;
    phi_stride  = 4*((n_phi + 1 + 3) / 4);
    prod_stride = 4*((npair_f + 3) / 4);

    Array<float> phi_host({nfreq, phi_stride}, af_uhost | af_zero);
    Array<float> prod_host({nfreq, prod_stride}, af_uhost | af_zero);
    vector<double> nb(n_phi+1);

    for (long f = 0; f < nfreq; f++) {
        eval_basis(kn, n_phi, double(f) + 0.5, j0[f], &nb[0]);
        for (long a = 0; a <= n_phi; a++)
            phi_host.data[f*phi_stride + a] = float(nb[a]);
        long p = 0;
        for (long a = 0; a <= n_phi; a++)
            for (long b = a; b <= n_phi; b++, p++)
                prod_host.data[f*prod_stride + p] = float(nb[a] * nb[b]);
    }

    phi_tab = phi_host.to_gpu();
    prod_tab = prod_host.to_gpu();

    // ---- Freq-ranges: split each non-empty knot interval into pieces of about
    // CHANNELS_PER_RANGE channels. A freq-range never crosses a knot, so j0 is fixed on
    // it, and because a zone boundary IS a knot it never crosses a zone either.
    vector<long> fr_lo, fr_hi, fr_j0, fr_zone;
    for (long f = 0; f < nfreq; ) {
        long g = f;
        while ((g < nfreq) && (j0[g] == j0[f]))
            g++;
        const long len = g - f;
        long k = (len + CHANNELS_PER_RANGE/2) / CHANNELS_PER_RANGE;
        if (k < 1)
            k = 1;
        for (long i = 0; i < k; i++) {
            const long lo = f + (len*i)/k;
            const long hi = f + (len*(i+1))/k;
            if (hi <= lo)
                continue;
            fr_lo.push_back(lo);
            fr_hi.push_back(hi);
            fr_j0.push_back(j0[f]);
            fr_zone.push_back(zone_of_coef[j0[f]]);
        }
        f = g;
    }
    nfrange = long(fr_lo.size());
    xassert_gt(nfrange, 0);

    Array<int> fr_host({nfrange, 4}, af_uhost | af_zero);
    for (long i = 0; i < nfrange; i++) {
        fr_host.data[4*i + 0] = int(fr_lo[i]);
        fr_host.data[4*i + 1] = int(fr_hi[i]);
        fr_host.data[4*i + 2] = int(fr_j0[i]);
        fr_host.data[4*i + 3] = int(fr_zone[i]);
    }
    fr_desc = fr_host.to_gpu();

    // ---- Zone descriptors. Zones are contiguous in both the coefficient index and the
    // channel index, so a zone's freq-ranges are a contiguous run.
    Array<int> zone_host({nzone, 4}, af_uhost | af_zero);
    nphi_zone_max = 0;
    for (long z = 0; z < nzone; z++) {
        long clo = -1, chi = -1;
        for (long j = 0; j < N_phi; j++) {
            if (zone_of_coef[j] != z)
                continue;
            if (clo < 0)
                clo = j;
            chi = j + 1;
        }
        // Every zone spans a non-empty channel range (its boundaries are distinct channel
        // indices), so it always holds at least one coefficient and one freq-range.
        xassert_ge(clo, 0);

        long flo = -1, fhi = -1;
        for (long i = 0; i < nfrange; i++) {
            if (fr_zone[i] != z)
                continue;
            if (flo < 0)
                flo = i;
            fhi = i + 1;
        }
        xassert_ge(flo, 0);

        zone_host.data[4*z + 0] = int(flo);
        zone_host.data[4*z + 1] = int(fhi);
        zone_host.data[4*z + 2] = int(clo);
        zone_host.data[4*z + 3] = int(chi - clo);
        nphi_zone_max = max(nphi_zone_max, chi - clo);
    }
    zone_desc = zone_host.to_gpu();

    // ---- The time basis (compile-time in (n, W), so this runs once and never changes).
    vector<double> P;
    build_time_basis(n, W, P);
    TimeStencils *tb = new TimeStencils;
    fill_stencils(*tb, P, int(n), int(W));
    tb_blob = tb;

    // ---- Kernel-2 block size. One thread per (beam, zone, output sample) solve, so the
    // shared memory scales with the block, and the grid is T/solve_threads blocks -- hence
    // solve_threads must divide T. Choose the largest that fits, since a larger block
    // amortizes the stage-1 staging over more solves.
    const long NB = ((n_phi > 1) ? n_phi : 1)*(n+1) + n;
    const long nblk_max = nphi_zone_max * (n+1);
    const long ncompz_max = nphi_zone_max * (n_phi+2);

    int shmem_max = 0;
    CUDA_CALL(cudaDeviceGetAttribute(&shmem_max, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));

    solve_threads = 0;
    for (long s: { 256L, 128L, 64L, 32L }) {
        if (T % s != 0)
            continue;
        const long bytes = ((nblk_max*(NB+1) + 2*nblk_max)*s + ncompz_max*(s + 2*W) + (s + 2*W)) * 4;
        if (bytes <= shmem_max) {
            solve_threads = s;
            break;
        }
    }
    if (solve_threads == 0) {
        stringstream ss;
        ss << "Detrender2d: the largest zone has " << nphi_zone_max << " basis functions,"
           << " which needs more shared memory than this GPU offers (" << shmem_max
           << " bytes) even at 32 threads per block. Use more zone boundaries, i.e."
           << " interior knots of multiplicity n_phi+1, to split the frequency band.";
        throw runtime_error(ss.str());
    }

    // ---- Per-launch scratch.
    const long ncomp = npair_f + n_phi + 1;
    gu = Array<float>({M, nfrange, ncomp, nbuf}, af_gpu | af_zero);
    acoef = Array<float>({M, N_phi, T}, af_gpu | af_zero);
    rmin = Array<float>({M, nzone, T}, af_gpu | af_zero);
}


Detrender2d::~Detrender2d()
{
    delete reinterpret_cast<TimeStencils *>(tb_blob);
    tb_blob = nullptr;
}


template<int NPHI>
static void _launch(const Detrender2d &d, float *data, unsigned char *mask,
                    float *gu, float *acoef, float *rmin,
                    const float *phi_tab, const float *prod_tab,
                    const int *fr_desc, const int *zone_desc,
                    long nphi_zone_max, long solve_threads,
                    long phi_stride, long prod_stride, const void *tb_blob,
                    cudaStream_t stream)
{
    const int NB = bandwidth(NPHI, int(d.n));
    const TimeStencils &tb = *reinterpret_cast<const TimeStencils *>(tb_blob);

    const int nbuf = int(d.nbuf);
    const int n_deg = int(d.n);
    const int W = int(d.W);
    const int T = int(d.T);
    const int S = int(solve_threads);

    // Kernel 1.
    {
        dim3 nblocks((nbuf + PASS_THREADS - 1)/PASS_THREADS, int(d.nfrange), int(d.M));
        detrend_2d_accum_kernel<NPHI> <<< nblocks, PASS_THREADS, 0, stream >>>
            (data, mask, gu, phi_tab, prod_tab, fr_desc,
             int(d.nfreq), int(d.nfrange), nbuf, int(phi_stride), int(prod_stride));
        CUDA_PEEK("detrend_2d_accum_kernel");
    }

    // Kernel 2. The shared-memory request usually exceeds the 48 KB default, so opt in.
    // Done once per (configuration, process) rather than per launch.
    {
        const long nblk_max = long(nphi_zone_max)*(long(d.n)+1);
        const long ncompz_max = long(nphi_zone_max)*(NPHI+2);
        const long shmem = ((nblk_max*(NB+1) + 2*nblk_max)*S + ncompz_max*(S + 2*W) + (S + 2*W)) * 4;

        // The attribute is per (kernel, device), but the requirement depends on the
        // largest ZONE of this instance, and several Detrender2d instances with
        // different knot vectors coexist in one process. Track the high-water mark:
        // setting it once would make a later instance with a bigger zone fail to launch
        // with "invalid argument", and setting it unconditionally would let a later
        // instance with a smaller zone lower it and break an earlier one.
        static long attr_shmem = 0;
        if (shmem > attr_shmem) {
            CUDA_CALL(cudaFuncSetAttribute(detrend_2d_solve_kernel<NPHI>,
                                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                                           int(shmem)));
            attr_shmem = shmem;
        }

        dim3 nblocks(int(T/S), int(d.nzone), int(d.M));
        detrend_2d_solve_kernel<NPHI> <<< nblocks, S, size_t(shmem), stream >>>
            (gu, acoef, rmin, zone_desc, fr_desc,
             int(d.nfrange), int(d.nzone), int(d.N_phi), nbuf, int(nphi_zone_max),
             n_deg, W, T, float(d.eta), float(d.eps), tb);
        CUDA_PEEK("detrend_2d_solve_kernel");
    }

    // Kernel 3.
    {
        dim3 nblocks((nbuf + PASS_THREADS - 1)/PASS_THREADS, int(d.nfrange), int(d.M));
        detrend_2d_subtract_kernel<NPHI> <<< nblocks, PASS_THREADS, 0, stream >>>
            (data, mask, acoef, rmin, phi_tab, fr_desc,
             int(d.nfreq), int(d.N_phi), int(d.nzone), nbuf, W, T, int(phi_stride), float(d.eps));
        CUDA_PEEK("detrend_2d_subtract_kernel");
    }
}


void Detrender2d::launch(Array<float> &data, Array<unsigned char> &mask, cudaStream_t stream) const
{
    xassert_eq(data.ndim, 3);
    xassert_shape_eq(data, ({M, nfreq, nbuf}));
    xassert_shape_eq(mask, ({M, nfreq, nbuf}));
    xassert(data.is_fully_contiguous());
    xassert(mask.is_fully_contiguous());
    xassert(data.on_gpu());
    xassert(mask.on_gpu());

    // One line per compiled n_phi; see detrender_2d_configs[] above.
    #define _DT2D_DISPATCH(P)                                                          \
        _launch<P> (*this, data.data, mask.data, gu.data, acoef.data, rmin.data,        \
                    phi_tab.data, prod_tab.data, fr_desc.data, zone_desc.data,          \
                    nphi_zone_max, solve_threads, phi_stride, prod_stride, tb_blob, stream)

    if (n_phi == 0)
        _DT2D_DISPATCH(0);
    else if (n_phi == 1)
        _DT2D_DISPATCH(1);
    else if (n_phi == 2)
        _DT2D_DISPATCH(2);
    else
        throw runtime_error("Detrender2d::launch: internal error, unhandled configuration");

    #undef _DT2D_DISPATCH
}


void Detrender2d::time_selected()
{
    // The timing configuration: 2 beams, 30000 channels, 4 equal zones with 3 equally
    // spaced simple interior knots each.
    const long nfreq = 30000;
    const long M = 2;
    const long nzone = 4;
    const long kint = 3;

    for (const Detrender2dConfig &c: detrender_2d_configs) {
        const long n_phi = c.n_phi;

        vector<long> knots;
        for (long i = 0; i <= n_phi; i++)
            knots.push_back(0);
        const long zw = nfreq / nzone;
        for (long z = 0; z < nzone; z++) {
            const long base = z*zw;
            for (long i = 1; i <= kint; i++)
                knots.push_back(base + (i*zw)/(kint+1));
            if (z < nzone-1)
                for (long i = 0; i <= n_phi; i++)
                    knots.push_back(base + zw);
        }
        for (long i = 0; i <= n_phi; i++)
            knots.push_back(nfreq);

        const long Tc = 2048;
        Detrender2d det(nfreq, knots, M, n_phi, /*n=*/2, /*W=*/4, Tc);

        // Global memory traffic: kernel 1 reads the whole buffer, kernel 3 reads and
        // writes the output region. Every byte of an ideal implementation is touched
        // exactly once per pass, so (time -> bandwidth) is the figure of merit: the
        // kernels are expected to be memory bound.
        const double nbytes = double(M) * double(nfreq)
            * (double(det.nbuf) + 2.0*double(det.T)) * 5.0;   // 4 bytes data + 1 byte mask

        // The kernels are branch-free and the work is mask-independent, so the timing
        // does not depend on the data. An all-valid mask is used so that the "normal"
        // path is what gets timed.
        Array<float> data({M, nfreq, det.nbuf}, af_gpu | af_zero);
        Array<unsigned char> mask({M, nfreq, det.nbuf}, af_gpu);
        CUDA_CALL(cudaMemset(mask.data, 1, M*nfreq*det.nbuf));

        cout << "\nDetrender2d::time_selected()\n"
             << "    (n_phi, n, W, T) = (" << det.n_phi << ", " << det.n << ", " << det.W
             << ", " << det.T << "), M = " << M << ", nfreq = " << nfreq << "\n"
             << "    N_phi = " << det.N_phi << ", nzone = " << det.nzone
             << ", nfrange = " << det.nfrange << ", solve_threads = " << det.solve_threads << "\n"
             << "    data = " << (double(M)*nfreq*det.nbuf*4 / 1.0e9) << " GB, "
             << "mask = " << (double(M)*nfreq*det.nbuf / 1.0e9) << " GB\n"
             << "    global memory traffic per launch = " << (nbytes / 1.0e9) << " GB\n"
             << endl;

        const int niter = 20;
        const int print_interval = 5;
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
