#ifndef _PIRATE_DETRENDER_KERNELS_HPP
#define _PIRATE_DETRENDER_KERNELS_HPP

// The cuda kernels of the 2-d spline detrender, and the host helpers that size their
// launches. Shared by GpuDetrenderLps2d (src_lib/DetrenderLps2d.cu) and by
// pirate::chimefrb::GpuSplineDetrender (src_lib/chimefrb/SplineDetrender.cu), which drives
// the same three-kernel pipeline with a different basis, weight convention and regulator;
// the two policy structs below are the knobs.
//
// This header is DEVICE code and is included from .cu files only. Detrender.hpp, the
// public header, deliberately knows nothing about it.

#ifndef __CUDACC__
#error "pirate/detrender_kernels.hpp contains device code; include it from a .cu file"
#endif

#include <algorithm>
#include <initializer_list>
#include <cuda_runtime.h>
#include <ksgpu/cuda_utils.hpp>

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// -------------------------------------------------------------------------------------------------
//
// Overview.
//
// Notation follows notes/detrending.tex, section "2-d detrending"; the numpy reference is
// pirate_frb/detrending/lps2d, and GpuDetrenderLps2d is compared to it by test_gpu_kernel()
// in that package.
//
// Three kernels, in stream order. The split is forced by the shape of the problem: the
// fit at one time sample couples every channel of a zone, so the full-resolution data
// must be reduced to sufficient statistics, solved, and then swept again. There is no
// single-pass formulation.
//
//   1. accumulate: (data, weights) -> (G, U), the per-freq-range sufficient statistics
//      G_jl[t] = sum_f w[f,t] phi_j(f) phi_l(f),   U_j[t] = sum_f w[f,t] phi_j(f) d[f,t].
//   2. solve: window-average (G,U) over 2W+1 samples, assemble the regularized normal
//      equations, factor, solve, and commit the baseline coefficients at the window
//      centre. One thread per (beam, zone, output sample).
//   3. subtract: evaluate the baseline at full resolution, subtract, expand the mask.
//
// The kernels know nothing about B-splines. What they know is:
//
//   - Per-channel TABLES of the n_phi+1 basis functions that are nonzero at that channel
//     (phi_tab) and their pairwise products (prod_tab), built by the host from whatever
//     basis it likes. Row f of phi_tab holds phi_{j0-n_phi+a}(f) for a = 0..n_phi.
//
//   - FREQ-RANGE descriptors, fr_desc[4*fr + {0,1,2,3}] = (c_lo, c_hi, j0, zone): a
//     contiguous run of channels [c_lo, c_hi) on which the same n_phi+1 basis functions
//     are nonzero, namely coefficients j0-n_phi .. j0. Their width is the occupancy knob:
//     it sets the block count in kernels 1 and 3 (see derive_channels_per_range()).
//
//   - ZONE descriptors, zone_desc[4*z + {0,1,2,3}] = (fr_lo, fr_hi, coef_lo, nphi_z): a
//     block of the fit that is exactly decoupled from the rest, occupying freq-ranges
//     [fr_lo, fr_hi) and coefficients [coef_lo, coef_lo + nphi_z). Both must be contiguous.
//
//   - A banded REGULATOR table (see kernel 2), the coefficient vector of the constant
//     function (unit_coef, needed only by WeightScaledStrength), and the time-axis
//     stencils (TimeStencils, from make_time_stencils()).
//
// Kernel 1 writes gu with layout (M, nfrange, NCOMP, nbuf), time fastest, where the NCOMP
// components are the (n_phi+1)(n_phi+2)/2 pair products in the order (0,0),(0,1),...,
// followed by the n_phi+1 data moments. Kernel 2 writes acoef (M, N_phi, T) and rmin
// (M, nzone, T); kernel 3 reads them.
//
// WHERE THE TIME GOES. Kernels 1 and 3 move 15 bytes of DRAM traffic per (beam, channel,
// output sample) with a uint8 mask and do 9 and 3 FMA respectively, i.e. 3.6 and 0.6
// flop/byte against an L40S balance point of 106. They are pure bandwidth with a 30x
// margin, so the only things that matter in them are full cache lines and enough warps in
// flight. Kernel 2 does ~2000 FMA per (beam, zone, output sample) but there are only
// M*nzone*T of those, which is ~3% of the chunk; it is written for clarity, not speed.
//
// CHUNK INVARIANCE. Output sample t is computed by an identical sequence of floating
// point operations whatever T is, so results are bit-identical across chunkings (given
// consistent padding, which is the caller's contract) and across runs -- PROVIDED the
// freq-range partition is held fixed, since it is part of the frequency summation order.
// The per-thread accumulation blocking is a compile-time constant, and there are NO
// atomics anywhere in the reduction. Reducing (G,U) with atomicAdd would be slightly
// simpler and about as fast, and would give up run-to-run determinism; do not do it
// without revisiting test_gpu_kernel()'s bit-identity assertions, which are the strongest
// structural tests we have.
//
// NAN SAFETY. Masked samples are allowed to hold anything, including NaN from a dropped
// packet. Every use of the data is a SELECT on the weight, never a multiply, because
// 0*nan = nan. This is checked by bit-identity under poisoning.
//
//
// TEMPLATE PARAMETERS.
//
//   NPHI   spline degree in frequency (n_phi in the reference and the tex). Compile-time
//          because it sizes register arrays (acc[NCOMP], a[NPHI+1]) and drives the
//          #pragma unroll in kernel 1's inner loop.
//
//   In     the input policy (kernels 1 and 3): what the second array is, how a weight is
//          read from it, and what kernel 3 writes back. MaskedInput or WeightedInput.
//
//   Reg    the regulator-strength policy (kernel 2): FixedStrength or WeightScaledStrength.
//
// The time-polynomial degree n, the window half-width W and the chunk length T are RUNTIME
// kernel arguments. None sizes a register array: T appears only as a stride and a bound,
// and W only as a loop bound and an offset into shared memory (kernel 2's parity fold is
// written with k on the outside for exactly this reason). W is bounded by MAX_W and n by
// MAX_NDEG only because the stencil struct is passed by value and needs a compile-time
// size; T must be a positive multiple of 32.


static constexpr int PASS_THREADS = 256;

// Largest supported window half-width and time-polynomial degree. They bound nothing in
// the algorithm; they give the by-value stencil struct and kernel 2's moment registers a
// compile-time size.
static constexpr int MAX_W = 16;
static constexpr int MAX_NDEG = 3;
static constexpr int MAX_NPAIR_T = (MAX_NDEG+1)*(MAX_NDEG+2)/2;

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
// sources with different bandwidths in j -- the data block (n_phi) and a regulator whose
// bandwidth does not depend on n_phi (D_1 has bandwidth 1 whatever n_phi is, since it is
// a difference penalty on coefficient indices). At n_phi = 0 the regulator is the wider
// of the two and a formula using n_phi alone would under-allocate.
static __host__ __device__ constexpr int bandwidth(int n_phi, int n)
{
    return ((n_phi > 1) ? n_phi : 1) * (n + 1) + n;
}

// Row width of the regulator table (kernel 2): bands 0 .. max(n_phi,1), matching the
// max() in bandwidth() so that every band the assembled matrix can hold has a slot.
static __host__ __device__ constexpr int reg_table_width(int n_phi)
{
    return ((n_phi > 1) ? n_phi : 1) + 1;
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

// Builds the stencils for the orthonormal polynomial basis of degree n on a window of
// 2W+1 samples. Defined in src_lib/DetrenderLps2d.cu.
TimeStencils make_time_stencils(long n, long W);


// -------------------------------------------------------------------------------------------------
//
// Input policies (kernels 1 and 3).
//
// load(): from the second array's element and the datum, produce the weight w and the
// weighted datum wd that kernel 1 accumulates. store(): what kernel 3 writes back, given
// the datum, the fitted model, and whether kernel 2 flagged the zone. weight_t is the
// element type of the second array; out_weight_t is the type kernel 3 points at it with,
// writable for a mask that gets expanded and const for weights that are never touched.
// Static member functions rather than lambdas, so that the header can be instantiated
// from more than one .cu file (notes/cpp.md).

// A uint8 mask, {0,1}-valued. Masked samples are SELECTED away, never multiplied, and
// kernel 3 writes a zero residual and clears the mask wherever the sample is masked or
// its zone was flagged. This is the GpuDetrenderLps2d convention.
struct MaskedInput
{
    using weight_t = unsigned char;
    using out_weight_t = unsigned char;

    static __device__ __forceinline__ void load(unsigned char m, float d, float &w, float &wd)
    {
        const bool mv = (m != 0);
        w  = mv ? 1.0f : 0.0f;
        wd = mv ? d : 0.0f;
    }

    static __device__ __forceinline__ void store(float *data, unsigned char *m, float dv, float model, bool bad)
    {
        // SELECT, do not multiply. 'dv - model' IS computed at masked channels and may be
        // NaN there; the select discards it. The arithmetic form mask*(d-model) would
        // propagate the NaN, since 0*nan = nan.
        const bool keep = (*m != 0) && !bad;
        *data = keep ? (dv - model) : 0.0f;
        *m = keep ? 1 : 0;
    }
};

// Real-valued nonnegative weights, as the chimefrb pipeline carries them. The fit is
// weighted least squares; kernel 3 subtracts the model at EVERY channel and never writes
// the weight array. This is the pirate::chimefrb::GpuSplineDetrender convention.
//
// The select on wt != 0 is deliberate: a plain wt*d would let a NaN at a zero-weight
// channel poison the whole time sample.
struct WeightedInput
{
    using weight_t = float;
    using out_weight_t = const float;

    static __device__ __forceinline__ void load(float wt, float d, float &w, float &wd)
    {
        w  = wt;
        wd = (wt != 0.0f) ? wt*d : 0.0f;
    }

    static __device__ __forceinline__ void store(float *data, const float *, float dv, float model, bool)
    {
        *data = dv - model;
    }
};


// Regulator-strength policies (kernel 2). With FixedStrength the regulator is
// reg_strength * R. With WeightScaledStrength it is (reg_strength * W_tot[t]) * R, where
// W_tot[t] is the total weight of the window-centre sample, and kernel 2 computes W_tot
// from the Gram matrix already in shared memory.
struct FixedStrength        { static constexpr bool scaled = false; };
struct WeightScaledStrength { static constexpr bool scaled = true;  };


// -------------------------------------------------------------------------------------------------
//
// Kernel 1: accumulate.
//
// gridDim = (ntile, nfrange, M), blockDim = PASS_THREADS. Thread i of tile b owns ONE
// buffer sample t = b*PASS_THREADS + i and loops over its freq-range's channels. A warp
// therefore reads 128 contiguous bytes of 'data' (one full cache line) and, with a uint8
// mask, 32 contiguous bytes of it (one 32-byte sector), so nothing is wasted at DRAM
// granularity.
//
// The basis tables are broadcast loads -- every thread in the block reads the same
// address -- so the ~1 MB of tables is served from L1/L2, not DRAM. They are read as
// scalars rather than 128-bit vectors: at 9 broadcast loads per 9 FMA the LSU ceiling is
// tens of TB/s, i.e. 30x above DRAM, so vectorizing them would buy nothing. The tables
// are padded to a multiple of 4 floats per channel anyway, so a future 128-bit load
// needs no change to the host side.

template<int NPHI, class In>
__global__ void __launch_bounds__(PASS_THREADS)
detrend_2d_accum_kernel(const float *data, const typename In::weight_t *wt, float *gu,
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
    wt += (long(m)*nfreq)*nbuf + t;

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

            // The policy SELECTS on the weight, never multiplies: dv may be NaN where the
            // weight is zero.
            float w, wd;
            In::load(wt[long(f)*nbuf], dv, w, wd);

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
// S is chosen by the host (choose_solve_threads()) from the shared-memory budget and must
// divide T. The shared layout is sized by the LARGEST zone so that every block agrees on
// the offsets; solve_shmem_bytes() is the one place the layout is measured.
//
// THE REGULATOR is a banded matrix R supplied by the host as reg_tab, with
// reg_tab[j*reg_table_width(NPHI) + b] = R_{j,j+b} for b = 0 .. nreg_bands-1, zero across
// zone boundaries and beyond a zone's last coefficient. The kernel adds strength * R to
// the frequency block of every time-polynomial degree q (Theta = I, see the stencils), so
// nreg_bands is the only thing that distinguishes one regulator's shape from another's:
// 2 for the first-difference penalty D_1, n_phi+1 for a penalty with the data block's
// own bandwidth. strength is reg_strength, times the sample's total weight under
// WeightScaledStrength; unit_coef is read only in that case.
//
// Stage 1 reduces the zone's freq-ranges into shared Z, cooperatively. Doing it once per
// block rather than once per thread is worth 8x in global loads: each thread would
// otherwise read nfr*9*(2W+1) floats of its own.

template<int NPHI, class Reg>
__global__ void __launch_bounds__(256)
detrend_2d_solve_kernel(const float *gu, float *acoef, float *rmin_out,
                        const int *zone_desc, const int *fr_desc,
                        const float *reg_tab, const float *unit_coef, int nreg_bands,
                        int nfrange, int nzone, int N_phi, int nbuf, int nphi_zone_max,
                        int n_deg, int W, int T, float reg_strength, float eps, TimeStencils tb)
{
    constexpr int NPAIR_F = (NPHI+1)*(NPHI+2)/2;
    constexpr int NCOMP   = NPAIR_F + (NPHI+1);
    constexpr int NREG    = reg_table_width(NPHI);
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
    // Must match solve_shmem_bytes().
    extern __shared__ float sh[];
    const int nblk_max   = nphi_zone_max * NQ;
    const int ncompz_max = nphi_zone_max * (NPHI+2);
    float *A     = sh;                                  // A[(I*(NB+1)+B)*S + tid]
    float *u     = A + long(nblk_max)*(NB+1)*S;         // u[I*S + tid]
    float *rs    = u + long(nblk_max)*S;                // rs[I*S + tid]
    float *Z     = rs + long(nblk_max)*S;               // Z[e*nloc + i]
    float *Zlive = Z + long(ncompz_max)*nloc;           // Zlive[i]
    float *Zwsum = Zlive + nloc;                        // Zwsum[i], WeightScaledStrength only

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

        if constexpr (Reg::scaled) {
            // The sample's total weight, W_tot = sum_f w_f = c^T G c with c = unit_coef the
            // coefficient vector of the constant function 1 (so that (Phi c)_f = 1 and
            // c^T G c = sum_f w_f). Diagonal band once, off-diagonal bands twice. Read from
            // the banded Z already in shared memory, so it costs no global traffic.
            float wsum = 0.0f;
            for (int j = 0; j < nphi_z; j++) {
                const float cj = unit_coef[coef_lo + j];
                for (int bd = 0; bd <= NPHI; bd++) {
                    if (j + bd >= nphi_z)
                        break;
                    const float cl = unit_coef[coef_lo + j + bd];
                    const float g = Z[(j*(NPHI+1) + bd)*nloc + i];
                    wsum += ((bd == 0) ? 1.0f : 2.0f) * cj * cl * g;
                }
            }
            Zwsum[i] = wsum;
        }
    }

    __syncthreads();

    // ---- Stage 2: one solve per thread, for output sample t = b*S + tid.
    const int t = b*S + tid;

    float strength = reg_strength;
    if constexpr (Reg::scaled)
        strength *= Zwsum[tid + W];                 // the window-centre sample

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

        // --- the regulator, strength * (R kron Theta), with Theta = I.
        //
        // Theta = I exactly, because the time basis is orthonormal -- so there is no
        // Kronecker contraction here, only the q == r diagonal, and R's band b lands at
        // band offset b*NQ of the assembled matrix. The kernel implements ONLY the
        // orthonormal basis; the reference's monomial option exists to cross-check
        // assembly and costs up to 5x in conditioning at n = 2.
        //
        // This add must stay AFTER the data scatter above and BEFORE the right-hand side:
        // the order of adds into each slot of A is part of the bit-identity contract.
        for (int bd = 0; bd < nreg_bands; bd++) {
            if (j + bd >= nphi_z)
                break;
            const float r = strength * reg_tab[(coef_lo + j)*NREG + bd];
            #pragma unroll
            for (int q = 0; q <= n_deg; q++) {
                const int I = j*NQ + q;
                A[(I*(NB+1) + bd*NQ)*S + tid] += r;
            }
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
// "expand" is the single '&& !bad' in MaskedInput::store(). There is no
// connected-component chase through G's zero pattern; that is what eta > 0 bought.
//
// PERFORMANCE, AND WHAT HAS ALREADY BEEN RULED OUT. This kernel is ~72% of the chunk and
// sustains only 62% of peak DRAM bandwidth, against kernel 1's 92%, and it reads about
// 250 MB more than it asks for (866 MB measured against 614 MB of useful traffic). The
// one structural difference from kernel 1 is that this is a read+write stream where that
// one is read-only, and ~62% is a plausible GDDR6 figure for a 1:1 mix. Three hypotheses
// were tested and are NOT the answer -- do not spend time on them again:
//
//   - Load-to-store dependency stalling. The body reads and writes the same address, so
//     each iteration carries a dependency. Unrolling 4x changed nothing.
//   - Basis-table locality. The phi_tab broadcasts are 11.5 M sectors and a block streams
//     ~480 KB, so the resident working set rivals the 96 MB L2. Staging the table into
//     shared memory moved DRAM traffic by 0.2%. The tables hit L2 fine.
//   - Sector misalignment. This one WAS real and is fixed below (tiling on the buffer
//     index rather than the output index). It cut L1 sectors 12% and left DRAM unchanged.
//
// What remains untried is widening the per-thread access: 4 samples per thread would make
// a warp write a full 128-byte line of mask instead of a 32-byte sector, which is the last
// partial write in the kernel. It requires 4 | W, 4 | T and 4 | nbuf.

template<int NPHI, class In>
__global__ void __launch_bounds__(PASS_THREADS)
detrend_2d_subtract_kernel(float *data, typename In::out_weight_t *wt,
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
    // there and the select in store() is belt and braces -- but the mask must be cleared
    // regardless, and that is what makes the zone actually disappear downstream.
    const bool bad = (rmin[(long(m)*nzone + z)*T + t] < eps);

    // Apply per-block and per-thread offsets to the full-resolution arrays.
    //   before: shape (M, nfreq, nbuf), contiguous
    //   after:  shape (nfreq,), stride nbuf, based at output sample t
    data += (long(m)*nfreq)*nbuf + bi;
    wt += (long(m)*nfreq)*nbuf + bi;

    #pragma unroll 4
    for (int f = c_lo; f < c_hi; f++) {
        // Summed by an explicit loop in a fixed order, matching the reference: the
        // grouping is part of the bit-identity contract, not an implementation detail.
        float model = 0.0f;
        #pragma unroll
        for (int aa = 0; aa <= NPHI; aa++)
            model += __ldg(phi_tab + long(f)*phi_stride + aa) * a[aa];

        const float dv = data[long(f)*nbuf];
        In::store(data + long(f)*nbuf, wt + long(f)*nbuf, dv, model, bad);
    }
}


// -------------------------------------------------------------------------------------------------
//
// Host helpers: launch geometry shared by every user of the kernels.


// Freq-range width, DERIVED from the instance size rather than fixed, because the right
// value moves by 4x across the sizes we care about and the penalty for getting it wrong is
// large in one direction. Kernels 1 and 3 launch M * nfrange * ntile blocks; at a fixed
// 512 channels a small instance (M=1, nfreq=4096, T=512) gets nfrange = 8 and therefore 24
// blocks on 142 SMs, and runs 43% slower than the same instance at 128 channels. Measured
// on an L40S:
//
//     channels_per_range   128     256     512    1875
//     M=2 F=30000 T=2048   3.99    3.91    4.11    4.35   ms
//     M=1 F=4096  T=512    0.108   0.158   0.158   0.158  ms
//
// Note the asymmetry: over-provisioning freq-ranges costs almost nothing (the extra load
// falls on kernel 2, which is ~3% of the chunk), while under-provisioning costs 40%. So
// the target block count is deliberately generous, and M is left OUT of the estimate --
// using M = 1 is the conservative choice, and it is what keeps the beam axis a spectator:
// one row's output never depends on how many rows were processed alongside it.
//
// 'cpr_min' is the floor. GpuDetrenderLps2d uses the default; a caller whose basis
// intervals are narrower than 128 channels (so that the default would give one
// freq-range per interval and too few blocks) passes a smaller one.
static constexpr long CPR_TARGET_BLOCKS = 1024;
static constexpr long CPR_MIN = 128;
static constexpr long CPR_MAX = 1024;

inline long derive_channels_per_range(long nfreq, long nbuf, long cpr_min = CPR_MIN)
{
    const long ntile = (nbuf + PASS_THREADS - 1) / PASS_THREADS;
    long cpr = (nfreq * ntile) / CPR_TARGET_BLOCKS;
    cpr = std::max(cpr, cpr_min);
    cpr = std::min(cpr, CPR_MAX);
    return cpr;
}


// Dynamic shared memory of one kernel-2 block of S threads. The one place the shared
// layout is measured; kernel 2's pointer arithmetic must agree with it.
inline long solve_shmem_bytes(long nblk_max, long NB, long ncompz_max, long S, long W, bool scaled)
{
    const long nloc = S + 2*W;
    long nfloats = (nblk_max*(NB+1) + 2*nblk_max)*S    // A, u, rs
                 + ncompz_max*nloc                     // Z
                 + nloc;                               // Zlive
    if (scaled)
        nfloats += nloc;                               // Zwsum
    return nfloats * 4;
}


// Exact blocks-per-SM for kernel 2, from the occupancy API rather than an estimate.
//
// Dividing the per-SM shared memory by the per-block request is off by one at the
// boundary and ignores register pressure entirely. Measured on an L40S at 256 threads:
// 49 KB per block gives 2 resident blocks and 50 KB gives 1, even though 2 x 50 KB is
// exactly the 100 KB the device reports per SM -- there is about 1 KB of reserve, and
// MaxSharedMemoryPerBlockOptin (99 KB) is the pool that actually predicts it. Rather than
// encode that, ask CUDA.
//
// The MaxDynamicSharedMemorySize attribute must be raised BEFORE querying, or the API
// reports 0 for any request above 48 KB. Both functions are per kernel INSTANTIATION:
// a permit granted to <NPHI, FixedStrength> says nothing about <NPHI, WeightScaledStrength>.
template<int NPHI, class Reg>
static int solve_blocks_per_sm(long threads, long shmem)
{
    int nb = 0;
    CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &nb, detrend_2d_solve_kernel<NPHI, Reg>, int(threads), size_t(shmem)));
    return nb;
}

template<int NPHI, class Reg>
static void solve_permit_shmem(long bytes)
{
    CUDA_CALL(cudaFuncSetAttribute(detrend_2d_solve_kernel<NPHI, Reg>,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize, int(bytes)));
}


// Kernel-2 block size. One thread per (beam, zone, output sample) solve, so the shared
// memory scales with the block, and the grid is T/S blocks -- hence S must divide T.
// Returns 0 if even 8 threads per block need more shared memory than the GPU offers, in
// which case the caller's remedy is more zone boundaries.
//
// MAXIMIZE THREADS PER SM, NOT THREADS PER BLOCK, and the difference is not academic.
// The obvious rule -- take the largest block that fits -- is actively wrong here,
// because shared memory per block is what caps blocks per SM. At GpuDetrenderLps2d's
// production configuration a 64-thread block needs 57.9 KB, and an SM has 100 KB, so
// exactly ONE block is resident: 64 threads. A 32-thread block needs 29.3 KB, so THREE
// are resident: 96 threads. Measured, the 32-thread choice runs kernel 2 in 112 us
// against 124 us, i.e. 10% faster, despite doing more work -- its stage-1 halo is
// amortized over half as many solves (40/32 against 72/64). Blocks per SM comes from the
// occupancy API, not arithmetic.
//
// Ties go to the largest block, since that is the one with the least halo redundancy.
// Sub-warp blocks (16, 8) are allowed and are not a mistake: they waste most of a warp,
// but a large zone running at quarter-warp occupancy is very much better than a
// configuration the kernel refuses outright, and at 8 threads a zone of ~60 basis
// functions still fits.
//
// Also raises the dynamic shared-memory limit of this instantiation to the device
// maximum, once, so that the occupancy queries here and every later launch see the same
// permission. (A per-instance value would have an ordering hazard: an instance with a
// small zone must never lower a limit an earlier instance with a big zone depends on.)
template<int NPHI, class Reg>
static long choose_solve_threads(long T, long nblk_max, long NB, long ncompz_max, long W)
{
    // The device the kernels will run on: the caller's current device, not device 0.
    int dev = 0, shmem_max = 0, max_threads_sm = 0;
    CUDA_CALL(cudaGetDevice(&dev));
    CUDA_CALL(cudaDeviceGetAttribute(&shmem_max, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    CUDA_CALL(cudaDeviceGetAttribute(&max_threads_sm, cudaDevAttrMaxThreadsPerMultiProcessor, dev));

    solve_permit_shmem<NPHI, Reg>(shmem_max);

    long best = 0;
    long best_threads_sm = -1;
    for (long s: { 256L, 128L, 64L, 32L, 16L, 8L }) {
        if (T % s != 0)
            continue;
        const long bytes = solve_shmem_bytes(nblk_max, NB, ncompz_max, s, W, Reg::scaled);
        if (bytes > shmem_max)
            continue;
        const long blocks = solve_blocks_per_sm<NPHI, Reg>(s, bytes);
        const long threads = std::min(blocks * s, long(max_threads_sm));
        if (threads > best_threads_sm) {
            best_threads_sm = threads;
            best = s;
        }
    }
    return best;
}


}  // namespace pirate

#endif  // _PIRATE_DETRENDER_KERNELS_HPP
