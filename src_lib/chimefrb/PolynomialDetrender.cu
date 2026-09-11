#include "../../include/pirate/chimefrb/PolynomialDetrender.hpp"

#include <sstream>
#include <iostream>
#include <ksgpu/xassert.hpp>
#include <ksgpu/cuda_utils.hpp>
#include <ksgpu/KernelTimer.hpp>

using namespace std;
using namespace ksgpu;

namespace pirate {
namespace chimefrb {
#if 0
}}  // editor auto-indent
#endif


// -------------------------------------------------------------------------------------------------
//
// poly_detrend_kernel<N>: one warp per row, everything in registers. N = polydeg + 1.
//
// A "row" is one (beam, channel, chunk): nt_chunk consecutive samples of one channel,
// contiguous in memory. The (M, nfreq, T) arrays are viewed as (nrows, n) with
// nrows = M*nfreq*(T/nt_chunk) and n = nt_chunk, which is exact because T is a multiple of
// nt_chunk. The block is (32, W): warp threadIdx.y of block blockIdx.x owns row
// blockIdx.x*W + threadIdx.y, and within the row lane l owns the sample pairs
// t = 2l + 64k and 2l + 1 + 64k, k = 0 .. n/64-1, as float2: each load or store
// instruction is 64-bit and covers two whole 128-byte lines of the row. (Two samples per
// lane is why nt_chunk must be a multiple of 64; 64-bit loads measured 13% faster than
// 32-bit at the full-band production shape, and 128-bit would buy little more on an L40S.)
//
// Three stages per row, all warp-uniform:
//
//   1. ACCUMULATE. Each lane sums, over its samples, the N(N+1)/2 entries of the lower
//      triangle of A_ab = sum_t w_t P_a(z_t) P_b(z_t) and the N entries of
//      v_a = sum_t w_t d_t P_a(z_t); a butterfly reduction then leaves the row totals on
//      every lane. P_a are the Legendre polynomials on the old code's grid
//      z_t = (2t+1-n)/n, evaluated on the fly by the three-term recurrence (no tables).
//      The weight-intensity product is a SELECT, wd = (w != 0) ? w*d : 0, so a NaN
//      intensity at a zero-weight sample contributes nothing (the old code multiplied).
//
//   2. GATE AND SOLVE, redundantly on all 32 lanes: everything is uniform, so there is no
//      divergence and nothing to broadcast. A is rescaled to unit diagonal and factored
//      pivot by pivot; at pivot j the Schur complement u_j = 1 - sum_{k<j} L_jk^2 must
//      exceed epsilon (the old code's u_j > epsilon*A_jj -- the rescaling does not change
//      the ratio), with the comparison written so that a NaN fails. If every pivot
//      passes, forward and back substitution give the coefficients.
//
//   3. SUBTRACT, or ZERO. A row that passed re-reads its intensities (from L2: the warp
//      loaded them moments ago), re-evaluates the polynomial and stores the difference. A
//      row that failed stores zero into its weights and does not touch its intensity.
//
// The sqrt and reciprocal in stage 2 are the correctly rounded intrinsics, not the
// approximate ones --use_fast_math would otherwise give: with those, scaling every weight
// by a power of four changes no bit of the output, which the unit test asserts.
//
// Register budget: the N(N+1)/2 + N accumulators plus the basis and coefficient vectors
// come to about a hundred registers at N = 9, so the block is capped at MAX_WARPS warps:
// a 512-thread __launch_bounds__ leaves 128 registers per thread, where a 1024-thread one
// would leave 64 and spill.

static constexpr int MAX_N = 9;        // polydeg <= 8
static constexpr int MAX_WARPS = 16;


// Index of (a, b), b <= a, in the packed lower triangle.
__host__ __device__ constexpr int tri(int a, int b) { return (a*(a+1))/2 + b; }


// The sum of x over the warp, on every lane.
__device__ __forceinline__ float warp_sum(float x)
{
    #pragma unroll
    for (int d = 16; d >= 1; d >>= 1)
        x += __shfl_xor_sync(0xffffffffu, x, d);
    return x;
}


// P_0(z) .. P_{N-1}(z) by the old code's recurrence, P_m = a_m z P_{m-1} + b_m P_{m-2} with
// a_m = (2m-1)/m and b_m = -(m-1)/m. The loop is fully unrolled, so the constants fold at
// compile time and no division runs per sample.
template<int N>
__device__ __forceinline__ void legendre_eval(float z, float P[N])
{
    P[0] = 1.0f;
    if constexpr (N > 1)
        P[1] = z;
    #pragma unroll
    for (int m = 2; m < N; m++) {
        const float a = float(2*m - 1) / float(m);
        const float b = -float(m - 1) / float(m);
        P[m] = a * z * P[m-1] + b * P[m-2];
    }
}


template<int N>
__global__ void __launch_bounds__(32 * MAX_WARPS)
poly_detrend_kernel(float *intensity, float *weights, long nrows, int n, float inv_n, float epsilon)
{
    constexpr int NA = (N*(N+1))/2;

    // Warp-uniform (r depends on threadIdx.y only), so the whole warp returns together and
    // every shuffle below has all 32 lanes.
    const long r = long(blockIdx.x) * blockDim.y + threadIdx.y;
    if (r >= nrows)
        return;

    const int lane = threadIdx.x;

    // Apply per-warp (= per-row) pointer offset.
    //   before: shape (nrows, n), contiguous
    //   after: shape (n,), contiguous
    intensity += r * long(n);
    weights += r * long(n);

    // ---- Stage 1: accumulate.

    float A[NA];
    float v[N];

    #pragma unroll
    for (int k = 0; k < NA; k++)
        A[k] = 0.0f;
    #pragma unroll
    for (int a = 0; a < N; a++)
        v[a] = 0.0f;

    for (int t0 = 2*lane; t0 < n; t0 += 64) {
        const float2 w2 = *reinterpret_cast<const float2 *> (weights + t0);
        const float2 d2 = *reinterpret_cast<const float2 *> (intensity + t0);

        #pragma unroll
        for (int u = 0; u < 2; u++) {
            const int t = t0 + u;
            // z = (2t+1-n)/n: an exact small integer times a reciprocal, one rounding.
            const float z = float(2*t + 1 - n) * inv_n;
            float P[N];
            legendre_eval<N> (z, P);

            const float w = u ? w2.y : w2.x;
            const float d = u ? d2.y : d2.x;
            const float wd = (w != 0.0f) ? (w * d) : 0.0f;

            #pragma unroll
            for (int a = 0; a < N; a++) {
                const float wp = w * P[a];
                #pragma unroll
                for (int b = 0; b <= a; b++)
                    A[tri(a,b)] += wp * P[b];
                v[a] += wd * P[a];
            }
        }
    }

    #pragma unroll
    for (int k = 0; k < NA; k++)
        A[k] = warp_sum(A[k]);
    #pragma unroll
    for (int a = 0; a < N; a++)
        v[a] = warp_sum(v[a]);

    // ---- Stage 2: rescale to unit diagonal, gate, Cholesky, solve.

    // s[a] = 1/sqrt(A_aa), or 1 where A_aa is not positive (no weight in the row, or a NaN
    // weight); the gate then fails at that pivot, since the Schur complement of a zero or
    // NaN diagonal entry is not > epsilon times it.
    float s[N];
    #pragma unroll
    for (int a = 0; a < N; a++) {
        const float daa = A[tri(a,a)];
        s[a] = (daa > 0.0f) ? __frcp_rn(__fsqrt_rn(daa)) : 1.0f;
    }

    #pragma unroll
    for (int a = 0; a < N; a++) {
        #pragma unroll
        for (int b = 0; b <= a; b++)
            A[tri(a,b)] *= s[a] * s[b];
    }

    // In-place Cholesky, L overwriting the lower triangle; rL[j] = 1/L_jj.
    bool ok = true;
    float rL[N];

    #pragma unroll
    for (int j = 0; j < N; j++) {
        const float ajj = A[tri(j,j)];          // 1, or 0 (see s[] above), or NaN
        float u = ajj;
        #pragma unroll
        for (int k = 0; k < j; k++)
            u -= A[tri(j,k)] * A[tri(j,k)];

        // The gate. A positive test, so that a NaN fails it.
        ok = ok && (u > epsilon * ajj);

        // Keep the factor finite when a pivot fails (the row is discarded anyway).
        const float ljj = (u > 0.0f) ? __fsqrt_rn(u) : 1.0f;
        A[tri(j,j)] = ljj;
        rL[j] = __frcp_rn(ljj);

        #pragma unroll
        for (int i = j+1; i < N; i++) {
            float x = A[tri(i,j)];
            #pragma unroll
            for (int k = 0; k < j; k++)
                x -= A[tri(i,k)] * A[tri(j,k)];
            A[tri(i,j)] = x * rL[j];
        }
    }

    // ---- Stage 3.

    if (!ok) {
        for (int t0 = 2*lane; t0 < n; t0 += 64)
            *reinterpret_cast<float2 *> (weights + t0) = make_float2(0.0f, 0.0f);
        return;
    }

    // L y = s*v (forward), then L^T y = y (backward, in place), then c = s*y.
    float y[N];

    #pragma unroll
    for (int a = 0; a < N; a++) {
        float x = s[a] * v[a];
        #pragma unroll
        for (int k = 0; k < a; k++)
            x -= A[tri(a,k)] * y[k];
        y[a] = x * rL[a];
    }
    #pragma unroll
    for (int a = N-1; a >= 0; a--) {
        float x = y[a];
        #pragma unroll
        for (int k = a+1; k < N; k++)
            x -= A[tri(k,a)] * y[k];
        y[a] = x * rL[a];
    }
    #pragma unroll
    for (int a = 0; a < N; a++)
        y[a] *= s[a];

    for (int t0 = 2*lane; t0 < n; t0 += 64) {
        float2 d2 = *reinterpret_cast<float2 *> (intensity + t0);
        float m[2];

        #pragma unroll
        for (int u = 0; u < 2; u++) {
            const float z = float(2*(t0+u) + 1 - n) * inv_n;
            float P[N];
            legendre_eval<N> (z, P);
            m[u] = 0.0f;
            #pragma unroll
            for (int a = 0; a < N; a++)
                m[u] += y[a] * P[a];
        }
        d2.x -= m[0];
        d2.y -= m[1];
        *reinterpret_cast<float2 *> (intensity + t0) = d2;
    }
}


// -------------------------------------------------------------------------------------------------


static long _checked_polydeg(long polydeg)
{
    if ((polydeg < 0) || (polydeg > MAX_N - 1)) {
        stringstream ss;
        ss << "GpuPolynomialDetrender: polydeg=" << polydeg << " is outside 0.." << (MAX_N - 1);
        throw runtime_error(ss.str());
    }
    return polydeg;
}


static double _checked_epsilon(double epsilon)
{
    if (!(epsilon > 0.0)) {
        stringstream ss;
        ss << "GpuPolynomialDetrender: epsilon=" << epsilon << " must be > 0";
        throw runtime_error(ss.str());
    }
    return epsilon;
}


static long _checked_nt_chunk(long nt_chunk)
{
    if ((nt_chunk <= 0) || (nt_chunk % 64 != 0)) {
        stringstream ss;
        ss << "GpuPolynomialDetrender: nt_chunk=" << nt_chunk << " must be a positive multiple of 64";
        throw runtime_error(ss.str());
    }
    return nt_chunk;
}


static long _checked_warps(long warps_per_block)
{
    if ((warps_per_block != 4) && (warps_per_block != 8) && (warps_per_block != 16)) {
        stringstream ss;
        ss << "GpuPolynomialDetrender: warps_per_block=" << warps_per_block
           << " is not supported (expected 4, 8 or 16)";
        throw runtime_error(ss.str());
    }
    return warps_per_block;
}


GpuPolynomialDetrender::GpuPolynomialDetrender(long polydeg_, double epsilon_, long nt_chunk_, long warps_per_block_) :
    polydeg(_checked_polydeg(polydeg_)),
    epsilon(_checked_epsilon(epsilon_)),
    nt_chunk(_checked_nt_chunk(nt_chunk_)),
    warps_per_block(_checked_warps(warps_per_block_))
{ }


template<int N>
static void _launch(float *intensity, float *weights, long nrows, long n, double epsilon,
                    long W, cudaStream_t stream)
{
    const long nblocks = (nrows + W - 1) / W;
    const dim3 nthreads(32, int(W));

    poly_detrend_kernel<N> <<< int(nblocks), nthreads, 0, stream >>>
        (intensity, weights, nrows, int(n), float(1.0 / double(n)), float(epsilon));
    CUDA_PEEK("poly_detrend_kernel");
}


void GpuPolynomialDetrender::launch(Array<float> &intensity, Array<float> &weights, cudaStream_t stream) const
{
    xassert_eq(intensity.ndim, 3);
    xassert_eq(weights.ndim, 3);
    for (int d = 0; d < 3; d++)
        xassert_eq(intensity.shape[d], weights.shape[d]);
    xassert(intensity.is_fully_contiguous());
    xassert(weights.is_fully_contiguous());
    xassert(intensity.on_gpu());
    xassert(weights.on_gpu());
    xassert(intensity.data != weights.data);

    // The kernel loads and stores float2, so both base pointers must be 8-byte aligned. A
    // cudaMalloc'ed array always is; a contiguous view starting at an odd element offset
    // of a larger array is not, and would fault inside the kernel rather than here.
    xassert((reinterpret_cast<uintptr_t>(intensity.data) & 7) == 0);
    xassert((reinterpret_cast<uintptr_t>(weights.data) & 7) == 0);

    const long M = intensity.shape[0];
    const long nfreq = intensity.shape[1];
    const long T = intensity.shape[2];

    if ((T <= 0) || (T % nt_chunk != 0)) {
        stringstream ss;
        ss << "GpuPolynomialDetrender::launch(): T=" << T << " must be a positive multiple of nt_chunk=" << nt_chunk;
        throw runtime_error(ss.str());
    }

    const long nrows = M * nfreq * (T / nt_chunk);
    if (nrows == 0)
        return;
    xassert_lt((nrows + warps_per_block - 1) / warps_per_block, (1L << 31));

    float *ip = intensity.data;
    float *wp = weights.data;
    const long W = warps_per_block;

    switch (polydeg) {
        case 0: _launch<1> (ip, wp, nrows, nt_chunk, epsilon, W, stream); break;
        case 1: _launch<2> (ip, wp, nrows, nt_chunk, epsilon, W, stream); break;
        case 2: _launch<3> (ip, wp, nrows, nt_chunk, epsilon, W, stream); break;
        case 3: _launch<4> (ip, wp, nrows, nt_chunk, epsilon, W, stream); break;
        case 4: _launch<5> (ip, wp, nrows, nt_chunk, epsilon, W, stream); break;
        case 5: _launch<6> (ip, wp, nrows, nt_chunk, epsilon, W, stream); break;
        case 6: _launch<7> (ip, wp, nrows, nt_chunk, epsilon, W, stream); break;
        case 7: _launch<8> (ip, wp, nrows, nt_chunk, epsilon, W, stream); break;
        case 8: _launch<9> (ip, wp, nrows, nt_chunk, epsilon, W, stream); break;
        default:
            throw logic_error("GpuPolynomialDetrender::launch(): polydeg out of range (constructor should have caught this)");
    }
}


// -------------------------------------------------------------------------------------------------


void GpuPolynomialDetrender::time_selected()
{
    // The production configuration (polydeg 4, epsilon 0.01, nt_chunk 1024) at the two
    // channel counts the old chain runs it at: the 16x-downsampled sub-pipelines and the
    // full band. 8 beams and one chunk per launch, as the other chimefrb timing functions
    // use. The kernel's work does not depend on the data except through the gate, which
    // unit weights never trip, so one random intensity array serves every launch.
    struct TimingConfig { long nfreq; const char *what; };
    const vector<TimingConfig> configs = {
        { 1024,  "sub-pipeline instances (16x downsampled, count-valued weights)" },
        { 16384, "top-level instance (full resolution, {0,1} weights)" },
    };
    const long M = 8, T = 1024, nt_chunk = 1024, polydeg = 4;
    const double epsilon = 0.01;
    const vector<long> warp_counts = { 4, 8, 16 };
    const int niter = 20;

    for (const TimingConfig &c: configs) {
        Array<float> intensity({M, c.nfreq, T}, af_gpu);
        Array<float> weights({M, c.nfreq, T}, af_gpu);
        {
            Array<float> hi({M, c.nfreq, T}, af_uhost);
            Array<float> hw({M, c.nfreq, T}, af_uhost);
            hi.randomize();
            for (long j = 0; j < M*c.nfreq*T; j++)
                hw.data[j] = 1.0f;
            intensity.fill(hi);
            weights.fill(hw);
        }

        // Global memory traffic of an ideal implementation: read the intensity and the
        // weights, write the intensity (12 bytes per sample). The weights are written only
        // on rows the gate fails, which unit weights never are.
        const double nbytes = 12.0 * double(M) * double(c.nfreq) * double(T);

        cout << "\nGpuPolynomialDetrender::time_selected()\n"
             << "    polydeg = " << polydeg << ", epsilon = " << epsilon << ", nt_chunk = " << nt_chunk
             << ":  " << c.what << "\n"
             << "    (M, nfreq, T) = (" << M << ", " << c.nfreq << ", " << T << ")"
             << ", global memory traffic per launch = " << (nbytes / 1.0e9) << " GB"
             << (c.nfreq * M * T * 4 * 2 < 96L*1024*1024 ? "  (fits in the L40S's 96 MB L2; the bandwidth below is optimistic)" : "")
             << endl;

        for (long W: warp_counts) {
            GpuPolynomialDetrender det(polydeg, epsilon, nt_chunk, W);

            KernelTimer kt(niter, 1);
            double dt = 0.0;
            while (kt.next()) {
                det.launch(intensity, weights, kt.stream);
                if (kt.warmed_up)
                    dt = kt.dt;
            }

            cout << "    warps_per_block = " << W
                 << ":  dt = " << (dt * 1.0e3) << " ms"
                 << ",  bandwidth = " << (nbytes / dt / 1.0e9) << " GB/s" << endl;
        }
    }
}


}}  // namespace pirate::chimefrb
