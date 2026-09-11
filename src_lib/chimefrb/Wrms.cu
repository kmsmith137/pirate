#include "../../include/pirate/chimefrb/Wrms.hpp"

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
// Shared helpers.


// Shared memory available to the "small L" kernel for staging its row, in bytes.
//
// A block gets 48 KB total without opting in via cudaFuncSetAttribute(), and the
// block-reduction buffer comes out of the SAME budget -- 3 floats per warp, so 384 bytes
// at the largest supported block size. Forgetting it does not cost performance: it makes
// the launch fail with "invalid argument" for L within 12 of the ceiling, a band narrow
// enough to hide behind a lot of testing before it shows up.
//
// Opting in would raise the ceiling on L from ~6000 to about 12000, which no caller in
// the RFI chain needs: its largest per-row case is L = 4096.
static constexpr long smem_reduction_bytes = 3 * 32 * long(sizeof(float));
static constexpr long smem_budget = 48*1024 - smem_reduction_bytes;

// Target samples per block on the "large L" path. Small enough that a plane spreads over
// many blocks (the AXIS_NONE caller has only a handful of rows, so per-row parallelism
// alone would leave the GPU idle), large enough that the per-block overhead is amortized.
static constexpr long target_chunk = 1024;
static constexpr long max_nchunk = 1024;   // so one block can always finalize a row


// The four kinds of update a step can apply. The partial sums are always taken about the
// current mean[], which the host zeroes before the first step, so "about zero" is the
// mean=0 case of "about the mean" and needs no separate code path.
enum StepKind {
    step_mean_only = 0,        // two-pass first half: set mean, leave var alone
    step_var_about_mean = 1,   // two-pass second half: set var from sums about the mean
    step_single_pass = 2,      // one-pass first pass: set both from sums about zero
    step_iterate = 3           // a refinement
};


// block_sum3(): replaces (a,b,c) by their sums over the block, on every thread.
// 'shm' must have room for 3*(NTHREADS/32) floats, and must not be in use elsewhere.
template<int NTHREADS>
__device__ __forceinline__ void block_sum3(float &a, float &b, float &c, float *shm)
{
    constexpr int NW = NTHREADS / 32;

    // A butterfly (xor) reduction, rather than a shfl_down tree, so that every lane ends
    // up with the warp total and no broadcast is needed.
    #pragma unroll
    for (int d = 16; d >= 1; d >>= 1) {
        a += __shfl_xor_sync(0xffffffffu, a, d);
        b += __shfl_xor_sync(0xffffffffu, b, d);
        c += __shfl_xor_sync(0xffffffffu, c, d);
    }

    if constexpr (NW > 1) {
        const int lane = threadIdx.x & 31;
        const int warp = threadIdx.x >> 5;

        if (lane == 0) {
            shm[warp] = a;
            shm[NW + warp] = b;
            shm[2*NW + warp] = c;
        }

        __syncthreads();

        a = b = c = 0.0f;
        #pragma unroll
        for (int w = 0; w < NW; w++) {
            a += shm[w];
            b += shm[NW + w];
            c += shm[2*NW + w];
        }

        // Every thread has read 'shm' before the next step overwrites it.
        __syncthreads();
    }
}


// _apply_step(): the update rule, given the block's summed (wsum, wisum, wiisum).
// Shared by both kernels, so that "the algorithm" lives in exactly one place.
//
// Note the denominator: following rf_kernels, a non-positive weight sum is replaced by 1
// rather than special-cased, which makes both outputs come out zero.
__device__ __forceinline__ void _apply_step(int kind, float wsum, float wisum, float wiisum,
                                            float &mean, float &var)
{
    const float den = (wsum > 0.0f) ? wsum : 1.0f;

    if (kind == step_mean_only) {
        mean = wisum / den;         // sums were taken about zero
        return;
    }

    if (kind == step_var_about_mean) {
        float v = wiisum / den;
        float cut = wrms_eps_2 * mean;
        var = (v >= cut*cut) ? v : 0.0f;
        return;
    }

    if (kind == step_single_pass) {
        float m = wisum / den;      // sums were taken about zero
        float v = wiisum / den - m*m;
        mean = m;
        var = (v >= wrms_eps_3 * m * m) ? v : 0.0f;
        return;
    }

    // step_iterate. Note that the first cutoff uses the OLD mean and the second uses the
    // INCREMENT, not the new mean.
    float dm = wisum / den;
    float v = wiisum / den - dm*dm;
    float cut2 = wrms_eps_2 * mean;

    if (v < cut2*cut2)
        v = 0.0f;
    if (v < wrms_eps_3 * dm * dm)
        v = 0.0f;

    mean = mean + dm;
    var = v;
}


// -------------------------------------------------------------------------------------------------
//
// Shared-memory kernel: one threadblock per row, row staged in shared memory once.
//
// The whole point is that the global array is read exactly once no matter how many
// refinements run. Everything after the staging loop is on-chip.


template<int NTHREADS>
__global__ void __launch_bounds__(NTHREADS)
wrms_smem_kernel(float *mean_out, float *var_out,
                 const float *in_i, const float *in_w,
                 int L, int niter, float iter_sigma, bool two_pass)
{
    extern __shared__ float smem[];       // 2*L floats: intensity, then weights
    __shared__ float red[3 * (NTHREADS/32)];

    float *si = smem;
    float *sw = smem + L;

    // Apply per-block (= per-row) pointer offset.
    //   before: shape (R, L), contiguous
    //   after: shape (L,), contiguous
    in_i += long(blockIdx.x) * L;
    in_w += long(blockIdx.x) * L;

    // Stage. The strided loop is coalesced, and needs no tail predicate for any L.
    for (int j = threadIdx.x; j < L; j += NTHREADS) {
        si[j] = in_i[j];
        sw[j] = in_w[j];
    }

    __syncthreads();

    float mean = 0.0f;
    float var = 0.0f;
    float wsum, wisum, wiisum;

    // First pass. The two-pass variant is two sweeps -- one for the mean, one for the
    // variance about it -- which is the numerically stabler of the two and is what the
    // production chain's first detrender iteration asks for.
    wsum = wisum = wiisum = 0.0f;
    for (int j = threadIdx.x; j < L; j += NTHREADS) {
        float wv = sw[j];
        float iv = si[j];
        wsum += wv;
        wisum += wv * iv;
        wiisum += wv * iv * iv;
    }
    block_sum3<NTHREADS> (wsum, wisum, wiisum, red);

    if (two_pass) {
        _apply_step(step_mean_only, wsum, wisum, wiisum, mean, var);

        wsum = wisum = wiisum = 0.0f;
        for (int j = threadIdx.x; j < L; j += NTHREADS) {
            float wv = sw[j];
            float dv = si[j] - mean;
            wsum += wv;
            wiisum += wv * dv * dv;
        }
        block_sum3<NTHREADS> (wsum, wisum, wiisum, red);
        _apply_step(step_var_about_mean, wsum, wisum, wiisum, mean, var);
    }
    else {
        _apply_step(step_single_pass, wsum, wisum, wiisum, mean, var);
    }

    // Refinements.
    for (int k = 1; k < niter; k++) {
        float thresh = iter_sigma * sqrtf(var);

        wsum = wisum = wiisum = 0.0f;
        for (int j = threadIdx.x; j < L; j += NTHREADS) {
            float dv = si[j] - mean;
            // Note: the weight is zeroed rather than the sample skipped, which is what
            // makes a fully-clipped row come out with wsum = 0 rather than undefined.
            float wv = wrms_survives(dv, thresh) ? sw[j] : 0.0f;
            wsum += wv;
            wisum += wv * dv;
            wiisum += wv * dv * dv;
        }
        block_sum3<NTHREADS> (wsum, wisum, wiisum, red);
        _apply_step(step_iterate, wsum, wisum, wiisum, mean, var);
    }

    if (threadIdx.x == 0) {
        mean_out[blockIdx.x] = mean;
        var_out[blockIdx.x] = var;
    }
}


// -------------------------------------------------------------------------------------------------
//
// Global-memory kernels: one step is a partial-sum kernel plus a finalize kernel.
//
// Used when a row does not fit in shared memory, which in the RFI chain means the
// AXIS_NONE clipper, whose "row" is a whole (F_ds, T_ds) plane. There are only a handful
// of rows there, so the work has to be spread along the row rather than across rows.


template<int NTHREADS>
__global__ void __launch_bounds__(NTHREADS)
wrms_partial_kernel(float *scratch, const float *mean, const float *var,
                    const float *in_i, const float *in_w,
                    long L, int nchunk, float iter_sigma, bool do_clip)
{
    __shared__ float red[3 * (NTHREADS/32)];

    const int c = blockIdx.x;
    const long r = blockIdx.y;

    // Apply per-block pointer offset.
    //   before: shape (R, L), contiguous
    //   after: shape (L,), contiguous
    in_i += r * L;
    in_w += r * L;

    const long chunk = (L + nchunk - 1) / nchunk;
    const long j0 = long(c) * chunk;
    const long j1 = min(L, j0 + chunk);

    const float m = mean[r];
    const float thresh = do_clip ? (iter_sigma * sqrtf(var[r])) : 0.0f;

    float wsum = 0.0f, wisum = 0.0f, wiisum = 0.0f;

    for (long j = j0 + threadIdx.x; j < j1; j += NTHREADS) {
        float dv = in_i[j] - m;
        float wv = in_w[j];
        if (do_clip && !wrms_survives(dv, thresh))
            wv = 0.0f;
        wsum += wv;
        wisum += wv * dv;
        wiisum += wv * dv * dv;
    }

    block_sum3<NTHREADS> (wsum, wisum, wiisum, red);

    if (threadIdx.x == 0) {
        float *p = scratch + (r * nchunk + c) * 3;
        p[0] = wsum;
        p[1] = wisum;
        p[2] = wiisum;
    }
}


template<int NTHREADS>
__global__ void __launch_bounds__(NTHREADS)
wrms_finalize_kernel(float *mean, float *var, const float *scratch, int nchunk, int kind)
{
    __shared__ float red[3 * (NTHREADS/32)];

    const long r = blockIdx.x;
    const float *p = scratch + r * nchunk * 3;

    float wsum = 0.0f, wisum = 0.0f, wiisum = 0.0f;

    for (int c = threadIdx.x; c < nchunk; c += NTHREADS) {
        wsum += p[3*c];
        wisum += p[3*c + 1];
        wiisum += p[3*c + 2];
    }

    block_sum3<NTHREADS> (wsum, wisum, wiisum, red);

    if (threadIdx.x == 0) {
        float m = mean[r];
        float v = var[r];
        _apply_step(kind, wsum, wisum, wiisum, m, v);
        mean[r] = m;
        var[r] = v;
    }
}


// -------------------------------------------------------------------------------------------------


GpuWrms::GpuWrms(long L_, long niter_, double iter_sigma_, bool two_pass_,
                 long threads_per_block_) :
    L(L_), niter(niter_), iter_sigma(iter_sigma_), two_pass(two_pass_),
    threads_per_block(threads_per_block_)
{
    if (L < 1)
        throw runtime_error("GpuWrms: expected L >= 1");
    if (niter < 1)
        throw runtime_error("GpuWrms: expected niter >= 1 (niter counts total passes,"
                            " so niter=1 means no refinement)");
    if (iter_sigma < 0.0)
        throw runtime_error("GpuWrms: expected iter_sigma >= 0");

    if ((threads_per_block != 128) && (threads_per_block != 256)
        && (threads_per_block != 512) && (threads_per_block != 1024)) {
        stringstream ss;
        ss << "GpuWrms: threads_per_block=" << threads_per_block
           << " is not supported (expected 128, 256, 512 or 1024)";
        throw runtime_error(ss.str());
    }
}


long GpuWrms::max_shared_L()
{
    return smem_budget / (2 * long(sizeof(float)));
}


bool GpuWrms::is_shared_memory_path() const
{
    return L <= max_shared_L();
}


// Helper for the global-memory path: how many blocks share one row.
static long _nchunk(long L)
{
    long n = (L + target_chunk - 1) / target_chunk;
    return max(1L, min(n, max_nchunk));
}


long GpuWrms::scratch_nelts(long R) const
{
    if (is_shared_memory_path())
        return 0;
    return 3 * R * _nchunk(L);
}


template<int NTHREADS>
static void _launch_smem(float *mean, float *var, const float *in_i, const float *in_w,
                         long R, long L, long niter, double iter_sigma, bool two_pass,
                         cudaStream_t stream)
{
    int nbytes = int(2 * L * sizeof(float));

    wrms_smem_kernel<NTHREADS> <<< R, NTHREADS, nbytes, stream >>>
        (mean, var, in_i, in_w, int(L), int(niter), float(iter_sigma), two_pass);

    CUDA_PEEK("wrms_smem_kernel");
}


template<int NTHREADS>
static void _launch_global(float *mean, float *var, const float *in_i, const float *in_w,
                           float *scratch, long R, long L, long niter, double iter_sigma,
                           bool two_pass, cudaStream_t stream)
{
    const long nchunk = _nchunk(L);
    const dim3 pgrid(nchunk, R);

    // The partial kernels accumulate about mean[], so mean must be zero before the first
    // step. (var is written before it is read, but zeroing it keeps a failed launch from
    // leaving something that looks like a result.)
    CUDA_CALL(cudaMemsetAsync(mean, 0, R * sizeof(float), stream));
    CUDA_CALL(cudaMemsetAsync(var, 0, R * sizeof(float), stream));

    auto step = [&](int kind, bool do_clip) {
        wrms_partial_kernel<NTHREADS> <<< pgrid, NTHREADS, 0, stream >>>
            (scratch, mean, var, in_i, in_w, L, int(nchunk), float(iter_sigma), do_clip);
        CUDA_PEEK("wrms_partial_kernel");

        wrms_finalize_kernel<NTHREADS> <<< R, NTHREADS, 0, stream >>>
            (mean, var, scratch, int(nchunk), kind);
        CUDA_PEEK("wrms_finalize_kernel");
    };

    if (two_pass) {
        step(step_mean_only, false);
        step(step_var_about_mean, false);
    }
    else
        step(step_single_pass, false);

    for (long k = 1; k < niter; k++)
        step(step_iterate, true);
}


void GpuWrms::launch(Array<float> &mean, Array<float> &var,
                     const Array<float> &in_i, const Array<float> &in_w,
                     Array<float> &scratch, cudaStream_t stream) const
{
    xassert_eq(in_i.ndim, 2);
    xassert_eq(in_i.shape[1], L);

    long R = in_i.shape[0];
    xassert_gt(R, 0);

    xassert_shape_eq(in_w, ({R, L}));
    xassert_shape_eq(mean, ({R}));
    xassert_shape_eq(var, ({R}));

    const std::initializer_list<const Array<float> *> arrays =
        { &mean, &var, &in_i, &in_w };

    for (const Array<float> *a: arrays) {
        xassert(a->is_fully_contiguous());
        xassert(a->on_gpu());
    }

    xassert(mean.data != in_i.data);
    xassert(mean.data != in_w.data);
    xassert(var.data != in_i.data);
    xassert(var.data != in_w.data);
    xassert(mean.data != var.data);

    long ns = scratch_nelts(R);
    float *sp = nullptr;

    if (ns > 0) {
        xassert_eq(scratch.ndim, 1);
        xassert_ge(scratch.shape[0], ns);
        xassert(scratch.is_fully_contiguous());
        xassert(scratch.on_gpu());
        sp = scratch.data;
    }

    if (is_shared_memory_path()) {
        switch (threads_per_block) {
            case 128:  _launch_smem<128>  (mean.data, var.data, in_i.data, in_w.data, R, L, niter, iter_sigma, two_pass, stream); return;
            case 256:  _launch_smem<256>  (mean.data, var.data, in_i.data, in_w.data, R, L, niter, iter_sigma, two_pass, stream); return;
            case 512:  _launch_smem<512>  (mean.data, var.data, in_i.data, in_w.data, R, L, niter, iter_sigma, two_pass, stream); return;
            case 1024: _launch_smem<1024> (mean.data, var.data, in_i.data, in_w.data, R, L, niter, iter_sigma, two_pass, stream); return;
        }
    }
    else {
        switch (threads_per_block) {
            case 128:  _launch_global<128>  (mean.data, var.data, in_i.data, in_w.data, sp, R, L, niter, iter_sigma, two_pass, stream); return;
            case 256:  _launch_global<256>  (mean.data, var.data, in_i.data, in_w.data, sp, R, L, niter, iter_sigma, two_pass, stream); return;
            case 512:  _launch_global<512>  (mean.data, var.data, in_i.data, in_w.data, sp, R, L, niter, iter_sigma, two_pass, stream); return;
            case 1024: _launch_global<1024> (mean.data, var.data, in_i.data, in_w.data, sp, R, L, niter, iter_sigma, two_pass, stream); return;
        }
    }

    throw runtime_error("GpuWrms::launch(): internal error, unhandled threads_per_block");
}


// -------------------------------------------------------------------------------------------------


// One row of time_selected(): a caller from the old search's production RFI config.
struct TimingConfig
{
    long L;
    long R;
    long niter;
    double iter_sigma;
    const char *what;
};


void GpuWrms::time_selected()
{
    // The distinct (L, niter) shapes the production chain asks for, at B = 8 beams.
    // two_pass is true throughout: it costs one extra on-chip sweep, and the point of
    // the timing is the global traffic.
    const vector<TimingConfig> configs = {
        { 4096,   8*1024, 1, 0.0, "std_dev_clipper(AXIS_TIME, 1, 1)" },
        { 4096,   8*1024, 9, 5.0, "intensity_clipper(AXIS_TIME, 1, 1)" },
        { 1024,   8*4096, 1, 0.0, "std_dev_clipper(AXIS_FREQ, 1, 1)" },
        { 1024,   8*4096, 9, 5.0, "intensity_clipper(AXIS_FREQ, 1, 1)" },
        { 512,    8*256,  9, 3.0, "intensity_clipper(AXIS_FREQ, 2, 16)" },
        { 131072, 8,      9, 3.0, "intensity_clipper(AXIS_NONE, 2, 16)" },
    };

    const vector<long> tpb_values = { 128, 256, 512, 1024 };
    const int niter_timing = 20;

    for (const TimingConfig &c: configs) {
        Array<float> in_i({c.R, c.L}, af_gpu | af_zero);
        Array<float> in_w({c.R, c.L}, af_gpu | af_zero);
        Array<float> mean({c.R}, af_gpu | af_zero);
        Array<float> var({c.R}, af_gpu | af_zero);

        // All-ones weights, so that the refinements take the "sample survives" path.
        // With all-zero weights every row would be degenerate, and we would be timing the
        // wrong branch -- an easy mistake, since af_zero is the convenient default.
        {
            Array<float> ones({c.R, c.L}, af_uhost);
            for (long j = 0; j < c.R * c.L; j++)
                ones.data[j] = 1.0f;
            in_w.fill(ones);
        }

        GpuWrms probe(c.L, c.niter, c.iter_sigma, true);
        bool shared = probe.is_shared_memory_path();

        // Global memory traffic. On the shared-memory path each input array is read
        // exactly once however many refinements run; on the global path, once per step.
        //
        // Two numbers, because on the global path they differ by a factor of nsteps and
        // only one of them is DRAM traffic: the re-reads hit L2 when the arrays fit in
        // it, which is why that path can report a "bandwidth" above the hardware's peak.
        // The unique footprint is what says whether that is happening.
        long nsteps = shared ? 1 : (2 + c.niter - 1);
        double unique_bytes = 2.0 * c.R * c.L * 4.0;
        double nbytes = unique_bytes * double(nsteps);

        cout << "\nGpuWrms::time_selected()\n"
             << "    (L, R, niter) = (" << c.L << ", " << c.R << ", " << c.niter << "):  "
             << c.what << "\n"
             << "    path = " << (shared ? "shared memory" : "global memory")
             << ", " << nsteps << " pass(es) over a " << (unique_bytes / 1.0e6)
             << " MB footprint = " << (nbytes / 1.0e9) << " GB requested" << endl;

        for (long tpb: tpb_values) {
            GpuWrms wrms(c.L, c.niter, c.iter_sigma, true, tpb);
            Array<float> scratch({max(1L, wrms.scratch_nelts(c.R))}, af_gpu | af_zero);

            KernelTimer kt(niter_timing, 1);
            double dt = 0.0;

            while (kt.next()) {
                wrms.launch(mean, var, in_i, in_w, scratch, kt.stream);
                if (kt.warmed_up)
                    dt = kt.dt;
            }

            cout << "    threads_per_block = " << tpb
                 << ":  dt = " << (dt * 1.0e3) << " ms"
                 << ",  bandwidth = " << (nbytes / dt / 1.0e9) << " GB/s" << endl;
        }
    }
}


}}  // namespace pirate::chimefrb
