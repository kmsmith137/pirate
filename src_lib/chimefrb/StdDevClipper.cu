#include "../../include/pirate/chimefrb/StdDevClipper.hpp"
#include "../../include/pirate/chimefrb/WiDownsampler.hpp"
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
// Step 3: sd_clip_1d_kernel, a port of rf_kernels std_dev_clipper::_clip_1d().
//
// One block per beam, over that beam's 'nrows' variances, in place. The block size is
// FIXED, not a knob, which is a deliberate exception to the port's "expose the occupancy
// knob" convention: this kernel runs B blocks over at most a few thousand floats, so its
// cost is launch latency, and no block size changes that. A knob would also invite a test
// comparing two block sizes on a sharp decision -- which two correct implementations need
// not agree on, since they sum in different orders -- so there is only one.
//
// float32 throughout, like the old code. The summation order differs from the old
// sequential loop, which moves vbar and s by roundoff; the tests bracket that. It is far
// below the float32 uncertainty of the variances themselves, so double accumulation would
// buy nothing measurable.
//
// The three passes re-read 'var' from L2 rather than staging it in shared memory: at most
// 64 KB per beam, which would not fit at the largest geometries, and the cost is not
// measurable at the smallest.

static constexpr int sd_clip_nthreads = 1024;


// _block_sum(): the sum of 'x' over the block, returned on every thread. 'shm' holds one
// value per warp and must not be in use elsewhere; the trailing __syncthreads() makes it
// safe to reuse on the next call.
template<typename T>
__device__ __forceinline__ T _block_sum(T x, T *shm)
{
    constexpr int NW = sd_clip_nthreads / 32;

    #pragma unroll
    for (int d = 16; d >= 1; d >>= 1)
        x += __shfl_xor_sync(0xffffffffu, x, d);

    if ((threadIdx.x & 31) == 0)
        shm[threadIdx.x >> 5] = x;

    __syncthreads();

    T t = 0;
    #pragma unroll
    for (int w = 0; w < NW; w++)
        t += shm[w];

    __syncthreads();
    return t;
}


__global__ void __launch_bounds__(sd_clip_nthreads)
sd_clip_1d_kernel(float *var, long nrows, float sigma)
{
    __shared__ int ishm[sd_clip_nthreads / 32];
    __shared__ float fshm[sd_clip_nthreads / 32];

    // Apply per-block (= per-beam) pointer offset.
    //   before: shape (B, nrows), contiguous
    //   after: shape (nrows,), contiguous
    var += long(blockIdx.x) * nrows;

    // Pass 1: count and sum the usable variances (those > 0).
    int n = 0;
    float acc1 = 0.0f;

    for (long i = threadIdx.x; i < nrows; i += sd_clip_nthreads) {
        float v = var[i];
        if (v > 0.0f) {
            n++;
            acc1 += v;
        }
    }

    n = _block_sum(n, ishm);
    acc1 = _block_sum(acc1, fshm);

    // At most one usable row: the whole beam is zeroed (the old code's "acc0 < 1.5"). 'n'
    // is block-uniform here, so every thread takes this branch or none does.
    if (n < 2) {
        for (long i = threadIdx.x; i < nrows; i += sd_clip_nthreads)
            var[i] = 0.0f;
        return;
    }

    // Pass 2: the standard deviation about the mean. Divides by n, not n-1, and both
    // statistics are fixed before anything is clipped: one pass, not an iteration.
    const float vbar = acc1 / float(n);
    float acc2 = 0.0f;

    for (long i = threadIdx.x; i < nrows; i += sd_clip_nthreads) {
        float v = var[i];
        if (v > 0.0f) {
            float d = v - vbar;
            acc2 += d * d;
        }
    }

    acc2 = _block_sum(acc2, fshm);

    const float s = sqrtf(acc2 / float(n));
    const float thresh = sigma * s;

    // Pass 3: clip. This is the old code's '>=' written out, and deliberately NOT a call
    // to wrms_survives(): that predicate is shared between GpuWrms's refinements and the
    // intensity clip because those two decisions must agree, while this is a different
    // decision on a different statistic. (The old code also visits the zero entries, where
    // the clip is a no-op; skipping them saves the stores.)
    for (long i = threadIdx.x; i < nrows; i += sd_clip_nthreads) {
        float v = var[i];
        if ((v > 0.0f) && (fabsf(v - vbar) >= thresh))
            var[i] = 0.0f;
    }
}


// -------------------------------------------------------------------------------------------------
//
// Step 4: sd_apply_kernel, zero the full-resolution weights of every killed row.
//
// STORES ZEROS ONLY. The old code ANDs a mask into the weights (AXIS_FREQ) or memsets whole
// rows (AXIS_TIME); storing +0.0f only where a row was killed is bit-identical to both, and
// touches the weights only in proportion to the killed fraction.
//
// A warp owns a 32-by-32 tile of downsampled cells: 32 consecutive t_ds (one per lane) by
// 32 consecutive f_ds. This differs from intensity_clip_kernel's 1-by-32 strip because the
// mask here is a function of the row alone -- there is no per-cell data to read -- so
// covering more cells per warp costs nothing, and a strip-per-warp grid would launch a
// million warps at production size to each read one float and exit.
//
// ONE LOAD PER WARP. The tile touches 32 rows (TIME: its 32 f_ds) or 32 columns (FREQ: its
// 32 t_ds), and the warp reads all 32 variances with a single coalesced load, then loops
// only over what was killed. So on clean data -- almost every tile -- a warp reads one
// cache line and exits. This is not a micro-optimization: steps 3-4 cost grows with the
// beam count, so it is per-warp work rather than launch latency, and a loop issuing 32
// loads per warp is where it went. (For FREQ the compiler cannot hoist a reload out of such
// a loop, since 'var' and 'weights' may alias as far as it knows.)
//
// No shared memory, no block reduction, no __syncthreads(): the warp count is a runtime
// knob (blockDim.y), as in intensity_clip_kernel.

__global__ void __launch_bounds__(1024)
sd_apply_kernel(float *weights, const float *var, int Df, int Dt,
                long F_ds, long T_ds, long ntiles, int axis)
{
    long g = long(blockIdx.x) * blockDim.y + threadIdx.y;

    // Warp-uniform (g depends only on threadIdx.y), so the whole warp returns together and
    // the ballot below always has all 32 lanes.
    if (g >= ntiles)
        return;

    const int lane = threadIdx.x;
    const long ntile_t = T_ds >> 5;
    const long ntile_f = F_ds >> 5;
    const long t_tile = g % ntile_t;  g /= ntile_t;
    const long f_tile = g % ntile_f;  g /= ntile_f;
    const long b = g;

    const long t_ds = (t_tile << 5) + lane;
    const long f0 = f_tile << 5;

    if (axis == int(ClipperAxis::TIME)) {
        // Lane j loads the variance of row f0+j, and a ballot tells every lane which of
        // the tile's 32 rows were killed. Each lane then zeroes its own cell in those rows.
        const float v = var[clipper_row(axis, b, f0 + lane, t_ds, F_ds, T_ds)];
        unsigned int killed = __ballot_sync(0xffffffffu, !(v > 0.0f));

        while (killed) {
            const int j = __ffs(killed) - 1;
            killed &= (killed - 1);
            clipper_zero_cell(weights, b, f0 + j, t_ds, F_ds, T_ds, Df, Dt);
        }
    }
    else {
        // FREQ: the row is this lane's column, the same for all 32 f_ds of the tile.
        if (var[clipper_row(axis, b, f0, t_ds, F_ds, T_ds)] > 0.0f)
            return;

        for (int j = 0; j < 32; j++)
            clipper_zero_cell(weights, b, f0 + j, t_ds, F_ds, T_ds, Df, Dt);
    }
}


// -------------------------------------------------------------------------------------------------


static ClipperAxis _checked_axis(ClipperAxis axis)
{
    if (axis == ClipperAxis::NONE)
        throw runtime_error("GpuStdDevClipper: AXIS_NONE is not supported"
                            " (rf_kernels::std_dev_clipper does not implement it)");
    return axis;
}


static double _checked_sigma(double sigma)
{
    if (sigma < 0.0)
        throw runtime_error("GpuStdDevClipper: expected sigma >= 0");
    return sigma;
}


static long _checked_warps(long warps_per_block)
{
    if ((warps_per_block != 4) && (warps_per_block != 8)
        && (warps_per_block != 16) && (warps_per_block != 32)) {
        stringstream ss;
        ss << "GpuStdDevClipper: warps_per_block=" << warps_per_block
           << " is not supported (expected 4, 8, 16 or 32)";
        throw runtime_error(ss.str());
    }
    return warps_per_block;
}


// Step 2 is a single pass (niter = 1), so the base gets iter_sigma = 0, which GpuWrms
// ignores at niter = 1.
GpuStdDevClipper::GpuStdDevClipper(long B_, long F_, long nt_chunk_, ClipperAxis axis_,
                                   double sigma_, long Df_, long Dt_, bool two_pass_,
                                   long warps_per_block_) :
    GpuClipperBase("GpuStdDevClipper", B_, F_, nt_chunk_, _checked_axis(axis_), Df_, Dt_,
                   1, 0.0, two_pass_),
    sigma(_checked_sigma(sigma_)),
    warps_per_block(_checked_warps(warps_per_block_))
{ }


void GpuStdDevClipper::launch(const Array<float> &intensity, Array<float> &weights,
                              Array<float> &scratch, cudaStream_t stream) const
{
    // Steps 1-2 (downsample, transpose if FREQ, GpuWrms at niter = 1), and all argument
    // checking. Only st.var is used; the mean is computed and discarded.
    StatisticOutputs st = _launch_statistic(intensity, weights, scratch, stream);

    // Step 3, in place on st.var.
    const long nrows = wrms_R / B;
    sd_clip_1d_kernel <<< B, sd_clip_nthreads, 0, stream >>>
        (st.var.data, nrows, float(sigma));
    CUDA_PEEK("sd_clip_1d_kernel");

    // Step 4.
    const long ntiles = B * (F_ds / 32) * (T_ds / 32);
    const long nblocks = (ntiles + warps_per_block - 1) / warps_per_block;
    const dim3 nthreads(32, warps_per_block);

    sd_apply_kernel <<< nblocks, nthreads, 0, stream >>>
        (weights.data, st.var.data, int(Df), int(Dt), F_ds, T_ds, ntiles, int(axis));
    CUDA_PEEK("sd_apply_kernel");
}


// -------------------------------------------------------------------------------------------------


void GpuStdDevClipper::time_selected()
{
    // The two std_dev_clippers in the production RFI config
    // (misc/chimefrb/configs/21-03-07-low-latency-uniform-badchannel-mask-noplot.json), each
    // of which appears there with both two_pass values: 36 AXIS_TIME and 24 AXIS_FREQ, all
    // at (Df,Dt) = (1,1) and sigma = 3.
    const vector<ClipperAxis> axes = { ClipperAxis::TIME, ClipperAxis::FREQ };

    const long B = 8, F = 1024, T = 4096;
    const double sigma = 3.0;
    const bool two_pass = true;
    const vector<long> warp_counts = { 4, 8, 16, 32 };
    const int niter_timing = 20;

    // Random intensity, and a FRESH copy of unit weights for every timed launch. Step 3
    // fires on this data -- about 0.3% of rows per launch, as in noise -- and the clipper
    // modifies its weights in place. Reusing one weights array would let killed rows pile up
    // over the timing loop, and step 4 zeroes every row whose variance is zero on every
    // launch, including rows killed on earlier launches, so in principle its measured cost
    // would grow with the length of the loop. (On this data the difference turned out to be
    // negligible, but it depends on how often step 3 fires, so the copies stay.)
    Array<float> intensity({B, F, T}, af_gpu);
    Array<float> weights({B, F, T}, af_gpu);
    {
        Array<float> hi({B, F, T}, af_uhost);
        Array<float> hw({B, F, T}, af_uhost);
        hi.randomize();
        for (long j = 0; j < B*F*T; j++)
            hw.data[j] = 1.0f;
        intensity.fill(hi);
        weights.fill(hw);
    }

    // One pristine copy per timed launch (plus a margin for warmup), refilled before each
    // warps_per_block setting.
    vector<Array<float>> wcopies;
    for (int i = 0; i < niter_timing + 4; i++)
        wcopies.push_back(Array<float> ({B, F, T}, af_gpu));

    for (ClipperAxis axis: axes) {
        const bool freq = (axis == ClipperAxis::FREQ);
        GpuStdDevClipper probe(B, F, T, axis, sigma, 1, 1, two_pass);
        Array<float> scratch({probe.scratch_nelts}, af_gpu | af_zero);

        // Predicted global memory traffic, following plans/chimefrb_std_dev_clipper.md
        // section 4: GpuWrms reads the (intensity, weights) pair once, the FREQ transpose
        // reads and writes it once more, and steps 3-4 touch only the R-float variance
        // array plus the killed rows' weights, which is negligible.
        const double full = 4.0 * B * F * T;
        const double nbytes = freq ? (6.0 * full) : (2.0 * full);

        cout << "\nGpuStdDevClipper::time_selected()\n"
             << "    axis=" << (freq ? "FREQ" : "TIME") << ", (Df,Dt)=(1,1), sigma=" << sigma
             << ", two_pass=" << two_pass << "\n"
             << "    (B, F, T) = (" << B << ", " << F << ", " << T << "),"
             << " (R, L) = (" << probe.wrms_R << ", " << probe.wrms_L << ")\n"
             << "    predicted traffic = " << (nbytes / 1.0e9) << " GB = "
             << (nbytes / full) << " full-resolution array passes" << endl;

        // Steps 1-2 alone, as the base class runs them, so that the cost of steps 3-4 can be
        // read off as the difference. (Built from the public sub-kernels rather than through
        // GpuClipperBase, whose launch helper is protected.)
        {
            Array<float> t_i, t_w;
            if (freq) {
                t_i = Array<float> ({B, T, F}, af_gpu | af_zero);
                t_w = Array<float> ({B, T, F}, af_gpu | af_zero);
            }
            Array<float> mean({probe.wrms_R}, af_gpu | af_zero);
            Array<float> var({probe.wrms_R}, af_gpu | af_zero);
            Array<float> empty;

            GpuWiDownsampler tr(1, 1, true);
            GpuWrms wrms(probe.wrms_L, 1, 0.0, two_pass);
            KernelTimer kt(niter_timing, 1);
            double dt = 0.0;

            while (kt.next()) {
                if (freq) {
                    tr.launch(t_i, t_w, intensity, weights, kt.stream);
                    wrms.launch(mean, var, t_i.reshape({probe.wrms_R, probe.wrms_L}),
                                t_w.reshape({probe.wrms_R, probe.wrms_L}), empty, kt.stream);
                }
                else {
                    wrms.launch(mean, var, intensity.reshape({probe.wrms_R, probe.wrms_L}),
                                weights.reshape({probe.wrms_R, probe.wrms_L}), empty, kt.stream);
                }
                if (kt.warmed_up)
                    dt = kt.dt;
            }

            cout << "    steps 1-2 only:       dt = " << (dt * 1.0e3) << " ms"
                 << ",  bandwidth = " << (nbytes / dt / 1.0e9) << " GB/s" << endl;
        }

        for (long W: warp_counts) {
            GpuStdDevClipper sd(B, F, T, axis, sigma, 1, 1, two_pass, W);
            for (Array<float> &wc: wcopies)
                wc.fill(weights);

            KernelTimer kt(niter_timing, 1);
            double dt = 0.0;
            long icall = 0;

            while (kt.next()) {
                Array<float> &wc = wcopies[(icall++) % long(wcopies.size())];
                sd.launch(intensity, wc, scratch, kt.stream);
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
