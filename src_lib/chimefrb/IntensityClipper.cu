#include "../../include/pirate/chimefrb/IntensityClipper.hpp"
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
// The kernel: apply a per-row threshold to the full-resolution weights.
//
// A warp owns one (beam, f_ds) and 32 consecutive downsampled time samples -- a 1-by-32
// strip of cells, covering a Df-by-(32*Dt) block of full-resolution weights. Lane 'l'
// owns cell l of the strip, entirely.
//
// IT WRITES ONLY ZEROS, AND USUALLY WRITES NOTHING. The old code ANDs a mask into the
// weights, which is a full read plus a full write of the (B,F,T) array. But (w & ~0) is
// w, so a surviving sample's store is a no-op, and (w & 0) is +0.0f for every w, so a
// masked sample's store is just a zero. Storing only where the mask fires is therefore
// bit-identical and touches the weights array only in proportion to the masked fraction:
// about 1e-6 on clean Gaussian data at sigma=5, and a few percent on RFI-heavy data. This
// is what turns a 2-pass kernel into a 1-pass one, and it is also why there is no
// shared-memory mask staging here -- with no read and almost no write there is nothing
// left to coalesce.
//
// Nothing is templated, and there is no block reduction, no shared memory and no
// __syncthreads(): the threads are completely independent. That is what makes the warp
// count a runtime parameter (taken from blockDim.y) rather than a template one, following
// GpuWiDownsampler; see the comment in WiDownsampler.cu for the measurement behind that
// rule. It is also what lets a warp return early on the last block without stranding
// anyone -- which most of them do, since most cells survive the clip.


__global__ void __launch_bounds__(1024)
intensity_clip_kernel(float *weights, const float *i_ds,
                      const float *mean, const float *var,
                      float sigma, int Df, int Dt,
                      long F_ds, long T_ds, long ntiles, int axis)
{
    const int lane = threadIdx.x;

    // Decompose the flat strip index into (b, f_ds, t_tile). Two integer divisions, once
    // per warp -- negligible next to the Df*Dt-element inner loop below, and it means the
    // only divisibility requirement is T_ds % 32 == 0 (no tail predicate is needed along
    // the frequency or beam axes).
    long g = long(blockIdx.x) * blockDim.y + threadIdx.y;

    if (g >= ntiles)
        return;

    const long ntile_t = T_ds >> 5;
    const long t_tile = g % ntile_t;  g /= ntile_t;
    const long f_ds = g % F_ds;       g /= F_ds;
    const long b = g;

    const long t_ds = (t_tile << 5) + lane;

    // The cell this lane owns. Consecutive lanes read consecutive floats, so the warp
    // reads exactly one 128-byte cache line.
    const float ival = i_ds[(b*F_ds + f_ds)*T_ds + t_ds];

    // Which statistic row this cell belongs to. Warp-uniform for TIME and NONE (a
    // broadcast load); for FREQ consecutive lanes read consecutive rows, which is another
    // single cache line. The branch inside clipper_row() is uniform across the whole grid.
    const long r = clipper_row(axis, b, f_ds, t_ds, F_ds, T_ds);

    // Note sigma here, not iter_sigma: the statistic's refinements used a different
    // threshold, and this is the final clip. Note also that var == 0 (a row with no usable
    // statistic) gives thresh == 0, and wrms_survives()'s strict '<' then masks the whole
    // row -- which is the intended behavior, not an edge case to guard against.
    const float thresh = sigma * sqrtf(var[r]);

    if (wrms_survives(ival - mean[r], thresh))
        return;

    // Only masked cells get here. See clipper_zero_cell() for the (deliberately strided)
    // store pattern.
    clipper_zero_cell(weights, b, f_ds, t_ds, F_ds, T_ds, Df, Dt);
}


// -------------------------------------------------------------------------------------------------


static double _checked_sigma(double sigma)
{
    if (sigma < 0.0)
        throw runtime_error("GpuIntensityClipper: expected sigma >= 0");
    return sigma;
}


static long _checked_warps(long warps_per_block)
{
    if ((warps_per_block != 4) && (warps_per_block != 8)
        && (warps_per_block != 16) && (warps_per_block != 32)) {
        stringstream ss;
        ss << "GpuIntensityClipper: warps_per_block=" << warps_per_block
           << " is not supported (expected 4, 8, 16 or 32)";
        throw runtime_error(ss.str());
    }
    return warps_per_block;
}


GpuIntensityClipper::GpuIntensityClipper(long B_, long F_, long nt_chunk_, ClipperAxis axis_,
                                         double sigma_, long Df_, long Dt_, long niter_,
                                         double iter_sigma_, bool two_pass_,
                                         long warps_per_block_) :
    GpuClipperBase("GpuIntensityClipper", B_, F_, nt_chunk_, axis_, Df_, Dt_,
                   niter_, iter_sigma_, two_pass_),
    sigma(_checked_sigma(sigma_)),
    warps_per_block(_checked_warps(warps_per_block_))
{ }


void GpuIntensityClipper::launch(const Array<float> &intensity, Array<float> &weights,
                                 Array<float> &scratch, cudaStream_t stream) const
{
    // Steps 1-2 (downsample, transpose if FREQ, GpuWrms), and all argument checking.
    StatisticOutputs st = _launch_statistic(intensity, weights, scratch, stream);

    // Step 3: the final clip. Note that this reads the UNTRANSPOSED downsampled intensity
    // (st.cell_i): the statistic wants frequency contiguous, and the mask application
    // wants time contiguous.
    long ntiles = B * F_ds * (T_ds / 32);
    long nblocks = (ntiles + warps_per_block - 1) / warps_per_block;
    dim3 nthreads(32, warps_per_block);

    intensity_clip_kernel <<< nblocks, nthreads, 0, stream >>>
        (weights.data, st.cell_i.data, st.mean.data, st.var.data, float(sigma),
         int(Df), int(Dt), F_ds, T_ds, ntiles, int(axis));

    CUDA_PEEK("intensity_clip_kernel");
}


// -------------------------------------------------------------------------------------------------


// One row of time_selected(): a clipper from the old search's production RFI config.
struct TimingConfig
{
    ClipperAxis axis;
    long Df;
    long Dt;
    double sigma;
    long niter;
    double iter_sigma;
};


static const char *_axis_name(ClipperAxis axis)
{
    switch (axis) {
        case ClipperAxis::FREQ: return "FREQ";
        case ClipperAxis::TIME: return "TIME";
        case ClipperAxis::NONE: return "NONE";
    }
    return "???";
}


void GpuIntensityClipper::time_selected()
{
    // The four distinct intensity_clippers in the production RFI config
    // (misc/chimefrb/configs/21-03-07-low-latency-uniform-badchannel-mask-noplot.json),
    // each of which appears there with both two_pass values. Note that iter_sigma is 3,
    // not 5, for the (2,16) pair: the two thresholds are different numbers.
    const vector<TimingConfig> configs = {
        { ClipperAxis::FREQ, 1,  1,  5.0, 9, 5.0 },
        { ClipperAxis::TIME, 1,  1,  5.0, 9, 5.0 },
        { ClipperAxis::NONE, 2,  16, 5.0, 9, 3.0 },
        { ClipperAxis::FREQ, 2,  16, 5.0, 9, 3.0 },
    };

    const long B = 8, F = 1024, T = 4096;
    const bool two_pass = true;
    const vector<long> warp_counts = { 4, 8, 16, 32 };
    const int niter_timing = 20;

    // Random intensity and unit weights. The data matters here in a way it does not for
    // the other two kernels: intensity_clip's write traffic is proportional to the masked
    // fraction (see the kernel comment), so timing it on all-zero data -- where every row
    // has var == 0 and is therefore entirely masked -- would measure the opposite of the
    // regime the RFI chain runs in. randomize() gives bounded values, so a 5-sigma clip
    // masks nothing and the reported bandwidth is the clean-data figure.
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

    for (const TimingConfig &c: configs) {
        const long F_ds = F / c.Df;
        const long T_ds = T / c.Dt;
        const bool need_ds = (c.Df != 1) || (c.Dt != 1);

        GpuIntensityClipper probe(B, F, T, c.axis, c.sigma, c.Df, c.Dt, c.niter,
                                  c.iter_sigma, two_pass);
        GpuWrms wprobe(probe.wrms_L, c.niter, c.iter_sigma, two_pass);

        // Predicted global memory traffic, following the cost model in
        // plans/chimefrb_intensity_clipper.md section 4: the downsample reads the
        // full-resolution pair and writes the downsampled one; the transpose reads and
        // writes the downsampled pair; GpuWrms reads it once (shared path) or once per
        // step (global path); and the clip reads the downsampled intensity and, on clean
        // data, writes essentially nothing.
        const double full = 4.0 * B * F * T;
        const double dsize = 4.0 * B * F_ds * T_ds;
        const long nsteps = wprobe.is_shared_memory_path() ? 1 : ((two_pass ? 2 : 1) + c.niter - 1);

        double nbytes = 0.0;
        if (need_ds)
            nbytes += 2.0*full + 2.0*dsize;
        if (c.axis == ClipperAxis::FREQ)
            nbytes += 4.0*dsize;
        nbytes += 2.0*dsize*double(nsteps) + dsize;

        Array<float> scratch({probe.scratch_nelts}, af_gpu | af_zero);

        cout << "\nGpuIntensityClipper::time_selected()\n"
             << "    axis=" << _axis_name(c.axis) << ", (Df,Dt)=(" << c.Df << "," << c.Dt
             << "), sigma=" << c.sigma << ", niter=" << c.niter
             << ", iter_sigma=" << c.iter_sigma << "\n"
             << "    (B, F, T) = (" << B << ", " << F << ", " << T << "),"
             << " (R, L) = (" << probe.wrms_R << ", " << probe.wrms_L << "),"
             << " GpuWrms path = " << (wprobe.is_shared_memory_path() ? "shared" : "global") << "\n"
             << "    predicted traffic = " << (nbytes / 1.0e9) << " GB = "
             << (nbytes / full) << " full-resolution array passes"
             << " (clean data: the clip writes ~nothing)" << endl;

        for (long W: warp_counts) {
            GpuIntensityClipper ic(B, F, T, c.axis, c.sigma, c.Df, c.Dt, c.niter,
                                   c.iter_sigma, two_pass, W);
            KernelTimer kt(niter_timing, 1);
            double dt = 0.0;

            while (kt.next()) {
                // Safe to reuse 'weights' across iterations: the clipper modifies it in
                // place, but on this data the clip fires on nothing, so it comes back
                // unchanged. (If it did fire, later iterations would time a smaller and
                // smaller amount of work.)
                ic.launch(intensity, weights, scratch, kt.stream);
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
