#include "../../include/pirate/chimefrb/WtUpsamplingKernel.hpp"

#include <random>
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
// The kernel: one lane per low-resolution cell.
//
// A warp owns one (beam, f_lo) row and 32 consecutive t_lo, the geometry of
// intensity_clip_kernel (IntensityClipper.cu). Each lane reads its cell's low-resolution
// weight -- together, one 128-byte load per warp -- and if the cell fails the test, zeroes
// the Df-by-Dt block of full-resolution weights behind it with zero_cell(). The
// full-resolution weights are never read: a kept weight is left bit-identical, and a masked
// one becomes +0.0, which is exactly what the old code's AND with an all-ones or all-zeros
// mask produces. So the traffic is one read of w_lores plus the masked cells.
//
// The last tile of a row may be partial (lanes past T_lo return), so there is no
// divisibility rule. Nothing is templated and nothing synchronizes or shuffles, so the warp
// count is a runtime parameter taken from blockDim.y (the rule in WiDownsamplingKernel.cu), and
// lanes may return independently.

__global__ void __launch_bounds__(1024)
weight_upsample_kernel(float *w_hires, const float *w_lores, float w_cutoff,
                       int Df, int Dt, long F_lo, long T_lo, long nwarps)
{
    // Decompose the warp index into (b, f_lo, t_tile): two integer divisions per warp.
    long g = long(blockIdx.x) * blockDim.y + threadIdx.y;

    if (g >= nwarps)
        return;

    const long ntile_t = (T_lo + 31) >> 5;
    const long t_tile = g % ntile_t;  g /= ntile_t;
    const long f_lo = g % F_lo;       g /= F_lo;
    const long b = g;

    const long t_lo = (t_tile << 5) + threadIdx.x;

    if (t_lo >= T_lo)
        return;

    // 'w_lores' has shape (B, F_lo, T_lo), contiguous.
    //
    // The old code's predicate, verbatim: a cell is KEPT iff w > w_cutoff. Written this way
    // round, not as "masked iff w <= w_cutoff", because the two differ when w is NaN: here, as
    // in the old code (whose AVX2 compare is ordered), a NaN weight masks its cell.
    if (w_lores[(b*F_lo + f_lo)*T_lo + t_lo] > w_cutoff)
        return;

    zero_cell(w_hires, b, f_lo, t_lo, F_lo, T_lo, Df, Dt);
}


// -------------------------------------------------------------------------------------------------


static long _checked_factor(const char *name, long D)
{
    if (D < 1) {
        stringstream ss;
        ss << "GpuWtUpsamplingKernel: expected " << name << " >= 1, got " << D;
        throw runtime_error(ss.str());
    }
    return D;
}


static double _checked_cutoff(double w_cutoff)
{
    // Written as !(w_cutoff >= 0) so that NaN throws too. (The old code checks w_cutoff < 0,
    // which lets NaN through, and a NaN cutoff would then mask everything.)
    if (!(w_cutoff >= 0.0)) {
        stringstream ss;
        ss << "GpuWtUpsamplingKernel: expected w_cutoff >= 0, got " << w_cutoff;
        throw runtime_error(ss.str());
    }
    return w_cutoff;
}


static long _checked_warps(long warps_per_block)
{
    if ((warps_per_block != 4) && (warps_per_block != 8)
        && (warps_per_block != 16) && (warps_per_block != 32)) {
        stringstream ss;
        ss << "GpuWtUpsamplingKernel: warps_per_block=" << warps_per_block
           << " is not supported (expected 4, 8, 16 or 32)";
        throw runtime_error(ss.str());
    }
    return warps_per_block;
}


GpuWtUpsamplingKernel::GpuWtUpsamplingKernel(long Df_, long Dt_, double w_cutoff_, long warps_per_block_) :
    Df(_checked_factor("Df", Df_)),
    Dt(_checked_factor("Dt", Dt_)),
    w_cutoff(_checked_cutoff(w_cutoff_)),
    warps_per_block(_checked_warps(warps_per_block_))
{ }


void GpuWtUpsamplingKernel::launch(Array<float> &w_hires, const Array<float> &w_lores,
                                   cudaStream_t stream) const
{
    xassert_eq(w_lores.ndim, 3);
    xassert_gt(w_lores.size, 0);

    const long B = w_lores.shape[0];
    const long F_lo = w_lores.shape[1];
    const long T_lo = w_lores.shape[2];

    xassert_shape_eq(w_hires, ({B, F_lo*Df, T_lo*Dt}));
    xassert(w_hires.is_fully_contiguous());
    xassert(w_lores.is_fully_contiguous());
    xassert(w_hires.on_gpu());
    xassert(w_lores.on_gpu());

    // Warps run in no particular order, so for (Df, Dt) != (1, 1) an overlap between the two
    // arrays would be a race. At (1, 1), where the shapes agree, passing one array twice is
    // the easy mistake to make.
    xassert(w_hires.data != w_lores.data);

    const long nwarps = B * F_lo * ((T_lo + 31) / 32);
    const long nblocks = (nwarps + warps_per_block - 1) / warps_per_block;
    const dim3 nthreads(32, warps_per_block);

    xassert_lt(nblocks, 1L << 31);

    weight_upsample_kernel <<< nblocks, nthreads, 0, stream >>>
        (w_hires.data, w_lores.data, float(w_cutoff), int(Df), int(Dt), F_lo, T_lo, nwarps);

    CUDA_PEEK("weight_upsample_kernel");
}


// -------------------------------------------------------------------------------------------------


static const int _ntiming_masks = 5;

static const char *_timing_mask_names[_ntiming_masks] = {
    "nothing masked (the floor: one read of w_lores, no stores)",
    "a contiguous run of F_lo/8 channels (the bad-channel shape: whole rows)",
    "every 32nd time sample (the std_dev axis=freq shape: isolated columns)",
    "3% of cells, independently (the intensity-clipper shape: isolated cells)",
    "everything masked (a memset of w_hires)"
};


// The low-resolution weights for mask number 'which' in time_selected(): 16 where the cell is
// kept (a count, as in the chain) and 0 where it is masked. Also returns the masked count.
static Array<float> _timing_lores(long B, long F_lo, long T_lo, int which, long &nmasked)
{
    Array<float> w({B, F_lo, T_lo}, af_uhost);
    std::mt19937 rng(137);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    nmasked = 0;

    for (long b = 0; b < B; b++) {
        for (long f = 0; f < F_lo; f++) {
            for (long t = 0; t < T_lo; t++) {
                bool masked;
                if (which == 0)
                    masked = false;
                else if (which == 1)
                    masked = (f >= F_lo/8) && (f < F_lo/4);
                else if (which == 2)
                    masked = ((t % 32) == 0);
                else if (which == 3)
                    masked = (u(rng) < 0.03f);
                else
                    masked = true;

                w.data[(b*F_lo + f)*T_lo + t] = masked ? 0.0f : 16.0f;
                nmasked += masked ? 1 : 0;
            }
        }
    }

    return w;
}


void GpuWtUpsamplingKernel::time_selected()
{
    // The production call: the wi_sub_pipeline upsample, 1024 -> 16384 channels at native
    // time resolution. 8 beams make w_hires 2.1 GB, so that every mask's writes go well past
    // the L40S's 96 MB L2. Its contents do not matter, since the kernel never reads it.
    const long B = 8, F_lo = 1024, T_lo = 4096, Df = 16, Dt = 1;
    const long F_hi = F_lo * Df;
    const long T_hi = T_lo * Dt;
    const vector<long> warp_counts = { 4, 8, 16, 32 };
    const int niter_timing = 20;

    Array<float> w_hires({B, F_hi, T_hi}, af_gpu | af_zero);
    Array<float> w_lores({B, F_lo, T_lo}, af_gpu);

    const double lores_bytes = 4.0 * B * F_lo * T_lo;
    const double hires_bytes = 4.0 * B * F_hi * T_hi;

    for (int which = 0; which < _ntiming_masks; which++) {
        long nmasked = 0;
        Array<float> h = _timing_lores(B, F_lo, T_lo, which, nmasked);
        w_lores.fill(h);

        // Predicted traffic: w_lores read once, and Df*Dt floats written per masked cell.
        // The old code's read-modify-write would read w_lores, and read and write all of
        // w_hires, whatever the mask.
        const double nbytes = lores_bytes + 4.0 * Df * Dt * nmasked;
        const double nbytes_rmw = lores_bytes + 2.0 * hires_bytes;

        cout << "\nGpuWtUpsamplingKernel::time_selected()\n"
             << "    (Df, Dt) = (" << Df << ", " << Dt << "), w_cutoff = 0, (B, F_lo, T_lo) = ("
             << B << ", " << F_lo << ", " << T_lo << ") -> (" << B << ", " << F_hi << ", "
             << T_hi << ")\n"
             << "    mask = " << _timing_mask_names[which] << ", "
             << (100.0 * nmasked / (B * F_lo * T_lo)) << "% of cells\n"
             << "    predicted traffic = " << (nbytes / 1.0e9) << " GB (the old code's"
             << " read-modify-write: " << (nbytes_rmw / 1.0e9) << " GB)" << endl;

        for (long W: warp_counts) {
            GpuWtUpsamplingKernel ups(Df, Dt, 0.0, W);
            KernelTimer kt(niter_timing, 1);
            double dt = 0.0;

            while (kt.next()) {
                ups.launch(w_hires, w_lores, kt.stream);
                if (kt.warmed_up)
                    dt = kt.dt;
            }

            cout << "    warps_per_block = " << W
                 << ":  dt = " << (dt * 1.0e3) << " ms"
                 << ",  bandwidth = " << (nbytes / dt / 1.0e9) << " GB/s" << endl;
        }

        if (nmasked == B * F_lo * T_lo) {
            KernelTimer kt(niter_timing, 1);
            double dt = 0.0;

            while (kt.next()) {
                CUDA_CALL(cudaMemsetAsync(w_hires.data, 0, size_t(hires_bytes), kt.stream));
                if (kt.warmed_up)
                    dt = kt.dt;
            }

            cout << "    cudaMemsetAsync of w_hires, for comparison:  dt = " << (dt * 1.0e3)
                 << " ms,  bandwidth = " << (hires_bytes / dt / 1.0e9) << " GB/s" << endl;
        }
    }
}


}}  // namespace pirate::chimefrb
