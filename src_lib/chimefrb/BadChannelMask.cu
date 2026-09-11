#include "../../include/pirate/chimefrb/BadChannelMask.hpp"

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
// The kernel: one warp per (beam, channel) row of the weights array.
//
// A warp whose channel is kept returns at once, having read one byte. A warp whose channel is
// masked stores zeros over its row, and never reads the weights. So the kernel's traffic is
// exactly the masked rows, and a masked channel comes out +0.0 whatever it held -- which is
// what the old code does (it stores a literal 0, not a multiply).
//
// The row loop's bound (t < T) doubles as the tail predicate, so there is no divisibility
// requirement on anything. Nothing is templated and nothing synchronizes, so the warp count is
// a runtime parameter, taken from blockDim.y (the rule in WiDownsampler.cu).
//
// Single-float stores, not float4: on an L40S, float4 stores measured no faster at any mask
// time_selected() tries, so the simpler loop stays.

__global__ void __launch_bounds__(1024)
badchannel_mask_kernel(float *weights, const uint8_t *keep, long F, long T, long nrows)
{
    // Rows of the (B, F, T) array are numbered row = b*F + f, so the channel is row % F: one
    // integer division per warp.
    const long row = long(blockIdx.x) * blockDim.y + threadIdx.y;

    if ((row >= nrows) || keep[row % F])
        return;

    // Per-warp offset. Before: 'weights' has shape (B*F, T), contiguous.
    // After: 'w' has shape (T,), contiguous. The per-thread offset is applied in the loop.
    float *w = weights + row*T;

    // Each store instruction writes 32 consecutive floats: one 128-byte cache line per warp
    // when the row is 128-byte aligned (T % 32 == 0, as in the chain), two otherwise.
    for (long t = threadIdx.x; t < T; t += 32)
        w[t] = 0.0f;
}


// -------------------------------------------------------------------------------------------------


// Throws unless 'keep' is a nonempty 1-d host array. Returns it, so that the constructor can
// validate it before initializing any other member from it.
static const Array<uint8_t> &_checked_keep(const Array<uint8_t> &keep)
{
    if (keep.ndim != 1)
        throw runtime_error("GpuBadChannelMask: expected a 1-d 'keep' array, got shape " + keep.shape_str());
    if (keep.size == 0)
        throw runtime_error("GpuBadChannelMask: the 'keep' array is empty");
    if (!keep.on_host())
        throw runtime_error("GpuBadChannelMask: 'keep' must be in host memory (the constructor copies it to the GPU)");
    return keep;
}


static long _count_masked(const Array<uint8_t> &keep)
{
    long n = 0;
    for (long f = 0; f < keep.shape[0]; f++)
        n += (keep.data[f * keep.strides[0]] == 0) ? 1 : 0;
    return n;
}


// A contiguous GPU copy of 'keep', normalized to 0/1. (The caller's array may have any stride.)
static Array<uint8_t> _gpu_keep(const Array<uint8_t> &keep)
{
    Array<uint8_t> h({keep.shape[0]}, af_uhost);
    for (long f = 0; f < keep.shape[0]; f++)
        h.data[f] = (keep.data[f * keep.strides[0]] != 0) ? 1 : 0;
    return h.to_gpu();
}


static long _checked_warps(long warps_per_block)
{
    if ((warps_per_block != 4) && (warps_per_block != 8)
        && (warps_per_block != 16) && (warps_per_block != 32)) {
        stringstream ss;
        ss << "GpuBadChannelMask: warps_per_block=" << warps_per_block
           << " is not supported (expected 4, 8, 16 or 32)";
        throw runtime_error(ss.str());
    }
    return warps_per_block;
}


GpuBadChannelMask::GpuBadChannelMask(const Array<uint8_t> &keep_, long warps_per_block_) :
    F(_checked_keep(keep_).shape[0]),
    nmasked(_count_masked(keep_)),
    keep(_gpu_keep(keep_)),
    warps_per_block(_checked_warps(warps_per_block_))
{ }


void GpuBadChannelMask::launch(Array<float> &weights, cudaStream_t stream) const
{
    xassert_eq(weights.ndim, 3);
    xassert_eq(weights.shape[1], F);
    xassert_gt(weights.size, 0);
    xassert(weights.is_fully_contiguous());
    xassert(weights.on_gpu());

    // The kernel would do nothing, but it would still cost a launch.
    if (nmasked == 0)
        return;

    const long nrows = weights.shape[0] * F;
    const long nblocks = (nrows + warps_per_block - 1) / warps_per_block;
    const dim3 nthreads(32, warps_per_block);

    badchannel_mask_kernel <<< nblocks, nthreads, 0, stream >>>
        (weights.data, keep.data, F, weights.shape[2], nrows);

    CUDA_PEEK("badchannel_mask_kernel");
}


// -------------------------------------------------------------------------------------------------


static const char *_timing_mask_names[4] = {
    "one channel",
    "every 8th channel",
    "one contiguous run of F/8 channels",
    "all channels"
};


// The 'keep' array for mask number 'which' in time_selected().
static Array<uint8_t> _timing_keep(long F, int which)
{
    Array<uint8_t> keep({F}, af_uhost);

    for (long f = 0; f < F; f++) {
        bool masked;
        if (which == 0)
            masked = (f == F/2);
        else if (which == 1)
            masked = ((f % 8) == 0);
        else if (which == 2)
            masked = (f >= F/8) && (f < F/4);
        else
            masked = true;
        keep.data[f] = masked ? 0 : 1;
    }

    return keep;
}


void GpuBadChannelMask::time_selected()
{
    // The production chain runs this once per chunk, at 1024 channels, with 12.6% of them
    // masked in a few contiguous runs. The four masks bracket that: the fixed cost of a launch
    // (one channel), an eighth of the channels spread evenly or bunched together, and all of
    // them, which is a plain memset and is timed against one.
    const long B = 8, F = 1024, T = 4096;
    const vector<long> warp_counts = { 4, 8, 16, 32 };
    const int niter_timing = 20;

    // The kernel never reads the weights, so one array serves every launch, and its contents
    // do not matter.
    Array<float> weights({B, F, T}, af_gpu | af_zero);
    const double full = 4.0 * B * F * T;

    for (int which = 0; which < 4; which++) {
        Array<uint8_t> keep = _timing_keep(F, which);
        GpuBadChannelMask probe(keep);

        // Predicted traffic: the masked rows, written once. Nothing is read but 'keep'.
        const double nbytes = 4.0 * B * T * probe.nmasked;

        cout << "\nGpuBadChannelMask::time_selected()\n"
             << "    mask = " << _timing_mask_names[which] << " (" << probe.nmasked << " of "
             << F << " channels), (B, F, T) = (" << B << ", " << F << ", " << T << ")\n"
             << "    predicted traffic = " << (nbytes / 1.0e6) << " MB of writes = "
             << (nbytes / full) << " full-array passes" << endl;

        for (long W: warp_counts) {
            GpuBadChannelMask bcm(keep, W);
            KernelTimer kt(niter_timing, 1);
            double dt = 0.0;

            while (kt.next()) {
                bcm.launch(weights, kt.stream);
                if (kt.warmed_up)
                    dt = kt.dt;
            }

            cout << "    warps_per_block = " << W
                 << ":  dt = " << (dt * 1.0e6) << " us"
                 << ",  bandwidth = " << (nbytes / dt / 1.0e9) << " GB/s" << endl;
        }

        if (probe.nmasked == F) {
            KernelTimer kt(niter_timing, 1);
            double dt = 0.0;

            while (kt.next()) {
                CUDA_CALL(cudaMemsetAsync(weights.data, 0, size_t(full), kt.stream));
                if (kt.warmed_up)
                    dt = kt.dt;
            }

            cout << "    cudaMemsetAsync of the whole array, for comparison:  dt = "
                 << (dt * 1.0e6) << " us,  bandwidth = " << (full / dt / 1.0e9) << " GB/s" << endl;
        }
    }
}


}}  // namespace pirate::chimefrb
