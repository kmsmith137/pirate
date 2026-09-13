#include "../../include/pirate/chimefrb/BadChannelMask.hpp"

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
//
// The MHz -> channel conversion, a port of rf_pipelines::badchannel_mask::_bind_transform()
// (extern/rf_pipelines/badchannel_mask.cpp), written line for line from its python
// transcription badchannel_keep() in pirate_frb/chimefrb/ReferenceBadChannelMask.py. That
// docstring states the rule and the old code's quirks; the unit test asserts that the two
// implementations agree exactly, which the two must-match details below are about.


// Channel i is masked iff a range reaches more than this fraction of a channel into it from
// both sides. Must equal FUDGE in ReferenceBadChannelMask.py.
static constexpr double BADCHANNEL_FUDGE = 1.0e-3;


// Returns the host 'keep' array, shape (nfreq,), 1 = keep, 0 = mask. Throws where the old
// code throws (see the constructor comment in the header).
static Array<uint8_t> _host_keep(const vector<pair<double,double>> &mask_ranges, long nfreq,
                                 pair<double,double> freq_range)
{
    const double flo = freq_range.first;
    const double fhi = freq_range.second;

    if (nfreq < 1)
        throw runtime_error("GpuBadChannelMask: expected nfreq >= 1");
    if (!(flo < fhi)) {
        stringstream ss;
        ss << "GpuBadChannelMask: expected freq_range=(lo, hi) with lo < hi, got (" << flo << ", " << fhi << ")";
        throw runtime_error(ss.str());
    }

    // Step 0, from the old constructor: every range must be nonempty.
    for (const auto &r: mask_ranges) {
        if (!(r.first < r.second)) {
            stringstream ss;
            ss << "GpuBadChannelMask: expected lo < hi in every mask range, got (" << r.first << ", " << r.second << ")";
            throw runtime_error(ss.str());
        }
    }

    // Step 1, from _bind_transform(): clip each range to the band. The three branches, and
    // the throw when none of them matches, are the old code's. Note that a range strictly
    // covering the whole band matches none of them.
    vector<pair<double,double>> clipped;
    for (const auto &r: mask_ranges) {
        const double lo = r.first, hi = r.second;
        if ((lo >= flo) && (hi <= fhi))
            clipped.push_back({lo, hi});
        else if ((lo < flo) && (hi >= flo) && (hi <= fhi))
            clipped.push_back({flo, hi});
        else if ((hi > fhi) && (lo <= fhi) && (lo >= flo))
            clipped.push_back({lo, fhi});
        else {
            stringstream ss;
            ss << "GpuBadChannelMask: the range (" << lo << ", " << hi << ") MHz lies entirely outside"
               << " the band [" << flo << ", " << fhi << "], or covers all of it; rf_pipelines rejects both";
            throw runtime_error(ss.str());
        }
    }

    // Step 2: to channel indices, in double precision, with the old code's expressions.
    const double scale = double(nfreq) / (fhi - flo);
    const double factor = scale * fhi;

    Array<uint8_t> keep({nfreq}, af_uhost);
    for (long f = 0; f < nfreq; f++)
        keep.data[f] = 1;

    for (const auto &r: clipped) {
        // The products go through volatile temporaries so that 'factor - x*scale' is rounded
        // TWICE, as the python reference rounds it. Host code is built with -march=x86-64-v3,
        // and GCC would otherwise be free to contract the multiply and subtract into one fused
        // instruction, which rounds once -- and a value within an ulp of an integer +/- FUDGE
        // could then land on the other side of floor() or ceil().
        volatile double t_hi = r.second * scale;
        volatile double t_lo = r.first * scale;
        long start = long(floor((factor - t_hi) + BADCHANNEL_FUDGE));
        long end = long(ceil((factor - t_lo) - BADCHANNEL_FUDGE));

        // The two ends are clamped differently, as in the old code: start into [0, nfreq-1]
        // and end into [0, nfreq]. start's upper clamp is what makes a range ending at the
        // bottom of the band mask the bottom channel even at zero width.
        start = min(max(start, 0L), nfreq - 1);
        end = min(max(end, 0L), nfreq);

        for (long f = start; f < end; f++)     // empty when end <= start
            keep.data[f] = 0;
    }

    return keep;
}


// -------------------------------------------------------------------------------------------------


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


static long _count_masked(const Array<uint8_t> &keep)
{
    long n = 0;
    for (long f = 0; f < keep.shape[0]; f++)
        n += (keep.data[f] == 0) ? 1 : 0;
    return n;
}


GpuBadChannelMask::GpuBadChannelMask(long nbeams_, long nfreq_, long ntime_,
                                     const vector<pair<double,double>> &mask_ranges_,
                                     pair<double,double> freq_range_, long warps_per_block_) :
    GpuBadChannelMask(nbeams_, nfreq_, ntime_, mask_ranges_, freq_range_, warps_per_block_,
                      _host_keep(mask_ranges_, nfreq_, freq_range_))
{ }


GpuBadChannelMask::GpuBadChannelMask(long nbeams_, long nfreq_, long ntime_,
                                     const vector<pair<double,double>> &mask_ranges_,
                                     pair<double,double> freq_range_, long warps_per_block_,
                                     const Array<uint8_t> &host_keep) :
    GpuTransformBase("GpuBadChannelMask", nbeams_, nfreq_, ntime_, /*scratch_nelts=*/0),
    mask_ranges(mask_ranges_),
    freq_range(freq_range_),
    warps_per_block(_checked_warps(warps_per_block_)),
    nmasked(_count_masked(host_keep)),
    keep(host_keep.to_gpu())
{ }


void GpuBadChannelMask::launch_checked(Array<float> &intensity, Array<float> &weights,
                                       Array<float> &scratch, cudaStream_t stream) const
{
    // The kernel would do nothing, but it would still cost a launch.
    if (nmasked == 0)
        return;

    const long nrows = nbeams * nfreq;
    const long nblocks = (nrows + warps_per_block - 1) / warps_per_block;
    const dim3 nthreads(32, warps_per_block);

    badchannel_mask_kernel <<< nblocks, nthreads, 0, stream >>>
        (weights.data, keep.data, nfreq, ntime, nrows);

    CUDA_PEEK("badchannel_mask_kernel");
}


// -------------------------------------------------------------------------------------------------


static const char *_timing_mask_names[4] = {
    "one channel",
    "every 8th channel",
    "one contiguous run of F/8 channels",
    "all channels"
};


// The MHz ranges for mask number 'which' in time_selected(), built from channel runs by the
// inverse of the conversion: channel edge i sits at fhi - i*(fhi-flo)/F, so the run [a, b) is
// the range (edge(b), edge(a)). Edge-aligned ranges convert back to exactly that run, because
// the fudge (1e-3 of a channel) absorbs the roundoff in the edge positions.
static vector<pair<double,double>> _timing_ranges(long F, int which, pair<double,double> band)
{
    const double flo = band.first, fhi = band.second;
    auto edge = [&](long i) { return fhi - double(i) * (fhi - flo) / double(F); };

    vector<pair<double,double>> ranges;
    if (which == 0)
        ranges.push_back({edge(F/2 + 1), edge(F/2)});
    else if (which == 1) {
        for (long f = 0; f < F; f += 8)
            ranges.push_back({edge(f+1), edge(f)});
    }
    else if (which == 2)
        ranges.push_back({edge(F/4), edge(F/8)});
    else
        ranges.push_back({flo, fhi});

    return ranges;
}


void GpuBadChannelMask::time_selected()
{
    // The production chain runs this once per chunk, at 1024 channels, with 12.6% of them
    // masked in a few contiguous runs. The four masks bracket that: the fixed cost of a launch
    // (one channel), an eighth of the channels spread evenly or bunched together, and all of
    // them, which is a plain memset and is timed against one.
    const long B = 8, F = 1024, T = 4096;
    const pair<double,double> band = { 400.0, 800.0 };
    const vector<long> warp_counts = { 4, 8, 16, 32 };
    const int niter_timing = 20;

    // The kernel never reads the weights, so one array serves every launch, and its contents
    // do not matter. The intensity is checked and never read; the scratch is unused.
    Array<float> intensity({B, F, T}, af_gpu | af_zero);
    Array<float> weights({B, F, T}, af_gpu | af_zero);
    Array<float> scratch({1}, af_gpu | af_zero);
    const double full = 4.0 * B * F * T;

    for (int which = 0; which < 4; which++) {
        vector<pair<double,double>> ranges = _timing_ranges(F, which, band);
        GpuBadChannelMask probe(B, F, T, ranges, band);

        // Predicted traffic: the masked rows, written once. Nothing is read but 'keep'.
        const double nbytes = 4.0 * B * T * probe.nmasked;

        cout << "\nGpuBadChannelMask::time_selected()\n"
             << "    mask = " << _timing_mask_names[which] << " (" << probe.nmasked << " of "
             << F << " channels), (B, F, T) = (" << B << ", " << F << ", " << T << ")\n"
             << "    predicted traffic = " << (nbytes / 1.0e6) << " MB of writes = "
             << (nbytes / full) << " full-array passes" << endl;

        for (long W: warp_counts) {
            GpuBadChannelMask bcm(B, F, T, ranges, band, W);
            KernelTimer kt(niter_timing, 1);
            double dt = 0.0;

            while (kt.next()) {
                bcm.launch(intensity, weights, scratch, kt.stream);
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
