#include "../../include/pirate/chimefrb/ChimeDequantizationKernel.hpp"

#include <vector>
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
// The kernel: one warp per (fine channel, 128 time samples), one lane per sample.
//
// A warp owns one fine channel 'f' and a tile of 128 consecutive time samples, and works
// through the tile in four iterations, lane L taking t = t0 + 32*j + L. Both stores are then
// 32 consecutive floats -- one full 128-byte cache line per warp per instruction, which is
// what makes a store-bound kernel run at memory bandwidth. The data loads are 32 consecutive
// BYTES per instruction (a quarter of a line), which is not ideal, but 'data' is one ninth
// of the traffic and the line is fetched once regardless.
//
// The (scale, offset) lookup and the mask byte are read redundantly: within one iteration the
// warp's 32 lanes span only 2 distinct (scale, offset) pairs and 4 distinct mask bytes. Those
// two arrays are together 0.4% of the traffic and the duplicate requests hit L1, so a shuffle
// broadcast would buy nothing measurable.
//
// The tail of a row is handled per lane, so nt need not be a multiple of anything (beyond the
// nt = 16*nt_coarse the constructor requires). Nothing is templated and nothing synchronizes
// or shuffles, so the warp count is a runtime parameter taken from blockDim.y, and lanes may
// return independently -- the rule in WiDownsamplingKernel.cu.

__global__ void __launch_bounds__(1024)
chime_dequantize_kernel(float *intensity, float *weights,
                        const float *scales, const float *offsets,
                        const uint8_t *data, const uint8_t *rfi_mask,
                        long nfreq, long nt, long nt_coarse, long i_fstride, long w_fstride,
                        int nupfreq, int fdiv, bool apply_rfimask, float scale, long nwarps)
{
    // Decompose the warp index into (f, t_tile): one integer division per warp.
    long g = long(blockIdx.x) * blockDim.y + threadIdx.y;

    if (g >= nwarps)
        return;

    const long ntile_t = (nt + 127) >> 7;
    const long t_tile = g % ntile_t;
    const long f = g / ntile_t;
    const long t0 = t_tile << 7;

    // Apply per-warp offsets. All five arrays become 1-d, indexed by the tile's time samples
    // (or, for the two coarse arrays, by itc = t >> 4).
    //   scales, offsets:  (nfreq_coarse, nt_coarse) contiguous -> (nt_coarse,)
    //   data:             (nfreq, nt) contiguous -> (nt,)
    //   intensity:        (nfreq, nt) with row stride i_fstride -> (nt,) contiguous
    //   weights:          (nfreq, nt) with row stride w_fstride -> (nt,) contiguous
    //   rfi_mask:         (nrfifreq, nt/8) contiguous -> (nt/8,)
    const long ifc = f / nupfreq;
    const float *sc = scales + ifc * nt_coarse;
    const float *of = offsets + ifc * nt_coarse;
    const uint8_t *src = data + f * nt;
    float *ip = intensity + f * i_fstride;
    float *wp = weights + f * w_fstride;
    const uint8_t *mrow = apply_rfimask ? (rfi_mask + (f / fdiv) * (nt >> 3)) : nullptr;

    for (int j = 0; j < 4; j++) {
        const long t = t0 + (j << 5) + long(threadIdx.x);

        // A lane's t only grows with j, so a lane that is past the end stays past it.
        if (t >= nt)
            break;

        // nt_per_packet == 16 (the constructor's rule), so itc is t >> 4.
        const long itc = t >> 4;
        const uint32_t u = src[t];

        // The scale is applied to the two coefficients FIRST, each product rounded once
        // (__fmul_rn is never contracted), and then ONE fused multiply-add, as the CPU
        // reference's _mm256_fmadd_ps is: written as fmaf() rather than (s*x + o) so that a
        // single rounding is not left to the compiler's contraction. This is what makes the
        // two bit-identical, and it is the operation order of ch_frb_io's decode(prescale).
        const float s = __fmul_rn(sc[itc], scale);
        const float o = __fmul_rn(of[itc], scale);
        float y = fmaf(s, float(u), o);
        float w = ((u != 0) && (u != 255)) ? 1.0f : 0.0f;

        if (apply_rfimask) {
            // SELECT, not a multiply: the reference ANDs with an all-ones/all-zeros mask, so
            // a masked sample is +0.0 whatever it held. Multiplying would give -0.0 for a
            // negative intensity, and NaN for an infinite one.
            const bool good = (mrow[t >> 3] >> (t & 7)) & 1;
            if (!good) {
                y = 0.0f;
                w = 0.0f;
            }
        }

        ip[t] = y;
        wp[t] = w;
    }
}


// -------------------------------------------------------------------------------------------------
//
// Constructor


// Validates the geometry and returns nupfreq. (One function rather than four, because most
// of the rules relate two arguments and the message should show both.)
static long _checked_geometry(long nfreq, long nt, long nfreq_coarse, long nt_coarse)
{
    stringstream ss;

    if ((nfreq < 1) || (nt < 1) || (nfreq_coarse < 1) || (nt_coarse < 1))
        ss << "expected positive (nfreq, nt, nfreq_coarse, nt_coarse), got ("
           << nfreq << ", " << nt << ", " << nfreq_coarse << ", " << nt_coarse << ")";

    else if ((nfreq % nfreq_coarse) != 0)
        ss << "nfreq (" << nfreq << ") is not a multiple of nfreq_coarse (" << nfreq_coarse
           << "), so nupfreq is not an integer";

    else if (nt != 16 * nt_coarse)
        ss << "expected nt == 16*nt_coarse, got nt=" << nt << " and nt_coarse=" << nt_coarse
           << " (nt_per_packet = " << (double(nt) / double(nt_coarse)) << "). This kernel"
           << " requires nt_per_packet == 16, as AssembledChunk's decode methods do: the"
           << " parser accepts any value, only the kernels are specialized";

    else
        return nfreq / nfreq_coarse;

    throw runtime_error("ChimeDequantizationKernel: " + ss.str());
}


static long _checked_warps(long warps_per_block)
{
    if ((warps_per_block != 4) && (warps_per_block != 8)
        && (warps_per_block != 16) && (warps_per_block != 32)) {
        stringstream ss;
        ss << "ChimeDequantizationKernel: warps_per_block=" << warps_per_block
           << " is not supported (expected 4, 8, 16 or 32)";
        throw runtime_error(ss.str());
    }
    return warps_per_block;
}


ChimeDequantizationKernel::ChimeDequantizationKernel(long nfreq_, long nt_, long nfreq_coarse_,
                                                     long nt_coarse_, long warps_per_block_) :
    nfreq(nfreq_),
    nt(nt_),
    nfreq_coarse(nfreq_coarse_),
    nt_coarse(nt_coarse_),
    nupfreq(_checked_geometry(nfreq_, nt_, nfreq_coarse_, nt_coarse_)),
    warps_per_block(_checked_warps(warps_per_block_))
{ }


// -------------------------------------------------------------------------------------------------
//
// launch()


// Checks one output array. Written out rather than xassert-ed because the two outputs go
// through the same code: an xassert would report a line number that does not say which array
// was wrong.
static void _check_output(const Array<float> &arr, const char *name, long nfreq, long nt)
{
    stringstream ss;

    if ((arr.ndim != 2) || (arr.shape[0] != nfreq) || (arr.shape[1] != nt))
        ss << name << ": expected shape (" << nfreq << ", " << nt << "), got " << arr.shape_str();

    else if (arr.strides[1] != 1)
        ss << name << ": expected a contiguous time axis (stride 1), got stride "
           << arr.strides[1];

    // The frequency stride is otherwise free -- that is the point of the array being only
    // partially contiguous -- but a stride below nt would make consecutive rows overlap, and
    // the kernel's warps would then race to write the same elements.
    else if (arr.strides[0] < nt)
        ss << name << ": frequency stride " << arr.strides[0] << " is smaller than nt=" << nt
           << ", so its rows overlap";

    else if (!arr.on_gpu())
        ss << name << ": expected an array in GPU memory";

    else
        return;

    throw runtime_error("ChimeDequantizationKernel::launch(): " + ss.str());
}


void ChimeDequantizationKernel::launch(Array<float> &intensity, Array<float> &weights,
                                       const Array<float> &scales, const Array<float> &offsets,
                                       const Array<uint8_t> &data, const Array<uint8_t> &rfi_mask,
                                       bool apply_rfimask, float scale, cudaStream_t stream) const
{
    _check_output(intensity, "intensity", nfreq, nt);
    _check_output(weights, "weights", nfreq, nt);

    // Warps run in no particular order, so one array passed twice would be a race between
    // the intensity and weights stores. (A partial overlap is the caller's responsibility --
    // there is nothing cheap to check.)
    xassert(intensity.data != weights.data);

    xassert_shape_eq(scales, ({nfreq_coarse, nt_coarse}));
    xassert_shape_eq(offsets, ({nfreq_coarse, nt_coarse}));
    xassert_shape_eq(data, ({nfreq, nt}));
    xassert(scales.is_fully_contiguous());
    xassert(offsets.is_fully_contiguous());
    xassert(data.is_fully_contiguous());
    xassert(scales.on_gpu());
    xassert(offsets.on_gpu());
    xassert(data.on_gpu());

    // 'fdiv' is the number of fine channels per mask row. The mask is unread (and may be
    // empty) when apply_rfimask is false, so nothing about it is checked in that case.
    long fdiv = 1;

    if (apply_rfimask) {
        xassert_eq(rfi_mask.ndim, 2);
        const long nrfifreq = rfi_mask.shape[0];
        xassert_gt(nrfifreq, 0);
        xassert_divisible(nfreq, nrfifreq);
        xassert_shape_eq(rfi_mask, ({nrfifreq, nt/8}));
        xassert(rfi_mask.is_fully_contiguous());
        xassert(rfi_mask.on_gpu());
        fdiv = nfreq / nrfifreq;
    }

    const long nwarps = nfreq * ((nt + 127) / 128);
    const long nblocks = (nwarps + warps_per_block - 1) / warps_per_block;
    const dim3 nthreads(32, warps_per_block);

    xassert_lt(nblocks, 1L << 31);

    chime_dequantize_kernel <<< nblocks, nthreads, 0, stream >>>
        (intensity.data, weights.data, scales.data, offsets.data, data.data, rfi_mask.data,
         nfreq, nt, nt_coarse, intensity.strides[0], weights.strides[0],
         int(nupfreq), int(fdiv), apply_rfimask, scale, nwarps);

    CUDA_PEEK("chime_dequantize_kernel");
}


// -------------------------------------------------------------------------------------------------
//
// time_selected()


void ChimeDequantizationKernel::time_selected()
{
    // The production CHIME geometry: one AssembledChunk, 16384 fine channels from 1024 coarse
    // ones, 1024 time samples, and an RFI mask at the coarse resolution. The outputs are
    // 134 MB, comfortably past an L40S's 96 MB L2, so the numbers are DRAM bandwidth.
    const long nfreq = 16384, nt = 1024, nfreq_coarse = 1024, nt_coarse = 64, nrfifreq = 1024;

    // The real call writes a chunk into one quarter of a pipeline block's time axis, so its
    // outputs have frequency stride 4096 rather than 1024. Timed alongside the contiguous
    // case, since the whole reason the kernel takes a stride is to support it.
    const long block_nt = 4096;

    const vector<long> warp_counts = { 4, 8, 16, 32 };
    const int niter_timing = 20;

    Array<uint8_t> data({nfreq, nt}, af_gpu | af_zero);
    Array<float> scales({nfreq_coarse, nt_coarse}, af_gpu | af_zero);
    Array<float> offsets({nfreq_coarse, nt_coarse}, af_gpu | af_zero);
    Array<uint8_t> rfi_mask({nrfifreq, nt/8}, af_gpu | af_zero);

    Array<float> out_i({nfreq, nt}, af_gpu | af_zero);
    Array<float> out_w({nfreq, nt}, af_gpu | af_zero);
    Array<float> block_i({nfreq, block_nt}, af_gpu | af_zero);
    Array<float> block_w({nfreq, block_nt}, af_gpu | af_zero);

    // The kernel's whole traffic: data in, the two float32 arrays out, plus the coarse arrays
    // and (when applied) the mask.
    const double out_bytes = 8.0 * nfreq * nt;
    const double coarse_bytes = 8.0 * nfreq_coarse * nt_coarse;
    const double mask_bytes = double(nrfifreq) * (nt/8);

    for (int im = 0; im < 2; im++) {
        const bool apply_rfimask = (im != 0);
        const double nbytes = out_bytes + double(nfreq)*nt + coarse_bytes
                              + (apply_rfimask ? mask_bytes : 0.0);

        cout << "\nChimeDequantizationKernel::time_selected()\n"
             << "    (nfreq, nt) = (" << nfreq << ", " << nt << "), (nfreq_coarse, nt_coarse) = ("
             << nfreq_coarse << ", " << nt_coarse << "), nrfifreq = " << nrfifreq
             << ", apply_rfimask = " << (apply_rfimask ? "true" : "false") << "\n"
             << "    traffic = " << (nbytes / 1.0e9) << " GB per chunk" << endl;

        for (long W: warp_counts) {
            ChimeDequantizationKernel k(nfreq, nt, nfreq_coarse, nt_coarse, W);
            KernelTimer kt(niter_timing, 1);
            double dt = 0.0;

            while (kt.next()) {
                k.launch(out_i, out_w, scales, offsets, data, rfi_mask, apply_rfimask, 1.0f, kt.stream);
                if (kt.warmed_up)
                    dt = kt.dt;
            }

            cout << "    warps_per_block = " << W
                 << ":  dt = " << (dt * 1.0e3) << " ms"
                 << ",  bandwidth = " << (nbytes / dt / 1.0e9) << " GB/s" << endl;
        }

        // The same launch, writing into a column slice of a (nfreq, 4096) pipeline block.
        {
            Array<float> slice_i = block_i.slice(1, 0, nt);
            Array<float> slice_w = block_w.slice(1, 0, nt);
            ChimeDequantizationKernel k(nfreq, nt, nfreq_coarse, nt_coarse);
            KernelTimer kt(niter_timing, 1);
            double dt = 0.0;

            while (kt.next()) {
                k.launch(slice_i, slice_w, scales, offsets, data, rfi_mask, apply_rfimask, 1.0f, kt.stream);
                if (kt.warmed_up)
                    dt = kt.dt;
            }

            cout << "    into a (nfreq, " << block_nt << ") block (frequency stride "
                 << block_nt << "), warps_per_block = " << k.warps_per_block
                 << ":  dt = " << (dt * 1.0e3) << " ms"
                 << ",  bandwidth = " << (nbytes / dt / 1.0e9) << " GB/s" << endl;
        }
    }

    // The ceiling: this kernel is store-bound, so a memset of the two output arrays is the
    // fastest anything writing them could possibly be.
    {
        KernelTimer kt(niter_timing, 1);
        double dt = 0.0;

        while (kt.next()) {
            CUDA_CALL(cudaMemsetAsync(out_i.data, 0, size_t(4L*nfreq*nt), kt.stream));
            CUDA_CALL(cudaMemsetAsync(out_w.data, 0, size_t(4L*nfreq*nt), kt.stream));
            if (kt.warmed_up)
                dt = kt.dt;
        }

        cout << "    cudaMemsetAsync of both output arrays, for comparison:  dt = "
             << (dt * 1.0e3) << " ms,  bandwidth = " << (out_bytes / dt / 1.0e9) << " GB/s"
             << endl;
    }
}


}}  // namespace pirate::chimefrb
