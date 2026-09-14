#include "../../include/pirate/chimefrb/RfiMaskPackingKernel.hpp"

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
// The kernel: one warp per (frequency channel, 1024 time samples), one lane per sample.
//
// The packing is __ballot_sync(), and it is not a trick that happens to work -- the ballot's
// bit N is lane N's predicate, so if lane L reads sample (t0 + 32j + L), the ballot IS the
// LSB-first packed word for those 32 samples. No shifts, no masks, no OR-reduction. Storing
// it puts bits 0-7 in the low byte, so byte k of the store is samples (t0+32j+8k .. +7) with
// the earliest in the LSB: exactly the file format. (That last step assumes a little-endian
// store, which CUDA is; the unit test's bit-order check is what would catch it if not.)
//
// WHY THE TILE IS 1024 SAMPLES, i.e. why the constructor requires nt % 1024 == 0. 1024
// samples is precisely the amount of input whose packed output is 32 uint32 = 128 bytes = one
// full cache line. Keeping ballot j on lane j leaves the warp holding the tile's 32 output
// words one per lane, in order, so the whole tile leaves in ONE cache-line-aligned store. A
// 512-sample tile would store half a line from half the warp; a 2048-sample tile would need
// two accumulators for no gain.
//
// The loads are 128 contiguous bytes per warp per instruction -- one full cache line if the
// row happens to start on a 128-byte boundary, two partial ones otherwise. We do not require
// the alignment (an arbitrary frequency stride is the point of the argument), and it costs
// little: consecutive iterations share the straddled line, so the extra DRAM traffic is about
// one line per row rather than a doubling. time_selected() measures it.
//
// Nothing is templated and nothing uses shared memory, so the warp count is a runtime
// parameter taken from blockDim.y. The early return is per-WARP, not per-lane, which is what
// makes the full-warp mask in __ballot_sync() legitimate.

__global__ void __launch_bounds__(1024)
rfi_mask_packing_kernel(uint8_t *rfi_mask, const float *weights,
                          long nt, long w_fstride, long nwarps)
{
    // Decompose the warp index into (f, t_tile): one integer division per warp.
    long g = long(blockIdx.x) * blockDim.y + threadIdx.y;

    if (g >= nwarps)
        return;

    const long ntile_t = nt >> 10;
    const long t_tile = g % ntile_t;
    const long f = g / ntile_t;
    const long t0 = t_tile << 10;

    // Apply per-warp offsets. Both arrays become 1-d and contiguous, indexed by the tile's
    // 1024 time samples (or, for the output, by the 32 words those samples pack into).
    //   weights:   (nfreq, nt) with row stride w_fstride -> (1024,) contiguous
    //   rfi_mask:  (nfreq, nt/8) contiguous -> (128,) bytes, viewed as (32,) uint32
    //
    // The uint32 view is why the output has to be fully contiguous: nt is a multiple of 1024,
    // so both f*(nt/8) and t0/8 are multiples of 128, and the cast is aligned.
    const float *src = weights + f * w_fstride + t0;
    uint32_t *dst = reinterpret_cast<uint32_t *> (rfi_mask + f * (nt >> 3) + (t0 >> 3));

    const int lane = threadIdx.x;
    uint32_t acc = 0;

    // Unrolled by 8 for memory-level parallelism: the body is load -> compare -> ballot, a
    // dependent chain, so 8 loads in flight per warp is what the unroll buys. The ballots
    // stay in order (they are convergent); only the loads move.
    #pragma unroll 8
    for (int j = 0; j < 32; j++) {
        const float w = src[(j << 5) + lane];
        const uint32_t b = __ballot_sync(0xffffffffU, w > 0.0f);

        // A select, not a branch: after the unroll 'j' is a constant, and every lane keeps
        // the one ballot it will store.
        if (lane == j)
            acc = b;
    }

    dst[lane] = acc;
}


// -------------------------------------------------------------------------------------------------
//
// Constructor


// Validates the geometry and returns nt. (One function rather than two, so that a message can
// name both arguments.)
static long _checked_geometry(long nfreq, long nt)
{
    stringstream ss;

    if ((nfreq < 1) || (nt < 1))
        ss << "expected positive (nfreq, nt), got (" << nfreq << ", " << nt << ")";

    else if ((nt % 1024) != 0)
        ss << "nt (" << nt << ") must be a multiple of 1024. This is the kernel's tiling, not"
           << " a property of the data: one warp packs 1024 time samples into the 128 bytes"
           << " that are exactly one cache line. Every chimefrb config runs the mask counter"
           << " at nt_chunk = 1024";

    else
        return nt;

    throw runtime_error("RfiMaskPackingKernel: " + ss.str());
}


static long _checked_warps(long warps_per_block)
{
    if ((warps_per_block != 4) && (warps_per_block != 8)
        && (warps_per_block != 16) && (warps_per_block != 32)) {
        stringstream ss;
        ss << "RfiMaskPackingKernel: warps_per_block=" << warps_per_block
           << " is not supported (expected 4, 8, 16 or 32)";
        throw runtime_error(ss.str());
    }
    return warps_per_block;
}


RfiMaskPackingKernel::RfiMaskPackingKernel(long nfreq_, long nt_, long warps_per_block_) :
    nfreq(nfreq_),
    nt(_checked_geometry(nfreq_, nt_)),
    warps_per_block(_checked_warps(warps_per_block_))
{ }


// -------------------------------------------------------------------------------------------------
//
// launch()


// Checks the input array. Written out rather than xassert-ed: a caller who hands over a
// transposed or overlapping array should be told which rule it broke, not a file and a line.
static void _check_weights(const Array<float> &weights, long nfreq, long nt)
{
    stringstream ss;

    if ((weights.ndim != 2) || (weights.shape[0] != nfreq) || (weights.shape[1] != nt))
        ss << "weights: expected shape (" << nfreq << ", " << nt << "), got "
           << weights.shape_str();

    else if (weights.strides[1] != 1)
        ss << "weights: expected a contiguous time axis (stride 1), got stride "
           << weights.strides[1];

    // The frequency stride is otherwise free -- that is the point of the array being only
    // partially contiguous -- but a stride below nt would make consecutive rows overlap, and
    // the kernel's warps would then read data another warp's row claims. Belt and braces:
    // ksgpu::Array refuses an overlapping array at construction (Array.cpp:172), so in
    // practice a caller hits that first, and this branch is unreachable through pybind11.
    else if (weights.strides[0] < nt)
        ss << "weights: frequency stride " << weights.strides[0] << " is smaller than nt="
           << nt << ", so its rows overlap";

    else
        return;

    throw runtime_error("RfiMaskPackingKernel::launch(): " + ss.str());
}


void RfiMaskPackingKernel::launch(Array<uint8_t> &rfi_mask, const Array<float> &weights,
                                    cudaStream_t stream) const
{
    xassert_shape_eq(rfi_mask, ({nfreq, nt/8}));
    xassert(rfi_mask.is_fully_contiguous());
    xassert(rfi_mask.on_gpu());
    xassert(weights.on_gpu());

    _check_weights(weights, nfreq, nt);

    // Warps run in no particular order, so a caller who packed a mask on top of its own
    // weights would be racing the reads against the writes.
    xassert((const void *) rfi_mask.data != (const void *) weights.data);

    const long nwarps = nfreq * (nt / 1024);
    const long nblocks = (nwarps + warps_per_block - 1) / warps_per_block;
    const dim3 nthreads(32, warps_per_block);

    xassert_lt(nblocks, 1L << 31);

    rfi_mask_packing_kernel <<< nblocks, nthreads, 0, stream >>>
        (rfi_mask.data, weights.data, nt, weights.strides[0], nwarps);

    CUDA_PEEK("rfi_mask_packing_kernel");
}


// -------------------------------------------------------------------------------------------------
//
// time_selected()


// The ceiling. This kernel reads 32 bytes for every byte it writes, so the reference is READ
// bandwidth -- not the memset that a store-bound kernel would compare against. Reads with
// float4 (the widest instruction) and reduces, so that no load can be optimized away.
__global__ void __launch_bounds__(256)
_read_bandwidth_kernel(float *out, const float4 *in, long n4)
{
    const long stride = long(gridDim.x) * blockDim.x;
    float s = 0.0f;

    for (long i = long(blockIdx.x) * blockDim.x + threadIdx.x; i < n4; i += stride) {
        float4 x = in[i];
        s += x.x + x.y + x.z + x.w;
    }

    // One atomic per warp, so that EVERY lane's loads are provably live. (Writing from lane 0
    // alone would leave the compiler free, in principle, to sink the loop into the branch.)
    for (int d = 16; d > 0; d >>= 1)
        s += __shfl_down_sync(0xffffffffU, s, d);

    if ((threadIdx.x & 31) == 0)
        atomicAdd(out, s);
}


// One timed case: pack 'weights' (which may be a strided view) into a fresh contiguous mask.
// 'what' labels the row; 'nbytes' is the traffic the bandwidth is computed from.
static void _time_one(const char *what, long nfreq, long nt, long warps_per_block,
                      const Array<float> &weights, Array<uint8_t> &rfi_mask, double nbytes)
{
    const int niter_timing = 20;

    RfiMaskPackingKernel k(nfreq, nt, warps_per_block);
    KernelTimer kt(niter_timing, 1);
    double dt = 0.0;

    while (kt.next()) {
        k.launch(rfi_mask, weights, kt.stream);
        if (kt.warmed_up)
            dt = kt.dt;
    }

    cout << "    " << what << ":  dt = " << (dt * 1.0e3) << " ms"
         << ",  bandwidth = " << (nbytes / dt / 1.0e9) << " GB/s" << endl;
}


void RfiMaskPackingKernel::time_selected()
{
    // Three geometries. The first is the real one -- the production chain runs its mask
    // counter inside a wi_sub_pipeline with Df=16, so nfreq is 16384/16 = 1024 (which is also
    // why real files have nrfifreq = 1024), at nt_chunk = 1024.
    //
    // READ THE FIRST TWO AS LATENCIES, NOT AS BANDWIDTHS. Their inputs are 4 MB and 16 MB,
    // well inside an L40S's 96 MB L2, so the reported GB/s is an L2 number several times
    // DRAM speed. That is not cheating -- in the real pipeline the chain has just written
    // these weights, so they plausibly ARE in L2 -- but only the third geometry, whose 256 MB
    // input cannot fit, measures the kernel against memory.
    struct Geom { long nfreq, nt; const char *what; };
    const vector<Geom> geoms = {
        { 1024, 1024, "production, one chunk (4 MB in, L2-resident: read dt, not GB/s)" },
        { 1024, 4096, "one pipeline block (16 MB in, L2-resident: read dt, not GB/s)" },
        { 16384, 4096, "large (256 MB in, exceeds L2: this one is DRAM bandwidth)" },
    };

    const vector<long> warp_counts = { 4, 8, 16, 32 };

    for (const Geom &g: geoms) {
        const double nbytes = 4.0 * g.nfreq * g.nt + double(g.nfreq) * (g.nt / 8);

        cout << "\nRfiMaskPackingKernel::time_selected()\n"
             << "    (nfreq, nt) = (" << g.nfreq << ", " << g.nt << "), " << g.what
             << ", traffic = " << (nbytes / 1.0e9) << " GB" << endl;

        Array<float> weights({g.nfreq, g.nt}, af_gpu | af_zero);
        Array<uint8_t> rfi_mask({g.nfreq, g.nt/8}, af_gpu | af_zero);

        for (long W: warp_counts) {
            stringstream ss;
            ss << "warps_per_block = " << W;
            _time_one(ss.str().c_str(), g.nfreq, g.nt, W, weights, rfi_mask, nbytes);
        }
    }

    // Packing out of a column window of a wider pipeline block, which is the production call
    // shape. Timed at an aligned column offset and at an odd one, since the kernel accepts
    // any frequency stride and the alignment note at the top of this file predicts the odd
    // one to be a little slower. Done at the production size (the real call) AND at the large
    // one, because only the latter is limited by memory, so only there can the alignment
    // effect show up at all.
    for (const Geom &g: { Geom{1024, 1024, ""}, Geom{16384, 4096, ""} }) {
        const long block_nt = g.nt + 128;   // a multiple of 32, so offset 0 keeps rows aligned
        const double nbytes = 4.0 * g.nfreq * g.nt + double(g.nfreq) * (g.nt / 8);

        Array<float> block({g.nfreq, block_nt}, af_gpu | af_zero);
        Array<uint8_t> rfi_mask({g.nfreq, g.nt/8}, af_gpu | af_zero);

        cout << "\n    (nfreq, nt) = (" << g.nfreq << ", " << g.nt << ") out of a (" << g.nfreq
             << ", " << block_nt << ") block (frequency stride " << block_nt
             << "), traffic = " << (nbytes / 1.0e9) << " GB" << endl;

        for (long off: { 0L, 1L }) {
            Array<float> slice = block.slice(1, off, off + g.nt);
            stringstream ss;
            ss << "column offset " << off << (off == 0 ? " (row-aligned loads)"
                                                       : " (loads straddle cache lines)");
            _time_one(ss.str().c_str(), g.nfreq, g.nt, 32, slice, rfi_mask, nbytes);
        }
    }

    // The ceiling, at the largest geometry: how fast the input can be read at all.
    {
        const long nfreq = 16384, nt = 4096;
        const long n4 = (nfreq * nt) / 4;
        const double nbytes = 4.0 * nfreq * nt;
        const int niter_timing = 20;

        Array<float> weights({nfreq, nt}, af_gpu | af_zero);
        Array<float> out({1}, af_gpu | af_zero);

        KernelTimer kt(niter_timing, 1);
        double dt = 0.0;

        while (kt.next()) {
            _read_bandwidth_kernel <<< 4096, 256, 0, kt.stream >>>
                (out.data, reinterpret_cast<const float4 *> (weights.data), n4);
            CUDA_PEEK("_read_bandwidth_kernel");
            if (kt.warmed_up)
                dt = kt.dt;
        }

        cout << "\n    read-only ceiling at (16384, 4096), for comparison:  dt = "
             << (dt * 1.0e3) << " ms,  bandwidth = " << (nbytes / dt / 1.0e9) << " GB/s"
             << endl;
    }
}


}}  // namespace pirate::chimefrb
