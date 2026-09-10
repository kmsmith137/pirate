#include "../../include/pirate/chimefrb/WiDownsampler.hpp"

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
// The kernel.
//
// One threadblock produces a 32-by-32 tile of output cells, and one thread walks the
// (Df,Dt) input block behind each of its cells. WARPS says how many warps share a tile,
// so each thread owns CELLS = 32/WARPS of them; it is a pure occupancy knob, and
// GpuWiDownsampler::time_selected() is what decides the default.
//
// The transposed and untransposed cases differ only in the last few lines: 'Transpose'
// stages the finished tile through shared memory so that the global write stays
// coalesced. When Transpose=false the shared arrays do not exist.


template<bool Transpose, int WARPS>
__global__ void __launch_bounds__(32*WARPS)
wi_downsample_kernel(float *out_i, float *out_w,
                     const float *in_i, const float *in_w,
                     int Df, int Dt, long F, long T, long F_ds, long T_ds)
{
    static_assert((WARPS >= 1) && (WARPS <= 32) && ((32 % WARPS) == 0));
    constexpr int CELLS = 32 / WARPS;

    // Within a tile, 'tx' is the time index and (ty + k*WARPS) is the frequency index,
    // both in units of output cells. Note that tx is the lane index, so a warp covers 32
    // consecutive output time samples, which is what makes the reads and writes below
    // cache-line-aligned when Dt=1.
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const long f0 = long(blockIdx.y) * 32;   // tile origin, in output cells
    const long t0 = long(blockIdx.x) * 32;
    const long b = blockIdx.z;

    // Apply per-block pointer offset to the input arrays.
    //   before: shape (B, F, T), contiguous
    //   after: shape (32*Df, 32*Dt), strides (T, 1)
    in_i += b*F*T + (f0*Df)*T + (t0*Dt);
    in_w += b*F*T + (f0*Df)*T + (t0*Dt);

    // Apply per-block pointer offset to the output arrays. In both cases the result is a
    // shape-(32,32) subarray, but the axis order differs:
    //   !Transpose: before shape (B, F_ds, T_ds); after [f_local][t_local], strides (T_ds, 1)
    //    Transpose: before shape (B, T_ds, F_ds); after [t_local][f_local], strides (F_ds, 1)
    if constexpr (Transpose) {
        out_i += b*F_ds*T_ds + t0*F_ds + f0;
        out_w += b*F_ds*T_ds + t0*F_ds + f0;
    }
    else {
        out_i += b*F_ds*T_ds + f0*T_ds + t0;
        out_w += b*F_ds*T_ds + f0*T_ds + t0;
    }

    // Reduce: one (Df,Dt) block per cell.
    //
    // Frequency is the outer loop and time is the inner loop, deliberately. The inner
    // loop then reads Dt consecutive floats, which can vectorize; the other order strides
    // by T and cannot. It also means the warp finishes one input row before touching the
    // next, so its live L1 footprint is one row-segment (32*Dt floats) rather than all Df
    // of them at once.

    float vi[CELLS];
    float vw[CELLS];

    #pragma unroll
    for (int k = 0; k < CELLS; k++) {
        const int f_local = ty + k*WARPS;
        const float *ip = in_i + long(f_local*Df)*T + long(tx)*Dt;
        const float *wp = in_w + long(f_local*Df)*T + long(tx)*Dt;

        float wsum = 0.0f;
        float wisum = 0.0f;

        for (int df = 0; df < Df; df++) {
            for (int dt = 0; dt < Dt; dt++) {
                float w = wp[dt];
                wsum += w;
                wisum += w * ip[dt];
            }
            ip += T;
            wp += T;
        }

        // Guarded divide, following rf_kernels: a cell with no weight gets intensity zero
        // rather than a NaN. Note that this is an exact test, not a threshold, since a sum
        // of nonnegative floats is zero iff every term is zero.
        vw[k] = wsum;
        vi[k] = (wsum > 0.0f) ? (wisum / wsum) : 0.0f;
    }

    if constexpr (Transpose) {
        // The 33 is the usual padding against shared-memory bank conflicts: it makes both
        // the row-wise write and the column-wise read below hit 32 distinct banks.
        __shared__ float tile_i[32][33];
        __shared__ float tile_w[32][33];

        #pragma unroll
        for (int k = 0; k < CELLS; k++) {
            const int f_local = ty + k*WARPS;
            tile_i[f_local][tx] = vi[k];
            tile_w[f_local][tx] = vw[k];
        }

        __syncthreads();

        // Now each thread writes the output cells with f_local = tx, so that consecutive
        // lanes write consecutive addresses along the (contiguous) frequency axis.
        #pragma unroll
        for (int k = 0; k < CELLS; k++) {
            const int t_local = ty + k*WARPS;
            out_i[t_local*F_ds + tx] = tile_i[tx][t_local];
            out_w[t_local*F_ds + tx] = tile_w[tx][t_local];
        }
    }
    else {
        #pragma unroll
        for (int k = 0; k < CELLS; k++) {
            const int f_local = ty + k*WARPS;
            out_i[f_local*T_ds + tx] = vi[k];
            out_w[f_local*T_ds + tx] = vw[k];
        }
    }
}


// -------------------------------------------------------------------------------------------------


GpuWiDownsampler::GpuWiDownsampler(long Df_, long Dt_, bool transpose_, long warps_per_block_) :
    Df(Df_), Dt(Dt_), transpose(transpose_), warps_per_block(warps_per_block_)
{
    if (Df < 1)
        throw runtime_error("GpuWiDownsampler: expected Df >= 1");
    if (Dt < 1)
        throw runtime_error("GpuWiDownsampler: expected Dt >= 1");

    if ((warps_per_block != 4) && (warps_per_block != 8)
        && (warps_per_block != 16) && (warps_per_block != 32)) {
        stringstream ss;
        ss << "GpuWiDownsampler: warps_per_block=" << warps_per_block
           << " is not supported (expected 4, 8, 16 or 32)";
        throw runtime_error(ss.str());
    }

    if ((Df == 1) && (Dt == 1) && !transpose)
        throw runtime_error("GpuWiDownsampler: (Df,Dt,transpose)=(1,1,false) is the identity."
                            " Use the input array directly, rather than paying for a copy.");
}


template<bool Transpose, int WARPS>
static void _launch(float *out_i, float *out_w, const float *in_i, const float *in_w,
                    long Df, long Dt, long B, long F, long T, cudaStream_t stream)
{
    long F_ds = F / Df;
    long T_ds = T / Dt;

    dim3 nblocks(T_ds/32, F_ds/32, B);
    dim3 nthreads(32, WARPS);

    wi_downsample_kernel<Transpose,WARPS> <<< nblocks, nthreads, 0, stream >>>
        (out_i, out_w, in_i, in_w, int(Df), int(Dt), F, T, F_ds, T_ds);

    CUDA_PEEK("wi_downsample_kernel");
}


void GpuWiDownsampler::launch(Array<float> &out_i, Array<float> &out_w,
                              const Array<float> &in_i, const Array<float> &in_w,
                              cudaStream_t stream) const
{
    xassert_eq(in_i.ndim, 3);

    long B = in_i.shape[0];
    long F = in_i.shape[1];
    long T = in_i.shape[2];

    xassert_gt(B, 0);
    xassert_shape_eq(in_w, ({B, F, T}));

    // The 32 is the output tile size: one threadblock produces a 32-by-32 tile of output
    // cells, and the kernel has no edge predication.
    if ((F % (32*Df)) || (T % (32*Dt))) {
        stringstream ss;
        ss << "GpuWiDownsampler::launch(): expected F divisible by 32*Df and T divisible"
           << " by 32*Dt, got (F,T)=(" << F << "," << T << ") with (Df,Dt)=("
           << Df << "," << Dt << ")";
        throw runtime_error(ss.str());
    }

    long F_ds = F / Df;
    long T_ds = T / Dt;

    if (transpose) {
        xassert_shape_eq(out_i, ({B, T_ds, F_ds}));
        xassert_shape_eq(out_w, ({B, T_ds, F_ds}));
    }
    else {
        xassert_shape_eq(out_i, ({B, F_ds, T_ds}));
        xassert_shape_eq(out_w, ({B, F_ds, T_ds}));
    }

    const std::initializer_list<const Array<float> *> arrays =
        { &out_i, &out_w, &in_i, &in_w };

    for (const Array<float> *a: arrays) {
        xassert(a->is_fully_contiguous());
        xassert(a->on_gpu());
    }

    // The kernel reads each input cell after writing no output, but only within a
    // threadblock; across blocks there is no ordering, so overlapping input and output is
    // a race. (1,1,transpose=true) is the tempting case: it looks like an in-place
    // transpose, and is not one.
    xassert(out_i.data != in_i.data);
    xassert(out_w.data != in_w.data);
    xassert(out_i.data != in_w.data);
    xassert(out_w.data != in_i.data);

    if (transpose) {
        switch (warps_per_block) {
            case 4:  _launch<true,4>  (out_i.data, out_w.data, in_i.data, in_w.data, Df, Dt, B, F, T, stream); return;
            case 8:  _launch<true,8>  (out_i.data, out_w.data, in_i.data, in_w.data, Df, Dt, B, F, T, stream); return;
            case 16: _launch<true,16> (out_i.data, out_w.data, in_i.data, in_w.data, Df, Dt, B, F, T, stream); return;
            case 32: _launch<true,32> (out_i.data, out_w.data, in_i.data, in_w.data, Df, Dt, B, F, T, stream); return;
        }
    }
    else {
        switch (warps_per_block) {
            case 4:  _launch<false,4>  (out_i.data, out_w.data, in_i.data, in_w.data, Df, Dt, B, F, T, stream); return;
            case 8:  _launch<false,8>  (out_i.data, out_w.data, in_i.data, in_w.data, Df, Dt, B, F, T, stream); return;
            case 16: _launch<false,16> (out_i.data, out_w.data, in_i.data, in_w.data, Df, Dt, B, F, T, stream); return;
            case 32: _launch<false,32> (out_i.data, out_w.data, in_i.data, in_w.data, Df, Dt, B, F, T, stream); return;
        }
    }

    throw runtime_error("GpuWiDownsampler::launch(): internal error, unhandled warps_per_block");
}


// -------------------------------------------------------------------------------------------------


// One row of time_selected(): the (Df, Dt, transpose) configuration, and an array
// geometry chosen so that the timing is not dominated by launch overhead.
struct TimingConfig
{
    long Df;
    long Dt;
    bool transpose;
    long B;
    long F;
    long T;
    const char *what;   // where this configuration comes up in the old RFI chain
};


void GpuWiDownsampler::time_selected()
{
    // The four configurations used by the old search's production RFI config
    // (misc/chimefrb/configs/21-03-07-low-latency-uniform-badchannel-mask-noplot.json).
    // Beam counts differ so that every row moves a comparable number of bytes.
    const vector<TimingConfig> configs = {
        { 16, 1,  false, 1, 16384, 4096, "16K -> 1K sub-pipeline downsample" },
        { 1,  1,  true,  8, 1024,  4096, "AXIS_FREQ clipper statistic (pure transpose)" },
        { 2,  16, true,  8, 1024,  4096, "AXIS_FREQ clipper statistic, (Df,Dt)=(2,16)" },
        { 2,  16, false, 8, 1024,  4096, "AXIS_NONE clipper, and AXIS_FREQ mask application" },
    };

    const vector<long> warp_counts = { 4, 8, 16, 32 };
    const int niter = 30;

    for (const TimingConfig &c: configs) {
        long F_ds = c.F / c.Df;
        long T_ds = c.T / c.Dt;

        Array<float> in_i({c.B, c.F, c.T}, af_gpu | af_zero);
        Array<float> in_w({c.B, c.F, c.T}, af_gpu | af_zero);

        vector<long> oshape = c.transpose ? vector<long>{c.B, T_ds, F_ds}
                                          : vector<long>{c.B, F_ds, T_ds};
        Array<float> out_i(oshape, af_gpu | af_zero);
        Array<float> out_w(oshape, af_gpu | af_zero);

        // Global memory traffic: both inputs are read once and both outputs are written
        // once, so this is also the DRAM traffic of an ideal implementation, and
        // (time -> bandwidth) is the figure of merit.
        double nbytes = 4.0 * (2.0*c.B*c.F*c.T + 2.0*c.B*F_ds*T_ds);

        cout << "\nGpuWiDownsampler::time_selected()\n"
             << "    (Df, Dt, transpose) = (" << c.Df << ", " << c.Dt << ", "
             << (c.transpose ? "true" : "false") << "):  " << c.what << "\n"
             << "    (B, F, T) = (" << c.B << ", " << c.F << ", " << c.T << ")"
             << " -> (" << c.B << ", " << (c.transpose ? T_ds : F_ds) << ", "
             << (c.transpose ? F_ds : T_ds) << ")\n"
             << "    global memory traffic per launch = " << (nbytes / 1.0e9) << " GB"
             << endl;

        for (long W: warp_counts) {
            GpuWiDownsampler ds(c.Df, c.Dt, c.transpose, W);
            KernelTimer kt(niter, 1);
            double dt = 0.0;

            while (kt.next()) {
                ds.launch(out_i, out_w, in_i, in_w, kt.stream);
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
