#include "../../include/pirate/chimefrb/IntensityClipper.hpp"
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
    // single cache line. The branch itself is uniform across the whole grid.
    long r;
    if (axis == int(ClipperAxis::TIME))
        r = b*F_ds + f_ds;
    else if (axis == int(ClipperAxis::FREQ))
        r = b*T_ds + t_ds;
    else
        r = b;

    // Note sigma here, not iter_sigma: the statistic's refinements used a different
    // threshold, and this is the final clip. Note also that var == 0 (a row with no usable
    // statistic) gives thresh == 0, and wrms_survives()'s strict '<' then masks the whole
    // row -- which is the intended behavior, not an edge case to guard against.
    const float thresh = sigma * sqrtf(var[r]);

    if (wrms_survives(ival - mean[r], thresh))
        return;

    // Apply per-warp/thread pointer offset to the full-resolution weights.
    //   before: shape (B, F_ds*Df, T_ds*Dt), contiguous
    //   after: shape (Df, Dt), strides (T_ds*Dt, 1)
    //
    // The store is 32-bit and, for Dt > 1, strided: the 32 lanes are Dt*4 bytes apart, so
    // one instruction touches up to 32 sectors rather than one cache line. Deliberate.
    // It runs only for masked cells, and when a whole bad channel is masked the warp's
    // Dt stores together cover 32*Dt*4 contiguous bytes anyway. GpuWiDownsampler reads
    // through exactly this pattern at 657 GB/s.
    const long T = T_ds * Dt;
    float *wp = weights + ((b*F_ds + f_ds)*Df)*T + t_ds*Dt;

    for (int df = 0; df < Df; df++) {
        for (int dt = 0; dt < Dt; dt++)
            wp[dt] = 0.0f;
        wp += T;
    }
}


// -------------------------------------------------------------------------------------------------
//
// Constructor helpers. These run from the initializer list, so that argument checking
// happens before the derived members (which divide by Df and Dt) are computed.


// Returns its first argument, so that it can initialize 'B'.
static long _checked_B(long B, long F, long nt_chunk, double sigma, long Df, long Dt,
                       long niter, double iter_sigma, long warps_per_block)
{
    if (B < 1)
        throw runtime_error("GpuIntensityClipper: expected B >= 1");
    if (Df < 1)
        throw runtime_error("GpuIntensityClipper: expected Df >= 1");
    if (Dt < 1)
        throw runtime_error("GpuIntensityClipper: expected Dt >= 1");
    if (niter < 1)
        throw runtime_error("GpuIntensityClipper: expected niter >= 1 (niter counts total"
                            " passes, so niter=1 means no refinement)");
    if (sigma < 0.0)
        throw runtime_error("GpuIntensityClipper: expected sigma >= 0");
    if (iter_sigma < 0.0)
        throw runtime_error("GpuIntensityClipper: expected iter_sigma >= 0");

    if ((warps_per_block != 4) && (warps_per_block != 8)
        && (warps_per_block != 16) && (warps_per_block != 32)) {
        stringstream ss;
        ss << "GpuIntensityClipper: warps_per_block=" << warps_per_block
           << " is not supported (expected 4, 8, 16 or 32)";
        throw runtime_error(ss.str());
    }

    // One uniform rule at every axis and every (Df,Dt). The 32 is GpuWiDownsampler's
    // output tile size and intensity_clip's warp width. It is slightly stronger than
    // strictly necessary -- AXIS_TIME and AXIS_NONE at (Df,Dt)=(1,1) run no downsampler,
    // and so need only nt_chunk % 32 == 0 -- but one rule is worth more than the extra
    // freedom, since the tests compare axes against each other on the same geometry.
    if ((F % (32*Df)) || (nt_chunk % (32*Dt))) {
        stringstream ss;
        ss << "GpuIntensityClipper: expected F divisible by 32*Df and nt_chunk divisible"
           << " by 32*Dt, got (F,nt_chunk)=(" << F << "," << nt_chunk << ") with (Df,Dt)=("
           << Df << "," << Dt << ")";
        throw runtime_error(ss.str());
    }

    return B;
}


static long _wrms_L(ClipperAxis axis, long F_ds, long T_ds)
{
    switch (axis) {
        case ClipperAxis::TIME: return T_ds;
        case ClipperAxis::FREQ: return F_ds;
        case ClipperAxis::NONE: return F_ds * T_ds;
    }
    throw runtime_error("GpuIntensityClipper: bad ClipperAxis");
}


static long _wrms_R(ClipperAxis axis, long B, long F_ds, long T_ds)
{
    switch (axis) {
        case ClipperAxis::TIME: return B * F_ds;
        case ClipperAxis::FREQ: return B * T_ds;
        case ClipperAxis::NONE: return B;
    }
    throw runtime_error("GpuIntensityClipper: bad ClipperAxis");
}


// Scratch layout, in float32 elements. Kept in one place, and in the same order that
// launch() carves it:
//
//    (I_ds, W_ds)   ncell each, only when (Df,Dt) != (1,1)
//    (I_t,  W_t)    ncell each, only when axis == FREQ
//    mean, var      wrms_R each
//    GpuWrms        its own scratch, nonzero only on the global-memory path
//
static long _scratch_nelts(ClipperAxis axis, long B, long F_ds, long T_ds,
                           long Df, long Dt, long niter, double iter_sigma, bool two_pass)
{
    const long ncell = B * F_ds * T_ds;
    const long R = _wrms_R(axis, B, F_ds, T_ds);
    const bool need_ds = (Df != 1) || (Dt != 1);

    long n = 2*R;
    if (need_ds)
        n += 2*ncell;
    if (axis == ClipperAxis::FREQ)
        n += 2*ncell;

    GpuWrms wrms(_wrms_L(axis, F_ds, T_ds), niter, iter_sigma, two_pass);
    return n + wrms.scratch_nelts(R);
}


GpuIntensityClipper::GpuIntensityClipper(long B_, long F_, long nt_chunk_, ClipperAxis axis_,
                                         double sigma_, long Df_, long Dt_, long niter_,
                                         double iter_sigma_, bool two_pass_,
                                         long warps_per_block_) :
    B(_checked_B(B_, F_, nt_chunk_, sigma_, Df_, Dt_, niter_, iter_sigma_, warps_per_block_)),
    F(F_), nt_chunk(nt_chunk_), axis(axis_), sigma(sigma_), Df(Df_), Dt(Dt_),
    niter(niter_), iter_sigma(iter_sigma_), two_pass(two_pass_),
    warps_per_block(warps_per_block_),
    F_ds(F_ / Df_), T_ds(nt_chunk_ / Dt_),
    wrms_L(_wrms_L(axis_, F_/Df_, nt_chunk_/Dt_)),
    wrms_R(_wrms_R(axis_, B_, F_/Df_, nt_chunk_/Dt_)),
    scratch_nelts(_scratch_nelts(axis_, B_, F_/Df_, nt_chunk_/Dt_, Df_, Dt_,
                                 niter_, iter_sigma_, two_pass_))
{ }


// -------------------------------------------------------------------------------------------------


// Carve the next sub-array out of the caller's scratch, advancing 'pos'.
static Array<float> _carve(Array<float> &scratch, long &pos, std::initializer_list<long> shape)
{
    long n = 1;
    for (long s: shape)
        n *= s;

    Array<float> ret = scratch.slice(0, pos, pos+n).reshape(shape);
    pos += n;
    return ret;
}


void GpuIntensityClipper::launch(const Array<float> &intensity, Array<float> &weights,
                                 Array<float> &scratch, cudaStream_t stream) const
{
    xassert_eq(intensity.ndim, 3);

    // Checked before the general shape assert, so that a caller with a longer time axis
    // is told what is actually going on. "T == N*nt_chunk is not implemented" and "bad
    // shape" are very different messages to be handed.
    if ((intensity.shape[0] == B) && (intensity.shape[1] == F)
        && (intensity.shape[2] != nt_chunk) && (intensity.shape[2] % nt_chunk == 0)) {
        stringstream ss;
        ss << "GpuIntensityClipper::launch(): got T=" << intensity.shape[2] << ", but this"
           << " class currently requires T == nt_chunk (=" << nt_chunk << "). Processing"
           << " T = N*nt_chunk samples in one call is not implemented yet -- it is a useful"
           << " generalization we may add later (see the chunking note in"
           << " IntensityClipper.hpp). For now, call launch() once per nt_chunk-sized"
           << " block, or use ReferenceIntensityClipper, which does implement it.";
        throw runtime_error(ss.str());
    }

    xassert_shape_eq(intensity, ({B, F, nt_chunk}));
    xassert_shape_eq(weights, ({B, F, nt_chunk}));

    const std::initializer_list<const Array<float> *> arrays =
        { &intensity, &weights, &scratch };

    for (const Array<float> *a: arrays) {
        xassert(a->is_fully_contiguous());
        xassert(a->on_gpu());
    }

    xassert_eq(scratch.ndim, 1);
    xassert_ge(scratch.shape[0], scratch_nelts);

    xassert(intensity.data != weights.data);
    xassert(scratch.data != intensity.data);
    xassert(scratch.data != weights.data);

    const bool need_ds = (Df != 1) || (Dt != 1);
    long pos = 0;

    // Step 1: downsample, unless (Df,Dt) = (1,1), which is the identity -- in which case
    // the full-resolution arrays ARE the downsampled ones and no copy is made.
    const Array<float> *cell_i = &intensity;
    const Array<float> *cell_w = &weights;
    Array<float> ds_i, ds_w;

    if (need_ds) {
        ds_i = _carve(scratch, pos, {B, F_ds, T_ds});
        ds_w = _carve(scratch, pos, {B, F_ds, T_ds});
        GpuWiDownsampler(Df, Dt, false).launch(ds_i, ds_w, intensity, weights, stream);
        cell_i = &ds_i;
        cell_w = &ds_w;
    }

    // Step 2: for AXIS_FREQ, transpose the (small) downsampled arrays so that the
    // statistic's row -- a frequency column -- is contiguous.
    //
    // Note that this is a transpose of the DOWNSAMPLED arrays, not a second downsample
    // with transpose=true. The two give bit-identical results (a cell's reduction order
    // is the same either way, and a transpose moves data without touching it), and this
    // one is much cheaper: it reads the full-resolution pair once instead of twice, at a
    // cost of a few passes over an array Df*Dt times smaller. At (2,16) that is 2.19
    // full-resolution passes against 4.
    const Array<float> *stat_i = cell_i;
    const Array<float> *stat_w = cell_w;
    Array<float> t_i, t_w;

    if (axis == ClipperAxis::FREQ) {
        t_i = _carve(scratch, pos, {B, T_ds, F_ds});
        t_w = _carve(scratch, pos, {B, T_ds, F_ds});
        GpuWiDownsampler(1, 1, true).launch(t_i, t_w, *cell_i, *cell_w, stream);
        stat_i = &t_i;
        stat_w = &t_w;
    }

    // Step 3: the statistic. All three axes are now "one output per row of a contiguous
    // (R, L) array", so this is a reshape and no data moves: (B,F_ds,T_ds) -> (B*F_ds,
    // T_ds) for TIME, (B,T_ds,F_ds) -> (B*T_ds, F_ds) for FREQ, and (B,F_ds,T_ds) ->
    // (B, F_ds*T_ds) for NONE, since a contiguous 3-d array is also a 2-d one.
    Array<float> mean = _carve(scratch, pos, {wrms_R});
    Array<float> var = _carve(scratch, pos, {wrms_R});

    GpuWrms wrms(wrms_L, niter, iter_sigma, two_pass);
    long nw = wrms.scratch_nelts(wrms_R);
    Array<float> wrms_scratch = (nw > 0) ? _carve(scratch, pos, {nw}) : Array<float>();

    wrms.launch(mean, var, stat_i->reshape({wrms_R, wrms_L}),
                stat_w->reshape({wrms_R, wrms_L}), wrms_scratch, stream);

    // Step 4: the final clip. Note that this reads the UNTRANSPOSED downsampled intensity
    // (cell_i, not stat_i): the statistic wants frequency contiguous, and the mask
    // application wants time contiguous.
    long ntiles = B * F_ds * (T_ds / 32);
    long nblocks = (ntiles + warps_per_block - 1) / warps_per_block;
    dim3 nthreads(32, warps_per_block);

    intensity_clip_kernel <<< nblocks, nthreads, 0, stream >>>
        (weights.data, cell_i->data, mean.data, var.data, float(sigma),
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
