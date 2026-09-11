#include "../../include/pirate/chimefrb/ClipperBase.hpp"
#include "../../include/pirate/chimefrb/WiDownsampler.hpp"
#include "../../include/pirate/chimefrb/Wrms.hpp"

#include <sstream>
#include <ksgpu/xassert.hpp>

using namespace std;
using namespace ksgpu;

namespace pirate {
namespace chimefrb {
#if 0
}}  // editor auto-indent
#endif


// -------------------------------------------------------------------------------------------------
//
// Constructor helpers. These run from the initializer list, so that argument checking
// happens before the derived members (which divide by Df and Dt) are computed.


// Returns its B argument, so that it can initialize 'B'.
static long _checked_B(const char *name, long B, long F, long nt_chunk, long Df, long Dt,
                       long niter, double iter_sigma)
{
    if (B < 1)
        throw runtime_error(string(name) + ": expected B >= 1");
    if (Df < 1)
        throw runtime_error(string(name) + ": expected Df >= 1");
    if (Dt < 1)
        throw runtime_error(string(name) + ": expected Dt >= 1");
    if (niter < 1)
        throw runtime_error(string(name) + ": expected niter >= 1 (niter counts total"
                            " passes, so niter=1 means no refinement)");
    if (iter_sigma < 0.0)
        throw runtime_error(string(name) + ": expected iter_sigma >= 0");

    // One uniform rule at every axis and every (Df,Dt). The 32 is GpuWiDownsampler's
    // output tile size and the clip kernels' warp width. It is slightly stronger than
    // strictly necessary -- AXIS_TIME and AXIS_NONE at (Df,Dt)=(1,1) run no downsampler,
    // and so need only nt_chunk % 32 == 0 -- but one rule is worth more than the extra
    // freedom, since the tests compare axes against each other on the same geometry.
    if ((F % (32*Df)) || (nt_chunk % (32*Dt))) {
        stringstream ss;
        ss << name << ": expected F divisible by 32*Df and nt_chunk divisible"
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
    throw runtime_error("GpuClipperBase: bad ClipperAxis");
}


static long _wrms_R(ClipperAxis axis, long B, long F_ds, long T_ds)
{
    switch (axis) {
        case ClipperAxis::TIME: return B * F_ds;
        case ClipperAxis::FREQ: return B * T_ds;
        case ClipperAxis::NONE: return B;
    }
    throw runtime_error("GpuClipperBase: bad ClipperAxis");
}


// Scratch layout, in float32 elements. Kept in one place, and in the same order that
// _launch_statistic() carves it:
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


GpuClipperBase::GpuClipperBase(const char *name_, long B_, long F_, long nt_chunk_,
                               ClipperAxis axis_, long Df_, long Dt_, long niter_,
                               double iter_sigma_, bool two_pass_) :
    B(_checked_B(name_, B_, F_, nt_chunk_, Df_, Dt_, niter_, iter_sigma_)),
    F(F_), nt_chunk(nt_chunk_), axis(axis_), Df(Df_), Dt(Dt_),
    niter(niter_), iter_sigma(iter_sigma_), two_pass(two_pass_),
    F_ds(F_ / Df_), T_ds(nt_chunk_ / Dt_),
    wrms_L(_wrms_L(axis_, F_/Df_, nt_chunk_/Dt_)),
    wrms_R(_wrms_R(axis_, B_, F_/Df_, nt_chunk_/Dt_)),
    scratch_nelts(_scratch_nelts(axis_, B_, F_/Df_, nt_chunk_/Dt_, Df_, Dt_,
                                 niter_, iter_sigma_, two_pass_)),
    name(name_)
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


GpuClipperBase::StatisticOutputs
GpuClipperBase::_launch_statistic(const Array<float> &intensity, const Array<float> &weights,
                                  Array<float> &scratch, cudaStream_t stream) const
{
    xassert_eq(intensity.ndim, 3);

    // Checked before the general shape assert, so that a caller with a longer time axis
    // is told what is actually going on. "T == N*nt_chunk is not implemented" and "bad
    // shape" are very different messages to be handed.
    if ((intensity.shape[0] == B) && (intensity.shape[1] == F)
        && (intensity.shape[2] != nt_chunk) && (intensity.shape[2] % nt_chunk == 0)) {
        stringstream ss;
        ss << name << "::launch(): got T=" << intensity.shape[2] << ", but this"
           << " class currently requires T == nt_chunk (=" << nt_chunk << "). Processing"
           << " T = N*nt_chunk samples in one call is not implemented yet -- it is a useful"
           << " generalization we may add later (see the chunking note in"
           << " ClipperBase.hpp). For now, call launch() once per nt_chunk-sized block, or"
           << " use the corresponding Reference* class in pirate_frb.chimefrb, which does"
           << " implement it.";
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

    StatisticOutputs out;

    // Step 1a: downsample, unless (Df,Dt) = (1,1), which is the identity -- in which case
    // the full-resolution arrays ARE the downsampled ones and no copy is made.
    Array<float> cell_w = weights;
    out.cell_i = intensity;

    if (need_ds) {
        out.cell_i = _carve(scratch, pos, {B, F_ds, T_ds});
        cell_w = _carve(scratch, pos, {B, F_ds, T_ds});
        GpuWiDownsampler(Df, Dt, false).launch(out.cell_i, cell_w, intensity, weights, stream);
    }

    // Step 1b: for AXIS_FREQ, transpose the (small) downsampled arrays so that the
    // statistic's row -- a frequency column -- is contiguous.
    //
    // Note that this is a transpose of the DOWNSAMPLED arrays, not a second downsample
    // with transpose=true. The two give bit-identical results (a cell's reduction order
    // is the same either way, and a transpose moves data without touching it), and this
    // one is much cheaper: it reads the full-resolution pair once instead of twice, at a
    // cost of a few passes over an array Df*Dt times smaller. At (2,16) that is 2.19
    // full-resolution passes against 4.
    Array<float> stat_i = out.cell_i;
    Array<float> stat_w = cell_w;

    if (axis == ClipperAxis::FREQ) {
        stat_i = _carve(scratch, pos, {B, T_ds, F_ds});
        stat_w = _carve(scratch, pos, {B, T_ds, F_ds});
        GpuWiDownsampler(1, 1, true).launch(stat_i, stat_w, out.cell_i, cell_w, stream);
    }

    // Step 2: the statistic. All three axes are now "one output per row of a contiguous
    // (R, L) array", so this is a reshape and no data moves: (B,F_ds,T_ds) -> (B*F_ds,
    // T_ds) for TIME, (B,T_ds,F_ds) -> (B*T_ds, F_ds) for FREQ, and (B,F_ds,T_ds) ->
    // (B, F_ds*T_ds) for NONE, since a contiguous 3-d array is also a 2-d one. The
    // device helper clipper_row() in ClipperBase.hpp is the inverse of this reshape.
    out.mean = _carve(scratch, pos, {wrms_R});
    out.var = _carve(scratch, pos, {wrms_R});

    GpuWrms wrms(wrms_L, niter, iter_sigma, two_pass);
    long nw = wrms.scratch_nelts(wrms_R);
    Array<float> wrms_scratch = (nw > 0) ? _carve(scratch, pos, {nw}) : Array<float>();

    wrms.launch(out.mean, out.var, stat_i.reshape({wrms_R, wrms_L}),
                stat_w.reshape({wrms_R, wrms_L}), wrms_scratch, stream);

    return out;
}


}}  // namespace pirate::chimefrb
