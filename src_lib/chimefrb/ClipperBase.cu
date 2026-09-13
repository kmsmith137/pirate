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
// Constructor helpers. The clipper's argument checks and its scratch size are computed by
// _checked_scratch_nelts(), which runs as the GpuTransformBase constructor's argument --
// i.e. before anything else, since the base class is initialized before any member. That
// is what lets the derived members below divide by Df and Dt safely, and what makes
// scratch_nelts (a const member of the base) known in time.


static void _check_args(const char *name, long nfreq, long ntime, long nt_chunk,
                        long Df, long Dt, long niter, double iter_sigma)
{
    // nbeams, nfreq and ntime >= 1 are GpuTransformBase's checks.
    if (nt_chunk < 1)
        throw runtime_error(string(name) + ": expected nt_chunk >= 1");
    if (ntime != nt_chunk) {
        stringstream ss;
        ss << name << ": got ntime=" << ntime << " and nt_chunk=" << nt_chunk << ", but this"
           << " class currently requires ntime == nt_chunk (the array must hold exactly one"
           << " chunk). Processing ntime = N*nt_chunk samples in one launch is not implemented"
           << " yet -- see the chunking note in ClipperBase.hpp. The corresponding Reference*"
           << " class in pirate_frb.chimefrb does implement it.";
        throw runtime_error(ss.str());
    }
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
    if ((nfreq % (32*Df)) || (nt_chunk % (32*Dt))) {
        stringstream ss;
        ss << name << ": expected nfreq divisible by 32*Df and nt_chunk divisible"
           << " by 32*Dt, got (nfreq,nt_chunk)=(" << nfreq << "," << nt_chunk << ") with (Df,Dt)=("
           << Df << "," << Dt << ")";
        throw runtime_error(ss.str());
    }
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


// Checks the arguments (above), then returns the scratch layout's size in float32 elements.
// The layout is kept in one place, and in the same order that _launch_statistic() carves it:
//
//    (I_ds, W_ds)   ncell each, only when (Df,Dt) != (1,1)
//    (I_t,  W_t)    ncell each, only when axis == FREQ
//    mean, var      wrms_R each
//    GpuWrms        its own scratch, nonzero only on the global-memory path
//
static long _checked_scratch_nelts(const char *name, long B, long nfreq, long ntime, long nt_chunk,
                                   ClipperAxis axis, long Df, long Dt, long niter,
                                   double iter_sigma, bool two_pass)
{
    _check_args(name, nfreq, ntime, nt_chunk, Df, Dt, niter, iter_sigma);

    const long F_ds = nfreq / Df, T_ds = nt_chunk / Dt;
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


GpuClipperBase::GpuClipperBase(const char *name_, long nbeams_, long nfreq_, long ntime_,
                               long nt_chunk_, ClipperAxis axis_, long Df_, long Dt_,
                               long niter_, double iter_sigma_, bool two_pass_) :
    GpuTransformBase(name_, nbeams_, nfreq_, ntime_,
                     _checked_scratch_nelts(name_, nbeams_, nfreq_, ntime_, nt_chunk_, axis_, Df_, Dt_,
                                            niter_, iter_sigma_, two_pass_)),
    nt_chunk(nt_chunk_), axis(axis_), Df(Df_), Dt(Dt_),
    niter(niter_), iter_sigma(iter_sigma_), two_pass(two_pass_),
    F_ds(nfreq_ / Df_), T_ds(nt_chunk_ / Dt_),
    wrms_L(_wrms_L(axis_, nfreq_/Df_, nt_chunk_/Dt_)),
    wrms_R(_wrms_R(axis_, nbeams_, nfreq_/Df_, nt_chunk_/Dt_))
{ }


// -------------------------------------------------------------------------------------------------


GpuClipperBase::StatisticOutputs
GpuClipperBase::_launch_statistic(const Array<float> &intensity, const Array<float> &weights,
                                  Array<float> &scratch, cudaStream_t stream) const
{
    const bool need_ds = (Df != 1) || (Dt != 1);
    long pos = 0;

    StatisticOutputs out;

    // Step 1a: downsample, unless (Df,Dt) = (1,1), which is the identity -- in which case
    // the full-resolution arrays ARE the downsampled ones and no copy is made.
    Array<float> cell_w = weights;
    out.cell_i = intensity;

    if (need_ds) {
        out.cell_i = carve_scratch(scratch, pos, {nbeams, F_ds, T_ds});
        cell_w = carve_scratch(scratch, pos, {nbeams, F_ds, T_ds});
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
        stat_i = carve_scratch(scratch, pos, {nbeams, T_ds, F_ds});
        stat_w = carve_scratch(scratch, pos, {nbeams, T_ds, F_ds});
        GpuWiDownsampler(1, 1, true).launch(stat_i, stat_w, out.cell_i, cell_w, stream);
    }

    // Step 2: the statistic. All three axes are now "one output per row of a contiguous
    // (R, L) array", so this is a reshape and no data moves: (nbeams,F_ds,T_ds) ->
    // (nbeams*F_ds, T_ds) for TIME, (nbeams,T_ds,F_ds) -> (nbeams*T_ds, F_ds) for FREQ, and
    // (nbeams,F_ds,T_ds) -> (nbeams, F_ds*T_ds) for NONE, since a contiguous 3-d array is
    // also a 2-d one. The
    // device helper clipper_row() in ClipperBase.hpp is the inverse of this reshape.
    out.mean = carve_scratch(scratch, pos, {wrms_R});
    out.var = carve_scratch(scratch, pos, {wrms_R});

    GpuWrms wrms(wrms_L, niter, iter_sigma, two_pass);
    long nw = wrms.scratch_nelts(wrms_R);
    Array<float> wrms_scratch = (nw > 0) ? carve_scratch(scratch, pos, {nw}) : Array<float>();

    wrms.launch(out.mean, out.var, stat_i.reshape({wrms_R, wrms_L}),
                stat_w.reshape({wrms_R, wrms_L}), wrms_scratch, stream);

    return out;
}


}}  // namespace pirate::chimefrb
