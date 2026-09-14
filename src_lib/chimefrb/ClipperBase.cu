#include "../../include/pirate/chimefrb/ClipperBase.hpp"
#include "../../include/pirate/chimefrb/WiDownsamplingKernel.hpp"
#include "../../include/pirate/chimefrb/WrmsKernel.hpp"
#include "../../include/pirate/inlines.hpp"   // xdiv()

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
// Constructor helpers. _checked_ntime() runs the clipper's argument checks and returns
// 'ntime', so that the constructor can call it as the GpuTransform constructor's ntime
// argument -- i.e. before any member is initialized, since a base class is initialized first.
// That ordering is load-bearing: F_ds and T_ds divide by Df and Dt, so a Df of 0 has to be
// rejected before they are computed. (They also use xdiv(), which asserts the divisor and
// the divisibility, so the member initializers are safe even if that ordering is ever
// disturbed -- but xdiv's message names a line of inlines.hpp, where _checked_ntime's names
// the parameter, so the check below is the one a caller wants to hit.)


static long _checked_ntime(const char *name, long nfreq, long ntime, long nt_chunk,
                           long Df, long Dt, long niter, double iter_sigma)
{
    // nbeams, nfreq and ntime >= 1 are GpuTransform's checks.
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

    // One uniform rule at every axis and every (Df,Dt). The 32 is GpuWiDownsamplingKernel's
    // output tile size and the clip kernels' warp width. It is slightly stronger than
    // strictly necessary -- ClipperAxis::TIME and ::NONE at (Df,Dt)=(1,1) run no downsampler,
    // and so need only nt_chunk % 32 == 0 -- but one rule is worth more than the extra
    // freedom, since the tests compare axes against each other on the same geometry.
    if ((nfreq % (32*Df)) || (nt_chunk % (32*Dt))) {
        stringstream ss;
        ss << name << ": expected nfreq divisible by 32*Df and nt_chunk divisible"
           << " by 32*Dt, got (nfreq,nt_chunk)=(" << nfreq << "," << nt_chunk << ") with (Df,Dt)=("
           << Df << "," << Dt << ")";
        throw runtime_error(ss.str());
    }

    return ntime;
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


GpuClipperBase::GpuClipperBase(const char *name_, long nbeams_, long nfreq_, long ntime_,
                               long nt_chunk_, ClipperAxis axis_, long Df_, long Dt_,
                               long niter_, double iter_sigma_, bool two_pass_) :
    GpuTransform(name_, nbeams_, nfreq_,
                 _checked_ntime(name_, nfreq_, ntime_, nt_chunk_, Df_, Dt_, niter_, iter_sigma_),
                 /*scratch_nelts=*/0),      // set in the body: the layout needs the members below
    nt_chunk(nt_chunk_), axis(axis_), Df(Df_), Dt(Dt_),
    niter(niter_), iter_sigma(iter_sigma_), two_pass(two_pass_),
    F_ds(xdiv(nfreq_, Df_)), T_ds(xdiv(nt_chunk_, Dt_)),
    wrms_L(_wrms_L(axis_, xdiv(nfreq_, Df_), xdiv(nt_chunk_, Dt_))),
    wrms_R(_wrms_R(axis_, nbeams_, xdiv(nfreq_, Df_), xdiv(nt_chunk_, Dt_)))
{
    ScratchLayout lay;
    _carve_statistic(lay);
    scratch_nelts = lay.nelts();
}


// -------------------------------------------------------------------------------------------------


// The per-launch workspace of steps 1-2, and the ONLY description of its layout: the
// constructor runs this in sizing mode for scratch_nelts, _launch_statistic() below runs it
// in carving mode for the arrays. Both take the same branches, since every condition is a
// const member -- which is what makes the two agree by construction rather than by review.
GpuClipperBase::StatisticScratch GpuClipperBase::_carve_statistic(ScratchLayout &lay) const
{
    StatisticScratch s;

    // The downsampled pair, only when (Df,Dt) != (1,1) -- at (1,1) the downsample is the
    // identity, and the full-resolution arrays are used in place.
    if ((Df != 1) || (Dt != 1)) {
        s.cell_i = lay.carve({nbeams, F_ds, T_ds});
        s.cell_w = lay.carve({nbeams, F_ds, T_ds});
    }

    // The transposed pair, only for ClipperAxis::FREQ (see _launch_statistic() for why).
    if (axis == ClipperAxis::FREQ) {
        s.stat_i = lay.carve({nbeams, T_ds, F_ds});
        s.stat_w = lay.carve({nbeams, T_ds, F_ds});
    }

    s.mean = lay.carve({wrms_R});
    s.var = lay.carve({wrms_R});

    // GpuWrmsKernel's own scratch: zero on the shared-memory path, and then nothing is carved.
    long nw = GpuWrmsKernel(wrms_L, niter, iter_sigma, two_pass).scratch_nelts(wrms_R);
    if (nw > 0)
        s.wrms = lay.carve({nw});

    return s;
}


GpuClipperBase::StatisticOutputs
GpuClipperBase::_launch_statistic(const Array<float> &intensity, const Array<float> &weights,
                                  Array<float> &scratch, cudaStream_t stream) const
{
    const bool need_ds = (Df != 1) || (Dt != 1);

    ScratchLayout lay(scratch);
    StatisticScratch s = _carve_statistic(lay);
    xassert_eq(lay.nelts(), scratch_nelts);   // the constructor sized from this same function

    StatisticOutputs out;

    // Step 1a: downsample, unless (Df,Dt) = (1,1), which is the identity -- in which case
    // the full-resolution arrays ARE the downsampled ones and no copy is made.
    Array<float> cell_w = weights;
    out.cell_i = intensity;

    if (need_ds) {
        out.cell_i = s.cell_i;
        cell_w = s.cell_w;
        GpuWiDownsamplingKernel(Df, Dt, false).launch(out.cell_i, cell_w, intensity, weights, stream);
    }

    // Step 1b: for ClipperAxis::FREQ, transpose the (small) downsampled arrays so that the
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
        stat_i = s.stat_i;
        stat_w = s.stat_w;
        GpuWiDownsamplingKernel(1, 1, true).launch(stat_i, stat_w, out.cell_i, cell_w, stream);
    }

    // Step 2: the statistic. All three axes are now "one output per row of a contiguous
    // (R, L) array", so this is a reshape and no data moves: (nbeams,F_ds,T_ds) ->
    // (nbeams*F_ds, T_ds) for TIME, (nbeams,T_ds,F_ds) -> (nbeams*T_ds, F_ds) for FREQ, and
    // (nbeams,F_ds,T_ds) -> (nbeams, F_ds*T_ds) for NONE, since a contiguous 3-d array is
    // also a 2-d one. The
    // device helper clipper_row() in ClipperBase.hpp is the inverse of this reshape.
    out.mean = s.mean;
    out.var = s.var;

    GpuWrmsKernel wrms(wrms_L, niter, iter_sigma, two_pass);
    wrms.launch(out.mean, out.var, stat_i.reshape({wrms_R, wrms_L}),
                stat_w.reshape({wrms_R, wrms_L}), s.wrms, stream);

    return out;
}


}}  // namespace pirate::chimefrb
