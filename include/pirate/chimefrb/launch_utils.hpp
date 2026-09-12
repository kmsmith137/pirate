#ifndef _PIRATE_CHIMEFRB_LAUNCH_UTILS_HPP
#define _PIRATE_CHIMEFRB_LAUNCH_UTILS_HPP

#include <initializer_list>
#include <ksgpu/Array.hpp>
#include <ksgpu/xassert.hpp>

namespace pirate {
namespace chimefrb {
#if 0
}}  // editor auto-indent
#endif


// Two helpers shared by the chimefrb transforms' launch() methods.
//
// Every chimefrb transform has the same launch() signature -- (intensity, weights, scratch,
// stream), with intensity and weights of shape (nbeams, nfreq, ntime) and scratch a 1-d
// float32 array of at least scratch_nelts elements -- so that a python pipeline class
// (pirate_frb.chimefrb.WiPipeline) can run any sequence of them. See
// pirate_frb/chimefrb/transform_io.py for the full statement of that interface.


// check_launch_args(): the argument checks every launch() makes. Shapes, dtype-agnostic
// contiguity, GPU residency, and no aliasing between the three arrays. 'scratch' may be
// empty (any ndim) when scratch_nelts == 0, and must otherwise be 1-d with at least
// scratch_nelts elements.
inline void check_launch_args(const ksgpu::Array<float> &intensity,
                              const ksgpu::Array<float> &weights,
                              const ksgpu::Array<float> &scratch,
                              long nbeams, long nfreq, long ntime, long scratch_nelts)
{
    xassert_shape_eq(intensity, ({nbeams, nfreq, ntime}));
    xassert_shape_eq(weights, ({nbeams, nfreq, ntime}));
    xassert(intensity.is_fully_contiguous());
    xassert(weights.is_fully_contiguous());
    xassert(intensity.on_gpu());
    xassert(weights.on_gpu());
    xassert(intensity.data != weights.data);

    if ((scratch_nelts > 0) || (scratch.size > 0)) {
        xassert_eq(scratch.ndim, 1);
        xassert_ge(scratch.shape[0], scratch_nelts);
        xassert(scratch.is_fully_contiguous());
        xassert(scratch.on_gpu());
        xassert(scratch.data != intensity.data);
        xassert(scratch.data != weights.data);
    }
}


// carve_scratch(): the next sub-array of the given shape out of the caller's 1-d scratch
// array, advancing 'pos'. A transform that lays its per-launch workspace out inside the
// caller's scratch (GpuClipperBase, GpuSplineDetrender) carves the pieces in a fixed
// order, so that its scratch_nelts is the sum of the pieces.
inline ksgpu::Array<float> carve_scratch(ksgpu::Array<float> &scratch, long &pos,
                                         std::initializer_list<long> shape)
{
    long n = 1;
    for (long s: shape)
        n *= s;

    xassert_le(pos + n, scratch.shape[0]);
    ksgpu::Array<float> ret = scratch.slice(0, pos, pos+n).reshape(shape);
    pos += n;
    return ret;
}


}}  // namespace pirate::chimefrb

#endif  // _PIRATE_CHIMEFRB_LAUNCH_UTILS_HPP
