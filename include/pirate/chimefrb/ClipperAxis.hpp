#ifndef _PIRATE_CHIMEFRB_CLIPPER_AXIS_HPP
#define _PIRATE_CHIMEFRB_CLIPPER_AXIS_HPP

namespace pirate {
namespace chimefrb {
#if 0
}}  // editor auto-indent
#endif


// Which axis a chimefrb clipper reduces along. Shared by GpuIntensityClipper and
// GpuStdDevClipper, which is why it lives in a header of its own rather than in either
// one's.
//
// The numeric values deliberately match rf_kernels::axis_type (core.hpp), and the numpy
// reference's AXIS_FREQ/AXIS_TIME/AXIS_NONE constants in
// pirate_frb/chimefrb/ReferenceIntensityClipper.py. A spot-check driver casts an integer
// straight to the old enum, so the three must not drift apart.

enum class ClipperAxis {
    FREQ = 0,   // one statistic per downsampled time sample, reducing over frequency
    TIME = 1,   // one statistic per downsampled frequency, reducing over time
    NONE = 2    // one statistic per beam, reducing over the whole plane
};


}}  // namespace pirate::chimefrb

#endif  // _PIRATE_CHIMEFRB_CLIPPER_AXIS_HPP
