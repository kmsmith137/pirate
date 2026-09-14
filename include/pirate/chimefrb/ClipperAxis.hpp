#ifndef _PIRATE_CHIMEFRB_CLIPPER_AXIS_HPP
#define _PIRATE_CHIMEFRB_CLIPPER_AXIS_HPP

#include <string>
#include <stdexcept>

namespace pirate {
namespace chimefrb {
#if 0
}}  // editor auto-indent
#endif


// Which axis a chimefrb clipper reduces along. Shared by GpuIntensityClipper and
// GpuStdDevClipper, which is why it lives in a header of its own rather than in either
// one's.
//
// The numeric values deliberately match rf_kernels::axis_type (core.hpp). A spot-check
// driver casts an integer straight to the old enum, so the two must not drift apart.
//
// Note the values are NOT array axis indices: for a (nbeams, nfreq, ntime) array, frequency
// is axis 1 and time is axis 2, whereas FREQ is 0 and TIME is 1. Nothing should index an
// array with a ClipperAxis.

enum class ClipperAxis {
    FREQ = 0,   // one statistic per downsampled time sample, reducing over frequency
    TIME = 1,   // one statistic per downsampled frequency, reducing over time
    NONE = 2    // one statistic per beam, reducing over the whole plane
};


// The axis NAMES: the one spelling used outside C++. A python caller passes one of these
// strings (a pybind11 type_caster in src_pybind11/pirate_pybind11_chimefrb.cpp converts, so
// python never sees the enum), a saved yaml file carries one, and the timing printouts use
// one. The legacy rf_pipelines json spells them 'AXIS_FREQ' and so on, which is mapped in
// pirate_frb/chimefrb/utils.py, the one place that old spelling survives.

inline const char *axis_to_string(ClipperAxis axis)
{
    switch (axis) {
        case ClipperAxis::FREQ: return "freq";
        case ClipperAxis::TIME: return "time";
        case ClipperAxis::NONE: return "none";
    }
    throw std::runtime_error("chimefrb::axis_to_string(): bad ClipperAxis");
}


// Throws on anything else, naming the three accepted spellings: the caller of a mistyped
// axis wants to be told what to write, not what went wrong. Exact match, no case folding --
// one spelling is the point.
inline ClipperAxis axis_from_string(const std::string &s)
{
    if (s == "freq") return ClipperAxis::FREQ;
    if (s == "time") return ClipperAxis::TIME;
    if (s == "none") return ClipperAxis::NONE;
    throw std::runtime_error("chimefrb: expected axis to be 'freq', 'time' or 'none', got '" + s + "'");
}


}}  // namespace pirate::chimefrb

#endif  // _PIRATE_CHIMEFRB_CLIPPER_AXIS_HPP
