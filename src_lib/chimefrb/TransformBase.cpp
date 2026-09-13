#include "../../include/pirate/chimefrb/TransformBase.hpp"

#include <cstdint>
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
// Constructor


static long _checked_positive(const string &name, const char *what, long x)
{
    if (x < 1) {
        stringstream ss;
        ss << name << ": expected " << what << " >= 1, got " << x;
        throw runtime_error(ss.str());
    }
    return x;
}


static long _checked_scratch_nelts(const string &name, long scratch_nelts)
{
    if (scratch_nelts < 0) {
        stringstream ss;
        ss << name << ": expected scratch_nelts >= 0, got " << scratch_nelts;
        throw runtime_error(ss.str());
    }
    return scratch_nelts;
}


GpuTransformBase::GpuTransformBase(const string &name_, long nbeams_, long nfreq_, long ntime_,
                                   long scratch_nelts_) :
    name(name_),
    nbeams(_checked_positive(name_, "nbeams", nbeams_)),
    nfreq(_checked_positive(name_, "nfreq", nfreq_)),
    ntime(_checked_positive(name_, "ntime", ntime_)),
    scratch_nelts(_checked_scratch_nelts(name_, scratch_nelts_))
{ }


// -------------------------------------------------------------------------------------------------
//
// launch(). The messages are written the way the python side of the code writes them
// ("expected 'weights' of shape (1, 64, 64), got (1, 64, 32)"), since a python caller who
// wrote the transform in python is the reader these checks are for.


// An array's shape as python prints it: "(1, 64, 32)", "(4096,)".
static string _shape_str(const Array<float> &a)
{
    stringstream ss;
    ss << "(";
    for (int i = 0; i < a.ndim; i++)
        ss << ((i > 0) ? ", " : "") << a.shape[i];
    ss << ((a.ndim == 1) ? ",)" : ")");
    return ss.str();
}


// True if the two arrays' byte ranges intersect. An empty array (data == nullptr, size 0)
// intersects nothing.
static bool _overlaps(const Array<float> &a, const Array<float> &b)
{
    const uintptr_t a0 = reinterpret_cast<uintptr_t>(a.data), a1 = a0 + sizeof(float) * a.size;
    const uintptr_t b0 = reinterpret_cast<uintptr_t>(b.data), b1 = b0 + sizeof(float) * b.size;
    return (a0 < b1) && (b0 < a1);
}


// The checks on 'intensity' and 'weights' ('what' names the argument in the message).
static void _check_data_array(const string &name, const char *what, const Array<float> &a,
                              long nbeams, long nfreq, long ntime)
{
    if ((a.ndim != 3) || (a.shape[0] != nbeams) || (a.shape[1] != nfreq) || (a.shape[2] != ntime)) {
        stringstream ss;
        ss << name << ".launch(): expected '" << what << "' of shape (" << nbeams << ", " << nfreq
           << ", " << ntime << "), got " << _shape_str(a);
        throw runtime_error(ss.str());
    }
    if (!a.is_fully_contiguous())
        throw runtime_error(name + ".launch(): expected '" + what + "' to be C-contiguous");
    if (!a.on_gpu())
        throw runtime_error(name + ".launch(): expected '" + what + "' to be on the GPU (got a host"
                            " array; pass a cupy array)");
}


void GpuTransformBase::launch(Array<float> &intensity, Array<float> &weights,
                              Array<float> &scratch, cudaStream_t stream) const
{
    _check_data_array(name, "intensity", intensity, nbeams, nfreq, ntime);
    _check_data_array(name, "weights", weights, nbeams, nfreq, ntime);

    if (intensity.data == weights.data)
        throw runtime_error(name + ".launch(): 'intensity' and 'weights' must be distinct arrays");

    // When scratch_nelts == 0 and the caller passed an empty array, there is nothing to
    // check. Any other scratch must be a 1-d contiguous GPU array that does not alias the
    // data, whatever scratch_nelts is.
    if ((scratch_nelts > 0) || (scratch.size > 0)) {
        if ((scratch.ndim != 1) || !scratch.is_fully_contiguous()) {
            stringstream ss;
            ss << name << ".launch(): expected 'scratch' to be a 1-d contiguous array, got shape "
               << _shape_str(scratch);
            throw runtime_error(ss.str());
        }
        if (!scratch.on_gpu())
            throw runtime_error(name + ".launch(): expected 'scratch' to be on the GPU");
        if (scratch.shape[0] < scratch_nelts) {
            stringstream ss;
            ss << name << ".launch(): 'scratch' has " << scratch.shape[0] << " elements, need at least "
               << scratch_nelts;
            throw runtime_error(ss.str());
        }
        if (_overlaps(scratch, intensity))
            throw runtime_error(name + ".launch(): 'scratch' overlaps 'intensity'");
        if (_overlaps(scratch, weights))
            throw runtime_error(name + ".launch(): 'scratch' overlaps 'weights'");
    }

    // launch_checked() sees exactly scratch_nelts elements, so that a subclass which carves
    // its workspace out of the array is checked against its own declared size, not against
    // whatever the caller happened to pass.
    Array<float> s = (scratch_nelts > 0) ? scratch.slice(0, 0, scratch_nelts) : Array<float>();
    launch_checked(intensity, weights, s, stream);
}


Array<float> GpuTransformBase::carve_scratch(Array<float> &scratch, long &pos,
                                             initializer_list<long> shape)
{
    long n = 1;
    for (long s: shape)
        n *= s;

    xassert_le(pos + n, scratch.shape[0]);
    Array<float> ret = scratch.slice(0, pos, pos+n).reshape(shape);
    pos += n;
    return ret;
}


}}  // namespace pirate::chimefrb
