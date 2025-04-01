#include "../include/pirate/trackers.hpp"

namespace pirate {
#if 0
}  // editor auto-indent
#endif


BandwidthTracker &BandwidthTracker::operator+=(const BandwidthTracker &x)
{
    nbytes_gmem += x.nbytes_gmem;
    nbytes_hmem += x.nbytes_hmem;
    nbytes_h2g += x.nbytes_h2g;
    nbytes_g2h += x.nbytes_g2h;
    kernel_launches += x.kernel_launches;
    memcpy_h2g_calls += x.memcpy_h2g_calls;
    memcpy_g2h_calls += x.memcpy_g2h_calls;
    memcpy_g2g_calls += x.memcpy_g2g_calls;
    
    return *this;
}


BandwidthTracker &BandwidthTracker::operator*=(long x)
{
    nbytes_gmem *= x;
    nbytes_hmem *= x;
    nbytes_h2g *= x;
    nbytes_g2h *= x;
    kernel_launches *= x;
    memcpy_h2g_calls *= x;
    memcpy_g2h_calls *= x;
    memcpy_g2g_calls *= x;

    return *this;
}


BandwidthTracker operator+(const BandwidthTracker &x, const BandwidthTracker &y)
{
    BandwidthTracker ret = x;
    ret += y;
    return ret;
}


BandwdithTracker operator*(long x, const BandwidthTracker &y)
{
    BandwidthTracker ret = y;
    ret *= x;
    return ret;
}


}  // namespace pirate
