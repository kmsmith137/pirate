#include "../include/pirate/internals/LaggedCacheLine.hpp"
#include <cassert>

using namespace std;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


LaggedCacheLine::LaggedCacheLine(int src_segment_)
    : src_segment(src_segment_)
{
    assert(src_segment >= 0);
}


void LaggedCacheLine::add_dst(int clag, int segment)
{
    assert(clag >= 0);
    assert(segment >= 0);
    assert(ndst < maxdst);

    // We expect (clag, segment) pairs to be sorted.
    if (ndst > 0) {
	int prev_clag = dst_clag[ndst-1];
	int prev_segment = dst_segment[ndst-1];
	assert((prev_clag < clag) || ((prev_clag == clag) && (prev_segment < segment)));
    }
	
    dst_clag[ndst] = clag;
    dst_segment[ndst] = segment;
    ndst++;
}


}  // namespace pirate
