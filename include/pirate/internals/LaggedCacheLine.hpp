#ifndef _PIRATE_INTERNALS_LAGGED_CACHE_LINE_HPP
#define _PIRATE_INTERNALS_LAGGED_CACHE_LINE_HPP

#include "../constants.hpp"  // constants::max_early_triggers_per_downsampling_level

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// -------------------------------------------------------------------------------------------------
//
// LaggedCacheLine: represents one source segment, and one or more destination cache lines,
// represented as (clag, dst_segment) pairs. Note that the (clag, dst_segment) pairs are
// lexicographically sorted.


struct LaggedCacheLine
{
    static constexpr int maxdst = constants::max_early_triggers_per_downsampling_level + 1;

    const int src_segment;
    
    int ndst = 0;
    int dst_clag[maxdst];
    int dst_segment[maxdst];

    LaggedCacheLine(int src_segment);
    
    void add_dst(int clag, int segment);
};


}  // namespace pirate

#endif // _PIRATE_INTERNALS_LAGGED_CACHE_LINE_HPP
