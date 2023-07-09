#ifndef _PIRATE_CONSTANTS_HPP
#define _PIRATE_CONSTANTS_HPP

namespace pirate {
#if 0
}  // editor auto-indent
#endif


struct constants
{
    static constexpr int bytes_per_gpu_cache_line = 128;

    // The "segment" is the unit of alignment for PCI-E transfers (or more generally,
    // transfers between dedispersion iobufs and ring buffers).
    //
    // Currently, I'm taking segments to correspond to GPU cache lines, but I may
    // generalize this later. This choice was motivated by a microbenchmark which
    // showed that PCI-E transfers were fast, as long as they were cache-aligned.
    
    static constexpr int bytes_per_segment = bytes_per_gpu_cache_line;
    
    // Currently all Dedispersers are two-stage, and each stage has rank <= 8,
    // so max total rank is 16.
    
    static constexpr int max_tree_rank = 16;

    // If you need to change 'max_downsampling_level', there should be no issues
    // (besides needing to recompile). However, if max_downsampling_level is
    // gratuitously large, then compilation time may be an issue.
    
    static constexpr int max_downsampling_level = 6;
    
    // If you need to change 'max_early_triggers_per_downsampling_level',
    // there should be no issues (besides needing to recompile). However,
    // if max_early_triggers_per_downsampling_level is gratuitously large,
    // then 'class DedispersionPlan' will be bloated.
    
    static constexpr int max_early_triggers_per_downsampling_level = 5;

    // I think this always makes sense.
    static_assert((bytes_per_segment % bytes_per_gpu_cache_line) == 0);

    // These assumptions are made all over the place.
    // (Placement of static_asserts in this source file is arbitrary.)
    static_assert(sizeof(int) == 4);
    static_assert(sizeof(long) == 8);
};


}  // namespace pirate

#endif // _PIRATE_CONSTANTS_HPP
