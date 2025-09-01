#ifndef _PIRATE_REFERENCE_TREE_HPP
#define _PIRATE_REFERENCE_TREE_HPP

#include <vector>
#include <memory>  // shared_ptr
#include <ksgpu/Array.hpp>


namespace pirate {
#if 0
}  // editor auto-indent
#endif


// ReferenceTree: simple, self-contained reference implementation of tree dedispersion.
//
// Processes input incrementally in chunks of shape (..., nfreq, ntime), where "..."
// represents an arbitrary number of spectator axes. The time axis must be contiguous.
// The number of frequencies must be a power of two.
//
// The RefrerenceTree is unaware of the larger dedispersion plan (stage1/stage2 split,
// early triggers, downsampling, etc.) but can be used as a "building block" to implement
// these features.


class ReferenceTree
{
public:
    // Frequency axis is second-to-last, inner (ntime*nspec) axis is last.
    // All other indices are spectators.
    ReferenceTree(const std::vector<long> &shape, long nspec);
    ReferenceTree(int ndim, const long *shape, long nspec);
    
    // Dedispersion is done in-place.
    // Array shape is (Spectator indices) + (nfreq, ntime*nspec).
    void dedisperse(ksgpu::Array<float> &arr);

    // Morally equivalent to make_shared<ReferenceTree> (...).
    // (Necessary since make_shared doesn't seem to work with initializer lists.)
    static std::shared_ptr<ReferenceTree> make(std::initializer_list<long> shape, long nspec);
    
protected:
    // (Spectator indices) + (nfreq, ntime*nspec)
    std::vector<long> shape;

    int ndim = -1;
    int rank = -1;
    long nfreq = -1;
    long ntime = -1;
    long nspec = -1;
    long ninner = -1;   // = ntime*nspec
    long npstate = 0;
    
    ksgpu::Array<float> pstate;          // can be large
    ksgpu::Array<float> scratch;         // always small (length (ntime+1)*nspec)
};


}  // namespace pirate

#endif // _PIRATE_REFERENCE_TREE_HPP
