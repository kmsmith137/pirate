#ifndef _PIRATE_INTERNALS_REFERENCE_TREE_HPP
#define _PIRATE_INTERNALS_REFERENCE_TREE_HPP

#include <vector>
#include <memory>  // shared_ptr
#include <gputils/Array.hpp>


namespace pirate {
#if 0
}  // editor auto-indent
#endif


// ReferenceTree: simple, self-contained reference implementation of tree dedispersion.
//
// Processes input incrementally in chunks of shape (..., nfreq, ..., ntime), where "..."
// represents an arbitrary number of spectator axes. The time axis must be last, and be
// contiguous (i.e. stride=1). The number of frequencies must be a power of two.
//
// The RefrerenceTree is unaware of the larger dedispersion plan (stage0/stage1 split,
// early triggers, downsampling, etc.) but can be used as a "building block" to implement
// these features.


class ReferenceTree
{
public:
    // By default, the frequency axis is second-to-last.
    ReferenceTree(std::initializer_list<long> shape);
    ReferenceTree(std::initializer_list<long> shape, int freq_axis);
    ReferenceTree(int ndim, const long *shape, int freq_axis);
    
    // Dedispersion is done in-place, on an array of shape 'shape'.
    void dedisperse(gputils::Array<float> &arr);

    // Morally equivalent to make_shared<ReferenceTree> (...).
    // (Necessary since make_shared doesn't seem to work with initializer lists.)
    static std::shared_ptr<ReferenceTree> make(std::initializer_list<long> shape);
    static std::shared_ptr<ReferenceTree> make(std::initializer_list<long> shape, int freq_axis);
    
protected:
    std::vector<long> shape;
    int freq_axis = -1;

    int ndim = -1;
    int nfreq = -1;
    int ntime = -1;
    int rank = -1;
    long nrstate = 0;
    
    gputils::Array<float> rstate;          // can be large
    gputils::Array<float> scratch;         // always small (length ntime+1)

    // Used temporarily in dedisperse().
    std::vector<long> tmp_shape;
    std::vector<long> tmp_strides;
};


}  // namespace pirate

#endif // _PIRATE_INTERNALS_REFERENCE_TREE_HPP
