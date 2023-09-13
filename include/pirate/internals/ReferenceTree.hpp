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
// Processes input incrementally in chunks of shape (2^rank, ntime).
//
// The RefrerenceTree is unaware of the larger dedispersion plan (stage0/stage1 split,
// early triggers, downsampling, etc.) but can be used as a "building block" to implement
// these features.


class ReferenceTree
{
public:
    ReferenceTree(int rank, int ntime);

    int rank = 0;
    int ntime = 0;
    int nrstate = 0;
    int nscratch = 0;

    // 2-d array of shape (2^rank, ntime).
    // Dedispersion is done in place -- output index is a bit-reversed delay.
    void dedisperse(gputils::Array<float> &arr, float *rstate, float *scratch) const; 
    void dedisperse(float *arr, int stride, float *rstate, float *scratch) const;

protected:
    std::shared_ptr<ReferenceTree> prev_tree;
    std::vector<int> lags;  // length 2^(rank-1)
};


}  // namespace pirate

#endif // _PIRATE_INTERNALS_REFERENCE_TREE_HPP
