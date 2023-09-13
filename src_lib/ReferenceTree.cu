#include "../include/pirate/internals/ReferenceTree.hpp"

#include "../include/pirate/internals/inlines.hpp"  // pow2(), bit_reverse_slow()
#include "../include/pirate/internals/utils.hpp"    // check_rank()

#include <cassert>

using namespace std;
using namespace gputils;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


ReferenceTree::ReferenceTree(int rank_, int ntime_) :
    rank(rank_),
    ntime(ntime_),
    nrstate(0),
    nscratch(ntime_)
{
    check_rank(rank, "ReferenceTree constructor");
    assert(ntime > 0);

    if (rank == 0)
	return;
    
    const int half_nfreq = 1 << (rank-1);
    this->lags.resize(half_nfreq);

    for (int j = 0; j < half_nfreq; j++) {
	int lag = bit_reverse_slow(j,rank-1) + 1;
	this->lags[j] = lag;
	this->nrstate += lag;
    }

    if (rank == 1)
	return;

    this->prev_tree = make_shared<ReferenceTree> (rank-1, ntime);
    this->nrstate += 2 * prev_tree->nrstate;
    this->nscratch = std::max(nscratch, prev_tree->nscratch);
}


void ReferenceTree::dedisperse(Array<float> &arr, float *rstate, float *scratch) const
{
    assert(arr.ndim == 2);
    assert(arr.shape[0] == pow2(rank));
    assert(arr.shape[1] == ntime);
    assert(arr.strides[1] == 1);

    this->dedisperse(arr.data, arr.strides[0], rstate, scratch);
}


void ReferenceTree::dedisperse(float *arr, int stride, float *rstate, float *scratch) const
{
    if (rank == 0)
	return;
    
    const int half_nfreq = 1 << (rank-1);
    
    if (rank > 1) {
	const int nrchild = prev_tree->nrstate;
	prev_tree->dedisperse(arr, stride, rstate, scratch);
	prev_tree->dedisperse(arr + half_nfreq * stride, stride, rstate + nrchild, scratch);
	rstate += 2*nrchild;
    }

    // Last tree iteration, with ring buffer lags.
    // This implementation is simple but slow!
    
    for (int j = 0; j < half_nfreq; j++) {
	float *row0 = arr + j * stride;
	float *row1 = row0 + half_nfreq * stride;
	
	int lag = lags[j];
	float x0 = (ntime >= lag) ? row0[ntime-lag] : rstate[ntime];

	int ncopy = std::min(lag, ntime);
	memcpy(scratch, row0 + ntime - ncopy, ncopy * sizeof(float));

	// FIXME planning trivial speedup here, after passing some unit tests.
	for (int t = ntime-1; t >= 0; t--) {
	    float y = row1[t];
	    float x1 = x0;
	    x0 = (t >= lag) ? row0[t-lag] : rstate[t];
	    
	    row0[t] = x1 + y;
	    row1[t] = x0 + y;
	}

	// Inefficient for (lag >= ntime).
	int ncomp = lag - ncopy;
	memmove(rstate, rstate + ncopy, ncomp * sizeof(float));
	memcpy(rstate + ncomp, scratch, ncopy * sizeof(float));
	
	rstate += lag;
    }
}


}  // namespace pirate
