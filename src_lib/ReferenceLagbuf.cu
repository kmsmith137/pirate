#include "../include/pirate/internals/ReferenceLagbuf.hpp"
#include <cassert>
#include <cstring>

using namespace std;
using namespace gputils;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


ReferenceLagbuf::ReferenceLagbuf(const vector<int> &lags_, int ntime_) :
    nchan(lags_.size()),
    ntime(ntime_),
    nrstate(0),
    lags(lags_)
{
    assert(nchan > 0);
    assert(ntime > 0);
    
    for (int c = 0; c < nchan; c++) {
	assert(lags[c] >= 0);
	nrstate += lags[c];
    }

    this->rstate = Array<float> ({nrstate}, af_uhost | af_zero);
    this->scratch = Array<float> ({ntime}, af_uhost | af_zero);
}


void ReferenceLagbuf::apply_lags(Array<float> &arr) const
{
    assert(arr.shape_equals({nchan,ntime}));
    assert(arr.strides[1] == 1);
    
    this->apply_lags(arr.data, arr.strides[0]);
}


void ReferenceLagbuf::apply_lags(float *arr, int stride) const
{
    float *rp = rstate.data;
    float *sp = scratch.data;
    
    for (int c = 0; c < nchan; c++) {
	int lag = lags[c];
	
	if (lag == 0)
	    continue;

	float *row = arr + c*stride;
	int n = std::min(lag, ntime);
	
	// Inefficient for (lag >= ntime).
	memcpy(sp, row + (ntime-n), n * sizeof(float));
	memmove(row+n, row, (ntime-n) * sizeof(float));
	memcpy(row, rp, n * sizeof(float));
	memmove(rp, rp+n, (lag-n)*sizeof(float));
	memcpy(rp + (lag-n), sp, n * sizeof(float));
	
	rp += lag;
    }
}


}  // namespace pirate
