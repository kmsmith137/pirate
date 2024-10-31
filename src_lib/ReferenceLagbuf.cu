#include "../include/pirate/internals/ReferenceLagbuf.hpp"
#include <cassert>
#include <cstring>

using namespace std;
using namespace gputils;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// -------------------------------------------------------------------------------------------------
//
// ReferenceLagbuf constructor.


static long _sum_lags(const int *lags, int nd, const ssize_t *shape, const ssize_t *strides)
{
    assert(nd > 0);
    
    long n = shape[0];
    long s = strides[0];    
    long ret = 0;
    
    if (nd == 1) {
	for (long i = 0; i < n; i++) {
	    int lag = lags[i*s];
	    assert(lag >= 0);
	    ret += lag;
	}
    }
    else {
	for (long i = 0; i < n; i++)
	    ret += _sum_lags(lags + i*s, nd-1, shape+1, strides+1);
    }

    return ret;
}


ReferenceLagbuf::ReferenceLagbuf(const gputils::Array<int> &lags_, int ntime_) :
    lags(lags_), ntime(ntime_)
{
    assert(ntime > 0);
    assert(lags.ndim > 0);

    int d = lags.ndim;
    long nr = _sum_lags(lags.data, d, lags.shape, lags.strides);
    
    expected_shape.resize(d+1);
    expected_shape[d] = ntime;
    for (int i = 0; i < d; i++)
	expected_shape[i] = lags.shape[i];

    this->rstate = Array<float> ({nr}, af_uhost | af_zero);
    this->scratch = Array<float> ({ntime}, af_uhost | af_zero);
}


// -------------------------------------------------------------------------------------------------
//
// ReferenceLagbuf::apply_lags()


// Applies (N-1)-dimensional lag array to N-dimensional data array.
static float *_apply_lags(float *data, const int *lags, float *rstate, float *scratch,
			  int data_ndim, const ssize_t *data_shape, const ssize_t *data_strides,
			  const ssize_t *lag_strides)
{
    if (data_ndim == 1) {
	int lag = *lags;
	int ntime = data_shape[0];
	int n = std::min(lag, ntime);
	
	// Inefficient for (lag >= ntime).
	memcpy(scratch, data + (ntime-n), n * sizeof(float));
	memmove(data+n, data, (ntime-n) * sizeof(float));
	memcpy(data, rstate, n * sizeof(float));
	memmove(rstate, rstate+n, (lag-n)*sizeof(float));
	memcpy(rstate + (lag-n), scratch, n * sizeof(float));
	
	return rstate + lag;
    }

    long n = data_shape[0];
    long ds = data_strides[0];
    long ls = lag_strides[0];

    for (long i = 0; i < n; i++) {
	rstate = _apply_lags(data + i*ds,     // data
			     lags + i*ls,     // lags
			     rstate,          // rstate
			     scratch,         // scratch
			     data_ndim-1,     // data_ndim
			     data_shape+1,    // shape
			     data_strides+1,  // data_strides
			     lag_strides+1);  // lag_strides
    }

    return rstate;
}


void ReferenceLagbuf::apply_lags(gputils::Array<float> &arr) const
{
    assert(arr.shape_equals(expected_shape));
    assert((arr.shape[arr.ndim-1] == 1) || (arr.strides[arr.ndim-1] == 1));
    
    float *rstate_end = _apply_lags(arr.data, lags.data, rstate.data, scratch.data, arr.ndim, arr.shape, arr.strides, lags.strides);				
    assert(rstate_end == rstate.data + rstate.size);
}


}  // namespace pirate
