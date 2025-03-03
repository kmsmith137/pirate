#include "../include/pirate/internals/ReferenceTree.hpp"

#include "../include/pirate/constants.hpp"
#include "../include/pirate/internals/inlines.hpp"  // pow2(), bit_reverse_slow()
#include "../include/pirate/internals/utils.hpp"    // check_rank()

#include <ksgpu/xassert.hpp>

using namespace std;
using namespace ksgpu;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// Assumes array has been transposed so that shape is (spectator_indices, nfreq, ntime)
static float *_dedisperse(float *arr, int ndim, const long *shape, const long *strides, int rank, int ntime, float *rp, float *sp)
{
    // Recursively reduce to the case ndim == 2.
    if (ndim > 2) {
	for (int i = 0; i < shape[0]; i++)
	    rp = _dedisperse(arr + i*strides[0], ndim-1, shape+1, strides+1, rank, ntime, rp, sp);
	return rp;
    }

    xassert(ndim == 2);    
    xassert(shape[0] == pow2(rank));
    xassert(shape[1] == ntime);
    xassert((ntime == 1) || (strides[1] == 1));

    long rstride = strides[0];
    
    for (int r = 0; r < rank; r++) {
	int ni = pow2(rank-r-1);
	int nj = pow2(r);

	// The index 'i' represents a coarse frequency.
	// The index 'j' represents a bit-reversed delay.
	
	for (int i = 0; i < ni; i++) {
	    for (int j = 0; j < nj; j++) {
		int row0 = i*(2*nj) + j;
		int row1 = row0 + nj;
		
		float *a0 = arr + (row0 * rstride);
		float *a1 = arr + (row1 * rstride);

		// FIXME precompute these!!!
		int lag = bit_reverse_slow(j,r) + 1;
		
		int n0 = min(lag, ntime+1);   // see "fill scratch" below, note (ntime+1) here
		int n1 = (ntime+1) - n0;      // see "fill scratch" below, note (ntime+1) here
		int m0 = max(lag-ntime, 0);   // see "advance ring buffer" below
		int m1 = lag - m0;            // see "advance ring buffer" below

		// Fill 'scratch' with (ntime+1) samples, obtained by applying 'lag'.
		memcpy(sp, rp, n0 * sizeof(float));
		memcpy(sp+n0, a0, n1 * sizeof(float));

		// Advance ring buffer.
		memmove(rp, rp+ntime, m0 * sizeof(float));
		memcpy(rp+m0, a0+ntime-m1, m1 * sizeof(float));
		rp += lag;

		// Apply dedispersion (sp,a1) -> (a0,a1).
		for (long it = 0; it < ntime; it++) {
		    float x0 = sp[it];
		    float x1 = sp[it+1];
		    float y = a1[it];
		    a0[it] = x1 + y;
		    a1[it] = x0 + y;
		}
	    }		
	}
    }

    return rp;
}

	
ReferenceTree::ReferenceTree(std::initializer_list<long> shape_) :
    ReferenceTree(shape_, shape_.size()-2)   // By default, the frequency axis is second-to-last.
{ }


ReferenceTree::ReferenceTree(std::initializer_list<long> shape_, int freq_axis_) :
    ReferenceTree(shape_.size(), shape_.begin(), freq_axis_)
{ }


ReferenceTree::ReferenceTree(int ndim_, const long *shape_, int freq_axis_) :
    freq_axis(freq_axis_), ndim(ndim_)
{
    xassert(ndim >= 2);
    xassert(freq_axis >= 0);
    xassert(freq_axis < ndim-1);
    
    this->shape.resize(ndim);
    this->nrstate = 1;

    for (int d = 0; d < ndim; d++) {
	shape[d] = shape_[d];
	xassert(shape[d] > 0);
	
	if ((d < ndim-1) && (d != freq_axis))
	    nrstate *= shape[d];
    }
    
    this->nfreq = shape[freq_axis];
    this->ntime = shape[ndim-1];
    
    xassert(is_power_of_two(nfreq));
    this->rank = integer_log2(nfreq);
    this->nrstate *= rstate_len(rank);   // rstate_len() is declared in utils.hpp
    
    this->tmp_shape.resize(ndim);
    this->tmp_strides.resize(ndim);
    this->rstate = Array<float> ({nrstate}, af_uhost | af_zero);
    this->scratch = Array<float> ({ntime+1}, af_uhost | af_zero);
}


void ReferenceTree::dedisperse(ksgpu::Array<float> &arr)
{
    xassert(arr.on_host());
    xassert(arr.shape_equals(this->shape));
    xassert((ntime == 1) || (arr.strides[ndim-1] == 1));
	
    if (rank == 0)
	return;

    for (int d = 0; d < ndim; d++) {
	tmp_shape[d] = arr.shape[d];
	tmp_strides[d] = arr.strides[d];
    }

    if (freq_axis != ndim-2) {
	std::swap(tmp_shape[freq_axis], tmp_shape[ndim-2]);
	std::swap(tmp_strides[freq_axis], tmp_strides[ndim-2]);
    }
    
    float *rp_end = _dedisperse(arr.data, ndim, &tmp_shape[0], &tmp_strides[0], rank, ntime, rstate.data, scratch.data);
    xassert(rp_end == rstate.data + nrstate);
}    


// static member function
shared_ptr<ReferenceTree> ReferenceTree::make(std::initializer_list<long> shape)
{
    return make_shared<ReferenceTree> (shape);
}

// static member function
shared_ptr<ReferenceTree> ReferenceTree::make(std::initializer_list<long> shape, int freq_axis)
{
    return make_shared<ReferenceTree> (shape, freq_axis);
}

	
}  // namespace pirate
