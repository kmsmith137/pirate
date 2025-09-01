#include "../include/pirate/ReferenceTree.hpp"
#include "../include/pirate/constants.hpp"
#include "../include/pirate/inlines.hpp"  // pow2(), bit_reverse_slow()
#include "../include/pirate/utils.hpp"    // check_rank()

#include <ksgpu/xassert.hpp>

using namespace std;
using namespace ksgpu;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// Assumes array has been transposed so that shape is (spectator_indices, nfreq, ntime*nspec)
static float *_dedisperse(float *arr, int ndim, const long *shape, const long *strides, int rank, long ntime, long nspec, float *rp, float *sp)
{
    // Recursively reduce to the case ndim == 2.
    if (ndim > 2) {
	for (long i = 0; i < shape[0]; i++)
	    rp = _dedisperse(arr + i*strides[0], ndim-1, shape+1, strides+1, rank, ntime, nspec, rp, sp);
	return rp;
    }

    xassert(ndim == 2);    
    xassert(shape[0] == pow2(rank));
    xassert(shape[1] == ntime * nspec);
    xassert((shape[1] == 1) || (strides[1] == 1));

    long ninner = ntime * nspec;
    long rstride = strides[0];
    
    for (int r = 0; r < rank; r++) {
	long ni = pow2(rank-r-1);
	long nj = pow2(r);

	// The index 'i' represents a coarse frequency.
	// The index 'j' represents a bit-reversed delay.
	
	for (long i = 0; i < ni; i++) {
	    for (long j = 0; j < nj; j++) {
		long row0 = i*(2*nj) + j;
		long row1 = row0 + nj;
		
		float *a0 = arr + (row0 * rstride);
		float *a1 = arr + (row1 * rstride);

		// FIXME precompute these!!!
		long lag1 = bit_reverse_slow(j,r) * nspec;
		long lag0 = lag1 + nspec;

		// Fill 'scratch' with n=(ntime+1)*nspec samples, obtained by applying lag0.
		long n = (ntime+1) * nspec;   // total samples in scratch buffer
		long n0 = min(lag0, n);       // number of samples which come from ring buffer
		long n1 = n - n0;             // number of samples which come from 'a0'
		memcpy(sp, rp, n0 * sizeof(float));
		memcpy(sp+n0, a0, n1 * sizeof(float));

		// Advance ring buffer (length lag0) by ninner samples.
		long m0 = max(lag0-ninner, 0L);  // number of samples which stay in ring buffer
		long m1 = lag0 - m0;             // number of samples which come from 'a0'
		memmove(rp, rp+ninner, m0 * sizeof(float));
		memcpy(rp+m0, a0+ninner-m1, m1 * sizeof(float));
		rp += lag0;

		// Apply dedispersion (sp,a1) -> (a0,a1).
		for (long k = 0; k < ninner; k++) {
		    float x0 = sp[k];
		    float x1 = sp[k+nspec];
		    float y = a1[k];
		    a0[k] = x1 + y;
		    a1[k] = x0 + y;
		}
	    }		
	}
    }

    return rp;
}


ReferenceTree::ReferenceTree(const std::vector<long> &shape_, long nspec_) :
    ReferenceTree(shape_.size(), &shape_[0], nspec_)
{ }


ReferenceTree::ReferenceTree(int ndim_, const long *shape_, long nspec_) :
    ndim(ndim_), nspec(nspec_)
{
    xassert(ndim >= 2);
    xassert(nspec > 0);
    xassert_divisible(shape_[ndim-1], nspec);
    
    this->shape.resize(ndim);

    for (int d = 0; d < ndim; d++) {
	shape[d] = shape_[d];
	xassert(shape[d] > 0);
    }
    
    this->nfreq = shape[ndim-2];
    this->ntime = xdiv(shape[ndim-1], nspec);
    
    xassert(is_power_of_two(nfreq));
    this->rank = integer_log2(nfreq);

    this->npstate = rstate_len(rank) * nspec;   // rstate_len() is declared in utils.hpp
    for (int d = 0; d < ndim-2; d++)
	npstate *= shape[d];
    
    this->pstate = Array<float> ({npstate}, af_uhost | af_zero);
    this->scratch = Array<float> ({(ntime+1)*nspec}, af_uhost | af_zero);
}


void ReferenceTree::dedisperse(ksgpu::Array<float> &arr)
{
    xassert(arr.on_host());
    xassert(arr.shape_equals(this->shape));
    xassert((ntime == 1) || (arr.strides[ndim-1] == 1));
	
    if (rank > 0) {
	float *rp_end = _dedisperse(arr.data, ndim, arr.shape, arr.strides, rank, ntime, nspec, pstate.data, scratch.data);
	xassert(rp_end == pstate.data + npstate);
    }
}    


// static member function
shared_ptr<ReferenceTree> ReferenceTree::make(std::initializer_list<long> shape, long nspec_)
{
    return make_shared<ReferenceTree> (shape.size(), shape.begin(), nspec_);
}

	
}  // namespace pirate
