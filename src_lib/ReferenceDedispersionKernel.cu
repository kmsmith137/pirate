#include "../include/pirate/internals/ReferenceDedispersionKernel.hpp"
#include "../include/pirate/internals/ReferenceLagbuf.hpp"
#include "../include/pirate/internals/ReferenceTree.hpp"
#include "../include/pirate/internals/inlines.hpp"     // pow2()
#include "../include/pirate/internals/utils.hpp"       // bit_reverse_slow()

using namespace std;
using namespace gputils;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


ReferenceDedispersionKernel::ReferenceDedispersionKernel(const Params &params_) :
    params(params_)
{
    // FIXME should have proper argument checking here.
    // Right now, I'm just making sure that everything is initialized.
	
    assert(params.rank >= 0);
    assert(params.ntime > 0);
    assert(params.nambient > 0);
    assert(params.nbeams > 0);
    assert(is_power_of_two(params.nambient));
    
    int B = params.nbeams;
    int A = params.nambient;
    int F = pow2(params.rank);
    int Ar = integer_log2(A);

    this->tree = ReferenceTree::make({B,A,F,params.ntime});

    if (!params.apply_input_residual_lags)
	return;

    // Remaining code initializes this->rlag_buf, in case (params.apply_input_residual_lags == false).

    if (params.nelts_per_segment <= 0)
	throw runtime_error("ReferenceDedispersionKernel: if params.apply_input_residual_lags==true,"
			    " then params.nelts_per_segment must be initialized and > 0" );
    
    Array<int> rlags({B,A,F}, af_uhost);
    
    for (int b = 0; b < B; b++) {
	for (int a = 0; a < A; a++) {
	    // Ambient index 'a' represents a bit-reversed coarse DM.
	    // Index 'f' represents a fine frequency.
	    for (int f = 0; f < F; f++) {
		int lag = rb_lag(f, a, Ar, params.rank, params.is_downsampled_tree);
		rlags.data[b*A*F + a*F + f] = lag % params.nelts_per_segment;  // residual lag
	    }
	}
    }
    
    this->rlag_buf = make_shared<ReferenceLagbuf> (rlags, params.ntime);
}


void ReferenceDedispersionKernel::apply(Array<float> &iobuf) const
{
    int B = params.nbeams;
    int A = params.nambient;
    int F = pow2(params.rank);

    assert(iobuf.shape_equals({B,A,F,params.ntime}));

    if (params.apply_input_residual_lags)
	rlag_buf->apply_lags(iobuf);

    tree->dedisperse(iobuf);
}


} // namespace pirate
