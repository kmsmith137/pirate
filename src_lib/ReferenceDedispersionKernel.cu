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
    params.validate();

    long B = params.beams_per_kernel_launch;
    long A = params.nambient;
    long F = pow2(params.rank);
    long T = params.ntime;
    long Ar = integer_log2(A);
    long N = xdiv(params.total_beams, B);

    this->trees.resize(N);
    for (long n = 0; n < N; n++)
	trees[n] = ReferenceTree::make({B,A,F,T});

    if (!params.apply_input_residual_lags)
	return;

    // Remaining code initializes this->rlag_bufs (only if params.apply_input_residual_lags == false).

    if (params.nelts_per_segment <= 0) {
	throw runtime_error("ReferenceDedispersionKernel: if params.apply_input_residual_lags==true,"
			    " then params.nelts_per_segment must be initialized and > 0" );
    }
    
    Array<int> rlags({B,A,F}, af_uhost);
    
    for (long b = 0; b < B; b++) {
	for (long a = 0; a < A; a++) {
	    // Ambient index 'a' represents a bit-reversed coarse DM.
	    // Index 'f' represents a fine frequency.
	    for (long f = 0; f < F; f++) {
		long lag = rb_lag(f, a, Ar, params.rank, params.input_is_downsampled_tree);
		rlags.data[b*A*F + a*F + f] = lag % params.nelts_per_segment;  // residual lag
	    }
	}
    }
    
    this->rlag_bufs.resize(N);
    for (long n = 0; n < N; n++)
	rlag_bufs[n] = make_shared<ReferenceLagbuf> (rlags, T);
}


void ReferenceDedispersionKernel::apply(Array<float> &in, Array<float> &out, long itime, long ibeam)
{
    long B = params.beams_per_kernel_launch;
    long A = params.nambient;
    long F = pow2(params.rank);
    long T = params.ntime;

    assert(in.shape_equals({B,A,F,T}));
    assert(out.shape_equals({B,A,F,T}));

    // Compare (itime, ibeam) with expected values.
    assert(itime == expected_itime);
    assert(ibeam == expected_ibeam);

    // Update expected (itime, ibeam).
    expected_ibeam += B;
    assert(expected_ibeam <= params.total_beams);
    
    if (expected_ibeam == params.total_beams) {
	expected_ibeam = 0;
	expected_itime++;
    }

    if (out.data != in.data)
	out.fill(in);

    int n = xdiv(ibeam, B);
    
    if (params.apply_input_residual_lags)
	rlag_bufs.at(n)->apply_lags(out);

    trees.at(n)->dedisperse(out);
}


} // namespace pirate
