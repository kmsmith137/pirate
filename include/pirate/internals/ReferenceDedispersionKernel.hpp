#ifndef _PIRATE_INTERNALS_REFERENCE_DEDISPERSION_KERNEL_HPP
#define _PIRATE_INTERNALS_REFERENCE_DEDISPERSION_KERNEL_HPP

#include <gputils/Array.hpp>

namespace pirate {
#if 0
}  // editor auto-indent
#endif


struct ReferenceTree;    // defined in ReferenceTree.hpp
struct ReferenceLagbuf;  // defined in ReferenceLagbuf.hpp


struct ReferenceDedispersionKernel
{
    struct Params {
	int rank = -1;	
	int ntime = 0;
	int nambient = 0;
	int nbeams = 0;

	// The 'nelts_per_segment' argument has the same meaning as DedisperisonPlan::nelts_per_segment.
	// It affects the computation of residual lags, and is only used if apply_input_residual_lags == true.
	//
	// Similarly, 'is_downsampled_tree' affects residual lags, and only used if apply_input_residual_lags == true.
	//
	// (Note that the ReferenceDedispersionKernel always uses float32, unlike the GpuDedispersionPlan
	// where there is some connection between the dtype and the value of 'nelts_per_segment'.)
	
	bool apply_input_residual_lags = false;
	bool is_downsampled_tree = false;  // only used if apply_input_residual_lags == true
	int nelts_per_segment = 0;         // only used if apply_input_residual_lags == true
    };

    ReferenceDedispersionKernel(const Params &params_);

    // iobuf has shape (nbeams, nambient, pow2(rank), ntime).
    void apply(gputils::Array<float> &iobuf) const;

    const Params params;
    std::shared_ptr<ReferenceTree> tree;
    
    // Only used if params.apply_input_residual_lags == true.
    std::shared_ptr<ReferenceLagbuf> rlag_buf;
};


}  // namespace pirate

#endif // _PIRATE_INTERNALS_REFERENCE_DEDISPERSION_KERNEL_HPP
