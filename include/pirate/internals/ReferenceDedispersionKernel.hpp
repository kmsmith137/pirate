#ifndef _PIRATE_INTERNALS_REFERENCE_DEDISPERSION_KERNEL_HPP
#define _PIRATE_INTERNALS_REFERENCE_DEDISPERSION_KERNEL_HPP

#include "GpuDedispersionKernel.hpp"  // GpuDedispersionKernel::Params

namespace pirate {
#if 0
}  // editor auto-indent
#endif


struct ReferenceTree;    // defined in ReferenceTree.hpp
struct ReferenceLagbuf;  // defined in ReferenceLagbuf.hpp


struct ReferenceDedispersionKernel
{
    using Params = GpuDedispersionKernel::Params;

    ReferenceDedispersionKernel(const Params &params);

    const Params params;
    
    // The 'in' and 'out' arrays have shape (params.beams_per_kernel, nambient, pow2(rank), ntime).
    // The 'itime' and 'ibeam' arguments are not logically necessary, but enable a debug check.
    void apply(gputils::Array<float> &in, gputils::Array<float> &out, long itime, long ibeam);

    // A bit awkward -- number of trees is (total_beams / beams_per_kernel).
    std::vector<std::shared_ptr<ReferenceTree>> trees;
    std::vector<std::shared_ptr<ReferenceLagbuf>> rlag_bufs;   // only used if params.apply_input_residual_lags == true.

    // Enables a debug check.
    long expected_itime = 0;
    long expected_ibeam = 0;    
};


}  // namespace pirate

#endif // _PIRATE_INTERNALS_REFERENCE_DEDISPERSION_KERNEL_HPP
