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

    // The 'in' and 'out' arrays are either dedispersion buffers or ringbufs, depending on
    // values of Params::input_is_ringbuf and Params::output_is_ringbuf. Shapes are:
    //
    //   - dedispersion buffer has shape (params.beams_per_kernel, nambient, pow2(rank), ntime).
    //   - ringbuf has 1-d shape (params.ringbuf_nseg * params.nelts_per_segment,)
    //
    // The 'itime' and 'ibeam' arguments are not logically necessary, but enable a debug check.
    // Warning: if 'in' is not a ringbuf, then apply() may modify 'in'!
    
    void apply(gputils::Array<float> &in, gputils::Array<float> &out, long itime, long ibeam);

    // A bit awkward -- number of trees is (total_beams / beams_per_kernel).
    std::vector<std::shared_ptr<ReferenceTree>> trees;
    std::vector<std::shared_ptr<ReferenceLagbuf>> rlag_bufs;   // only used if params.apply_input_residual_lags == true.

    // Enables a debug check.
    long expected_itime = 0;
    long expected_ibeam = 0;

    // Used internally
    void _copy_to_ringbuf(const gputils::Array<float> &in, gputils::Array<float> &out, long rb_pos);
    void _copy_from_ringbuf(const gputils::Array<float> &in, gputils::Array<float> &out, long rb_pos);
};


}  // namespace pirate

#endif // _PIRATE_INTERNALS_REFERENCE_DEDISPERSION_KERNEL_HPP
