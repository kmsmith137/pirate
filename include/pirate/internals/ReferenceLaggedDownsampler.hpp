#ifndef _PIRATE_INTERNALS_REFERENCE_LAGGED_DOWNSAMPLER_HPP
#define _PIRATE_INTERNALS_REFERENCE_LAGGED_DOWNSAMPLER_HPP

#include <vector>
#include <memory>  // shared_ptr
#include <gputils/Array.hpp>


namespace pirate {
#if 0
}  // editor auto-indent
#endif

// Defined in include/pirate/internals/ReferenceLagbuf.hpp
class ReferenceLagbuf;


struct ReferenceLaggedDownsampler
{
    // A potential source of confusion: denote
    //
    //    ld_nds = GpuLaggedDownsampler::Params::num_downsmapling_levels
    //    dc_nds = DedispersionConfig::num_downsampling_levels
    //
    // Then ld_nds = (dc_nds - 1)!
    
    struct Params
    {
	int small_input_rank = -1;
	int large_input_rank = -1;
	int num_downsampling_levels = -1;
	int nbeams = 0;
	long ntime = 0;
    };

    ReferenceLaggedDownsampler(const Params &params);

    // apply(): the input/output arrays have the following shapes:
    //
    //   in.shape = (nbeams, 2^large_input_rank, ntime)
    //   out[i].shape = (nbeams, 2^(large_input_rank-1), ntime/2^(i+1))
    //   out.size() = Params::num_downsampling_levels
    //
    // Note: if nbeams == 1, then the beam axis can be omitted, i.e. the following are okay:
    //   in.shape = (2^large_input_rank, ntime)
    //   out[i].shape = (2^(large_input_rank-1), ntime/2^(i+1))
    //
    // Note: the ReferenceLaggedDownsampler stores all ring buffer state needed to call
    // apply() incrementally.
    
    void apply(const gputils::Array<float> &in, std::vector<gputils::Array<float>> &out);
    void apply(const gputils::Array<float> &in, gputils::Array<float> *outp);

    const Params params;

    std::shared_ptr<ReferenceLagbuf> lagbuf_small;
    std::shared_ptr<ReferenceLagbuf> lagbuf_large;
    std::shared_ptr<ReferenceLaggedDownsampler> next;
};


}  // namespace pirate

#endif // _PIRATE_INTERNALS_REFERENCE_LAGGED_DOWNSAMPLER_HPP
