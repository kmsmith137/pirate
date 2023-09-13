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
    struct Params
    {
	int small_input_rank = 0;
	int large_input_rank = 0;
	int num_downsampling_levels = 0;
	int nbeams = 0;
	long ntime = 0;
    };

    const Params params;

    std::shared_ptr<ReferenceLagbuf> lagbuf_small;
    std::shared_ptr<ReferenceLagbuf> lagbuf_large;
    std::shared_ptr<ReferenceLaggedDownsampler> next;

    ReferenceLaggedDownsampler(const Params &params);
    
    void apply(const gputils::Array<float> &in, std::vector<gputils::Array<float>> &out);
    void apply(const gputils::Array<float> &in, gputils::Array<float> *outp);
};


}  // namespace pirate

#endif // _PIRATE_INTERNALS_REFERENCE_LAGGED_DOWNSAMPLER_HPP
