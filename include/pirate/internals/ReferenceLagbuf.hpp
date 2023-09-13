#ifndef _PIRATE_INTERNALS_REFERENCE_LAGBUF_HPP
#define _PIRATE_INTERNALS_REFERENCE_LAGBUF_HPP

#include <vector>
#include <gputils/Array.hpp>


namespace pirate {
#if 0
}  // editor auto-indent
#endif


class ReferenceLagbuf
{
public:
    // ReferenceLagbuf: a very simple class which applies a channel-dependent lag
    // (specified by a length-nchan integer-valued vector of lags) incrementally to
    // an input array of shape (nchan, ntime).
    
    ReferenceLagbuf(const std::vector<int> &lags, int ntime);

    int nchan = 0; // lags.size()
    int ntime = 0;
    int nrstate = 0;

    // 2-d array of shape (nchan, ntime).
    // Lags are applied in place.
    void apply_lags(gputils::Array<float> &arr) const;
    void apply_lags(float *arr, int stride) const;

protected:
    std::vector<int> lags;  // length nchan
    gputils::Array<float> rstate;
    gputils::Array<float> scratch;
};
				       

}  // namespace pirate

#endif // _PIRATE_INTERNALS_REFERENCE_LAGBUF_HPP
