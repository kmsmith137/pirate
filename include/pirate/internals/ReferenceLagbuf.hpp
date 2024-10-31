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
    // ReferenceLagbuf: applies a lag incrementally to an input array of shape
    // (n_0, n_1, ..., n_{d-1}, T). The lags are specified as an integer-valued
    // array of shape (n_0, n_1, ..., n_{d-1}).
    
    ReferenceLagbuf(const gputils::Array<int> &lags, int ntime);

    std::vector<ssize_t> expected_shape;  // (n_0, ..., n_{d-1}, T)
    int ntime = 0;

    void apply_lags(gputils::Array<float> &arr) const;

protected:
    gputils::Array<int> lags;     // shape (n_0, ..., n_{d-1})
    gputils::Array<float> rstate;
    gputils::Array<float> scratch;
};


}  // namespace pirate

#endif // _PIRATE_INTERNALS_REFERENCE_LAGBUF_HPP
