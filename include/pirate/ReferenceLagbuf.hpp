#ifndef _PIRATE_REFERENCE_LAGBUF_HPP
#define _PIRATE_REFERENCE_LAGBUF_HPP

#include <vector>
#include <ksgpu/Array.hpp>


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
    
    ReferenceLagbuf(const ksgpu::Array<int> &lags, int ntime);

    std::vector<long> expected_shape;  // (n_0, ..., n_{d-1}, T)
    ksgpu::Array<int> lags;            // shape (n_0, ..., n_{d-1})
    int ntime = 0;

    void apply_lags(ksgpu::Array<float> &arr) const;

    static void test();

protected:
    ksgpu::Array<float> rstate;
    ksgpu::Array<float> scratch;
};


}  // namespace pirate

#endif // _PIRATE_REFERENCE_LAGBUF_HPP
