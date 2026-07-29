#ifndef _PIRATE_DETRENDER_1D_HPP
#define _PIRATE_DETRENDER_1D_HPP

#include <tuple>
#include <vector>
#include <cuda_runtime.h>
#include <ksgpu/Array.hpp>

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// Detrender1d: the 1-d time detrender, a masked, adaptively centered moving local
// polynomial fit. The algorithm is specified in notes/tree_dedispersion.tex, section
// "Time detrending algorithm 1: local polynomial subtraction"; pirate_frb/detrending_1d
// is the pure-numpy reference that this kernel is validated against.
//
// Operates in place on a (data, mask) pair, independently for each row (one row per
// (beam, freq) pair). For each output sample t, a degree-n polynomial is fit to the
// valid samples of the window [t-W, t+W] and evaluated back at t; the fit is
// subtracted from the data, and the sample is dropped ("mask expansion") if its
// window is too ill-conditioned to determine the fit.
//
// The kernel writes only the middle T samples of each row, i.e. buffer samples
// [W, W+T). The 2W padding samples are read but not written, and the caller is
// responsible for the buffer shift between chunks (each output sample is committed
// once and forever, so the detrender never revisits it).
//
// Where the expanded mask is false, the residual is written as zero rather than
// left untouched. This matches Detrender.detrend_chunk() in the numpy reference,
// and means that a consumer which forgets to check the mask sees zeros rather than
// raw un-detrended intensity.
//
// (n, W, T) are compile-time parameters of the cuda kernel, so only the combinations
// listed in the constructor's error message exist. The number of rows M is runtime.

struct Detrender1d
{
    // Throws unless (n, W, T) is one of the compiled configurations.
    Detrender1d(long n, long W, long T = 2048);

    // Mask-expansion threshold on the conditioning statistic rmin, and the NaN guard
    // used where rmin is below it. Neither is a regularizer: see the "Cholesky, and
    // the conditioning statistic" discussion in notes/tree_dedispersion.tex.
    //
    // Note eps is inert at n=1, where rmin is {0,1}-valued and mask expansion reduces
    // to "the window holds at least 2 valid samples".
    static constexpr float eps = 1.0e-3f;
    static constexpr float mu = 1.0e-30f;

    const long n;      // polynomial degree
    const long W;      // window half-width (window is 2W+1 samples)
    const long T;      // output samples per row (chunk size)
    const long nbuf;   // buffer samples per row, = T + 2W

    // launch(): asynchronously launch kernel, and return without synchronizing stream.
    // Note: stream=NULL is allowed, but is not the default.
    //
    //   data: shape (M, nbuf), float32, fully contiguous, on GPU. Modified in place.
    //   mask: shape (M, nbuf), uint8, fully contiguous, on GPU. Modified in place,
    //         and {0,1}-valued on both input and output.
    //
    // The caller must treat the output mask as the authoritative one: it is the input
    // mask with the ill-conditioned windows removed, so it can only lose samples.
    void launch(ksgpu::Array<float> &data,
                ksgpu::Array<unsigned char> &mask,
                cudaStream_t stream) const;

    // The compiled (n, W, T) configurations, i.e. the arguments the constructor accepts.
    static std::vector<std::tuple<long,long,long>> configs();

    // Static timing function (called via 'python -m pirate_frb time --dt1d').
    // Times every compiled configuration.
    static void time_selected();
};


}  // namespace pirate

#endif // _PIRATE_DETRENDER_1D_HPP
