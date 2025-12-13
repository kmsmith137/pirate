#ifndef _PIRATE_REFERENCE_TREE_HPP
#define _PIRATE_REFERENCE_TREE_HPP

#include "FrequencySubbands.hpp"

#include <vector>
#include <memory>  // shared_ptr
#include <ksgpu/Array.hpp>


namespace pirate {
#if 0
}  // editor auto-indent
#endif


// ReferenceTree: simple, self-contained reference implementation of tree dedispersion.
//
// Processes input incrementally in chunks of shape (..., nfreq, ntime), where "..."
// represents an arbitrary number of spectator axes. The time axis must be contiguous.
// The number of frequencies must be a power of two.
//
// The RefrerenceTree is unaware of the larger dedispersion plan (stage1/stage2 split,
// early triggers, downsampling, etc.) but can be used as a "building block" to implement
// these features.


class ReferenceTree
{
public:
    // Frequency axis is second-to-last, inner (ntime*nspec) axis is last.
    // All other indices are spectators.
    ReferenceTree(const std::vector<long> &shape, long nspec);
    ReferenceTree(int ndim, const long *shape, long nspec);
    
    // Dedispersion is done in-place.
    // Array shape is (Spectator indices) + (nfreq, ntime*nspec).
    void dedisperse(ksgpu::Array<float> &arr);

    // Morally equivalent to make_shared<ReferenceTree> (...).
    // (Necessary since make_shared doesn't seem to work with initializer lists.)
    static std::shared_ptr<ReferenceTree> make(std::initializer_list<long> shape, long nspec);
    
protected:
    // (Spectator indices) + (nfreq, ntime*nspec)
    std::vector<long> shape;

    int ndim = -1;
    int rank = -1;
    long nfreq = -1;
    long ntime = -1;
    long nspec = -1;
    long ninner = -1;   // = ntime*nspec
    long npstate = 0;
    
    ksgpu::Array<float> pstate;          // can be large
    ksgpu::Array<float> scratch;         // always small (length (ntime+1)*nspec)
};


// -------------------------------------------------------------------------------------------------
//
// FIXME: currently, 'struct ReferenceTree' and 'struct ReferenceTreeWithSubbands' are independent
// classes. They should be combined into one class (or at least share a base class).


struct ReferenceTreeWithSubbands
{
    struct Params
    {
        int num_beams = 0;
        int amb_rank = 0;
        int dd_rank = 0;
        int ntime = 0;
        std::vector<long> subband_counts;
    };

    Params params;
    FrequencySubbands fs;

    ksgpu::Array<float> pstate;
    ksgpu::Array<float> scratch;


    ReferenceTreeWithSubbands(const Params &params_);

    // Dedisperses 'buf' in place, writes subbands to 'out'.
    // buf.shape = { num_beams, 2^amb_rank, 2^dd_rank, ntime }.
    // out.shape = { num_beams, 2^(amb_rank + dd_rank - pf_rank), M, ntime }
    void dedisperse(ksgpu::Array<float> &buf, ksgpu::Array<float> &out);

    static void test();

    // Helper for dedisperse().
    // buf shape: (pow2(dd_rank), ntime).
    // out shape: (pow2(dd_rank-pf_rank), fs.M, ntime).
    float *dedisperse_2d(float *bufp, long buf_dstride, float *outp, long out_dstride, long out_mstride, float *ps);

    // Helper for dedisperse()
    // 'dst' and 'src' have shape (2, ntime).
    // dst==src is okay.
    inline float *dedisperse_1d(float *dst, long ds, float *src, long ss, float *ps, long lag);
};


}  // namespace pirate

#endif // _PIRATE_REFERENCE_TREE_HPP
