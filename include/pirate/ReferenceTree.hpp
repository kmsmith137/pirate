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

// See ReferenceLagbuf.{hpp,cu}.
class ReferenceLagbuf;


// ReferenceTree: reference implementation of tree dedispersion.
//
// Processes input incrementally in chunks of shape 
//   (nbeams, pow2(amb_rank), pow2(dd_rank), ntime * nspec).
//
// The RefrerenceTree is unaware of the larger dedispersion plan (stage1/stage2 split,
// early triggers, downsampling, etc.) but can be used as a "building block" to implement
// these features.


struct ReferenceTree
{
    struct Params
    {
        long num_beams = 0;
        long amb_rank = 0;
        long dd_rank = 0;
        long ntime = 0;
        long nspec = 0;
        std::vector<long> subband_counts;
    };

    Params params;
    FrequencySubbands frequency_subbands;

    std::vector<float> pstate;   // sized lazily on the first dedisperse() (see dedisperse_1d)
    ksgpu::Array<float> scratch;

    std::shared_ptr<ReferenceLagbuf> final_lagbuf;

    ReferenceTree(const Params &params_);

    // Dedisperses 'buf' in place, and writes subbands to 'out'.
    //   buf.shape = { num_beams, 2^amb_rank, 2^dd_rank, ntime * nspec}.
    //   out.shape = { num_beams, 2^(amb_rank + dd_rank - pf_rank), M, ntime * nspec }.
    //
    // Note: if M=1 (no subbands), then the 'out' argument is optional, and
    // an empty (size-zero) array can be passed instead.

    void dedisperse(ksgpu::Array<float> &buf, ksgpu::Array<float> &out);

    static void test_basics();
    static void test_subbands();
    
    // Helper for dedisperse().
    // buf shape: (pow2(dd_rank), ntime * nspec).
    // out shape: (pow2(dd_rank-pf_rank), fs.M, ntime * nspec). Can be NULL.
    // 'ps_used' is the current offset into 'pstate'; returns the updated offset.
    long dedisperse_2d(float *bufp, long buf_dstride, float *outp, long out_dstride, long out_mstride, long ps_used);

    // Helper for dedisperse()
    // 'dst' and 'src' have shape (2, ntime * nspec), and dst==src is okay.
    // 'ps_used' is the current offset into 'pstate' (grown here if needed); returns the updated offset.
    inline long dedisperse_1d(float *dst, long ds, float *src, long ss, long ps_used, long lag);
};


}  // namespace pirate

#endif // _PIRATE_REFERENCE_TREE_HPP
