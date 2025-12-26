#ifndef _PIRATE_UTILS_HPP
#define _PIRATE_UTILS_HPP

#include <string>
#include <ksgpu/Array.hpp>

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// Arguments must satisfy 0 <= i < pow2(nbits).
extern int bit_reverse_slow(int i, int nbits);

// If n=2^r, returns value of r.
// If n is not a power of 2, throws an exception.
extern int integer_log2(long n);

// rb_lag(): returns lag needed for two-stage dedispersion.
// The index 0 <= freq_coarse < pow2(stage2_rank) represents a coarse frequency.
// The index 0 <= dm_brev < pow2(stage1_rank) represents a **bit-reversed** delay.
// If uflag=true, then we're computing the upper half of a (stage1_rank+stage2_rank+1) tree.
extern int rb_lag(int freq_coarse, int dm_brev, int stage1_rank, int stage2_rank, bool uflag=false);

// Downsamples (freq,time) array by a factor 2 along either frequency or time axis.
// Each pair of elements will be averaged/summed, depending on whether the 'normalize' flag is true/false.
extern void reference_downsample_freq(const ksgpu::Array<float> &in, ksgpu::Array<float> &out, bool normalize);
extern void reference_downsample_time(const ksgpu::Array<float> &in, ksgpu::Array<float> &out, bool normalize);

// dedisperse_non_incremental(): currently only used for testing the ReferenceTree,
// but I could imagine this being useful elsewhere some day. Dedispersion is done in
// place -- output index is a bit-reversed delay.
//
// Note: Input is a 2-d array with shape (nfreq, ntime*nspec).

extern void dedisperse_non_incremental(ksgpu::Array<float> &arr, long nspec);

// dedispersion_delay(): returns the dedispersion delay for a given (freq, dm_brev) pair.
// Used for testing.
extern long dedispersion_delay(int rank, long freq, long dm_brev);

extern std::string hex_str(uint x);


}  // namespace pirate

#endif // _PIRATE_UTILS_HPP

