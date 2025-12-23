#ifndef _PIRATE_UTILS_HPP
#define _PIRATE_UTILS_HPP

#include <string>
#include <ksgpu/Array.hpp>

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// Throws exception if rank is invalid (otherwise returns 'rank').
extern int check_rank(int rank, const char *where=nullptr, int min_rank=0);

// Arguments must satisfy 0 <= i < pow2(nbits).
extern int bit_reverse_slow(int i, int nbits);

// If n=2^r, returns value of r.
// If n is not a power of 2, throws an exception.
extern int integer_log2(long n);

// rb_lag(): returns lag needed for two-stage dedispersion.
// The index 0 <= i < pow2(rank1) represents a coarse frequency.
// The index 0 <= j < pow2(rank0) represents a **bit-reversed** delay.
// If uflag=true, then we're computing the upper half of a (rank0+rank1+1) tree.
extern int rb_lag(int i, int j, int rank0, int rank1, bool uflag=false);

// Downsamples (freq,time) array by a factor 2 along either frequency or time axis.
// Each pair of elements will be averaged/summed, depending on whether the 'normalize' flag is true/false.
extern void reference_downsample_freq(const ksgpu::Array<float> &in, ksgpu::Array<float> &out, bool normalize);
extern void reference_downsample_time(const ksgpu::Array<float> &in, ksgpu::Array<float> &out, bool normalize);

// Reduces (dm_brev, time) array by a factor 2, by keeping only odd (dm_brev)-indices.
// FIXME if I ever implement Array<float>::slice() with strides, then this would be a special case.
extern void reference_extract_odd_channels(const ksgpu::Array<float> &in, ksgpu::Array<float> &out);

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

