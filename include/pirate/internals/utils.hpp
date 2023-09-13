#ifndef _PIRATE_INTERNALS_UTILS_HPP
#define _PIRATE_INTERNALS_UTILS_HPP

#include <gputils/Array.hpp>

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

// Returns total ring buffer size needed for tree of given rank (assuming no padding).
extern ssize_t rstate_len(int rank);

// Returns total ring buffer size needed by the tree downsampler. (The
// "tree downsampler" operates on 2^(output_rank+1) input channels, and
// outputs 2^(output_rank) output channels.)
extern ssize_t rstate_ds_len(int output_rank);

// rb_lag(): returns lag needed for two-stage dedispersion.
// The index 0 <= i < pow2(rank1) represents a coarse frequency.
// The index 0 <= j < pow2(rank0) represents a **bit-reversed** delay.
// If uflag=true, then we're computing the upper half of a (rank0+rank1+1) tree.
extern int rb_lag(int i, int j, int rank0, int rank1, bool uflag=false);


// dedisperse_non_incremental(): currently only used for testing the ReferenceTree,
// but I could imagine this being useful elsewhere some day. Dedispersion is done in
// place -- output index is a bit-reversed delay.
extern void dedisperse_non_incremental(gputils::Array<float> &arr);

// Downsamples (freq,time) array by a factor 2 along either frequency or time axis.
// Each pair of elements will be averaged/summed, depending on whether the 'normalize' flag is true/false.
extern void reference_downsample_freq(const gputils::Array<float> &in, gputils::Array<float> &out, bool normalize);
extern void reference_downsample_time(const gputils::Array<float> &in, gputils::Array<float> &out, bool normalize);

// Reduces (dm_brev, time) array by a factor 2, by keeping only odd (dm_brev)-indices.
// FIXME if I ever implement Array<float>::slice() with strides, then this would be a special case.
extern void reference_extract_odd_channels(const gputils::Array<float> &in, gputils::Array<float> &out);

// lag_non_incremental() is only used for testing the ReferenceLagbuf.
// Lagging is done in place.
extern void lag_non_incremental(gputils::Array<float> &arr, const std::vector<int> &lags);


// Setup for mean_bytes_per_unaligned_chunk():
//
// We have a long array in GPU global memory
//
//   arr[nouter][nbytes];
//
// where:
//
//   nouter >> 1
//   0 < nbytes <= constants::bytes_per_gpu_cache_line
//   base address of 'arr' is aligned
//
// Suppose we read one "chunk" of the form arr[i,:]. Depending on the (assumed random) value
// of i, the memory controller will read either one or two cache lines. This function returns
// the expectation value (expressed as byte count, not cache line count).

extern int mean_bytes_per_unaligned_chunk(int nbytes);


}  // namespace pirate

#endif // _PIRATE_INTERNALS_UTILS_HPP

