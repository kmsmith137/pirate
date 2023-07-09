#ifndef _PIRATE_INTERNALS_UTILS_HPP
#define _PIRATE_INTERNALS_UTILS_HPP

#include "../DedispersionConfig.hpp"

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

// Returns my current guess for the CHORD dedispersion config.
// FIXME will go away, after I define a YAML format for configs.
extern DedispersionConfig
make_chord_dedispersion_config(const std::string &compressed_dtype = "int8",
			       const std::string &uncompressed_dtype = "float32");


}  // namespace pirate

#endif // _PIRATE_INTERNALS_UTILS_HPP

