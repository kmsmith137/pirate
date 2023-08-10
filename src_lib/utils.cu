#include "../include/pirate/internals/utils.hpp"
#include "../include/pirate/internals/inlines.hpp"  // pow2()
#include "../include/pirate/constants.hpp"          // constants::max_tree_rank

#include <cassert>
#include <sstream>
#include <stdexcept>

using namespace std;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


int check_rank(int rank, const char *where, int min_rank)
{
    if ((rank >= min_rank) && (rank <= constants::max_tree_rank))
	return rank;

    if (!where)
	where = "check_rank";
	    
    stringstream ss;
    ss << where << ": rank=" << rank << " is out-of-range (min_rank=" << min_rank
       << ", max_rank=" << constants::max_tree_rank << ")";
    
    throw runtime_error(ss.str());
}


int bit_reverse_slow(int i, int nbits)
{
    assert((nbits >= 0) && (nbits <= 30));
    assert((i >= 0) && (i < (1 << nbits)));
    
    int j = 0;
    
    while (nbits > 0) {
	j = (j << 1) | (i & 1);
	i >>= 1;
	nbits--;
    }

    return j;
}


extern int integer_log2(long n)
{
    float f = (n > 0) ? (1.414f * n) : 1.0f;
    int p = log2f(f);

    // If this fails, then n is not a power of 2.
    assert(n == (1L << p));

    return p;
}


int rb_lag(int i, int j, int rank0, int rank1, bool uflag)
{
    assert(rank0 >= 0);
    assert(rank1 >= 0);
    assert((rank0+rank1) <= constants::max_tree_rank);

    int n0 = 1 << rank0;
    int n1 = 1 << rank1;
    
    assert((i >= 0) && (i < n1));
    assert((j >= 0) && (j < n0));

    int dm = bit_reverse_slow(j, rank0);
    
    if (uflag)
	dm += n0;

    int lag = (n1-1-i) * dm;
    assert(lag >= 0);

    return lag;
}


ssize_t rstate_len(int rk)
{
    check_rank(rk, "rstate_len");
    
    if (rk <= 1)
	return rk;  // Covers cases rk=0, rk=1.
    
    return pow2(2*rk-2) + (rk-1) * pow2(rk-2);
}


// FIXME needs unit test.
ssize_t rstate_ds_len(int rk)
{
    check_rank(rk, "rstate_ds_len");
    
    int nchan = pow2(rk);
    return (nchan*(nchan+1))/2;
}


DedispersionConfig make_chord_dedispersion_config(const string &compressed_dtype, const string &uncompressed_dtype)
{
    DedispersionConfig config;
    config.tree_rank = 15;
    config.num_downsampling_levels = 5;
    config.time_samples_per_chunk = 2048;
    //config.num_downsampling_levels = 1;

    config.uncompressed_dtype = uncompressed_dtype;
    config.compressed_dtype = compressed_dtype;
    
    config.beams_per_gpu = 128;
    config.beams_per_batch = 2;    // ?
    config.num_active_batches = 2;  // ?
    config.gmem_nbytes_per_gpu = 32L * 1024L * 1024L * 1024L;  // A40 assumed
    
    config.add_early_triggers(1, {13});
    config.add_early_triggers(2, {12,13});
    config.add_early_triggers(3, {11,12,13});
    config.add_early_triggers(4, {10,11,12,13});
    // config.force_ring_buffers_to_host = true;

    config.validate();
    return config;
}


}  // namespace pirate