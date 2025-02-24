#include "../include/pirate/internals/utils.hpp"
#include "../include/pirate/internals/inlines.hpp"  // pow2()
#include "../include/pirate/constants.hpp"          // constants::max_tree_rank

#include <cassert>
#include <sstream>
#include <stdexcept>

using namespace std;
using namespace ksgpu;

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


// Only used in mean_bytes_per_unaligned_chunk().
int gcd(int m, int n)
{
    assert(m > 0);
    assert(n > 0);

    if (m > n)
	std::swap(m, n);

    while (m > 0) {
	int mold = m;
	m = n % m;
	n = mold;
    }

    return n;
}


int mean_bytes_per_unaligned_chunk(int nbytes)
{
    assert(nbytes > 0);
    assert(nbytes <= constants::bytes_per_gpu_cache_line);

    int g = gcd(nbytes, constants::bytes_per_gpu_cache_line);
    
    // int n = xdiv(constants::bytes_per_gpu_cache_line, g);
    // int m = xdiv(nbytes, g);
    //
    // n possible alignments
    //   (n-m+1) alignments produce 1 cache line
    //   (m-1) alignments produce 2 cache lines
    //
    // Expected number of cache lines = (n+m-1)/n
    // Expected number of bytes = (n+m-1)/n * (n*g) = (n*g + m*g - g)

    return constants::bytes_per_gpu_cache_line + nbytes - g;  // (n*g + m*g - g)
}


void reference_downsample_freq(const Array<float> &in, Array<float> &out, bool normalize)
{
    assert(out.ndim == 2);
    assert(out.strides[1] == 1);

    assert(in.shape_equals({ 2*out.shape[0], out.shape[1] }));
    assert(in.strides[1] == 1);

    float w = normalize ? 0.5 : 1.0;
    int nchan_out = out.shape[0];
    int nt = out.shape[1];

    for (int c = 0; c < nchan_out; c++) {
	const float *src_row0 = in.data + (2*c) * in.strides[0];
	const float *src_row1 = in.data + (2*c+1) * in.strides[0];
	float *dst_row = out.data + c * out.strides[0];

	for (int t = 0; t < nt; t++)
	    dst_row[t] = w * (src_row0[t] + src_row1[t]);
    }
}

    
void reference_downsample_time(const Array<float> &in, Array<float> &out, bool normalize)
{
    assert(out.ndim == 2);
    assert(out.strides[1] == 1);

    assert(in.shape_equals({ out.shape[0], 2*out.shape[1] }));
    assert(in.strides[1] == 1);

    float w = normalize ? 0.5 : 1.0;
    int nchan = out.shape[0];
    int nt_out = out.shape[1];

    for (int c = 0; c < nchan; c++) {
	const float *src_row = in.data + c * in.strides[0];
	float *dst_row = out.data + c * out.strides[0];

	for (int t = 0; t < nt_out; t++)
	    dst_row[t] = w * (src_row[2*t] + src_row[2*t+1]);
    }
}


void reference_extract_odd_channels(const Array<float> &in, Array<float> &out)
{
    assert(out.ndim == 2);
    assert(out.strides[1] == 1);

    assert(in.shape_equals({ 2*out.shape[0], out.shape[1] }));
    assert(in.strides[1] == 1);

    int nchan_out = out.shape[0];
    int nt = out.shape[1];

    for (int c = 0; c < nchan_out; c++) {
	memcpy(out.data + c * out.strides[0],
	       in.data + (2*c+1) * in.strides[0],
	       nt * sizeof(float));
    }
}


void lag_non_incremental(Array<float> &arr, const vector<int> &lags)
{
    assert(arr.ndim == 2);
    assert(arr.shape[0] == lags.size());
    assert(arr.strides[1] == 1);

    int nchan = arr.shape[0];
    int ntime = arr.shape[1];
	
    for (int c = 0; c < nchan; c++) {
	assert(lags[c] >= 0);
	int lag = std::min(lags[c], ntime);
	
	float *row = arr.data + c*arr.strides[0];
	memmove(row+lag, row, (ntime-lag) * sizeof(float));
	memset(row, 0, lag * sizeof(float));
    }
}


void dedisperse_non_incremental(Array<float> &arr)
{
    assert(arr.ndim == 2);

    int nfreq = arr.shape[0];
    int ntime = arr.shape[1];
    assert(nfreq > 0);
    assert(ntime > 0);
    assert((ntime ==1) || (arr.strides[1] == 1));
    
    int rank = integer_log2(nfreq);

    for (int r = 0; r < rank; r++) {
	int pr = pow2(r);
	
	for (int i = 0; i < nfreq; i += 2*pr) {
	    for (int j = 0; j < pr; j++) {
		float *row0 = arr.data + (i+j)*arr.strides[0];
		float *row1 = row0 + pr*arr.strides[0];
		
		int lag = bit_reverse_slow(j,r) + 1;
		float x0 = (ntime >= lag) ? row0[ntime-lag] : 0.0;
		
		for (int t = ntime-1; t >= 0; t--) {
		    float y = row1[t];
		    float x1 = x0;
		    x0 = (t >= lag) ? row0[t-lag] : 0.0;

		    row0[t] = x1 + y;
		    row1[t] = x0 + y;
		}
	    }
	}
    }
}


}  // namespace pirate
