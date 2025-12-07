#include "../include/pirate/utils.hpp"
#include "../include/pirate/inlines.hpp"    // pow2(), xdiv()
#include "../include/pirate/constants.hpp"  // constants::max_tree_rank

#include <sstream>
#include <stdexcept>
#include <iomanip>

#include <ksgpu/xassert.hpp>

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
    xassert((nbits >= 0) && (nbits <= 30));
    xassert((i >= 0) && (i < (1 << nbits)));
    
    int j = 0;
    
    while (nbits > 0) {
        j = (j << 1) | (i & 1);
        i >>= 1;
        nbits--;
    }

    return j;
}


int integer_log2(long n)
{
    float f = (n > 0) ? (1.414f * n) : 1.0f;
    int p = log2f(f);

    // If this fails, then n is not a power of 2.
    xassert(n == (1L << p));

    return p;
}

int rb_lag(int i, int j, int rank0, int rank1, bool uflag)
{
    xassert(rank0 >= 0);
    xassert(rank1 >= 0);
    xassert((rank0+rank1) <= constants::max_tree_rank);

    int n0 = 1 << rank0;
    int n1 = 1 << rank1;
    
    xassert((i >= 0) && (i < n1));
    xassert((j >= 0) && (j < n0));

    int dm = bit_reverse_slow(j, rank0);
    
    if (uflag)
        dm += n0;

    int lag = (n1-1-i) * dm;
    xassert(lag >= 0);

    return lag;
}


long rstate_len(int rk)
{
    check_rank(rk, "rstate_len");
    
    if (rk <= 1)
        return rk;  // Covers cases rk=0, rk=1.
    
    return pow2(2*rk-2) + (rk-1) * pow2(rk-2);
}


// FIXME needs unit test.
long rstate_ds_len(int rk)
{
    check_rank(rk, "rstate_ds_len");
    
    int nchan = pow2(rk);
    return (nchan*(nchan+1))/2;
}


// Only used in mean_bytes_per_unaligned_chunk().
int gcd(int m, int n)
{
    xassert(m > 0);
    xassert(n > 0);

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
    xassert(nbytes > 0);
    xassert(nbytes <= constants::bytes_per_gpu_cache_line);

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
    xassert(out.ndim == 2);
    xassert(out.strides[1] == 1);

    xassert(in.shape_equals({ 2*out.shape[0], out.shape[1] }));
    xassert(in.strides[1] == 1);

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
    xassert(out.ndim == 2);
    xassert(out.strides[1] == 1);

    xassert(in.shape_equals({ out.shape[0], 2*out.shape[1] }));
    xassert(in.strides[1] == 1);

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
    xassert(out.ndim == 2);
    xassert(out.strides[1] == 1);

    xassert(in.shape_equals({ 2*out.shape[0], out.shape[1] }));
    xassert(in.strides[1] == 1);

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
    xassert(arr.ndim == 2);
    xassert(arr.shape[0] == long(lags.size()));
    xassert(arr.strides[1] == 1);

    int nchan = arr.shape[0];
    int ntime = arr.shape[1];
        
    for (int c = 0; c < nchan; c++) {
        xassert(lags[c] >= 0);
        int lag = std::min(lags[c], ntime);
        
        float *row = arr.data + c*arr.strides[0];
        memmove(row+lag, row, (ntime-lag) * sizeof(float));
        memset(row, 0, lag * sizeof(float));
    }
}


void dedisperse_non_incremental(Array<float> &arr, long nspec)
{
    xassert(arr.ndim == 2);
    long nfreq = arr.shape[0];
    long ninner = arr.shape[1];
    
    xassert(nspec > 0);
    xassert(nfreq > 0);
    xassert(ninner > 0);
    xassert((ninner == 1) || (arr.strides[1] == 1));
    xassert(is_power_of_two(nfreq));
    xassert_divisible(ninner, nspec);
    
    int rank = integer_log2(nfreq);
    // long ntime = xdiv(ninner, nspec);   // not actually needed

    for (int r = 0; r < rank; r++) {
        int pr = pow2(r);
        
        for (int i = 0; i < nfreq; i += 2*pr) {
            for (int j = 0; j < pr; j++) {
                float *row0 = arr.data + (i+j)*arr.strides[0];
                float *row1 = row0 + pr*arr.strides[0];
                
                long lag1 = bit_reverse_slow(j,r) * nspec;
                long lag0 = lag1 + nspec;

                for (int k = ninner-1; k >= 0; k--) {
                    float x0 = (k >= lag0) ? row0[k-lag0] : 0.0f;
                    float x1 = (k >= lag1) ? row0[k-lag1] : 0.0f;
                    float y = row1[k];

                    row0[k] = x1 + y;
                    row1[k] = x0 + y;
                }
            }
        }
    }
}


string hex_str(uint x)
{
    stringstream ss;
    ss << std::hex << "0x" << x;
    return ss.str();
}


// Called by 'python -m pirate_frb scratch'. Intended for quick throwaway tests.
void scratch()
{
    cout << "pirate::scratch() called -- this is a place for quick throwaway tests" << endl;

    // Constructing an int4 ksgpu::Array is no problem...
    Dtype dt_int4 = Dtype::from_str("int4");
    Array<void> arr(dt_int4, {3,3,7}, af_rhost | af_zero);

    // ... but currently, no helpful accessors are implemented, so in order to get/set
    // array elements, we need to reinterpret_cast a bare pointer. For example, to set
    // arr[0,0,5]=3, one would need to do something like this:

    unsigned char *p8 = (unsigned char *) arr.data;
    p8[2] = 0x30;  // arr[0,0,5] = 3

    // This also works.
    uint *p32 = (uint *) arr.data;
    p32[0] = 0x300000;  // arr[0,0,5] = 3
}


}  // namespace pirate
