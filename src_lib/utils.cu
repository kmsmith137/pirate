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

int rb_lag(int freq_coarse, int dm_brev, int stage1_rank, int stage2_rank, bool uflag)
{
    xassert(stage1_rank >= 0);
    xassert(stage2_rank >= 0);
    xassert_le(stage1_rank+stage2_rank, constants::max_tree_rank);

    int ndm = (1 << stage1_rank);
    int nfreq = (1 << stage2_rank);
    
    xassert((freq_coarse >= 0) && (freq_coarse < nfreq));
    xassert((dm_brev >= 0) && (dm_brev < ndm));

    int dm = bit_reverse_slow(dm_brev, stage1_rank);
    
    if (uflag)
        dm += ndm;

    int lag = (nfreq-1-freq_coarse) * dm;
    xassert(lag >= 0);

    return lag;
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

long dedispersion_delay(int rank, long freq, long dm_brev)
{
    long delay = 0;
    long delay0 = 0;

    for (int r = 0; r < rank; r++) {
        long d = (dm_brev & 1) ? (delay0+1) : delay0;
        delay += ((freq & 1) ? 0 : d);
        delay0 += d;
        dm_brev >>= 1;
        freq >>= 1;
    }

    return delay;
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


}  // namespace pirate
