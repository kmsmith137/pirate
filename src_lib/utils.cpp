#include "../include/pirate/utils.hpp"
#include "../include/pirate/inlines.hpp"    // pow2(), xdiv()
#include "../include/pirate/constants.hpp"  // constants::max_tree_rank

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <iomanip>

#include <sys/mman.h>
#include <cuda_runtime.h>

#include <ksgpu/cuda_utils.hpp>   // CUDA_CALL
#include <ksgpu/xassert.hpp>

using namespace std;
using namespace ksgpu;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// -------------------------------------------------------------------------------------------------
//
// safe_memcpy_*: split host<->device copies at absolute
// cuda_host_register_chunk_size-aligned host addresses, to work around
// the cudaMemcpyAsync()-spanning-two-registrations failure. See comments
// in utils.hpp and constants.hpp.


// 'host_ptr' is the host side of the copy (src for h2g, dst for g2h).
// 'dev_ptr' is the device side. 'do_one' is invoked once per piece with
// (host_p, dev_p, this_nbytes), where the piece [host_p, host_p+this_nbytes)
// is guaranteed not to straddle a chunk boundary.
template<typename Fn>
static inline void _split_at_chunk_boundaries(void *host_ptr, void *dev_ptr,
                                              long nbytes, Fn &&do_one)
{
    constexpr long chunk = constants::cuda_host_register_chunk_size;
    constexpr long mask  = chunk - 1;

    char *h = static_cast<char *>(host_ptr);
    char *d = static_cast<char *>(dev_ptr);
    long  n = nbytes;

    while (n > 0) {
        // Bytes from h to the next chunk boundary at or after h+1.
        // (If h is already on a boundary, this is exactly 'chunk'.)
        uintptr_t up = reinterpret_cast<uintptr_t>(h);
        long to_bdy  = static_cast<long>(chunk - (up & mask));
        long this_sz = std::min(n, to_bdy);
        do_one(h, d, this_sz);
        h += this_sz; d += this_sz; n -= this_sz;
    }
}


void safe_memcpy_h2g_sync(void *dst, const void *src, long nbytes)
{
    xassert(nbytes >= 0);
    _split_at_chunk_boundaries(const_cast<void *>(src), dst, nbytes,
        [](char *h, char *d, long n) {
            CUDA_CALL(cudaMemcpy(d, h, n, cudaMemcpyHostToDevice));
        });
}


void safe_memcpy_g2h_sync(void *dst, const void *src, long nbytes)
{
    xassert(nbytes >= 0);
    _split_at_chunk_boundaries(dst, const_cast<void *>(src), nbytes,
        [](char *h, char *d, long n) {
            CUDA_CALL(cudaMemcpy(h, d, n, cudaMemcpyDeviceToHost));
        });
}


void safe_memcpy_h2g_async(void *dst, const void *src, long nbytes,
                            cudaStream_t stream)
{
    xassert(nbytes >= 0);
    _split_at_chunk_boundaries(const_cast<void *>(src), dst, nbytes,
        [stream](char *h, char *d, long n) {
            CUDA_CALL(cudaMemcpyAsync(d, h, n, cudaMemcpyHostToDevice, stream));
        });
}


void safe_memcpy_g2h_async(void *dst, const void *src, long nbytes,
                            cudaStream_t stream)
{
    xassert(nbytes >= 0);
    _split_at_chunk_boundaries(dst, const_cast<void *>(src), nbytes,
        [stream](char *h, char *d, long n) {
            CUDA_CALL(cudaMemcpyAsync(h, d, n, cudaMemcpyDeviceToHost, stream));
        });
}


// -------------------------------------------------------------------------------------------------
//
// revisit_512gb_inner(): diagnostic for the 'pirate_frb revisit_512gb'
// subcommand. mmap nbytes, prefault if 4 KiB pages, attempt a single
// cudaHostRegister(), report. Returns true if the register call
// succeeded.


bool revisit_512gb_inner(long nbytes, bool use_hugepages)
{
    using clk = std::chrono::steady_clock;
    auto sec = [](clk::time_point t) {
        return std::chrono::duration<double>(clk::now() - t).count();
    };

    // CUDA versions (helpful when comparing future runs to today's).
    int rt = 0, drv = 0;
    cudaRuntimeGetVersion(&rt);
    cudaDriverGetVersion(&drv);
    auto fmt = [](int v) {
        return std::to_string(v / 1000) + "." + std::to_string((v % 1000) / 10);
    };
    cout << "  CUDA runtime version: " << fmt(rt) << "\n";
    cout << "  CUDA driver version:  " << fmt(drv) << "\n";

    // mmap.
    int mflags = MAP_PRIVATE | MAP_ANONYMOUS;
    if (use_hugepages)
        mflags |= MAP_HUGETLB;
    auto t = clk::now();
    void *base = mmap(nullptr, nbytes, PROT_READ | PROT_WRITE, mflags, -1, 0);
    if (base == MAP_FAILED) {
        int e = errno;
        cout << "  mmap " << (nbytes >> 30) << " GiB FAILED: "
             << std::strerror(e) << "\n" << std::flush;
        return false;
    }
    cout << "  mmap " << (nbytes >> 30) << " GiB ("
         << (use_hugepages ? "MAP_HUGETLB" : "4 KiB pages") << "): OK in "
         << std::fixed << std::setprecision(2) << sec(t) << "s\n" << std::flush;

    // Prefault. Hugepages are pre-committed by mmap, but 4 KiB pages
    // would otherwise be faulted lazily inside cudaHostRegister, making
    // its timing harder to interpret.
    if (!use_hugepages) {
        cout << "  prefaulting 4 KiB pages..." << std::flush;
        t = clk::now();
        constexpr long page = 4096;
        char *cp = static_cast<char *>(base);
        for (long off = 0; off < nbytes; off += page)
            cp[off] = 0;
        cout << " done in " << std::fixed << std::setprecision(2)
             << sec(t) << "s\n" << std::flush;
    }

    // The actual test: single cudaHostRegister() over the whole region.
    cout << "  cudaHostRegister(" << (nbytes >> 30) << " GiB)..." << std::flush;
    t = clk::now();
    cudaError_t err = cudaHostRegister(base, nbytes, cudaHostRegisterDefault);
    double reg_secs = sec(t);

    bool ok = (err == cudaSuccess);
    if (ok) {
        cout << " OK in " << std::fixed << std::setprecision(2)
             << reg_secs << "s\n" << std::flush;
        cudaHostUnregister(base);
    } else {
        cout << " FAILED after " << std::fixed << std::setprecision(2)
             << reg_secs << "s\n"
             << "    err=" << int(err) << " ("
             << cudaGetErrorString(err) << ")\n" << std::flush;
        cudaGetLastError();   // clear sticky error
    }

    if (munmap(base, nbytes) != 0)
        cout << "  warning: munmap failed: " << std::strerror(errno) << "\n";

    return ok;
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


void reference_downsample_freq(const Array<float> &in, Array<float> &out)
{
    xassert(out.ndim == 2);
    xassert(out.strides[1] == 1);

    xassert(in.shape_equals({ 2*out.shape[0], out.shape[1] }));
    xassert(in.strides[1] == 1);

    float w = 0.7071067811865476f;  // 1/sqrt(2)
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

    
void reference_downsample_time(const Array<float> &in, Array<float> &out)
{
    xassert(out.ndim == 2);
    xassert(out.strides[1] == 1);

    xassert(in.shape_equals({ out.shape[0], 2*out.shape[1] }));
    xassert(in.strides[1] == 1);

    float w = 0.7071067811865476f;  // 1/sqrt(2)
    int nchan = out.shape[0];
    int nt_out = out.shape[1];

    for (int c = 0; c < nchan; c++) {
        const float *src_row = in.data + c * in.strides[0];
        float *dst_row = out.data + c * out.strides[0];

        for (int t = 0; t < nt_out; t++)
            dst_row[t] = w * (src_row[2*t] + src_row[2*t+1]);
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
    static constexpr float rsqrt2 = 0.7071067811865476f;

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

                    row0[k] = rsqrt2 * (x1 + y);
                    row1[k] = rsqrt2 * (x0 + y);
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
