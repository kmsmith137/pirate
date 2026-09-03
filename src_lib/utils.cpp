#include "../include/pirate/utils.hpp"
#include "../include/pirate/inlines.hpp"    // pow2(), xdiv(), integer_log2()
#include "../include/pirate/constants.hpp"  // constants::max_tree_rank

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <limits>
#include <string_view>
#include <thread>
#include <vector>
#include <iomanip>

#include <sys/mman.h>
#include <unistd.h>         // write()
#include <cuda_runtime.h>

#include <ksgpu/cuda_utils.hpp>   // CUDA_CALL
#include <ksgpu/xassert.hpp>
#include <ksgpu/rand_utils.hpp>  // rand_int()

using namespace std;
using namespace ksgpu;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// -------------------------------------------------------------------------------------------------
//
// Serialized output: atomic_print() and class AtomicPrint. See utils.hpp for
// the rationale and for usage of the one-liner and block forms.


void atomic_print(std::string_view s, int fd) noexcept
{
    // An AtomicPrint that was never streamed to prints nothing.
    if (s.empty())
        return;

    // Process-global: shared by every C++ caller and, through the pybind11
    // binding, by every python caller in this process.
    static std::mutex lock;

    // Copy only in the case where we must append the newline.
    string tmp;
    if (s.back() != '\n') {
        tmp = string(s) + '\n';
        s = tmp;
    }

    // Everything above is formatting; the critical section below is
    // write-only, so a slow write never serializes message construction.
    lock_guard<std::mutex> l(lock);

    const char *p = s.data();
    long n = (long)s.size();

    while (n > 0) {
        ssize_t w = ::write(fd, p, n);

        if (w < 0) {
            // Retry a signal-interrupted write; otherwise give up silently
            // (see the noexcept/best-effort rationale in utils.hpp).
            if (errno == EINTR)
                continue;
            return;
        }

        p += w;
        n -= (long)w;
    }

    // Note on the loop: we hold the mutex across a short write, so
    // intra-process atomicity is unconditional. Cross-process atomicity
    // relies on the message going out in a single write(2), which the kernel
    // guarantees for pipes up to PIPE_BUF (4 KiB on linux) and which holds in
    // practice for regular files and ttys. A multi-KiB block sharing a pipe
    // with another process could therefore split; our lines are far smaller.
}


AtomicPrint::AtomicPrint(int fd_) :
    fd(fd_)
{
    xassert_ge(fd_, 0);
}


AtomicPrint::~AtomicPrint()
{
    // Destructors are implicitly noexcept, and atomic_print() is noexcept, but
    // ss.str() allocates and could in principle throw -- which would be
    // std::terminate during unwinding. Belt-and-braces.
    try {
        atomic_print(ss.str(), fd);
    } catch (...) {
    }
}


void test_atomic_print(int fd, long nthreads, long nlines, long line_pad)
{
    xassert_ge(fd, 0);
    xassert_gt(nthreads, 0);
    xassert_gt(nlines, 0);
    xassert_gt(line_pad, 0);

    vector<std::thread> threads;
    threads.reserve(nthreads);

    for (long t = 0; t < nthreads; t++) {
        threads.emplace_back([fd,nlines,line_pad,t]() {
            for (long i = 0; i < nlines; i++) {
                // Rotate through all three call styles, so the test exercises
                // the one-liner form, the multi-line block form, and a direct
                // atomic_print() -- including a line long enough to make a
                // splice obvious if the funnel ever regresses.
                switch (i % 3) {
                case 0:
                    AtomicPrint(fd) << "cpp t=" << t << " i=" << i << " oneliner "
                                  << string(line_pad, 'x');
                    break;

                case 1: {
                    AtomicPrint a(fd);
                    a << "cpp t=" << t << " i=" << i << " block1 " << string(line_pad, 'y') << "\n";
                    a << "cpp t=" << t << " i=" << i << " block2 " << string(line_pad, 'y') << "\n";
                    break;
                }

                default: {
                    stringstream ss;
                    ss << "cpp t=" << t << " i=" << i << " direct " << string(line_pad, 'z');
                    atomic_print(ss.str(), fd);
                    break;
                }
                }
            }
        });
    }

    for (auto &t: threads)
        t.join();
}


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
// revisit_512gb_inner(): diagnostic for the 'pirate_frb dev revisit_512gb'
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
        constexpr long page = constants::host_page_size;
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


// -------------------------------------------------------------------------------------------------
//
// test_utils(): unit test for bit_reverse_slow() above, and for the integer/bit helpers in
// inlines.hpp. Dispatched from 'python -m pirate_frb test --util'.
//
// The helpers are piecewise-constant between consecutive powers of two, so the test sweeps a
// dense range of small values plus a window around every binade boundary, rather than sampling
// uniformly. That is the point: an off-by-one binade AT a boundary is the failure mode these
// helpers are prone to, and uniform sampling essentially never lands on one. Some random values
// are swept too, in case that reasoning is incomplete.
//
// Each helper is checked against a reference below. The references are written as obvious loops,
// so that they share no logic with the shift-and-mask implementations they check.


static int ref_popcount(long x)
{
    int n = 0;
    for (int i = 0; i < 64; i++)
        n += (x >> i) & 1;
    return n;
}


static int ref_bit_length(long x)
{
    xassert(x >= 0);

    int n = 0;
    while (x) {
        n++;
        x >>= 1;
    }

    return n;
}


// Largest power of two <= x, or 0 if x < 1.
static long ref_bit_floor(long x)
{
    if (x < 1)
        return 0;

    long p = 1;
    while (p <= x/2)   // "p <= x/2", not "2*p <= x", so that the loop cannot overflow
        p *= 2;

    return p;
}


// Smallest power of two >= x, or 1 if x <= 1.
static long ref_bit_ceil(long x)
{
    xassert_le(x, 1L << 62);   // otherwise the answer is not representable

    long p = 1;
    while (p < x)
        p *= 2;

    return p;
}


// Places bit b of 'i' at position (nbits-1-b), rather than shifting bits out of one
// word and into another as bit_reverse_slow() does.
static int ref_bit_reverse(int i, int nbits)
{
    int j = 0;
    for (int b = 0; b < nbits; b++)
        if ((i >> b) & 1)
            j |= 1 << (nbits-1-b);

    return j;
}


// Checks that f() throws, i.e. that it lands on one of the xasserts in the helper under test.
template<typename F>
static void check_throws(F f, const string &what)
{
    try {
        f();
    }
    catch (const std::exception &) {
        return;
    }

    throw runtime_error("test_utils: expected an exception from " + what + ", but none was thrown");
}


void test_utils()
{
    // Values swept below. All are nonnegative: the helpers take a 'long', but are only
    // meaningful on nonnegative arguments (e.g. bit_floor(-1) would shift into the sign bit).

    vector<long> vals;

    for (long n = 0; n <= 4096; n++)
        vals.push_back(n);

    for (int k = 12; k <= 62; k++)
        for (long d = -2; d <= 2; d++)
            vals.push_back((1L << k) + d);

    for (int i = 0; i < 1000; i++)
        vals.push_back(rand_int(0, 1L << 62));

    // is_power_of_two(), popcount(), bit_length(), bit_floor().

    for (long n: vals) {
        long bf = ref_bit_floor(n);
        xassert_eq(is_power_of_two(n), ((n >= 1) && (bf == n)));
        xassert_eq(popcount(n), ref_popcount(n));
        xassert_eq(bit_length(n), ref_bit_length(n));
        xassert_eq(bit_floor(n), bf);
    }

    // round_up_to_power_of_two(), round_down_to_power_of_two().
    // Note that round_up_to_power_of_two() is only defined for n <= 2^62.

    for (long n: vals) {
        if (n <= (1L << 62))
            xassert_eq(round_up_to_power_of_two(n), ref_bit_ceil(n));
        if (n >= 1)
            xassert_eq(round_down_to_power_of_two(n), ref_bit_floor(n));
    }

    for (long n: { 0L, -1L, numeric_limits<long>::min() })
        check_throws([n]() { round_down_to_power_of_two(n); },
                     "round_down_to_power_of_two(" + to_string(n) + ")");

    // integer_log2(), on its domain (exact powers of two) and off it. The off-domain list
    // includes both neighbours of every binade boundary, since "one away from a power of
    // two" is the input most likely to be mistaken for a power of two.

    for (int k = 0; k <= 62; k++)
        xassert_eq(integer_log2(1L << k), k);

    vector<long> non_powers = { 0, -1, -2, 3, 5, 6, 7, 9, 100,
                                numeric_limits<long>::min(), numeric_limits<long>::max() };

    for (int k = 2; k <= 62; k++) {
        non_powers.push_back((1L << k) - 1);
        non_powers.push_back((1L << k) + 1);
        non_powers.push_back(-(1L << k));
    }

    for (long n: non_powers)
        check_throws([n]() { integer_log2(n); }, "integer_log2(" + to_string(n) + ")");

    // pow2().

    for (int k = 0; k <= 32; k++) {
        long p = 1;
        for (int j = 0; j < k; j++)
            p *= 2;
        xassert_eq(pow2(k), p);
    }

    check_throws([]() { pow2(-1); }, "pow2(-1)");
    check_throws([]() { pow2(33); }, "pow2(33)");

    // align_up(). The reference uses a remainder, rather than align_up()'s bitmask.

    for (int k = 0; k <= 20; k++) {
        long a = 1L << k;
        for (long n = 0; n <= 1025; n++) {
            long r = n % a;
            xassert_eq(align_up(n,a), (r ? (n+a-r) : n));
        }
    }

    check_throws([]() { align_up(-1, 8); }, "align_up(-1,8)");
    check_throws([]() { align_up(8, 0); }, "align_up(8,0)");
    check_throws([]() { align_up(8, 6); }, "align_up(8,6)");   // nalign not a power of two

    // xdiv(), xmod().

    for (long m = 0; m <= 100; m++) {
        for (long n = 1; n <= 20; n++) {
            xassert_eq(xmod(m,n), m % n);
            if ((m % n) == 0)
                xassert_eq(xdiv(m,n), m/n);
            else
                check_throws([m,n]() { xdiv(m,n); },
                             "xdiv(" + to_string(m) + "," + to_string(n) + ")");
        }
    }

    check_throws([]() { xdiv(-4, 2); }, "xdiv(-4,2)");
    check_throws([]() { xdiv(4, 0); }, "xdiv(4,0)");
    check_throws([]() { xmod(-4, 2); }, "xmod(-4,2)");
    check_throws([]() { xmod(4, 0); }, "xmod(4,0)");

    // bit_reverse_slow(): exhaustive for nbits <= 14, checked both against ref_bit_reverse()
    // and against the property that reversing twice is the identity.

    long nrev = 0;

    for (int nbits = 0; nbits <= 14; nbits++) {
        for (int i = 0; i < (1 << nbits); i++) {
            int j = bit_reverse_slow(i, nbits);
            xassert_eq(j, ref_bit_reverse(i, nbits));
            xassert_eq(bit_reverse_slow(j, nbits), i);
            nrev++;
        }
    }

    check_throws([]() { bit_reverse_slow(0, -1); }, "bit_reverse_slow(0,-1)");
    check_throws([]() { bit_reverse_slow(0, 31); }, "bit_reverse_slow(0,31)");
    check_throws([]() { bit_reverse_slow(-1, 4); }, "bit_reverse_slow(-1,4)");
    check_throws([]() { bit_reverse_slow(16, 4); }, "bit_reverse_slow(16,4)");

    AtomicPrint() << "test_utils: " << vals.size() << " values swept through the inlines.hpp"
                  << " integer helpers, " << nrev << " bit_reverse_slow() calls exhausted"
                  << " (nbits <= 14)";
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
