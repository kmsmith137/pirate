#include "../include/pirate/avx2_utils.hpp"

#include <immintrin.h>
#include <random>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <thread>
#include <chrono>
#include <algorithm>
#include <cstdio>

using namespace std;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// avx2_simulate_4bit_noise(): generate quantized-Gaussian int4 noise, packed 2 nibbles/byte.
//
// The chain is: uint32 RNG -> quantized Gaussian in [-7,7] -> packed int4. The quantization is
// done by inverse-CDF: the output has only 15 levels, so instead of computing a Gaussian (with
// log/sqrt), we map the uniform uint32 straight to a level with 14 precomputed thresholds:
//     level = -7 + #{ j : u >= U[j] },   U[j] = 2^32 * Phi((-6.5+j)/sigma),  sigma = 2.5.
// (These U[j] are the bin edges of the quantized+clamped Gaussian.)
//
// Packing matches GpuDequantizationKernel: signed two's-complement nibbles, low nibble = even
// index; each output uint32 packs 8 samples with sample j in bits [4j, 4j+4). The value -8 is
// never produced (it is reserved as a "masked" sentinel there).


static const double SIGMA = 2.5;

// 14 signed-flipped thresholds: (U[j]-1) ^ 0x80000000. A signed AVX2 compare of (u ^ 0x80000000)
// against this then computes the unsigned predicate "u >= U[j]" (AVX2 has no unsigned compare).
struct Thresholds {
    int32_t thr_s[14];
    Thresholds() {
        for (int j = 0; j < 14; j++) {
            double b = -6.5 + j;                                        // bin edge between levels (-7+j),(-6+j)
            double phi = 0.5 * std::erfc(-(b / SIGMA) / std::sqrt(2.0));  // Phi(b/sigma)
            long v = std::lround(phi * 4294967296.0);                   // * 2^32
            if (v < 1) v = 1;
            if (v > 4294967295L) v = 4294967295L;
            thr_s[j] = (int32_t) (((uint32_t) v - 1u) ^ 0x80000000u);
        }
    }
};

// Meyers singleton: thread-safe one-time init (std::erfc is a runtime call).
static const Thresholds &thresholds()
{
    static const Thresholds t;
    return t;
}


// Recover the 14 uint32 CDF thresholds U[j] from the (sign-flipped) Thresholds.
static void recover_uint32_thresholds(uint32_t U[14])
{
    const int32_t *thr_s = thresholds().thr_s;
    for (int j = 0; j < 14; j++)
        U[j] = ((uint32_t) thr_s[j] ^ 0x80000000u) + 1u;      // invert thr_s[j] = (U[j]-1) ^ 0x80000000
}

// Probability of output level L (in [-7,7]) implied by the integer thresholds U[].
static double discrete_level_prob(int L, const uint32_t U[14])
{
    const double inv2p32 = 1.0 / 4294967296.0;
    if (L == -7) return U[0] * inv2p32;                        // g < -6.5  (lower clamp)
    if (L ==  7) return (4294967296.0 - U[13]) * inv2p32;      // g >= 6.5  (upper clamp)
    return (double) (U[L+7] - U[L+6]) * inv2p32;
}


double avx2_4bit_noise_variance()
{
    uint32_t U[14];
    recover_uint32_thresholds(U);
    double mean = 0.0, m2 = 0.0;
    for (int L = -7; L <= 7; L++) {
        double p = discrete_level_prob(L, U);
        mean += p * (double) L;
        m2   += p * (double) L * (double) L;
    }
    return m2 - mean * mean;                                    // symmetric, so mean ~ 0
}


// ---- vectorized xoshiro128++ (8 independent lanes / streams) ----

template<int K> static inline __m256i rotl8(__m256i x)
{
    return _mm256_or_si256(_mm256_slli_epi32(x, K), _mm256_srli_epi32(x, 32 - K));
}

static inline __m256i xoshiro8_next(__m256i &s0, __m256i &s1, __m256i &s2, __m256i &s3)
{
    __m256i result = _mm256_add_epi32(rotl8<7>(_mm256_add_epi32(s0, s3)), s0);
    __m256i t = _mm256_slli_epi32(s1, 9);
    s2 = _mm256_xor_si256(s2, s0);
    s3 = _mm256_xor_si256(s3, s1);
    s1 = _mm256_xor_si256(s1, s2);
    s0 = _mm256_xor_si256(s0, s3);
    s2 = _mm256_xor_si256(s2, t);
    s3 = rotl8<11>(s3);
    return result;
}


// Per-thread RNG state. Layout: t_state[w*8 + l] = word w (0..3) of lane l (0..7), i.e. each lane
// is an independent 128-bit xoshiro128++. Seeded once per thread from std::random_device.
static thread_local uint32_t t_state[32];
static thread_local bool t_seeded = false;

static void seed_state()
{
    std::random_device rd;
    for (int i = 0; i < 32; i++)
        t_state[i] = (uint32_t) rd();
    // xoshiro requires nonzero state per lane (all-zero would be a fixed point). Probability
    // 2^-128 per lane, but guard it anyway.
    for (int lane = 0; lane < 8; lane++)
        if ((t_state[lane] | t_state[8+lane] | t_state[16+lane] | t_state[24+lane]) == 0u)
            t_state[lane] = 1u;
    t_seeded = true;
}


void avx2_simulate_4bit_noise(unsigned int *dst, long nelts_4bit)
{
    if (nelts_4bit < 0 || (nelts_4bit % 64) != 0) {
        std::stringstream ss;
        ss << "avx2_simulate_4bit_noise: nelts_4bit (=" << nelts_4bit << ") must be a nonnegative multiple of 64";
        throw std::runtime_error(ss.str());
    }
    if (nelts_4bit == 0)
        return;

    if (!t_seeded)
        seed_state();

    // Load per-thread state (unaligned; t_state is a plain uint32 array).
    __m256i s0 = _mm256_loadu_si256((const __m256i*)(t_state +  0));
    __m256i s1 = _mm256_loadu_si256((const __m256i*)(t_state +  8));
    __m256i s2 = _mm256_loadu_si256((const __m256i*)(t_state + 16));
    __m256i s3 = _mm256_loadu_si256((const __m256i*)(t_state + 24));

    const int32_t *thr_s = thresholds().thr_s;
    const __m256i msb     = _mm256_set1_epi32((int) 0x80000000u);
    const __m256i seven   = _mm256_set1_epi32(7);
    const __m256i nibmask = _mm256_set1_epi32(0xF);
    const __m256i shifts  = _mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28);   // lane l -> bits [4l,4l+4)
    __m256i thr[14];
    for (int j = 0; j < 14; j++) thr[j] = _mm256_set1_epi32(thr_s[j]);

    long nwords = nelts_4bit / 8;                          // each uint32 packs 8 int4 samples
    for (long w = 0; w < nwords; w++) {
        __m256i us = _mm256_xor_si256(xoshiro8_next(s0, s1, s2, s3), msb);
        __m256i acc = _mm256_setzero_si256();
        for (int j = 0; j < 14; j++)
            acc = _mm256_sub_epi32(acc, _mm256_cmpgt_epi32(us, thr[j]));   // acc += (u >= U[j])
        __m256i level = _mm256_sub_epi32(acc, seven);                     // signed level in [-7,7]

        // Pack the 8 lanes into one uint32: nibble(l) = (level_l & 0xF) placed at bits [4l,4l+4).
        __m256i nib = _mm256_sllv_epi32(_mm256_and_si256(level, nibmask), shifts);
        __m128i lo = _mm256_castsi256_si128(nib);
        __m128i hi = _mm256_extracti128_si256(nib, 1);
        __m128i o  = _mm_or_si128(lo, hi);                                        // OR lanes 0-3 with 4-7
        o = _mm_or_si128(o, _mm_shuffle_epi32(o, _MM_SHUFFLE(1, 0, 3, 2)));       // OR the two 64-bit halves
        o = _mm_or_si128(o, _mm_shuffle_epi32(o, _MM_SHUFFLE(2, 3, 0, 1)));       // OR the two 32-bit halves
        dst[w] = (unsigned int) _mm_cvtsi128_si32(o);
    }

    // Store state back so the stream continues across calls on this thread.
    _mm256_storeu_si256((__m256i*)(t_state +  0), s0);
    _mm256_storeu_si256((__m256i*)(t_state +  8), s1);
    _mm256_storeu_si256((__m256i*)(t_state + 16), s2);
    _mm256_storeu_si256((__m256i*)(t_state + 24), s3);
}


// =================================================================================================
//
// Unit test + timing, dispatched from 'python -m pirate_frb test --sim' / 'time --sim'.


// Standard normal CDF Phi(z), used to compute the expected quantized-Gaussian level probabilities.
static inline double std_normal_cdf(double z)
{
    return 0.5 * std::erfc(-z / std::sqrt(2.0));
}

// Expected probability of output level L (in [-7,7]) for the quantized+clamped Gaussian(sigma=SIGMA).
static double expected_level_prob(int L)
{
    if (L == -7) return std_normal_cdf(-6.5 / SIGMA);            // g < -6.5  (lower clamp)
    if (L ==  7) return 1.0 - std_normal_cdf(6.5 / SIGMA);       // g >= 6.5  (upper clamp)
    return std_normal_cdf((L + 0.5) / SIGMA) - std_normal_cdf((L - 0.5) / SIGMA);
}


void test_avx2_simulate_4bit_noise()
{
    // Self-consistency: the integer thresholds baked into the SIMD kernel should reproduce
    // expected_level_prob() when differenced and scaled by 2^-32, up to the <= 2^-32 rounding of
    // each threshold. (Cheap check that the kernel's quantization boundaries match the continuous
    // distribution the histogram below is compared against.)
    {
        uint32_t U[14];
        recover_uint32_thresholds(U);
        const double inv2p32 = 1.0 / 4294967296.0;
        for (int L = -7; L <= 7; L++) {
            double disc = discrete_level_prob(L, U);
            double cont = expected_level_prob(L);
            if (std::abs(disc - cont) > 4.0 * inv2p32) {
                std::stringstream ss;
                ss << "test_avx2_simulate_4bit_noise: threshold/prob mismatch at level " << L
                   << " -- from thresholds " << disc << " vs expected " << cont;
                throw std::runtime_error(ss.str());
            }
        }
    }

    const long n = 1L << 24;                     // ~16.8M samples
    long nwords = n / 8;
    std::vector<unsigned int> buf(nwords, 0u);

    avx2_simulate_4bit_noise(buf.data(), n);

    long hist[16] = {0};                         // index = level + 8
    for (long w = 0; w < nwords; w++) {
        unsigned int x = buf[w];
        for (int k = 0; k < 8; k++) {
            int nib = (x >> (4*k)) & 0xF;
            int level = (nib >= 8) ? (nib - 16) : nib;   // signed two's complement
            hist[level + 8]++;
        }
    }

    // The -8 value is reserved as a sentinel and must never be produced.
    if (hist[0] != 0) {
        std::stringstream ss;
        ss << "test_avx2_simulate_4bit_noise: FAILED -- produced " << hist[0] << " forbidden -8 values";
        throw std::runtime_error(ss.str());
    }

    // Per-level statistical check vs the quantized-Gaussian(rms=2.5) distribution. Tolerance is
    // 8 sigma (binomial), chosen so a correct generator ~never fails while any real bug -- wrong
    // sigma, wrong thresholds, mis-packing -- shifts levels by many sigma and is caught.
    double mean = 0, var = 0, max_sigma = 0;
    for (int level = -7; level <= 7; level++) {
        double p = expected_level_prob(level);
        double exp_c = n * p;
        double sd = std::sqrt(n * p * (1.0 - p));
        double obs_c = (double) hist[level + 8];
        double dev = std::abs(obs_c - exp_c);
        if (sd > 0 && dev / sd > max_sigma) max_sigma = dev / sd;
        if (dev > 8.0 * sd + 5.0) {
            std::stringstream ss;
            ss << "test_avx2_simulate_4bit_noise: FAILED at level " << level
               << " -- observed " << (long) obs_c << ", expected " << exp_c << " +/- " << sd;
            throw std::runtime_error(ss.str());
        }
        double frac = obs_c / n;
        mean += level * frac;
        var  += level * (double) level * frac;
    }
    var -= mean * mean;

    printf("    test_avx2_simulate_4bit_noise: pass  (n=%ld, mean=%+.4f, rms=%.4f [analytic %.4f], max deviation=%.1f sigma)\n",
           n, mean, std::sqrt(var), std::sqrt(avx2_4bit_noise_variance()), max_sigma);
}


void time_avx2_simulate_4bit_noise(long nthreads)
{
    if (nthreads < 1) nthreads = 1;
    const long N = 1L << 27;                      // samples per thread
    const long nwords = N / 8;

    // One reusable output buffer per thread (~64 MB each); pre-touched by zero-init.
    std::vector<std::vector<unsigned int>> bufs((size_t) nthreads);
    for (long t = 0; t < nthreads; t++)
        bufs[t].assign((size_t) nwords, 0u);

    // (a) 'nth' threads, one length-N call each (into bufs[t]).
    auto run_threads = [&](long nth) {
        std::vector<std::thread> threads;
        threads.reserve((size_t) nth);
        auto t0 = std::chrono::steady_clock::now();
        for (long t = 0; t < nth; t++)
            threads.emplace_back([&bufs, N, t]() { avx2_simulate_4bit_noise(bufs[t].data(), N); });
        for (auto &th : threads) th.join();
        auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double>(t1 - t0).count();
    };

    // (b) single thread, 'ncalls' back-to-back length-'call_len' calls filling bufs[0] (total = N).
    // Isolates the per-call overhead (state load/store, threshold broadcast) at fixed total work.
    auto run_chunked = [&](long ncalls, long call_len) {
        long words_per_call = call_len / 8;
        auto t0 = std::chrono::steady_clock::now();
        for (long c = 0; c < ncalls; c++)
            avx2_simulate_4bit_noise(bufs[0].data() + c * words_per_call, call_len);
        auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double>(t1 - t0).count();
    };

    auto median3 = [](auto &&fn) {
        fn();                                     // warmup (untimed)
        double d[3];
        for (int r = 0; r < 3; r++) d[r] = fn();
        std::sort(d, d + 3);
        return d[1];
    };

    double t1 = median3([&] { return run_chunked(1L << 7, 1L << 20); });   // 2^7 x 2^20 = 2^27
    double g1 = (double) N / t1 / 1e9;

    double t2 = median3([&] { return run_chunked(1L << 17, 1L << 10); });  // 2^17 x 2^10 = 2^27
    double g2 = (double) N / t2 / 1e9;

    double t3 = median3([&] { return run_chunked(1, N); });
    double g3 = (double) N / t3 / 1e9;

    double tN = median3([&] { return run_threads(nthreads); });
    double total_N = (double) (nthreads * N) / tN / 1e9;
    double perthread_N = total_N / nthreads;

    printf("avx2_simulate_4bit_noise() throughput  (nthreads = %ld, 2^27 samples per variant)\n", nthreads);
    printf("   1 thread,    2^7 x 2^20:    %6.2f Gsamp/s\n", g1);
    printf("   1 thread,    2^17 x 2^10:   %6.2f Gsamp/s\n", g2);
    printf("   1 thread,    1 x 2^27:      %6.2f Gsamp/s\n", g3);
    printf("   %2ld threads,  1 x 2^27 ea:   %7.2f Gsamp/s total,  %5.2f per-thread   (%.1fx vs 1 thread)\n",
           nthreads, total_N, perthread_N, total_N / g3);
}


}  // namespace pirate
