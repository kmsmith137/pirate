#include "../include/pirate/avx2_utils.hpp"

#include <immintrin.h>
#include <random>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <stdexcept>

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


}  // namespace pirate
