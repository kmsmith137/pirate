#ifndef _PIRATE_AVX2_UTILS_HPP
#define _PIRATE_AVX2_UTILS_HPP

// Note: this header is deliberately free of <immintrin.h> (and any AVX intrinsics), so it can be
// safely #included from .cu translation units. The AVX2 implementation lives in avx2_utils.cpp,
// which is compiled by the host compiler (nvcc's CUDA frontend chokes on <immintrin.h>).

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// Simulates Gaussian random noise with rms=2.5 (this is the rms of the pre-quantized Gaussian;
// the int4 quantization and the [-7,7] clamp perturb the actual output rms slightly, to ~2.51).
// Quantizes to int4, clamped to [-7,7], i.e. the value -8 is not allowed.
// Quantization is +/- symmetric, i.e. 4-bit output value 0x1 corresponds to range [0.5,1.5].
// Packs the result into (nelts_4bit/2) bytes, or equivalently (nelts_4bit/8) uints.
// Throws an exception if nelts_4bit is not a multiple of 64 (convenient for simd alignment).
//
// Thread-safe: the rng state is per-thread (thread_local), initialized once per thread from
// std::random_device.

void avx2_simulate_4bit_noise(unsigned int *dst, long nelts_4bit);


}  // namespace pirate

#endif  // _PIRATE_AVX2_UTILS_HPP
