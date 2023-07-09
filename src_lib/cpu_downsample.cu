#include <cassert>
#include <sstream>
#include <stdexcept>

#include "../include/pirate/avx256/downsample.hpp"
#include "../include/pirate/internals/cpu_downsample.hpp"

using namespace std;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


static exception bad_bit_depth(const char *where, int src_bit_depth)
{
    stringstream ss;
    ss << where << ": src_bit_depth=" << src_bit_depth << " not supported";
    return runtime_error(ss.str());
}
				

ssize_t cpu_downsample_src_bytes_per_chunk(int src_bit_depth)
{
    if (src_bit_depth == 4)
	return avx256_4bit_downsampler::src_bytes_per_chunk;
    else if (src_bit_depth == 5)
	return avx256_5bit_downsampler::src_bytes_per_chunk;
    else if (src_bit_depth == 6)
	return avx256_6bit_downsampler::src_bytes_per_chunk;
    else if (src_bit_depth == 7)
	return avx256_7bit_downsampler::src_bytes_per_chunk;
    else
	throw bad_bit_depth("cpu_downsample_src_bytes_per_chunk()", src_bit_depth);
}


template<typename T>
static void _cpu_downsample(const uint8_t *src, uint8_t *dst, ssize_t src_nbytes, ssize_t dst_nbytes)
{
    constexpr int S = T::src_bytes_per_chunk;
    constexpr int D = T::dst_bytes_per_chunk;
    
    // FIXME pointer alignment asserts
    assert(src_nbytes >= 0);
    assert(dst_nbytes >= 0);
    assert((src_nbytes * D) == (dst_nbytes * S));
    assert((src_nbytes % S) == 0);

    T kernel(dst);
    
    for (ssize_t i = 0; i < src_nbytes; i += T::src_bytes_per_chunk)
	kernel.advance_chunk(src+i);

    // Kernel uses streaming writes, so don't forget the fence!!
    _mm_mfence();    
}


void cpu_downsample(int src_bit_depth, const uint8_t *src, uint8_t *dst, ssize_t src_nbytes, ssize_t dst_nbytes)
{
    if (src_bit_depth == 4)
	_cpu_downsample <avx256_4bit_downsampler> (src, dst, src_nbytes, dst_nbytes);
    else if (src_bit_depth == 5)
	_cpu_downsample <avx256_5bit_downsampler> (src, dst, src_nbytes, dst_nbytes);
    else if (src_bit_depth == 6)
	_cpu_downsample <avx256_6bit_downsampler> (src, dst, src_nbytes, dst_nbytes);
    else if (src_bit_depth == 7)
	_cpu_downsample <avx256_7bit_downsampler> (src, dst, src_nbytes, dst_nbytes);
    else
	throw bad_bit_depth("cpu_downsample()", src_bit_depth);
}


}  // namespace pirate
