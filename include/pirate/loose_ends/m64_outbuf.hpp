#ifndef _PIRATE_LOOSE_ENDS_AVX256_M64_OUTBUF_HPP
#define _PIRATE_LOOSE_ENDS_AVX256_M64_OUTBUF_HPP

#include <immintrin.h>
#include <ksgpu/xassert.hpp>

namespace pirate {
#if 0
}  // editor auto-indent
#endif


struct avx256_m64_outbuf
{
    // Streaming writes are faster, but don't forget _mm_mfence()!
    static constexpr bool use_streaming_writes = true;

    using I64 = long long;
    
    I64 *out;
    I64 out_buf;    
    __m256i c64;

    
    avx256_m64_outbuf(uint8_t *out_)
    {
    	out = (I64 *) out_;
	out_buf = 0;
	c64 = _mm256_set1_epi64x(64);
    }


    inline void flush64(I64 x)
    {
	if constexpr (use_streaming_writes)
	    _mm_stream_si64(out, out_buf);
	else
	    *out = out_buf;
	
	out_buf = x;
	out++;
    }
    
    // 'P_reg' must contain [P3,P2,P1,P0]
    
    template<int P0, int P1, int P2, int P3, int P4>
    inline void advance(__m256i x, __m256i P_reg)
    {
	static_assert(P0 >= 0);
	static_assert((P1 >= P0) && (P1 <= P0+64));
	static_assert((P2 >= P1) && (P2 <= P1+64));
	static_assert((P3 >= P2) && (P3 <= P2+64));
	static_assert((P4 >= P3) && (P4 <= P3+64));

#if 0
	xassert(_mm256_extract_epi64(P_reg,0) == P0);
	xassert(_mm256_extract_epi64(P_reg,1) == P1);
	xassert(_mm256_extract_epi64(P_reg,2) == P2);
	xassert(_mm256_extract_epi64(P_reg,3) == P3);
#endif

	// P_reg &= 0x3f
	P_reg = _mm256_slli_epi64(P_reg, 58);
	P_reg = _mm256_srli_epi64(P_reg, 58);
	
	// Think of (x << P) as a 128-bit integer 'z'. After this block:
	//
	//   a = (low 64 bits of z)
	//   b = (high 64 bits of z)

	__m256i a = _mm256_sllv_epi64(x, P_reg);
	__m256i b = _mm256_srlv_epi64(x, _mm256_sub_epi64(c64, P_reg));
	
	__m128i a01 = _mm256_extracti128_si256(a, 0);
	__m128i a23 = _mm256_extracti128_si256(a, 1);
	__m128i b01 = _mm256_extracti128_si256(b, 0);
	__m128i b23 = _mm256_extracti128_si256(b, 1);
	
	out_buf |= _mm_extract_epi64(a01, 0);

	if constexpr ((P0/64) != (P1/64))
	    flush64(_mm_extract_epi64(b01, 0));

	out_buf |= _mm_extract_epi64(a01, 1);
	
	if constexpr ((P1/64) != (P2/64))
	    flush64(_mm_extract_epi64(b01, 1));

	out_buf |= _mm_extract_epi64(a23, 0);
	
	if constexpr ((P2/64) != (P3/64))
	    flush64(_mm_extract_epi64(b23, 0));

	out_buf |= _mm_extract_epi64(a23, 1);
	
	if constexpr ((P3/64) != (P4/64))
	    flush64(_mm_extract_epi64(b23, 1));
    }
};


}  // namespace pirate

#endif  // _PIRATE_LOOSE_ENDS_AVX256_M64_OUTBUF_HPP
