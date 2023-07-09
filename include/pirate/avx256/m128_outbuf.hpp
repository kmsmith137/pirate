#ifndef _PIRATE_AVX256_M128_OUTBUF_HPP
#define _PIRATE_AVX256_M128_OUTBUF_HPP

#include <immintrin.h>

namespace pirate {
#if 0
}  // editor auto-indent
#endif


struct avx256_m128_outbuf
{
    // Streaming writes are faster, but don't forget _mm_mfence()!
    static constexpr bool use_streaming_writes = true;
    
    __m128i *out;
    __m128i out_buf;    

    
    avx256_m128_outbuf(uint8_t *out_)
    {
    	out = (__m128i *) out_;
	out_buf = _mm_setzero_si128();
    }

    
    template<int P, int N>
    inline void advance(__m128i x)
    {
	constexpr int Q = P % 16;
	static_assert((N >= 0) && (N < 16));

	if constexpr (Q == 0)
	    out_buf = x;
	else
	    out_buf = _mm_or_si128(out_buf, _mm_slli_si128(x, Q));

	if constexpr (Q + N >= 16) {
	    if constexpr (use_streaming_writes)
		_mm_stream_si128(out, out_buf);
	    else
		_mm_store_si128(out, out_buf);
	    out++;
	}

	if constexpr (Q + N > 16)
	    out_buf = _mm_srli_si128(x, 16-Q);
    }
};


}  // namespace pirate

#endif  // _PIRATE_AVX256_M128_OUTBUF_HPP
