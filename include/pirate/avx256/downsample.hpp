#ifndef _PIRATE_AVX256_DOWNSAMPLE_HPP
#define _PIRATE_AVX256_DOWNSAMPLE_HPP

#include <cstdint>
#include <immintrin.h>
#include "m64_outbuf.hpp"
#include "m128_outbuf.hpp"

namespace pirate {
#if 0
}  // editor auto-indent
#endif

// Note: downsampling kernels use streaming writes, so don't forget _mm_mfence()!


// -------------------------------------------------------------------------------------------------


template<int N>
inline __m256i avx256_ds_add(__m256i x, __m256i mask)
{
    __m256i y;
    
    y = _mm256_srli_epi64(x, N);
    x = _mm256_and_si256(mask, x);
    y = _mm256_and_si256(mask, y);
    x = _mm256_add_epi64(x, y);
    
    return x;
}


template<int N>
inline __m256i avx256_ds_shift(__m256i x, __m256i mask)
{
    __m256i y;
    
    y = _mm256_andnot_si256(mask, x);
    y = _mm256_srli_epi64(y, N);
    x = _mm256_and_si256(mask, x);
    x = _mm256_or_si256(x, y);

    return x;
}


// -------------------------------------------------------------------------------------------------


template<int S>
struct avx256_m64_splitter
{
    __m256i in_prev;
    __m256i S_reg;

    avx256_m64_splitter()
    {
	in_prev = _mm256_setzero_si256();
	S_reg = _mm256_set1_epi64x(S);
    }
    
    inline void split(__m256i &a, __m256i &b, __m256i in, __m256i R_reg)
    {
	// Have: in = [ x7 x6 x5 x4 ], in_prev = [ x3 x2 x1 x0 ]
	// Want: in_prev = [ x6 x5 x4 x3 ]

	in_prev = _mm256_blend_epi32(in_prev, in, 0x3f);     // [ x3 x6 x5 x4 ],  0x3f = (00111111)_2
	in_prev = _mm256_permute4x64_epi64(in_prev, 0x93);   // [ x6 x5 x4 x3 ],  0x93 = (2103)_4

	// Within each 64-bit lane, think of (in + in_prev) as a logical array 'x' of (64+R) bits.
	//
	// Have:
	//   in = x[64+R:R]
	//   in_prev = x[R:0] + junk[64-R]
	//
	// Want:
	//   a = x[64:0]
	//   b = zero[S-R] + x[(64+R):S]

	__m256i SR = _mm256_sub_epi64(S_reg, R_reg);
	__m256i t = _mm256_srlv_epi64(in_prev, SR);   // zero[S-R] + x[R:0] + junk[64-S]
	__m256i u = _mm256_srli_epi64(t, 64-S);       // zero[64-R] + x[R:0]
	__m256i v = _mm256_sllv_epi64(in, R_reg);     // x[64:R] + zero[R]
	
	a = _mm256_or_si256(u, v);            // x[64:0]
	b = _mm256_srlv_epi64(in, SR);        // zero[S-R] + x[64+R:S]

	in_prev = in;
    }
};


// -------------------------------------------------------------------------------------------------


// 4-bit -> 5-bit
struct avx256_4bit_downsampler
{
    static constexpr int src_bytes_per_chunk = 8*32;
    static constexpr int dst_bytes_per_chunk = 5*32;

    avx256_m128_outbuf obuf;
    
    __m256i mask0;
    __m256i mask1;
    __m256i mask2;
    __m256i mask3;
    __m256i mask4;

    
    avx256_4bit_downsampler(uint8_t *out_)
	: obuf(out_)
    {
	mask0 = _mm256_set1_epi8(0x0f);
	mask1 = _mm256_set1_epi16(0x00ff);
	mask2 = _mm256_set1_epi32(0x0000ffff);
	mask3 = _mm256_set1_epi64x(0x00000000ffffffffL);
	mask4 = _mm256_set_epi64x(0, 0xffffffffffffffffL, 0, 0xffffffffffffffffL);
    }

    
    template<int I>
    inline void advance1(__m256i x)
    {
	// Have: each 64-bit integer in 'x' looks like
	//   x16 + ... + x0    where xi is 4-bit
	//
	// Want: each 64-bit integer in 'x' looks like
	//   zero[24] + (x16+x15) + ... + (x1+x0)   where (xi+xj) is 5-bit
	
	__m256i y;
	
	x = avx256_ds_add<4> (x, mask0);
	x = avx256_ds_shift<3> (x, mask1);
	x = avx256_ds_shift<6> (x, mask2);
	x = avx256_ds_shift<12> (x, mask3);

	// Last shift done by hand (rather than calling avx256_ds_shift), since it crosses 64-bit boundaries.
	y = _mm256_andnot_si256(mask4, x);
	y = _mm256_srli_si256(y, 3);
	x = _mm256_and_si256(mask4, x);
	x = _mm256_or_si256(x, y);

	// Write output
	obuf.advance <20*I,10> (_mm256_extracti128_si256(x, 0));
	obuf.advance <20*I+10,10> (_mm256_extracti128_si256(x, 1));
    }


    // Warning: segfaults if pointer is not aligned!
    inline void advance_chunk(const uint8_t *in_)
    {
	const __m256i *in = (const __m256i *) (in_);
	
	advance1<0> (_mm256_load_si256(in));
	advance1<1> (_mm256_load_si256(in+1));
	advance1<2> (_mm256_load_si256(in+2));
	advance1<3> (_mm256_load_si256(in+3));
	advance1<4> (_mm256_load_si256(in+4));
	advance1<5> (_mm256_load_si256(in+5));
	advance1<6> (_mm256_load_si256(in+6));
	advance1<7> (_mm256_load_si256(in+7));
    }
};


// -------------------------------------------------------------------------------------------------


// 5 bit -> 6 bit
struct avx256_5bit_downsampler
{
    static constexpr int src_bytes_per_chunk = 5*32;
    static constexpr int dst_bytes_per_chunk = 3*32;
    
    avx256_m64_outbuf obuf;
    avx256_m64_splitter<40> sp40;
    
    __m256i R_reg;
    __m256i P_reg;

    __m256i c4;
    __m256i c6;
    __m256i c150;

    // See misc/make-dsmasks.py
    __m256i dsmask_5_4;
    __m256i dsmask_10_2;
    __m256i dsmask_20_1;
    

    static constexpr int nbits_out(int nbits_in)
    {
	return 6 * (nbits_in / 10);
    }

    
    avx256_5bit_downsampler(uint8_t *out_)
	: obuf(out_)
    {
    	R_reg = _mm256_set_epi64x(2, 8, 4, 0);
	P_reg = _mm256_set_epi64x(nbits_out(192), nbits_out(128), nbits_out(64), 0);
	
	c4 = _mm256_set1_epi64x(4);
	c6 = _mm256_set1_epi64x(6);
	c150 = _mm256_set1_epi64x(150);

	// See misc/mask-dsmasks.py
	dsmask_5_4 = _mm256_set1_epi64x(0x7c1f07c1fL);
	dsmask_10_2 = _mm256_set1_epi64x(0x3ff003ffL);
	dsmask_20_1 = _mm256_set1_epi64x(0xfffffL);	
    }


    inline __m256i inner_downsample(__m256i x)
    {
	// Input: each 64-bit integer in 'x' should look like
	//   junk[24] + x7 + x6 + x5 + x4 + x3 + x2 + x1 + x0    where xi is 5-bit
	//
	// Output: each 64-bit integer will look like
	//   zero[40] + (x7+x6) + (x5+x4) + (x3+x2) + (x1+x0)   where (xi+xj) is 6-bit

	x = avx256_ds_add<5> (x, dsmask_5_4);
	x = avx256_ds_shift<4> (x, dsmask_10_2);
	x = avx256_ds_shift<8> (x, dsmask_20_1);
	
	return x;
    }

    
    template<int I>
    inline void advance1(__m256i in)
    {
	__m256i a, b;
	sp40.split(a, b, in, R_reg);

	// Clear junk bits in 'b'.
	//
	// Have:
	//
	//   b = zero[40-R] + x[(64+R):40]
	//
	//     = zero[36] + junk[8] + x[60:40]  if R=0,2,4
	//       zero[32] + junk[2] + x[70:40]  if R=6,8
	
	__m256i lflag = _mm256_cmpgt_epi32(c6, R_reg);     // note that _epi32 is okay here
	__m256i mask = _mm256_andnot_si256(lflag, dsmask_10_2);   // [0,0,0,0] or [0,1,0,1]
	mask = _mm256_or_si256(mask, dsmask_20_1);                // [0,0,1,1] or [0,1,1,1]
	b = _mm256_and_si256(mask, b);

	// Downsample. After this block:
	//
	//  a = zero[28] + y[36:0]   if R < 6
	//      zero[22] + y[42:0]   if R >= 6

	a = inner_downsample(a);
	b = inner_downsample(b);
	b = _mm256_slli_epi64(b, 24);
	a = _mm256_or_si256(a, b);

	// Write output

	constexpr int P0 = nbits_out(256*I);
	constexpr int P1 = nbits_out(256*I + 64);
	constexpr int P2 = nbits_out(256*I + 128);
	constexpr int P3 = nbits_out(256*I + 192);
	constexpr int P4 = nbits_out(256*I + 256);
	
	obuf.advance<P0,P1,P2,P3,P4> (a, P_reg);

	// Advance to next iteration of the loop.
	//
	// R = (R+6)  if R < 4
	//   = (R-4)  if R >= 4
	//
	// P = (P+150) if R < 4
	//   = (P+156) if R >= 4

	lflag = _mm256_cmpgt_epi32(c4, R_reg);    // note that _epi32 is okay here
	R_reg = _mm256_add_epi64(R_reg, _mm256_and_si256(lflag,c6));
	R_reg = _mm256_sub_epi64(R_reg, _mm256_andnot_si256(lflag,c4));

	P_reg = _mm256_add_epi64(P_reg, c150);
	P_reg = _mm256_add_epi64(P_reg, _mm256_andnot_si256(lflag,c6));
    }


    // Warning: segfaults if pointer is not aligned!
    inline void advance_chunk(const uint8_t *in_)
    {
	const __m256i *in = (const __m256i *) (in_);
	
	advance1<0> (_mm256_load_si256(in));
	advance1<1> (_mm256_load_si256(in+1));
	advance1<2> (_mm256_load_si256(in+2));
	advance1<3> (_mm256_load_si256(in+3));
	advance1<4> (_mm256_load_si256(in+4));
    }
};


// -------------------------------------------------------------------------------------------------


// 6 bit -> 7 bit
struct avx256_6bit_downsampler
{
    static constexpr int src_bytes_per_chunk = 12*32;
    static constexpr int dst_bytes_per_chunk = 7*32;
    
    avx256_m64_outbuf obuf;
    avx256_m64_splitter<48> sp48;
    
    __m256i R_reg;
    __m256i P_reg;

    __m256i c4;
    __m256i c7;
    __m256i c147;
    
    // See misc/make-dsmasks.py
    __m256i dsmask_6_4;
    __m256i dsmask_12_2;
    __m256i dsmask_24_1;
    

    static constexpr int nbits_out(int nbits_in)
    {
	return 7 * (nbits_in / 12);
    }

    
    avx256_6bit_downsampler(uint8_t *out_)
	: obuf(out_)
    {
    	R_reg = _mm256_set_epi64x(0, 8, 4, 0);
	P_reg = _mm256_set_epi64x(nbits_out(192), nbits_out(128), nbits_out(64), 0);
	
	c4 = _mm256_set1_epi64x(4);
	c7 = _mm256_set1_epi64x(7);
	c147 = _mm256_set1_epi64x(147);

	// See misc/mask-dsmasks.py
	dsmask_6_4 = _mm256_set1_epi64x(0x3f03f03f03fL);
	dsmask_12_2 = _mm256_set1_epi64x(0xfff000fffL);
	dsmask_24_1 = _mm256_set1_epi64x(0xffffffL);
    }

    
    template<int I>
    inline void advance1(__m256i in)
    {
	__m256i a, b;
	sp48.split(a, b, in, R_reg);

	// Clear junk bits in 'b'.
	//
	// Have:
	//
	//   b = zero[48-R] + x[(64+R):48]
	//
	//     = zero[44] + junk[8] + x[12:0]   if R = 0 or 4
	//       zero[40] + x[24:0]             if R = 8

	__m256i hflag = _mm256_cmpgt_epi32(R_reg, c4);  // note that _epi32 is okay here
	__m256i mask = _mm256_or_si256(hflag, dsmask_12_2);
	b = _mm256_and_si256(mask, b);

	// Downsample. After this block:
	//
	//  a = zero[29] + y[35:0]   if R = 0 or 4
	//      zero[22] + y[42:0]   if R = 8

	a = avx256_ds_add<6> (a, dsmask_6_4);
	a = avx256_ds_shift<5> (a, dsmask_12_2);
	a = avx256_ds_shift<10> (a, dsmask_24_1);
	
	b = avx256_ds_add<6> (b, dsmask_6_4);
	b = avx256_ds_shift<5> (b, dsmask_12_2);
	// Need last ds_shift for 'a' but not 'b'

	b = _mm256_slli_epi64(b, 28);
	a = _mm256_or_si256(a, b);

	// Write output

	constexpr int P0 = nbits_out(256*I);
	constexpr int P1 = nbits_out(256*I + 64);
	constexpr int P2 = nbits_out(256*I + 128);
	constexpr int P3 = nbits_out(256*I + 192);
	constexpr int P4 = nbits_out(256*I + 256);
	
	obuf.advance<P0,P1,P2,P3,P4> (a, P_reg);

	// Advance to next iteration of the loop.
	//
	// R = (R+4)  if R = 0 or 4
	//   = 0      if R = 8
	//
	// P = (P+147) if R = 0 or 4
	//   = (P+154) if R = 8

	R_reg = _mm256_add_epi64(R_reg, c4);
	R_reg = _mm256_andnot_si256(hflag, R_reg);

	P_reg = _mm256_add_epi64(P_reg, c147);
	P_reg = _mm256_add_epi64(P_reg, _mm256_and_si256(hflag,c7));
    }


    // Warning: segfaults if pointer is not aligned!
    inline void advance_chunk(const uint8_t *in_)
    {
	const __m256i *in = (const __m256i *) (in_);
	
	advance1<0> (_mm256_load_si256(in));
	advance1<1> (_mm256_load_si256(in+1));
	advance1<2> (_mm256_load_si256(in+2));
	advance1<3> (_mm256_load_si256(in+3));
	advance1<4> (_mm256_load_si256(in+4));
	advance1<5> (_mm256_load_si256(in+5));
	advance1<6> (_mm256_load_si256(in+6));
	advance1<7> (_mm256_load_si256(in+7));
	advance1<8> (_mm256_load_si256(in+8));
	advance1<9> (_mm256_load_si256(in+9));
	advance1<10> (_mm256_load_si256(in+10));
	advance1<11> (_mm256_load_si256(in+11));
    }
};


// -------------------------------------------------------------------------------------------------


// 7 bit -> 8 bit
struct avx256_7bit_downsampler
{
    static constexpr int src_bytes_per_chunk = 7*32;
    static constexpr int dst_bytes_per_chunk = 4*32;
    
    avx256_m128_outbuf obuf;
    avx256_m64_splitter<56> sp56;
    
    __m256i R_reg;

    __m256i c4;
    __m256i c10;
    
    // See misc/make-dsmasks.py
    __m256i dsmask_7_4;
    __m256i dsmask_14_2;
    __m256i dsmask_28_1;
    __m256i mask64;
    

    static constexpr int nbytes_out(int nbits_in)
    {
	return (nbits_in / 14);
    }

    
    avx256_7bit_downsampler(uint8_t *out_)
	: obuf(out_)
    {
    	R_reg = _mm256_set_epi64x(10, 2, 8, 0);
	
	c4 = _mm256_set1_epi64x(4);
	c10 = _mm256_set1_epi64x(10);
	
	// See misc/mask-dsmasks.py
	dsmask_7_4 = _mm256_set1_epi64x(0x1fc07f01fc07fL);
	dsmask_14_2 = _mm256_set1_epi64x(0x3fff0003fffL);
	dsmask_28_1 = _mm256_set1_epi64x(0xfffffffL);
	mask64 = _mm256_set_epi64x(0, 0xffffffffffffffffL, 0, 0xffffffffffffffffL);
    }

    
    template<int I>
    inline void advance1(__m256i in)
    {
	constexpr int P0 = nbytes_out(256*I);
	constexpr int P1 = nbytes_out(256*I + 64);
	constexpr int P2 = nbytes_out(256*I + 128);
	constexpr int P3 = nbytes_out(256*I + 192);
	constexpr int P4 = nbytes_out(256*I + 256);

	// Byte shifts within 128-bit lanes
	constexpr int M0 = 8 - (P1-P0);
	constexpr int M1 = 8 - (P3-P2);
	
	// Byte counts within 128-bit lanes
	constexpr int N0 = P2-P0;
	constexpr int N1 = P4-P2;

	__m256i a, b;
	sp56.split(a, b, in, R_reg);
	
	// Clear junk bits in 'b'.
	//
	// Have:
	//
	//   b = zero[56-R] + x[(64+R):56]
	//
	//     = zero[52] + junk[12]            if R = 0,2,4
	//       zero[44] + junk[6] + x[14:0]   if R = 6,8,10,12

	__m256i hflag = _mm256_cmpgt_epi32(R_reg, c4);  // note that epi32 is okay here
	__m256i mask = _mm256_and_si256(hflag, dsmask_14_2);
	b = _mm256_and_si256(mask, b);

	// Downsample. After this block:
	//
	//  a = zero[29] + y[35:0]   if R = 0,2,4
	//      zero[22] + y[42:0]   if R = 6,6,10,12

	a = avx256_ds_add<7> (a, dsmask_7_4);
	a = avx256_ds_shift<6> (a, dsmask_14_2);
	a = avx256_ds_shift<12> (a, dsmask_28_1);

	// Need ds_shifts for 'a' but not 'b'
	b = avx256_ds_add<7> (b, dsmask_7_4);

	b = _mm256_slli_epi64(b, 32);
	a = _mm256_or_si256(a, b);

	// Shift within 128-bit lanes
	
	b = _mm256_andnot_si256(mask64, a);
	__m256i b0 = _mm256_srli_si256(b, M0);
	__m256i b1 = _mm256_srli_si256(b, M1);
	b = _mm256_blend_epi32(b0, b1, 0xf0);
	
	a = _mm256_and_si256(mask64, a);
	a = _mm256_or_si256(a, b);

	// Write output

	obuf.advance <P0,N0> (_mm256_extracti128_si256(a,0));
	obuf.advance <P2,N1> (_mm256_extracti128_si256(a,1));

	// Advance to next iteration of the loop.
	//
	// R = (R+4)   if R = 0,2,4,6,8
	//   = (R-10)  if R = 10,12

	__m256i lflag = _mm256_cmpgt_epi32(c10, R_reg);  // note that epi32 is okay here
	R_reg = _mm256_add_epi64(R_reg, _mm256_and_si256(lflag, c4));
	R_reg = _mm256_sub_epi64(R_reg, _mm256_andnot_si256(lflag, c10));
    }


    // Warning: segfaults if pointer is not aligned!
    inline void advance_chunk(const uint8_t *in_)
    {
	const __m256i *in = (const __m256i *) (in_);
	
	advance1<0> (_mm256_load_si256(in));
	advance1<1> (_mm256_load_si256(in+1));
	advance1<2> (_mm256_load_si256(in+2));
	advance1<3> (_mm256_load_si256(in+3));
	advance1<4> (_mm256_load_si256(in+4));
	advance1<5> (_mm256_load_si256(in+5));
	advance1<6> (_mm256_load_si256(in+6));
    }
};


}  // namespace pirate

#endif // _PIRATE_AVX256_DOWNSAMPLE_HPP
