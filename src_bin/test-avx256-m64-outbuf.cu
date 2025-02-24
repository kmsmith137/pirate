#include <iostream>
#include <ksgpu/Array.hpp>

#include "../include/pirate/avx256/m64_outbuf.hpp"
#include "../include/pirate/avx256/downsample.hpp"
#include "../include/pirate/internals/bitvec.hpp"

using namespace std;
using namespace ksgpu;
using namespace pirate;


template<int P0, int P1, int P2, int P3, int P4>
static void test_avx256_m64_outbuf()
{
    int Q0 = P0 % 64;
    
    vector<bool> bvec0 = make_bitvec(Q0);
    vector<bool> bvec1 = make_bitvec(P1-P0);
    vector<bool> bvec2 = make_bitvec(P2-P1);
    vector<bool> bvec3 = make_bitvec(P3-P2);
    vector<bool> bvec4 = make_bitvec(P4-P3);

    vector<bool> bvec = concat_bitvec(bvec0, bvec1);
    bvec = concat_bitvec(bvec, bvec2);
    bvec = concat_bitvec(bvec, bvec3);
    bvec = concat_bitvec(bvec, bvec4);

    Array<uint8_t> ref_out({40}, af_uhost | af_zero);
    write_bitvec(ref_out.data, bvec);
    
    uint64_t x0 = bitvec_to_uint64(bvec0);
    uint64_t x1 = bitvec_to_uint64(bvec1);
    uint64_t x2 = bitvec_to_uint64(bvec2);
    uint64_t x3 = bitvec_to_uint64(bvec3);
    uint64_t x4 = bitvec_to_uint64(bvec4);
	
    __m256i x = _mm256_set_epi64x(x4,x3,x2,x1);
    __m256i P = _mm256_set_epi64x(P3,P2,P1,P0);

    Array<uint8_t> kern_out({40}, af_uhost | af_zero);
    avx256_m64_outbuf buf(kern_out.data);
    
    buf.out_buf = x0;
    buf.advance<P0,P1,P2,P3,P4> (x, P);
    *(buf.out) = buf.out_buf;  // flush

    // Kernel uses streaming writes, so don't forget the fence!!
    _mm_mfence();
    
    bool fail = false;
    
    for (int i = 0; i < 320; i++) {
	int kern_bit = (kern_out.data[i/8] >> (i%8)) & 1;
	int ref_bit = (ref_out.data[i/8] >> (i%8)) & 1;
	
	if (kern_bit == ref_bit)
	    continue;

	if (!fail) {
	    cout << "test_avx256_m64_outbuf failed: P0=" << P0 << ", P1=" << P1
		 << ", P2=" << P2 << ", P3=" << P3 << ", P4=" << P4 << endl;
	}
    
	cout << "  bit " << i << ": kern=" << kern_bit << ", ref=" << ref_bit << endl;
	fail = true;
    }

    if (fail)
	exit(1);
}


template<int I>
static void test_d4()
{
    if constexpr (I > 0)
	test_d4<I-1> ();
    
    constexpr int P0 = 160*I;
    constexpr int P1 = 160*I + 40;
    constexpr int P2 = 160*I + 80;
    constexpr int P3 = 160*I + 120;
    constexpr int P4 = 160*I + 160;

    test_avx256_m64_outbuf<P0,P1,P2,P3,P4> ();
}


template<int I>
static void test_d5()
{
    if constexpr (I > 0)
	test_d5<I-1> ();
    
    constexpr int P0 = avx256_5bit_downsampler::nbits_out(256*I);
    constexpr int P1 = avx256_5bit_downsampler::nbits_out(256*I + 64);
    constexpr int P2 = avx256_5bit_downsampler::nbits_out(256*I + 128);
    constexpr int P3 = avx256_5bit_downsampler::nbits_out(256*I + 192);
    constexpr int P4 = avx256_5bit_downsampler::nbits_out(256*I + 256);

    test_avx256_m64_outbuf<P0,P1,P2,P3,P4> ();
}



int main(int argc, char **argv)
{
    for (int i = 0; i < 10; i++) {
	test_d4<7> ();
	test_d5<4> ();
    }

    cout << "test-avx256-m64-outbuf: pass" << endl;
    return 0;
}
