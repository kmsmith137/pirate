#include <iostream>
#include <ksgpu/Array.hpp>

#include "../include/pirate/internals/bitvec.hpp"
#include "../include/pirate/internals/cpu_downsample.hpp"

using namespace std;
using namespace ksgpu;
using namespace pirate;


static void test_cpu_downsampler(int src_bit_depth, int nchunks)
{
    int S = src_bit_depth;
    int D = src_bit_depth+1;
    
    int src_nbytes = nchunks * cpu_downsample_src_bytes_per_chunk(S);
    int src_nbits = 8 * src_nbytes;
    xassert_divisible(src_nbits, 2*S);

    int dst_nelts = src_nbits / (2*S);
    int dst_nbits = dst_nelts * D;
    int dst_nbytes = dst_nbits / 8;
    xassert(dst_nbits == (8 * dst_nbytes));
    
    vector<bool> src_bitvec = make_bitvec(src_nbits);
    
    Array<unsigned char> src({src_nbytes}, af_uhost | af_zero);
    write_bitvec(src.data, src_bitvec);

    Array<unsigned char> dst({dst_nbytes}, af_uhost | af_zero);
    cpu_downsample(S, src.data, dst.data, src_nbytes, dst_nbytes);

    vector<bool> dst_bitvec = read_bitvec(dst.data, dst_nbits);
    bool fail = false;
    
    for (int i = 0; i < dst_nelts; i++) {
	int x = bitvec_to_uint64(src_bitvec, S, (2*i) * S);
	int y = bitvec_to_uint64(src_bitvec, S, (2*i+1) * S);
	int z = bitvec_to_uint64(dst_bitvec, D, i*D);

	if (z == x+y)
	    continue;
	
	if (!fail)
	    cout << "test_cpu_downsampler failed: src_bit_depth=" << src_bit_depth << ", nchunks=" << nchunks << endl;
	
	cout << "   at i=" << i
	     << ": x = src[" << ((2*i+1)*S) << ":" << ((2*i)*S) << "] = " << x
	     << ", y = src[" << ((2*i+2)*S) << ":" << ((2*i+1)*S) << "] = " << y
	     << ", z = dst[" << ((i+1)*D) << ":" << (i*D) << "] = " << z
	     << ", expected (x+y)=" << (x+y) << endl;
	
	fail = true;
    }

    if (fail)
	exit(1);    
}


// -------------------------------------------------------------------------------------------------


int main(int argc, char **argv)
{
    for (int src_bit_depth = 4; src_bit_depth <= 7; src_bit_depth++) {
	for (int i = 1; i <= 10; i++)
	    test_cpu_downsampler(src_bit_depth, i);
	cout << "test-cpu-downsampler(src_bit_depth=" << src_bit_depth << "): pass" << endl;
    }
    
    return 0;
}
