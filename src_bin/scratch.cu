// A placeholder file (already integrated into Makefile) for debugging

#include <ksgpu.hpp>
// #include "../include/pirate/file_utils.hpp"

using namespace std;
using namespace ksgpu;
// using namespace pirate;


// -------------------------------------------------------------------------------------------------
//
// Input:
// b <-> ti0   th0 th1 th2 th3 th4 <-> ti1 ti2 ti3 ti4 ti5
//
// Output:
// b <-> ti5   th0 th1 th2 th3 th4 <-> ti0 ti1 ti2 ti3 ti4


__device__ __forceinline__ __half2 float16_transpose(__half2 x)
{
    const uint lane0 = (threadIdx.x >> 1) & 0xf;
    const uint lane1 = lane0 | 0x10;

    __half2 y0 = __shfl_sync(0xffffffff, x, lane0);
    __half2 y1 = __shfl_sync(0xffffffff, x, lane1);

    return (threadIdx.x & 1) ? __highs2half2(y0,y1) : __lows2half2(y0,y1);
}


// Launch with 32 threads and 1 block.
__global__ void ft_kernel(__half2 *p)
{
    p[threadIdx.x] = float16_transpose(p[threadIdx.x]);
}


static void test_ft()
{
    Array<float> src({32,2}, af_rhost | af_random);  // (register, thread, simd)
    Array<float> dst({32,2}, af_rhost | af_random);  // (register, thread, simd)
    
    for (int ti = 0; ti < 64; ti++) {
	int ti0 = ti & 1;
	int ti5 = ti >> 5;
	int ti04 = ti & 0x1f;
	int ti15 = ti >> 1;
	
	// Input:
	// b <-> ti0   th0 th1 th2 th3 th4 <-> ti1 ti2 ti3 ti4 ti5
	//
	// Output:
	// b <-> ti5   th0 th1 th2 th3 th4 <-> ti0 ti1 ti2 ti3 ti4
	
	dst.at({ ti04, ti5 }) = src.at({ ti15, ti0 });
    }

    Array<__half> garr = src.template convert<__half> ();
    garr = garr.to_gpu();

    ft_kernel<<<1,32>>> ((half2 *) garr.data);
    CUDA_PEEK("ft_kernel launch");

    assert_arrays_equal(dst, garr, "host", "gpu", {"th","b"});
    cout << "test_ft: pass" << endl;
}


// -------------------------------------------------------------------------------------------------

// Input:
// b <-> ti0   th0 th1 th2 th3 th4 <-> ti1 ti2 ti3 ti4 ti5    r <-> s
//
// Output:
// b <-> ti5   th0 th1 th2 th3 th4 <-> s ti1 ti2 ti3 ti4    r <-> ti0

__device__ __forceinline__ void initial_float16_transpose2(__half2 &x, __half2 &y)
{
    // Input:
    // b <-> ti0   th0 th1 th2 th3 th4 <-> ti1 ti2 ti3 ti4 ti5    r <-> s
    //
    // __shfl_sync()  s == ti5 (mod 2)
    // __shfl_sync()  s != ti5 (mod 2)
    //
    // Intermediate:
    // b <-> ti0   th0 th1 th2 th3 th4 <-> s ti1 ti2 ti3 ti4    r <-> ti5
    
    __half2 src0 = (threadIdx.x & 0x10) ? y : x;  // s == ti5 (mod 2)
    __half2 src1 = (threadIdx.x & 0x10) ? x : y;  // s != ti5 (mod 2)

    const uint lane0 = ((threadIdx.x >> 1) & 0xf) | (threadIdx.x << 4);  // lop3
    const uint lane1 = lane0 ^ 0x10;

    src0 = __shfl_sync(0xffffffff, src0, lane0);
    src1 = __shfl_sync(0xffffffff, src1, lane1);

    __half2 z0 = (threadIdx.x & 0x1) ? src1 : src0;
    __half2 z1 = (threadIdx.x & 0x1) ? src0 : src1;

    // Local transpose:
    // b <-> ti5   th0 th1 th2 th3 th4 <-> s ti1 ti2 ti3 ti4    r <-> ti0

    x = __lows2half2(z0, z1);
    y = __highs2half2(z0, z1);
}


// Launch with 32 threads and 1 block.
__global__ void ift2_kernel(__half2 *p)
{
    __half2 x = p[threadIdx.x];
    __half2 y = p[threadIdx.x + 32];

    initial_float16_transpose2(x, y);

    p[threadIdx.x] = x;
    p[threadIdx.x + 32] = y;
}


static void test_ift2()
{
    Array<float> src({2,32,2}, af_rhost | af_random);  // (register, thread, simd)
    Array<float> dst({2,32,2}, af_rhost | af_random);  // (register, thread, simd)
    
    for (int s = 0; s < 2; s++) {
	for (int ti = 0; ti < 64; ti++) {
	    int ti0 = ti & 1;
	    int ti5 = ti >> 5;
	    int ti15 = ti >> 1;
	    int ti14 = ti15 & 0xf;
	    
	    // Input:
	    // b <-> ti0   th0 th1 th2 th3 th4 <-> ti1 ti2 ti3 ti4 ti5    r <-> s
	    //
	    // Output:
	    // b <-> ti5   th0 th1 th2 th3 th4 <-> s ti1 ti2 ti3 ti4    r <-> ti0

	    // src.at({s, ti15, ti0 }) = 64*s + ti;
	    // src.at({s, ti15, ti0 }) = ((s==0) && (ti==0)) ? 1 : 0;
	    dst.at({ ti0, (2*ti14+s), ti5 }) = src.at({ s, ti15, ti0 });
	}
    }

    Array<__half> garr = src.template convert<__half> ();
    garr = garr.to_gpu();

    ift2_kernel<<<1,32>>> ((half2 *) garr.data);
    CUDA_PEEK("ift2_kernel launch");

    assert_arrays_equal(dst, garr, "host", "gpu", {"r","th","b"});
    cout << "test_ift2: pass" << endl;
}


// -------------------------------------------------------------------------------------------------


int main(int argc, char **argv)
{
    test_ft();
    test_ift2();
    return 0;
}
