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


__device__ __forceinline__ __half2 initial_float16_transpose1(__half2 x)
{
    const uint lane0 = (threadIdx.x >> 1) & 0xf;
    const uint lane1 = lane0 | 0x10;

    __half2 y0 = __shfl_sync(0xffffffff, x, lane0);
    __half2 y1 = __shfl_sync(0xffffffff, x, lane1);

    return (threadIdx.x & 1) ? __highs2half2(y0,y1) : __lows2half2(y0,y1);
}


// Launch with 32 threads and 1 block.
__global__ void ift1_kernel(__half2 *p)
{
    p[threadIdx.x] = initial_float16_transpose1(p[threadIdx.x]);
}


static void test_ift1()
{
    cout << "test_ift1" << endl;

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

    ift1_kernel<<<1,32>>> ((half2 *) garr.data);
    CUDA_PEEK("ift1_kernel launch");

    assert_arrays_equal(dst, garr, "host", "gpu", {"th","b"});
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
    cout << "test_ift2" << endl;

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
}


// -------------------------------------------------------------------------------------------------
//
// final_float16_transpose1
//
// Input (where 1 <= M <= 5)
// b <-> ti_M   th0 th1 th2 th3 th4 <->  ti_{M+1} ..ti_5 ti_0 .. ti_{M-1}
//
// Output:
// b <-> ti_0   th0 th1 th2 th3 th4 <-> ti_1 ti_2 ti_3 ti_4 ti_5


__device__ __forceinline__ __half2 final_float16_transpose1(__half2 x, int M)
{
    // M=1
    // b <-> ti1   threads <-> ti2 ... ti5 ti0
    // b <-> ti0   threads <-> ti1 ... ti5

    // lane0 = (source for ti0=0)
    uint lane0a = (threadIdx.x << (6-M));
    uint lane0b = (threadIdx.x >> M) & ((1 << (5-M)) - 1);
    uint lane0 = lane0a | lane0b;

    // lane1 = (source for ti0=1)
    uint lane1 = lane0 ^ (1 << (5-M));

    __half2 y0 = __shfl_sync(0xffffffff, x, lane0);  // ti0=0
    __half2 y1 = __shfl_sync(0xffffffff, x, lane1);  // ti0=1

    return (threadIdx.x & (1 << (M-1))) ? __highs2half2(y0,y1) : __lows2half2(y0,y1);
}


__global__ void fft1_kernel(__half2 *p, int M)
{
    p[threadIdx.x] = final_float16_transpose1(p[threadIdx.x], M);
}



static void test_fft1()
{
    Array<float> src({32,2}, af_rhost | af_random);  // (register, thread, simd)
    Array<float> dst({32,2}, af_rhost | af_random);  // (register, thread, simd)
    
    for (int M = 1; M <= 5; M++) {
	cout << "test_fft1: M=" << M << endl;
	
	for (int ti = 0; ti < 64; ti++) {
	    int ti0 = ti & 1;
	    int tiM = (ti >> M) & 1;
	    
	    int ti15 = ti >> 1;
	    int ti_0_M1 = ti & ((1 << M) - 1);  // ti_0 ... ti_{M-1}
	    int ti_M1_5 = ti >> (M+1);          // ti_{M+1} ... ti_5
	    
	    // Input:
	    // b <-> ti_M   th0 th1 th2 th3 th4 <-> ti_{M+1} ... ti_5 ti_0 ... ti_{M-1}
	    //
	    // Output:
	    // b <-> ti0   th0 th1 th2 th3 th4 <-> ti1 ti2 ti3 ti4 ti5

	    dst.at({ ti15, ti0 }) = src.at({ ti_M1_5 | (ti_0_M1 << (5-M)) , tiM });
	}

	Array<__half> garr = src.template convert<__half> ();
	garr = garr.to_gpu();

	fft1_kernel<<<1,32>>> ((half2 *) garr.data, M);
	CUDA_PEEK("ift2_kernel launch");

	assert_arrays_equal(dst, garr, "host", "gpu", {"th","b"});
    }
}



// -------------------------------------------------------------------------------------------------
//
// final_float16_transpose2
// 
// Input (where 1 <= M <= 5)
// b <-> ti_M   th0 th1 th2 th3 th4 <->  ti_{M+1} ..ti_5 ti_0 .. ti_{M-1}    r <-> s
//
// Output:
// b <-> ti_0   th0 th1 th2 th3 th4 <-> ti_1 ti_2 ti_3 ti_4 ti_5   r <-> s


#if 0

__device__ __forceinline__ void final_float16_transpose2(__half2 &x0, __half2 &x1, int M)
{
    __half2 y0 = __lows2half2(x0, x1);
    __half2 y1 = __highs2half2(x0, x1);
    
    // b <-> s   th0 th1 th2 th3 th4 <-> ti_{M+1} ..ti_5 ti_0 .. ti_{M-1}    r <-> ti_M

    __half2 z0 = (threadIdx.x & (1 << (M-1))) ? y1 : y0;
    __half2 z1 = (threadIdx.x & (1 << (M-1))) ? y0 : y1;

    // b <-> s   th0 th1 th2 th3 th4 <-> ti_{M+1} ..ti_5 ti_0 .. ti_{M-1}    r <-> (ti_M ^ ti0)

    uint lane00 = (threadIdx.x << (6-M));
    uint lanexx = (threadIdx.x << (5-M)) & xx;
    uint lane1 = ;
    
    z0 = __shfl_sync(0xffffffff, z0, lane0);
    z1 = __shfl_sync(0xffffffff, z1, lane1);

    // b <-> s   th0 th1 th2 th3 th4 <-> ti_1 ti_2 ... ti_5    r <-> (ti_M ^ ti0)

    y0 = ;  // conditional move
    y1 = ;  // conditional move
    
    // b <-> s   th0 th1 th2 th3 th4 <-> ti_1 ti_2 ... ti_5    r <-> ti0

    x0 = __lows2half2(y0, y1);
    x1 = __highs2half2(y0, y1);
    
    // b <-> ti0    th0 th1 th2 th3 th4 <-> ti_1 ti_2 ... ti_5    r <-> s
}

#endif


// -------------------------------------------------------------------------------------------------



int main(int argc, char **argv)
{
    test_ift1();
    test_ift2();
    test_fft1();
    return 0;
}
