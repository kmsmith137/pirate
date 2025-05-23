#include "../include/pirate/cuda_kernels/peak_finding.hpp"

#include <ksgpu/Array.hpp>
#include <ksgpu/rand_utils.hpp>

using namespace std;
using namespace ksgpu;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// Launch with one warp!
// 'pstate' has length (O*I).
// 'data' has shape (O, 32*ninner).

template<int I, int O>
__global__ void pf_ringbuf_test_kernel(int *data, int *pstate, int ninner, int data_stride)
{
    int x[O];

    pf_ringbuf<int,I,O> rb;
    rb.load(pstate);

    for (int n = 0; n < ninner; n++) {
	#pragma unroll
	for (int r = 0; r < O; r++)
	    x[r] = data[r*data_stride + 32*n + threadIdx.x];
	
	rb.template multi_advance<0,O> (x);
	
	#pragma unroll
	for (int r = 0; r < O; r++)
	    data[r*data_stride + 32*n + threadIdx.x] = x[r];
    }

    rb.store(pstate);
}


// Function type for pf_ringbuf_test_kernel<I,O>
using rb_kernel_t = void (*)(int *, int *, int, int);


// Ugh
template<int I>
static rb_kernel_t get_rb_kernel1(int O)
{
    if (O == 1)
	return pf_ringbuf_test_kernel<I,1>;
    if (O == 2)
	return pf_ringbuf_test_kernel<I,2>;
    if (O == 3)
	return pf_ringbuf_test_kernel<I,3>;
    if (O == 4)
	return pf_ringbuf_test_kernel<I,4>;
    if (O == 5)
	return pf_ringbuf_test_kernel<I,5>;
    if (O == 6)
	return pf_ringbuf_test_kernel<I,6>;
    if (O == 7)
	return pf_ringbuf_test_kernel<I,7>;
    if (O == 8)
	return pf_ringbuf_test_kernel<I,8>;
    if (O == 9)
	return pf_ringbuf_test_kernel<I,9>;
    
    throw runtime_error("get_rb_kernel: invalid O");
}


// Ugh
static rb_kernel_t get_rb_kernel(int I, int O)
{
    if (I == 1)
	return get_rb_kernel1<1> (O);
    if (I == 2)
	return get_rb_kernel1<2> (O);
    if (I == 4)
	return get_rb_kernel1<4> (O);
    if (I == 8)
	return get_rb_kernel1<8> (O);
    if (I == 16)
	return get_rb_kernel1<16> (O);
    
    throw runtime_error("get_rb_kernel: invalid I");
}


void test_gpu_pf_ringbuf()
{
    int I = (1 << rand_int(0,5));
    int O = rand_int(1, 10);
    int ninner = rand_int(1, 10);
    int nouter = rand_int(1, 10);
    rb_kernel_t kernel = get_rb_kernel(I,O);
    
    cout << "test_gpu_ringbuf: I=" << I << ", O=" << O << ", ninner=" << ninner << ", nouter=" << nouter << endl;

    int T = 32*nouter*ninner;
    Array<int> hdata({O,T+I}, af_random | af_rhost);
    
    Array<int> pstate = hdata.slice(1,0,I);   // shape (O,I)
    pstate = pstate.clone().to_gpu();

    Array<int> gdata = hdata.slice(1,I,T+I);  // shape (O,T)
    gdata = gdata.to_gpu();

    for (int n = 0; n < nouter; n++)
	kernel<<<1,32>>> (gdata.data + 32*ninner*n, pstate.data, ninner, gdata.strides[0]);

    assert_arrays_equal(
	hdata.slice(1,0,T),  // shape (O,T)
	gdata,
	"host", "gpu",
	{"r","t"}
    );
}


}  // namespace pirate
