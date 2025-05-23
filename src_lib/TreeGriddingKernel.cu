#include "../include/pirate/TreeGriddingKernel.hpp" 
#include "../include/pirate/inlines.hpp"   // xdiv()

#include <cuda_fp16.h>
#include <ksgpu/xassert.hpp>
#include <ksgpu/cuda_utils.hpp>
#include <ksgpu/device_transposes.hpp>   // FULL_MASK

using namespace std;
using namespace ksgpu;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// -------------------------------------------------------------------------------------------------


const TreeGriddingKernelParams &TreeGriddingKernelParams::validate() const
{
    Dtype fp16(df_float, 16);
    Dtype fp32(df_float, 32);
    
    xassert(nfreq > 0);
    xassert(nchan > 0);
    xassert(ntime > 0);
    xassert(beams_per_batch > 0);    
    xassert((dtype == fp16) || (dtype == fp32));
    xassert_divisible(ntime, xdiv(1024,dtype.nbits));
    
    xassert(channel_map.on_host());
    xassert_shape_eq(channel_map, ({nchan+1}));
    xassert(channel_map.is_fully_contiguous());

    return *this;
}


// -------------------------------------------------------------------------------------------------


ReferenceTreeGriddingKernel::ReferenceTreeGriddingKernel(const TreeGriddingKernelParams &params_) :
    params(params_.validate())
{ }


void ReferenceTreeGriddingKernel::apply(Array<float> &out, const Array<float> &in)
{
    long B = params.beams_per_batch;
    long N = params.nchan;
    long F = params.nfreq;
    long T = params.ntime;
    
    xassert(out.on_host());
    xassert_shape_eq(out, ({B,N,T}));
    xassert(out.get_ncontig() >= 2);
    
    xassert(in.on_host());
    xassert_shape_eq(in, ({B,F,T}));
    xassert(in.get_ncontig() >= 2);

    for (long b = 0; b < B; b++) {
	float *outp = out.data + b * out.strides[0];
	const float *inp = in.data + b * in.strides[0];

	memset(outp, 0, N * T * sizeof(float));
	       
	for (long n = 0; n < N; n++) {
	    float f0 = params.channel_map.data[n];
	    float f1 = params.channel_map.data[n+1];

	    long if0 = max(long(f0), 0L);
	    long if1 = min(long(f1)+1, F);

	    for (long f = if0; f < if1; f++) {
		float flo = max(f0, float(f));
		float fhi = min(f1, float(f)+1);
		float w = max(fhi-flo, 0.0f);
		
		for (long t = 0; t < T; t++)
		    outp[n*T + t] += w * inp[f*T + t];
	    }
	}
    }
}


// -------------------------------------------------------------------------------------------------


GpuTreeGriddingKernel::GpuTreeGriddingKernel(const TreeGriddingKernelParams &params_) :
    params(params_.validate())
{
    long S = xdiv(params.dtype.nbits, 8);
    long B = params.beams_per_batch;
    long N = params.nchan;
    long F = params.nfreq;
    long T = params.ntime;
    
    this->bw_per_launch.nbytes_gmem = B * (N+F) * T * S;
    this->nchan_per_thread = 4;   // reasonable default (?)

    long ny = (N + nchan_per_thread - 1) / nchan_per_thread;
    ksgpu::assign_kernel_dims(this->nblocks, this->nthreads, T, ny, B);
}


void GpuTreeGriddingKernel::allocate()
{
    if (is_allocated)
	throw runtime_error("double call to GpuTreeGriddingKernel::allocate()");

    // Copy host -> GPU.
    this->gpu_channel_map = params.channel_map.to_gpu();
    this->is_allocated = true;
}



inline __device__ void _set_zero(float &x) { x = 0.0f; }
inline __device__ void _set_zero(__half2 &x) { x = __float2half2_rn(0.0f); }

inline __device__ float _mult(float x, float y) { return x*y; }
inline __device__ __half2 _mult(float x, __half2 y) { return __float2half2_rn(x) * y; }


// cuda kernel supports a flexible thread/block mapping:
//
//   threadIdx.x = blockIdx.x = time index  (0 <= t < T)
//   threadIdx.y = blockIdx.y = tree channel index  (0 <= n < N/Nbs)
//   threadIdx.z = blockIdx.z = beam index  (0 <= b < B)

template<typename T32>
__global__ void gpu_tree_gridding_kernel(
    T32 *out,
    const T32 *in,
    const float *channel_map,
    long out_bstride32,
    long in_bstride32,
    int nchan_per_thread,   // must be <= 31
    int nbeams,  // beams per batch
    int nchan,   // number of tree channels
    int nfreq,
    int ntime32)
{
    static_assert(sizeof(T32) == 4);

    int t = (blockIdx.x * blockDim.x + threadIdx.x);   // time index
    int n = (blockIdx.y * blockDim.y + threadIdx.y) * nchan_per_thread;   // base tree channel index
    int b = (blockIdx.z * blockDim.z + threadIdx.z);   // beam index

    if ((t >= ntime32) || (n >= nchan) || (b >= nbeams))
	return;
    
    // Number of tree channels processed by thread.
    int M = min(nchan_per_thread, nchan - n);
    
    // Absorb (t,n,b) into pointers.
    // After this, each thread can treat 'out' as length-M array with stride ntime32,
    // and 'in' as length-nfreq arary with stride ntime32.
    out += (b * out_bstride32) + (long(n) * long(ntime32)) + t;
    in += (b * in_bstride32) + t;

    // Read channel_map[n:(n+M+1)]
    int ix = min(threadIdx.x & 0x1f, M);
    float cmap = channel_map[n + ix];

    // Code optimization is imperfect here, but should still
    // be fast enough to be memory bandwidth limited.
    
    for (int m = 0; m < M; m++) {
	float f0 = __shfl_sync(FULL_MASK, cmap, m);
	float f1 = __shfl_sync(FULL_MASK, cmap, m+1);
	
	int if0 = max(int(f0), 0);
	int if1 = min(int(f1)+1, nfreq);

	T32 t;
	_set_zero(t);

	for (int f = if0; f < if1; f++) {
	    float flo = max(f0, float(f));
	    float fhi = min(f1, float(f)+1);
	    float w = fhi - flo;
	    t += _mult(w, in[f*ntime32]);
	}

	out[m*ntime32] = t;
    }
}


template<typename T32>
static void _launch(const GpuTreeGriddingKernel &k, Array<void> &out, const Array<void> &in, cudaStream_t stream)
{
    int s = xdiv(32, k.params.dtype.nbits);

    // Note that beam_strides and 'ntime' get divided by s.
    
    gpu_tree_gridding_kernel<T32>
	<<< k.nblocks, k.nthreads, 0, stream >>>
	(reinterpret_cast<T32 *> (out.data),       // T32 *out,
	 reinterpret_cast<const T32 *> (in.data),  // const T32 *in,
	 k.gpu_channel_map.data,                   // const float *channel_map,
	 xdiv(out.strides[0], s),                  // long out_bstride32,
	 xdiv(in.strides[0], s),                   // long in_bstride32,
	 k.nchan_per_thread,                       // int nchan_per_thread
	 k.params.beams_per_batch,                 // int beams per batch
	 k.params.nchan,                           // int nchan (number of tree channels)
	 k.params.nfreq,                           // int nfreq
	 xdiv(k.params.ntime, s));                 // int ntime32
}


void GpuTreeGriddingKernel::launch(Array<void> &out, const Array<void> &in, cudaStream_t stream)
{
    long B = params.beams_per_batch;
    long N = params.nchan;
    long F = params.nfreq;
    long T = params.ntime;
    
    xassert(out.on_gpu());
    xassert_shape_eq(out, ({B,N,T}));
    xassert_eq(out.dtype, params.dtype);
    xassert(out.get_ncontig() >= 2);
    
    xassert(in.on_gpu());
    xassert_shape_eq(in, ({B,F,T}));
    xassert_eq(in.dtype, params.dtype);
    xassert(in.get_ncontig() >= 2);

    xassert(this->is_allocated);
    
    Dtype fp16(df_float, 16);
    Dtype fp32(df_float, 32);

    if (params.dtype == fp32)
	_launch<float> (*this, out, in, stream);
    else if (params.dtype == fp16)
	_launch<__half2> (*this, out, in, stream);
    else
	throw runtime_error("GpuTreeGriddingKernel::launch(): couldn't find kernel matching dtype");
    
    CUDA_PEEK("GpuTreeGriddingKernel::launch");
}


}  // namespace pirate
