#include "../include/pirate/TreeGriddingKernel.hpp"
#include "../include/pirate/BumpAllocator.hpp"
#include "../include/pirate/inlines.hpp"   // xdiv(), align_up(), pow2()
#include "../include/pirate/DedispersionConfig.hpp"  // for make_channel_map()

#include <cuda_fp16.h>
#include <algorithm>  // std::sort()
#include <ksgpu/xassert.hpp>
#include <ksgpu/cuda_utils.hpp>
#include <ksgpu/rand_utils.hpp>
#include <ksgpu/test_utils.hpp>
#include <ksgpu/KernelTimer.hpp>

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

    // Check that channel_map values are in-range and monotonically decreasing.
    for (long i = 0; i <= nchan; i++) {
	    double c = channel_map.data[i];
	    xassert((c >= 0) && (c <= nfreq));
	    if (i > 0)
	        xassert(channel_map.data[i-1] > c);
    }
    
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
            // channel_map is monotonically decreasing, so channel_map[n+1] < channel_map[n].
            // f0 is the lower bound, f1 is the upper bound.
            double f0 = params.channel_map.data[n+1];
            double f1 = params.channel_map.data[n];

            long if0 = max(long(f0), 0L);
            long if1 = min(long(f1)+1, F);

            for (long f = if0; f < if1; f++) {
                double flo = max(f0, double(f));
                double fhi = min(f1, double(f)+1);
                double w = max(fhi-flo, 0.0);
                
                for (long t = 0; t < T; t++)
                    outp[n*T + t] += float(w) * inp[f*T + t];
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

    resource_tracker.add_kernel("tree_gridding", B * (N+F+1) * T * S);
    resource_tracker.add_gmem_footprint("channel_map", (N+1) * sizeof(double), true);

    this->nchan_per_thread = 4;   // reasonable default (?)
    long ny = (N + nchan_per_thread - 1) / nchan_per_thread;
    ksgpu::assign_kernel_dims(this->nblocks, this->nthreads, T, ny, B);
}


void GpuTreeGriddingKernel::allocate(BumpAllocator &allocator)
{
    if (is_allocated)
        throw runtime_error("double call to GpuTreeGriddingKernel::allocate()");

    if (!(allocator.aflags & af_gpu))
        throw runtime_error("GpuTreeGriddingKernel::allocate(): allocator.aflags must contain af_gpu");
    if (!(allocator.aflags & af_zero))
        throw runtime_error("GpuTreeGriddingKernel::allocate(): allocator.aflags must contain af_zero");

    long nbytes_before = allocator.nbytes_allocated.load();

    // Copy host -> GPU.
    this->gpu_channel_map = allocator.allocate_array<double>({params.nchan + 1});
    this->gpu_channel_map.fill(params.channel_map);

    long nbytes_allocated = allocator.nbytes_allocated.load() - nbytes_before;
    xassert_eq(nbytes_allocated, resource_tracker.get_gmem_footprint("channel_map"));

    this->is_allocated = true;
}



inline __device__ void _set_zero(float &x) { x = 0.0f; }
inline __device__ void _set_zero(__half2 &x) { x = __float2half2_rn(0.0f); }

inline __device__ float _mult(double w, float y) { return float(w) * y; }
inline __device__ __half2 _mult(double w, __half2 y) { return __float2half2_rn(float(w)) * y; }


// cuda kernel supports a flexible thread/block mapping:
//
//   threadIdx.x = blockIdx.x = time index  (0 <= t < T)
//   threadIdx.y = blockIdx.y = tree channel index  (0 <= n < N/Nbs)
//   threadIdx.z = blockIdx.z = beam index  (0 <= b < B)
//
// FIXME: using double precision in a GPU kernel!! This is a temporary kludge.

template<typename T32>
__global__ void gpu_tree_gridding_kernel(
    T32 *out,
    const T32 *in,
    const double *channel_map,
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
    double cmap = channel_map[n + ix];

    // Code optimization is imperfect here, but should still
    // be fast enough to be memory bandwidth limited.
    
    for (int m = 0; m < M; m++) {
        // channel_map is monotonically decreasing, so channel_map[n+1] < channel_map[n].
        // f0 is the lower bound, f1 is the upper bound.
        double f0 = __shfl_sync(~0u, cmap, m+1);
        double f1 = __shfl_sync(~0u, cmap, m);
        
        int if0 = max(int(f0), 0);
        int if1 = min(int(f1)+1, nfreq);

        T32 t;
        _set_zero(t);

        for (int f = if0; f < if1; f++) {
            double flo = max(f0, double(f));
            double fhi = min(f1, double(f)+1);
            double w = fhi - flo;
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
         k.gpu_channel_map.data,                   // const double *channel_map,
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


// -------------------------------------------------------------------------------------------------
//
// GpuTreeGriddingKernel::test()


void GpuTreeGriddingKernel::test()
{
    TreeGriddingKernelParams params;
    params.dtype = (rand_uniform() < 0.5) ? Dtype(df_float,16) : Dtype(df_float,32);

    auto v = ksgpu::random_integers_with_bounded_product(3, 1000);
    int Ncl = xdiv(1024, params.dtype.nbits);   // elements per cache line
    int B = v[0];
    int F = v[1];
    int T = v[2] * Ncl;
    int N = rand_int(1, 2000/(v[0]*v[2]));
    
    long hs_in = rand_int(F*T, 2*F*T);   // host beam stride, input array
    long gs_in = rand_int(F*T, 2*F*T);   // gpu beam stride, input array
    long hs_out = rand_int(N*T, 2*N*T);  // host beam stride, output array
    long gs_out = rand_int(N*T, 2*N*T);  // gpu beam stride, output array

    gs_in = align_up(gs_in, Ncl);
    gs_out = align_up(gs_out, Ncl);
    
    params.beams_per_batch = B;
    params.nfreq = F;
    params.nchan = N;
    params.ntime = T;

    cout << "TreeGriddingKernel::test()\n"
         << "    dtype = " << params.dtype << "\n"
         << "    beams_per_batch = " << B << "\n"
         << "    nfreq = " << F << "\n"
         << "    nchan = " << N << "\n"
         << "    ntime = " << T << "\n"
         << "    host_src_beam_stride = " << hs_in << "\n"
         << "    host_dst_beam_stride = " << hs_out << "\n"
         << "    gpu_src_beam_stride = " << gs_in << "\n"
         << "    gpu_dst_beam_stride = " << gs_out << endl;
    
    // Generate a monotonically decreasing channel_map: cvec[0]=F, cvec[N]=0.
    vector<double> cvec(N+1);
    cvec[0] = F;
    cvec[N] = 0;
    for (int i = 1; i < N; i++)
        cvec[i] = rand_uniform(0, F);

    std::sort(cvec.begin(), cvec.end(), std::greater<double>());

    params.channel_map = Array<double> ({N+1}, af_rhost | af_zero);
    memcpy(params.channel_map.data, &cvec[0], (N+1) * sizeof(double));
    
    Array<float> hsrc({B,F,T}, {hs_in,T,1}, af_uhost | af_random | af_guard);
    Array<float> hdst({B,N,T}, {hs_out,T,1}, af_uhost | af_zero | af_guard);

    Array<void> gsrc(params.dtype, {B,F,T}, {gs_in,T,1}, af_gpu | af_zero | af_guard);
    Array<void> gdst(params.dtype, {B,N,T}, {gs_out,T,1}, af_gpu | af_zero | af_guard);
    gsrc.fill(hsrc.convert(params.dtype));

    ReferenceTreeGriddingKernel hkernel(params);
    hkernel.apply(hdst, hsrc);

    GpuTreeGriddingKernel gkernel(params);
    BumpAllocator allocator(af_gpu | af_zero, -1);  // dummy allocator
    gkernel.allocate(allocator);
    gkernel.launch(gdst, gsrc, nullptr);   // null stream
    
    assert_arrays_equal(hdst, gdst, "hdst", "gdst", {"b","n","t"});
}


// -------------------------------------------------------------------------------------------------
//
// GpuTreeGriddingKernel::time()


void GpuTreeGriddingKernel::time()
{
    Dtype fp16(df_float, 16);
    Dtype fp32(df_float, 32);

    // Construct a throwaway (CHORD-like) DedispersionConfig to get channel_map via make_channel_map().
    DedispersionConfig dconfig = DedispersionConfig::make_mini_chord(fp16);  // dtype doesn't matter for channel_map
    
    // Generate channel_map (stored in CPU memory).
    Array<double> channel_map = dconfig.make_channel_map();
    
    long nfreq = dconfig.get_total_nfreq();  // 28160
    long nchan = pow2(dconfig.tree_rank);    // 65536
    long ntime = 2048;
    long beams_per_batch = 4;
    long nstreams = 2;  // for latency hiding

    // Time both float32 and float16.
    for (int pass = 0; pass < 2; pass++) {
        Dtype dtype = (pass == 0) ? fp32 : fp16;
        
        TreeGriddingKernelParams params;
        params.dtype = dtype;
        params.nfreq = nfreq;
        params.nchan = nchan;
        params.ntime = ntime;
        params.beams_per_batch = beams_per_batch;
        params.channel_map = channel_map;
        
        GpuTreeGriddingKernel kernel(params);
        BumpAllocator time_allocator(af_gpu | af_zero, -1);  // dummy allocator
        kernel.allocate(time_allocator);
        
        // Allocate GPU arrays.
        Array<void> gsrc(dtype, {nstreams, beams_per_batch, nfreq, ntime}, af_gpu | af_zero);
        Array<void> gdst(dtype, {nstreams, beams_per_batch, nchan, ntime}, af_gpu | af_zero);
        
        // Print header.
        double input_gb = double(beams_per_batch) * nfreq * ntime * (dtype.nbits / 8) / 1.0e9;
        double output_gb = double(beams_per_batch) * nchan * ntime * (dtype.nbits / 8) / 1.0e9;
        double bw_per_launch = 1.0e-9 * kernel.resource_tracker.get_gmem_bw("tree_gridding");

        cout << "\nGpuTreeGriddingKernel::time()\n"
             << "    dtype = " << dtype << "\n"
             << "    nfreq = " << nfreq << "\n"
             << "    nchan = " << nchan << "\n"
             << "    ntime = " << ntime << "\n"
             << "    beams_per_batch = " << beams_per_batch << "\n"
             << "    input size per batch = " << input_gb << " GB\n"
             << "    output size per batch = " << output_gb << " GB\n"
             << "    bandwidth per launch = " << bw_per_launch << " GB\n"
             << endl;
        
        // Use KernelTimer with 500 iterations, 2 streams.
        int niter = 500;
        int print_interval = 50;
        KernelTimer kt(niter, nstreams);
        
        while (kt.next()) {
            Array<void> s = gsrc.slice(0, kt.istream);
            Array<void> d = gdst.slice(0, kt.istream);

            kernel.launch(d, s, kt.stream);
            
            if (kt.warmed_up && ((kt.curr_iteration+1) % print_interval == 0)) {
                cout << "    iter " << (kt.curr_iteration+1) << "/" << niter
                     << ": dt = " << (kt.dt * 1.0e3) << " ms"
                     << ", bandwidth = " << (bw_per_launch / kt.dt) << " GB/s" << endl;
            }
        }
    }
}


}  // namespace pirate
