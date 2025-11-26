#include <iostream>
#include <ksgpu/Array.hpp>
#include <ksgpu/cuda_utils.hpp>
#include <ksgpu/test_utils.hpp>
#include <ksgpu/xassert.hpp>

#include "../include/pirate/loose_ends/gpu_downsample.hpp"
#include "../include/pirate/tests.hpp"

using namespace std;
using namespace ksgpu;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// FIXME could move somewhere more general
static Array<float> make_random_weights(const vector<long> &shape)
{
    Array<float> ret(shape, af_rhost);
    xassert(ret.is_fully_contiguous());

    for (long i = 0; i < ret.size; i++)
        ret.data[i] = rand_uniform();
    
    return ret;
}


// FIXME currently assuming transpose_output=true
static void test_gpu_downsample(int Df, int Dt, int nbeams, int nfreq_dst, int ntime_dst, int src_bstride, int src_fstride, int dst_bstride, int dst_tstride)
{
    cout << "test_gpu_downsample: (Df,Dt)=(" << Df << "," << Dt << "),"
         << " (nbeams,nfreq_dst,ntime_dst)=(" << nbeams << "," << nfreq_dst << "," << ntime_dst << "),"
         << " src (bstride,fstride)=(" << src_bstride << "," << src_fstride << "),"
         << " dst (bstride,tstride)=(" << dst_bstride << "," << dst_tstride << ")"
         << endl;

    int nfreq_src = Df * nfreq_dst;
    int ntime_src = Dt * ntime_dst;
    
    Array<float> srci_cpu({nbeams, nfreq_src, ntime_src}, af_rhost | af_random);
    Array<float> srcw_cpu = make_random_weights({nbeams, nfreq_src, ntime_src});
    
    Array<float> dsti_cpu({nbeams, ntime_dst, nfreq_dst}, af_rhost);
    Array<float> dstw_cpu({nbeams, ntime_dst, nfreq_dst}, af_rhost);

    for (auto ix = dsti_cpu.ix_start(); dsti_cpu.ix_valid(ix); dsti_cpu.ix_next(ix)) {
        int b = ix[0];
        int tds = ix[1];
        int fds = ix[2];
        
        float wisum = 0.0;
        float wsum = 0.0;
        
        for (int f = Df*fds; f < Df*(fds+1); f++) {
            for (int t = Dt*tds; t < Dt*(tds+1); t++) {
                float w = srcw_cpu.at({b,f,t});
                wisum += w * srci_cpu.at({b,f,t});
                wsum += w;
            }
        }

        dsti_cpu.at(ix) = (wsum > 0.0) ? (wisum/wsum) : 0.0;
        dstw_cpu.at(ix) = wsum;
    }
    
    Array<float> srci_gpu({nbeams, nfreq_src, ntime_src}, {src_bstride, src_fstride, 1}, af_gpu);
    Array<float> srcw_gpu({nbeams, nfreq_src, ntime_src}, {src_bstride, src_fstride, 1}, af_gpu);
    Array<float> dsti_gpu({nbeams, ntime_dst, nfreq_dst}, {dst_bstride, dst_tstride, 1}, af_gpu);
    Array<float> dstw_gpu({nbeams, ntime_dst, nfreq_dst}, {dst_bstride, dst_tstride, 1}, af_gpu);

    srci_gpu.fill(srci_cpu);
    srcw_gpu.fill(srcw_cpu);
    
    launch_downsample(dsti_gpu, dstw_gpu, srci_gpu, srcw_gpu, Df, Dt, true);
    CUDA_PEEK("launch_downsample");
    CUDA_CALL(cudaDeviceSynchronize());

    assert_arrays_equal(dsti_cpu, dsti_gpu, "cpu", "gpu", {"beam","time","freq"});
    assert_arrays_equal(dstw_cpu, dstw_gpu, "cpu", "gpu", {"beam","time","freq"});
}


void test_gpu_downsample()
{
    int Df = rand_int(1,6);
    int Dt = (rand_uniform() < 0.5) ? (1 << rand_int(0,2)) : (4 * rand_int(1,4));
    
    long nbeams = rand_int(1, 6);
    long nfreq_dst = 32 * rand_int(1, 6);
    long ntime_dst = 32 * rand_int(1, 6);
    long nfreq_src = Df * nfreq_dst;
    long ntime_src = Dt * ntime_dst;
    
    auto src_strides = make_random_strides({nbeams, nfreq_src, ntime_src}, 1, 4);  // ncontig=1, nalign=4
    auto dst_strides = make_random_strides({nbeams, ntime_dst, nfreq_dst}, 1, 4);  // ncontig=1, nalign=4
    
    int src_bstride = src_strides[0];
    int src_fstride = src_strides[1];
    int dst_bstride = dst_strides[0];
    int dst_tstride = dst_strides[1];
    
    test_gpu_downsample(Df, Dt, nbeams, nfreq_dst, ntime_dst, src_bstride, src_fstride, dst_bstride, dst_tstride);
}


}  // namespace pirate
