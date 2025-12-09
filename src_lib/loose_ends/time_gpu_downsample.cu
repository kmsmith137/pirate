#include <iostream>
#include <ksgpu/Array.hpp>
#include <ksgpu/KernelTimer.hpp>
#include <ksgpu/string_utils.hpp>
#include <ksgpu/xassert.hpp>

#include "../../include/pirate/loose_ends/timing.hpp"
#include "../../include/pirate/loose_ends/gpu_downsample.hpp"

using namespace std;
using namespace ksgpu;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


static void time_one_gpu_downsample(int Df, int Dt)
{
    bool transpose_output = true;  // FIXME
    
    xassert((Df >= 1) && ((2048 % Df) == 0));
    xassert((Dt >= 1) && ((2048 % Dt) == 0));
    
    long nbeams = 64;
    long nfreq_src = 2048;
    long ntime_src = 2048;
    long ninner = 100;
    long nouter = 20;
    long nstreams = 2;
    
    long nfreq_dst = nfreq_src / Df;
    long ntime_dst = ntime_src / Dt;
    double gmem_gb = 8. * nbeams * ninner * double(nfreq_src * ntime_src + nfreq_dst * ntime_dst) / pow(2,30.);
    
    Array<float> src_si({nstreams, nbeams, nfreq_src, ntime_src}, af_gpu | af_zero);
    Array<float> src_sw({nstreams, nbeams, nfreq_src, ntime_src}, af_gpu | af_zero);
    Array<float> dst_si({nstreams, nbeams, ntime_dst, nfreq_dst}, af_gpu | af_zero);
    Array<float> dst_sw({nstreams, nbeams, ntime_dst, nfreq_dst}, af_gpu | af_zero);

    stringstream sp_name;
    sp_name << "gpu_downsample(Df=" << Df << ",Dt=" << Dt << ",transpose_output=" << transpose_output << ")";
    string name = sp_name.str();

    KernelTimer kt(nouter, nstreams);

    while (kt.next()) {
        Array<float> src_i = src_si.slice(0, kt.istream);
        Array<float> src_w = src_sw.slice(0, kt.istream);
        Array<float> dst_i = dst_si.slice(0, kt.istream);
        Array<float> dst_w = dst_sw.slice(0, kt.istream);
        
        for (int j = 0; j < ninner; j++)
            launch_downsample(dst_i, dst_w, src_i, src_w, Df, Dt, transpose_output, kt.stream);

        if (kt.warmed_up) {
            double gb_per_sec = gmem_gb / kt.dt;
            cout << name << " global memory (GB/s): " << gb_per_sec << endl;
        }
    }
}


void time_gpu_downsample()
{
    for (int Df: {1,4,16})
        for (int Dt: {1,4,16})
            time_one_gpu_downsample(Df,Dt);
}


}  // namespace pirate
