#include <iostream>
#include <ksgpu/Array.hpp>
#include <ksgpu/KernelTimer.hpp>

#include "../../include/pirate/loose_ends/timing.hpp"
#include "../../include/pirate/loose_ends/gpu_transpose.hpp"

using namespace std;
using namespace ksgpu;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


void time_gpu_transpose()
{
    int nx = 2048;
    int ny = 2048;
    int nz = 64;
    int ninner = 100;
    int nouter = 20;
    int nstreams = 4;
    double gmem_gb = 8. * nx*ny*nz * double(ninner) / pow(2,30.);
    
    vector<Array<float>> src(nstreams);
    vector<Array<float>> dst(nstreams);

    for (int istream = 0; istream < nstreams; istream++) {
        src[istream] = Array<float> ({nz,ny,nx}, af_gpu | af_zero);
        dst[istream] = Array<float> ({nz,nx,ny}, af_gpu | af_zero);
    }

    KernelTimer kt(nstreams);

    for (int i = 0; i < nouter; i++) {
        for (int j = 0; j < ninner; j++)
            launch_transpose(dst[kt.istream], src[kt.istream], kt.stream);

        if (kt.advance()) {
            double gb_per_sec = gmem_gb / kt.dt;
            cout << "gpu_transpose global memory (GB/s): " << gb_per_sec << endl;
        }
    }
}


}  // namespace pirate
