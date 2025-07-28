#include "../include/pirate/PeakFindingKernel.hpp"
#include "../include/pirate/inlines.hpp"  // xdiv()

#include <ksgpu/cuda_utils.hpp>
#include <ksgpu/time_utils.hpp>

using namespace std;
using namespace ksgpu;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


static void time_gpu_peak_finding_kernel(const PeakFindingKernelParams &params, long nouter, long ninner, long nstreams)
{
    cout << "\ntime_pf_kernel: start\n"
	 << "   dtype = " << params.dtype.str() << "\n"
	 << "   M = dm_downsampling_factor = " << params.dm_downsampling_factor << "\n"
	 << "   E = max_kernel_width = " << params.max_kernel_width << "\n"
	 << "   Dout = time_downsampling_factor = " << params.time_downsampling_factor << "\n"
	 << "   beams_per_batch = " << params.beams_per_batch << "\n"
	 << "   ndm_in = " << params.ndm_in << "\n"
	 << "   nt_in = " << params.nt_in << "\n"
	 << endl;

    GpuPeakFindingKernel gpu_kernel(params);
    gpu_kernel.allocate();

    long S = nstreams;
    long B = params.beams_per_batch;   // total_beams is ignored
    long P = gpu_kernel.nprofiles;
    long Min = params.ndm_in;
    long Tin = params.nt_in;
    long Mout = xdiv(Min, params.dm_downsampling_factor);
    long Tout = xdiv(Tin, params.time_downsampling_factor);
    double gb = 1.0e-9 * ninner * gpu_kernel.bw_per_launch.nbytes_gmem;
    
    Array<float> wt({S,B,P,Min}, af_gpu | af_zero);
    Array<float> in({S,B,Min,Tin}, af_gpu | af_zero);
    Array<float> out_max({S,B,P,Mout,Tout}, af_gpu | af_zero);
    Array<float> out_ssq({S,B,P,Mout,Tout}, af_gpu | af_zero);

    xassert(nouter >= nstreams+1);
    vector<ksgpu::CudaStreamWrapper> streams(S);  // creates S new streams
    vector<struct timeval> tv(nouter);

    for (int i = 0; i < nouter; i++) {
	int s = i % S;
	cudaStreamSynchronize(streams[s]);
	tv[i] = get_time();

	for (int j = 0; j < ninner; j++) {
	    gpu_kernel.launch(
	        out_max.slice(0,s),
		out_ssq.slice(0,s),
		in.slice(0,s),
		wt.slice(0,s),
		0, streams[s]
	    );
	}
	
	int k = i - (i/(2*S))*S;
	if (i > k)
	    cout << "        " << ((i-k) * gb / time_diff(tv[k],tv[i])) << " GB/s\n";
    }
}


static void time_gpu_peak_finding_kernel(int E, int M, int Dout)
{
    long nstreams = 2;
    long nouter = 30;
    long ninner = 50;
    
    PeakFindingKernelParams p;
    p.dtype = Dtype::native<float> ();
    p.dm_downsampling_factor = M;
    p.time_downsampling_factor = Dout;
    p.max_kernel_width = E;
    p.beams_per_batch = 1;
    p.total_beams = p.beams_per_batch;
    p.ndm_in = (32768 / M) * M;
    p.nt_in = 2048;

    time_gpu_peak_finding_kernel(p, nouter, ninner, nstreams);
}


void time_gpu_peak_finding_kernels()
{
    // I may add more kernels later.
    // FIXME Increasing M from 16 -> 50 makes the kernel run a little slower, I wonder why?
    
    time_gpu_peak_finding_kernel(32, 16, 16);
    time_gpu_peak_finding_kernel(32, 50, 16);
}    


}  // namespace pirate
