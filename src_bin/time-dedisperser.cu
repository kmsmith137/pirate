#include "../include/pirate/DedispersionPlan.hpp"
#include "../include/pirate/DedispersionConfig.hpp"
#include "../include/pirate/Dedisperser.hpp"
#include "../include/pirate/inlines.hpp"  // xdiv()

#include <ksgpu/cuda_utils.hpp>  // CudaStreamWrapper, CUDA_CALL()
#include <ksgpu/time_utils.hpp>  // get_time(), time_diff()

using namespace std;
using namespace ksgpu;
using namespace pirate;


int main(int argc, char **argv)
{
    long nchunks = 500;
    long beams_per_gpu = 12;
    long beams_per_batch = 1;
    long num_active_batches = 3;
    bool use_copy_engine = true;

    ChimeDedisperser dd(beams_per_gpu, num_active_batches, beams_per_batch, use_copy_engine);
    dd.initialize();

    vector<struct timeval> timestamps(nchunks+1);
    timestamps[0] = ksgpu::get_time();
    
    for (long it_chunk = 0; it_chunk < nchunks; it_chunk++) {
	dd.run(it_chunk);
	
	long ihi = it_chunk+1;
	long ilo = (ihi/2);
	timestamps[ihi] = ksgpu::get_time();

	// lbs = "Logical beam-seconds" of data.
	double lbs = (ihi - ilo) * (beams_per_gpu) * (1.0e-3 * dd.config.time_samples_per_chunk);
	double dt_sec = ksgpu::time_diff(timestamps[ilo], timestamps[ihi]);
	double real_time_beams = lbs / dt_sec;
	
	cout << "After " << (it_chunk+1)
	     << " chunks: real-time CHIME beams = "
	     << real_time_beams << " (per-GPU, not per-node)\n";
    }

    return 0;
}

