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
#if 0
    
    // I did some performance tuning, and found that the following values worked well:
    //
    //   config.beams_per_batch = 1
    //   config.num_active_batches = 3
    //
    // This may change when RFI removal is incorporated (we may want to increase
    // beams_per_batch, in order to reduce the number of kernel launches).
    
    DedispersionConfig config;
    config.tree_rank = 15;
    config.num_downsampling_levels = 4;      // max DM 13K
    config.time_samples_per_chunk = 2048;    // dedisperse in 2-second chunks
    config.dtype = Dtype::native<__half>();  // float16
    config.beams_per_gpu = 12;  // ?
    config.beams_per_batch = 1;
    config.num_active_batches = 3;
    // No early triggers

    long nchunks = 500;
    long nstreams = config.num_active_batches;
    long nbatches = xdiv(config.beams_per_gpu, config.beams_per_batch);
    long nfreq = 16384;  // used in "extra" copies, see below.

    cout << "time-dedispersion: currently hardcoding a simplified CHIME-like setup" << endl;
    config.print(cout, 4);
    config.validate();

    shared_ptr<DedispersionPlan> plan = make_shared<DedispersionPlan> (config);
    GpuDedisperser dedisperser(plan);
    dedisperser.allocate();  // note: all buffers are initialized or zeroed

    vector<ksgpu::CudaStreamWrapper> streams;
    for (long i = 0; i < nstreams; i++)
	streams.push_back(ksgpu::CudaStreamWrapper());

    // FIXME currently, gridding and peak-finding are not implemented.
    // As a kludge, we put in some extra GPU->GPU memcopies with the same bandwidth.

    // Total (i.e. src+dst) extra elements (not bytes) per batch.
    long extra_nelts = config.beams_per_batch * nfreq * config.time_samples_per_chunk;
    extra_nelts += dedisperser.stage1_dd_bufs[0].get_nelts();  // includes factor 'beams_per_batch'
    extra_nelts += dedisperser.stage2_dd_bufs[0].get_nelts();  // includes factor 'beams_per_batch'

    // "One-sided" (i.e. src or dst only) extra bytes (not elements) per batch
    long extra_nbytes = align_up((extra_nelts * config.dtype.nbits) / (2*8), 256);
    
    // Length-2 axis is {dst,src}
    Array<char> extra_buffers({ nstreams, 2, extra_nbytes }, af_zero | af_gpu);

    vector<struct timeval> timestamps(nchunks+1);
    timestamps[0] = ksgpu::get_time();
	
    long istream = 0;
    for (long it_chunk = 0; it_chunk < nchunks; it_chunk++) {
	for (long ibatch = 0; ibatch < nbatches; ibatch++) {
	    
	    // Call to cudaStreamSynchronize() prevents streams from getting too out-of-sync.
	    // Later, I'll replace this with something better (involving cudaEvents).
	    
	    CudaStreamWrapper &s = streams.at(istream);
	    CUDA_CALL(cudaStreamSynchronize(s));

	    // "Extra" GPU->GPU memcopies (see above).
	    char *xdst = extra_buffers.data + (2*istream) * extra_nbytes;
	    char *xsrc = extra_buffers.data + (2*istream+1) * extra_nbytes;
	    CUDA_CALL(cudaMemcpyAsync(xdst, xsrc, extra_nbytes, cudaMemcpyDeviceToDevice, s));
	    
	    dedisperser.launch(ibatch, it_chunk, istream, s);
	    istream = (istream + 1) % nstreams;
	}

	long ihi = it_chunk+1;
	long ilo = (ihi/2);
	timestamps[ihi] = ksgpu::get_time();

	// lbs = "Logical beam-seconds" of data.
	double lbs = (ihi - ilo) * (config.beams_per_gpu) * (1.0e-3 * config.time_samples_per_chunk);
	double dt_sec = ksgpu::time_diff(timestamps[ilo], timestamps[ihi]);
	double real_time_beams = lbs / dt_sec;
	
	cout << "After " << (it_chunk+1)
	     << " chunks: real-time CHIME beams = "
	     << real_time_beams << " (per-GPU, not per-node)\n";
    }

#else

    long nchunks = 500;
    long beams_per_gpu = 12;
    long beams_per_batch = 1;
    long num_active_batches = 3;
    bool use_copy_engine = false;

    ChimeDedisperser dd(beams_per_gpu, num_active_batches, beams_per_batch, use_copy_engine);
    dd.initialize();

    vector<struct timeval> timestamps(nchunks+1);
    timestamps[0] = ksgpu::get_time();
    
    for (long it_chunk = 0; it_chunk < nchunks; it_chunk++) {
	dd.run(1);
	
	long ihi = it_chunk+1;
	long ilo = (ihi/2);
	timestamps[ihi] = ksgpu::get_time();

	// lbs = "Logical beam-seconds" of data.
	double lbs = (ihi - ilo) * (config.beams_per_gpu) * (1.0e-3 * config.time_samples_per_chunk);
	double dt_sec = ksgpu::time_diff(timestamps[ilo], timestamps[ihi]);
	double real_time_beams = lbs / dt_sec;
	
	cout << "After " << (it_chunk+1)
	     << " chunks: real-time CHIME beams = "
	     << real_time_beams << " (per-GPU, not per-node)\n";
    }

#endif

    return 0;
}

