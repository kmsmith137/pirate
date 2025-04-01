#include "../include/pirate/Dedisperser.hpp"
#include "../include/pirate/DedispersionPlan.hpp"
#include "../include/pirate/inlines.hpp"  // xdiv(), pow2()

using namespace std;
using namespace ksgpu;

namespace pirate {
#if 0
}  // editor auto-indent
#endif

//
// ChimeDedisperser: a temporary hack for timing.
//


ChimeDedisperser::ChimeDedisperser(int beams_per_gpu_, int num_active_batches_, int beams_per_batch_, bool use_copy_engine_)
{
    this->use_copy_engine = use_copy_engine_;
    
    config.tree_rank = 15;
    config.num_downsampling_levels = 4;      // max DM 13K
    config.time_samples_per_chunk = 2048;    // dedisperse in 2-second chunks
    config.dtype = Dtype::native<__half>();  // float16
    config.beams_per_gpu = beams_per_gpu_;
    config.beams_per_batch = beams_per_batch_;
    config.num_active_batches = num_active_batches_;
    // No early triggers
    
    config.validate();
}


void ChimeDedisperser::initialize()
{
    if (plan)
	raise runtime_error("Double call to ChimeDedisperser::initialize()");

    long nstreams = config.num_active_batches;

    this->plan = make_shared<DedispersionPlan> (config);
    this->dedisperser = make_shared<GpuDedisperser> (plan);
    this->dedisperser->allocate();  // note: all buffers are zeroed or initialized
    
    for (long i = 0; i < nstreams; i++)
	this->streams.push_back(ksgpu::CudaStreamWrapper());

    // FIXME currently, gridding and peak-finding are not implemented.
    // As a kludge, we put in some extra GPU->GPU memcopies with the same bandwidth.

    // Total (i.e. src+dst) extra elements (not bytes) per batch.
    long extra_nelts = config.beams_per_batch * nfreq * config.time_samples_per_chunk;
    extra_nelts += dedisperser->stage1_dd_bufs[0].get_nelts();  // includes factor 'beams_per_batch'
    extra_nelts += dedisperser->stage2_dd_bufs[0].get_nelts();  // includes factor 'beams_per_batch'

    // "One-sided" (i.e. src or dst only) extra bytes (not elements) per batch
    long extra_nbytes = align_up((extra_nelts * config.dtype.nbits) / (2*8), 256);
    
    // Length-2 axis is {dst,src}
    this->extra_bufffers = Array<char> extra_buffers({ nstreams, 2, extra_nbytes }, af_zero | af_gpu);
}


void ChimeDedisperser::run(long nchunks)
{
    if (!plan)
	raise runtime_error("Must call ChimeDedisperser::initialize() before ChimeDedisperser::run()");
    
    long nstreams = config.num_active_batches;
    long nbatches = xdiv(config.beams_per_gpu, config.beams_per_batch);
    long extra_nbytes = extra_buffers.shape[2];    
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

	    if (use_copy_engine)
		CUDA_CALL(cudaMemcpyAsync(xdst, xsrc, extra_nbytes, cudaMemcpyDeviceToDevice, s));
	    else
		ksgpu::launch_memcpy_kernel(xdst, xsrc, extra_nbytes, s);
	    
	    dedisperser.launch(ibatch, it_chunk, istream, s);
	    istream = (istream + 1) % nstreams;
	}
    }
}


}  // namespace pirate
