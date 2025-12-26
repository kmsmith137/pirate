#include <ksgpu/memcpy_kernels.hpp>
#include "../include/pirate/Dedisperser.hpp"
#include "../include/pirate/DedispersionPlan.hpp"
#include "../include/pirate/BumpAllocator.hpp"
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

    // Frequency band: CHIME [400,800] MHz
    config.zone_nfreq = { nfreq };
    config.zone_freq_edges = { 400.0, 800.0 };

    // Disable frequency subbands (search only full band)
    config.frequency_subband_counts = { 1 };

    // Peak finding params: one per downsampling level
    config.peak_finding_params.resize(config.num_downsampling_levels);
    for (int i = 0; i < config.num_downsampling_levels; i++) {
        config.peak_finding_params[i].max_width = 16;
        config.peak_finding_params[i].wt_dm_downsampling = 64;
        config.peak_finding_params[i].wt_time_downsampling = 64;
    }

    // No early triggers
    config.validate();
}


void ChimeDedisperser::initialize()
{
    if (plan)
        throw runtime_error("Double call to ChimeDedisperser::initialize()");

    long nstreams = config.num_active_batches;
    long nbatches = xdiv(config.beams_per_gpu, config.beams_per_batch);

    this->plan = make_shared<DedispersionPlan> (config);
    this->dedisperser = make_shared<GpuDedisperser> (plan);

    BumpAllocator gpu_allocator(af_gpu | af_zero, -1);     // dummy allocator
    BumpAllocator host_allocator(af_rhost | af_zero, -1);  // dummy allocator

    this->dedisperser->allocate(gpu_allocator, host_allocator);  // note: all buffers are zeroed or initialized
    this->streams = CudaStreamWrapper::create_vector(nstreams);
    
    // FIXME currently, gridding and peak-finding are not implemented.
    // As a kludge, we put in some extra GPU->GPU memcopies with the same bandwidth.

    // Total (i.e. src+dst) extra elements (not bytes) per batch.
    long extra_nelts = config.beams_per_batch * nfreq * config.time_samples_per_chunk;
    extra_nelts += dedisperser->stage1_dd_bufs[0].get_nelts();  // includes factor 'beams_per_batch'
    // extra_nelts += dedisperser->stage2_dd_bufs[0].get_nelts();  // includes factor 'beams_per_batch'

    // "One-sided" (i.e. src or dst only) extra bytes (not elements) per batch
    this->extra_nbytes_per_batch = align_up((extra_nelts * config.dtype.nbits) / (2*8), 256);
    
    // Length-2 axis is {dst,src}
    this->extra_buffers = Array<char> ({ nstreams, 2, extra_nbytes_per_batch }, af_zero | af_gpu);

    // BandwidthTracker
    this->bw_per_run_call = nbatches * dedisperser->bw_per_launch;
    this->bw_per_run_call.nbytes_gmem += 2 * nbatches * extra_nbytes_per_batch;  // factor 2 is from {dst,src}.

    if (use_copy_engine)
        this->bw_per_run_call.memcpy_g2g_calls += nbatches;
    else
        this->bw_per_run_call.kernel_launches += nbatches;
}


void ChimeDedisperser::run(long ichunk)
{
    if (!plan)
        throw runtime_error("Must call ChimeDedisperser::initialize() before ChimeDedisperser::run()");

    long nstreams = config.num_active_batches;
    long nbatches = xdiv(config.beams_per_gpu, config.beams_per_batch);
    
    for (long ibatch = 0; ibatch < nbatches; ibatch++) {
        // Call to cudaStreamSynchronize() prevents streams from getting too out-of-sync.
        // Later, I'll replace this with something better (involving cudaEvents).
        CudaStreamWrapper &s = streams.at(istream);
        CUDA_CALL(cudaStreamSynchronize(s));

        // "Extra" GPU->GPU memcopies (see above).
        char *xdst = extra_buffers.data + (2*istream) * extra_nbytes_per_batch;
        char *xsrc = extra_buffers.data + (2*istream+1) * extra_nbytes_per_batch;
        
        if (use_copy_engine)
            CUDA_CALL(cudaMemcpyAsync(xdst, xsrc, extra_nbytes_per_batch, cudaMemcpyDeviceToDevice, s));
        else
            ksgpu::launch_memcpy_kernel(xdst, xsrc, extra_nbytes_per_batch, s);
        
        dedisperser->launch(ichunk, ibatch, istream, s);
        istream = (istream + 1) % nstreams;
    }
}


}  // namespace pirate
