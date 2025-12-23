#ifndef _PIRATE_DEDISPERSER_HPP
#define _PIRATE_DEDISPERSER_HPP

#include "DedispersionConfig.hpp"
#include "DedispersionBuffer.hpp"
#include "trackers.hpp"  // BandwidthTracker

#include <vector>
#include <memory>  // shared_ptr
#include <iostream>
#include <ksgpu/Array.hpp>
#include <ksgpu/cuda_utils.hpp>  // CudaStreamWrapper

namespace pirate {
#if 0
}  // editor auto-indent
#endif


struct DedispersionPlan;              // defined in DedispersionPlan.hpp
struct GpuDedispersionKernel;         // defined in DedispersionKernel.hpp
struct GpuLaggedDownsamplingKernel;   // defined in LaggedDownsamplingKernel.hpp
struct CpuRingbufCopyKernel;          // defined in RingbufCopyKernel.hpp
struct GpuRingbufCopyKernel;          // defined in RingbufCopyKernel.hpp
struct ReferenceTreeGriddingKernel;   // defined in TreeGriddingKernel.hpp
struct GpuTreeGriddingKernel;         // defined in TreeGriddingKernel.hpp
struct ReferencePeakFindingKernel;    // defined in PeakFindingKernel.hpp


// -------------------------------------------------------------------------------------------------
//
// GpuDedisperser (defined in src_lib/GpuDedisperser.cu)
//
// Warning: not thread-safe!


struct GpuDedisperser
{
    GpuDedisperser(const std::shared_ptr<DedispersionPlan> &plan);
    
    std::shared_ptr<DedispersionPlan> plan;
    
    const DedispersionConfig config;   // same as plan->config

    ksgpu::Dtype dtype;           // = (config.dtype)
    long nfreq = 0;               // = (config.get_total_nfreq())
    long input_rank = 0;          // = (config.tree_rank)
    long input_ntime = 0;         // = (config.time_samples_per_chunk)
    long total_beams = 0;         // = (config.beams_per_gpu)
    long beams_per_batch = 0;     // = (config.beams_per_batch)
    long nstreams = 0;            // = (config.num_active_batches)
    long nbatches = 0;            // = (total_beams / beams_per_batch)
    long gpu_ringbuf_nelts = 0;   // = (mega_ringbuf->gpu_global_nseg * plan->nelts_per_segment)
    long host_ringbuf_nelts = 0;  // = (mega_ringbuf->host_global_nseg * plan->nelts_per_segment)

    long output_ntrees = 0;
    std::vector<long> output_rank;      // length output_ntrees
    std::vector<long> output_ntime;     // length output_ntrees, equal to (input_time / pow2(output_ds_level[:]))
    std::vector<long> output_ds_level;  // length output_ntrees

    // "outer" vector has length nstreams
    // "inner" array has shape (beams_per_batch, nfreq, ntime), uses config.dtype
    std::vector<ksgpu::Array<void>> input_arrays;
    
    std::vector<DedispersionBuffer> stage1_dd_bufs;  // length nstreams
    std::vector<DedispersionBuffer> stage2_dd_bufs;  // length nstreams
    
    ksgpu::Array<void> gpu_ringbuf;
    ksgpu::Array<void> host_ringbuf;

    std::shared_ptr<GpuTreeGriddingKernel> tree_gridding_kernel;
    std::vector<std::shared_ptr<GpuDedispersionKernel>> stage1_dd_kernels;
    std::vector<std::shared_ptr<GpuDedispersionKernel>> stage2_dd_kernels;
    std::shared_ptr<GpuLaggedDownsamplingKernel> lds_kernel;
    std::shared_ptr<GpuRingbufCopyKernel> g2g_copy_kernel;
    std::shared_ptr<CpuRingbufCopyKernel> h2h_copy_kernel;

    bool is_allocated = false;

    // Bandwidth per call to GpuDedisperser::launch().
    // To get bandwidth per time chunk, multiply by 'nbatches'.
    BandwidthTracker bw_per_launch;

    // Note: allocate() initializes or zeroes all arrays (i.e. no array is left uninitialized)
    void allocate();

    // launch() interface needs some explanation:
    //
    //  - Caller is responsible for creating/managing (nstreams) cuda streams,
    //    and ensuring that the (istream, stream) arguments are always consistent.
    //
    //  - Before calling launch(), caller must queue kernels to 'stream' which
    //    populate the input buffer (stage1_dd_kernels[istream].bufs[0]).
    //
    //  - launch() returns asynchronously. When 'stream' is synchronized,
    //    the output buffers (stage2_dd_kernels[istream].bufs[:]) will be populated.
    //
    // Reminder: a "chunk" is a range of time indices, and a "batch" is a range of beam indices.
    //
    // FIXME interface will evolve over time (e.g. cudaEvents).

    void launch(long ichunk, long ibatch, long istream, cudaStream_t stream);

    // Static member function: runs one randomized test iteration.
    static void test();

    // Static member function: runs one test iteration with specified configuration.
    static void test_one(const DedispersionConfig &config, int nchunks, bool host_only=false);
};


// -------------------------------------------------------------------------------------------------
//
// ReferenceDedisperser (defined in src_lib/ReferenceDedisperser.cu)
//
// Sophistication == 0:
//
//   - Uses one-stage dedispersion instead of two stages.
//   - In downsampled trees, compute twice as many DMs as necessary, then drop the bottom half.
//   - Each early trigger is computed in an independent tree, by disregarding some input channels.
//
// Sophistication == 1:
//
//   - Uses same two-stage tree/lag structure as plan.
//   - Lags are applied with a per-tree ReferenceLagbuf, rather than using ring/staging buffers.
//   - Lags are split into segments + residuals, but not further split into chunks.
//
// Sophistication == 2:
//
//   - As close to GPU implementation as possible!
//
// Note on Dcore: in order for the ReferenceDedisperser to perfectly mimic the GPU kernel, we
// need a constructor argument 'Dcore' which contains Dcore values from the GpuPeakFindingKernels.
// See PeakFindingKernel.hpp for the meaning of Dcore, and GpuDedisperser.cu for example code
// to initialize the Dcore vector.

struct ReferenceDedisperserBase
{
    // Constructor not intended to be called directly -- use make() below.
    ReferenceDedisperserBase(
        const std::shared_ptr<DedispersionPlan> &plan, 
        const std::vector<long> &Dcore,
        int sophistication
    );
    
    std::shared_ptr<DedispersionPlan> plan;
    const DedispersionConfig config;     // same as plan->config
    const std::vector<long> Dcore;       // see above
    const int sophistication;            // see above
    
    long nfreq = 0;               // same as config.get_total_nfreq()
    long input_rank = 0;          // same as config.tree_rank
    long input_ntime = 0;         // same as config.time_samples_per_chunk
    long total_beams = 0;         // same as config.beams_per_gpu
    long beams_per_batch = 0;     // same as config.beams_per_batch
    long nbatches = 0;            // = (total_beams / beams_per_batch)

    long output_ntrees = 0;             // same as plan->stage2_ntrees
    std::vector<long> output_rank;      // length output_ntrees
    std::vector<long> output_ntime;     // length output_ntrees, equal to (input_time / pow2(output_ds_level[:]))
    std::vector<long> output_ds_level;  // length output_ntrees

    std::shared_ptr<ReferenceTreeGriddingKernel> tree_gridding_kernel;

    // To process multiple chunks, call the dedisperse() method in a loop.
    // Reminder: a "chunk" is a range of time indices, and a "batch" is a range of beam indices.
    virtual void dedisperse(long ichunk, long ibatch) = 0;

    // Retrieves the peak-finding kernel for the i-th tree.
    // Implemented for sophistication=1 or 2, throws an exception if sophistication=0.
    // (This is because the sophistication=0 kernel uses different tree sizes.)
    // This is a little awkward, but I think it's the least awkward approach.
    virtual std::shared_ptr<ReferencePeakFindingKernel> get_pf_kernel(long itree) = 0;

    // Before calling dedisperse(), caller should fill 'input_array'.
    // Shape is (beams_per_batch, nfreq, input_ntime).
    ksgpu::Array<float> input_array;

    // Befre calling dedisperse(), caller should fill 'wt_arrays' (peak-finding weights).
    // Shape is (beams_per_batch, ndm_wt, nt_wt, nprofiles, nsubbands)
    //   where (ndm_wt, nt_wt) can be found in plan->stage2_pf_params[itree].
    //   and (nprofiles, nsubbands) can be found in plan->stage2_trees[itree].
    std::vector<ksgpu::Array<float>> wt_arrays;    // length output_ntrees

    // After dedisperse() completes, dedispersion output is stored in 'output_arrays'.
    // output_arrays[i] has shape (beams_per_batch, 2^output_rank[i], output_ntime[i])
    std::vector<ksgpu::Array<float>> output_arrays;   // length output_ntrees
   
    // After dedisperse() completes, peak-finding output is stored in 'out_max', 'out_argmax'.
    // Shape is (beams_per_batch, ndm_out, nt_out)
    //   where (ndm_out, nt_out) can be found in plan->stage2_pf_params[itree]
    std::vector<ksgpu::Array<float>> out_max;     // length output_ntrees
    std::vector<ksgpu::Array<uint>> out_argmax;   // length output_ntrees

    // Factory function -- constructs ReferenceDedisperser of specified sophistication.
    static std::shared_ptr<ReferenceDedisperserBase> make(
        const std::shared_ptr<DedispersionPlan> &plan,
        const std::vector<long> &Dcore,
        int sophistication
    );

    // Helper methods called by subclass constructors.
    void _init_output_arrays(std::vector<ksgpu::Array<float>> &out);
    void _init_output_arrays(std::vector<ksgpu::Array<void>> &out);
};


// -------------------------------------------------------------------------------------------------
//
// ChimeDedisperser: a temporary hack for timing.
//
// Warning: not thread-safe!


struct ChimeDedisperser
{
    // I did some performance tuning, and found that the following values worked well:
    //
    //   config.beams_per_batch = 1
    //   config.num_active_batches = 3
    //
    // This may change when RFI removal is incorporated (we may want to increase
    // beams_per_batch, in order to reduce the number of kernel launches).

    ChimeDedisperser(int beams_per_gpu=12, int num_active_batches=3, int beams_per_batch=1, bool use_copy_engine=false);
    
    // Call with current GPU appropriately set.
    void initialize();
    
    // Dedisperses a data "cube" with shape (config.beams_per_gpu, nfreq, config.time_samples_per_chunk)
    // Note 'beams_per_gpu' here, not 'beams_per_batch'!
    void run(long ichunk);
    
    bool use_copy_engine = false;
    int nfreq = 16384;
    int istream = 0;
    
    DedispersionConfig config;
    std::shared_ptr<DedispersionPlan> plan;
    std::shared_ptr<GpuDedisperser> dedisperser;
    std::vector<ksgpu::CudaStreamWrapper> streams;

    // FIXME currently, gridding and peak-finding are not implemented.
    // As a kludge, we put in some extra GPU->GPU memcopies with the same bandwidth.
    ksgpu::Array<char> extra_buffers;  // shape (num_active_batches, 2, extra_nbytes_per_batch)
    long extra_nbytes_per_batch = 0;   // FIXME doesn't get initialized until initialize() is called.

    // Bandwidth per call to ChimeDedisperser::run().
    // This corresponds to 'beams_per_gpu' beams, not 'beams_per_batch' beams.
    // FIXME: doesn't currently get initialized until initialize() is called.
    BandwidthTracker bw_per_run_call;
};


}  // namespace pirate

#endif // _PIRATE_DEDISPERSER_HPP
