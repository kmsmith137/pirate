#ifndef _PIRATE_DEDISPERSER_HPP
#define _PIRATE_DEDISPERSER_HPP

#include "DedispersionConfig.hpp"
#include "DedispersionBuffer.hpp"
#include "DedispersionTree.hpp"
#include "ResourceTracker.hpp"

#include <mutex>
#include <thread>
#include <vector>
#include <memory>
#include <iostream>
#include <condition_variable>

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
struct CoalescedDdKernel2;            // defined in CoalescedDdKernel2.hpp
struct CudaEventRingbuf;              // defined in CudaEventRingbuf.hpp
struct CudaStreamPool;                // defined in CudaStreamPool.hpp
struct BumpAllocator;                 // defined in BumpAllocator.hpp
struct MegaRingbuf;                   // defined in MegaRingbuf.hpp


// -------------------------------------------------------------------------------------------------
//
// GpuDedisperser (defined in src_lib/GpuDedisperser.cu)


struct GpuDedisperser
{
    struct Params 
    {
        std::shared_ptr<DedispersionPlan> plan;
        std::shared_ptr<CudaStreamPool> stream_pool;

        // Set sizes of dedispersion output and peak-finding weights buffers.
        // If uninitialized, default is plan->num_active_batches.
        long nbatches_out = 0;
        long nbatches_wt = 0;    // infrequently used (only in a unit test)
        
        // detect_deadlocks=true: assumes that {acquire,release}_input() is called
        // on the same thread as {acquire,release}_output(), and detect deadlocks
        // accordingly.
        //
        // detect_deadlocks=false: assumes that {acquire,release}_input() is called
        // on a different thread as {acquire,release}_output(). In this case, the
        // deadlock-checking logic is disabled.

        bool detect_deadlocks = true;
    };

    GpuDedisperser(const Params &params);
    
    Params params;

    DedispersionConfig config;
    std::shared_ptr<DedispersionPlan> plan;
    std::shared_ptr<MegaRingbuf> mega_ringbuf;
    std::shared_ptr<CudaStreamPool> stream_pool;
 
    // Some key members of DedispersionPlan, copied into GpuDedisperser for convenience.
    ksgpu::Dtype dtype;                    // same as plan->dtype
    long nfreq = 0;                        // same as plan->freq
    long nt_in = 0;                        // same as plan->nt_in
    long total_beams = 0;                  // same as plan->beams_per_gpu
    long beams_per_batch = 0;              // same as plan->beams_per_batch
    long nstreams = 0;                     // same as plan->num_active_batches
    long nbatches = 0;                     // = (total_beams / beams_per_batch)
    long ntrees = 0;                       // same as plan->ntrees
    std::vector<DedispersionTree> trees;   // same as plan->trees

    // --------------------------  High-level API ---------------------------
    //
    // acquire_input): after call, 'stream' sees empty input buffer.
    // release_input(): before call, 'stream' must see full input buffer.
    // acquire_output(): after call, 'stream' sees full output buffer.
    // release_output(): before call, 'stream' must see empty output buffer.
    //
    // FIXME: I may rethink this API later. (Do I want acquire_weights()?
    // What should be the return type of acquire_output()?)
    //
    // FIXME: currently there's no explicit API protecting the pf_weights.
    // The caller is responsible for thinking through race conditions involving
    // the pf_weights (e.g. in GpuDedisperser::test_one(), where the weights
    // are updated during the unit test). This issue may go away on its own,
    // since my tentative long-term plan is to generate the pf_weights
    // "dynamically" as part of the compute graph.

    void allocate(BumpAllocator &gpu_allocator, BumpAllocator &host_allocator);

    ksgpu::Array<void> acquire_input(long ichunk, long ibatch, cudaStream_t stream);
    void release_input(long ichunk, long ibatch, cudaStream_t stream);

    void acquire_output(long ichunk, long ibatch, cudaStream_t stream);
    void release_output(long ichunk, long ibatch, cudaStream_t stream);

    // Thread-backed class pattern: stop the worker thread and put the object
    // into a "stopped" state. The first caller sets 'error'. If e is null,
    // represents normal termination; if non-null, represents an error.
    // Entry points called after stop() will rethrow the stored exception.
    void stop(std::exception_ptr e = std::exception_ptr());

    ~GpuDedisperser();

    // Static member function: runs one randomized test iteration.
    static void test_random();

    // Static member function: do one test/timing run with specified configuration.
    static void test_one(const DedispersionConfig &config, int nchunks, bool host_only=false);
    static void time_one(const DedispersionConfig &config, long niterations, bool use_hugepages);

    
    // --------------------------  Public members  --------------------------

    // "inner" array shape (nstreams, beams_per_batch, nfreq, ntime).
    ksgpu::Array<void> input_arrays;       // config.dtype

    // "outer" vector has length ntrees
    // "inner" array shape = this->extended_wt_shapes[itree], see below
    std::vector<ksgpu::Array<void>> wt_arrays;

    // "outer" vector has length ntrees
    // "inner" array shape (nbatches_out, beams_per_batch, t.ndm_out, t.nt_out)
    //    where t= plan->trees.at(itree)

    std::vector<ksgpu::Array<void>> out_max;     // config.dtype
    std::vector<ksgpu::Array<uint>> out_argmax;  // uint dtype

    bool is_allocated = false;
    ResourceTracker resource_tracker;  // all rates are "per batch"
    
    // -----------------------------  Internals  -----------------------------

    std::vector<DedispersionBuffer> stage1_dd_bufs;  // length nstreams
    
    long gpu_ringbuf_nelts = 0;
    long host_ringbuf_nelts = 0;
    ksgpu::Array<void> gpu_ringbuf;
    ksgpu::Array<void> host_ringbuf;

    std::shared_ptr<GpuTreeGriddingKernel> tree_gridding_kernel;
    std::vector<std::shared_ptr<GpuDedispersionKernel>> stage1_dd_kernels;
    std::vector<std::shared_ptr<CoalescedDdKernel2>> cdd2_kernels;
    std::shared_ptr<GpuLaggedDownsamplingKernel> lds_kernel;
    std::shared_ptr<GpuRingbufCopyKernel> g2g_copy_kernel;
    std::shared_ptr<CpuRingbufCopyKernel> h2h_copy_kernel;

    // Peak-finding weights use a complicated GPU memory layout.
    // The helper class 'GpuPfWeightLayout' is intended to hide the details.
    // We have one GpuPfWeightLayout per output tree: cdd2_kernels[itree]->pf_weight_layout.
    // The shape for one beam batch is here: cdd2_kernels[itree]->expected_wt_shape.
    // The "extended" shapes below adds an outer index with length (nbatches_wt).

    std::vector<std::vector<long>> extended_wt_shapes;   // length ntrees
    std::vector<std::vector<long>> extended_wt_strides;  // length ntrees
    
    // Helpers for individual compute steps. 
    // (These are called once each, so are logically unncecessary, but I like the separation
    // between "compute logic" and "synchronization logic".)

    void _launch_tree_gridding(long ichunk, long ibatch, cudaStream_t stream);
    void _launch_lagged_downsampler(long ichunk, long ibatch, cudaStream_t stream);
    void _launch_dd_stage1(long ichunk, long ibatch, cudaStream_t stream);
    void _launch_et_g2g(long ichunk, long ibatch, cudaStream_t stream);
    void _launch_g2h(long ichunk, long ibatch, cudaStream_t stream);
    void _launch_h2g(long ichunk, long ibatch, cudaStream_t stream);
    void _launch_cdd2(long ichunk, long ibatch, cudaStream_t stream);

    // These helpers will run from the et_h2h thread, not the main thread.
    void _do_et_h2h(long ichunk, long ibatch);     // CPU kernel, not GPU kernel
    void _launch_et_h2g(long ichuink, long ibatch, cudaStream_t stream);

    // These methods contain the difficult code :)
    void _launch_dedispersion_kernels(long ichunk, long ibatch, cudaStream_t stream);
    void _worker_main();
    void worker_main();

    // Thread-backed class pattern: helpers.
    void _throw_if_stopped(const char *method_name);

    // The CudaEventRingbufs keep track of lagged dependencies between kernels.
    std::shared_ptr<CudaEventRingbuf> evrb_tree_gridding;
    std::shared_ptr<CudaEventRingbuf> evrb_g2g;
    std::shared_ptr<CudaEventRingbuf> evrb_g2h;
    std::shared_ptr<CudaEventRingbuf> evrb_h2g;
    std::shared_ptr<CudaEventRingbuf> evrb_cdd2;
    std::shared_ptr<CudaEventRingbuf> evrb_et_h2g;
    std::shared_ptr<CudaEventRingbuf> evrb_output;

    // These members help keep track of lags between kernels.
    long host_seq_lag = 0;     // has_host_ringbuf ? (mega_ringbuf->min_host_clag * nbatches) : (2*nstreams)
    long et_seq_headroom = 0;  // has_early_triggers ? (mega_ringbuf->min_et_headroom * nbatches) : (2*nstreams)
    long et_seq_lag = 0;       // has_early_triggers ? (mega_ringbuf->min_et_clag * nbatches) : (2*nstreams)

    std::mutex mutex;

    long curr_input_ichunk = 0;
    long curr_input_ibatch = 0;
    bool curr_input_acquired = false;

    long curr_output_ichunk = 0;
    long curr_output_ibatch = 0;
    bool curr_output_acquired = false;

    // Thread-backed class pattern: worker thread and stopped state.
    // Note: no condition_variable needed to wake up worker thread, since the worker does
    // all of its waiting on condition_variables which are members of CudaEventRingbuf.
    std::thread worker;
    bool is_stopped = false;
    std::exception_ptr error;
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
    std::vector<long> Dcore;       // see above
    int sophistication;            // see above

    // Some key members of DedispersionPlan, copied into ReferenceD
    DedispersionConfig config;             // same as plan->config
    ksgpu::Dtype dtype;                    // same as plan->dtype
    long nfreq = 0;                        // same as plan->nfreq
    long nt_in = 0;                        // same as plan->nt_in
    long total_beams = 0;                  // same as plan->beams_per_gpu
    long beams_per_batch = 0;              // same as plan->beams_per_batch
    long num_downsampling_levels = 0;      // same as plan->num_downsampling_levels
    long nbatches = 0;                     // = (total_beams / beams_per_batch)
    long ntrees = 0;                       // same as plan->ntrees
    std::vector<DedispersionTree> trees;   // same as plan->trees

    std::shared_ptr<ReferenceTreeGriddingKernel> tree_gridding_kernel;
    std::vector<std::shared_ptr<ReferencePeakFindingKernel>> pf_kernels;  // length ntrees

    // To process multiple chunks, call the dedisperse() method in a loop.
    // Reminder: a "chunk" is a range of time indices, and a "batch" is a range of beam indices.
    virtual void dedisperse(long ichunk, long ibatch) = 0;

    // Before calling dedisperse(), caller should fill 'input_array'.
    // Shape is (beams_per_batch, nfreq, nt_in).
    ksgpu::Array<float> input_array;

    // Befre calling dedisperse(), caller should fill 'wt_arrays' (peak-finding weights).
    // Shape is (beams_per_batch, t.ndm_wt, t.nt_wt, t.nprofiles, t.frequency_subbands.F)
    //   where t = plan->trees.at(itree).
    std::vector<ksgpu::Array<float>> wt_arrays;    // length ntrees

    // After dedisperse() completes, peak-finding output is stored in 'out_max', 'out_argmax'.
    // Shape is (beams_per_batch, t.ndm_out, t.nt_out)
    //   where t = plan->trees.at(itree)
    std::vector<ksgpu::Array<float>> out_max;     // length ntrees
    std::vector<ksgpu::Array<uint>> out_argmax;   // length ntrees

    // Factory function -- constructs ReferenceDedisperser of specified sophistication.
    static std::shared_ptr<ReferenceDedisperserBase> make(
        const std::shared_ptr<DedispersionPlan> &plan,
        const std::vector<long> &Dcore,
        int sophistication
    );
};


}  // namespace pirate

#endif // _PIRATE_DEDISPERSER_HPP
