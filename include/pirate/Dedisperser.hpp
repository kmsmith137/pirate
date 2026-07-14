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
        // Must be a gpu_runnable (hence non-incomplete) plan: the constructor asserts
        // plan->params.gpu_runnable, since the Dcore values must match the compiled kernels.
        std::shared_ptr<DedispersionPlan> plan;
        std::shared_ptr<CudaStreamPool> stream_pool;

        // Set sizes of dedispersion output and peak-finding weights buffers.
        // If uninitialized, default is plan->num_active_batches.
        long nbatches_out = 0;
        long nbatches_wt = 0;    // infrequently used (only in a unit test)

        // Number of independent downstream consumers of the dedispersion
        // output. Each consumer drives its own acquire_output / release_output
        // progress cursor and back-pressure. num_consumers == 0 is allowed
        // (the dedisperser drops outputs as soon as cdd2 produces them).
        // Constructor throws if num_consumers < 0; the default of -1 is a
        // "you forgot to set it" sentinel.
        long num_consumers = -1;

        // CUDA device id. The GpuDedisperser's worker thread will call
        // cudaSetDevice(cuda_device_id) on entry, so all kernel launches
        // and async copies driven by the worker thread hit the right
        // device. ~GpuDedisperser also temporarily switches to this device
        // for its cleanup work. Required (constructor asserts >= 0).
        int cuda_device_id = -1;

        // If true, each consumer must interleave its acquire_output() /
        // release_output() calls (acquire seq_id N, release seq_id N, acquire
        // N+1, ...). If false, a consumer may acquire_output() several batches
        // (consecutive seq_ids) before releasing them -- appropriate for an
        // asynchronous consumer.
        //
        // Currently the same for all consumers, but could be made per-consumer.
        bool synchronous = true;

        // This optional member is only used to set Outputs::ichunk_fpga_based
        // (to Outputs::ichunk_zero-based + initial_chunk).
        long initial_chunk = 0;
    };

    
    // struct Outputs: dedisperserion output buffers, for one time chunk and
    // multiple beams. Used in two contexts:
    //
    //  - GpuDedisperser::output_ringbuf is a ring buffer (shared between
    //    dedisperser and grouper), containing outputs for many beams.
    //
    //  - GpuDedisperser::acquire_output() returns outputs for one
    //    batch of beams.

    struct Outputs {
        // Note that this is the dtype of the 'out_max' array only (out_argmax is always uint).
        ksgpu::Dtype dtype;
        
        // Beam count, whose value is context-dependent:
        //   - GpuDedisperser::output_ringbuf: nbeams = nbatches_out * beams_per_batch.
        //   - Slice returned by acquire_output(): nbeams = beams_per_batch.
        long nbeams = 0;

        long ichunk_zero_based = 0;    // chunk index, relative to first dedisperser output
        long ichunk_fpga_based = 0;    // chunk index, relative to fpga seq 0
        long ibeam = 0;                // beam index (not beam_id!!) of first beam in Outputs
        
        std::vector<long> ndm_out;     // length ntrees
        std::vector<long> nt_out;      // length ntrees

        // "Outer" length ntrees.
        // "Inner" shape (nbeams, ndm_out[i], nt_out[i]).
        
        std::vector<ksgpu::Array<void>> out_max;     // config.dtype
        std::vector<ksgpu::Array<uint>> out_argmax;  // uint dtype

        // Caller must initialize {dtype, nbeams, ndm_out, nt_out} before calling.
        void allocate(BumpAllocator &gpu_allocator);

        // Slice along beam axis and return a "view".
        Outputs slice(long start_beam_index, long end_beam_index) const;
    };

    // Factory function to create GpuDedisperser (preferred over direct construction).
    static std::shared_ptr<GpuDedisperser> create(const Params &params);
    
    // Noncopyable and nonmoveable
    GpuDedisperser(const GpuDedisperser&) = delete;
    GpuDedisperser& operator=(const GpuDedisperser&) = delete;
    GpuDedisperser(GpuDedisperser&&) = delete;
    GpuDedisperser& operator=(GpuDedisperser&&) = delete;
    
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

    void allocate(BumpAllocator &gpu_allocator, BumpAllocator &host_allocator);

    // All four progress-cursor methods take a 'seq_id' argument: the global
    // batch index 0, 1, 2, ..., related to the (ichunk, ibatch) chunk/batch
    // pair by seq_id = ichunk*nbatches + ibatch. acquire_input / release_input
    // (and each consumer's acquire_output / release_output) must be called with
    // seq_id = 0, 1, 2, ... in order.
    //
    // By default (Params::synchronous==true) a consumer must also INTERLEAVE its
    // acquire_output()/release_output() calls (acquire N, release N, acquire N+1,
    // ...). With Params::synchronous==false a consumer may acquire_output()
    // several batches before releasing them; the only requirement is that release
    // stays behind acquire (a batch can't be released before it is acquired).
    //
    // acquire_input(seq_id, stream): after call, 'stream' sees empty input
    //                  buffer. Returns the input buffer as ksgpu::Array<void>
    //                  with shape (beams_per_batch, nfreq, nt_in), valid until
    //                  the matching release_input_and_launch_dd_kernels() call.
    // release_input_and_launch_dd_kernels(seq_id, stream): before call,
    //                  'stream' must see full input buffer.
    // acquire_output(consumer_id, seq_id, stream, sync, noreturn):
    //                  after call, 'stream' sees full output buffer (if
    //                  sync=true, the host thread blocks instead and 'stream'
    //                  is ignored). Returns a GpuDedisperser::Outputs struct
    //                  holding list-of-Array views of out_max and out_argmax,
    //                  valid until the matching release_output() call (if
    //                  noreturn=true, returns an empty Outputs). consumer_id
    //                  must be in [0, params.num_consumers).
    // release_output(consumer_id, seq_id, stream):
    //                  before call, 'stream' must see empty output buffer.
    //                  consumer_id must be in [0, params.num_consumers).
    //
    // FIXME: currently there's no explicit API protecting the pf_weights.
    // The caller is responsible for thinking through race conditions involving
    // the pf_weights (e.g. in GpuDedisperser::test_one(), where the weights
    // are updated during the unit test). This issue may go away on its own,
    // since my tentative long-term plan is to generate the pf_weights
    // "dynamically" as part of the compute graph.

    ksgpu::Array<void> acquire_input (long seq_id, cudaStream_t stream);
    void               release_input_and_launch_dd_kernels (long seq_id, cudaStream_t stream);

    // If sync=true, then ignore 'stream' and synchronize the host thread
    // (call evrb_cdd2->synchronize() instead of evrb_cdd2->wait()).
    // If noreturn=true, then return an empty Outputs object (reduces overhead
    // for callers who don't need it).
    Outputs            acquire_output(long consumer_id, long seq_id, cudaStream_t stream,
                                      bool sync=false, bool noreturn=false);
    void               release_output(long consumer_id, long seq_id, cudaStream_t stream);

    // (No separate view methods; both input and output return their views
    // directly from acquire.)

    // Thread-backed class pattern: stop the worker thread and put the object
    // into a "stopped" state. The first caller sets 'error'. If e is null,
    // represents normal termination; if non-null, represents an error.
    // Entry points called after stop() will rethrow the stored exception.
    void stop(std::exception_ptr e = std::exception_ptr()) const;

    ~GpuDedisperser();

    // Static member function: runs one randomized test iteration.
    static void test_random();

    // Static member function: do one test run with specified configuration.
    // nbatches_out=0 defaults to nstreams; nbatches_wt=0 defaults to nbatches_out.
    static void test_one(const DedispersionConfig &config, long nchunks,
                         long nbatches_out=0, long nbatches_wt=0, bool host_only=false);

    // Run timing benchmark (C++ version). Entry point: throws on a stopped
    // (or never-allocated) instance; any throw stops the GpuDedisperser.
    // To run from command line: 'python -m pirate_frb time_dedisperser config.yml'.
    // (Note that the --python flag will run the python version of the timing benchmark, 
    //  which is in pirate_frb.utils.time_cupy_dedisperser().)
    void time(BumpAllocator &gpu_allocator, BumpAllocator &cpu_allocator, long niterations);

    // Fills the peak-finding weight arrays (wt_arrays) with NON-random analytic weights,
    // derived from a PfAvarApproximation built from (plan, freq_variances). All weight
    // slots and all beams get identical weights. Must be called after allocate(). Blocks
    // (calls cudaDeviceSynchronize) before returning, so the weights are in place on the
    // GPU when it returns. Entry point: throws on a stopped (or never-allocated)
    // instance; any throw stops the GpuDedisperser.
    void fill_analytic_weights(const ksgpu::Array<double> &freq_variances);

    // Copies host-side peak-finding weights to the GPU weight arrays (wt_arrays[itree]) for a
    // single tree, filling all 'nbatches_wt' weight slots. 'pf_weights' has shape
    // (nbatches_wt, beams_per_batch, t.ndm_wt, t.nt_wt, t.nprofiles, t.frequency_subbands.N),
    // with t = plan->trees[itree]. Unlike fill_analytic_weights(), the weights may differ per
    // slot and per beam. Must be called after allocate(). Does NOT synchronize -- the caller
    // owns any race conditions on the pf_weights (see the FIXME in the High-level API section).
    // Entry point: throws on a stopped (or never-allocated) instance; any throw
    // stops the GpuDedisperser.
    void fill_all_weights(long itree, const ksgpu::Array<float> &pf_weights);

    // --------------------------  Public members  --------------------------

protected:
    // Constructor is protected - use create() factory function instead.
    GpuDedisperser(const Params &params);

public:

    // "inner" array shape (nstreams, beams_per_batch, nfreq, ntime).
    ksgpu::Array<void> input_arrays;       // config.dtype

    // "outer" vector has length ntrees
    // "inner" array shape = this->extended_wt_shapes[itree], see below
    std::vector<ksgpu::Array<void>> wt_arrays;

    // GpuDedisperser-owned output ringbuf (see struct Outputs above). Its
    // out_max/out_argmax have inner shape (nbatches_out * beams_per_batch,
    // t.ndm_out, t.nt_out) with t = plan->trees.at(itree); the leading axis
    // flattens (nbatches_out, beams_per_batch), so per-batch slot 'iout'
    // occupies beam range [iout*beams_per_batch, (iout+1)*beams_per_batch),
    // extracted via output_ringbuf.slice().
    Outputs output_ringbuf;

    // allocate() bookkeeping, both protected by 'mutex'. allocate_called is
    // set at the START of allocate() (double-call guard -- correct even for
    // concurrent calls); is_allocated is set at the END, once the buffers
    // exist and the worker thread is spawned.
    bool allocate_called = false;
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
    void _launch_dedispersion_kernels(long seq_id, cudaStream_t stream);
    void _worker_main();
    void worker_main();

    // Entry-point bodies. The public methods are thin wrappers that stop the
    // GpuDedisperser if the body throws (see notes/stoppable_class.md).
    void _allocate(BumpAllocator &gpu_allocator, BumpAllocator &host_allocator);
    ksgpu::Array<void> _acquire_input(long seq_id, cudaStream_t stream);
    void _release_input_and_launch_dd_kernels(long seq_id, cudaStream_t stream);
    Outputs _acquire_output(long consumer_id, long seq_id, cudaStream_t stream, bool sync, bool noreturn);
    void _release_output(long consumer_id, long seq_id, cudaStream_t stream);
    void _time(BumpAllocator &gpu_allocator, BumpAllocator &cpu_allocator, long niterations);
    void _fill_analytic_weights(const ksgpu::Array<double> &freq_variances);
    void _fill_all_weights(long itree, const ksgpu::Array<float> &pf_weights);

    // Thread-backed class pattern: helpers. Both must be called with 'mutex' held.
    void _throw_if_stopped(const char *method_name);
    void _throw_if_unallocated(const char *method_name);

    // The CudaEventRingbufs keep track of lagged dependencies between kernels.
    std::shared_ptr<CudaEventRingbuf> evrb_tree_gridding;
    std::shared_ptr<CudaEventRingbuf> evrb_g2g;
    std::shared_ptr<CudaEventRingbuf> evrb_g2h;
    std::shared_ptr<CudaEventRingbuf> evrb_h2g;
    std::shared_ptr<CudaEventRingbuf> evrb_cdd2;
    std::shared_ptr<CudaEventRingbuf> evrb_et_h2g;

    // Length params.num_consumers. Each consumer's release_output() events
    // go into its own ring; the cdd2 kernel waits on all N rings.
    std::vector<std::shared_ptr<CudaEventRingbuf>> evrb_release_output;

    // These members help keep track of lags between kernels.
    long host_seq_lag = 0;     // has_host_ringbuf ? (mega_ringbuf->min_host_clag * nbatches) : (2*nstreams)
    long et_seq_headroom = 0;  // has_early_triggers ? (mega_ringbuf->min_et_headroom * nbatches) : (2*nstreams)
    long et_seq_lag = 0;       // has_early_triggers ? (mega_ringbuf->min_et_clag * nbatches) : (2*nstreams)

    // Stop-pattern state ('mutable' since stop() is const -- see
    // notes/stoppable_class.md). is_stopped/error are protected by 'mutex'.
    mutable std::mutex mutex;
    mutable bool is_stopped = false;
    mutable std::exception_ptr error;

    long curr_input_seq_id = 0;
    bool curr_input_acquired = false;

    // Per-consumer progress cursors. Each vector has length params.num_consumers.
    // curr_output_acquire_seq_id: next seq_id to be passed to acquire_output().
    // curr_output_release_seq_id: next seq_id to be passed to release_output().
    // Invariant: curr_output_release_seq_id <= curr_output_acquire_seq_id (a batch
    // can't be released before it is acquired). In synchronous mode the two
    // cursors differ by at most 1 (acquire/release must interleave); in
    // asynchronous mode (params.synchronous==false) acquire may run arbitrarily
    // far ahead of release.
    std::vector<long> curr_output_acquire_seq_id;
    std::vector<long> curr_output_release_seq_id;

    // Thread-backed class pattern: worker thread. Spawned at the end of
    // allocate(), with the handle published under 'mutex' (see "Spawning,
    // joining, and teardown" in notes/cpp.md); joined in ~GpuDedisperser.
    // Note: no condition_variable needed to wake up worker thread, since the worker does
    // all of its waiting on condition_variables which are members of CudaEventRingbuf.
    std::thread worker;
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
// Note on Dcore: the per-tree peak-finding Dcore values come from the plan
// (plan->stage2_pf_params[:].Dcore, filled from the cdd2 kernel registry), so the
// reference peak-finders mimic the GPU kernels whenever the corresponding cdd2 kernels
// are compiled into the build. See PeakFindingKernelParams::Dcore for details.

struct ReferenceDedisperserBase
{
    struct Params {
        std::shared_ptr<DedispersionPlan> plan;
        int sophistication = -1;        // 0, 1, or 2 (see above)
        bool enable_variances = false;  // if true, allocate + fill out_var

        // If true, the tree gridding kernel is skipped: 'input_array' has shape
        // (beams_per_batch, pow2(toplevel_tree_rank), nt_in) and is interpreted as an
        // already-gridded toplevel tree-domain array. Used by unit tests that need to
        // inject probes into specific tree-freq channels (see test_decode_argmax).
        bool tree_domain_input = false;
    };

    // Constructor not intended to be called directly -- use make() below, which
    // dispatches on Params::sophistication.
    ReferenceDedisperserBase(const Params &params);

    Params params;   // construction parameters

    // Some key members of DedispersionPlan, copied in for convenience.
    DedispersionConfig config;             // same as params.plan->config
    ksgpu::Dtype dtype;                    // same as params.plan->dtype
    long nfreq = 0;                        // same as params.plan->nfreq
    long nt_in = 0;                        // same as params.plan->nt_in
    long total_beams = 0;                  // same as params.plan->beams_per_gpu
    long beams_per_batch = 0;              // same as params.plan->beams_per_batch
    long num_primary_trees = 0;      // same as params.plan->num_primary_trees
    long nbatches = 0;                     // = (total_beams / beams_per_batch)
    long ntrees = 0;                       // same as params.plan->ntrees
    std::vector<DedispersionTree> trees;   // same as params.plan->trees

    std::shared_ptr<ReferenceTreeGriddingKernel> tree_gridding_kernel;
    std::vector<std::shared_ptr<ReferencePeakFindingKernel>> pf_kernels;  // length ntrees

    // To process multiple chunks, call the dedisperse() method in a loop.
    // Reminder: a "chunk" is a range of time indices, and a "batch" is a range of beam indices.
    virtual void dedisperse(long ichunk, long ibatch) = 0;

    // Before calling dedisperse(), caller should fill 'input_array'.
    // Shape is (beams_per_batch, nfreq, nt_in), or (beams_per_batch,
    // pow2(toplevel_tree_rank), nt_in) if Params::tree_domain_input is set.
    ksgpu::Array<float> input_array;

    // Befre calling dedisperse(), caller should fill 'wt_arrays' (peak-finding weights).
    // Shape is (beams_per_batch, t.ndm_wt, t.nt_wt, t.nprofiles, t.frequency_subbands.N)
    //   where t = plan->trees.at(itree).
    std::vector<ksgpu::Array<float>> wt_arrays;    // length ntrees

    // After dedisperse() completes, peak-finding output is stored in 'out_max', 'out_argmax'.
    // Shape is (beams_per_batch, t.ndm_out, t.nt_out)
    //   where t = plan->trees.at(itree)
    std::vector<ksgpu::Array<float>> out_max;     // length ntrees
    std::vector<ksgpu::Array<uint>> out_argmax;   // length ntrees

    // Per-chunk peak-finding variance, only allocated if Params::enable_variances (else empty).
    // Shape is (beams_per_batch, t.ndm_out, t.frequency_subbands.M, t.nprofiles).
    // OVERWRITTEN each dedisperse() call. See ReferencePeakFindingKernel::apply().
    std::vector<ksgpu::Array<double>> out_var;    // length ntrees (elements empty if disabled)

    // Factory function -- constructs ReferenceDedisperser of the sophistication in 'params'.
    static std::shared_ptr<ReferenceDedisperserBase> make(const Params &params);
};


}  // namespace pirate

#endif // _PIRATE_DEDISPERSER_HPP
