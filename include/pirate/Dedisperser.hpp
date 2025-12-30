#ifndef _PIRATE_DEDISPERSER_HPP
#define _PIRATE_DEDISPERSER_HPP

#include "BumpAllocator.hpp"
#include "DedispersionConfig.hpp"
#include "DedispersionBuffer.hpp"
#include "DedispersionTree.hpp"
#include "ResourceTracker.hpp"

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
struct CoalescedDdKernel2;            // defined in CoalescedDdKernel2.hpp


// -------------------------------------------------------------------------------------------------
//
// GpuDedisperser (defined in src_lib/GpuDedisperser.cu)
//
// Warning: not thread-safe!


struct GpuDedisperser
{
    GpuDedisperser(const std::shared_ptr<DedispersionPlan> &plan);
    
    std::shared_ptr<DedispersionPlan> plan;
    
    // Some key members of DedispersionPlan, copied into GpuDedisperser for convenience.
    DedispersionConfig config;             // same as plan->config
    ksgpu::Dtype dtype;                    // same as plan->dtype
    long nfreq = 0;                        // same as plan->freq
    long nt_in = 0;                        // same as plan->nt_in
    long total_beams = 0;                  // same as plan->beams_per_gpu
    long beams_per_batch = 0;              // same as plan->beams_per_batch
    long nstreams = 0;                     // same as plan->num_active_batches
    long nbatches = 0;                     // = (total_beams / beams_per_batch)
    long ntrees = 0;                       // same as plan->ntrees
    std::vector<DedispersionTree> trees;   // same as plan->trees

    // "inner" array shape (nstreams, beams_per_batch, nfreq, ntime).
    ksgpu::Array<void> input_arrays;       // config.dtype

    // "outer" vector has length ntrees
    // "inner" array shape = this->extended_wt_shapes[itree], see below
    std::vector<ksgpu::Array<void>> wt_arrays;

    // "outer" vector has length ntrees
    // "inner" array shape (nstreams, beams_per_batch, t.ndm_out, t.nt_out)
    //    where t= plan->trees.at(itree)
    //
    // Note: currently using a "short" (length-nstreams) ring buffer for 
    // dedispersion outputs. In the future when I implement RPC postprocessing,
    // it may make sense to have a configurable buffer length.

    std::vector<ksgpu::Array<void>> out_max;     // config.dtype
    std::vector<ksgpu::Array<uint>> out_argmax;  // uint dtype

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

    bool is_allocated = false;

    // Peak-finding weights use a complicated GPU memory layout.
    // The helper class 'GpuPfWeightLayout' is intended to hide the details.
    // We have one GpuPfWeightLayout per output tree: cdd2_kernels[itree]->pf_weight_layout.
    // The shape for one beam batch is here: cdd2_kernels[itree]->expected_wt_shape.
    // The "extended" shapes below adds an outer length-nstreams index.
    std::vector<std::vector<long>> extended_wt_shapes;   // length ntrees
    std::vector<std::vector<long>> extended_wt_strides;  // length ntrees

    // All rates are "per call to launch()".
    ResourceTracker resource_tracker;

    // Note: allocate() initializes or zeroes all arrays (i.e. no array is left uninitialized)
    void allocate(BumpAllocator &gpu_allocator, BumpAllocator &host_allocator);

    // Reminder: a "chunk" is a range of time indices, and a "batch" is a range of beam indices.
    // XXX temporary kludge: uses default stream for all kernels/copies!!!

    void launch(long ichunk, long ibatch);

    // Static member function: runs one randomized test iteration.
    static void test();

    // Static member function: runs one test iteration with specified configuration.
    static void test_one(const DedispersionConfig &config, int nchunks, bool host_only=false);

    void _launch_tree_gridding(long ichunk, long ibatch);
    void _launch_lagged_downsampler(long ichunk, long ibatch);
    void _launch_dd_stage1(long ichunk, long ibatch);
    void _launch_et_g2g(long ichunk, long ibatch);   // copies from 'gpu' zones to 'g2h' zones
    void _run_et_h2h(long ichunk, long ibatch);      // on CPU! copies from 'host' zones to 'et_host' zone
    void _launch_et_h2g(long ichuink, long ibatch);  // copies from 'et_host' zone to 'et_gpu' zone
    void _launch_g2h(long ichunk, long ibatch);      // this is the main gpu->host copy
    void _launch_h2g(long ichunk, long ibatch);      // this is the main host->gpu copy
    void _launch_cdd2(long ichunk, long ibatch);
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
