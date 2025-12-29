#ifndef _PIRATE_LAGGED_DOWNSAMPLING_KERNEL_HPP
#define _PIRATE_LAGGED_DOWNSAMPLING_KERNEL_HPP

#include <vector>
#include <memory>
#include <iostream>

#include <ksgpu/Dtype.hpp>
#include <ksgpu/Array.hpp>

#include "BumpAllocator.hpp"
#include "ResourceTracker.hpp"


namespace pirate {
#if 0
}  // editor auto-indent
#endif

struct DedispersionBuffer;  // defined in DedispersionBuffer.hpp
struct ReferenceLagbuf;     // defined in ReferenceLagbuf.hpp


struct LaggedDownsamplingKernelParams
{
    // The input to the lagged downsampling kernel is an array of shape
    //
    //   (beams_per_batch, pow2(input_total_rank), ntime).
    //
    // The output is a length (num_downsampling_levels - 1) sequence of arrays,
    // indexed by 1 <= ids < num_downsampling_levels, with array shape
    //
    //   (beams_per_batch, pow2(input_total_rank-1), ntime / 2^ids).
    //
    // The 'output_dd_rank' parameter is the dedispersion rank of the
    // "stage1" transform that will subseqeuntly be applied to the output
    // arrays (which must be the same for all values of ids), satisfying:
    //
    //   0 <= output_dd_rank <= (input_total_rank-1).
    
    ksgpu::Dtype dtype;                 // same as DedispersionConfig::dtype
    long input_total_rank = -1;         // same as DedispersionConfig::tree_rank;
    long output_dd_rank = -1;           // same as DedispersionPlan::stage1_dd_rank[1]
    long num_downsampling_levels = -1;  // same as DedispersionConfig::num_downsampling_levels
    long total_beams = 0;               // same as DedispersionConfig::beams_per_gpu
    long beams_per_batch = 0;           // same as DedispersionConfig::beams_per_batch
    long ntime = 0;                     // same as DedispersionConfig::time_samples_per_chunk

    LaggedDownsamplingKernelParams() { }
    LaggedDownsamplingKernelParams(const LaggedDownsamplingKernelParams &) = default;

    void validate() const;  // throws an exception if anything is wrong

    // Emit C++ code to initialize this LaggedDownsamplingKernelParams.
    // (Sometimes convenient in unit tests.)
    void emit_cpp(std::ostream &os=std::cout, const char *name="params", int indent=4);
};


// Notes:
//
//   - The reference kernel allocates persistent state in its constructor, and does
//     not define an allocate() method.
//
//   - The reference kernel uses float32, regardless of the dtype specified in 'params'.

struct ReferenceLaggedDownsamplingKernel
{
    LaggedDownsamplingKernelParams params;

    ReferenceLaggedDownsamplingKernel(const LaggedDownsamplingKernelParams &params);

    void apply(DedispersionBuffer &buf, long ibatch);
    
    int nbatches = 0;  // same as (params.total_beams / params.beams_per_batch)
    std::vector<std::shared_ptr<ReferenceLagbuf>> lagbufs_small;  // length nbatches
    std::vector<std::shared_ptr<ReferenceLagbuf>> lagbufs_large;  // length nbatches
    std::shared_ptr<ReferenceLaggedDownsamplingKernel> next;

    // Helper for apply().
    void _apply(const ksgpu::Array<float> &in, ksgpu::Array<void> *outp, long ibatch);
};
    

// Note: the gpu kernel allocates persistent state in GpuLaggedDownsamplingKernel::allocate().
class GpuLaggedDownsamplingKernel
{
public:
    LaggedDownsamplingKernelParams params;
    bool is_allocated = false;
    int nbatches = 0;   // (total_beams / beams_per_batch)
    
    // Factory function used to construct new GpuLaggedDownsamplingKernel objects.
    static std::shared_ptr<GpuLaggedDownsamplingKernel> make(const LaggedDownsamplingKernelParams &params);

    // Note: allocate() initializes or zeroes all arrays (i.e. no array is left uninitialized)
    void allocate(BumpAllocator &allocator);

    // One call to launch() processes an array of shape (beams_per_batch, pow2(input_total_rank), ntime).
    // The NULL stream is allowed, but is not the default.
    // Reminder: a "chunk" is a range of time indices, and a "batch" is a range of beam indices.
    virtual void launch(DedispersionBuffer &buf, long ichunk, long ibatch, cudaStream_t stream) = 0;
    
    // Parameters computed in constructor.
    int shmem_nbytes_per_threadblock = 0;
    int state_nelts_per_beam = 0;

    // These parameters (also computed in constructor) determine how the kernel is divided
    // into threadblocks. See GpuLaggedDownsamplingKernel.cu for more info.
    int M_W = 0;
    int M_B = 0;
    int A_W = 0;
    int A_B = 0;

    // Shape (total_beams, state_nelts_per_beam).
    ksgpu::Array<void> persistent_state;

   // All rates are "per call to launch()".
    ResourceTracker resource_tracker;

    // Static member function: runs one randomized test iteration.
    static void test();

    // Static member function: run timing for representative kernels.
    static void time();

protected:
    // Constructor is protected -- use GpuLaggedDownsamplingKernel::make() instead.
    GpuLaggedDownsamplingKernel(const LaggedDownsamplingKernelParams &params);
};


}  // namespace pirate

#endif // _PIRATE_LAGGED_DOWNSAMPLING_KERNEL_HPP
