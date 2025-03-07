#ifndef _PIRATE_INTERNALS_LAGGED_DOWNSAMPLING_KERNEL_HPP
#define _PIRATE_INTERNALS_LAGGED_DOWNSAMPLING_KERNEL_HPP

#include <vector>
#include <memory>
#include <iostream>

#include "ReferenceLagbuf.hpp"

#include <ksgpu/Dtype.hpp>
#include <ksgpu/Array.hpp>


namespace pirate {
#if 0
}  // editor auto-indent
#endif


// Defined in DedispersionBuffer.hpp
struct DedispersionBuffer;


struct LaggedDownsamplingKernelParams
{
    // FIXME rename and explain the *_input_rank params.
    ksgpu::Dtype dtype;                 // same as DedispersionConfig::dtype
    long small_input_rank = -1;         // same as DedispersionPlan::stage1_trees[1].rank0 + 1
    long large_input_rank = -1;         // same as DedispersionConfig::tree_rank;
    long num_downsampling_levels = -1;  // same as DedispersionConfig::num_downsampling_levels
    long total_beams = 0;               // same as DedispersionConfig::beams_per_gpu
    long beams_per_batch = 0;           // same as DedispersionConfig::beams_per_batch
    long ntime = 0;                     // same as DedispersionConfig::time_samples_per_chunk

    LaggedDownsamplingKernelParams() { }
    LaggedDownsamplingKernelParams(const LaggedDownsamplingKernelParams &) = default;

    void print(std::ostream &os=std::cout, int indent=0) const;
    void validate() const;  // throws an exception if anything is wrong
};


// Note: the reference kernel allocates persistent state in the constructor.
struct ReferenceLaggedDownsamplingKernel
{
    const LaggedDownsamplingKernelParams params;

    ReferenceLaggedDownsamplingKernel(const LaggedDownsamplingKernelParams &params);

    void apply(DedispersionBuffer &buf, long ibatch);
    
    int nbatches = 0;  // same as (params.total_beams / params.beams_per_batch)
    std::vector<ReferenceLagbuf> lagbufs_small;  // length nbatches
    std::vector<ReferenceLagbuf> lagbufs_large;  // length nbatches
    std::shared_ptr<ReferenceLaggedDownsamplingKernel> next;

    // Helper for apply().
    void _apply(const ksgpu::Array<float> &in, ksgpu::Array<void> *outp, long ibatch);
};
    

// Note: the gpu kernel allocates persistent state in GpuLaggedDownsamplingKernel::allocate().
class GpuLaggedDownsamplingKernel
{
public:
    const LaggedDownsamplingKernelParams params;
    
    // Factory function used to construct new GpuLaggedDownsamplingKernel objects.
    static std::shared_ptr<GpuLaggedDownsamplingKernel> make(const LaggedDownsamplingKernelParams &params);

    void allocate();
    bool is_allocated() const;
    
    // NULL stream is allowed, but is not the default.
    virtual void launch(DedispersionBuffer &buf, long ibatch, long it_chunk, cudaStream_t stream) = 0;
    
    void print(std::ostream &os=std::cout, int indent=0) const;
    
    // Parameters computed in constructor.
    int nbatches = 0;
    int shmem_nbytes_per_threadblock = 0;
    int state_nelts_per_beam = 0;
    
    // These parameters (also computed in constructor) determine how the kernel is divided
    // into threadblocks. See GpuLaggedDownsamplingKernel.cu for more info.
    int M_W = 0;
    int M_B = 0;
    int A_W = 0;
    int A_B = 0;

protected:
    // Constructor is protected -- use GpuLaggedDownsamplingKernel::make() instead.
    GpuLaggedDownsamplingKernel(const LaggedDownsamplingKernelParams &params);

    // Shape (total_beams, state_nelts_per_beam).
    ksgpu::Array<void> persistent_state;
};


}  // namespace pirate

#endif // _PIRATE_INTERNALS_AGGED_DOWNSAMPLING_KERNEL_HPP
