#ifndef _PIRATE_INTERNALS_LAGGED_DOWNSAMPLING_KERNEL_HPP
#define _PIRATE_INTERNALS_LAGGED_DOWNSAMPLING_KERNEL_HPP

#include <vector>
#include <memory>
#include <iostream>

#include "DedispersionBuffers.hpp"  // DedispersionInbuf, DedispersionInbufParams
#include "ReferenceLagbuf.hpp"

#include <ksgpu/Dtype.hpp>
#include <ksgpu/Array.hpp>


namespace pirate {
#if 0
}  // editor auto-indent
#endif

class DedispersionPlan;


// Note: the reference kernel allocates persistent state in the constructor.
struct ReferenceLaggedDownsamplingKernel
{
    const DedispersionInbufParams params;

    ReferenceLaggedDownsamplingKernel(const DedispersionInbufParams &params);

    void apply(DedispersionInbuf &buf, long ibatch);
    
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
    const DedispersionInbufParams params;
    
    // Factory function used to construct new GpuLaggedDownsamplingKernel objects.
    static std::shared_ptr<GpuLaggedDownsamplingKernel> make(const DedispersionInbufParams &params);

    void allocate();
    bool is_allocated() const;
    
    // NULL stream is allowed, but is not the default.
    virtual void launch(DedispersionInbuf &buf, long ibatch, long it_chunk, cudaStream_t stream) = 0;
    
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
    GpuLaggedDownsamplingKernel(const DedispersionInbufParams &params);

    // Shape (total_beams, state_nelts_per_beam).
    ksgpu::Array<void> persistent_state;
};


}  // namespace pirate

#endif // _PIRATE_INTERNALS_AGGED_DOWNSAMPLING_KERNEL_HPP
