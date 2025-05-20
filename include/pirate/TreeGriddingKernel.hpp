#ifndef _PIRATE_TREE_GRIDDING_KERNEL_HPP
#define _PIRATE_TREE_GRIDDING_KERNEL_HPP

#include <ksgpu/Array.hpp>
#include "trackers.hpp"  // BandwidthTracker

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// Defined in TreeGriddingKernel.cu
struct TreeGriddingKernelParams
{
    ksgpu::Dtype dtype;
    long nfreq = 0;
    long nchan = 0;   // number of tree channels
    long ntime = 0;
    long beams_per_batch = 0;
    
    // Length (nchan+1), values are in [0,nfreq].
    ksgpu::Array<float> channel_map;

    // Validates params, and returns reference to 'this'.
    const TreeGriddingKernelParams &validate() const;
};


// Defined in TreeGriddingKernel.cu
struct ReferenceTreeGriddingKernel
{
    ReferenceTreeGriddingKernel(const TreeGriddingKernelParams &params);

    // Note: params.dtype is ignored in reference kernel (dtype of apply() is always float).
    // Output array shape is (beams_per_batch, nchan, ntime), and inner two indices must be contiguous.
    // Input array shape is (beams_per_batch, nfreq, ntime), and inner two indices must be contiguous.
    void apply(ksgpu::Array<float> &out, const ksgpu::Array<float> &in);

    const TreeGriddingKernelParams params;
};


// Defined in TreeGriddingKernel.cu
struct GpuTreeGriddingKernel
{
    GpuTreeGriddingKernel(const TreeGriddingKernelParams &params);

    void allocate();

    // launch(): asynchronously launch kernel, and return without synchronizing streams.
    // Note: stream=NULL is allowed, but is not the default.
    void launch(ksgpu::Array<void> &out, const ksgpu::Array<void> &in, cudaStream_t stream);

    const TreeGriddingKernelParams params;

    bool is_allocated = false;
    ksgpu::Array<float> gpu_channel_map;
    
    // Bandwidth per call to GpuTreeGriddingKernel::launch().
    // To get bandwidth per time chunk, multiply by (total_beams / beams_per_batch).
    BandwidthTracker bw_per_launch;

    // For kernel launch.
    dim3 nblocks;
    dim3 nthreads;
    int nchan_per_thread;
};


}  // namespace pirate

#endif // _PIRATE_TREE_GRIDDING_KERNEL_HPP
