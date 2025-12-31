#ifndef _PIRATE_TREE_GRIDDING_KERNEL_HPP
#define _PIRATE_TREE_GRIDDING_KERNEL_HPP

#include <ksgpu/Array.hpp>
#include "BumpAllocator.hpp"
#include "ResourceTracker.hpp"

namespace pirate {
#if 0
}  // editor auto-indent
#endif


struct TreeGriddingKernelParams
{
    ksgpu::Dtype dtype;
    long nfreq = 0;   // number of input frequency channels
    long nchan = 0;   // number of output tree channels
    long ntime = 0;   // time samples per chunk
    long beams_per_batch = 0;

    // Length (nchan+1), values are in [0,nfreq], stored in CPU memory.
    // Defines the mapping between output "tree" channels and frequency channels.
    // The array must be monotonically decreasing (channel_map[i] > channel_map[i+1]).
    //
    // Given tree channel 0 <= itree < ntree, the values of channel_map[itree+1] and
    // channel_map[itree] define the edges of the tree channel in frequency space
    // (i.e., the interval [channel_map[itree+1], channel_map[itree])).
    //
    // NOTE: we use double precision, since weights are computed by differencing
    // (channel_map[i+1] - channel_map[i]), which loses a lot of relative precision.

    ksgpu::Array<double> channel_map;

    // Validates params, and returns reference to 'this'.
    const TreeGriddingKernelParams &validate() const;
};


struct ReferenceTreeGriddingKernel
{
    ReferenceTreeGriddingKernel(const TreeGriddingKernelParams &params);

    // Rebins input frequency channels into output "tree" channels, using weighted sums.
    //
    // For each tree channel n (0 <= n < nchan), the channel_map defines floating-point
    // boundaries [f0, f1) = [channel_map[n+1], channel_map[n]). Note that channel_map
    // is monotonically decreasing, so channel_map[n+1] < channel_map[n]. The output is:
    //
    //   out[b,n,t] = sum_f w[n,f] * in[b,f,t]
    //
    // where w[n,f] is the fractional overlap between the tree channel [f0,f1) and the
    // frequency bin [f, f+1). That is, w = max(min(f1,f+1) - max(f0,f), 0). This gives
    // weight 1 for fully contained frequency channels, and fractional weights at edges.
    //
    // Note: params.dtype is ignored in reference kernel (dtype of apply() is always float).
    // Output array shape is (beams_per_batch, nchan, ntime), and inner two indices must be contiguous.
    // Input array shape is (beams_per_batch, nfreq, ntime), and inner two indices must be contiguous.
    void apply(ksgpu::Array<float> &out, const ksgpu::Array<float> &in);

    const TreeGriddingKernelParams params;
};


struct GpuTreeGriddingKernel
{
    GpuTreeGriddingKernel(const TreeGriddingKernelParams &params);

    void allocate(BumpAllocator &allocator);

    // launch(): asynchronously launch kernel, and return without synchronizing stream.
    // Note: stream=NULL is allowed, but is not the default.
    void launch(ksgpu::Array<void> &out, const ksgpu::Array<void> &in, cudaStream_t stream);

    // Reminder: contains 'channel_map', which lives in host memory.
    const TreeGriddingKernelParams params;

    // If is_allocated=true, then 'gpu_channel_map' is a copy of 'channel_map' in GPU memory.
    // FIXME: using double precision in a GPU kernel!! This is a temporary kludge.
    ksgpu::Array<double> gpu_channel_map;
    bool is_allocated = false;

    // All rates are "per call to launch()".
    ResourceTracker resource_tracker;

    // For kernel launch.
    dim3 nblocks;
    dim3 nthreads;
    int nchan_per_thread;

    static void test_random();  // runs one randomized test iteration
    static void time_selected();  // does a timing run with chord-like parameters
};


}  // namespace pirate

#endif // _PIRATE_TREE_GRIDDING_KERNEL_HPP
