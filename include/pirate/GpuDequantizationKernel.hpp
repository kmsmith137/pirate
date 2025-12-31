#ifndef _PIRATE_GPU_DEQUANTIZATION_KERNEL_HPP
#define _PIRATE_GPU_DEQUANTIZATION_KERNEL_HPP

#include <ksgpu/Array.hpp>
#include <ksgpu/Dtype.hpp>
#include "ResourceTracker.hpp"

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// GpuDequantizationKernel: convert int4 array to float32 or float16.
//
// Input:  contiguous array of shape (nbeams, nfreq, ntime), dtype int4
// Output: contiguous array of shape (nbeams, nfreq, ntime), dtype float32 or float16
//
// The int4 values are interpreted as signed two's complement (-8 to +7).
// Nibble packing: low nibble = even index, high nibble = odd index.

struct GpuDequantizationKernel
{
    // Constructor args: (dtype, nbeams, nfreq, ntime)
    // dtype: output dtype (must be float32 or float16)
    // Throws exception if dtype is invalid, or if ntime is not divisible by 256.
    GpuDequantizationKernel(ksgpu::Dtype dtype, long nbeams, long nfreq, long ntime);
    
    // Reference implementation (CPU, always outputs float32 regardless of this->dtype).
    // Input: shape (nbeams, nfreq, ntime), dtype int4, fully contiguous, on host
    // Output: shape (nbeams, nfreq, ntime), dtype float32, fully contiguous, on host
    void apply_reference(ksgpu::Array<float> &out, const ksgpu::Array<void> &in) const;
    
    // GPU kernel launch (async, does not sync stream).
    // Input: shape (nbeams, nfreq, ntime), dtype int4, fully contiguous, on GPU
    // Output: shape (nbeams, nfreq, ntime), dtype matches this->dtype, fully contiguous, on GPU
    void launch(ksgpu::Array<void> &out, const ksgpu::Array<void> &in, cudaStream_t stream) const;
    
    // Static test function (called via 'python -m pirate_frb test --gdqk')
    static void test_random();
    
    // Static timing function
    static void time_selected();
    
    // Members
    ksgpu::Dtype dtype;  // output dtype (float32 or float16)
    long nbeams;
    long nfreq;
    long ntime;
    
    ResourceTracker resource_tracker;

    // Kernel launch config
    dim3 nblocks;
    dim3 nthreads;
};


}  // namespace pirate

#endif // _PIRATE_GPU_DEQUANTIZATION_KERNEL_HPP

