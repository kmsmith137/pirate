#ifndef _PIRATE_RINGBUF_COPY_KERNEL_HPP
#define _PIRATE_RINGBUF_COPY_KERNEL_HPP

#include <ksgpu/Array.hpp>
#include "BumpAllocator.hpp"
#include "trackers.hpp"  // BandwidthTracker

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// Defined in RingbufCopyKernel.cu
struct RingbufCopyKernelParams
{
    long total_beams = 0;
    long beams_per_batch = 0;
    long nelts_per_segment = 0;

    // Octuples array can either be size-zero, or shape-(2N,4) contiguous.
    ksgpu::Array<uint> octuples;

    // Validates params, and returns reference to 'this'.
    const RingbufCopyKernelParams &validate() const;
};


// Defined in RingbufCopyKernel.cu
struct CpuRingbufCopyKernel
{
    CpuRingbufCopyKernel(const RingbufCopyKernelParams &params);

    // Reminder: a "chunk" is a range of time indices, and a "batch" is a range of beam indices.
    void apply(ksgpu::Array<void> &ringbuf, long ichunk, long ibatch);

    const RingbufCopyKernelParams params;
    const int noctuples;   // = (octuples.size / 8)
    
    // Bandwidth per call to CpuRingbufCopyKernel::launch().
    // To get bandwidth per time chunk, multiply by (total_beams / beams_per_batch).
    BandwidthTracker bw_per_launch;       // all arrays including octuples
    BandwidthTracker bw_core_per_launch;  // only input/output arrays
};


// Defined in RingbufCopyKernel.cu
struct GpuRingbufCopyKernel
{
    GpuRingbufCopyKernel(const RingbufCopyKernelParams &params);

    void allocate(BumpAllocator &allocator);

    // launch(): asynchronously launch copy kernel, and return without synchronizing streams.
    // Reminder: a "chunk" is a range of time indices, and a "batch" is a range of beam indices.
    // Note: stream=NULL is allowed, but is not the default.
    void launch(ksgpu::Array<void> &ringbuf, long ichunk, long ibatch, cudaStream_t stream);

    const RingbufCopyKernelParams params;
    const int noctuples;     // = (octuples.size / 8)

    bool is_allocated = false;
    ksgpu::Array<uint> gpu_octuples;

    // GPU memory footprint (in bytes), computed in constructor, checked in allocate().
    long gmem_footprint_nbytes = 0;
    
    // Bandwidth per call to GpuRingbufCopyKernel::launch().
    // To get bandwidth per time chunk, multiply by (total_beams / beams_per_batch).
    BandwidthTracker bw_per_launch;       // all gpu arrays including octuples
    BandwidthTracker bw_core_per_launch;  // only input/output arrays

    // Static member function: runs one randomized test iteration.
    static void test();
};


}  // namespace pirate

#endif // _PIRATE_RINGBUF_COPY_KERNEL_HPP
