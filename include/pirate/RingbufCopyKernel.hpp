#ifndef _PIRATE_RINGBUF_COPY_KERNEL_HPP
#define _PIRATE_RINGBUF_COPY_KERNEL_HPP

#include <ksgpu/Array.hpp>
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
    
    void apply(ksgpu::Array<void> &ringbuf, long ibatch, long it_chunk);

    const RingbufCopyKernelParams params;
    const int noctuples;   // = (octuples.size / 8)
    
    // Bandwidth per call to CpuRingbufCopyKernel::launch().
    // To get bandwidth per time chunk, multiply by (total_beams / beams_per_batch).
    BandwidthTracker bw_per_launch;
};


// Defined in RingbufCopyKernel.cu
struct GpuRingbufCopyKernel
{
    GpuRingbufCopyKernel(const RingbufCopyKernelParams &params);

    void allocate();

    // launch(): asynchronously launch copy kernel, and return without synchronizing streams.
    // Note: stream=NULL is allowed, but is not the default.
    void launch(ksgpu::Array<void> &ringbuf, long ibatch, long it_chunk, cudaStream_t stream);

    const RingbufCopyKernelParams params;
    const int noctuples;     // = (octuples.size / 8)

    bool is_allocated = false;
    ksgpu::Array<uint> gpu_octuples;
    
    // Bandwidth per call to GpuRingbufCopyKernel::launch().
    // To get bandwidth per time chunk, multiply by (total_beams / beams_per_batch).
    BandwidthTracker bw_per_launch;

    // Static member function: runs one randomized test iteration.
    static void test();
};


}  // namespace pirate

#endif // _PIRATE_RINGBUF_COPY_KERNEL_HPP
