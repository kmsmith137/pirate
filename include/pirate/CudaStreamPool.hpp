#ifndef _PIRATE_CUDA_STREAM_POOL_HPP
#define _PIRATE_CUDA_STREAM_POOL_HPP

#include <vector>
#include <memory>
#include <ksgpu/cuda_utils.hpp>

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// CudaStreamPool: A collection of CUDA streams for GPU computation.
//
// Contains low-priority and high-priority streams for GPU<->Host transfers,
// plus a configurable number of compute streams.
//
// CudaStreamPool is noncopyable and should always be accessed via shared_ptr.
// Use CudaStreamPool::create() to construct.


struct CudaStreamPool
{
    // Global ID to identify stream pool, obtained by incrementing thread-safe global variable.
    int pool_id = 0;
    int num_compute_streams = 0;
    
    // "Low-priority" is cuda priority 0, and "high-priority" is cuda priority (-1).
    // Suggest using low priority for dedispersion ringbufs, and high-priority for
    // everything else. I think this simple heuristic will help with throughput.
    
    ksgpu::CudaStreamWrapper low_priority_g2h;   // GPU to host
    ksgpu::CudaStreamWrapper low_priority_h2g;   // Host to GPU
    ksgpu::CudaStreamWrapper high_priority_g2h;  // GPU to host
    ksgpu::CudaStreamWrapper high_priority_h2g;  // Host to GPU
    
    // Compute streams (default priority = 0).
    std::vector<ksgpu::CudaStreamWrapper> compute_streams;

    // Factory function to create a CudaStreamPool.
    // The num_compute_streams argument must be > 0.
    static std::shared_ptr<CudaStreamPool> create(int num_compute_streams);

    // Noncopyable.
    CudaStreamPool(const CudaStreamPool &) = delete;
    CudaStreamPool &operator=(const CudaStreamPool &) = delete;

private:
    // Private constructor - use CudaStreamPool::create() instead.
    CudaStreamPool(int num_compute_streams);
};


}  // namespace pirate

#endif // _PIRATE_CUDA_STREAM_POOL_HPP

