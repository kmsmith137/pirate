#include "../include/pirate/CudaStreamPool.hpp"

#include <atomic>
#include <stdexcept>

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// Thread-safe global counter for pool IDs.
static std::atomic<int> next_pool_id{1};


CudaStreamPool::CudaStreamPool(int num_compute_streams_, int compute_stream_priority)
    : num_compute_streams(num_compute_streams_)
{
    if (num_compute_streams <= 0)
        throw std::runtime_error("CudaStreamPool: num_compute_streams must be > 0");
    
    // Assign unique pool ID.
    pool_id = next_pool_id.fetch_add(1);
    
    // Create low-priority transfer streams (default priority = 0).
    low_priority_g2h_stream = ksgpu::CudaStreamWrapper::create(0);
    low_priority_h2g_stream = ksgpu::CudaStreamWrapper::create(0);
    
    // Create high-priority transfer streams (priority = -1).
    high_priority_g2h_stream = ksgpu::CudaStreamWrapper::create(-1);
    high_priority_h2g_stream = ksgpu::CudaStreamWrapper::create(-1);
    
    // Create compute streams.
    compute_streams = ksgpu::CudaStreamWrapper::create_vector(num_compute_streams, compute_stream_priority);
}


std::shared_ptr<CudaStreamPool> CudaStreamPool::create(int num_compute_streams, int compute_stream_priority)
{
    // Note: can't use std::make_shared since constructor is private.
    return std::shared_ptr<CudaStreamPool>(new CudaStreamPool(num_compute_streams, compute_stream_priority));
}


}  // namespace pirate

