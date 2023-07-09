#ifndef _PIRATE_DEDISPERSER_HPP
#define _PIRATE_DEDISPERSER_HPP

#include <memory>  // shared_ptr

#include "DedispersionConfig.hpp"

namespace pirate {
#if 0
}  // editor auto-indent
#endif

struct DedispersionPlan;  // defined in ./DedispersionPlan.hpp
struct CacheLineRingbuf;  // defined in ./internals/CacheLineRingbuf.hpp


// On a machine with multiple GPUs, you should make one Dedisperser per GPU.
// (It's okay if the DedispersionPlan is shared.)

struct Dedisperser
{
    const DedispersionConfig config;
    std::shared_ptr<DedispersionPlan> plan;
    std::shared_ptr<CacheLineRingbuf> cache_line_ringbuf;
    
    std::shared_ptr<char> host_buffer;
    std::shared_ptr<char> gpu_buffer;

    Dedisperser(const DedispersionConfig &config);

    // This alternate constructor is useful for sharing DedispersionPlans between Dedispersers.
    Dedisperser(const std::shared_ptr<DedispersionPlan> &plan);

    // Must call from thread whose current CUDA device is set appropriately!
    void allocate();

    void launch_h2g_copies(ssize_t chunk, int beam, cudaStream_t stream);
    void launch_g2h_copies(ssize_t chunk, int beam, cudaStream_t stream);
};


}  // namespace pirate

#endif // _PIRATE_DEDISPERSER_HPP
