#ifndef _PIRATE_ASSEMBLED_FRAME_HPP
#define _PIRATE_ASSEMBLED_FRAME_HPP

#include "SlabAllocator.hpp"

#include <ksgpu/Array.hpp>

#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <vector>

namespace pirate {
#if 0
}  // editor auto-indent
#endif


struct AssembledFrame
{
    long nfreq = 0;
    long ntime = 0;
    long beam_id = 0;
    long time_chunk_index = 0;   // 0, 1, 2, ...
    // time_chunk_index will be replaced by an FPGA count in the future (?)

    // dtype int4, shape (nfreq, ntime).
    // This array is allocated on memory returned from a SlabAllocator.
    ksgpu::Array<void> data;
};


struct AssembledFrameAllocator
{
    AssembledFrameAllocator(const std::shared_ptr<SlabAllocator> &slab_allocator, int num_consumers);

    long nfreq = 0;
    long time_samples_per_chunk = 0;
    std::vector<int> beam_ids;

    // Each consumer calls initalize() once, to establish the number of frequency channels,
    // number of time samples per chunk, and a list of beam IDs. If any of this data mismatches
    // between two consumers, an exception is thrown. In particular, each consumer must process
    // the same beam_ids in the same order.
    //
    // Consumers run in different threads, so everything needs to be thread-safe.

    void initialize(int consumer_id, long nfreq, long time_samples_per_chunk, const std::vector<int> &beam_ids);

    // Each consumer calls get_frame() in a loop.
    // Frames are returned in the following ordering:
    //
    //   for i in 0,1,2,...
    //      for j in 0,...,B-1,  where B=beam_ids.size()
    //          The (i*B+j)-th call to get_frame() returns time_chunk_index=i, beam_id=beam_ids[j]
    //
    // All consumers receive the same sequence of frames, and corresponding frames are
    // shared: the N-th call to get_frame() returns the same shared_ptr<AssembledFrame>
    // for all consumers (not a copy).
    //
    // The allocator holds a reference to each frame until all consumers have received it.
    // When the last consumer receives a frame, the allocator drops its reference. (If all
    // consumers have also dropped their references, then the frame is deallocated.)
    //
    // Frame memory is allocated from 'slab_allocator', with nbytes = (nfreq * time_samples_per_chunk) / 2.

    std::shared_ptr<AssembledFrame> get_frame(int consumer_id);

private:
    std::shared_ptr<SlabAllocator> slab_allocator;
    int num_consumers;
    
    mutable std::mutex lock;
    
    // Per-consumer state
    std::vector<bool> is_initialized;      // has consumer called initialize()?
    std::vector<long> consumer_next_index; // next sequence index for each consumer
    int num_initialized = 0;               // count of consumers that have initialized
    
    // Queue of frames: (frame, num_consumers_received)
    std::deque<std::pair<std::shared_ptr<AssembledFrame>, int>> frame_queue;
    long queue_start_index = 0;  // sequence index of first frame in queue
    
    // Helper to create a new frame for the given sequence index.
    std::shared_ptr<AssembledFrame> create_frame(long seq_index);
};


}  // namespace pirate

#endif // _PIRATE_ASSEMBLED_FRAME_HPP
