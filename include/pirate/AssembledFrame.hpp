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
    //  - The first call to get_frame() returns a frame with time_chunk_index=0 and beam_id=beam_ids[0].
    //  - The next (nbeams-1) calls return frames with time_chunk_index=0 and beam_ids[1:]
    //  - The next (nbeams) calls return time_chunk_index=1,
    //     etc.
    //
    // Thus, each consumer receives a sequence of frames with a preset pattern of time and beam indices.
    //
    // IMPORANT: at a given point in the sequence, the frames received by the different consumers must be 
    // shared_ptrs to the same AssembledFrame object. That is, all consumers get the same AssembledFrame
    // on the first call (time_index=0, beam_ids[0]). On the second call, the AssembledFrame is different
    // from the first call, but the same for all consumers (and so on for the third, fourth, ... call).
    //
    // To make this work, the AssembledFrameAllocator is responsible for keeping an internal queue of
    // AssembledFrames that have been sent to one consumer, but not yet sent to all consumers. When an
    // AssembledFrame exits the queue (i.e. when it is sent to the last consumer), its shared_ptr reference
    // is dropped (no longer held by the AssembledFrameAllocator).
    //
    // The AssembledFrameAllocator creates new frames with 
    //   auto frame = make_shared<AssembledFrame>();
    //
    // and initializes frame->data so that it points to a slab from 'slab_allocator' with
    // nbytes = (nfreq * time_samples_per_chunk) / 2.

    std::shared_ptr<AssembledFrame> get_frame(int consumer_id);

private:
    std::shared_ptr<SlabAllocator> slab_allocator;
    int num_consumers;
    
    mutable std::mutex lock;
    std::condition_variable cv;
    
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
