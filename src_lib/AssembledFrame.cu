#include "../include/pirate/AssembledFrame.hpp"

#include <ksgpu/xassert.hpp>

#include <sstream>
#include <stdexcept>

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// -------------------------------------------------------------------------------------------------
//
// Constructor


AssembledFrameAllocator::AssembledFrameAllocator(const std::shared_ptr<SlabAllocator> &slab_allocator_, int num_consumers_)
    : slab_allocator(slab_allocator_), num_consumers(num_consumers_)
{
    if (!slab_allocator) {
        throw std::runtime_error("AssembledFrameAllocator: slab_allocator is null");
    }
    if (num_consumers <= 0) {
        std::stringstream ss;
        ss << "AssembledFrameAllocator: num_consumers=" << num_consumers << " must be positive";
        throw std::runtime_error(ss.str());
    }
    
    is_initialized.resize(num_consumers, false);
    consumer_next_index.resize(num_consumers, 0);
}


// -------------------------------------------------------------------------------------------------
//
// initialize()


void AssembledFrameAllocator::initialize(int consumer_id, long nfreq_, long time_samples_per_chunk_, const std::vector<int> &beam_ids_)
{
    // Validate consumer_id
    if ((consumer_id < 0) || (consumer_id >= num_consumers)) {
        std::stringstream ss;
        ss << "AssembledFrameAllocator::initialize(): consumer_id=" << consumer_id
           << " is out of range [0, " << num_consumers << ")";
        throw std::runtime_error(ss.str());
    }
    
    // Validate arguments
    if (nfreq_ <= 0) {
        std::stringstream ss;
        ss << "AssembledFrameAllocator::initialize(): nfreq=" << nfreq_ << " must be positive";
        throw std::runtime_error(ss.str());
    }
    if (time_samples_per_chunk_ <= 0) {
        std::stringstream ss;
        ss << "AssembledFrameAllocator::initialize(): time_samples_per_chunk=" << time_samples_per_chunk_ << " must be positive";
        throw std::runtime_error(ss.str());
    }
    if (beam_ids_.empty()) {
        throw std::runtime_error("AssembledFrameAllocator::initialize(): beam_ids must be non-empty");
    }
    
    std::unique_lock<std::mutex> guard(lock);
    
    // Check for double initialization
    if (is_initialized[consumer_id]) {
        std::stringstream ss;
        ss << "AssembledFrameAllocator::initialize(): consumer_id=" << consumer_id << " already initialized";
        throw std::runtime_error(ss.str());
    }
    
    if (num_initialized == 0) {
        // First consumer: establish parameters
        nfreq = nfreq_;
        time_samples_per_chunk = time_samples_per_chunk_;
        beam_ids = beam_ids_;
    }
    else {
        // Subsequent consumers: validate parameters match
        if (nfreq_ != nfreq) {
            std::stringstream ss;
            ss << "AssembledFrameAllocator::initialize(): consumer_id=" << consumer_id
               << " has nfreq=" << nfreq_ << ", expected " << nfreq;
            throw std::runtime_error(ss.str());
        }
        if (time_samples_per_chunk_ != time_samples_per_chunk) {
            std::stringstream ss;
            ss << "AssembledFrameAllocator::initialize(): consumer_id=" << consumer_id
               << " has time_samples_per_chunk=" << time_samples_per_chunk_
               << ", expected " << time_samples_per_chunk;
            throw std::runtime_error(ss.str());
        }
        if (beam_ids_ != beam_ids) {
            std::stringstream ss;
            ss << "AssembledFrameAllocator::initialize(): consumer_id=" << consumer_id
               << " has mismatched beam_ids";
            throw std::runtime_error(ss.str());
        }
    }
    
    is_initialized[consumer_id] = true;
    num_initialized++;
}


// -------------------------------------------------------------------------------------------------
//
// create_frame(): helper to create a new frame for a given sequence index


std::shared_ptr<AssembledFrame> AssembledFrameAllocator::create_frame(long seq_index)
{
    // Must be called with lock held.
    // Assumes initialization is complete.
    
    long nbeams = beam_ids.size();
    long time_chunk_index = seq_index / nbeams;
    int beam_index = seq_index % nbeams;
    
    auto frame = std::make_shared<AssembledFrame>();
    frame->nfreq = nfreq;
    frame->ntime = time_samples_per_chunk;
    frame->beam_id = beam_ids[beam_index];
    frame->time_chunk_index = time_chunk_index;
    
    // Allocate data using slab allocator.
    // int4 dtype means 4 bits per element, so nbytes = (nfreq * ntime) / 2.
    long nbytes = (nfreq * time_samples_per_chunk) / 2;
    std::shared_ptr<void> slab = slab_allocator->get_slab(nbytes, /*blocking=*/true);
    
    // Initialize ksgpu::Array<void> manually.
    frame->data.data = slab.get();
    frame->data.ndim = 2;
    frame->data.shape[0] = nfreq;
    frame->data.shape[1] = time_samples_per_chunk;
    frame->data.size = nfreq * time_samples_per_chunk;
    frame->data.strides[0] = time_samples_per_chunk;
    frame->data.strides[1] = 1;
    frame->data.dtype = ksgpu::Dtype(ksgpu::df_int, 4);
    frame->data.aflags = slab_allocator->aflags;
    frame->data.base = slab;
    frame->data.check_invariants("AssembledFrameAllocator::create_frame()");
    
    return frame;
}


// -------------------------------------------------------------------------------------------------
//
// get_frame()


std::shared_ptr<AssembledFrame> AssembledFrameAllocator::get_frame(int consumer_id)
{
    // Validate consumer_id
    if ((consumer_id < 0) || (consumer_id >= num_consumers)) {
        std::stringstream ss;
        ss << "AssembledFrameAllocator::get_frame(): consumer_id=" << consumer_id
           << " is out of range [0, " << num_consumers << ")";
        throw std::runtime_error(ss.str());
    }
    
    std::unique_lock<std::mutex> guard(lock);
    
    // Check that this consumer has been initialized
    if (!is_initialized[consumer_id]) {
        std::stringstream ss;
        ss << "AssembledFrameAllocator::get_frame(): consumer_id=" << consumer_id
           << " has not called initialize()";
        throw std::runtime_error(ss.str());
    }
    
    // Get this consumer's next sequence index
    long seq_index = consumer_next_index[consumer_id];
    long queue_pos = seq_index - queue_start_index;
    
    xassert(queue_pos >= 0);
    
    // Extend queue if needed (create new frames)
    while (queue_pos >= (long)frame_queue.size()) {
        long new_seq_index = queue_start_index + frame_queue.size();
        auto frame = create_frame(new_seq_index);
        frame_queue.push_back({frame, 0});
    }
    
    // Get frame and increment receive count
    auto &[frame, num_received] = frame_queue[queue_pos];
    num_received++;
    
    // Increment this consumer's sequence index
    consumer_next_index[consumer_id]++;
    
    // Pop frames from front that all consumers have received
    while (!frame_queue.empty() && (frame_queue.front().second == num_consumers)) {
        frame_queue.pop_front();
        queue_start_index++;
    }
    
    return frame;
}


}  // namespace pirate

