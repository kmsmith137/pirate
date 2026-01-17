#include "../include/pirate/AssembledFrame.hpp"
#include "../include/pirate/inlines.hpp"  // xdiv()

#include <ksgpu/xassert.hpp>
#include <ksgpu/mem_utils.hpp>

#include <sstream>
#include <stdexcept>

using namespace std;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// -------------------------------------------------------------------------------------------------
//
// Constructor


AssembledFrameAllocator::AssembledFrameAllocator(const shared_ptr<SlabAllocator> &slab_allocator_, int num_consumers_)
    : slab_allocator(slab_allocator_), num_consumers(num_consumers_), is_dummy_mode(slab_allocator_->is_dummy())
{
    xassert(slab_allocator);
    xassert_gt(num_consumers, 0);

    if (!ksgpu::af_on_host(slab_allocator->aflags))
        throw runtime_error("AssembledFrameAllocator: slab_allocator must be on host");
    
    is_initialized.resize(num_consumers, false);
    consumer_next_index.resize(num_consumers, 0);
    
    // Spawn worker thread if not in dummy mode.
    if (!is_dummy_mode)
        worker_thread = thread(&AssembledFrameAllocator::worker_main, this);
}


// -------------------------------------------------------------------------------------------------
//
// Destructor


AssembledFrameAllocator::~AssembledFrameAllocator()
{
    this->stop();
    if (worker_thread.joinable())
        worker_thread.join();
}


// -------------------------------------------------------------------------------------------------
//
// stop()


void AssembledFrameAllocator::stop(exception_ptr e)
{
    unique_lock<mutex> guard(lock);

    if (is_stopped)
        return;

    error = e;
    is_stopped = true;
    
    guard.unlock();
    cv.notify_all();

    // Propagate stop to SlabAllocator (wakes up worker thread if blocked in get_slab).
    slab_allocator->stop(e);
}


// -------------------------------------------------------------------------------------------------
//
// _throw_if_stopped(): helper for entry points. Caller must hold lock.


void AssembledFrameAllocator::_throw_if_stopped(const char *method_name)
{
    if (error)
        rethrow_exception(error);
    
    if (is_stopped)
        throw runtime_error(string(method_name) + " called on stopped instance");
}


// -------------------------------------------------------------------------------------------------
//
// Worker thread (non-dummy mode only)
//
// The worker thread pre-initializes frames and pushes them directly to frame_queue.
// This reduces latency for callers of get_frame(), since the memset() is already done.


void AssembledFrameAllocator::_worker_main()
{
    unique_lock<mutex> guard(lock);
        
    // Wait until: stopped, or initialization is complete.
    // We need nfreq, time_samples_per_chunk, beam_ids to create frames.
    while (!is_stopped && (num_initialized == 0))
        cv.wait(guard);
        
    // Main loop: create frames and add directly to frame_queue.
    while (!is_stopped)
        _create_frame(guard);
}


void AssembledFrameAllocator::worker_main()
{
    try {
        _worker_main();  // only returns if is_stopped
    } catch (...) {
        stop(current_exception());
    }
}


// -------------------------------------------------------------------------------------------------
//
// Create a new frame and add it to the queue.
// Caller must currently hold lock via 'guard'.
// The lock will be dropped and re-acquired.


void AssembledFrameAllocator::_create_frame(unique_lock<mutex> &guard)
{
    xassert(num_initialized > 0);

    // If two threads are creating a frame simultaneously, then something is wrong.
    xassert(!frame_creation_underway);
    frame_creation_underway = true;

    // Drop lock while allocating.
    guard.unlock();

    // Allocate data using slab allocator.
    // int4 dtype means 4 bits per element, so nbytes = (nfreq * ntime) / 2.
    // Array is initialized to all (-8) values.
    long nbytes = nfreq * xdiv(time_samples_per_chunk, 2);
    shared_ptr<void> slab = slab_allocator->get_slab(nbytes, /*blocking=*/true);
    memset(slab.get(), 0x88, nbytes);

    // Note that frame->{beam_id, time_chunk_index} are initialized later.
    auto frame = make_shared<AssembledFrame>();
    frame->nfreq = nfreq;
    frame->ntime = time_samples_per_chunk;
    
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
    frame->data.check_invariants("AssembledFrameAllocator::create_frame_unlocked()");
    
    guard.lock();

    // Get seq_index at end of queue, after acquiring the lock 
    // (in case seq_index changed while the lock was dropped).

    long nbeams = beam_ids.size();
    long seq_index = queue_start_index + frame_queue.size();

    frame->time_chunk_index = seq_index / nbeams;
    frame->beam_id = beam_ids[seq_index % nbeams];

    this->frame_queue.push_back({frame, 0});
    this->frame_creation_underway = false;

    cv.notify_all();  // Wake up any waiting get_frame() callers
}


// -------------------------------------------------------------------------------------------------
//
// initialize() - Entry point


void AssembledFrameAllocator::initialize(int consumer_id, long nfreq_, long time_samples_per_chunk_, const vector<long> &beam_ids_)
{
    {
        unique_lock<mutex> guard(lock);
        _throw_if_stopped("AssembledFrameAllocator::initialize");
    }
    
    try {
        _initialize(consumer_id, nfreq_, time_samples_per_chunk_, beam_ids_);
    } catch (...) {
        this->stop(current_exception());
        throw;
    }
}


void AssembledFrameAllocator::_initialize(int consumer_id, long nfreq_, long time_samples_per_chunk_, const vector<long> &beam_ids_)
{
    xassert_ge(consumer_id, 0);
    xassert_lt(consumer_id, num_consumers);
    xassert_gt(nfreq_, 0L);
    xassert_gt(time_samples_per_chunk_, 0L);
    xassert(!beam_ids_.empty());
    
    unique_lock<mutex> guard(lock);
    
    // Check for double initialization
    if (is_initialized[consumer_id]) {
        stringstream ss;
        ss << "AssembledFrameAllocator::initialize(): consumer_id=" << consumer_id << " already initialized";
        throw runtime_error(ss.str());
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
            stringstream ss;
            ss << "AssembledFrameAllocator::initialize(): consumer_id=" << consumer_id
               << " has nfreq=" << nfreq_ << ", expected " << nfreq;
            throw runtime_error(ss.str());
        }
        if (time_samples_per_chunk_ != time_samples_per_chunk) {
            stringstream ss;
            ss << "AssembledFrameAllocator::initialize(): consumer_id=" << consumer_id
               << " has time_samples_per_chunk=" << time_samples_per_chunk_
               << ", expected " << time_samples_per_chunk;
            throw runtime_error(ss.str());
        }
        if (beam_ids_ != beam_ids) {
            stringstream ss;
            ss << "AssembledFrameAllocator::initialize(): consumer_id=" << consumer_id
               << " has mismatched beam_ids";
            throw runtime_error(ss.str());
        }
    }
    
    is_initialized[consumer_id] = true;
    num_initialized++;
    
    // Notify worker thread that initialization is complete.
    cv.notify_all();
}


// -------------------------------------------------------------------------------------------------
//
// get_frame() - Entry point


shared_ptr<AssembledFrame> AssembledFrameAllocator::get_frame(int consumer_id)
{
    {
        unique_lock<mutex> guard(lock);
        _throw_if_stopped("AssembledFrameAllocator::get_frame");
    }
    
    try {
        return _get_frame(consumer_id);
    } catch (...) {
        this->stop(current_exception());
        throw;
    }
}


shared_ptr<AssembledFrame> AssembledFrameAllocator::_get_frame(int consumer_id)
{
    xassert_ge(consumer_id, 0);
    xassert_lt(consumer_id, num_consumers);
    
    unique_lock<mutex> guard(lock);
    
    // Check that this consumer has been initialized
    if (!is_initialized[consumer_id]) {
        stringstream ss;
        ss << "AssembledFrameAllocator::get_frame(): consumer_id=" << consumer_id
           << " has not called initialize()";
        throw runtime_error(ss.str());
    }

    for (;;) {
        if (is_stopped)
            _throw_if_stopped("AssembledFrameAllocator::get_frame");

        // Get this consumer's next sequence index.
        // Note that we re-read this data in each iteration of the for-loop,
        // in case values changed while we dropped the lock.

        long seq_index = consumer_next_index[consumer_id];
        long queue_pos = seq_index - queue_start_index; 
        long queue_size = frame_queue.size();

        xassert(queue_pos >= 0);
        xassert(queue_pos <= queue_size);

        if (queue_pos >= queue_size) {
            // Frame is not in queue. If another thread is currently creating a new frame, then
            // wait for that thread to finish. Otherwise, current thread creates a new frame.

            if (!is_dummy_mode || frame_creation_underway)
                cv.wait(guard);
            else
                _create_frame(guard);

            // Back to top of loop, since lock was dropped in either cv.wait() or _create_frame().
            continue;
        }

        // Get frame and increment receive count.
        auto &[frame_ref, num_received] = frame_queue[queue_pos];
        num_received++;
        
        // IMPORTANT: Make a copy of the shared_ptr BEFORE popping from queue!
        // frame_ref becomes a dangling reference after pop_front().
        auto result = frame_ref;
    
        // Increment this consumer's sequence index, and update first_unreceived_index.
        // (first_unreceived_index is the max of all consumer_next_index values.)
        consumer_next_index[consumer_id]++;
        if (consumer_next_index[consumer_id] > first_unreceived_index) {
            first_unreceived_index = consumer_next_index[consumer_id];
            cv.notify_all();  // Wake up block_until_low_memory() if waiting
        }
    
        // Pop frames from front that all consumers have received.
        while (!frame_queue.empty() && (frame_queue.front().second == num_consumers)) {
            frame_queue.pop_front();
            queue_start_index++;
        }
    
        return result;
    }
}


// -------------------------------------------------------------------------------------------------
//
// num_free_frames() and num_total_frames()


long AssembledFrameAllocator::num_free_frames() const
{
    // Number of pre-initialized frames waiting for first consumer.
    unique_lock<mutex> guard(lock);
    long queue_end_index = queue_start_index + frame_queue.size();
    long num_preinitialized = queue_end_index - first_unreceived_index;
    guard.unlock();    

    return num_preinitialized + slab_allocator->num_free_slabs();
}


long AssembledFrameAllocator::num_total_frames() const
{
    return slab_allocator->num_total_slabs();
}


// -------------------------------------------------------------------------------------------------
//
// block_until_low_memory() - Entry point


void AssembledFrameAllocator::block_until_low_memory(long nframe_threshold)
{
    try {
        _block_until_low_memory(nframe_threshold);
    } catch (...) {
        this->stop(current_exception());
        throw;
    }
}


void AssembledFrameAllocator::_block_until_low_memory(long nframe_threshold)
{
    unique_lock<mutex> guard(lock);

    for (;;) {
        _throw_if_stopped("AssembledFrameAllocator::block_until_low_memory");
        guard.unlock();

        // Block until slab allocator is empty (all slabs allocated).
        slab_allocator->block_until_empty();
        
        // Check num_preinitialized under lock.
        guard.lock();
        _throw_if_stopped("AssembledFrameAllocator::block_until_low_memory");
        
        long queue_end_index = queue_start_index + frame_queue.size();
        long num_preinitialized = queue_end_index - first_unreceived_index;
        
        if (num_preinitialized <= nframe_threshold)
            return;
        
        // Wait for something to change (either num_preinitialized decreases,
        // or a slab is returned and a new frame is created).
        cv.wait(guard);
    }
}


}  // namespace pirate
