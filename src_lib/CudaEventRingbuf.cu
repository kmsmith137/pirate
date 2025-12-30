#include "../include/pirate/CudaEventRingbuf.hpp"

#include <sstream>
#include <stdexcept>
#include <ksgpu/cuda_utils.hpp>
#include <ksgpu/xassert.hpp>

namespace pirate {
#if 0
}  // editor auto-indent
#endif


CudaEventRingbuf::CudaEventRingbuf(const std::string &name_, int nconsumers_, 
                                    long max_size_, bool blocking_sync_)
    : name(name_),
      nconsumers(nconsumers_),
      max_size(max_size_),
      blocking_sync(blocking_sync_)
{
    if (nconsumers < 0) {
        std::ostringstream ss;
        ss << "CudaEventRingbuf '" << name << "': nconsumers=" << nconsumers << " must be >= 0";
        throw std::runtime_error(ss.str());
    }

    if (max_size <= 0) {
        std::ostringstream ss;
        ss << "CudaEventRingbuf '" << name << "': max_size=" << max_size << " must be > 0";
        throw std::runtime_error(ss.str());
    }

    // Set up event flags.
    event_flags = cudaEventDisableTiming;
    if (blocking_sync)
        event_flags |= cudaEventBlockingSync;

    // Pre-allocate counters.
    produced.resize(max_size, false);
    acquired.resize(max_size, 0);
    released.resize(max_size, 0);
    
    // Pre-allocate events (only if nconsumers > 0).
    if (nconsumers > 0) {
        events.resize(max_size);
        for (long i = 0; i < max_size; i++)
            CUDA_CALL(cudaEventCreateWithFlags(&events[i], event_flags));
    }
}


CudaEventRingbuf::~CudaEventRingbuf()
{
    for (size_t i = 0; i < events.size(); i++)
        cudaEventDestroy(events[i]);
}


void CudaEventRingbuf::stop(std::exception_ptr e)
{
    std::unique_lock<std::mutex> lock(mutex);
    
    if (!is_stopped) {
        is_stopped = true;
        error = e;
    }
    
    lock.unlock();
    cv.notify_all();
}


void CudaEventRingbuf::_throw_if_stopped(const char *method_name)
{
    // Caller must hold mutex.
    if (!is_stopped)
        return;
    
    if (error)
        std::rethrow_exception(error);
    
    std::ostringstream ss;
    ss << "CudaEventRingbuf::" << method_name << "(): called on stopped instance '" << this->name << "'";
    throw std::runtime_error(ss.str());
}


void CudaEventRingbuf::record(cudaStream_t stream, long seq_id)
{
    try {
        // Get the slot for this event.
        long slot = seq_id % max_size;
        
        std::unique_lock<std::mutex> lock(mutex);
        
        _throw_if_stopped("record");
            
        // Check that seq_id matches expected value.
        if (seq_id != seq_end) {
            std::ostringstream ss;
            ss << "CudaEventRingbuf '" << name << "' record(): expected seq_id=" 
               << seq_end << ", got seq_id=" << seq_id;
            throw std::runtime_error(ss.str());
        }

        // Check ring buffer capacity.
        if (seq_end - seq_start >= max_size) {
            std::ostringstream ss;
            ss << "CudaEventRingbuf '" << name << "' record(): ring buffer overflow."
               << " Either max_size=" << max_size << " was too small, or nconsumers="
               << nconsumers << " is too large.";
            throw std::runtime_error(ss.str());
        }
            
        // Reset counters for the new seq_id at this slot.
        produced[slot] = false;
        acquired[slot] = 0;
        released[slot] = 0;
            
        // Reserve the slot by incrementing seq_end.
        seq_end++;
        
        // Record the event without holding the lock (only if nconsumers > 0).
        if (nconsumers > 0) {
            lock.unlock();
            CUDA_CALL(cudaEventRecord(events[slot], stream));
            lock.lock();
        }

        produced[slot] = true;
        
        // If nconsumers == 0, slots are immediately recyclable.
        if (nconsumers == 0)
            seq_start = seq_end;
        
        lock.unlock();
            
        // Wake up any threads waiting in synchronize_with_producer().
        cv.notify_all();
    }
    catch (...) {
        stop(std::current_exception());
        throw;
    }
}


cudaEvent_t CudaEventRingbuf::_acquire(long seq_id, bool blocking)
{
    if (nconsumers == 0) {
        std::ostringstream ss;
        ss << "CudaEventRingbuf '" << name << "': wait()/synchronize() called with nconsumers=0";
        throw std::runtime_error(ss.str());
    }

    long slot = seq_id % max_size;

    std::unique_lock<std::mutex> lock(mutex);
    
    for (;;) {
        _throw_if_stopped("wait/synchronize");
        
        // Check if seq_id is too old (already recycled). This would indicate
        // overconsumption, so use the same exception text as below.
        if (seq_id < seq_start) {
            std::ostringstream ss;
            ss << "CudaEventRingbuf '" << name << "': seq_id=" << seq_id 
               << " consumed more than nconsumers=" << nconsumers << " times";
            throw std::runtime_error(ss.str());
        }
    
        // Check if event is available (in range AND cudaEventRecord has completed).
        if ((seq_id < seq_end) && produced[slot])
            break;

        // Event not yet available.
        if (!blocking) {
            std::ostringstream ss;
            ss << "CudaEventRingbuf '" << name << "': seq_id=" << seq_id
               << " not yet produced (seq_end=" << seq_end
               << "), and blocking=false";
            throw std::runtime_error(ss.str());
        }
        
        cv.wait(lock);
    }

    // Check for over-consumption.
    if (acquired[slot] >= nconsumers) {
        std::ostringstream ss;
        ss << "CudaEventRingbuf '" << name << "': seq_id=" << seq_id 
           << " consumed more than nconsumers=" << nconsumers << " times";
        throw std::runtime_error(ss.str());
    }
    
    // Acquire the event.
    acquired[slot]++;
    
    return events[slot];
}


void CudaEventRingbuf::_release(long seq_id)
{
    long slot = seq_id % max_size;

    std::unique_lock<std::mutex> lock(mutex);
    
    // Some paranoid asserts.
    xassert(seq_id >= seq_start);
    xassert(seq_id < seq_end);
    xassert(released[slot] < nconsumers);

    // Release the event.
    released[slot]++;

    if (released[slot] < nconsumers)
        return;

    // Advance seq_start past all fully-released events.
    while (seq_start < seq_end) {
        long front_slot = seq_start % max_size;
            
        if (released[front_slot] < nconsumers)
            break;
            
        seq_start++;
    }
        
    // Wake up any waiting threads (producers waiting for capacity, or
    // consumers that might have been waiting on stale check).
    lock.unlock();
    cv.notify_all();
}


void CudaEventRingbuf::wait(cudaStream_t stream, long seq_id, bool blocking)
{
    if ((seq_id < 0) && (nconsumers > 0))
        return;

    try {
        cudaEvent_t event = _acquire(seq_id, blocking);
        CUDA_CALL(cudaStreamWaitEvent(stream, event, 0));
        _release(seq_id);
    }
    catch (...) {
        stop(std::current_exception());
        throw;
    }
}


void CudaEventRingbuf::synchronize(long seq_id, bool blocking)
{
    if ((seq_id < 0) && (nconsumers > 0))
        return;
    
    try {
        cudaEvent_t event = _acquire(seq_id, blocking);
        CUDA_CALL(cudaEventSynchronize(event));
        _release(seq_id);
    }
    catch (...) {
        stop(std::current_exception());
        throw;
    }
}


void CudaEventRingbuf::synchronize_with_producer(long seq_id)
{
    if (seq_id < 0)
        return;

    try {
        long slot = seq_id % max_size;

        std::unique_lock<std::mutex> lock(mutex);
        
        for (;;) {
            _throw_if_stopped("synchronize_with_producer");
            
            if (seq_id < seq_start)
                return;
            if ((seq_id < seq_end) && produced[slot])
                return;
            cv.wait(lock);
        }
    }
    catch (...) {
        stop(std::current_exception());
        throw;
    }
}


}  // namespace pirate
