#ifndef _PIRATE_CUDA_EVENT_RINGBUF_HPP
#define _PIRATE_CUDA_EVENT_RINGBUF_HPP

#include <string>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <cuda_runtime.h>

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// CudaEventRingbuf: A thread-safe ring buffer for (cuda event, seq_id) pairs.
//
// For each seq_id=0,1,..., a cuda event is produced once and consumed 'nconsumers'
// times. This class is useful for synchronizing GPU work between a producer stream
// and one or more consumer streams.
//
// Producer:
//   - record(stream, seq_id): Records a cuda event from 'stream' for the given seq_id.
//     The seq_id argument must be 0,1,2,... in the first, second, ... call.
//
// Consumer:
//   - wait(stream, seq_id, blocking): Makes 'stream' wait on the event for seq_id.
//   - synchronize(seq_id): Makes the calling thread wait on the event for seq_id.
//
// Each seq_id can be consumed at most 'nconsumers' times. If consumed more, an
// exception is thrown. If the ring buffer exceeds max_size, an exception is thrown.
//
// Special case nconsumers=0: wait() and synchronize() throw exceptions, but
// record() and synchronize_with_producer() are allowed. No cuda events are
// allocated. This is useful for pure producer-consumer synchronization without
// GPU event overhead.
//
// CudaEventRingbuf is noncopyable.


struct CudaEventRingbuf
{
    // Constructor.
    //   - name: Used in error messages.
    //   - nconsumers: Number of times each event can be consumed (0 is allowed, see above).
    //   - max_size: Maximum ring buffer size (throws if exceeded).
    //   - blocking_sync: If true, then synchronize() will block instead of busy-waiting.
    
    CudaEventRingbuf(const std::string &name, int nconsumers, 
                     long max_size=1000, bool blocking_sync = true);
    
    ~CudaEventRingbuf();

    // Noncopyable.
    CudaEventRingbuf(const CudaEventRingbuf &) = delete;
    CudaEventRingbuf &operator=(const CudaEventRingbuf &) = delete;

    // Producer: records a cuda event from 'stream', and saves it in the ring buffer.
    // The seq_id argument is expected to be 0,1,2,... in the first, second, ... call.
    void record(cudaStream_t stream, long seq_id);

    // Consumer: retrieve event from ringbuf, and call cudaStreamWaitEvent(stream, event).
    //
    // If the seq_id has not yet been produced (via record()):
    //   - If blocking=false (default), throws an exception.
    //   - If blocking=true, the calling thread blocks until another thread produces the seq_id.

    void wait(cudaStream_t stream, long seq_id, bool blocking = false);

    // Consumer: retrieve event from ringbuf, and call cudaEventSynchronize(event).
    //
    // If the seq_id has not yet been produced (via record()):
    //   - If blocking=false (default), throws an exception.
    //   - If blocking=true, the calling thread blocks until another thread produces the seq_id.

    void synchronize(long seq_id, bool blocking = false);

    // Blocks calling thread until an event with specified seq_id has been produced.
    // Does not consume an event, or modify the ringbuf.
    void synchronize_with_producer(long seq_id);

    // ----- Internals -----

    std::string name;
    int nconsumers = 1;
    long max_size = 1000;
    bool blocking_sync = true;
    unsigned int event_flags = 0;

    // Ring buffer state (all protected by mutex).
    // - 'events' is a fixed-size array of cuda events (size == max_size).
    // - 'produced[slot]' is true iff cudaEventRecord() has completed for the current seq_id at that slot.
    // - 'acquired[slot]' is the number of consumers who have acquired events[slot] for the current seq_id.
    // - 'released[slot]' is the number of consumers who have released events[slot] for the current seq_id.
    // - An event at seq_id is stored at slot (seq_id % max_size), and is valid iff
    //   seq_start <= seq_id < seq_end AND produced[slot] is true.
    // - The slot can be recycled (for a new seq_id) only when released[slot] == nconsumers,
    //   ensuring no consumer is still using the event.
    
    std::vector<cudaEvent_t> events;
    std::vector<bool> produced;  // per-slot: true iff cudaEventRecord() has completed
    std::vector<int> acquired;   // per-slot acquisition count (protected by mutex)
    std::vector<int> released;   // per-slot release count (protected by mutex)
    
    std::mutex mutex;
    std::condition_variable cv;  // signaled when seq_start/seq_end/produced changes
    long seq_start = 0;          // lowest valid seq_id in ring buffer
    long seq_end = 0;            // one past highest valid seq_id (next seq_id to be recorded)
    
    // Helper: acquire event at seq_id for consumption.
    // If blocking=false and seq_id not yet produced, throws an exception.
    // If blocking=true and seq_id not yet produced, waits for it.
    // Throws if seq_id is stale (already recycled) or over-consumed.
    cudaEvent_t _acquire(long seq_id, bool blocking);
    
    // Helper: release event at seq_id after CUDA call is complete.
    // If all nconsumers have released, advances seq_start to recycle the slot.
    void _release(long seq_id);
};


}  // namespace pirate

#endif // _PIRATE_CUDA_EVENT_RINGBUF_HPP
