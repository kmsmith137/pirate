#ifndef _PIRATE_ASSEMBLED_FRAME_HPP
#define _PIRATE_ASSEMBLED_FRAME_HPP

#include "SlabAllocator.hpp"

#include <ksgpu/Array.hpp>

#include <mutex>
#include <deque>
#include <vector>
#include <thread>
#include <memory>
#include <exception>
#include <filesystem>
#include <condition_variable>


namespace pirate {
#if 0
}  // editor auto-indent
#endif


struct AssembledFrame
{
    // Initialized at creation, not protected by lock.
    long nfreq = 0;
    long ntime = 0;
    long beam_id = 0;
    long time_chunk_index = 0;   // 0, 1, 2, ...
    // time_chunk_index will be replaced by an FPGA count in the future (?)

    // dtype int4, shape (nfreq, ntime).
    // Newly allocated frames have all 'data' elements set to (-8).
    //
    // Warning: if the AssembledFrame has been "reaped" under memory pressure,
    // then 'data' is an empty array. The array state (empty vs nonempty) is
    // protected by the lock, but the array contents are not lock-protected.

    ksgpu::Array<void> data;

    // ASDF file I/O.
    // Note: int4 dtype is stored as uint8 with shape (nfreq, ntime/2).
    //
    // Note: if sync=true in write_asdf(), then we fsync() the file at the end.
    // This is the default, to avoid runaway page cache usage. I tested that it
    // still gives full ssd bandwith, if multiple writer threads are used.
    //
    // A future project: faster asdf?
    //
    //  - Currently, memory bandwidth of AssembledFrame::write_asdf() is 4-5x the memory
    //    footprint. (Empirical, with 'pirate_frb hwtest configs/hwtest/cf00_asdf.yml'
    //    and 'pcm-memory'.) This is potentially an issue (see 'misc/plan_hmem_bw.py').
    //
    //  - For binary "blobs", writing with open(..., O_DIRECT) -> write() is 1x the
    //    footprint, and does not pollute the page cache, which seems important in
    //    the larger footprint.
    //
    //  - Achieving this performance in a general purpose asdf implementation seems
    //    challenging due to dual alignment requirements (O_DIRECT requires all writes
    //    to be logically aligned on 4k boundaries within the file, and pointers to be
    //    4k-aligned).
    //
    //  - Idea: write a short standalone asdf writer which is integrated with the
    //    AssembledFrameAllocator (see below). The allocator uses a single memory
    //    slab, which is both "backing" memory for the C++ arrays in the real-time
    //    server, and "backing" memory for the asdf file, so that the file can be
    //    written with a single call to write().
    //
    //  - Layout is nontrivial -- need to leave space for the yaml header, and gaps
    //    between the arrays. There are some mild alignment requirements (C++ code
    //    generally assumes that Arrays are 128-byte aligned, and O_DIRECT assumes
    //    that the base pointer is page-aligned. I think that the asdf format is
    //    flexible enough that these alignment requirements are not a problem.
    //
    //  - Note: according to google gemini, to determine the O_DIRECT page alignment
    //    requirement, you should call ioctl(fd, BLKPBSZGET), not BLKSSZGET.
    
    void write_asdf(const std::string &filename, bool sync=true) const;
    static std::shared_ptr<AssembledFrame> from_asdf(const std::string &filename);    // Call without lock held.

    // --------------------  private/internal  --------------------
    
    // Call with lock held!
    // Called by reaper thread, or ssd writer thread.
    void _reap_locked();

    // Create a random AssembledFrame (for testing). Note: ntime must be even.
    static std::shared_ptr<AssembledFrame> make_random(long nfreq, long ntime, long beam_id, long time_chunk_index);
    static std::shared_ptr<AssembledFrame> make_random();  // randomizes all parameters
    
    // Unit test for ASDF I/O.
    static void test_asdf();
    
    // Members after this point are internal state.
    // These members are protected by the mutex, and are not saved to the ASDF file.

    mutable std::mutex mutex;
    long finalize_count = 0;    // incremented by FrbServer worker thread(s)

    // NOTE: all save_paths received from RPC clients MUST be validated with
    // pirate::is_safe_relpath(). If this check fails, then an error is returned in
    // the write request, and the save_path must not be added to the AssembledFrame.

    std::vector<std::filesystem::path> save_paths;

    // Save state. These members are protected by the lock, and can only be modified
    // by member functions of 'class FileWriter'. In particular, when an RPC thread
    // wants to add a save_path, it should acquire the lock, call save_paths.push_back(),
    // drop the lock, and call FileWriter::process_frame().

    std::exception_ptr save_error = nullptr;
    bool in_ssd_queue = false;
    bool in_nfs_queue = false;
    bool on_ssd = false;
    long nfs_count = 0;
};


// AssembledFrameAllocator: Thread-backed class that allocates AssembledFrames.
//
// In non-dummy mode, a worker thread pre-initializes frames (calls memset to fill
// with 0x88) to reduce latency for callers of get_frame(). The worker thread
// inherits its vcpu affinity from the caller of the constructor. Python callers
// should call the AssembledFrameAllocator constructor within a ThreadAffinity
// context manager.
//
// In dummy mode (slab_allocator->is_dummy()), no worker thread is created, and
// frames are initialized synchronously by the caller of get_frame().
//
// Entry points (following thread-backed class pattern): initialize(), get_frame().
// These will throw if the allocator is stopped, and will call stop() on exception.

struct AssembledFrameAllocator
{
    AssembledFrameAllocator(const std::shared_ptr<SlabAllocator> &slab_allocator, int num_consumers);
    ~AssembledFrameAllocator();

    // Non-copyable, non-movable (required for thread-backed class pattern).
    AssembledFrameAllocator(const AssembledFrameAllocator &) = delete;
    AssembledFrameAllocator &operator=(const AssembledFrameAllocator &) = delete;
    AssembledFrameAllocator(AssembledFrameAllocator &&) = delete;
    AssembledFrameAllocator &operator=(AssembledFrameAllocator &&) = delete;

    long nfreq = 0;
    long time_samples_per_chunk = 0;
    std::vector<long> beam_ids;

    // Each consumer calls initalize() once, to establish the number of frequency channels,
    // number of time samples per chunk, and a list of beam IDs. If any of this data mismatches
    // between two consumers, an exception is thrown. In particular, each consumer must process
    // the same beam_ids in the same order.
    //
    // Consumers run in different threads, so everything needs to be thread-safe.
    //
    // Entry point: throws if stopped, calls stop() on exception.

    void initialize(int consumer_id, long nfreq, long time_samples_per_chunk, const std::vector<long> &beam_ids);

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
    //
    // Entry point: throws if stopped, calls stop() on exception.

    std::shared_ptr<AssembledFrame> get_frame(int consumer_id);

    // Returns the number of "available" frames: pre-initialized frames waiting for their first
    // consumer, plus free slabs in the underlying slab_allocator.
    // If permissive=false (default): throws exception in dummy mode or if not initialized.
    // If permissive=true: returns 0 in dummy mode or if not initialized.
    long num_free_frames(bool permissive = false) const;

    // Entry point: Block until slab allocator is empty (all slabs in use), AND the number of
    // pre-initialized frames waiting for first consumer is <= nframe_threshold.
    // Throws exception in dummy mode, or if stop() is called from another thread.
    void block_until_low_memory(long nframe_threshold);

    // Returns the total number of frames (same as num_total_slabs() from the underlying slab_allocator).
    // Throws exception in dummy mode.
    // If blocking=false (default) and not initialized, throws exception.
    // If blocking=true and not initialized, blocks until initialized.
    long num_total_frames(bool blocking = false) const;

    // Returns true if in dummy mode.
    bool is_dummy() const { return is_dummy_mode; }

    // Stop the allocator. After calling stop(), entry points will throw.
    // If 'e' is non-null, it represents an error; if null, it's normal termination.
    // Thread-safe; first call sets the error.
    void stop(std::exception_ptr e = nullptr);

private:
    std::shared_ptr<SlabAllocator> slab_allocator;
    int num_consumers;
    bool is_dummy_mode;  // cached from slab_allocator->is_dummy()
    
    mutable std::mutex lock;
    std::condition_variable cv;  // signaled on: stop, frame added to queue, frame received by all consumers
    
    // Thread-backed class pattern: stopped state and error
    bool is_stopped = false;
    std::exception_ptr error;
    std::thread worker_thread;
    
    // Per-consumer state
    std::vector<bool> is_initialized;      // has consumer called initialize()?
    std::vector<long> consumer_next_index; // next sequence index for each consumer
    int num_initialized = 0;               // count of consumers that have initialized
    
    // Queue of frames: (frame, num_consumers_received).
    // In non-dummy mode, the worker thread pushes pre-initialized frames to the back.
    // In dummy mode, get_frame() creates frames synchronously and pushes to the back.
    // Frames are popped from the front when all consumers have received them.
    //
    // The early part of the queue contains frames with (num_consumers_received > 0) that
    // are waiting for subsequent consumers. The later part contains frames with
    // (num_consumers_received == 0) that have been pre-initialized, waiting for their
    // first consumer. The boundary between these regions is at first_unreceived_index.
    std::deque<std::pair<std::shared_ptr<AssembledFrame>, int>> frame_queue;
    long queue_start_index = 0;             // sequence index of first frame in queue
    long first_unreceived_index = 0;        // sequence index of first frame not yet received by any consumer
    bool frame_creation_underway = false;   // is there a thread in the process of creating a new frame?

    // Create a new frame and add it to the queue.
    // Caller must currently hold lock via 'guard'.
    // The lock will be dropped and re-acquired.
    void _create_frame(std::unique_lock<std::mutex> &lock);
    
    // Helper for entry points. Caller must hold lock.
    void _throw_if_stopped(const char *method_name);
    
    // Worker thread functions (non-dummy mode only).
    void _worker_main();
    void worker_main();
    
    // Internal implementations of entry points.
    void _initialize(int consumer_id, long nfreq, long time_samples_per_chunk, const std::vector<long> &beam_ids);
    std::shared_ptr<AssembledFrame> _get_frame(int consumer_id);
    void _block_until_low_memory(long nframe_threshold);
};


}  // namespace pirate

#endif // _PIRATE_ASSEMBLED_FRAME_HPP
