#ifndef _PIRATE_SLAB_ALLOCATOR_HPP
#define _PIRATE_SLAB_ALLOCATOR_HPP

#include "constants.hpp"
#include "BumpAllocator.hpp"

#include <condition_variable>
#include <exception>
#include <memory>
#include <mutex>
#include <vector>

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// SlabAllocator: A thread-safe pool allocator for fixed-size memory "slabs".
// Stoppable class: see notes/stoppable_class.md.
//
// The allocator manages a large pre-allocated memory region, which is
// subdivided into fixed-size slabs. Slabs are returned to the pool when their
// reference count drops to zero, making them available for future allocations.
//
// Slab size is established by the first call to get_slab(). All subsequent calls
// must request the same size; otherwise an exception is thrown.
//
// It's okay if the slabs outlive the allocator (but memory isn't reclaimed until
// the last slab is destroyed).
//
// Key features:
//   - Thread-safe (uses a lock for the free list)
//   - Slabs hold a reference to the SlabAllocator, preventing the underlying
//     memory from being freed while any slab is still in use
//   - Compatible with weak_ptr (control blocks are allocated separately)
//   - Supports blocking mode: get_slab() can wait for a slab to be returned
//
// Dummy mode (capacity < 0):
//   - No base memory is pre-allocated
//   - get_slab() allocates fresh memory using af_alloc() for each request
//   - num_total_slabs() and num_free_slabs() throw exceptions


class SlabAllocator : public std::enable_shared_from_this<SlabAllocator>
{
public:
    static constexpr int nalign = constants::bytes_per_gpu_cache_line;
    
    // Factory method: create SlabAllocator that allocates new memory using af_alloc().
    // The 'aflags' are memory allocation flags from ksgpu/mem_utils.hpp.
    // If capacity < 0, operates in "dummy" mode (see class comment).
    static std::shared_ptr<SlabAllocator> create(int aflags, long capacity);
    
    // Factory method: create SlabAllocator that gets memory from an existing BumpAllocator.
    // The aflags are inherited from the BumpAllocator.
    // Throws exception if BumpAllocator is in dummy mode.
    //
    // If the BumpAllocator is async, the SlabAllocator is itself "async":
    // its constructor returns immediately, and the b->allocate_bytes() call
    // is deferred to the first get_slab(). Async-init failures from `b`
    // surface from either get_slab() or wait_until_initialized().
    static std::shared_ptr<SlabAllocator> create(const std::shared_ptr<BumpAllocator> &b, long nbytes);
    
    // Non-copyable, non-movable.
    SlabAllocator(const SlabAllocator &) = delete;
    SlabAllocator &operator=(const SlabAllocator &) = delete;
    SlabAllocator(SlabAllocator &&) = delete;
    SlabAllocator &operator=(SlabAllocator &&) = delete;

    // Allocate a slab of the specified size. The first call establishes the
    // slab size; all subsequent calls must use the same size.
    //
    // The returned shared_ptr holds a reference to the SlabAllocator, ensuring
    // the underlying memory is not freed while the slab is in use. When the
    // shared_ptr's reference count drops to zero, the slab is returned to the
    // pool for reuse.
    //
    // If blocking=false (default) and no slabs are available, throws an exception.
    // If blocking=true, waits until a slab becomes available.
    //
    // In dummy mode, always allocates fresh memory using af_alloc().
    std::shared_ptr<void> get_slab(long nbytes, bool blocking = false);
    
    // Returns the number of slabs currently available in the free list.
    // Throws exception in dummy mode or if not initialized.
    long num_free_slabs() const;
    
    // Returns the total number of slabs in the pool.
    // Throws exception in dummy mode.
    // If blocking=false (default) and not initialized, throws exception.
    // If blocking=true and not initialized, blocks until initialized.
    long num_total_slabs(bool blocking = false) const;
    
    // Returns the established slab size.
    // Throws exception if not initialized.
    long get_slab_size() const;
    
    // Returns true if the SlabAllocator is ready to serve get_slab() calls
    // without blocking on async init. Semantics:
    //   - Dummy mode (no underlying BumpAllocator): always true.
    //   - Non-dummy mode: delegates to bump_allocator->is_initialized().
    //
    // Note: does NOT check whether slab_size has been established (that's
    // user-pattern state, established on the first get_slab() call).
    bool is_initialized() const;
    
    // Returns true if in dummy mode (capacity < 0).
    bool is_dummy() const { return capacity < 0; }
    
    // Block until there are no free slabs (all slabs are in use).
    // If slab size has not been established yet, blocks until it is.
    // Throws exception in dummy mode, or if stop() is called from another thread.
    void block_until_empty();

    // In async-aware mode (underlying BumpAllocator was constructed async),
    // delegates to bump_allocator->wait_until_initialized(). In dummy mode
    // (no underlying BumpAllocator), no-op. In sync mode (BumpAllocator was
    // sync), bump_allocator's wait is a no-op too.
    //
    // Note: this does NOT trigger the deferred b->allocate_bytes() call; the
    // first get_slab() does that. The purpose of calling this method
    // explicitly is to surface async-init failures eagerly rather than from
    // the first get_slab() (which may run later, from a worker thread).
    void wait_until_initialized();

    // Stop the allocator. Any thread blocked in get_slab() will wake up and throw.
    // If 'e' is non-null, it represents an error; if null, it's normal termination.
    // Thread-safe; first call sets the error.
    // In non-dummy mode, also propagates stop(e) to the underlying BumpAllocator
    // (per the thread-backed-class pattern).
    void stop(std::exception_ptr e = nullptr) const;

    const int aflags;           // allocation flags from ksgpu
    const long capacity;        // total bytes in base region, or < 0 for dummy mode

protected:
    // Protected constructors - use create() factory methods instead.
    SlabAllocator(int aflags, long capacity);
    SlabAllocator(const std::shared_ptr<BumpAllocator> &b, long nbytes);

private:
    // The underlying memory region. Set in the constructor for the
    // (aflags, capacity) factory; set lazily on first get_slab() for the
    // BumpAllocator-backed factory (so the constructor doesn't block on
    // an async BumpAllocator's init).
    std::shared_ptr<void> base;

    // Held only in non-dummy BumpAllocator-backed mode. Keeps the
    // BumpAllocator alive until first get_slab() does the deferred
    // allocate_bytes(). Stays alive afterward for the SlabAllocator's
    // lifetime (cheap; one shared_ptr).
    std::shared_ptr<BumpAllocator> bump_allocator;

    // Stop-pattern state ('mutable' since stop() is const -- see
    // notes/stoppable_class.md). is_stopped/error are protected by 'lock'.
    mutable std::mutex lock;
    mutable std::condition_variable cv;  // signaled when a slab is returned, initialized, or stop() is called
    mutable bool is_stopped = false;
    mutable std::exception_ptr error;

    // Slab management. These are protected by 'lock'.
    long slab_size = -1;            // slab size in bytes (established by first get_slab)
    long num_slabs = 0;             // total number of slabs
    std::vector<void *> free_list;  // stack of free slab pointers

    // True while a get_slab() caller is performing the deferred
    // bump_allocator->allocate_bytes() with 'lock' released (so that stop()
    // is not blocked behind the BumpAllocator's async init). Protected by
    // 'lock'; other get_slab() callers wait on 'cv' while it is set.
    bool init_underway = false;

    // Helper for blocking operations. Caller must hold lock.
    void _throw_if_stopped() const;

    // Entry-point body; get_slab() is a thin wrapper that stops the
    // allocator if this throws (see notes/stoppable_class.md).
    std::shared_ptr<void> _get_slab(long nbytes, bool blocking);

    // Helper called when a slab's refcount drops to zero.
    void return_slab(void *slab_ptr);
};


}  // namespace pirate

#endif // _PIRATE_SLAB_ALLOCATOR_HPP
