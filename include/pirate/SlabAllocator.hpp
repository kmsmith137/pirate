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
//
// Entry points vs accessors (see notes/stoppable_class.md):
//   - Entry points -- throw/rethrow the saved error when stopped, and any
//     throw stops the allocator: get_slab(), block_until_empty(),
//     num_total_slabs(), wait_until_initialized(). Rule of thumb: methods
//     that can block are entry points.
//   - Stopped-tolerant informational accessors -- no stopped-state check;
//     last-known values remain meaningful for diagnostics after a stop:
//     num_free_slabs(), get_slab_size(), is_initialized(), is_dummy().


class SlabAllocator : public std::enable_shared_from_this<SlabAllocator>
{
public:
    static constexpr int nalign = constants::bytes_per_gpu_cache_line;
    
    // Factory method: create SlabAllocator that allocates new memory using af_alloc().
    // The 'aflags' are memory allocation flags from ksgpu/mem_utils.hpp.
    // If capacity < 0, operates in "dummy" mode (see class comment).
    // capacity == 0 is rejected (throws): a zero-capacity pool could never
    // serve a get_slab() call.
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
    // If permissive=false (default): throws in dummy mode, or if the slab
    // pool has not been created yet (i.e. no get_slab() call has completed).
    // If permissive=true: returns 0 in those two cases instead -- never
    // throws (there are no other throw paths). Does NOT throw on a stopped
    // allocator in either mode (stopped-tolerant informational accessor).
    long num_free_slabs(bool permissive = false) const;

    // Returns the total number of slabs in the pool.
    // Throws exception in dummy mode.
    // If blocking=false (default) and the slab pool has not been created
    // yet, throws exception. If blocking=true, blocks until it is created.
    long num_total_slabs(bool blocking = false) const;
    
    // Returns the established slab size.
    // Throws exception if the slab size has not been established yet (by
    // the entry of the first get_slab() call). Does NOT throw on a stopped
    // allocator (stopped-tolerant informational accessor).
    long get_slab_size() const;
    
    // Returns true if the SlabAllocator is ready to serve get_slab() calls
    // without blocking on async init. Semantics:
    //   - No underlying BumpAllocator (dummy and aflags modes): always true.
    //   - Bump-backed mode: delegates to bump_allocator->is_initialized().
    //
    // Note: does NOT check whether slab_size has been established (that's
    // user-pattern state, established on the first get_slab() call).
    bool is_initialized() const;
    
    // Returns true if in dummy mode (capacity < 0).
    bool is_dummy() const { return capacity < 0; }
    
    // Block until there are no free slabs (all slabs are in use).
    // If the slab pool has not been created yet (no completed get_slab()
    // call), blocks until it is.
    // Throws exception in dummy mode, or if stop() is called from another thread.
    void block_until_empty();

    // In async-aware mode (underlying BumpAllocator was constructed async),
    // delegates to bump_allocator->wait_until_initialized(). In dummy and
    // aflags modes (no underlying BumpAllocator), no-op. In sync mode
    // (BumpAllocator was sync), bump_allocator's wait is a no-op too.
    //
    // Note: this does NOT trigger the deferred b->allocate_bytes() call; the
    // first get_slab() does that. The purpose of calling this method
    // explicitly is to surface async-init failures eagerly rather than from
    // the first get_slab() (which may run later, from a worker thread).
    //
    // Throws on a stopped allocator (rethrows the saved error, or the
    // generic message on a clean stop), uniformly across modes.
    //
    // Deliberately no timeout_ms param (unlike BumpAllocator): callers wait
    // out slow async inits on the BumpAllocator itself (see run_server.py
    // phase 2), so this call should always be fast in practice.
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
    //
    // One condition variable per wait-predicate, so a targeted notify can
    // never be "lost" waking a waiter with a different predicate:
    //   free_cv  -- a slab was returned to the free list; awaited by
    //               get_slab(blocking=true). return_slab() uses notify_one,
    //               which is sound here BECAUSE all free_cv waiters share
    //               the same predicate and one returned slab satisfies
    //               exactly one of them.
    //   init_cv  -- the deferred BumpAllocator init completed, or failed
    //               (init_underway reset on the throw path); awaited by
    //               get_slab() callers that lost the init_underway race.
    //   size_cv  -- the pool was materialized (num_slabs set, free list
    //               filled, on the first completed get_slab()); awaited
    //               by num_total_slabs(blocking=true).
    //   empty_cv -- the free list became empty; awaited by
    //               block_until_empty().
    // stop() notify_all's all four.
    mutable std::mutex lock;
    mutable std::condition_variable free_cv;
    mutable std::condition_variable init_cv;
    mutable std::condition_variable size_cv;
    mutable std::condition_variable empty_cv;
    mutable bool is_stopped = false;
    mutable std::exception_ptr error;

    // Slab management. These are protected by 'lock'.
    //
    // slab_size is committed at first get_slab() ENTRY (see the establish-
    // or-throw logic at the top of _get_slab); num_slabs is set later,
    // atomically with the free-list fill, and doubles as the "pool
    // materialized" flag in wait predicates (dummy mode: stays 0).
    long slab_size = -1;            // slab size in bytes
    long num_slabs = 0;             // total number of slabs
    std::vector<void *> free_list;  // stack of free slab pointers

    // True while a get_slab() caller is performing the deferred
    // bump_allocator->allocate_bytes() with 'lock' released (so that stop()
    // is not blocked behind the BumpAllocator's async init). Protected by
    // 'lock'; other get_slab() callers wait on 'init_cv' while it is set.
    bool init_underway = false;

    // Helper for blocking operations. Caller must hold lock. Rethrows the
    // saved error if non-null; otherwise throws a generic
    // "<method_name> called on stopped instance".
    void _throw_if_stopped(const char *method_name) const;

    // Entry-point body; get_slab() is a thin wrapper that stops the
    // allocator if this throws (see notes/stoppable_class.md).
    std::shared_ptr<void> _get_slab(long nbytes, bool blocking);

    // Helper called when a slab's refcount drops to zero.
    void return_slab(void *slab_ptr);
};


}  // namespace pirate

#endif // _PIRATE_SLAB_ALLOCATOR_HPP
