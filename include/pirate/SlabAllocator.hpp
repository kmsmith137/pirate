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
// The allocator carves fixed-size slabs from a BumpAllocator ON DEMAND: each
// get_slab() call takes a slab from the free list if one is available, and
// otherwise carves one new slab from the BumpAllocator. When the
// BumpAllocator is exhausted (allocate_bytes() returns nullptr), that fact
// is cached and get_slab() is thereafter served from returned slabs only.
// Slabs are returned to the free list when their reference count drops to
// zero, making them available for future allocations.
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
// Dummy mode (constructed via the create(aflags) factory):
//   - No BumpAllocator is involved
//   - get_slab() allocates fresh memory using af_alloc() for each request
//
// Entry points vs accessors (see notes/stoppable_class.md):
//   - Entry points -- throw/rethrow the saved error when stopped, and any
//     throw stops the allocator: get_slab(), block_until_empty(),
//     wait_until_initialized(). Rule of thumb: methods that can block are
//     entry points.
//   - Stopped-tolerant informational accessors -- no stopped-state check;
//     last-known values remain meaningful for diagnostics after a stop:
//     is_initialized().


class SlabAllocator : public std::enable_shared_from_this<SlabAllocator>
{
public:
    static constexpr int nalign = constants::bytes_per_gpu_cache_line;

    // Factory method: create SlabAllocator that carves slabs from an existing
    // BumpAllocator, one at a time, as get_slab() calls arrive. There is no
    // up-front carve amount: the SlabAllocator draws from the BumpAllocator
    // until allocate_bytes() reports exhaustion.
    // The aflags are inherited from the BumpAllocator.
    // Throws exception if BumpAllocator is in dummy mode.
    //
    // If the BumpAllocator is async, the SlabAllocator constructor still
    // returns immediately: each per-slab b->allocate_bytes() call blocks on
    // the async init internally (in practice only the first one waits).
    // Async-init failures from `b` surface from either get_slab() or
    // wait_until_initialized().
    static std::shared_ptr<SlabAllocator> create(const std::shared_ptr<BumpAllocator> &b);

    // Factory method: create SlabAllocator in "dummy" mode (see class comment).
    // The 'aflags' are memory allocation flags from ksgpu/mem_utils.hpp, used
    // for the per-slab af_alloc() calls in get_slab().
    static std::shared_ptr<SlabAllocator> create(int aflags);

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
    // If no free slab is available, a new one is carved from the
    // BumpAllocator. Once the BumpAllocator is exhausted:
    //   - blocking=false (default): throws an exception;
    //   - blocking=true: waits until a slab is returned to the pool.
    //
    // In dummy mode, always allocates fresh memory using af_alloc().
    std::shared_ptr<void> get_slab(long nbytes, bool blocking = false);

    // Returns true if the SlabAllocator is ready to serve get_slab() calls
    // without blocking on async init. Semantics:
    //   - Dummy mode (no underlying BumpAllocator): always true.
    //   - Bump-backed mode: delegates to bump_allocator->is_initialized().
    //
    // Note: does NOT check whether slab_size has been established (that's
    // user-pattern state, established on the first get_slab() call).
    bool is_initialized() const;

    // Block until the BumpAllocator is exhausted AND there are no free slabs
    // (i.e. every carved slab is in use). Both conditions are required:
    // before exhaustion, an empty free list is not memory pressure -- the
    // next get_slab() would simply carve another slab.
    // Throws exception in dummy mode, or if stop() is called from another thread.
    void block_until_empty();

    // In async-aware mode (underlying BumpAllocator was constructed async),
    // delegates to bump_allocator->wait_until_initialized(). In dummy mode
    // (no underlying BumpAllocator), no-op. In sync mode (BumpAllocator was
    // sync), bump_allocator's wait is a no-op too.
    //
    // Note: this does NOT carve any slabs; get_slab() does that. The purpose
    // of calling this method explicitly is to surface async-init failures
    // eagerly rather than from the first get_slab() (which may run later,
    // from a worker thread).
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

    const int aflags;    // allocation flags from ksgpu (inherited from the BumpAllocator in non-dummy mode)
    const bool is_dummy; // constant after construction, not lock-protected

protected:
    // Protected constructors - use create() factory methods instead.
    explicit SlabAllocator(const std::shared_ptr<BumpAllocator> &b);
    explicit SlabAllocator(int aflags);   // dummy mode

private:
    // Null in dummy mode. MUST stay alive for the SlabAllocator's lifetime:
    // slabs are raw pointers into the BumpAllocator's base region, and this
    // member is what keeps that region valid (BumpAllocator::base is a
    // 'const' member, freed only when the BumpAllocator is destroyed; slabs
    // keep the SlabAllocator alive via their deleters, and the SlabAllocator
    // keeps the BumpAllocator alive via this member).
    std::shared_ptr<BumpAllocator> bump_allocator;

    // Stop-pattern state ('mutable' since stop() is const -- see
    // notes/stoppable_class.md). is_stopped/error are protected by 'lock'.
    //
    // One condition variable per wait-predicate, so a targeted notify can
    // never be "lost" waking a waiter with a different predicate:
    //   free_cv  -- a slab was returned to the free list; awaited by
    //               get_slab(blocking=true) after BumpAllocator exhaustion.
    //               return_slab() uses notify_one, which is sound here
    //               BECAUSE all free_cv waiters share the same predicate and
    //               one returned slab satisfies exactly one of them.
    //   empty_cv -- awaited by block_until_empty(); predicate is
    //               (bump_allocator_empty && free_list.empty()). Signaled
    //               on BOTH true-ward transitions: (1) the free list's
    //               empty-transition on pop, in _get_slab(); (2)
    //               bump_allocator_empty set true, in _get_slab().
    // stop() notify_all's both.
    mutable std::mutex lock;
    mutable std::condition_variable free_cv;
    mutable std::condition_variable empty_cv;
    mutable bool is_stopped = false;
    mutable std::exception_ptr error;

    // Slab management. These are protected by 'lock'.
    //
    // slab_size is committed at first get_slab() ENTRY (see the establish-
    // or-throw logic at the top of _get_slab): 'lock' is released during the
    // carve, so a check-only test at entry could go stale before the size
    // was committed. Set-once, with no cv notify: no wait predicate tests it.
    //
    // num_slabs_allocated counts slabs carved from the BumpAllocator so far
    // (monotonically increasing; stays 0 in dummy mode).
    //
    // Invariant: free_list.capacity() >= num_slabs_allocated at all times.
    // Since (slabs handed out) + free_list.size() <= num_slabs_allocated,
    // this guarantees that return_slab()'s push_back never reallocates,
    // hence never throws (it runs inside a slab deleter). The carve path in
    // _get_slab() grows free_list (amortized doubling) BEFORE incrementing
    // num_slabs_allocated and before the new slab pointer escapes.
    long slab_size = -1;              // slab size in bytes
    long num_slabs_allocated = 0;     // slabs carved from the BumpAllocator so far
    std::vector<void *> free_list;    // stack of free slab pointers

    // True once bump_allocator->allocate_bytes() has returned nullptr
    // (capacity exhausted); cached so allocate_bytes() is never called
    // again. Protected by 'lock'. Not meaningful in dummy mode.
    bool bump_allocator_empty = false;

    // Helper for blocking operations. Caller must hold lock. Rethrows the
    // saved error if non-null; otherwise throws a generic
    // "<method_name> called on stopped instance".
    void _throw_if_stopped(const char *method_name) const;

    // Entry-point body; get_slab() is a thin wrapper that stops the
    // allocator if this throws (see notes/stoppable_class.md).
    std::shared_ptr<void> _get_slab(long nbytes, bool blocking);

    // Wraps a slab pointer in a shared_ptr whose deleter returns the slab to
    // the pool (and keeps 'this' alive via a captured shared_ptr).
    std::shared_ptr<void> _wrap_slab(void *slab_ptr);

    // Helper called when a slab's refcount drops to zero.
    void return_slab(void *slab_ptr);
};


}  // namespace pirate

#endif // _PIRATE_SLAB_ALLOCATOR_HPP
