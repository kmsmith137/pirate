#include "../include/pirate/SlabAllocator.hpp"
#include "../include/pirate/inlines.hpp"     // align_up()

#include <algorithm>   // std::max
#include <stdexcept>
#include <sstream>
#include <ksgpu/xassert.hpp>
#include <ksgpu/mem_utils.hpp>

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// -------------------------------------------------------------------------------------------------
//
// SlabAllocator factory methods and constructors


std::shared_ptr<SlabAllocator> SlabAllocator::create(const std::shared_ptr<BumpAllocator> &b)
{
    return std::shared_ptr<SlabAllocator>(new SlabAllocator(b));
}


std::shared_ptr<SlabAllocator> SlabAllocator::create(int aflags)
{
    return std::shared_ptr<SlabAllocator>(new SlabAllocator(aflags));
}


// Dummy-mode constructor: no BumpAllocator. Each get_slab() call allocates
// fresh memory with _af_alloc(aflags). (In non-dummy mode the aflags are
// validated by the BumpAllocator's own constructor.)
SlabAllocator::SlabAllocator(int aflags_)
    : aflags(aflags_), is_dummy(true)
{
    ksgpu::check_aflags(aflags, "SlabAllocator constructor");

    if (aflags & ksgpu::af_random)
        throw std::runtime_error("SlabAllocator constructor: af_random flag is not supported");

    if (aflags & ksgpu::af_guard)
        throw std::runtime_error("SlabAllocator constructor: af_guard flag is not supported");
}


SlabAllocator::SlabAllocator(const std::shared_ptr<BumpAllocator> &b)
    : aflags(b->aflags), is_dummy(false), bump_allocator(b)
{
    if (b->capacity < 0)
        throw std::runtime_error("SlabAllocator constructor: BumpAllocator is in dummy mode");

    // No memory is taken from the BumpAllocator here: slabs are carved one
    // at a time, on demand, by get_slab(). This also means the constructor
    // never blocks on an async BumpAllocator's init (each per-slab
    // allocate_bytes() call blocks internally instead).
}


// -------------------------------------------------------------------------------------------------
//
// stop() and _throw_if_stopped()


void SlabAllocator::stop(std::exception_ptr e) const
{
    // Snapshot the bump_allocator pointer under the lock, then release the
    // lock before propagating stop() downstream -- avoids holding two
    // locks at once. (No actual deadlock is reachable since BumpAllocator's
    // stop never calls back into SlabAllocator, but the discipline keeps
    // lock-order reasoning trivially correct.)
    std::shared_ptr<BumpAllocator> ba_to_notify;
    {
        std::lock_guard<std::mutex> guard(lock);
        if (is_stopped)
            return;
        is_stopped = true;
        error = e;
        ba_to_notify = bump_allocator;  // null in dummy mode
    }
    // Notify after releasing the mutex so woken threads aren't immediately
    // blocked re-acquiring it. Stop notifies every cv of the class.
    free_cv.notify_all();
    empty_cv.notify_all();
    if (ba_to_notify)
        ba_to_notify->stop(e);
}


void SlabAllocator::wait_until_initialized()
{
    try {
        // Check our OWN stopped state first, so the stopped behavior is
        // uniform across modes. Without this, an error-stopped dummy-mode
        // allocator (no underlying BumpAllocator) would silently succeed,
        // while bump-backed mode rethrows the root cause via the
        // BumpAllocator's readiness gate.
        {
            std::lock_guard<std::mutex> guard(lock);
            _throw_if_stopped("SlabAllocator::wait_until_initialized");
        }

        // No-op in dummy mode (no underlying BumpAllocator).
        if (!bump_allocator)
            return;
        // Delegates to the BumpAllocator's wait. Does NOT carve any slabs --
        // get_slab() does that. The purpose of explicit wait is to surface
        // async-init failures eagerly.
        bump_allocator->wait_until_initialized();
    } catch (...) {
        stop(std::current_exception());
        throw;
    }
}


void SlabAllocator::_throw_if_stopped(const char *method_name) const
{
    if (error)
        std::rethrow_exception(error);

    if (is_stopped)
        throw std::runtime_error(std::string(method_name) + " called on stopped instance");
}


void SlabAllocator::block_until_empty()
{
    try {
        if (is_dummy)
            throw std::runtime_error("SlabAllocator::block_until_empty(): not available in dummy mode");

        std::unique_lock<std::mutex> guard(lock);
        _throw_if_stopped("SlabAllocator::block_until_empty");

        // "Empty" = the BumpAllocator is exhausted AND every carved slab is
        // in use. Both conjuncts are required: before exhaustion, an empty
        // free list is not memory pressure (the next get_slab() would just
        // carve another slab). Both true-ward transitions notify empty_cv
        // (see the cv roster in SlabAllocator.hpp), so no wakeup is lost
        // regardless of which conjunct becomes true last.
        while (!bump_allocator_empty || !free_list.empty()) {
            empty_cv.wait(guard);
            _throw_if_stopped("SlabAllocator::block_until_empty");
        }
    } catch (...) {
        stop(std::current_exception());
        throw;
    }
}


// -------------------------------------------------------------------------------------------------
//
// get_slab(): the main allocation method


std::shared_ptr<void> SlabAllocator::get_slab(long nbytes, bool blocking)
{
    // Per the strict stoppable-class policy (notes/stoppable_class.md), ANY
    // exception thrown from an entry point stops the allocator -- including
    // argument errors (bad nbytes, size mismatch) and the non-blocking
    // "pool exhausted" throw.
    try {
        return _get_slab(nbytes, blocking);
    } catch (...) {
        stop(std::current_exception());
        throw;
    }
}


std::shared_ptr<void> SlabAllocator::_get_slab(long nbytes, bool blocking)
{
    if (nbytes <= 0) {
        std::stringstream ss;
        ss << "SlabAllocator::get_slab(): nbytes=" << nbytes << " must be positive";
        throw std::runtime_error(ss.str());
    }

    long aligned_nbytes = align_up(nbytes, nalign);

    std::unique_lock<std::mutex> guard(lock);
    _throw_if_stopped("SlabAllocator::get_slab");

    // The first call establishes the slab size; subsequent calls throw on a
    // mismatch. Establish-at-entry (all modes) is what makes the size check
    // race-free: 'lock' is released during the carve below, so a check-only
    // test here could go stale before the size was committed (handing out a
    // wrong-size slab). Set-once, with no cv notify: no wait predicate
    // tests slab_size.
    if (slab_size < 0)
        slab_size = aligned_nbytes;
    else if (aligned_nbytes != slab_size) {
        std::stringstream ss;
        ss << "SlabAllocator::get_slab(): requested size " << nbytes
           << " (aligned: " << aligned_nbytes << ") does not match established slab size "
           << slab_size;
        throw std::runtime_error(ss.str());
    }

    if (is_dummy) {
        guard.unlock();
        // Use the local 'aligned_nbytes', not 'slab_size': the lock was just
        // released, so re-reading the member here would be a data race.
        return ksgpu::_af_alloc(ksgpu::Dtype(ksgpu::df_uint,8), aligned_nbytes, aflags);
    }

    for (;;) {
        // 1. Serve from the free list if possible.
        if (!free_list.empty()) {
            void *slab_ptr = free_list.back();
            free_list.pop_back();
            bool notify = free_list.empty();  // true-ward transition of the block_until_empty() predicate
            guard.unlock();
            if (notify)
                empty_cv.notify_all();
            return _wrap_slab(slab_ptr);
        }

        // 2. Carve a new slab from the BumpAllocator, unless it is known to
        // be exhausted. The carve runs with 'lock' RELEASED: allocate_bytes()
        // can block on the bump's async init, stop() needs 'lock', and
        // stop()'s cascade into the BumpAllocator is the very call that
        // unblocks the wait -- holding 'lock' across it would deadlock
        // stop() behind us. Use the local 'aligned_nbytes' (== slab_size),
        // not the member: re-reading slab_size with the lock released would
        // be a data race. (Concurrent carvers are fine: each caller needs
        // its own slab, and BumpAllocator is internally thread-safe.)
        if (!bump_allocator_empty) {
            guard.unlock();
            void *ptr = bump_allocator->allocate_bytes(aligned_nbytes, /*throw_on_failure=*/ false);
            // (If the bump is stopped/failed, allocate_bytes() throws; the
            // get_slab() wrapper then stops this allocator too.)
            guard.lock();
            _throw_if_stopped("SlabAllocator::get_slab");

            if (ptr != nullptr) {
                uintptr_t p = reinterpret_cast<uintptr_t>(ptr);
                xassert((p % nalign) == 0);

                // Maintain the return_slab() no-throw invariant
                // (free_list.capacity() >= num_slabs_allocated -- see the
                // member comment in SlabAllocator.hpp) BEFORE the new slab
                // can escape. Amortized doubling keeps ramp-up O(n) rather
                // than O(n^2); if reserve() throws, num_slabs_allocated has
                // not yet been incremented, so the invariant still holds
                // for all previously handed-out slabs (the carved bytes are
                // abandoned; the bump never reclaims them anyway).
                if (long(free_list.capacity()) < num_slabs_allocated + 1)
                    free_list.reserve(std::max(2 * long(free_list.capacity()), 16L));
                num_slabs_allocated++;
                guard.unlock();
                return _wrap_slab(ptr);
            }

            // The BumpAllocator is exhausted; cache that fact so we never
            // call allocate_bytes() again. (Another carver may have beaten
            // us to it while 'lock' was released -- then it already
            // notified.) This is a true-ward transition of the
            // block_until_empty() predicate, so it must notify empty_cv.
            // Under-lock notify: legitimate here, the lock is deliberately
            // held into the next loop iteration (see notes/cpp.md).
            if (!bump_allocator_empty) {
                bump_allocator_empty = true;
                empty_cv.notify_all();
            }
            // Loop: slabs may have been returned while 'lock' was released.
            continue;
        }

        // 3. Free list empty AND BumpAllocator exhausted: wait or throw.
        if (!blocking) {
            std::stringstream ss;
            ss << "SlabAllocator::get_slab(): pool exhausted (" << num_slabs_allocated
               << " slabs carved from the BumpAllocator, all in use)";
            throw std::runtime_error(ss.str());
        }

        free_cv.wait(guard);
        _throw_if_stopped("SlabAllocator::get_slab");
    }
}


// -------------------------------------------------------------------------------------------------
//
// Helper methods


std::shared_ptr<void> SlabAllocator::_wrap_slab(void *slab_ptr)
{
    // Create shared_ptr with a custom deleter that returns the slab to the
    // pool. The captured shared_ptr<SlabAllocator> ensures the allocator
    // (and, transitively, the BumpAllocator's memory region) stays alive as
    // long as any slab is in use.
    std::shared_ptr<SlabAllocator> self = shared_from_this();

    return std::shared_ptr<void>(slab_ptr, [self, slab_ptr](void *) {
        self->return_slab(slab_ptr);
    });
}


void SlabAllocator::return_slab(void *slab_ptr)
{
    // Runs inside a slab shared_ptr deleter, so it must never throw. The
    // push_back can't: _get_slab()'s carve path maintains
    // free_list.capacity() >= num_slabs_allocated, and
    // (slabs handed out) + free_list.size() <= num_slabs_allocated (each
    // carved slab has exactly one deleter, invoked once), so the push_back
    // never reallocates.
    std::unique_lock<std::mutex> guard(lock);
    free_list.push_back(slab_ptr);
    guard.unlock();
    // notify_one is sound here: every free_cv waiter has the same predicate
    // (free list non-empty), and one returned slab satisfies exactly one
    // waiter. See the cv comments in SlabAllocator.hpp.
    free_cv.notify_one();
}


bool SlabAllocator::is_initialized() const
{
    // No stopped-check: deliberately usable on a stopped allocator
    // (stopped-tolerant informational accessor -- see the entry-point
    // classification in SlabAllocator.hpp).
    //
    // No underlying BumpAllocator -- dummy mode (each get_slab call
    // allocates fresh memory): always "ready".
    if (!bump_allocator)
        return true;
    // Bump-backed: delegate to the underlying BumpAllocator.
    return bump_allocator->is_initialized();
}


}  // namespace pirate
