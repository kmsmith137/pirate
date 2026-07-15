#include "../include/pirate/SlabAllocator.hpp"
#include "../include/pirate/inlines.hpp"     // align_up()

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


std::shared_ptr<SlabAllocator> SlabAllocator::create(int aflags, long capacity)
{
    return std::shared_ptr<SlabAllocator>(new SlabAllocator(aflags, capacity));
}


std::shared_ptr<SlabAllocator> SlabAllocator::create(const std::shared_ptr<BumpAllocator> &b, long nbytes)
{
    return std::shared_ptr<SlabAllocator>(new SlabAllocator(b, nbytes));
}


SlabAllocator::SlabAllocator(int aflags_, long capacity_)
    : aflags(aflags_), capacity(capacity_)
{
    ksgpu::check_aflags(aflags, "SlabAllocator constructor");
    
    if (aflags & ksgpu::af_random)
        throw std::runtime_error("SlabAllocator constructor: af_random flag is not supported");
    
    if (aflags & ksgpu::af_guard)
        throw std::runtime_error("SlabAllocator constructor: af_guard flag is not supported");
    
    if (!is_dummy()) {
        // Normal mode: pre-allocate base region.
        long aligned_capacity = align_up(capacity, nalign);
        
        // Allocate base region using dtype with 1 byte per element.
        this->base = ksgpu::_af_alloc(ksgpu::Dtype(ksgpu::df_uint, 8), aligned_capacity, aflags);
        
        // Verify that returned pointer is aligned.
        uintptr_t p = reinterpret_cast<uintptr_t>(base.get());
        xassert((p % nalign) == 0);
    }
    // else: dummy mode, base remains empty
}


SlabAllocator::SlabAllocator(const std::shared_ptr<BumpAllocator> &b, long nbytes)
    : aflags(b->aflags), capacity(nbytes), bump_allocator(b)
{
    if (nbytes <= 0) {
        std::stringstream ss;
        ss << "SlabAllocator constructor: nbytes=" << nbytes << " must be positive";
        throw std::runtime_error(ss.str());
    }

    if (b->capacity < 0)
        throw std::runtime_error("SlabAllocator constructor: BumpAllocator is in dummy mode");

    // Lazy init: defer b->allocate_bytes() and b->get_base() to first
    // get_slab(). This way the SlabAllocator constructor doesn't block on
    // the BumpAllocator's async init, allowing multiple async BumpAllocators
    // to be constructed concurrently without serialization on the
    // SlabAllocator's init step.
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
        ba_to_notify = bump_allocator;  // may be null in dummy mode
        free_cv.notify_all();
        init_cv.notify_all();
        size_cv.notify_all();
        empty_cv.notify_all();
    }
    if (ba_to_notify)
        ba_to_notify->stop(e);
}


void SlabAllocator::wait_until_initialized()
{
    try {
        // Check our OWN stopped state first, so the stopped behavior is
        // uniform across modes. Without this, an error-stopped allocator
        // with no underlying BumpAllocator (dummy or aflags mode) would
        // silently succeed, while bump-backed mode rethrows the root cause
        // via the BumpAllocator's readiness gate.
        {
            std::lock_guard<std::mutex> guard(lock);
            _throw_if_stopped("SlabAllocator::wait_until_initialized");
        }

        // No-op in dummy/aflags mode (no underlying BumpAllocator).
        if (!bump_allocator)
            return;
        // Delegates to the BumpAllocator's wait. Does NOT trigger the deferred
        // b->allocate_bytes() / b->get_base() calls -- those happen on first
        // get_slab(). The purpose of explicit wait is to surface async-init
        // failures eagerly.
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
        if (is_dummy())
            throw std::runtime_error("SlabAllocator::block_until_empty(): not available in dummy mode");

        std::unique_lock<std::mutex> guard(lock);
        _throw_if_stopped("SlabAllocator::block_until_empty");

        // Wait until the pool is materialized and the free list is empty.
        // The gate is 'num_slabs' (set atomically with the free-list fill),
        // NOT 'slab_size' (committed earlier, at first get_slab() entry):
        // an empty not-yet-filled free list must not satisfy this wait.
        // The predicate can only become true at an empty-transition (which
        // implies the pool exists), so empty_cv alone covers it.
        while ((num_slabs == 0) || !free_list.empty()) {
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
    // "no free slabs" throw.
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
    // race-free: 'lock' is released during the lazy-init wait below, so a
    // check-only test here could go stale before the size was committed
    // (handing out a wrong-size slab). Set-once, with no cv notify: no wait
    // predicate tests slab_size -- it is committed BEFORE the pool exists,
    // and predicates that need "pool materialized" test 'num_slabs' (set
    // atomically with the free-list fill) instead.
    if (slab_size < 0)
        slab_size = aligned_nbytes;
    else if (aligned_nbytes != slab_size) {
        std::stringstream ss;
        ss << "SlabAllocator::get_slab(): requested size " << nbytes
           << " (aligned: " << aligned_nbytes << ") does not match established slab size "
           << slab_size;
        throw std::runtime_error(ss.str());
    }

    if (is_dummy()) {
        guard.unlock();
        // Use the local 'aligned_nbytes', not 'slab_size': the lock was just
        // released, so re-reading the member here would be a data race.
        return ksgpu::_af_alloc(ksgpu::Dtype(ksgpu::df_uint,8), aligned_nbytes, aflags);
    }

    // Lazy init from the BumpAllocator (deferred from the constructor so
    // multiple async BumpAllocators can be constructed without
    // serialization). The first caller performs the (potentially
    // long-blocking) b->allocate_bytes() with 'lock' RELEASED: stop() needs
    // 'lock', and its cascade into the BumpAllocator is the very call that
    // unblocks the wait -- holding 'lock' across the wait would deadlock
    // stop() behind it. Other get_slab() callers wait on 'init_cv' until the
    // init completes (or the allocator is stopped).
    while (!base) {
        if (init_underway) {
            // Another thread is performing the lazy init; wait for it.
            init_cv.wait(guard);
            _throw_if_stopped("SlabAllocator::get_slab");
            continue;
        }

        init_underway = true;
        guard.unlock();

        std::shared_ptr<void> new_base;

        try {
            // b->allocate_bytes() blocks on async init and rethrows on
            // async-init failure. (If stop() is called while we are blocked
            // here, its cascade into the BumpAllocator makes this call throw.)
            long aligned_capacity = align_up(capacity, nalign);
            void *ptr = bump_allocator->allocate_bytes(aligned_capacity);
            uintptr_t p = reinterpret_cast<uintptr_t>(ptr);
            xassert((p % nalign) == 0);
            std::shared_ptr<void> bump_base = bump_allocator->get_base();
            new_base = std::shared_ptr<void>(bump_base, ptr);
        } catch (...) {
            // The try/catch wrapper in get_slab() stops the allocator
            // (waking any waiters, which then throw via _throw_if_stopped);
            // here we just clear 'init_underway' and rethrow.
            {
                std::lock_guard<std::mutex> g2(lock);
                init_underway = false;
            }
            throw;
        }

        guard.lock();
        this->base = new_base;
        this->init_underway = false;
        init_cv.notify_all();  // wake threads waiting on 'init_underway'

        // stop() may have been called while 'lock' was released.
        _throw_if_stopped("SlabAllocator::get_slab");
    }

    if (num_slabs == 0) {
        // First completed allocation: materialize the pool. Validate before
        // committing 'num_slabs' (the "pool exists" flag tested by wait
        // predicates). The throw below stops the allocator (via the
        // try/catch wrapper in get_slab()), so the slab_size committed at
        // entry is only ever left without a pool on a stopped instance.
        long aligned_capacity = align_up(capacity, nalign);
        long new_num_slabs = aligned_capacity / slab_size;

        if (new_num_slabs <= 0) {
            std::stringstream ss;
            ss << "SlabAllocator::get_slab(): capacity=" << capacity
                << " is too small for slab_size=" << slab_size;
            throw std::runtime_error(ss.str());
        }

        num_slabs = new_num_slabs;

        // Initialize free list with all slabs.
        char *base_ptr = static_cast<char *>(base.get());
        free_list.reserve(num_slabs);
        for (long i = 0; i < num_slabs; i++)
            free_list.push_back(base_ptr + i * slab_size);

        // Wake any thread waiting in num_total_slabs(blocking=true) for the
        // pool to be created. (block_until_empty() waiters don't need this
        // wake: their predicate can only become true at an empty-transition,
        // notified on empty_cv below.)
        size_cv.notify_all();
    }

    // Wait for a slab if blocking, otherwise throw.
    while (free_list.empty()) {
        if (!blocking) {
            std::stringstream ss;
            ss << "SlabAllocator::get_slab(): no free slabs available (total slabs: "
               << num_slabs << ")";
            throw std::runtime_error(ss.str());
        }

        free_cv.wait(guard);
        _throw_if_stopped("SlabAllocator::get_slab");
    }

    void *slab_ptr = free_list.back();
    free_list.pop_back();
    bool notify = free_list.empty();  // wake up block_until_empty() if free list is now empty
    guard.unlock();

    if (notify)
        empty_cv.notify_all();

    // Create shared_ptr with a custom deleter that returns the slab to the pool.
    // The captured shared_ptr<SlabAllocator> ensures the allocator (and its
    // underlying memory) stays alive as long as any slab is in use.
    std::shared_ptr<SlabAllocator> self = shared_from_this();
    
    return std::shared_ptr<void>(slab_ptr, [self, slab_ptr](void *) {
        self->return_slab(slab_ptr);
    });
}


// -------------------------------------------------------------------------------------------------
//
// Helper methods


void SlabAllocator::return_slab(void *slab_ptr)
{
    std::unique_lock<std::mutex> guard(lock);
    free_list.push_back(slab_ptr);
    guard.unlock();
    // notify_one is sound here: every free_cv waiter has the same predicate
    // (free list non-empty), and one returned slab satisfies exactly one
    // waiter. See the cv comments in SlabAllocator.hpp.
    free_cv.notify_one();
}


long SlabAllocator::num_free_slabs() const
{
    // No stopped-check: deliberately usable on a stopped allocator
    // (stopped-tolerant informational accessor -- see the entry-point
    // classification in SlabAllocator.hpp).
    if (is_dummy())
        throw std::runtime_error("SlabAllocator::num_free_slabs(): not available in dummy mode");

    std::lock_guard<std::mutex> guard(lock);

    // Gate on 'num_slabs', not 'slab_size': the latter is committed at first
    // get_slab() entry, before the pool exists (see _get_slab).
    if (num_slabs == 0)
        throw std::runtime_error("SlabAllocator::num_free_slabs(): slab pool has not been created yet");

    return static_cast<long>(free_list.size());
}


long SlabAllocator::num_total_slabs(bool blocking) const
{
    try {
        if (is_dummy())
            throw std::runtime_error("SlabAllocator::num_total_slabs(): not available in dummy mode");

        std::unique_lock<std::mutex> guard(lock);
        _throw_if_stopped("SlabAllocator::num_total_slabs");

        // Gate on 'num_slabs', not 'slab_size': the latter is committed at
        // first get_slab() entry, before the pool exists (see _get_slab),
        // and waiters here return num_slabs.
        while (num_slabs == 0) {
            if (!blocking)
                throw std::runtime_error("SlabAllocator::num_total_slabs(): slab pool has not been created yet");
            size_cv.wait(guard);
            _throw_if_stopped("SlabAllocator::num_total_slabs");
        }

        return num_slabs;
    } catch (...) {
        stop(std::current_exception());
        throw;
    }
}


long SlabAllocator::get_slab_size() const
{
    // No stopped-check: deliberately usable on a stopped allocator
    // (stopped-tolerant informational accessor -- see the entry-point
    // classification in SlabAllocator.hpp).
    std::lock_guard<std::mutex> guard(lock);

    if (slab_size < 0)
        throw std::runtime_error("SlabAllocator::get_slab_size(): slab size has not been established yet");

    return slab_size;
}


bool SlabAllocator::is_initialized() const
{
    // Dummy mode: no underlying BumpAllocator; always "ready" (each
    // get_slab call allocates fresh memory).
    if (!bump_allocator)
        return true;
    // Non-dummy: delegate to the underlying BumpAllocator. (Note: we do
    // NOT check whether this->base has been set -- that happens lazily
    // on first get_slab(), after the BumpAllocator is initialized.)
    return bump_allocator->is_initialized();
}


}  // namespace pirate
