#include "../include/pirate/SlabAllocator.hpp"
#include "../include/pirate/inlines.hpp"  // align_up()

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


void SlabAllocator::stop(std::exception_ptr e)
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
        cv.notify_all();
    }
    if (ba_to_notify)
        ba_to_notify->stop(e);
}


void SlabAllocator::wait_until_initialized()
{
    // No-op in dummy mode (no underlying BumpAllocator).
    if (!bump_allocator)
        return;
    // Delegates to the BumpAllocator's wait. Does NOT trigger the deferred
    // b->allocate_bytes() / b->get_base() calls -- those happen on first
    // get_slab(). The purpose of explicit wait is to surface async-init
    // failures eagerly.
    bump_allocator->wait_until_initialized();
}


void SlabAllocator::_throw_if_stopped() const
{
    if (error)
        std::rethrow_exception(error);
    
    if (is_stopped)
        throw std::runtime_error("SlabAllocator method called on stopped instance");
}


void SlabAllocator::block_until_empty()
{
    if (is_dummy())
        throw std::runtime_error("SlabAllocator::block_until_empty(): not available in dummy mode");
    
    std::unique_lock<std::mutex> guard(lock);
    _throw_if_stopped();

    // Wait until slab size is established and free list is empty.
    while ((slab_size < 0) || !free_list.empty()) {
        cv.wait(guard);
        _throw_if_stopped();
    }
}


// -------------------------------------------------------------------------------------------------
//
// get_slab(): the main allocation method


std::shared_ptr<void> SlabAllocator::get_slab(long nbytes, bool blocking)
{
    if (nbytes <= 0) {
        std::stringstream ss;
        ss << "SlabAllocator::get_slab(): nbytes=" << nbytes << " must be positive";
        throw std::runtime_error(ss.str());
    }

    long aligned_nbytes = align_up(nbytes, nalign);

    std::unique_lock<std::mutex> guard(lock);
    _throw_if_stopped();

    if ((slab_size >= 0) && (aligned_nbytes != slab_size)) {
        std::stringstream ss;
        ss << "SlabAllocator::get_slab(): requested size " << nbytes
           << " (aligned: " << aligned_nbytes << ") does not match established slab size "
           << slab_size;
        throw std::runtime_error(ss.str());
    }

    if (is_dummy()) {
        slab_size = aligned_nbytes;
        guard.unlock();
        return ksgpu::_af_alloc(ksgpu::Dtype(ksgpu::df_uint,8), slab_size, aflags);
    }

    // Lazy init from the BumpAllocator (deferred from the constructor so
    // multiple async BumpAllocators can be constructed without
    // serialization). The first caller performs the (potentially
    // long-blocking) b->allocate_bytes() with 'lock' RELEASED: stop() needs
    // 'lock', and its cascade into the BumpAllocator is the very call that
    // unblocks the wait -- holding 'lock' across the wait would deadlock
    // stop() behind it. Other get_slab() callers wait on 'cv' until the
    // init completes (or the allocator is stopped).
    while (!base) {
        if (init_underway) {
            // Another thread is performing the lazy init; wait for it.
            cv.wait(guard);
            _throw_if_stopped();
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
            // Surface the failure to this SlabAllocator's subsequent callers
            // and propagate to the BumpAllocator. stop() wakes any waiters,
            // which then throw via _throw_if_stopped(); clearing
            // 'init_underway' afterwards is just hygiene.
            auto e = std::current_exception();
            this->stop(e);

            {
                std::lock_guard<std::mutex> g2(lock);
                init_underway = false;
            }

            std::rethrow_exception(e);
        }

        guard.lock();
        this->base = new_base;
        this->init_underway = false;
        cv.notify_all();  // wake threads waiting on 'init_underway'

        // stop() may have been called while 'lock' was released.
        _throw_if_stopped();
    }

    if (slab_size < 0) {
        // Validate before committing 'slab_size' / 'num_slabs': an oversized
        // first request must not leave the allocator in a broken state
        // (slab size established, but zero slabs -- later blocking callers
        // would hang forever). Per the stoppable-class pattern, this throw
        // also stops the allocator, since it indicates a misconfiguration
        // rather than a recoverable condition.
        long aligned_capacity = align_up(capacity, nalign);
        long new_num_slabs = aligned_capacity / aligned_nbytes;

        if (new_num_slabs <= 0) {
            std::stringstream ss;
            ss << "SlabAllocator::get_slab(): capacity=" << capacity
                << " is too small for slab_size=" << aligned_nbytes;
            auto e = std::make_exception_ptr(std::runtime_error(ss.str()));
            guard.unlock();
            this->stop(e);
            std::rethrow_exception(e);
        }

        slab_size = aligned_nbytes;
        num_slabs = new_num_slabs;

        // First allocation: initialize free list with all slabs.
        char *base_ptr = static_cast<char *>(base.get());
        free_list.reserve(num_slabs);
        for (long i = 0; i < num_slabs; i++)
            free_list.push_back(base_ptr + i * slab_size);

        // Wake up any thread waiting in block_until_empty() for initialization.
        cv.notify_all();
    }

    // Wait for a slab if blocking, otherwise throw.
    while (free_list.empty()) {
        if (!blocking) {
            std::stringstream ss;
            ss << "SlabAllocator::get_slab(): no free slabs available (total slabs: "
               << num_slabs << ")";
            throw std::runtime_error(ss.str());
        }

        cv.wait(guard);
        _throw_if_stopped();
    }

    void *slab_ptr = free_list.back();
    free_list.pop_back();
    bool notify = free_list.empty();  // wake up block_until_empty() if free list is now empty
    guard.unlock();
    
    if (notify)
        cv.notify_all();

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
    cv.notify_one();  // wake up one waiting thread, if any
}


long SlabAllocator::num_free_slabs() const
{
    if (is_dummy())
        throw std::runtime_error("SlabAllocator::num_free_slabs(): not available in dummy mode");
    
    std::lock_guard<std::mutex> guard(lock);
    
    if (slab_size < 0)
        throw std::runtime_error("SlabAllocator::num_free_slabs(): slab size has not been established yet");
    
    return static_cast<long>(free_list.size());
}


long SlabAllocator::num_total_slabs(bool blocking) const
{
    if (is_dummy())
        throw std::runtime_error("SlabAllocator::num_total_slabs(): not available in dummy mode");
    
    std::unique_lock<std::mutex> guard(lock);
    _throw_if_stopped();

    while (slab_size < 0) {
        if (!blocking)
            throw std::runtime_error("SlabAllocator::num_total_slabs(): slab size has not been established yet");
        cv.wait(guard);
        _throw_if_stopped();
    }
    
    return num_slabs;
}


long SlabAllocator::get_slab_size() const
{
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
