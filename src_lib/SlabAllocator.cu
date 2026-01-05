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
// SlabAllocator constructors


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


SlabAllocator::SlabAllocator(BumpAllocator &b, long nbytes)
    : aflags(b.aflags), capacity(nbytes)
{
    if (nbytes <= 0) {
        std::stringstream ss;
        ss << "SlabAllocator constructor: nbytes=" << nbytes << " must be positive";
        throw std::runtime_error(ss.str());
    }
    
    if (b.capacity < 0)
        throw std::runtime_error("SlabAllocator constructor: BumpAllocator is in dummy mode");
    
    // Get memory from the BumpAllocator.
    long aligned_capacity = align_up(capacity, nalign);
    void *ptr = b.allocate_bytes(aligned_capacity);
    
    // Verify alignment.
    uintptr_t p = reinterpret_cast<uintptr_t>(ptr);
    xassert((p % nalign) == 0);
    
    // Create a shared_ptr that keeps the BumpAllocator's base alive.
    // The deleter is a no-op since BumpAllocator manages the lifetime.
    std::shared_ptr<void> bump_base = b.get_base();
    this->base = std::shared_ptr<void>(bump_base, ptr);
}


// -------------------------------------------------------------------------------------------------
//
// _check_nbytes(): helper for slab size validation
// Caller must hold 'lock'.

bool SlabAllocator::_check_nbytes(long nbytes)
{    
    if (nbytes <= 0) {
        std::stringstream ss;
        ss << "SlabAllocator::get_slab(): nbytes=" << nbytes << " must be positive";
        throw std::runtime_error(ss.str());
    }

    long aligned_nbytes = align_up(nbytes, nalign);
    
    if (slab_size < 0) {
        slab_size = aligned_nbytes;
        return true;
    }
    
    if (aligned_nbytes != slab_size) {
        std::stringstream ss;
        ss << "SlabAllocator::get_slab(): requested size " << nbytes
           << " (aligned: " << aligned_nbytes << ") does not match established slab size "
           << slab_size;
        throw std::runtime_error(ss.str());
    }
    
    return false;
}


// -------------------------------------------------------------------------------------------------
//
// _allocate_slab(): helper that pops a slab from the free list (or waits if blocking)


void *SlabAllocator::_allocate_slab(long nbytes, bool blocking)
{
    xassert(!is_dummy());
    
    std::unique_lock<std::mutex> guard(lock);
    
    if (_check_nbytes(nbytes)) {
        // First allocation: initialize free list with all slabs.
        long aligned_capacity = align_up(capacity, nalign);
        num_slabs = aligned_capacity / slab_size;
        
        if (num_slabs <= 0) {
            std::stringstream ss;
            ss << "SlabAllocator::get_slab(): capacity=" << capacity
               << " is too small for slab_size=" << slab_size;
            throw std::runtime_error(ss.str());
        }
        
        char *base_ptr = static_cast<char *>(base.get());
        free_list.reserve(num_slabs);
        for (long i = 0; i < num_slabs; i++) {
            free_list.push_back(base_ptr + i * slab_size);
        }
    }
    
    // Wait for a slab if blocking, otherwise throw.
    if (blocking) {
        cv.wait(guard, [this]() { return !free_list.empty(); });
    }
    else if (free_list.empty()) {
        std::stringstream ss;
        ss << "SlabAllocator::get_slab(): no free slabs available (total slabs: "
           << num_slabs << ")";
        throw std::runtime_error(ss.str());
    }
    
    void *slab_ptr = free_list.back();
    free_list.pop_back();
    return slab_ptr;
}


// -------------------------------------------------------------------------------------------------
//
// get_slab(): the main allocation method


std::shared_ptr<void> SlabAllocator::get_slab(long nbytes, bool blocking)
{
    if (is_dummy()) {
        // Dummy mode: allocate fresh memory using af_alloc().
        std::unique_lock<std::mutex> guard(lock);
        _check_nbytes(nbytes);
        guard.unlock();
        
        return ksgpu::_af_alloc(ksgpu::Dtype(ksgpu::df_uint, 8), slab_size, aflags);
    }
    
    // Normal mode: get slab from pool.
    void *slab_ptr = _allocate_slab(nbytes, blocking);
    
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
    return static_cast<long>(free_list.size());
}


long SlabAllocator::num_total_slabs() const
{
    if (is_dummy())
        throw std::runtime_error("SlabAllocator::num_total_slabs(): not available in dummy mode");
    
    std::lock_guard<std::mutex> guard(lock);
    return num_slabs;
}


long SlabAllocator::get_slab_size() const
{
    std::lock_guard<std::mutex> guard(lock);
    return slab_size;
}


}  // namespace pirate
