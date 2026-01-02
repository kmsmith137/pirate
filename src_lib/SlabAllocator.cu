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


SlabAllocator::SlabAllocator(long nbytes, int aflags_)
    : aflags(aflags_)
{
    if (nbytes <= 0) {
        std::stringstream ss;
        ss << "SlabAllocator constructor: nbytes=" << nbytes << " must be positive";
        throw std::runtime_error(ss.str());
    }
    
    ksgpu::check_aflags(aflags, "SlabAllocator constructor");
    
    if (aflags & ksgpu::af_random)
        throw std::runtime_error("SlabAllocator constructor: af_random flag is not supported");
    
    if (aflags & ksgpu::af_guard)
        throw std::runtime_error("SlabAllocator constructor: af_guard flag is not supported");
    
    if (aflags & ksgpu::af_gpu)
        throw std::runtime_error("SlabAllocator constructor: af_gpu flag is not supported (host memory only)");
    
    // Round capacity up to alignment boundary.
    this->capacity = align_up(nbytes, nalign);
    
    // Allocate base region using dtype with 1 byte per element.
    this->base = ksgpu::_af_alloc(ksgpu::Dtype(ksgpu::df_uint, 8), capacity, aflags);
    
    // Verify that returned pointer is aligned.
    uintptr_t p = reinterpret_cast<uintptr_t>(base.get());
    xassert((p % nalign) == 0);
}


SlabAllocator::SlabAllocator(BumpAllocator &b, long nbytes)
    : aflags(b.aflags)
{
    if (nbytes <= 0) {
        std::stringstream ss;
        ss << "SlabAllocator constructor: nbytes=" << nbytes << " must be positive";
        throw std::runtime_error(ss.str());
    }
    
    if (b.capacity < 0)
        throw std::runtime_error("SlabAllocator constructor: BumpAllocator is in dummy mode");
    
    if (aflags & ksgpu::af_gpu)
        throw std::runtime_error("SlabAllocator constructor: af_gpu flag is not supported (host memory only)");
    
    // Get memory from the BumpAllocator.
    this->capacity = align_up(nbytes, nalign);
    void *ptr = b.allocate_bytes(capacity);
    
    // Verify alignment.
    uintptr_t p = reinterpret_cast<uintptr_t>(ptr);
    xassert((p % nalign) == 0);
    
    // Create a shared_ptr that keeps the BumpAllocator's base alive.
    // The deleter is a no-op since BumpAllocator manages the lifetime.
    std::shared_ptr<void> bump_base = b.get_base();
    this->base = std::shared_ptr<void>(bump_base, ptr);
}


SlabAllocator::~SlabAllocator()
{
    // The free_list should contain all slabs when the allocator is destroyed.
    // We don't enforce this (slabs might outlive the allocator), but it's
    // worth noting that leaked slabs will keep the base memory alive.
}


// -------------------------------------------------------------------------------------------------
//
// _allocate_slab(): helper that pops a slab from the free list


void *SlabAllocator::_allocate_slab(long nbytes)
{
    if (nbytes <= 0) {
        std::stringstream ss;
        ss << "SlabAllocator::get_slab(): nbytes=" << nbytes << " must be positive";
        throw std::runtime_error(ss.str());
    }
    
    // Round up to alignment.
    long aligned_nbytes = align_up(nbytes, nalign);
    
    std::lock_guard<std::mutex> guard(lock);
    
    // First allocation establishes the slab size.
    if (slab_size < 0) {
        slab_size = aligned_nbytes;
        
        // Calculate total number of slabs.
        num_slabs = capacity / slab_size;
        
        if (num_slabs <= 0) {
            std::stringstream ss;
            ss << "SlabAllocator::get_slab(): capacity=" << capacity
               << " is too small for slab_size=" << slab_size;
            throw std::runtime_error(ss.str());
        }
        
        // Initialize free list with all slabs.
        char *base_ptr = static_cast<char *>(base.get());
        free_list.reserve(num_slabs);
        for (long i = 0; i < num_slabs; i++) {
            free_list.push_back(base_ptr + i * slab_size);
        }
    }
    else if (aligned_nbytes != slab_size) {
        std::stringstream ss;
        ss << "SlabAllocator::get_slab(): requested size " << nbytes
           << " (aligned: " << aligned_nbytes << ") does not match established slab size "
           << slab_size;
        throw std::runtime_error(ss.str());
    }
    
    // Pop a slab from the free list.
    if (free_list.empty()) {
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


std::shared_ptr<void> SlabAllocator::get_slab(long nbytes)
{
    void *slab_ptr = _allocate_slab(nbytes);
    
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
    std::lock_guard<std::mutex> guard(lock);
    free_list.push_back(slab_ptr);
}


long SlabAllocator::num_free_slabs() const
{
    std::lock_guard<std::mutex> guard(lock);
    return static_cast<long>(free_list.size());
}


long SlabAllocator::num_total_slabs() const
{
    std::lock_guard<std::mutex> guard(lock);
    return num_slabs;
}


long SlabAllocator::get_slab_size() const
{
    std::lock_guard<std::mutex> guard(lock);
    return slab_size;
}


}  // namespace pirate
