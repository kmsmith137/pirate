#include "../include/pirate/SlabAllocator.hpp"
#include "../include/pirate/inlines.hpp"  // align_up(), is_aligned()

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
// SlabHeader: stored at the start of each slab for host memory allocations.
// This holds the reference to keep the SlabAllocator alive.


struct SlabAllocator::SlabHeader
{
    std::shared_ptr<SlabAllocator> allocator;
    void *slab_ptr;  // pointer to start of internal slab (for returning to free list)
    
    SlabHeader(std::shared_ptr<SlabAllocator> a, void *p)
        : allocator(std::move(a)), slab_ptr(p) { }
    
    ~SlabHeader()
    {
        if (allocator)
            allocator->return_slab(slab_ptr);
    }
};


// -------------------------------------------------------------------------------------------------
//
// InPlaceAllocator: custom allocator that uses pre-existing memory rather than malloc.
// Used with std::allocate_shared to embed the control block in the slab.


template<typename T>
struct SlabAllocator::InPlaceAllocator
{
    using value_type = T;
    
    void *mem;
    
    explicit InPlaceAllocator(void *m) : mem(m) { }
    
    template<typename U>
    InPlaceAllocator(const InPlaceAllocator<U> &other) : mem(other.mem) { }
    
    T *allocate(std::size_t n)
    {
        // std::allocate_shared allocates space for both the control block and
        // the SlabHeader in a single allocation. Verify it fits in our reserved space.
        std::size_t total_bytes = n * sizeof(T);
        if (total_bytes > SlabAllocator::control_overhead) {
            std::stringstream ss;
            ss << "SlabAllocator: control block size (" << total_bytes
               << " bytes) exceeds reserved overhead (" << SlabAllocator::control_overhead
               << " bytes). This is a build configuration issue.";
            throw std::runtime_error(ss.str());
        }
        return static_cast<T *>(mem);
    }
    
    void deallocate(T *, std::size_t) noexcept
    {
        // No-op: the memory is part of the slab, not separately allocated.
    }
    
    template<typename U>
    bool operator==(const InPlaceAllocator<U> &) const { return true; }
    
    template<typename U>
    bool operator!=(const InPlaceAllocator<U> &other) const { return !(*this == other); }
};


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
        
        // We embed the control block at the start of each slab.
        internal_slab_size = align_up(aligned_nbytes + control_overhead, nalign);
        
        // Calculate total number of slabs.
        num_slabs = capacity / internal_slab_size;
        
        if (num_slabs <= 0) {
            std::stringstream ss;
            ss << "SlabAllocator::get_slab(): capacity=" << capacity
               << " is too small for slab_size=" << slab_size
               << " (internal_slab_size=" << internal_slab_size << ")";
            throw std::runtime_error(ss.str());
        }
        
        // Initialize free list with all slabs.
        char *base_ptr = static_cast<char *>(base.get());
        free_list.reserve(num_slabs);
        for (long i = 0; i < num_slabs; i++) {
            free_list.push_back(base_ptr + i * internal_slab_size);
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
    
    // Embed the control block at the start of the slab using std::allocate_shared
    // with a custom allocator, avoiding any malloc for the shared_ptr machinery.
    //
    // Layout: [SlabHeader + control block] [padding] [user data]
    //         ^-- slab_ptr                           ^-- user_ptr
    
    void *user_ptr = static_cast<char *>(slab_ptr) + control_overhead;
    
    // Create the shared_ptr using in-place allocation.
    InPlaceAllocator<SlabHeader> alloc(slab_ptr);
    std::shared_ptr<SlabHeader> header = std::allocate_shared<SlabHeader>(
        alloc, shared_from_this(), slab_ptr);
    
    // Create an aliasing shared_ptr that points to the user data region.
    // This shares ownership with 'header', so when all references to the
    // user data are dropped, the SlabHeader destructor runs.
    return std::shared_ptr<void>(header, user_ptr);
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

