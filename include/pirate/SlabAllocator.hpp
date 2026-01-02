#ifndef _PIRATE_SLAB_ALLOCATOR_HPP
#define _PIRATE_SLAB_ALLOCATOR_HPP

#include "constants.hpp"
#include "BumpAllocator.hpp"

#include <memory>
#include <mutex>
#include <vector>

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// SlabAllocator: A thread-safe pool allocator for fixed-size memory "slabs".
//
// The allocator manages a large pre-allocated host memory region, which is
// subdivided into fixed-size slabs. Slabs are returned to the pool when their
// reference count drops to zero, making them available for future allocations.
//
// Slab size is established by the first call to get_slab(). All subsequent calls
// must request the same size; otherwise an exception is thrown.
//
// Key features:
//   - Host memory only (GPU memory is not supported)
//   - Thread-safe (uses a lock for the free list)
//   - Slabs hold a reference to the SlabAllocator, preventing the underlying
//     memory from being freed while any slab is still in use
//   - Compatible with weak_ptr (control blocks are allocated separately)


class SlabAllocator : public std::enable_shared_from_this<SlabAllocator>
{
public:
    static constexpr int nalign = constants::bytes_per_gpu_cache_line;
    
    // Construct SlabAllocator that allocates new memory using af_alloc().
    // The 'aflags' are memory allocation flags from ksgpu/mem_utils.hpp.
    // GPU memory (af_gpu) is not supported; use af_rhost or af_uhost.
    SlabAllocator(long nbytes, int aflags);
    
    // Construct SlabAllocator that gets memory from an existing BumpAllocator.
    // The aflags are inherited from the BumpAllocator.
    // GPU memory (af_gpu) is not supported.
    SlabAllocator(BumpAllocator &b, long nbytes);
    
    ~SlabAllocator();
    
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
    std::shared_ptr<void> get_slab(long nbytes);
    
    // Returns the number of slabs currently available in the free list.
    long num_free_slabs() const;
    
    // Returns the total number of slabs in the pool.
    long num_total_slabs() const;
    
    // Returns the established slab size, or -1 if no slabs have been allocated yet.
    long get_slab_size() const;

private:
    // The underlying memory region.
    std::shared_ptr<void> base;
    long capacity = 0;       // total bytes in base region
    int aflags = 0;          // allocation flags from ksgpu
    
    // Slab management. These are protected by 'lock'.
    mutable std::mutex lock;
    long slab_size = -1;        // slab size in bytes (established by first get_slab)
    long num_slabs = 0;         // total number of slabs
    std::vector<void *> free_list;  // stack of free slab pointers
    
    // Helper called when a slab's refcount drops to zero.
    void return_slab(void *slab_ptr);
    
    // Helper that allocates a slab (validates nbytes, pops from free list).
    void *_allocate_slab(long nbytes);
};


}  // namespace pirate

#endif // _PIRATE_SLAB_ALLOCATOR_HPP
