#include "../include/pirate/BumpAllocator.hpp"
#include "../include/pirate/inlines.hpp"  // align_up()

#include <stdexcept>
#include <sstream>
#include <ksgpu/xassert.hpp>

namespace pirate {
#if 0
}  // editor auto-indent
#endif


BumpAllocator::BumpAllocator(int aflags_, long capacity_)
    : aflags(aflags_), capacity(capacity_)
{
    ksgpu::check_aflags(aflags, "BumpAllocator constructor");
    
    if (aflags & ksgpu::af_random)
        throw std::runtime_error("BumpAllocator constructor: af_random flag is not supported");
    
    if (aflags & ksgpu::af_guard)
        throw std::runtime_error("BumpAllocator constructor: af_guard flags is not supported");

    if (capacity >= 0) {
        // Round capacity up to alignment boundary.
        capacity = align_up(capacity, nalign);
        
        // Allocate base region.
        // We use nelts=capacity with dtype that has 1 byte per element.
        base = ksgpu::_af_alloc(ksgpu::Dtype(ksgpu::df_uint, 8), capacity, aflags);
        
        // Verify that returned pointer is aligned.
        uintptr_t p = reinterpret_cast<uintptr_t>(base.get());
        xassert((p % nalign) == 0);
    }
}


std::shared_ptr<void> BumpAllocator::get_base() const
{
    if (capacity < 0)
        throw std::runtime_error("BumpAllocator::get_base() called in dummy mode (capacity < 0)");
    return base;
}


void *BumpAllocator::allocate_bytes(long nbytes)
{
    if (capacity < 0)
        throw std::runtime_error("BumpAllocator::allocate_bytes() called in dummy mode (capacity < 0)");
    
    if (nbytes < 0) {
        std::stringstream ss;
        ss << "BumpAllocator::allocate_bytes(): nbytes=" << nbytes << " is negative";
        throw std::runtime_error(ss.str());
    }
    
    if (nbytes == 0)
        return nullptr;

    // Round up to alignment boundary.
    long aligned_nbytes = align_up(nbytes, nalign);
    
    // Atomically claim space from the buffer.
    long old_offset = nbytes_allocated.fetch_add(aligned_nbytes);
    long new_offset = old_offset + aligned_nbytes;
    
    if (new_offset > capacity) {
        // Roll back the allocation.
        nbytes_allocated.fetch_sub(aligned_nbytes);
        
        std::stringstream ss;
        ss << "BumpAllocator::allocate_bytes(): allocation of " << nbytes << " bytes would exceed capacity "
           << capacity << " (currently allocated: " << old_offset << ")";
        throw std::runtime_error(ss.str());
    }
    
    char *base_ptr = static_cast<char *>(base.get());
    return base_ptr + old_offset;
}


// Note: caller has checked 'dtype'.
ksgpu::Array<void> BumpAllocator::_allocate_array_internal(ksgpu::Dtype dtype, int ndim, const long *shape, const long *strides)
{
    ksgpu::Array<void> ret;
    
    // _array_init_dchecked(..., allocate=false) initializes all Array members 
    // except 'data' and 'base', and returns element count needed for allocation.
    long nalloc = ksgpu::_array_init_dchecked(ret, dtype, ndim, shape, strides, aflags, false);
    long nbytes = nalloc * (dtype.nbits / 8);
    
    if (ret.size > 0) {
        if (capacity < 0) {
            // In dummy mode, allocate a fresh array with af_alloc().
            ret.base = ksgpu::_af_alloc(dtype, nalloc, aflags);
            ret.data = ret.base.get();
            nbytes_allocated.fetch_add(align_up(nbytes, nalign));
        }
        else {
            // Normal mode: allocate from bump allocator.
            ret.base = this->base;
            ret.data = this->allocate_bytes(nbytes);
        }
    }

    ksgpu::_check_array_invariants_except_dtype(ret, "BumpAllocator::allocate_array()");
    return ret;
}


}  // namespace pirate

