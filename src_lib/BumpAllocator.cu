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
    if (capacity >= 0) {
        // Round capacity up to alignment boundary.
        capacity = align_up(capacity, constants::bytes_per_gpu_cache_line);
        
        // Allocate base region.
        // We use nelts=capacity with dtype that has 1 byte per element.
        base = ksgpu::_af_alloc(ksgpu::Dtype(ksgpu::df_uint, 8), capacity, aflags);
        
        // Verify that returned pointer is aligned.
        uintptr_t p = reinterpret_cast<uintptr_t>(base.get());
        xassert((p % constants::bytes_per_gpu_cache_line) == 0);
    }
    else {
        // Dummy mode: just validate aflags.
        ksgpu::check_aflags(aflags, "BumpAllocator constructor");
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
    
    // Round up to alignment boundary.
    long aligned_nbytes = align_up(nbytes, constants::bytes_per_gpu_cache_line);
    
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


// Static member function.
long BumpAllocator::_compute_array_nbytes(ksgpu::Dtype dtype, int ndim, const long *shape, const long *strides)
{
    if (!dtype.is_valid())
        throw std::runtime_error("BumpAllocator::allocate_array(): invalid dtype");
    
    if ((ndim < 0) || (ndim > ksgpu::ArrayMaxDim)) {
        std::stringstream ss;
        ss << "BumpAllocator::allocate_array(): ndim=" << ndim << " is out of range [0," << ksgpu::ArrayMaxDim << "]";
        throw std::runtime_error(ss.str());
    }
    
    long bytes_per_elt = dtype.nbits / 8;
    if ((dtype.nbits % 8) != 0) {
        std::stringstream ss;
        ss << "BumpAllocator::allocate_array(): dtype " << dtype << " has non-byte-aligned size";
        throw std::runtime_error(ss.str());
    }

    // Compute array size. For empty arrays (size==0), return 0 bytes.
    long size = 1;
    for (int d = 0; d < ndim; d++) {
        if (shape[d] < 0) {
            std::stringstream ss;
            ss << "BumpAllocator::allocate_array(): shape[" << d << "]=" << shape[d] << " is negative";
            throw std::runtime_error(ss.str());
        }
        size *= shape[d];
    }
    
    if (size == 0)
        return 0;
    
    // If strides are provided, validate them and compute the required buffer size.
    // Otherwise, assume contiguous layout.
    
    if (strides != nullptr) {
        // For arrays with explicit strides, we need to compute the total buffer size
        // by finding the maximum offset that any element can have.
        //
        // For each axis, the maximum index is (shape[d]-1), so the maximum offset
        // from varying that axis is (shape[d]-1) * abs(strides[d]).
        //
        // Note: we allow negative strides, but the returned pointer will point to
        // the element at index (0,0,...,0), which may not be at the start of the buffer.
        
        long max_offset = 0;
        long min_offset = 0;
        
        for (int d = 0; d < ndim; d++) {
            if (shape[d] == 0)
                continue;
            if (strides[d] >= 0)
                max_offset += (shape[d] - 1) * strides[d];
            else
                min_offset += (shape[d] - 1) * strides[d];
        }
        
        // The buffer needs to cover from min_offset to max_offset (inclusive).
        return (max_offset - min_offset + 1) * bytes_per_elt;
    }
    else {
        // Contiguous layout.
        return size * bytes_per_elt;
    }
}


ksgpu::Array<void> BumpAllocator::_allocate_array_internal(ksgpu::Dtype dtype, int ndim, const long *shape, const long *strides)
{
    // Compute contiguous strides if not provided.
    long contiguous_strides[ksgpu::ArrayMaxDim];
    
    if (strides == nullptr) {
        long stride = 1;
        for (int d = ndim - 1; d >= 0; d--) {
            contiguous_strides[d] = stride;
            stride *= shape[d];
        }
        strides = contiguous_strides;
    }
    
    long nbytes = _compute_array_nbytes(dtype, ndim, shape, strides);

    // In dummy mode, allocate a fresh array with af_alloc().
    if (capacity < 0) {
        long nelts = (dtype.nbits > 0) ? ((nbytes * 8) / dtype.nbits) : 0;
        
        // Note: for zero-size arrays, we create an empty array.
        if (nelts == 0) {
            ksgpu::Array<void> ret;
            ret.dtype = dtype;
            ret.ndim = ndim;
            ret.size = 0;
            ret.aflags = aflags;
            for (int d = 0; d < ndim; d++) {
                ret.shape[d] = shape[d];
                ret.strides[d] = strides[d];
            }
            for (int d = ndim; d < ksgpu::ArrayMaxDim; d++) {
                ret.shape[d] = 0;
                ret.strides[d] = 0;
            }
            return ret;
        }
        
        // Allocate memory and track the allocation size.
        std::shared_ptr<void> arr_base = ksgpu::_af_alloc(dtype, nelts, aflags);
        nbytes_allocated.fetch_add(align_up(nbytes, constants::bytes_per_gpu_cache_line));
        
        // Build the array.
        ksgpu::Array<void> ret;
        ret.dtype = dtype;
        ret.ndim = ndim;
        ret.aflags = aflags;
        ret.base = arr_base;
        
        ret.size = 1;
        for (int d = 0; d < ndim; d++) {
            ret.shape[d] = shape[d];
            ret.strides[d] = strides[d];
            ret.size *= shape[d];
        }
        for (int d = ndim; d < ksgpu::ArrayMaxDim; d++) {
            ret.shape[d] = 0;
            ret.strides[d] = 0;
        }
        
        // For negative strides, we need to offset the data pointer.
        // The data pointer should point to element (0,0,...,0).
        long min_offset = 0;
        for (int d = 0; d < ndim; d++) {
            if ((shape[d] > 0) && (strides[d] < 0))
                min_offset += (shape[d] - 1) * strides[d];
        }
        
        long bytes_per_elt = dtype.nbits / 8;
        char *base_ptr = static_cast<char *>(arr_base.get());
        ret.data = base_ptr - (min_offset * bytes_per_elt);
        
        ret.check_invariants("BumpAllocator::allocate_array()");
        return ret;
    }
    
    // Normal mode: allocate from bump allocator.
    ksgpu::Array<void> ret;
    ret.dtype = dtype;
    ret.ndim = ndim;
    ret.aflags = aflags;
    ret.base = base;
    
    ret.size = 1;
    for (int d = 0; d < ndim; d++) {
        ret.shape[d] = shape[d];
        ret.strides[d] = strides[d];
        ret.size *= shape[d];
    }
    for (int d = ndim; d < ksgpu::ArrayMaxDim; d++) {
        ret.shape[d] = 0;
        ret.strides[d] = 0;
    }
    
    if (ret.size == 0) {
        // Empty array: no memory needed.
        ret.data = nullptr;
        return ret;
    }
    
    // Allocate memory.
    void *ptr = allocate_bytes(nbytes);
    
    // For negative strides, offset the data pointer.
    long min_offset = 0;
    for (int d = 0; d < ndim; d++) {
        if ((shape[d] > 0) && (strides[d] < 0))
            min_offset += (shape[d] - 1) * strides[d];
    }
    
    long bytes_per_elt = dtype.nbits / 8;
    char *base_ptr = static_cast<char *>(ptr);
    ret.data = base_ptr - (min_offset * bytes_per_elt);
    
    ret.check_invariants("BumpAllocator::allocate_array()");
    return ret;
}


}  // namespace pirate

