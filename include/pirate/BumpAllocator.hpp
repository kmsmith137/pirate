#ifndef _PIRATE_BUMP_ALLOCATOR_HPP
#define _PIRATE_BUMP_ALLOCATOR_HPP

#include "constants.hpp"

#include <atomic>
#include <vector>
#include <memory>
#include <initializer_list>
#include <ksgpu/Array.hpp>

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// BumpAllocator: A thread-safe bump allocator that supports GPU/host memory.
//
// The allocator operates in one of two modes, depending on the 'capacity' arg
// to the constructor:
//
//   - capacity >= 0: The constructor pre-allocates a "base" memory region using
//     ksgpu::af_alloc(). Calls to allocate_bytes() or allocate_array() return
//     pointers/arrays within this region. The lifetime of all allocations is tied
//     to the base shared_ptr.
//
//   - capacity < 0: "Dummy" mode. No memory is allocated in the constructor.
//     allocate_bytes() and get_base() throw exceptions. allocate_array() allocates
//     memory on-the-fly with af_alloc(), where each Array has its own independent
//     base pointer.
//
// Thread safety: All public methods are thread-safe. The allocate_bytes() and
// allocate_array() methods use atomic operations to update the byte offset.
//
// Alignment: All allocations are aligned to constants::bytes_per_gpu_cache_line
// (currently 128 bytes).


struct BumpAllocator
{
    static constexpr int nalign = constants::bytes_per_gpu_cache_line;

    // Constructor. The 'aflags' are memory allocation flags from ksgpu/mem_utils.hpp.
    // If capacity >= 0, allocates a base memory region of 'capacity' bytes.
    // If capacity < 0, operates in "dummy" mode (see class comment above).
    BumpAllocator(int aflags, long capacity);

    // Noncopyable
    BumpAllocator(const BumpAllocator &) = delete;
    BumpAllocator &operator=(const BumpAllocator &) = delete;

    // Number of bytes allocated so far (aligned to cache line size).
    // This counter is always valid, even in dummy mode.
    std::atomic<long> nbytes_allocated{0};

    // Returns the base shared_ptr. Throws an exception in dummy mode.
    std::shared_ptr<void> get_base() const;

    // Allocates 'nbytes' from the base region, returns pointer.
    // Throws an exception in dummy mode, or if allocation would exceed capacity.
    // The returned pointer is aligned to constants::bytes_per_gpu_cache_line.
    // Warning: caller is responsible for keeping a reference to the base shared_ptr!
    void *allocate_bytes(long nbytes);

    // Allocate an Array with the given shape and default (contiguous) strides.
    // If T==void, throws an exception (use the version with explicit dtype).
    // The returned ksgpu::Array keeps a reference to the BumpAllocator base shared_ptr.
    template<typename T>
    ksgpu::Array<T> allocate_array(std::initializer_list<long> shape);

    // Allocate an Array with the given dtype and shape.
    // If T==void, the returned array has the given dtype.
    // If T!=void, throws an exception if T and dtype do not match.
    template<typename T>
    ksgpu::Array<T> allocate_array(ksgpu::Dtype dtype, std::initializer_list<long> shape);

    // Allocate an Array with explicit shape and strides.
    // If T==void, throws an exception (use the version with explicit dtype).
    template<typename T>
    ksgpu::Array<T> allocate_array(const std::vector<long> &shape, const std::vector<long> &strides);

    // Allocate an Array with explicit dtype, shape, and strides.
    // If T==void, the returned array has the given dtype.
    // If T!=void, throws an exception if T and dtype do not match.
    template<typename T>
    ksgpu::Array<T> allocate_array(ksgpu::Dtype dtype, const std::vector<long> &shape, const std::vector<long> &strides);

    
    // ----- Internals -----

    int aflags = 0;
    long capacity = -1;      // -1 means dummy mode
    std::shared_ptr<void> base;

    // Helper: allocates array, used by all allocate_array() overloads.
    // Handles both normal mode and dummy mode.
    ksgpu::Array<void> _allocate_array_internal(ksgpu::Dtype dtype, int ndim, const long *shape, const long *strides);
};


// -------------------------------------------------------------------------------------------------
//
// Template implementations.


template<typename T>
ksgpu::Array<T> BumpAllocator::allocate_array(std::initializer_list<long> shape)
{
    static_assert(!std::is_void_v<T>, "BumpAllocator::allocate_array<void>() requires explicit dtype");
    ksgpu::Dtype dtype = ksgpu::Dtype::native<T>();
    ksgpu::Array<void> ret = _allocate_array_internal(dtype, shape.size(), shape.begin(), nullptr);
    return ret.template cast<T>("BumpAllocator::allocate_array()");
}


template<typename T>
ksgpu::Array<T> BumpAllocator::allocate_array(ksgpu::Dtype dtype, std::initializer_list<long> shape)
{
    ksgpu::_check_dtype<T>(dtype, "BumpAllocator::allocate_array()");
    ksgpu::Array<void> ret = _allocate_array_internal(dtype, shape.size(), shape.begin(), nullptr);
    return ret.template cast<T>("BumpAllocator::allocate_array()");
}


template<typename T>
ksgpu::Array<T> BumpAllocator::allocate_array(const std::vector<long> &shape, const std::vector<long> &strides)
{
    static_assert(!std::is_void_v<T>, "BumpAllocator::allocate_array<void>() requires explicit dtype");
    
    if (shape.size() != strides.size())
        throw std::runtime_error("BumpAllocator::allocate_array(): shape/strides size mismatch");
    
    ksgpu::Dtype dtype = ksgpu::Dtype::native<T>();
    ksgpu::Array<void> ret = _allocate_array_internal(dtype, shape.size(), &shape[0], &strides[0]);
    return ret.template cast<T>("BumpAllocator::allocate_array()");
}


template<typename T>
ksgpu::Array<T> BumpAllocator::allocate_array(ksgpu::Dtype dtype, const std::vector<long> &shape, const std::vector<long> &strides)
{
    ksgpu::_check_dtype<T>(dtype, "BumpAllocator::allocate_array()");
    
    if (shape.size() != strides.size())
        throw std::runtime_error("BumpAllocator::allocate_array(): shape/strides size mismatch");
    
    ksgpu::Array<void> ret = _allocate_array_internal(dtype, shape.size(), &shape[0], &strides[0]);
    return ret.template cast<T>("BumpAllocator::allocate_array()");
}


}  // namespace pirate

#endif // _PIRATE_BUMP_ALLOCATOR_HPP

