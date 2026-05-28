#ifndef _PIRATE_BUMP_ALLOCATOR_HPP
#define _PIRATE_BUMP_ALLOCATOR_HPP

#include "constants.hpp"

#include <atomic>
#include <condition_variable>
#include <exception>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include <ksgpu/Array.hpp>

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// State held by the base shared_ptr's deleter in async mode. Captured by
// value into the deleter so it outlives BumpAllocator if needed (e.g., a
// SlabAllocator still holds a reference to base after BumpAllocator dies).
struct AsyncDeleterState {
    long size = 0;
    int cuda_device = -1;
    std::atomic<int> n_registered{0};            // case 1 only; always 0 in cases 2, 3
    std::vector<long> reg_chunk_offsets;         // case 1 only; size n_registered_chunks + 1
};


// BumpAllocator: A thread-safe bump allocator that supports GPU/host memory.
//
// Sync mode (async=false, default):
//   - capacity >= 0: pre-allocates a "base" region via ksgpu::af_alloc().
//   - capacity < 0: "dummy" mode. allocate_bytes()/get_base() throw.
//
// Async mode (async=true):
//   - constructor returns immediately; allocation + zeroing happens on
//     worker threads.
//   - public methods (allocate_bytes, get_base, allocate_array) block
//     until init complete; failures rethrow.
//   - wait_until_initialized() blocks explicitly.
//   - aflags must equal exactly one of:
//       * af_mmap_huge | af_rhost | af_zero  (case 1: mmap + chunked register)
//       * af_rhost | af_zero                 (case 2: cudaHostAlloc + parallel zero)
//       * af_gpu | af_zero                   (case 3: cudaMalloc + cudaMemset)
//   - cuda_device must be >= 0.
//   - nthreads must be >= 2 for cases 1 and 2; ignored for case 3.

struct BumpAllocator
{
    static constexpr int nalign = constants::bytes_per_gpu_cache_line;

    // The nthreads and cuda_device args are ignored when async=false.
    BumpAllocator(int aflags, long capacity,
                  bool async = false, int nthreads = 0, int cuda_device = -1);
    ~BumpAllocator();

    // Noncopyable, nonmovable.
    BumpAllocator(const BumpAllocator &) = delete;
    BumpAllocator &operator=(const BumpAllocator &) = delete;
    BumpAllocator(BumpAllocator &&) = delete;
    BumpAllocator &operator=(BumpAllocator &&) = delete;

    // Number of bytes allocated so far (aligned to cache line size).
    // This counter is always valid, even in dummy mode.
    std::atomic<long> nbytes_allocated{0};

    // Returns the base shared_ptr. Throws in dummy mode. In async mode,
    // blocks until init complete (or rethrows async-init exception).
    std::shared_ptr<void> get_base() const;

    // Allocates 'nbytes' from the base region, returns pointer.
    // Throws in dummy mode, or if allocation would exceed capacity.
    // In async mode, blocks until init complete (or rethrows).
    // Warning: caller is responsible for keeping a reference to the base shared_ptr!
    void *allocate_bytes(long nbytes);

    // In async mode: blocks until init complete (returns) or async init
    // failed (rethrows the captured exception). In sync mode: returns
    // immediately. Safe to call any number of times.
    void wait_until_initialized();

    // Returns true if the allocator is ready to serve allocations (sync
    // mode: always true after ctor returns; async mode: true after workers
    // have finished initialization). Returns false if init is still in
    // progress OR if the allocator has been stopped (with or without
    // error). Use wait_until_initialized() to block; this is a
    // non-blocking poll.
    bool is_initialized() const;

    // Stop the allocator. Idempotent. If 'e' is non-null, it's stored as
    // the error; first stop wins. Workers see the stop_flag and exit.
    // Subsequent calls to allocate_bytes/get_base/wait_until_initialized
    // throw (rethrowing the stored error if any).
    void stop(std::exception_ptr e = nullptr);

    template<typename T>
    ksgpu::Array<T> allocate_array(std::initializer_list<long> shape);

    template<typename T>
    ksgpu::Array<T> allocate_array(ksgpu::Dtype dtype, std::initializer_list<long> shape);

    template<typename T>
    ksgpu::Array<T> allocate_array(const std::vector<long> &shape, const std::vector<long> &strides);

    template<typename T>
    ksgpu::Array<T> allocate_array(ksgpu::Dtype dtype, const std::vector<long> &shape, const std::vector<long> &strides);


    // ----- Internals -----

    int aflags = 0;
    long capacity = -1;      // -1 means dummy mode
    std::shared_ptr<void> base;

    // Helper: allocates array, used by all allocate_array() overloads.
    // Handles both normal mode and dummy mode.
    ksgpu::Array<void> _allocate_array_internal(ksgpu::Dtype dtype, int ndim, const long *shape, const long *strides);

    // Async state. Sync mode leaves these mostly unused (is_initialized
    // is set to true at end of sync ctor, so the blocking helper is a
    // single uncontended mutex acquire).
    mutable std::mutex _mutex;
    mutable std::condition_variable _cv;
    bool _is_initialized = false;
    bool _is_stopped = false;
    std::exception_ptr _error;
    std::atomic<bool> _stop_flag{false};

    // Held in async mode (null in sync mode). Captured by value into base's
    // deleter so the counter + size outlive BumpAllocator if external refs
    // (e.g. a SlabAllocator slab) keep `base` alive.
    std::shared_ptr<AsyncDeleterState> _async_state;

    // Async worker threads. Empty in sync mode.
    std::vector<std::thread> _workers;

    // Transient chunking state used by workers in cases 1 and 2.
    std::atomic<long> _next_zero_chunk{0};
    std::vector<std::atomic<int>> _super_done;   // case 1: per-super-chunk completion counter
    std::atomic<int> _workers_remaining{0};      // case 2: last-out-finalizes counter
    long _nzero_chunks = 0;
    long _nreg_chunks = 0;
    // Case 1 only: register chunks aligned to absolute
    // constants::cuda_host_register_chunk_size-aligned host addresses
    // (so the head and tail register chunks may be partial). Zero chunks
    // are inserted not to straddle register-chunk boundaries; entries are
    // monotonic with _zero_chunk_starts[0] = 0 and
    // _zero_chunk_starts[_nzero_chunks] = size. _super_of_zero_chunk[i]
    // maps each zero chunk to the register chunk it lies within.
    // _zero_chunks_per_super[s] holds the count for completion-tracking.
    std::vector<long> _zero_chunk_starts;
    std::vector<int>  _super_of_zero_chunk;
    std::vector<int>  _zero_chunks_per_super;

    // Blocking helper for async mode (no-op in sync mode).
    void _block_until_ready_or_throw() const;

    // Marked private-ish (in struct so accessible to internals but
    // not part of the user API).
    void _finalize_initialized();

    // Worker bodies (one per case).
    void _zero_worker_case1();
    void _registrar_worker_case1();
    void _zero_worker_case2();
    void _memset_worker_case3();

    // Per-case async init helpers (called by ctor).
    void _async_init_case1(long capacity_arg, int nthreads, int cuda_device);
    void _async_init_case2(long capacity_arg, int nthreads, int cuda_device);
    void _async_init_case3(long capacity_arg, int cuda_device);
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
