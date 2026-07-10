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



// BumpAllocator: A thread-safe bump allocator that supports GPU/host memory.
//
// Intended for allocating huge memory regions at the beginning of programs,
// and carving them up afterwards.
//
// This started out simple and got complicated!
//
//   - The CUDA driver caps memory registrations (cudaHostAlloc(),
//     cudaHostRegister()) at ~511 GiB (undocumented!!)
//     Re-test on newer CUDA / driver versions with:
//        python -m pirate_frb revisit_512gb [-H]
//
//     To work around this, we register memory in chunks
//     (constants::cuda_host_register_chunk_size, currently 64 GiB)
//     aligned on pointer addresses.
//
//   - This created a new problem: cudaMemcpy*() fails if it crosses
//     registration chunk boundaries. We solved this problem as follows.
//
//      In situations where a cudaMemcpy* may be backed by a BumpAllocator,
//      we call safe_memcpy_{h2g,g2h}_{sync,async}() (see utils.hpp) which
//      splits host<->device copies at chunk boundaries.
//
//   - Initial registration and (especially) zeroing can be slow.
//     To solve this, we implemented an 'async' mode which allows
//     the caller to do other things in parallel, and also speeds
//     up the initialization by dispatching multiple threads.
//
// Sync mode (async=false, default):
//   - capacity > 0: allocate, optionally register, optionally zero --
//     all on the main thread before the ctor returns.
//   - capacity == 0: no-op.
//   - capacity < 0: "dummy" mode. allocate_bytes()/get_base() throw.
//
// Async mode (async=true):
//   - ctor returns immediately; allocation is done synchronously but
//     zeroing and chunked register run on worker threads. Public
//     methods (allocate_bytes, get_base, allocate_array, ...) block
//     until init completes; failures rethrow.
//   - capacity >= 0 required (no dummy mode).
//
// Argument requirements:
//   - cuda_device >= 0 if af_gpu is set (always: ctor uses it to pick
//     the device for cudaMalloc / cudaMemset).
//   - cuda_device >= 0 if (af_rhost) AND async (the registrar worker
//     calls cudaSetDevice).
//   - For sync mode + af_rhost, cuda_device is ignored: the chunked
//     register runs on the main thread against the caller's current
//     CUDA device.
//   - nthreads >= 2 if af_zero AND af_rhost AND async (1 registrar
//     thread + >= 1 zero worker).
//   - nthreads >= 1 if af_zero AND af_uhost AND async (>= 1 zero
//     worker; no registrar).
//   - Otherwise nthreads is ignored.
//
// aflags must be: exactly one of {af_uhost, af_rhost, af_gpu}, plus an
// optional subset of {af_mmap_huge, af_zero}. Any other flag triggers
// a constructor error. af_mmap_huge is rejected with af_gpu.
//
// Note: after construction, the `capacity` member may be larger than
// the value passed in. It is rounded up to a multiple of
// max(nalign, page-size implied by aflags) (4 KiB for af_uhost/af_rhost
// without af_mmap_huge, 2 MiB with af_mmap_huge, nalign=128 B for af_gpu).
//
// Thread-backed class (see notes/thread_backed_class.md).


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

    // Returns the base shared_ptr. Throws in dummy mode (capacity < 0).
    // For capacity == 0 ("empty" mode), returns a null shared_ptr (no
    // backing memory was allocated). In async mode, blocks until init
    // complete (or rethrows async-init exception).
    std::shared_ptr<void> get_base() const;

    // Allocates 'nbytes' from the base region, returns pointer.
    // Throws in dummy mode, or if allocation would exceed capacity.
    // In async mode, blocks until init complete (or rethrows).
    // Warning: caller is responsible for keeping a reference to the base shared_ptr!
    void *allocate_bytes(long nbytes);

    // In async mode: blocks until init complete (returns) or async init
    // failed (rethrows the captured exception). In sync mode: returns
    // immediately. Safe to call any number of times.
    void wait_until_initialized() const;

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
    void stop(std::exception_ptr e = nullptr) const;

    template<typename T>
    ksgpu::Array<T> allocate_array(std::initializer_list<long> shape);

    template<typename T>
    ksgpu::Array<T> allocate_array(ksgpu::Dtype dtype, std::initializer_list<long> shape);

    template<typename T>
    ksgpu::Array<T> allocate_array(const std::vector<long> &shape, const std::vector<long> &strides);

    template<typename T>
    ksgpu::Array<T> allocate_array(ksgpu::Dtype dtype, const std::vector<long> &shape, const std::vector<long> &strides);


    // ----- Internals -----

    const int aflags;
    const long capacity;      // -1 means dummy mode

    // State held by the base shared_ptr's deleter. Captured by value
    // into the deleter so it outlives BumpAllocator if needed (e.g., a
    // SlabAllocator still holds a reference to base after BumpAllocator
    // dies).
    struct DeleterState {
        std::atomic<int> n_registered{0};            // chunked-register paths only
        std::vector<long> reg_chunk_offsets;         // chunked-register paths only
    };

    // _nreg_chunks and _deleter_state are declared before `base`
    // because they are written by _allocate_base() (which initializes
    // `base` from the ctor's init list). Member init order follows
    // declaration order, so they must precede `base`.
    //
    // _nreg_chunks > 0 iff af_rhost (set by _setup_rhost_deleter).
    // _deleter_state is non-null iff af_rhost, in both sync and async
    // modes; non-rhost paths rely on ksgpu's own deleter.
    long _nreg_chunks = 0;
    std::shared_ptr<DeleterState> _deleter_state;

    // const after ctor: assigned exactly once via the init list.
    const std::shared_ptr<void> base;

    // Helper: allocates array, used by all allocate_array() overloads.
    // Handles both normal mode and dummy mode.
    ksgpu::Array<void> _allocate_array_internal(ksgpu::Dtype dtype, int ndim, const long *shape, const long *strides);

    // Entry-point body; allocate_bytes() is a thin wrapper that stops the
    // allocator if this throws (see notes/stoppable_class.md).
    void *_allocate_bytes(long nbytes);

    // State machine. Sync mode leaves the mutex/cv mostly unused
    // (is_initialized is set true at end of sync ctor, so the blocking
    // helper is a single uncontended mutex acquire).
    // The stop-pattern members are 'mutable' since stop() is const
    // (see notes/stoppable_class.md).
    mutable std::mutex _mutex;
    mutable std::condition_variable _cv;
    mutable bool _is_stopped = false;
    mutable std::exception_ptr _error;
    mutable std::atomic<bool> _stop_flag{false};

    bool _is_initialized = false;

    // Async worker threads. Empty in sync mode.
    std::vector<std::thread> _workers;
    int _async_cuda_device = -1;     // captured by workers for cudaSetDevice()

    // Async chunking state. The zero-chunk partition is built so no
    // zero chunk straddles a register-chunk boundary:
    // _zero_chunk_starts[i+1] - _zero_chunk_starts[i] is at most
    // zero_chunk_bytes, and _super_of_zero_chunk[i] is the register
    // chunk index that contains zero chunk i. _zero_chunks_per_super[s]
    // is the count the registrar waits for. When there is no registrar
    // (zero workers only), _super_done is empty and _super_of_zero_chunk
    // is empty too (zero workers skip the signal).
    std::atomic<long> _next_zero_chunk{0};
    std::vector<std::atomic<int>> _super_done;
    std::atomic<int>  _workers_remaining{0};   // last-out-finalizes counter (zero-only path)
    long _nzero_chunks = 0;
    std::vector<long> _zero_chunk_starts;
    std::vector<int>  _super_of_zero_chunk;
    std::vector<int>  _zero_chunks_per_super;

    // Blocking helper for async mode (no-op in sync mode). Throws on a
    // stopped allocator: the saved error if non-null, else a generic
    // "<method_name> called on stopped instance".
    void _block_until_ready_or_throw(const char *method_name) const;

    // Marked private-ish (in struct so accessible to internals but
    // not part of the user API).
    void _finalize_initialized();

    // Allocates the backing region (using ksgpu) and, for af_rhost,
    // wraps it in a chained shared_ptr whose deleter unregisters the
    // successfully-registered chunks before dropping the ksgpu keepalive.
    // Writes _nreg_chunks and _deleter_state (for the af_rhost path).
    // Called from the ctor's init list; reads `this->aflags` and
    // `this->capacity` (initialized first per declaration order).
    // Returns null for dummy/empty mode (capacity <= 0).
    std::shared_ptr<void> _allocate_base(int cuda_device);

    // Init paths (called from the ctor body, after `base` is set).
    void _init_sync(int cuda_device);
    void _init_async(int nthreads, int cuda_device);

    // Wraps `ksgpu_base` in a chained shared_ptr whose deleter
    // unregisters the successfully-registered chunks, then drops a
    // captured keepalive so the ksgpu deleter (munmap) fires.
    // Initializes _deleter_state and _nreg_chunks. Returns the chained
    // shared_ptr (used to initialize `base` in the init list).
    std::shared_ptr<void> _setup_rhost_deleter(std::shared_ptr<void> ksgpu_base);

    // Serial chunked-register loop, used by the sync path. Throws on
    // cudaHostRegister failure; the chained deleter (already installed
    // by _setup_rhost_deleter) handles partial-failure cleanup during
    // the ctor stack unwind.
    void _register_chunks_serially();

    // Build _zero_chunk_starts / _super_of_zero_chunk /
    // _zero_chunks_per_super / _super_done for the async path. If
    // has_zero_workers is false, only _super_done is sized (when a
    // registrar is present): _zero_chunk_starts, _super_of_zero_chunk,
    // and _zero_chunks_per_super stay empty, so no zero work is
    // dispatched.
    void _build_async_chunk_layout(bool has_zero_workers);

    // Worker bodies.
    void _zero_worker();
    void _registrar_worker();
    void _gpu_memset_worker();

    // Static helper: build the absolute-aligned register chunk offset
    // table. Returns a vector of size n_chunks + 1; chunk i covers
    // [returned[i], returned[i+1]). The first chunk may be partial; all
    // others are constants::cuda_host_register_chunk_size except the
    // last which may also be partial.
    static std::vector<long>
    _build_reg_chunk_offsets(const void *base_raw, long size);
};


// -------------------------------------------------------------------------------------------------
//
// Template implementations.


// Note: the allocate_array() overloads are entry points; per the strict
// stoppable-class policy (notes/stoppable_class.md), ANY throw (including
// dtype/shape argument errors) stops the allocator.

template<typename T>
ksgpu::Array<T> BumpAllocator::allocate_array(std::initializer_list<long> shape)
{
    static_assert(!std::is_void_v<T>, "BumpAllocator::allocate_array<void>() requires explicit dtype");
    try {
        ksgpu::Dtype dtype = ksgpu::Dtype::native<T>();
        ksgpu::Array<void> ret = _allocate_array_internal(dtype, shape.size(), shape.begin(), nullptr);
        return ret.template cast<T>("BumpAllocator::allocate_array()");
    } catch (...) {
        stop(std::current_exception());
        throw;
    }
}


template<typename T>
ksgpu::Array<T> BumpAllocator::allocate_array(ksgpu::Dtype dtype, std::initializer_list<long> shape)
{
    try {
        ksgpu::_check_dtype<T>(dtype, "BumpAllocator::allocate_array()");
        ksgpu::Array<void> ret = _allocate_array_internal(dtype, shape.size(), shape.begin(), nullptr);
        return ret.template cast<T>("BumpAllocator::allocate_array()");
    } catch (...) {
        stop(std::current_exception());
        throw;
    }
}


template<typename T>
ksgpu::Array<T> BumpAllocator::allocate_array(const std::vector<long> &shape, const std::vector<long> &strides)
{
    static_assert(!std::is_void_v<T>, "BumpAllocator::allocate_array<void>() requires explicit dtype");
    try {
        if (shape.size() != strides.size())
            throw std::runtime_error("BumpAllocator::allocate_array(): shape/strides size mismatch");

        ksgpu::Dtype dtype = ksgpu::Dtype::native<T>();
        ksgpu::Array<void> ret = _allocate_array_internal(dtype, shape.size(), &shape[0], &strides[0]);
        return ret.template cast<T>("BumpAllocator::allocate_array()");
    } catch (...) {
        stop(std::current_exception());
        throw;
    }
}


template<typename T>
ksgpu::Array<T> BumpAllocator::allocate_array(ksgpu::Dtype dtype, const std::vector<long> &shape, const std::vector<long> &strides)
{
    try {
        ksgpu::_check_dtype<T>(dtype, "BumpAllocator::allocate_array()");

        if (shape.size() != strides.size())
            throw std::runtime_error("BumpAllocator::allocate_array(): shape/strides size mismatch");

        ksgpu::Array<void> ret = _allocate_array_internal(dtype, shape.size(), &shape[0], &strides[0]);
        return ret.template cast<T>("BumpAllocator::allocate_array()");
    } catch (...) {
        stop(std::current_exception());
        throw;
    }
}


}  // namespace pirate

#endif // _PIRATE_BUMP_ALLOCATOR_HPP
