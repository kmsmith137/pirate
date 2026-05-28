#include "../include/pirate/BumpAllocator.hpp"
#include "../include/pirate/constants.hpp"  // cuda_host_register_chunk_size
#include "../include/pirate/inlines.hpp"    // align_up()

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <thread>

#include <sys/mman.h>

#include <ksgpu/cuda_utils.hpp>
#include <ksgpu/mem_utils.hpp>
#include <ksgpu/xassert.hpp>

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// Async mode constants. Worker chunking parameters.
static constexpr long zero_chunk_bytes = 128L << 20;     // 128 MiB
static constexpr long reg_chunk_bytes = constants::cuda_host_register_chunk_size;  // 64 GiB
static constexpr long hugepage_size = 2L << 20;          // 2 MiB


// -------------------------------------------------------------------------------------------------
// Constructor / destructor


BumpAllocator::BumpAllocator(int aflags_, long capacity_, bool async, int nthreads, int cuda_device)
    : aflags(aflags_), capacity(capacity_)
{
    ksgpu::check_aflags(aflags, "BumpAllocator constructor");

    if (aflags & ksgpu::af_random)
        throw std::runtime_error("BumpAllocator constructor: af_random flag is not supported");

    if (aflags & ksgpu::af_guard)
        throw std::runtime_error("BumpAllocator constructor: af_guard flag is not supported");

    if (!async) {
        // ----- Sync mode. nthreads and cuda_device are ignored. -----
        if (capacity >= 0) {
            capacity = align_up(capacity, nalign);
            base = ksgpu::_af_alloc(ksgpu::Dtype(ksgpu::df_uint, 8), capacity, aflags);
            uintptr_t p = reinterpret_cast<uintptr_t>(base.get());
            xassert((p % nalign) == 0);
        }
        _is_initialized = true;
        return;
    }

    // ----- Async mode: precondition checks. -----
    if (capacity < 0)
        throw std::runtime_error(
            "BumpAllocator(async): capacity must be >= 0 (dummy mode not supported in async)");
    if (cuda_device < 0)
        throw std::runtime_error(
            "BumpAllocator(async): cuda_device must be >= 0; caller must specify explicitly");

    const int case1_flags = ksgpu::af_mmap_huge | ksgpu::af_rhost | ksgpu::af_zero;
    const int case2_flags = ksgpu::af_rhost | ksgpu::af_zero;
    const int case3_flags = ksgpu::af_gpu | ksgpu::af_zero;

    if (aflags == case1_flags) {
        if (nthreads < 2)
            throw std::runtime_error("BumpAllocator(async, case 1): nthreads must be >= 2");
        _async_init_case1(capacity, nthreads, cuda_device);
    }
    else if (aflags == case2_flags) {
        if (nthreads < 2)
            throw std::runtime_error("BumpAllocator(async, case 2): nthreads must be >= 2");
        _async_init_case2(capacity, nthreads, cuda_device);
    }
    else if (aflags == case3_flags) {
        // nthreads is ignored.
        _async_init_case3(capacity, cuda_device);
    }
    else {
        std::stringstream ss;
        ss << "BumpAllocator(async): aflags=0x" << std::hex << aflags
           << " is not supported. Must be exactly one of: "
           << "(af_mmap_huge|af_rhost|af_zero), (af_rhost|af_zero), or (af_gpu|af_zero). "
           << "Generality may be added in the future.";
        throw std::runtime_error(ss.str());
    }
}


BumpAllocator::~BumpAllocator()
{
    stop();
    for (auto &t : _workers)
        if (t.joinable())
            t.join();
    // Member destructors run after this body. The `base` shared_ptr's
    // deleter (set in the async ctor, or by ksgpu in sync ctor) handles
    // cleanup when the last reference drops.
}


// -------------------------------------------------------------------------------------------------
// Async init helpers


void BumpAllocator::_async_init_case1(long capacity_arg, int nthreads, int cuda_device)
{
    capacity = align_up(capacity_arg, nalign);
    long mmap_size = align_up(capacity, hugepage_size);

    void *base_raw = mmap(nullptr, mmap_size, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
    if (base_raw == MAP_FAILED) {
        int e = errno;
        std::stringstream ss;
        ss << "BumpAllocator(async, case 1): mmap(MAP_HUGETLB, " << mmap_size
           << " bytes) failed: " << strerror(e);
        throw std::runtime_error(ss.str());
    }

    // RAII guard: munmap if anything between here and base setup throws.
    std::unique_ptr<void, std::function<void(void *)>> mmap_guard(
        base_raw, [mmap_size](void *p) { munmap(p, mmap_size); });

    // Register chunk boundaries: aligned to absolute
    // cuda_host_register_chunk_size-aligned host addresses. The first
    // chunk runs from base to the next aligned address (head chunk), then
    // full chunks of reg_chunk_bytes each, then a partial trailing chunk
    // (if mmap_size isn't a multiple of reg_chunk_bytes). All entries
    // are byte offsets relative to base_raw.
    std::vector<long> reg_offsets;
    reg_offsets.push_back(0);
    {
        uintptr_t up = reinterpret_cast<uintptr_t>(base_raw);
        long head = static_cast<long>(reg_chunk_bytes - (up & (reg_chunk_bytes - 1)));
        if (head >= mmap_size) {
            // Whole allocation fits in the head chunk; no further boundaries needed.
        } else {
            long off = head;
            while (off < mmap_size) {
                reg_offsets.push_back(off);
                off += reg_chunk_bytes;
            }
        }
    }
    reg_offsets.push_back(mmap_size);
    _nreg_chunks = static_cast<long>(reg_offsets.size()) - 1;

    _async_state = std::make_shared<AsyncDeleterState>();
    _async_state->size = mmap_size;
    _async_state->cuda_device = cuda_device;
    _async_state->reg_chunk_offsets = reg_offsets;  // captured by deleter

    base = std::shared_ptr<void>(base_raw, [state = _async_state](void *p) {
        int n = state->n_registered.load(std::memory_order_acquire);
        char *cp = static_cast<char *>(p);
        for (int s = 0; s < n; s++) {
            long off = state->reg_chunk_offsets[s];
            cudaError_t err = cudaHostUnregister(cp + off);
            if (err != cudaSuccess) {
                std::fprintf(stderr,
                    "BumpAllocator(async, case 1) deleter: cudaHostUnregister "
                    "chunk %d (offset=%ld) failed: %s\n",
                    s, off, cudaGetErrorString(err));
            }
        }
        if (munmap(p, state->size) != 0) {
            std::fprintf(stderr,
                "BumpAllocator(async, case 1) deleter: munmap failed: %s\n",
                std::strerror(errno));
        }
    });
    mmap_guard.release();   // base now owns the cleanup

    // Zero chunks: partition [0, mmap_size) into pieces of <= zero_chunk_bytes,
    // making sure no zero chunk straddles a register-chunk boundary. We do
    // this per register chunk so a straddler is split at the boundary.
    _zero_chunk_starts.clear();
    _super_of_zero_chunk.clear();
    _zero_chunks_per_super.assign(_nreg_chunks, 0);
    _zero_chunk_starts.push_back(0);
    for (long s = 0; s < _nreg_chunks; s++) {
        long lo = reg_offsets[s];
        long hi = reg_offsets[s + 1];
        long cur = lo;
        while (cur < hi) {
            long next = std::min(cur + zero_chunk_bytes, hi);
            _zero_chunk_starts.push_back(next);
            _super_of_zero_chunk.push_back(static_cast<int>(s));
            _zero_chunks_per_super[s]++;
            cur = next;
        }
    }
    _nzero_chunks = static_cast<long>(_super_of_zero_chunk.size());
    _next_zero_chunk.store(0);

    _super_done = std::vector<std::atomic<int>>(_nreg_chunks);
    for (long i = 0; i < _nreg_chunks; i++)
        _super_done[i].store(0, std::memory_order_relaxed);

    // Spawn workers: 1 registrar + (nthreads-1) zero workers. Each worker
    // captures a `base` keepalive so the deleter can't fire while a worker
    // is alive.
    _workers.reserve(nthreads);
    try {
        _workers.emplace_back([this, keepalive = base]() {
            _registrar_worker_case1();
        });
        for (int i = 1; i < nthreads; i++) {
            _workers.emplace_back([this, keepalive = base]() {
                _zero_worker_case1();
            });
        }
    } catch (...) {
        stop(std::current_exception());
        for (auto &t : _workers)
            if (t.joinable()) t.join();
        throw;
    }
}


void BumpAllocator::_async_init_case2(long capacity_arg, int nthreads, int cuda_device)
{
    capacity = align_up(capacity_arg, nalign);

    void *p = nullptr;
    {
        ksgpu::CudaSetDevice _scoped(cuda_device);
        cudaError_t err = cudaHostAlloc(&p, capacity, 0);
        if (err != cudaSuccess)
            throw ksgpu::make_cuda_exception(err, "cudaHostAlloc", __FILE__, __LINE__);
    }

    std::unique_ptr<void, std::function<void(void *)>> host_guard(
        p, [](void *q) { cudaFreeHost(q); });

    _async_state = std::make_shared<AsyncDeleterState>();
    _async_state->size = capacity;
    _async_state->cuda_device = cuda_device;

    base = std::shared_ptr<void>(p, [state = _async_state](void *q) {
        // cudaFreeHost doesn't require a current device on UVA systems.
        cudaError_t err = cudaFreeHost(q);
        if (err != cudaSuccess) {
            fprintf(stderr,
                    "BumpAllocator(async, case 2): cudaFreeHost failed: %s\n",
                    cudaGetErrorString(err));
            exit(1);
        }
    });
    host_guard.release();

    _nzero_chunks = (capacity + zero_chunk_bytes - 1) / zero_chunk_bytes;
    _next_zero_chunk.store(0);
    _workers_remaining.store(nthreads);

    _workers.reserve(nthreads);
    try {
        for (int i = 0; i < nthreads; i++) {
            _workers.emplace_back([this, keepalive = base]() {
                _zero_worker_case2();
            });
        }
    } catch (...) {
        stop(std::current_exception());
        for (auto &t : _workers)
            if (t.joinable()) t.join();
        throw;
    }
}


void BumpAllocator::_async_init_case3(long capacity_arg, int cuda_device)
{
    capacity = align_up(capacity_arg, nalign);

    void *p = nullptr;
    {
        ksgpu::CudaSetDevice _scoped(cuda_device);
        cudaError_t err = cudaMalloc(&p, capacity);
        if (err != cudaSuccess)
            throw ksgpu::make_cuda_exception(err, "cudaMalloc", __FILE__, __LINE__);
    }

    std::unique_ptr<void, std::function<void(void *)>> gpu_guard(
        p, [dev = cuda_device](void *q) {
            ksgpu::CudaSetDevice _scoped(dev);
            cudaFree(q);
        });

    _async_state = std::make_shared<AsyncDeleterState>();
    _async_state->size = capacity;
    _async_state->cuda_device = cuda_device;

    base = std::shared_ptr<void>(p, [state = _async_state](void *q) {
        ksgpu::CudaSetDevice _scoped(state->cuda_device);
        cudaError_t err = cudaFree(q);
        if (err != cudaSuccess) {
            fprintf(stderr,
                    "BumpAllocator(async, case 3): cudaFree failed: %s\n",
                    cudaGetErrorString(err));
            exit(1);
        }
    });
    gpu_guard.release();

    // Spawn exactly 1 memset worker (case 3 ignores nthreads).
    _workers.reserve(1);
    try {
        _workers.emplace_back([this, keepalive = base]() {
            _memset_worker_case3();
        });
    } catch (...) {
        stop(std::current_exception());
        for (auto &t : _workers)
            if (t.joinable()) t.join();
        throw;
    }
}


// -------------------------------------------------------------------------------------------------
// Worker bodies


void BumpAllocator::_zero_worker_case1()
{
    try {
        while (true) {
            if (_stop_flag.load(std::memory_order_relaxed)) return;
            long i = _next_zero_chunk.fetch_add(1, std::memory_order_relaxed);
            if (i >= _nzero_chunks) return;
            long off = _zero_chunk_starts[i];
            long sz  = _zero_chunk_starts[i + 1] - off;
            memset(static_cast<char *>(base.get()) + off, 0, sz);
            _super_done[_super_of_zero_chunk[i]].fetch_add(1, std::memory_order_release);
        }
    } catch (...) {
        stop(std::current_exception());
    }
}


void BumpAllocator::_registrar_worker_case1()
{
    try {
        // Workers start with current device 0 by default; set explicitly.
        CUDA_CALL(cudaSetDevice(_async_state->cuda_device));

        const auto &offsets = _async_state->reg_chunk_offsets;

        for (long s = 0; s < _nreg_chunks; s++) {
            int needed = _zero_chunks_per_super[s];

            while (_super_done[s].load(std::memory_order_acquire) < needed) {
                if (_stop_flag.load(std::memory_order_relaxed)) return;
                std::this_thread::yield();
            }

            long off = offsets[s];
            long sz  = offsets[s + 1] - off;
            cudaError_t err = cudaHostRegister(
                static_cast<char *>(base.get()) + off, sz, cudaHostRegisterDefault);
            if (err != cudaSuccess) {
                stop(std::make_exception_ptr(
                    ksgpu::make_cuda_exception(err, "cudaHostRegister", __FILE__, __LINE__)));
                return;
            }
            // Always increment AFTER a successful register, BEFORE the next
            // stop_flag check. If we increment before the cudaHostRegister
            // or skip the increment on a late stop, we'd leak the
            // registration.
            _async_state->n_registered.fetch_add(1, std::memory_order_release);
        }

        _finalize_initialized();
    } catch (...) {
        stop(std::current_exception());
    }
}


void BumpAllocator::_zero_worker_case2()
{
    try {
        while (true) {
            if (_stop_flag.load(std::memory_order_relaxed)) break;
            long i = _next_zero_chunk.fetch_add(1, std::memory_order_relaxed);
            if (i >= _nzero_chunks) break;
            long offset = i * zero_chunk_bytes;
            long sz = std::min(zero_chunk_bytes, _async_state->size - offset);
            memset(static_cast<char *>(base.get()) + offset, 0, sz);
        }
        // Last out finalizes.
        if (_workers_remaining.fetch_sub(1, std::memory_order_acq_rel) == 1)
            _finalize_initialized();
    } catch (...) {
        stop(std::current_exception());
        // Decrement so the counter still reaches zero (other workers
        // may also be exiting through their normal path).
        _workers_remaining.fetch_sub(1, std::memory_order_acq_rel);
    }
}


void BumpAllocator::_memset_worker_case3()
{
    try {
        CUDA_CALL(cudaSetDevice(_async_state->cuda_device));
        CUDA_CALL(cudaMemset(base.get(), 0, _async_state->size));
        _finalize_initialized();
    } catch (...) {
        stop(std::current_exception());
    }
}


// -------------------------------------------------------------------------------------------------
// State machine: stop, finalize, blocking helper


void BumpAllocator::stop(std::exception_ptr e)
{
    std::lock_guard<std::mutex> guard(_mutex);
    if (_is_stopped) return;       // first stop wins
    _is_stopped = true;
    _error = e;                    // may be null (normal stop) or non-null (error)
    _stop_flag.store(true, std::memory_order_relaxed);
    _cv.notify_all();
}


void BumpAllocator::_finalize_initialized()
{
    std::lock_guard<std::mutex> guard(_mutex);
    if (_is_stopped) return;       // a worker errored out and stopped us
    _is_initialized = true;
    _cv.notify_all();
}


void BumpAllocator::_block_until_ready_or_throw() const
{
    std::unique_lock<std::mutex> guard(_mutex);
    while (!_is_initialized && !_is_stopped)
        _cv.wait(guard);
    if (_is_stopped && _error)
        std::rethrow_exception(_error);
    if (_is_stopped)
        throw std::runtime_error("BumpAllocator method called on stopped instance");
}


void BumpAllocator::wait_until_initialized()
{
    _block_until_ready_or_throw();
}


bool BumpAllocator::is_initialized() const
{
    std::lock_guard<std::mutex> guard(_mutex);
    // Return true only if init has completed AND we haven't been stopped.
    // A stopped allocator -- whether with an error or via clean shutdown
    // (e.g., destructor) -- is not "ready to use".
    return _is_initialized && !_is_stopped;
}


// -------------------------------------------------------------------------------------------------
// Public allocation methods. All gate on _block_until_ready_or_throw (no-op
// in sync mode since _is_initialized is true from end of ctor).


std::shared_ptr<void> BumpAllocator::get_base() const
{
    if (capacity < 0)
        throw std::runtime_error("BumpAllocator::get_base() called in dummy mode (capacity < 0)");
    _block_until_ready_or_throw();
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

    _block_until_ready_or_throw();

    // Round up to alignment boundary.
    long aligned_nbytes = align_up(nbytes, nalign);

    // Atomically claim space using a compare-exchange loop. A simpler
    // fetch_add(aligned_nbytes) + roll-back-on-overflow pattern is racy:
    // the failing thread's transient over-increment of 'nbytes_allocated'
    // can cause concurrent allocations that would have fit to be rejected.
    // The CAS loop only advances 'nbytes_allocated' when the allocation
    // succeeds.
    long old_offset = nbytes_allocated.load(std::memory_order_relaxed);
    long new_offset;
    do {
        new_offset = old_offset + aligned_nbytes;
        if (new_offset > capacity) {
            std::stringstream ss;
            ss << "BumpAllocator::allocate_bytes(): allocation of " << nbytes
               << " bytes would exceed capacity " << capacity
               << " (currently allocated: " << old_offset << ")";
            throw std::runtime_error(ss.str());
        }
    } while (!nbytes_allocated.compare_exchange_weak(old_offset, new_offset));

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
    // Use ceiling division to handle sub-byte types (e.g., int4)
    long nbytes = (nalloc * dtype.nbits + 7) / 8;

    if (ret.size > 0) {
        if (capacity < 0) {
            // In dummy mode, allocate a fresh array with af_alloc().
            // Dummy mode is sync-only (precondition rejects async + dummy),
            // so no need to block here.
            ret.base = ksgpu::_af_alloc(dtype, nalloc, aflags);
            ret.data = ret.base.get();
            nbytes_allocated.fetch_add(align_up(nbytes, nalign));
        }
        else {
            // Normal mode: allocate from bump allocator. Blocks until ready
            // in async mode.
            _block_until_ready_or_throw();
            ret.base = this->base;
            ret.data = this->allocate_bytes(nbytes);
        }
    }

    ksgpu::_check_array_invariants_except_dtype(ret, "BumpAllocator::allocate_array()");
    return ret;
}


}  // namespace pirate
