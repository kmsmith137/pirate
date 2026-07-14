#include "../include/pirate/BumpAllocator.hpp"
#include "../include/pirate/inlines.hpp"    // align_up()

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
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

// The BumpAllocator code is intended to be as simple and readable
// as possible. We avoid special-case logic and write "unified" functions
// which test flags in a logical sequence. We don't take different code
// paths depending on whether nbytes >= 511 GiB. (In particular, we always
// implement af_rhost allocations by mmap-ing a page-aligned byte count,
// followed by "chunked" cudaHostRegister() calls aligned on pointer
// addresses, since this is the only viable approach for nbytes > 511 GiB.)

// Worker chunking parameters.
// Note: reg_chunk_bytes is 64 GiB by default. Lower for stress testing.
static constexpr long reg_chunk_bytes = constants::cuda_host_register_chunk_size;
static constexpr long zero_chunk_bytes = 128L << 20;     // 128 MiB
static constexpr long hugepage_size = constants::host_hugepage_size;   // 2 MiB
static constexpr long host_page_size = constants::host_page_size;      // 4 KiB


// -------------------------------------------------------------------------------------------------
// File-local helpers


// Validate aflags: exactly one of {af_uhost, af_rhost, af_gpu}, plus an
// optional subset of {af_mmap_huge, af_zero}. No other flags allowed.
// af_mmap_huge is not meaningful for af_gpu and is rejected if combined.
// Returns the validated aflags so callers can invoke this from the ctor's
// initializer list.
static int _validate_aflags(int aflags)
{
    const int allowed = ksgpu::af_uhost | ksgpu::af_rhost | ksgpu::af_gpu
                      | ksgpu::af_mmap_huge | ksgpu::af_zero;
    int extra = aflags & ~allowed;
    if (extra != 0) {
        std::stringstream ss;
        ss << "BumpAllocator: aflags=0x" << std::hex << aflags
           << " contains unsupported bits 0x" << extra
           << " (allowed: af_uhost | af_rhost | af_gpu | af_mmap_huge | af_zero)";
        throw std::runtime_error(ss.str());
    }
    int loc = !!(aflags & ksgpu::af_uhost)
            + !!(aflags & ksgpu::af_rhost)
            + !!(aflags & ksgpu::af_gpu);
    if (loc != 1)
        throw std::runtime_error(
            "BumpAllocator: aflags must contain exactly one of "
            "{af_uhost, af_rhost, af_gpu}");
    if ((aflags & ksgpu::af_gpu) && (aflags & ksgpu::af_mmap_huge))
        throw std::runtime_error(
            "BumpAllocator: af_mmap_huge cannot be combined with af_gpu "
            "(huge-page mapping is a host-memory concept)");
    return aflags;
}


// Compute the flags to pass to ksgpu::_af_alloc. For host paths we
// substitute af_rhost -> af_uhost (we own the registration) and force
// an mmap flag so the base is host-page-aligned (a cudaHostRegister
// requirement). af_zero is NOT forwarded; zeroing is owned by
// BumpAllocator.
static int _compute_alloc_flags(int aflags)
{
    if (aflags & ksgpu::af_gpu)
        return ksgpu::af_gpu;
    int mmap_flag = (aflags & ksgpu::af_mmap_huge)
                  ? ksgpu::af_mmap_huge
                  : ksgpu::af_mmap_small;
    return ksgpu::af_uhost | mmap_flag;
}


// Page-size alignment for the chosen allocation path.
static long _alloc_align(int aflags)
{
    if (aflags & ksgpu::af_gpu)
        return BumpAllocator::nalign;            // cudaMalloc alignment is much bigger
    if (aflags & ksgpu::af_mmap_huge)
        return hugepage_size;                    // 2 MiB
    return host_page_size;                       // 4 KiB
}


// Validate sync/async + dummy combination, and align capacity to the
// page/cache-line size implied by aflags. Returns capacity unchanged
// for capacity <= 0 (dummy or empty modes). Called from the ctor init
// list so that the const `capacity` member can be initialized directly.
static long _validate_and_align_capacity(int aflags, long capacity, bool async)
{
    if (async && capacity < 0)
        throw std::runtime_error(
            "BumpAllocator(async): capacity must be >= 0 (dummy mode not supported)");
    if (capacity <= 0)
        return capacity;
    return align_up(capacity,
                    std::max((long) BumpAllocator::nalign, _alloc_align(aflags)));
}


// -------------------------------------------------------------------------------------------------
// Constructor / destructor


BumpAllocator::BumpAllocator(int aflags_, long capacity_, bool async,
                              int nthreads, int cuda_device)
    : aflags(_validate_aflags(aflags_)),
      capacity(_validate_and_align_capacity(aflags_, capacity_, async)),
      base(_allocate_base(cuda_device))
{
    if (capacity <= 0) {
        // capacity < 0: dummy mode (sync only). capacity == 0: empty.
        // Either way, `base` is null and there is no work to do.
        _is_initialized = true;
        return;
    }

    if (async)
        _init_async(nthreads, cuda_device);
    else
        _init_sync(cuda_device);
}


BumpAllocator::~BumpAllocator()
{
    stop();
    for (auto &t : _workers)
        if (t.joinable())
            t.join();
    // Member destructors run after this body. The `base` shared_ptr's
    // deleter handles cleanup when the last reference drops.
}


// -------------------------------------------------------------------------------------------------
// Init paths


std::shared_ptr<void> BumpAllocator::_allocate_base(int cuda_device)
{
    // Called from the ctor init list: this->aflags and this->capacity are
    // already initialized (preceding members in declaration order), but
    // every member declared after `base` is not yet constructed.
    // _nreg_chunks and _deleter_state are declared before `base`
    // specifically so this function may write to them.

    if (capacity <= 0)
        return nullptr;     // dummy mode (< 0) or empty (== 0): no allocation.

    const bool is_gpu = aflags & ksgpu::af_gpu;
    const bool is_rhost = aflags & ksgpu::af_rhost;

    if (is_gpu && cuda_device < 0)
        throw std::runtime_error(
            "BumpAllocator: cuda_device must be >= 0 when af_gpu is set");

    // Allocate. For af_gpu, ensure cudaMalloc happens on the requested
    // device (ksgpu records this as the free-time device too).
    //
    // Note that the initial allocation (either mmap() or cudaMalloc()) is
    // synchronous, even if the BumpAllocator has async=true. This detail
    // is important for the GpuDedisperser, since it uses cudaIPC to share
    // memory with the grouper, and this only works if the initial allocation
    // is done with cudaMalloc (not cudaMallocAsync).
    
    int alloc_flags = _compute_alloc_flags(aflags);
    std::optional<ksgpu::CudaSetDevice> _scoped;
    if (is_gpu)
        _scoped.emplace(cuda_device);
    std::shared_ptr<void> ksgpu_base = ksgpu::_af_alloc(
        ksgpu::Dtype(ksgpu::df_uint, 8), capacity, alloc_flags);

    uintptr_t p = reinterpret_cast<uintptr_t>(ksgpu_base.get());
    xassert((p % nalign) == 0);

    if (is_rhost)
        return _setup_rhost_deleter(std::move(ksgpu_base));
    return ksgpu_base;
}


void BumpAllocator::_init_sync(int cuda_device)
{
    const bool is_gpu    = aflags & ksgpu::af_gpu;
    const bool is_rhost  = aflags & ksgpu::af_rhost;
    const bool need_zero = aflags & ksgpu::af_zero;

    // Chunked register on the main thread. A throw mid-loop unwinds the
    // ctor; the chained deleter installed by _allocate_base unregisters
    // the successfully-registered chunks and then triggers the ksgpu
    // deleter (munmap).
    if (is_rhost)
        _register_chunks_serially();

    // Zero pass. For mmap-based host paths (we always force af_mmap_huge
    // or af_mmap_small), MAP_ANONYMOUS guarantees zero, so we only need
    // an explicit memset for af_gpu.
    if (need_zero && is_gpu) {
        ksgpu::CudaSetDevice _scoped(cuda_device);
        CUDA_CALL(cudaMemset(base.get(), 0, capacity));
    }

    _is_initialized = true;
}


void BumpAllocator::_init_async(int nthreads, int cuda_device)
{
    const bool is_gpu    = aflags & ksgpu::af_gpu;
    const bool is_uhost  = aflags & ksgpu::af_uhost;
    const bool is_rhost  = aflags & ksgpu::af_rhost;
    const bool need_zero = aflags & ksgpu::af_zero;

    // Async-specific argument validation.
    if (is_rhost && cuda_device < 0)
        throw std::runtime_error(
            "BumpAllocator(async): cuda_device must be >= 0 when af_rhost is set");
    if (need_zero && is_rhost && nthreads < 2)
        throw std::runtime_error(
            "BumpAllocator(async): nthreads must be >= 2 for af_rhost + af_zero "
            "(1 registrar + >= 1 zero worker)");
    if (need_zero && is_uhost && nthreads < 1)
        throw std::runtime_error(
            "BumpAllocator(async): nthreads must be >= 1 for af_uhost + af_zero");

    // Async work to do (some combination of these may be needed).
    const bool host_memset = need_zero && (is_uhost || is_rhost);
    const bool registrar   = is_rhost;
    const bool gpu_memset  = need_zero && is_gpu;

    if (!host_memset && !registrar && !gpu_memset) {
        // af_uhost or af_gpu without af_zero: nothing to do async.
        _is_initialized = true;
        return;
    }

    _async_cuda_device = cuda_device;

    // Build _super_done (if registrar) and zero-chunk partition (if
    // host_memset). _build_async_chunk_layout handles both based on
    // _nreg_chunks (which is > 0 iff is_rhost, set by
    // _setup_rhost_deleter) and the explicit has_zero_workers flag.
    if (host_memset || registrar)
        _build_async_chunk_layout(host_memset);

    int n_zero_workers = host_memset
                       ? (registrar ? nthreads - 1 : nthreads)
                       : 0;
    if (host_memset)
        _workers_remaining.store(n_zero_workers);

    int total_workers = n_zero_workers + (registrar ? 1 : 0) + (gpu_memset ? 1 : 0);
    _workers.reserve(total_workers);
    try {
        if (registrar)
            _workers.emplace_back([this, keepalive = base]() {
                _registrar_worker();
            });
        for (int i = 0; i < n_zero_workers; i++)
            _workers.emplace_back([this, keepalive = base]() {
                _zero_worker();
            });
        if (gpu_memset)
            _workers.emplace_back([this, keepalive = base]() {
                _gpu_memset_worker();
            });
    } catch (...) {
        stop(std::current_exception());
        for (auto &t : _workers)
            if (t.joinable()) t.join();
        throw;
    }
}


// -------------------------------------------------------------------------------------------------
// Setup helpers


std::vector<long>
BumpAllocator::_build_reg_chunk_offsets(const void *base_raw, long size)
{
    // Register chunk boundaries at absolute reg_chunk_bytes-aligned host
    // addresses. First chunk runs from base_raw to the next aligned
    // address (head; possibly partial), then full chunks of
    // reg_chunk_bytes, then a partial trailing chunk if size isn't a
    // multiple of reg_chunk_bytes. Entries are byte offsets relative
    // to base_raw, with offsets[0] == 0 and offsets[back] == size.
    std::vector<long> offsets;
    offsets.push_back(0);
    uintptr_t up = reinterpret_cast<uintptr_t>(base_raw);
    long head = static_cast<long>(reg_chunk_bytes - (up & (reg_chunk_bytes - 1)));
    if (head < size) {
        long off = head;
        while (off < size) {
            offsets.push_back(off);
            off += reg_chunk_bytes;
        }
    }
    offsets.push_back(size);
    return offsets;
}


std::shared_ptr<void> BumpAllocator::_setup_rhost_deleter(std::shared_ptr<void> ksgpu_base)
{
    std::vector<long> reg_offsets =
        _build_reg_chunk_offsets(ksgpu_base.get(), capacity);
    _nreg_chunks = static_cast<long>(reg_offsets.size()) - 1;

    _deleter_state = std::make_shared<DeleterState>();
    _deleter_state->reg_chunk_offsets = reg_offsets;

    // Wrap base in a chained shared_ptr whose deleter unregisters
    // successful chunks, then drops the captured keepalive (ksgpu's
    // shared_ptr) so ksgpu's deleter (munmap) fires.
    //
    // Read the raw pointer BEFORE std::move'ing ksgpu_base into the
    // lambda capture, since std::move leaves the source in a moved-from
    // (empty) state where .get() may return nullptr.
    void *raw = ksgpu_base.get();
    return std::shared_ptr<void>(
        raw,
        [state = _deleter_state, keepalive = std::move(ksgpu_base)](void *p) {
            int n = state->n_registered.load(std::memory_order_acquire);
            char *cp = static_cast<char *>(p);
            for (int s = 0; s < n; s++) {
                long off = state->reg_chunk_offsets[s];
                cudaError_t err = cudaHostUnregister(cp + off);
                if (err != cudaSuccess) {
                    std::fprintf(stderr,
                        "BumpAllocator deleter: cudaHostUnregister chunk "
                        "%d (offset=%ld) failed: %s\n",
                        s, off, cudaGetErrorString(err));
                }
            }
            // keepalive drops at end of lambda -> ksgpu deleter (munmap).
        });
}


void BumpAllocator::_register_chunks_serially()
{
    const auto &offsets = _deleter_state->reg_chunk_offsets;
    char *cp = static_cast<char *>(this->base.get());
    long n_chunks = static_cast<long>(offsets.size()) - 1;
    for (long s = 0; s < n_chunks; s++) {
        long off = offsets[s];
        long sz  = offsets[s + 1] - off;
        cudaError_t err = cudaHostRegister(cp + off, sz, cudaHostRegisterDefault);
        if (err != cudaSuccess)
            throw ksgpu::make_cuda_exception(err, "cudaHostRegister",
                                              __FILE__, __LINE__);
        // Increment AFTER success, so a throw mid-loop leaves
        // n_registered as the count of successfully-registered chunks.
        _deleter_state->n_registered.fetch_add(1, std::memory_order_release);
    }
}


void BumpAllocator::_build_async_chunk_layout(bool has_zero_workers)
{
    // _nreg_chunks > 0 iff _setup_rhost_deleter was called, iff af_rhost.
    // Size _super_done unconditionally when we have a registrar (zero
    // workers may or may not be present; the registrar still reads it).
    if (_nreg_chunks > 0) {
        _super_done = std::vector<std::atomic<int>>(_nreg_chunks);
        // Explicit zero-init: std::atomic<int>'s default ctor does
        // value-initialize to 0 (so the vector ctor leaves these at 0),
        // but that corner of the spec is easy to forget -- make the
        // invariant visible.
        for (long s = 0; s < _nreg_chunks; s++)
            _super_done[s].store(0, std::memory_order_relaxed);
    }

    if (!has_zero_workers)
        return;

    _zero_chunk_starts.clear();
    _super_of_zero_chunk.clear();
    _zero_chunks_per_super.clear();
    _zero_chunk_starts.push_back(0);

    if (_nreg_chunks > 0) {
        // With a registrar: clip zero chunks at every register boundary
        // so no zero chunk straddles, and each zero chunk signals
        // exactly one _super_done counter.
        const auto &reg_offsets = _deleter_state->reg_chunk_offsets;
        _zero_chunks_per_super.assign(_nreg_chunks, 0);
        for (long s = 0; s < _nreg_chunks; s++) {
            long lo = reg_offsets[s];
            long hi = reg_offsets[s + 1];
            for (long cur = lo; cur < hi; ) {
                long next = std::min(cur + zero_chunk_bytes, hi);
                _zero_chunk_starts.push_back(next);
                _super_of_zero_chunk.push_back(static_cast<int>(s));
                _zero_chunks_per_super[s]++;
                cur = next;
            }
        }
    } else {
        // No registrar: uniform zero chunks. _super_of_zero_chunk and
        // _zero_chunks_per_super stay empty (zero workers don't signal).
        for (long cur = 0; cur < capacity; ) {
            long next = std::min(cur + zero_chunk_bytes, capacity);
            _zero_chunk_starts.push_back(next);
            cur = next;
        }
    }
    _nzero_chunks = static_cast<long>(_zero_chunk_starts.size()) - 1;
    _next_zero_chunk.store(0);
}


// -------------------------------------------------------------------------------------------------
// Worker bodies


void BumpAllocator::_zero_worker()
{
    // _nreg_chunks > 0 iff there's a registrar worker running.
    const bool has_registrar = (_nreg_chunks > 0);

    try {
        while (true) {
            if (_stop_flag.load(std::memory_order_relaxed)) break;
            long i = _next_zero_chunk.fetch_add(1, std::memory_order_relaxed);
            if (i >= _nzero_chunks) break;
            long off = _zero_chunk_starts[i];
            long sz  = _zero_chunk_starts[i + 1] - off;
            memset(static_cast<char *>(base.get()) + off, 0, sz);
            if (has_registrar) {
                int super = _super_of_zero_chunk[i];
                _super_done[super].fetch_add(1, std::memory_order_release);
            }
        }
    } catch (...) {
        stop(std::current_exception());
    }

    // Last-out finalizes. (When a registrar is present, the registrar
    // finalizes after its last chunk -- by construction it finishes after
    // all zero workers.) This block must run on EVERY path -- including
    // after a caught error above -- so it sits outside the main try; its
    // own try/catch keeps a pathological throw (e.g. a mutex failure in
    // _finalize_initialized) from escaping the thread (std::terminate).
    try {
        if (!has_registrar) {
            if (_workers_remaining.fetch_sub(1, std::memory_order_acq_rel) == 1)
                _finalize_initialized();
        }
    } catch (...) {
        stop(std::current_exception());
    }
}


void BumpAllocator::_registrar_worker()
{
    try {
        // Workers start with current device 0 by default; set explicitly.
        CUDA_CALL(cudaSetDevice(_async_cuda_device));

        const auto &offsets = _deleter_state->reg_chunk_offsets;
        for (long s = 0; s < _nreg_chunks; s++) {
            // Check the stop flag once per chunk. Without this, the fast path
            // (spin-wait predicate already satisfied, e.g. no zero workers)
            // would proceed straight to cudaHostRegister on every remaining
            // chunk, and a stop()/destructor could block for the full
            // remaining registration time.
            if (_stop_flag.load(std::memory_order_relaxed)) return;

            // If zero workers are present, wait until they've completed
            // all zero chunks belonging to this super. If not, needed=0
            // and we proceed immediately.
            int needed = _zero_chunks_per_super.empty()
                       ? 0
                       : _zero_chunks_per_super[s];
            while (_super_done[s].load(std::memory_order_acquire) < needed) {
                if (_stop_flag.load(std::memory_order_relaxed)) return;
                std::this_thread::yield();
            }

            long off = offsets[s];
            long sz  = offsets[s + 1] - off;
            cudaError_t err = cudaHostRegister(
                static_cast<char *>(base.get()) + off, sz,
                cudaHostRegisterDefault);
            if (err != cudaSuccess) {
                stop(std::make_exception_ptr(
                    ksgpu::make_cuda_exception(err, "cudaHostRegister",
                                                __FILE__, __LINE__)));
                return;
            }
            // Increment AFTER success. The deleter scans
            // n_registered chunks; an off-by-one would leak a
            // registration or unregister a never-registered chunk.
            _deleter_state->n_registered.fetch_add(1, std::memory_order_release);
        }
        _finalize_initialized();
    } catch (...) {
        stop(std::current_exception());
    }
}


void BumpAllocator::_gpu_memset_worker()
{
    try {
        CUDA_CALL(cudaSetDevice(_async_cuda_device));
        CUDA_CALL(cudaMemset(base.get(), 0, capacity));
        _finalize_initialized();
    } catch (...) {
        stop(std::current_exception());
    }
}


// -------------------------------------------------------------------------------------------------
// State machine: stop, finalize, blocking helper


void BumpAllocator::stop(std::exception_ptr e) const
{
    {
        std::lock_guard<std::mutex> guard(_mutex);
        if (_is_stopped) return;       // first stop wins
        _is_stopped = true;
        _error = e;                    // may be null (normal stop) or non-null (error)
        _stop_flag.store(true, std::memory_order_relaxed);
    }
    // Notify after releasing the mutex so woken threads aren't immediately
    // blocked re-acquiring it.
    _cv.notify_all();
}


void BumpAllocator::_finalize_initialized()
{
    {
        std::lock_guard<std::mutex> guard(_mutex);
        if (_is_stopped) return;       // a worker errored out and stopped us
        _is_initialized = true;
    }
    _cv.notify_all();
}


void BumpAllocator::_block_until_ready_or_throw(const char *method_name) const
{
    std::unique_lock<std::mutex> guard(_mutex);
    while (!_is_initialized && !_is_stopped)
        _cv.wait(guard);
    if (_is_stopped && _error)
        std::rethrow_exception(_error);
    if (_is_stopped)
        throw std::runtime_error(std::string(method_name) + " called on stopped instance");
}


void BumpAllocator::wait_until_initialized() const
{
    _block_until_ready_or_throw("BumpAllocator::wait_until_initialized");
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


// Note: the public allocation methods are entry points; per the strict
// stoppable-class policy (notes/stoppable_class.md), ANY throw (including
// dummy-mode and argument errors) stops the allocator.

std::shared_ptr<void> BumpAllocator::get_base() const
{
    try {
        if (capacity < 0)
            throw std::runtime_error("BumpAllocator::get_base() called in dummy mode (capacity < 0)");
        _block_until_ready_or_throw("BumpAllocator::get_base");
        return base;
    } catch (...) {
        stop(std::current_exception());
        throw;
    }
}


void *BumpAllocator::allocate_bytes(long nbytes)
{
    try {
        return _allocate_bytes(nbytes);
    } catch (...) {
        stop(std::current_exception());
        throw;
    }
}


void *BumpAllocator::_allocate_bytes(long nbytes)
{
    if (capacity < 0)
        throw std::runtime_error("BumpAllocator::allocate_bytes() called in dummy mode (capacity < 0)");

    if (nbytes < 0) {
        std::stringstream ss;
        ss << "BumpAllocator::allocate_bytes(): nbytes=" << nbytes << " is negative";
        throw std::runtime_error(ss.str());
    }

    _block_until_ready_or_throw("BumpAllocator::allocate_bytes");

    // The nbytes == 0 no-op comes AFTER the readiness gate, so a zero-size
    // allocation on a stopped allocator throws like any other allocation
    // (and never returns nullptr from a not-yet-initialized allocator).
    if (nbytes == 0)
        return nullptr;

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
        // Block (and throw) on stop in both branches. In dummy mode
        // (sync-only) this is a non-blocking check: it returns
        // immediately unless stop() has been called.
        _block_until_ready_or_throw("BumpAllocator::allocate_array");
        if (capacity < 0) {
            // Dummy mode: allocate a fresh array with af_alloc().
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
