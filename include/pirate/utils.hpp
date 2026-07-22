#ifndef _PIRATE_UTILS_HPP
#define _PIRATE_UTILS_HPP

#include <string>
#include <sstream>
#include <string_view>
#include <cuda_runtime.h>   // cudaStream_t
#include <ksgpu/Array.hpp>

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// -------------------------------------------------------------------------------------------------
//
// Serialized output: atomic_print() and class AtomicPrint.
//
// ALL pirate output goes through this funnel (C++ and python alike), so that
// concurrent writers can never interleave mid-line. Two real bugs motivated
// it: chained operator<< from concurrent FrbServer threads, and python's
// print() -- which emits the message and its newline as two separate stream
// writes, letting a sibling thread slip in between.
//
// The funnel formats each message completely, then emits it with ONE write(2)
// under a process-global mutex. Both halves matter:
//
//   - the mutex serializes every in-process caller, C++ threads and python
//     threads alike (python reaches the same mutex through the pybind11
//     binding of atomic_print);
//
//   - the single syscall makes a line unsplittable even ACROSS PROCESSES
//     sharing the fd -- which no in-process lock can do. run_toy_grouper
//     spawns one child subprocess per grouper address, all inheriting the
//     parent's stdout, so this case is real.
//
// A side benefit: write(2) bypasses stdio and python buffering, so output
// reaches a redirected log immediately, with no flush() or PYTHONUNBUFFERED
// needed on our side.


// atomic_print(): the shared primitive underneath all pirate output.
//
// Intended for PYTHON callers, via the pybind11 binding of the same name.
// C++ callers should use class AtomicPrint below, which composes a message and
// then hands it here.
//
// Appends '\n' if 's' is nonempty and does not already end with one, then
// emits it as described above. Empty input is a no-op (an AtomicPrint that was
// never streamed to prints nothing) -- to print a blank line, pass "\n".
//
// Best-effort and noexcept: write errors (EPIPE, EBADF, ...) are swallowed,
// so this is safe to call from destructors and teardown paths. This is a
// deliberate exception to the usual check-and-throw rule, because
// ~AtomicPrint() calls it during exception unwinding; the argument checking
// lives in the AtomicPrint constructor instead.
extern void atomic_print(std::string_view s, int fd=1) noexcept;


// AtomicPrint: RAII line/block builder, and the interface for C++ callers.
//
// One-liner form -- emits at the end of the full expression, appending the
// newline for you:
//
//     AtomicPrint() << "FrbServer: beamset=" << b << ", ichunk=" << ichunk;
//     AtomicPrint(2) << "FrbServer: " << err_msg;      // 2 = stderr
//
// Block form -- accumulates, then emits once at end of scope as a single
// contiguous write, so the whole block lands together:
//
//     AtomicPrint a;
//     a << "FrbServer: connected to grouper at " << addr << "\n";
//     a << "FrbServer: waiting for X-engine node(s) at " << rpc_addr << "\n";
//
// Use it for ALL C++ output, including from single-threaded call sites where
// nothing could interleave today. Uniformity is the point: it keeps the
// "no raw std::cout/printf in src_lib" invariant greppable (see notes/cpp.md),
// and a site that is single-threaded today may not stay that way.
//
// Keep the scope tight: nothing is visible until emission, so an AtomicPrint
// held across a blocking call hides its output for the duration.
class AtomicPrint {
public:
    // Throws if fd < 0. Any valid fd is accepted (1 = stdout, 2 = stderr).
    explicit AtomicPrint(int fd=1);

    // Guard-like: scope determines the emission point, so copying or moving
    // one of these would make the output location hard to reason about.
    AtomicPrint(const AtomicPrint &) = delete;
    AtomicPrint &operator=(const AtomicPrint &) = delete;

    // Emits whatever has been composed -- including on an exception-unwinding
    // path, which is deliberate: a partial line shows how far we got.
    ~AtomicPrint();

    // Must stay a MEMBER template. A free operator<<(AtomicPrint &, const T &)
    // would not bind to the temporary in the one-liner form above.
    template<typename T>
    AtomicPrint &operator<<(const T &t) { ss << t; return *this; }

protected:
    std::ostringstream ss;
    int fd = 1;
};


// test_atomic_print(): concurrency smoke test, called from
// 'python -m pirate_frb test --aout' (see pirate_frb/tests/test_atomic_out.py).
// Spawns 'nthreads' threads which each emit 'nlines' lines to 'fd', mixing the
// one-liner and block forms of AtomicPrint with direct atomic_print() calls.
// The caller checks the resulting fd for spliced or missing lines.
extern void test_atomic_print(int fd, long nthreads, long nlines);


// safe_memcpy_{h2g,g2h}_{sync,async}(): wrappers around
// cudaMemcpy{,Async}() that split a copy at every absolute
// pirate::constants::cuda_host_register_chunk_size-aligned host address
// inside the requested range. Use these whenever the host pointer COULD
// lie in a hugepage-backed BumpAllocator -- such BumpAllocators register
// their backing memory in chunks (since cudaHostRegister has a ~511 GiB
// per-call ceiling), and a cudaMemcpyAsync() spanning two registration
// chunks would fail with cudaErrorInvalidValue.
//
// Splitting is unconditional (the wrappers do not check whether the
// host pointer is actually in a chunked allocator), which is harmless
// for non-chunked pointers and avoids a cudaPointerGetAttributes()
// lookup per call. nbytes == 0 is a no-op; nbytes < 0 throws.

extern void safe_memcpy_h2g_sync (void *dst, const void *src, long nbytes);
extern void safe_memcpy_g2h_sync (void *dst, const void *src, long nbytes);
extern void safe_memcpy_h2g_async(void *dst, const void *src, long nbytes,
                                   cudaStream_t stream);
extern void safe_memcpy_g2h_async(void *dst, const void *src, long nbytes,
                                   cudaStream_t stream);


// Diagnostic for 'pirate_frb revisit_512gb': mmap nbytes (with hugepages
// iff use_hugepages), prefault (for 4 KiB pages only -- hugepages are
// pre-committed by mmap), and attempt a single cudaHostRegister() on
// the entire region. Reports progress + result to stdout. Returns true
// if cudaHostRegister succeeded (which would mean the historical
// ~511 GiB driver cap has been lifted). Cleans up on either path.
extern bool revisit_512gb_inner(long nbytes, bool use_hugepages);


// Arguments must satisfy 0 <= i < pow2(nbits).
extern int bit_reverse_slow(int i, int nbits);

// If n=2^r, returns value of r.
// If n is not a power of 2, throws an exception.
extern int integer_log2(long n);

// rb_lag(): returns lag needed for two-stage dedispersion.
// The index 0 <= freq_coarse < pow2(stage2_rank) represents a coarse frequency.
// The index 0 <= dm_brev < pow2(stage1_rank) represents a **bit-reversed** delay.
// If uflag=true, then we're computing the upper half of a (stage1_rank+stage2_rank+1) tree.
extern int rb_lag(int freq_coarse, int dm_brev, int stage1_rank, int stage2_rank, bool uflag=false);

// Downsamples (freq,time) array by a factor 2 along either frequency or time axis.
// "Variance-preserving" normalization 1/sqrt(2).
extern void reference_downsample_freq(const ksgpu::Array<float> &in, ksgpu::Array<float> &out);
extern void reference_downsample_time(const ksgpu::Array<float> &in, ksgpu::Array<float> &out);

// dedisperse_non_incremental(): currently only used for testing the ReferenceTree,
// but I could imagine this being useful elsewhere some day. Dedispersion is done in
// place -- output index is a bit-reversed delay.
//
// Note: Input is a 2-d array with shape (nfreq, ntime*nspec).

extern void dedisperse_non_incremental(ksgpu::Array<float> &arr, long nspec);

// dedispersion_delay(): returns the dedispersion delay for a given (freq, dm_brev) pair.
// Used for testing.
extern long dedispersion_delay(int rank, long freq, long dm_brev);

extern std::string hex_str(uint x);


}  // namespace pirate

#endif // _PIRATE_UTILS_HPP

