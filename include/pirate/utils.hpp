#ifndef _PIRATE_UTILS_HPP
#define _PIRATE_UTILS_HPP

#include <string>
#include <vector>
#include <utility>
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
//
// 'line_pad' is the length of each line's filler run, and is a parameter because
// message length is what the atomicity property turns on: one write(2) per message
// is unsplittable up to PIPE_BUF, and only beyond it does a short write become a
// possibility at all. The caller draws it.
extern void test_atomic_print(int fd, long nthreads, long nlines, long line_pad = 200);


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


// Diagnostic for 'pirate_frb dev revisit_512gb': mmap nbytes (with hugepages
// iff use_hugepages), prefault (for 4 KiB pages only -- hugepages are
// pre-committed by mmap), and attempt a single cudaHostRegister() on
// the entire region. Reports progress + result to stdout. Returns true
// if cudaHostRegister succeeded (which would mean the historical
// ~511 GiB driver cap has been lifted). Cleans up on either path.
extern bool revisit_512gb_inner(long nbytes, bool use_hugepages);


// Arguments must satisfy 0 <= i < pow2(nbits).
// Note: we haven't bothered writing a fast bit_reverse function, since we don't
// currently need bit-reversal in any critical paths. The "_slow" name is intended to
// remind the caller that we could write a fast one if needed.
extern int bit_reverse_slow(int i, int nbits);

// test_utils(): unit test for bit_reverse_slow() above, and for the integer/bit
// helpers in inlines.hpp (is_power_of_two(), pow2(), popcount(), bit_length(),
// bit_floor(), integer_log2(), align_up(), round_{up,down}_to_power_of_two(),
// xdiv(), xmod()). Throws an exception on failure.
//
// Called from 'python -m pirate_frb test --util'. It exhausts the interesting
// part of its parameter space in one call and takes milliseconds, so the caller
// runs it once per invocation rather than once per test iteration.
extern void test_utils();

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


// -------------------------------------------------------------------------------------------------
//
// Shape draws shared by the kernel test_random()s.


// The streaming shape that every kernel test_random() needs: how many chunks, how long a
// chunk, how many beams per batch, how many batches. All of it comes out of ONE bounded
// product, so the total array footprint stays under 'budget' however the draw splits it --
// which is the property each of these tests needs and none can get from four independent
// draws.
//
// Shared because a v[] index means nothing on its own -- three kernel tests need the same
// index-to-meaning mapping, and a private copy of it in each would let them drift into
// covering different shapes with nothing to make that visible.
//
// 'nt_in_divisor' is the granularity nt_in_per_chunk must be a multiple of -- 32 for a
// float32 segment, or whatever random_nt_in_granularity() returned. 'nextra' asks for that
// many further factors OF THE SAME PRODUCT, returned in 'extra': a caller uses them for the
// per-kernel quantities that also have to fit inside the budget (an ambient rank, a
// weight-array DM count), and taking them from the same product is what keeps the budget a
// budget.
struct RandomKernelShape
{
    long nchunks = 0;
    long nt_in_per_chunk = 0;      // a multiple of the caller's nt_in_divisor
    long beams_per_batch = 0;
    long num_batches = 0;
    long total_beams = 0;          // = beams_per_batch * num_batches
    std::vector<long> extra;       // 'nextra' more factors of the same bounded product
};

extern RandomKernelShape random_kernel_shape(long budget, long nt_in_divisor, int nextra = 0);


// The input-time granularity the peak-finding and cdd2 kernels impose, as
// {nt_in_per_wt, nt_in_divisor}.
//
// nt_in_per_wt is the number of input samples one weight-array entry covers. A kernel with
// Tinner > 1 has no freedom there -- the weight stride is fixed by the register layout -- so
// the only randomized case is Tinner == 1, where the kernel accepts any power-of-two multiple
// of a segment and the draw picks one. nt_in_divisor is then the granularity
// PeakFindingKernelParams::validate() requires of nt_in.
extern std::pair<long,long> random_nt_in_granularity(long simd_width, long Tinner);


}  // namespace pirate

#endif // _PIRATE_UTILS_HPP

