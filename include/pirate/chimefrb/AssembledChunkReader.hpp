#ifndef _PIRATE_CHIMEFRB_ASSEMBLED_CHUNK_READER_HPP
#define _PIRATE_CHIMEFRB_ASSEMBLED_CHUNK_READER_HPP

#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <thread>
#include <exception>
#include <condition_variable>

#include "AssembledChunk.hpp"

namespace pirate {
class SlabAllocator;   // defined in SlabAllocator.hpp

namespace chimefrb {
#if 0
}}  // editor auto-indent
#endif


// AssembledChunkReader: reads a list of chimefrb data files with a pool of threads, and
// hands them to the caller one at a time, IN FILENAME ORDER.
//
//   AssembledChunkReader reader(filenames, 4);
//   while (auto chunk = reader.get_chunk())
//       ...   // chunks arrive in the order of 'filenames', whatever order they were read in
//
// From the caller's side this is the serial loop
//
//   for (const string &fn: filenames)
//       use(AssembledChunk::from_msgpack(fn));
//
// with the same order and the same exceptions -- but up to 'nthreads' files are read ahead
// of the caller, so a caller that does real work per chunk (a GPU RFI chain, say) does not
// wait on the disk. Reading a file is mostly I/O plus one large memcpy, which is what the
// pool overlaps; see AssembledChunk.hpp for the file format.
//
// MEMORY. At most 'nthreads' chunks exist inside the reader at once (some read, some being
// read). At the production CHIME geometry one chunk owns a ~17 MB buffer, so the default
// nthreads=4 costs about 68 MB. A caller that RETAINS the chunks it takes defeats that
// bound: a chunk keeps its buffer alive, and so does any array view of it.
//
// READ ERRORS ARE DELIVERED IN ORDER, which is a deliberate departure from the usual
// thread-backed rule that a worker throwing calls stop() at once. A worker that fails on
// file 7 stores the exception with file 7 and carries on; get_chunk() rethrows it when it
// reaches that file, so files 0-6 are still delivered first, exactly as a serial loop would.
// Rethrowing then stops the reader, so nothing is read past the failure and every later
// get_chunk() rethrows the same error.
//
// The exception is from_msgpack()'s own, unwrapped. Every error the FILE can cause names it
// ("chimefrb: <filename>: ..."), which is why the reader adds nothing -- but a resource
// failure inside from_msgpack() (an allocation, an Array invariant) does not, and is then
// indistinguishable from the same failure anywhere else.
//
// Follows the "thread-backed class" pattern (notes/thread_backed_class.md): stop() puts the
// object in a stopped state and wakes every thread, and the destructor calls stop() and
// joins. Stop-reporting follows the documented "done-value" variant (see "Error reporting"
// in notes/stoppable_class.md): a null stop() is normal termination and get_chunk() then
// returns an empty pointer, while stop(e) is an error shutdown and get_chunk() rethrows e.
//
// The worker threads inherit the constructing thread's vcpu affinity.

struct AssembledChunkReader
{
    // Reads 'filename_list' in order, with min(nthreads, filename_list.size()) worker
    // threads. An empty list is legal: get_chunk() returns an empty pointer immediately.
    //
    // 'nthreads' must be >= 1, and is clamped to the number of files. The clamped value is
    // BOTH the worker count and the read-ahead depth (see 'nthreads' below).
    //
    // If 'allocator' is non-null, every chunk's buffer is taken from it instead of being
    // allocated per file. Three things to know before passing one:
    //
    //   - A SlabAllocator serves ONE slab size, fixed by its first use, so every file in
    //     the list must have identical parameters (nupfreq, nt_per_packet, nrfifreq). A
    //     file that does not is reported as that file's read error, naming the parameters.
    //
    //   - THE READER STOPS THE ALLOCATOR when it stops, which includes its own destructor,
    //     and SlabAllocator::stop() propagates on into the underlying BumpAllocator. So the
    //     allocator must be this reader's alone -- do not share one with anything that
    //     outlives the reader. (The cascade is not optional: an exhausted pool blocks
    //     get_slab(), and only the allocator's own stop() can wake a worker parked there.)
    //
    //   - THE POOL MUST HOLD AT LEAST 'nthreads' SLABS. This is a requirement, not a
    //     performance hint: a smaller pool DEADLOCKS. The reader keeps up to 'nthreads'
    //     files outstanding, and an already-read one holds its slab until get_chunk()
    //     hands it over -- so if a later file takes the last slab, the worker reading the
    //     file the caller is waiting for blocks in get_slab() and nothing can progress.
    //     (Verified: 4 threads against a one-slab pool hangs on the first try.) A dummy-mode
    //     SlabAllocator, which af_alloc()s every slab and never blocks, is always safe.
    AssembledChunkReader(const std::vector<std::string> &filename_list, long nthreads = 4,
                         const std::shared_ptr<SlabAllocator> &allocator
                             = std::shared_ptr<SlabAllocator>());

    // Noncopyable, nonmovable.
    AssembledChunkReader(const AssembledChunkReader &) = delete;
    AssembledChunkReader &operator=(const AssembledChunkReader &) = delete;
    AssembledChunkReader(AssembledChunkReader &&) = delete;
    AssembledChunkReader &operator=(AssembledChunkReader &&) = delete;

    // Destructor: stop() (which also stops the allocator, if there is one), then join.
    ~AssembledChunkReader();

    // Entry point: block until the next file in 'filenames' has been read, and return it.
    //
    // Returns an empty pointer at the end of the list, and on every call after that. Also
    // returns an empty pointer if the reader was cleanly stopped, even with chunks already
    // read; rethrows the stored error if it was error-stopped.
    //
    // Rethrows this file's own read error, if it had one, having first stopped the reader
    // (see READ ERRORS above, including what its text does and does not name).
    //
    // Thread-safe, but meant for ONE consumer thread: it is the "i-th call returns the i-th
    // file" contract that makes this a drop-in for a serial loop. With several consumers
    // each chunk is still delivered exactly once, but which thread gets which is undefined.
    std::shared_ptr<AssembledChunk> get_chunk();

    // Put the reader into the stopped state and wake every thread. First caller wins
    // (stores 'e'); later callers return immediately. Also stops 'allocator' if there is
    // one -- see the constructor. Thread-safe, callable from any thread.
    void stop(std::exception_ptr e = nullptr) const;

    // Stopped-tolerant informational accessor: no stopped-state check, since its
    // last-known value stays meaningful after a stop.
    bool get_is_stopped() const;

    // ----- Constants (immutable after construction; NOT lock-protected) -----

    const std::vector<std::string> filenames;
    const long nfiles;

    // Worker threads, and the read-ahead depth: at most 'nthreads' files are outstanding
    // (claimed by a worker but not yet returned by get_chunk()) at any instant. One number
    // serves as both -- see AssembledChunkReader.cpp for why a deeper queue would not help.
    // Clamped to nfiles by the constructor, so it is 0 for an empty file list.
    const long nthreads;

    // The caller's allocator, or null. Stopped by stop() -- see the constructor.
    const std::shared_ptr<SlabAllocator> allocator;

    // ----- Synchronization (everything below is protected by 'lock') -----

    mutable std::mutex lock;

    // cv_ready -- waiter: the consumer in get_chunk() (predicate: the slot for
    //   'next_return' is ready, or the list is exhausted, or stopped).
    //   Signaled on: a worker publishing a completed slot, stop(). notify_all rather than
    //   notify_one, because get_chunk() permits (does not expect) several consumers, and
    //   one completed file satisfies only the one waiting for that index.
    mutable std::condition_variable cv_ready;

    // cv_claim -- waiters: idle worker threads (predicate: a file is left AND the window
    //   is open, i.e. next_claim < nfiles && next_claim < next_return + nthreads; or
    //   stopped). Signaled on: get_chunk() advancing 'next_return' (notify_one), stop()
    //   (notify_all).
    mutable std::condition_variable cv_claim;

    mutable bool is_stopped = false;
    mutable std::exception_ptr error;

    // One file's result. 'chunk' and 'error' are exclusive: a slot holds one or the other.
    struct Slot
    {
        std::shared_ptr<AssembledChunk> chunk;
        std::exception_ptr error;
        bool ready = false;
    };

    // The reorder buffer: file index j lives in ring[j % nthreads]. The modulo is exact,
    // not a heuristic -- slot j % nthreads is freed exactly when file j - nthreads is
    // returned, which is the same moment file j becomes claimable.
    std::vector<Slot> ring;

    long next_claim = 0;    // next file index a worker may claim
    long next_return = 0;   // next file index get_chunk() will return

    std::vector<std::thread> workers;

private:
    // Worker thread main, and its mandatory catch-all wrapper (an exception escaping a
    // thread is std::terminate). Note a FILE's read error never reaches the wrapper: it is
    // caught in _worker_main() and stored in the file's slot.
    void _worker_main();
    void worker_main();
};


}}  // namespace pirate::chimefrb

#endif // _PIRATE_CHIMEFRB_ASSEMBLED_CHUNK_READER_HPP
