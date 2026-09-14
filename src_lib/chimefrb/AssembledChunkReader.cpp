#include "../../include/pirate/chimefrb/AssembledChunkReader.hpp"
#include "../../include/pirate/SlabAllocator.hpp"

#include <algorithm>
#include <stdexcept>
#include <sstream>

using namespace std;

namespace pirate {
namespace chimefrb {
#if 0
}}  // editor auto-indent
#endif


// The read-ahead window is the thread count, with no separate knob, and it is worth saying
// why that does not throttle the pool. A worker may claim file j only while
// j < next_return + nthreads, so at most 'nthreads' files are outstanding. Then:
//
//   - If reading is the bottleneck (the case the pool exists for), the consumer is parked
//     at next_return, every slot in the window is claimable, and all threads are busy.
//   - If the CONSUMER is the bottleneck, steady state is a full window of already-read
//     chunks and idle threads; each get_chunk() frees one slot and wakes one worker. One
//     read in flight, which is all that is needed when the consumer is the slower side.
//
// A deeper queue would only help a bursty consumer, and would cost a proportional amount of
// memory -- a chunk is ~17 MB at the production geometry.


AssembledChunkReader::AssembledChunkReader(const vector<string> &filename_list, long nthreads_,
                                           const shared_ptr<SlabAllocator> &allocator_) :
    filenames(filename_list),
    nfiles(filename_list.size()),
    // Clamped: spawning four threads to read two files is pure waste. Zero threads for an
    // empty file list is fine -- get_chunk() returns an empty pointer without waiting.
    nthreads(min(nthreads_, long(filename_list.size()))),
    allocator(allocator_)
{
    if (nthreads_ < 1)
        throw runtime_error("AssembledChunkReader: expected nthreads >= 1, got "
                            + to_string(nthreads_));

    for (long i = 0; i < nfiles; i++)
        if (filenames[i].empty())
            throw runtime_error("AssembledChunkReader: filename_list[" + to_string(i)
                                + "] is an empty string");

    // max(1) so the ring is never zero-length; with nfiles == 0 nothing ever indexes it.
    ring.resize(max(nthreads, 1L));
    workers.reserve(nthreads);

    // Spawn the pool. On a mid-spawn throw, wake and join whatever started, then rethrow:
    // a joinable std::thread member reaching its destructor is std::terminate. We set
    // is_stopped by hand rather than calling stop(), which would also stop the caller's
    // allocator -- and a construction failure should leave the caller's object alone.
    try {
        for (long i = 0; i < nthreads; i++)
            workers.push_back(std::thread(&AssembledChunkReader::worker_main, this));
    } catch (...) {
        {
            lock_guard<mutex> lk(lock);
            is_stopped = true;
        }
        cv_ready.notify_all();
        cv_claim.notify_all();

        for (auto &w: workers)
            if (w.joinable())
                w.join();
        throw;
    }
}


AssembledChunkReader::~AssembledChunkReader()
{
    this->stop();

    for (auto &w: workers)
        if (w.joinable())
            w.join();
}


void AssembledChunkReader::stop(std::exception_ptr e) const
{
    {
        lock_guard<mutex> lk(lock);
        if (is_stopped)
            return;              // first caller wins; idempotent
        is_stopped = true;
        error = e;
    }

    // Wake every waiter. Setting is_stopped under the lock before these notifies is what
    // makes a lost wakeup impossible (every waiter rechecks the predicate).
    cv_ready.notify_all();       // the consumer in get_chunk()
    cv_claim.notify_all();       // idle workers

    // Unblock a worker parked in SlabAllocator::get_slab() on an exhausted pool -- our own
    // cvs cannot reach it. MUST be called with 'lock' released (never hold the reader's
    // lock while calling into the allocator). 'e' is forwarded rather than dropped, so that
    // the allocator's other users see the root cause (see notes/stoppable_class.md).
    if (allocator)
        allocator->stop(e);
}


bool AssembledChunkReader::get_is_stopped() const
{
    lock_guard<mutex> lk(lock);
    return is_stopped;           // no _throw_if_stopped: deliberate, this is an accessor
}


std::shared_ptr<AssembledChunk> AssembledChunkReader::get_chunk()
{
    unique_lock<mutex> lk(lock);

    for (;;) {
        // A stop takes precedence over already-read chunks. On an error stop, rethrow the
        // root cause; on a clean stop, return an empty pointer (the "done" value).
        if (error)
            std::rethrow_exception(error);
        if (is_stopped)
            return nullptr;
        if (next_return >= nfiles)
            return nullptr;      // end of the list -- NOT a stop; the reader stays usable

        Slot &s = ring[next_return % nthreads];
        if (s.ready) {
            shared_ptr<AssembledChunk> chunk = std::move(s.chunk);
            std::exception_ptr e = s.error;
            s = Slot();          // free the ring slot
            next_return++;
            lk.unlock();

            // notify_one is sound: every cv_claim waiter has the same predicate, and one
            // freed slot admits exactly one new claim (work-queue handoff).
            cv_claim.notify_one();

            if (e) {
                // This file failed. Delivering the error here rather than when it happened
                // is what keeps the earlier files' chunks (see the header). Stopping first
                // is the entry-point rule, and it also keeps the workers from reading on.
                this->stop(e);
                std::rethrow_exception(e);
            }
            return chunk;
        }

        cv_ready.wait(lk);
    }
}


void AssembledChunkReader::_worker_main()
{
    unique_lock<mutex> lk(lock);

    for (;;) {
        // Wait only while a file is LEFT but the window is shut. Waiting when every file is
        // claimed would park this worker forever: nothing reopens that predicate.
        while (!is_stopped && (next_claim < nfiles) && (next_claim >= next_return + nthreads))
            cv_claim.wait(lk);

        if (is_stopped || (next_claim >= nfiles))
            return;              // stopped, or every file is claimed and this worker is done

        long j = next_claim++;
        lk.unlock();

        // 'filenames' is const and fully built before any thread was spawned, so it needs
        // no lock. from_msgpack() is reentrant (its own fd, pread() only, no mutable
        // globals), so several workers may be inside it at once.
        shared_ptr<AssembledChunk> chunk;
        std::exception_ptr e;

        try {
            chunk = AssembledChunk::from_msgpack(filenames[j], allocator);
        } catch (...) {
            e = std::current_exception();
        }

        lk.lock();
        Slot &s = ring[j % nthreads];
        s.chunk = std::move(chunk);
        s.error = e;
        s.ready = true;
        lk.unlock();

        cv_ready.notify_all();
        lk.lock();
    }
}


void AssembledChunkReader::worker_main()
{
    try {
        _worker_main();
    } catch (...) {
        // A file's read error never gets here -- _worker_main() catches it and stores it in
        // that file's slot. This is for something structural (a bad_alloc, say), where
        // stopping the whole reader is the right answer.
        this->stop(std::current_exception());
    }
}


}}  // namespace pirate::chimefrb
