#include "../include/pirate/FileWriter.hpp"
#include "../include/pirate/AssembledFrame.hpp"
#include "../include/pirate/file_utils.hpp"

#include <sstream>

#include <ksgpu/xassert.hpp>

using namespace std;
using namespace ksgpu;

namespace fs = std::filesystem;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


FileWriter::FileWriter(const Params &params_) : params(params_)
{
    xassert(params.ssd_root.is_absolute());
    xassert(params.nfs_root.is_absolute());
    xassert(params.num_ssd_threads > 0);
    xassert(params.num_nfs_threads > 0);
    xassert(params.max_subscriber_backlog >= 1);

    // Create directories, if they don't already exist.
    pirate::create_directories(params.ssd_root);
    pirate::create_directories(params.nfs_root);

    // All members are initialized before this point.
    // Now safe to start the worker threads.
    //
    // If thread creation fails partway (e.g. std::system_error on thread
    // resource exhaustion), stop and join the already-spawned threads before
    // rethrowing. Without this, the constructor would exit by exception with
    // joinable std::thread members, and their destructors would call
    // std::terminate.
    try {
        for (long i = 0; i < params.num_ssd_threads; i++)
            ssd_threads.emplace_back(&FileWriter::ssd_thread_main, this);
        for (long i = 0; i < params.num_nfs_threads; i++)
            nfs_threads.emplace_back(&FileWriter::nfs_thread_main, this);
    } catch (...) {
        this->stop(std::current_exception());

        for (auto &t : ssd_threads)
            if (t.joinable())
                t.join();

        for (auto &t : nfs_threads)
            if (t.joinable())
                t.join();

        throw;
    }
}


FileWriter::~FileWriter()
{
    this->stop();

    for (auto &t : ssd_threads)
        if (t.joinable())
            t.join();
    
    for (auto &t : nfs_threads)
        if (t.joinable())
            t.join();
}


void FileWriter::stop(std::exception_ptr e) const
{
    unique_lock<std::mutex> lock(mutex);
    if (is_stopped)
        return;

    is_stopped = true;
    error = e;

    auto subscribers = _get_rpc_subscribers();  // call with lock held
    lock.unlock();

    ssd_cv.notify_all();
    nfs_cv.notify_all();
    ssd_clear_cv.notify_all();

    for (const auto &s: subscribers) {
        unique_lock<std::mutex> subscriber_lock(s->mutex);        
        if (!s->is_stopped) {
            s->is_stopped = true;
            s->error = e;
            subscriber_lock.unlock();
            s->cv.notify_all();
        }
    }
}


void FileWriter::ssd_thread_main()
{
    try {
        _ssd_thread_main();
    } catch (...) {
        stop(std::current_exception());
    }
}


void FileWriter::nfs_thread_main()
{
    try {
        _nfs_thread_main();
    } catch (...) {
        stop(std::current_exception());
    }
}


void FileWriter::process_frame(const shared_ptr<AssembledFrame> &frame)
{
    // Per the strict stoppable-class policy (notes/stoppable_class.md), ANY
    // exception thrown from an entry point (including the argument check)
    // stops the FileWriter.
    try {
        xassert(frame);
        _process_frame(frame);
    } catch (...) {
        stop(std::current_exception());
        throw;
    }
}


void FileWriter::_process_frame(const shared_ptr<AssembledFrame> &frame)
{
    // Stopped-check FIRST -- before the no-enqueue early return, and before
    // mutating the frame's in_*_queue flags. On an already-stopped writer we
    // throw with the frame untouched (no flags set that no worker will ever
    // clear). If stop() races in between this check and the push below, the
    // second check throws with the flags set -- benign, since the whole
    // pipeline is coming down at that point.
    {
        lock_guard<std::mutex> state_lock(this->mutex);
        _throw_if_stopped("FileWriter::process_frame");
    }

    unique_lock<std::mutex> frame_lock(frame->mutex);

    long npaths = frame->save_paths.size();
    xassert(npaths > 0);

    // An AssembledFrame is ready for the nfs queue if it is either on disk, or a save_error has ben set.
    bool nfs_queue_ready = frame->on_ssd || (frame->nfs_count > 0) || (frame->save_error != nullptr);

    bool enqueue_ssd = !frame->in_ssd_queue && !nfs_queue_ready;
    bool enqueue_nfs = !frame->in_nfs_queue && (frame->nfs_count < npaths) && nfs_queue_ready;

    if (enqueue_ssd)
        frame->in_ssd_queue = true;
    if (enqueue_nfs)
        frame->in_nfs_queue = true;

    xassert(!frame->in_ssd_queue || !frame->in_nfs_queue);
    frame_lock.unlock();

    if (!enqueue_ssd && !enqueue_nfs)
        return;

    unique_lock<std::mutex> state_lock(this->mutex);
    _throw_if_stopped("FileWriter::process_frame");

    if (enqueue_ssd)
        this->ssd_queue.push(frame);
    if (enqueue_nfs)
        this->nfs_queue.push(frame);

    state_lock.unlock();

    // Wake one ssd/nfs thread per pushed frame. notify_one is sound: all
    // waiters on each cv share the same predicate (queue non-empty), and one
    // pushed frame is consumed by exactly one thread.
    if (enqueue_ssd)
        ssd_cv.notify_one();
    if (enqueue_nfs)
        nfs_cv.notify_one();
}


void FileWriter::add_subscriber(const shared_ptr<RpcSubscriber> &subscriber)
{
    try {
        xassert(subscriber);

        lock_guard<std::mutex> lock(mutex);
        _throw_if_stopped("FileWriter::add_subscriber");

        rpc_subscribers.push_back(subscriber);
    } catch (...) {
        stop(std::current_exception());
        throw;
    }
}


void FileWriter::_ssd_thread_main()
{
    unique_lock<std::mutex> state_lock(mutex);

    for (;;) {
        // At top of loop, lock is held.
        // Pop a frame from the ssd_queue.
        for (;;) {
            if (is_stopped)
                return;
            if (!ssd_queue.empty())
                break;
            ssd_cv.wait(state_lock);
        }

        shared_ptr<AssembledFrame> frame = ssd_queue.front();
        ssd_queue.pop();
        bool ssd_now_empty = ssd_queue.empty();   // snapshot under the lock
        state_lock.unlock();

        // If this pop emptied the ssd_queue, wake the nfs threads that are
        // waiting for it to clear (see _nfs_thread_main()). notify_all: every
        // waiter's predicate becomes true simultaneously.
        if (ssd_now_empty)
            ssd_clear_cv.notify_all();

        // Some paranoid checks.
        unique_lock<std::mutex> frame_lock(frame->mutex);
        _ssd_worker_checks(frame);

        // To avoid racing against another thread that modifes frame->save_paths,
        // we copy save_paths[0] to a local variable, then drop the lock, then
        // write the file.

        string path = frame->save_paths[0].path;
        frame_lock.unlock();

        std::exception_ptr eptr = nullptr;

        try {
            _write_to_ssd(frame, path);
        } catch (...) {
            eptr = std::current_exception();
        }

        frame_lock.lock();
        _ssd_worker_checks(frame);

        frame->save_error = eptr;
        frame->in_ssd_queue = false;
        frame->in_nfs_queue = true;
        frame->on_ssd = !eptr;
        
        // The frame can be reaped now: either the data is safely on the SSD,
        // or the write FAILED (save_error set) and the data will never reach
        // disk -- keeping it would leak the slab (see _reap_locked()).
        frame->_reap_locked();
        frame_lock.unlock();

        state_lock.lock();
        nfs_queue.push(frame);
        // Wake one nfs thread for the pushed frame (work-queue handoff --
        // see the ssd_cv/nfs_cv comment in FileWriter.hpp).
        nfs_cv.notify_one();
        // Lock remains held for next loop iteration.
    }
}


// Frames in the SSD queue should satisfy these conditions.
// Called by the SSD thread, with frame->mutex held.
void FileWriter::_ssd_worker_checks(const shared_ptr<AssembledFrame> &frame)
{
    xassert(frame->save_paths.size() > 0);
    xassert(frame->save_error == nullptr);
    xassert(frame->in_ssd_queue);
    xassert(!frame->in_nfs_queue);
    xassert(!frame->on_ssd);
    xassert(frame->nfs_count == 0);
}


void FileWriter::_nfs_thread_main()
{
    // Outer loop over frames
    for (;;) {
        unique_lock<std::mutex> state_lock(mutex);

        // Pop a frame from the nfs_queue.
        for (;;) {
            if (this->is_stopped)
                return;
            if (!nfs_queue.empty())
                break;
            nfs_cv.wait(state_lock);
        }

        shared_ptr<AssembledFrame> frame = nfs_queue.front();
        nfs_queue.pop();  

        xassert(frame);

        // Before the nfs thread processes the frame, we wait for the ssd queue
        // to clear. This ensures that under memory pressure, the ssd threads have
        // near-100% access to SSD bandwidth.
        //
        // Deliberate approximation: the predicate tests only the QUEUE, so up to
        // num_ssd_threads popped-and-mid-write frames are invisible to it, and a
        // small bounded amount of nfs work can overlap those writes. Tracking
        // in-flight writes too (a counter) isn't worth the complexity: the gate
        // matters in the heavily-backlogged case, where the queue is rarely empty.

        for (;;) {
            if (this->is_stopped)
                return;
            if (ssd_queue.empty())
                break;
            ssd_clear_cv.wait(state_lock);
        }
        
        state_lock.unlock();

        unique_lock<std::mutex> frame_lock(frame->mutex);
        xassert(frame->save_paths.size() > 0);

        fs::path primary_path = frame->save_paths[0].path;

        // Inner loop over frames->save_paths.
        for (;;) {
            // At top of loop, frame_lock is held, but not state_lock.

            // Recheck is_stopped once per save_path entry, so that a stopped
            // FileWriter abandons the remaining (possibly slow) NFS operations
            // for this frame promptly, instead of processing every remaining
            // entry first. (Drop frame_lock before taking the state mutex:
            // this class never holds both locks simultaneously.)
            frame_lock.unlock();
            {
                unique_lock<std::mutex> stop_check_lock(mutex);
                if (this->is_stopped)
                    return;
            }
            frame_lock.lock();

            //
            // General comment: if the frame_lock is dropped and reacquired, then frame->save_paths
            // may be modified (by an RPC caller appending new save_paths). We generally handle this
            // by using a continue-statement to go back to the top of the inner loop, to start the
            // processing logic from scratch.
            //
            // Another consequence of this: we always copy save_paths to local variables before
            // dropping the frame_lock, in order to avoid racing against another thread that modifies
            // frame->save_paths. (This happens in several places in the code below.)

            // Recheck these asserts every time the lock is dropped and reacquired.
            xassert(!frame->in_ssd_queue);
            xassert(frame->in_nfs_queue);

            // Delete the SSD copy once save_paths[0] has been PROCESSED --
            // either copied to NFS (nfs_count bumped by the success path) or
            // given up on (nfs_count bumped by the save_error branch below).
            // The error case is deliberate: the SSD is a staging area, not a
            // backlog, so when the NFS write fails the data is dropped (the
            // failure was already reported to subscribers as an errored
            // WriteStatus) rather than left to accumulate on the SSD.
            // Retaining errored files would only make sense together with
            // NFS-retry logic and reboot recovery, which we don't implement.
            if (frame->on_ssd && (frame->nfs_count > 0)) {
                frame_lock.unlock();

                // Note: prints a warning (rather than throwing an exception) if something goes wrong.
                _try_to_delete_from_ssd(primary_path);

                frame_lock.lock();
                frame->on_ssd = false;
            }

            long nfs_count = frame->nfs_count;

            if (nfs_count >= long(frame->save_paths.size())) {
                frame->in_nfs_queue = false;
                break;  // note: only exit from inner loop
            }

            if (frame->save_error) {
                shared_ptr<FileStream> stream = frame->save_paths[nfs_count].stream;
                WriteStatus ws;
                ws.save_path = frame->save_paths[nfs_count].path;
                ws.stream_name = stream ? stream->stream_name : "";
                ws.error = frame->save_error;
                frame_lock.unlock();

                // Rule R2 (see FileStream's thread-safety comment): bump the
                // stream counter BEFORE emitting the notification, so "client
                // received the notification" implies "counter reflects it".
                if (stream)
                    stream->num_files_errored++;
                _update_rpc_subscribers(ws);

                // Re-acquire frame_lock, before going back to top of loop.
                frame_lock.lock();
                frame->nfs_count++;
                continue;
            }

            // Duplicate-path skip. If save_paths[nfs_count] repeats the path of
            // an earlier entry, the file is already on disk under this exact
            // name: entries are processed strictly in order, and reaching this
            // point with save_error == nullptr means every j < nfs_count
            // completed successfully (an NFS failure sets save_error without
            // incrementing nfs_count, diverting later entries to the branch
            // above). Skip the filesystem operation -- a hardlink onto an
            // existing dst would fail and poison save_error -- but still emit
            // a success WriteStatus and still advance nfs_count: RPC clients
            // rely on one notification per queued entry (a silent skip would
            // hang a client waiting on this filename).
            if (nfs_count > 0) {
                bool duplicate = false;
                for (long j = 0; j < nfs_count; j++) {
                    if (frame->save_paths[j].path == frame->save_paths[nfs_count].path) {
                        duplicate = true;
                        break;
                    }
                }
                if (duplicate) {
                    shared_ptr<FileStream> stream = frame->save_paths[nfs_count].stream;
                    WriteStatus ws;
                    ws.save_path = frame->save_paths[nfs_count].path;
                    ws.stream_name = stream ? stream->stream_name : "";
                    frame_lock.unlock();

                    // A duplicate-skip counts as WRITTEN (the file is on disk
                    // under this exact name). Rule R2: bump before notifying.
                    if (stream)
                        stream->num_files_written++;
                    _update_rpc_subscribers(ws);

                    // Re-acquire frame_lock, before going back to top of loop.
                    frame_lock.lock();
                    frame->nfs_count++;
                    continue;
                }
            }

            fs::path secondary_path = frame->save_paths[nfs_count].path;
            shared_ptr<FileStream> secondary_stream = frame->save_paths[nfs_count].stream;
            frame_lock.unlock();

            try {
                if (nfs_count == 0)
                    _copy_from_ssd_to_nfs(primary_path);
                else
                    _hardlink_in_nfs(primary_path, secondary_path);
            } catch (...) {
                // No counter bump here: nfs_count is not incremented, so this
                // entry is re-processed by the save_error branch above, which
                // counts it as errored (exactly one count per entry).
                frame_lock.lock();
                frame->save_error = std::current_exception();
                continue;  // back to top of loop, with lock held, and without incrementing frame->nfs_count.
            }

            // Rule R2: bump the stream counter BEFORE emitting the notification.
            if (secondary_stream)
                secondary_stream->num_files_written++;

            WriteStatus ws;
            ws.save_path = secondary_path;
            ws.stream_name = secondary_stream ? secondary_stream->stream_name : "";
            _update_rpc_subscribers(ws);

            frame_lock.lock();

            // Check that frame->nfs_count didn't change while we dropped the lock.
            xassert(frame->nfs_count == nfs_count);
            frame->nfs_count++;
        } // inner loop over frame->save_paths
    }  // outer loop over frames
}


// Note on stop-responsiveness: the file I/O in the helpers below (write+fsync
// on SSD; copy/hardlink on NFS) is blocking, and cannot be interrupted by
// stop() -- there is no practical way to abort a thread that is mid-syscall
// in a write to a wedged filesystem (e.g. a hard-mounted NFS server that has
// gone away). Consequence: stop() itself returns promptly, but ~FileWriter()
// joins the worker threads and can hang until the in-flight filesystem
// operation completes. The is_stopped rechecks in the worker loops bound the
// damage to one in-flight operation per thread.

void FileWriter::_write_to_ssd(const std::shared_ptr<AssembledFrame> &frame, const fs::path &path)
{
    fs::path full_path = params.ssd_root / path;
    pirate::create_directories(full_path.parent_path());  // wraps fs::create_directories()

    // Write through temp file
    TmpFileGuard guard(full_path);
    frame->write_asdf(guard.tmp_filename);
    guard.commit();
}


void FileWriter::_copy_from_ssd_to_nfs(const fs::path &path)
{
    fs::path src_path = params.ssd_root / path;
    fs::path dst_path = params.nfs_root / path;
    pirate::create_directories(dst_path.parent_path());  // wraps fs::create_directories()

    // Write through temp file
    TmpFileGuard guard(dst_path);
    pirate::copy_file(src_path, guard.tmp_filename);  // wraps fs::copy_file()
    guard.commit();
}


void FileWriter::_hardlink_in_nfs(const fs::path &src_path, const fs::path &dst_path)
{
    fs::path src_full = params.nfs_root / src_path;
    fs::path dst_full = params.nfs_root / dst_path;

    pirate::create_directories(dst_full.parent_path());  // wraps fs::create_directories()
    pirate::create_hard_link(src_full, dst_full);        // wraps fs::create_hard_link()
}


// Prints a warning (rather than throwing an exception) if anything goes wrong.
void FileWriter::_try_to_delete_from_ssd(const fs::path &path)
{
    fs::path full_path = params.ssd_root / path;

    try {
        // remove_file() returns true if file was deleted, false if it didn't exist.
        if (!pirate::remove_file(full_path))
            cout << "Warning: file not found when deleting " << full_path << " from SSD" << endl;
    }
    catch (std::exception &e) {
        cout << "Warning: deletion from SSD failed: " << e.what() << endl;
    }
    catch (...) {
        cout << "Warning: unknown error while deleting " << full_path << " from SSD" << endl;
    }
}


// Call with lock held!!
// Returns this->rpc_subscribers, calling lock() to convert weak_ptr -> shared_ptr.
// Any subscribers which do not lock() successfully are "pruned" from this->rpc_subscribers.
vector<shared_ptr<FileWriter::RpcSubscriber>> FileWriter::_get_rpc_subscribers() const
{
    vector<shared_ptr<FileWriter::RpcSubscriber>> ret;

    long i = 0;
    while (i < long(rpc_subscribers.size())) {
        shared_ptr<RpcSubscriber> s = rpc_subscribers[i].lock();

        if (s) {
            ret.push_back(s);
            i++;
        }
        else {
            rpc_subscribers[i] = std::move(rpc_subscribers.back());
            rpc_subscribers.pop_back();
            // don't increment i
        }
    }

    return ret;
}


void FileWriter::_update_rpc_subscribers(const WriteStatus &write_status)
{
    unique_lock<std::mutex> lock(mutex);
    auto subscribers = _get_rpc_subscribers();  // call with lock held
    lock.unlock();

    for (const auto &s: subscribers) {
        unique_lock<std::mutex> lock(s->mutex);

        if (s->is_stopped)
            continue;

        // Push-time filter: stream-triggered notifications only go to
        // subscribers that opted in. (This filter used to live at pop time,
        // in the SubscribeFiles handler; filtering here keeps unwanted
        // entries out of the queue entirely, so a subscriber can never
        // overflow on notifications it never asked for.)
        if (!s->subscribe_streams && !write_status.stream_name.empty())
            continue;

        if ((long)s->queue.size() >= params.max_subscriber_backlog) {
            // Overflow: stop this subscriber and free its queue NOW, so
            // server memory stays bounded even if the client never reads
            // again. The handler observes error/is_stopped the next time it
            // makes progress -- its blocking Write() may stay parked until
            // the client reads or disconnects, but that costs no more than
            // any idle subscriber's handler thread does. The subscriber
            // stays in rpc_subscribers (skipped via is_stopped above) until
            // the handler exits and the weak_ptr is pruned.
            std::stringstream ss;
            ss << "SubscribeFiles: subscriber fell behind (queue reached "
               << params.max_subscriber_backlog << " entries); notifications "
               << "were dropped -- resubscribe and resynchronize";
            s->is_stopped = true;
            s->error = std::make_exception_ptr(runtime_error(ss.str()));
            std::queue<WriteStatus>().swap(s->queue);
            lock.unlock();
            // Wake the handler if it's in its cv wait (it is almost always
            // parked in Write() instead -- the queue can only build while
            // the handler isn't popping -- but this covers the rare
            // interleavings).
            s->cv.notify_one();
            cout << "FileWriter: stopping slow SubscribeFiles subscriber "
                 << "(queue reached " << params.max_subscriber_backlog
                 << " entries)" << endl;
            continue;
        }

        s->queue.push(write_status);
        lock.unlock();
        // notify_one is sound here: the only waiter on a subscriber's cv
        // is the single SubscribeFiles handler thread that created it.
        s->cv.notify_one();
    }
}

 // Helper for entry points. Caller must hold mutex.
void FileWriter::_throw_if_stopped(const char *method_name)
{
    if (error)
        std::rethrow_exception(error);
    if (is_stopped)
        throw runtime_error(string(method_name) + " called on stopped instance");
}


// -------------------------------------------------------------------------------------------------


// See FileWriter.hpp for the contract (nonempty, safe, canonical relative path).
void validate_acqdir(const string &acqdir)
{
    if (acqdir.empty())
        throw runtime_error("acqdir must be a nonempty string");

    fs::path p(acqdir);

    // The canonical-form check (lexically_normal round-trip) compares STRINGS,
    // not fs::path objects -- path equality decomposes elementwise and would
    // accept e.g. "foo//bar". The has_filename() check rejects a trailing '/'
    // (which lexically_normal preserves), and the "." check rejects the one
    // normalized path that would alias the roots themselves.
    if (!is_safe_relpath(p)
        || (p.lexically_normal().string() != acqdir)
        || !p.has_filename()
        || (acqdir == "."))
    {
        throw runtime_error("invalid acqdir '" + acqdir + "': must be a normalized relative path"
                            " (no leading/trailing '/', no '//', no '.' or '..' components)");
    }
}


// See FileWriter.hpp for the contract; keep in sync with the python-side
// parser (pirate_frb.utils.list_acqdir's _FRAME_RE).
string make_acq_relpath(const string &acqdir, long beam_id, long time_chunk_index)
{
    return acqdir + "/frame_b" + to_string(beam_id)
                  + "_t" + to_string(time_chunk_index) + ".asdf";
}


string make_acq_relpath(const string &acqdir, const shared_ptr<AssembledFrame> &frame)
{
    xassert(frame);
    return make_acq_relpath(acqdir, frame->beam_id, frame->time_chunk_index);
}


}  // namespace pirate
