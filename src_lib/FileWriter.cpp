#include "../include/pirate/FileWriter.hpp"
#include "../include/pirate/AssembledFrame.hpp"
#include "../include/pirate/file_utils.hpp"

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

    // All members are initialized before this point.
    // Now safe to start the worker threads.
    for (long i = 0; i < params.num_ssd_threads; i++)
        ssd_threads.emplace_back(&FileWriter::ssd_thread_main, this);
    for (long i = 0; i < params.num_nfs_threads; i++)
        nfs_threads.emplace_back(&FileWriter::nfs_thread_main, this);
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


void FileWriter::stop(std::exception_ptr e)
{
    unique_lock<std::mutex> lock(mutex);
    if (is_stopped)
        return;

    is_stopped = true;
    error = e;

    auto subscribers = _get_rpc_subscribers();  // call with lock held
    lock.unlock();

    cv.notify_all();

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
    xassert(frame);

    try {
        _process_frame(frame);
    } catch (...) {
        this->stop(std::current_exception());
        throw;
    }
}


void FileWriter::_process_frame(const shared_ptr<AssembledFrame> &frame)
{
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

    lock_guard<std::mutex> state_lock(this->mutex);
    _throw_if_stopped("process_frame");

    if (enqueue_ssd)
        this->ssd_queue.push(frame);
    if (enqueue_nfs)
        this->nfs_queue.push(frame);

    // Wake up ssd and nfs threads.
    cv.notify_all();
}


void FileWriter::add_subscriber(const shared_ptr<RpcSubscriber> &subscriber)
{
    xassert(subscriber);

    lock_guard<std::mutex> lock(mutex);
    _throw_if_stopped("add_subscriber");

    rpc_subscribers.push_back(subscriber);
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
            cv.wait(state_lock);
        }

        shared_ptr<AssembledFrame> frame = ssd_queue.front();
        ssd_queue.pop();
        state_lock.unlock();

        // Wake up nfs threads that are waiting for the ssd queue to clear.
        // (See _nfs_thread_main().)
        cv.notify_all();

        // Some paranoid checks.
        unique_lock<std::mutex> frame_lock(frame->mutex);
        _ssd_worker_checks(frame);

        // To avoid racing against another thread that modifes frame->save_paths,
        // we copy save_paths[0] to a local variable, then drop the lock, then
        // write the file.

        string path = frame->save_paths[0];
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
        frame_lock.unlock();

        state_lock.lock();
        nfs_queue.push(frame);
        // Wake up nfs threads that are waiting for the nfs queue.
        cv.notify_all();
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
            cv.wait(state_lock);
        }

        shared_ptr<AssembledFrame> frame = nfs_queue.front();
        nfs_queue.pop();  

        xassert(frame);

        // Before the nfs thread processes the frame, we wait for the ssd queue
        // to clear. This ensures that under memory pressure, the ssd threads have
        // near-100% access to SSD bandwidth.

        for (;;) {
            if (this->is_stopped)
                return;
            if (ssd_queue.empty())
                break;
            cv.wait(state_lock);
        }
        
        state_lock.unlock();

        unique_lock<std::mutex> frame_lock(frame->mutex);
        xassert(frame->save_paths.size() > 0);

        fs::path primary_path = frame->save_paths[0];

        // Inner loop over frames->save_paths.
        for (;;) {
            // At top of loop, frame_lock is held, but not state_lock.
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

            // After a frame has been copied to NFS, it can be deleted from SSD.
            // The frame has been copied to NFS iff (nfs_count > 0).
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
                WriteStatus ws;
                ws.save_path = frame->save_paths[nfs_count];
                ws.error = frame->save_error;
                frame_lock.unlock();

                _update_rpc_subscribers(ws);

                // Re-acquire frame_lock, before going back to top of loop.
                frame_lock.lock();
                frame->nfs_count++;
                continue;
            }

            fs::path secondary_path = frame->save_paths[nfs_count];
            frame_lock.unlock();
    
            try {
                if (nfs_count == 0)
                    _copy_from_ssd_to_nfs(primary_path);
                else
                    _hardlink_in_nfs(primary_path, secondary_path);
            } catch (...) {
                frame_lock.lock();
                frame->save_error = std::current_exception();
                continue;  // back to top of loop, with lock held, and without incrementing frame->nfs_count.
            }

            WriteStatus ws;
            ws.save_path = secondary_path;
            _update_rpc_subscribers(ws);

            frame_lock.lock();

            // Check that frame->nfs_count didn't change while we dropped the lock.
            xassert(frame->nfs_count == nfs_count);
            frame->nfs_count++;
        } // inner loop over frame->save_paths
    }  // outer loop over frames
}


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
vector<shared_ptr<FileWriter::RpcSubscriber>> FileWriter::_get_rpc_subscribers()
{
    vector<shared_ptr<FileWriter::RpcSubscriber>> ret;

    ulong i = 0;
    while (i < rpc_subscribers.size()) {
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
        if (!s->is_stopped) {
            s->queue.push(write_status);
            lock.unlock();
            s->cv.notify_all();
        }
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


}  // namespace pirate
