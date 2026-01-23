#include "../include/pirate/FileWriter.hpp"
#include "../include/pirate/AssembledFrame.hpp"

using namespace std;
using namespace ksgpu;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


void FileWriter::process_frame(const shared_ptr<AssembledFrame> &frame)
{
    unique_lock<std::mutex> frame_lock(frame->mutex);

    long num_filenames = frame->filenames.size();
    xassert(num_filenames > 0);

    bool on_disk = frame->on_ssd || (frame->nfs_count > 0);
    bool enqueue_ssd = !frame->in_ssd_queue && !on_disk;
    bool enqueue_nfs = !frame->in_nfs_queue && (frame->nfs_count < num_filenames) && on_disk;

    if (enqueue_ssd)
        frame->in_ssd_queue = true;
    if (enqueue_nfs)
        frame->in_nfs_queue = true;

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


 // Helper for entry points. Caller must hold mutex.
void FileWriter::_throw_if_stopped(const char *method_name)
{
    if (error)
        std::rethrow_exception(error);
    if (is_stopped)
        throw runtime_error(string(method_name) + " called on stopped instance");
}



}  // namespace pirate
