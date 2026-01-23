#ifndef _PIRATE_FILE_WRITER_HPP
#define _PIRATE_FILE_WRITER_HPP

#include <mutex>
#include <queue>
#include <string>
#include <memory>
#include <condition_variable>

namespace pirate {
#if 0
}  // editor auto-indent
#endif

struct AssembledFrame;  // AssembledFrame.hpp


struct FileWriter
{
public:
   struct Params
    {
        std::string ssd_root;
        std::string nfs_root;
        int num_ssd_threads = 4;
        int num_nfs_threads = 2;
    };

    struct Tracker
    {
        std::mutex mutex;
        std::condition_variable cv;
        std::queue<std::pair<std::string,int>> queue;  // (filename, status) pairs
    };

    FileWriter(const Params &params);

    // process_frame(): adds frame to ssd/nfs queues if needed.
    //  - called by RPC thread, after a new filename is appended.
    //  - called by SSD thread, after writing to local disk.
    //  - this is an "entry point", in the sense defined in notes/thread_backed_class.md.
    //  - caller must not hold this->lock or frame->lock.

    void process_frame(const std::shared_ptr<AssembledFrame> &frame);

    void add_tracker(const std::shared_ptr<Tracker> &tracker);


    // --------------------------------------------------

private:
    const Params params;

    std::mutex mutex;
    std::condition_variable cv;
    bool is_started = false;
    bool is_stopped = false;
    std::exception_ptr error;

    std::queue<std::shared_ptr<AssembledFrame>> ssd_queue;
    std::queue<std::shared_ptr<AssembledFrame>> nfs_queue;
    std::vector<std::weak_ptr<Tracker>> trackers;

    std::vector<std::shared_ptr<Tracker>> _get_trackers();

    // Helper for entry points. Caller must hold mutex.
    void _throw_if_stopped(const char *method_name);
};


}  // namespace pirate

#endif  // _PIRATE_FILE_WRITER_HPP