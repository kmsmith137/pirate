#ifndef _PIRATE_FILE_WRITER_HPP
#define _PIRATE_FILE_WRITER_HPP

#include <mutex>
#include <queue>
#include <string>
#include <memory>
#include <thread>
#include <vector>
#include <filesystem>
#include <condition_variable>

namespace pirate {
#if 0
}  // editor auto-indent
#endif

struct AssembledFrame;  // AssembledFrame.hpp


// Note: the FileWriter spawns worker threads in its constructor.
// (Total threads = Params::num_ssd_threads + Params::num_nfs_threads.)
//
// All worker threads inherit their vCPU affinity from the caller of
// the constructor. Python callers should call the FileWriter constructor
// within a ThreadAffinity context manager.

struct FileWriter
{
public:
    // Constructor arguments.
    struct Params
    {
        std::filesystem::path ssd_root;
        std::filesystem::path nfs_root;
        int num_ssd_threads = 4;
        int num_nfs_threads = 2;
    };

    // Status of write request, reported to RPC subscribers.
    struct WriteStatus
    {
        std::filesystem::path save_path;
        std::exception_ptr error;  // empty pointer if write was successful
    };

    struct RpcSubscriber
    {
        std::mutex mutex;
        std::condition_variable cv;
        std::queue<WriteStatus> queue;
        std::exception_ptr error;
        bool is_stopped = false;
    };

    FileWriter(const Params &params);
    ~FileWriter();

    // Noncopyable, nonmoveable.
    FileWriter(const FileWriter &) = delete;
    FileWriter(FileWriter &&) = delete;
    FileWriter &operator=(const FileWriter &) = delete;
    FileWriter &operator=(FileWriter &&) = delete;

    void stop(std::exception_ptr e = nullptr);

    // process_frame(): adds frame to ssd/nfs queues if needed.
    //  - called by RPC thread, after a new filename is appended.
    //  - called by SSD thread, after writing to local disk.
    //  - this is an "entry point", in the sense defined in notes/thread_backed_class.md.
    //  - caller must not hold this->lock or frame->lock.

    void process_frame(const std::shared_ptr<AssembledFrame> &frame);

    void add_subscriber(const std::shared_ptr<RpcSubscriber> &subscriber);


    // --------------------------------------------------

private:
    const Params params;

    std::mutex mutex;
    std::condition_variable cv;
    bool is_stopped = false;
    std::exception_ptr error;

    std::queue<std::shared_ptr<AssembledFrame>> ssd_queue;
    std::queue<std::shared_ptr<AssembledFrame>> nfs_queue;

    std::vector<std::weak_ptr<RpcSubscriber>> rpc_subscribers;

    std::vector<std::thread> ssd_threads;
    std::vector<std::thread> nfs_threads;

    void ssd_thread_main();
    void nfs_thread_main();

    void _ssd_thread_main();
    void _nfs_thread_main();
    void _process_frame(const std::shared_ptr<AssembledFrame> &frame);

    // All paths are relative to either params.ssd_root, or params.nfs_root.
    // These functions assume that paths have been validated with pirate::is_safe_relpath().
    // (This check happens before the paths get added to the AssembledFrame, in the write_request RPC.)
    void _hardlink_in_nfs(const std::filesystem::path &src_path, const std::filesystem::path &dst_path);
    void _write_to_ssd(const std::shared_ptr<AssembledFrame> &frame, const std::filesystem::path &path);
    void _copy_from_ssd_to_nfs(const std::filesystem::path &path);
    void _try_to_delete_from_ssd(const std::filesystem::path &path);

    // More helper functions called internally.
    void _ssd_worker_checks(const std::shared_ptr<AssembledFrame> &frame);
    void _update_rpc_subscribers(const WriteStatus &write_status);

    std::vector<std::shared_ptr<RpcSubscriber>> _get_rpc_subscribers();

    // Helper for entry points. Caller must hold mutex.
    void _throw_if_stopped(const char *method_name);
};


// FilenamePattern: helper class to expand 'pattern' strings, such as
// "dir1/dir2/file_(BEAM)_(CHUNK).asdf"

struct FilenamePattern
{
    // Constructor validates that 'pattern' contains exactly one "(BEAM)" and one "(CHUNK)".
    // Throws an exception if validation fails.
    FilenamePattern(const std::string &pattern);

    // Replace "(BEAM)" with f->beam_id and "(CHUNK)" with f->time_chunk_index.
    std::string expand(const std::shared_ptr<AssembledFrame> &f) const;

private:
    std::string pattern;
    std::size_t beam_pos;   // position of "(BEAM)" in pattern
    std::size_t chunk_pos;  // position of "(CHUNK)" in pattern
};


}  // namespace pirate

#endif  // _PIRATE_FILE_WRITER_HPP
