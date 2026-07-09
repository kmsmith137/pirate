#ifndef _PIRATE_FILE_WRITER_HPP
#define _PIRATE_FILE_WRITER_HPP

#include <atomic>
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
// Thread-backed class (see notes/thread_backed_class.md).
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
        std::string stream_name;      // "" = WriteFiles-triggered; nonempty = stream (StartStream RPC)
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

    // Public so the FrbServer RPC handler (e.g. GetConfig) can read the
    // construction-time config. Field is const, so external reads are safe.
    const Params params;

private:

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


// Acquisition-directory helpers. RPC callers (WriteFiles, StartStream) specify
// only an acquisition directory 'acqdir'; filenames are always of the form
//
//     {acqdir}/frame_b{beam_id}_t{time_chunk_index}.asdf
//
// relative to both file-writing roots (written to {ssd_root}/... and copied to
// {nfs_root}/...; the user-visible final file is the nfs_root one).

// Validates an RPC-supplied acqdir: nonempty, a safe relative path (no absolute
// paths, no ".." escapes -- see is_safe_relpath), and in canonical form (no
// leading/trailing '/', no '//', no '.' or '..' components). May be multi-level
// ("foo/bar/baz"). Throws runtime_error on failure. Canonical form matters
// beyond cosmetics: relpaths derived from acqdir are compared as strings by
// FileWriter's NFS duplicate-skip, so two spellings of the same directory
// (e.g. "foo" vs "foo/") must be impossible.
void validate_acqdir(const std::string &acqdir);

// Builds the roots-relative path for one frame of an acquisition:
// "{acqdir}/frame_b{beam_id}_t{time_chunk_index}.asdf" (unpadded decimal).
// This fixed naming scheme is parsed by python-side tooling
// (pirate_frb.utils.list_acqdir's _FRAME_RE) -- keep the two in sync.
std::string make_acq_relpath(const std::string &acqdir,
                             const std::shared_ptr<AssembledFrame> &frame);


// FileStream: one registered "stream" (StartStream RPC): frames matching
// (beam_ids x chunk range) are queued for disk writing automatically as they
// are processed. Created by FrbServer's StartStream handler; referenced by
// FrbServer's active/inactive stream lists, and by AssembledFrame::SaveRequest
// entries -- the latter is what gives the FileWriter worker threads access to
// the stream throughout the file-writing code (to bump the written/errored
// counters and tag notifications with stream_name).
//
// THREAD-SAFETY: the fields fall into three groups.
//
//   (i) Immutable after publication. Set by the StartStream handler before
//       the stream is pushed into FrbServer::active_streams (a push made
//       under FrbServer::mutex, which is also how every other thread
//       discovers the stream) -- so any thread that can see the stream can
//       read these freely.
//
//   (ii) Atomic counters, readable anywhere. Two ordering rules keep them
//       sound (and must be preserved if new call sites are added):
//         R1: num_files_queued is incremented by the frame_finalizing_thread
//             at MATCH time, BEFORE the corresponding save_paths push.
//         R2: num_files_written / num_files_errored are incremented by the
//             FileWriter NFS threads at completion, BEFORE the corresponding
//             subscriber notification is emitted.
//       Consequences: written + errored <= queued at every instant, and once
//       the stream is deactivated and the FileWriter drains,
//       written + errored == queued ("fully drained") -- which is how
//       ShowStreams distinguishes DRAINING from INACTIVE.
//
//   (iii) Deactivation fields, guarded by the owning FrbServer's mutex:
//       written exactly once, in the same critical section that moves the
//       stream from FrbServer::active_streams to the inactive ring, and read
//       only under that mutex (ShowStreams / CancelStream). FileWriter code
//       must never touch these.

struct FileStream
{
    // (i) Immutable after publication.
    std::string stream_name;          // nonempty; unique among ACTIVE streams
    std::string acqdir;            // acquisition directory (validated with validate_acqdir)
    std::vector<long> beam_ids;    // original args (nonempty, validated, distinct)
    std::vector<int> beam_indices; // parallel: position in metadata->beam_ids
    long fpga_seq_start = 0;       // original args (echoed by ShowStreams)
    long fpga_seq_end = 0;
    long chunk_first = 0;          // derived: fpga_seq_start / seq_per_chunk
    long chunk_last = 0;           // derived: (fpga_seq_end-1) / seq_per_chunk, INCLUSIVE
    long started_at_unix_ns = 0;   // wall clock, stamped by StartStream

    // (ii) Atomic counters (rules R1/R2 above).
    std::atomic<long> num_files_queued = 0;
    std::atomic<long> num_files_written = 0;   // includes duplicate-skips (file IS on disk)
    std::atomic<long> num_files_errored = 0;

    // (iii) Deactivation fields -- guarded by the owning FrbServer's mutex.
    bool cancelled = false;             // true = CancelStream, false = expired
    long deactivated_at_unix_ns = 0;    // wall clock; 0 while active
};


}  // namespace pirate

#endif  // _PIRATE_FILE_WRITER_HPP
