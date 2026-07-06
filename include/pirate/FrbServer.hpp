#ifndef _PIRATE_FRB_SERVER_HPP
#define _PIRATE_FRB_SERVER_HPP

#include "Receiver.hpp"
#include "AssembledFrame.hpp"
#include "DedispersionConfig.hpp"
#include "FileWriter.hpp"      // FileStream, FilenamePattern
#include "XEngineMetadata.hpp"
#include "constants.hpp"       // inactive_file_stream_capacity

#include <ksgpu/xassert.hpp>

#include <array>
#include <condition_variable>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

// Forward declarations
namespace grpc { class Server; }

// Forward-declare the generated proto type used by _fill_handshake(), so this
// widely-included header does not pull in grpc++/protobuf. (The .cpp includes
// the generated headers.)
namespace frb::grouper::v1 { class Handshake; }

namespace pirate {
#if 0
}  // editor auto-indent
#endif


struct BumpAllocator;     // BumpAllocator.hpp (only shared_ptr<> members here)
struct DedispersionPlan;  // DedispersionPlan.hpp
struct GpuDedisperser;    // Dedisperser.hpp
struct CudaStreamPool;    // CudaStreamPool.hpp
struct CudaEventRingbuf;  // CudaEventRingbuf.hpp
struct FrbRpcService;     // defined in FrbServer.cpp


// FrbServer: a thread-backed class that manages Receivers and an RPC service.
//
// Backing threads: one worker thread per Receiver, a reaper thread, a
// processing thread, and a frame-finalizing thread (see below). These
// threads inherit their vcpu affinity from the caller
// of FrbServer::start(). Python callers should call FrbServer.start() within
// a ThreadAffinity context manager.
//
// FrbServer construction takes a DedispersionConfig (params.config_prefilled)
// whose four metadata-dependent members (zone_nfreq, zone_freq_edges,
// time_sample_ms, beams_per_gpu) will be overwritten by the processing
// thread once XEngineMetadata is available. The "filled" config is
// reachable as FrbServer::plan->config after the plan has been built.
//
// The processing thread also builds a GpuDedisperser (alongside a private
// CudaStreamPool) once the plan exists. 'plan' and 'dedisperser' (along with
// evrb_dq / evrb_h2g and dedisperser_is_initialized) become
// non-null atomically (published in the same critical section).
//
// processing_thread: after publishing, snapshots rb_curr = rb_processed once
// rb_initialized is true, then loops over (ichunk, ibatch, beam) doing a
// per-beam H2G copy of the assembled frame into per-stream GPU scratch
// (bumping the local rb_curr), per-batch dequantization into the dedisperser
// input buffer, and per-batch release_input_and_launch_dd_kernels.
// It does NOT bump rb_processed.
//
// frame_finalizing_thread: bridges H2G-copy completion (signaled via
// evrb_h2g) to the FrbServer ringbuf accounting. For each fired
// event it bumps rb_processed by beams_per_batch under 'mutex' and notifies
// cv (waking the reaper).
//
// Note that FrbServer::start() also spawns a grpc service with its own worker
// threads. These threads will be unpinned (default system-wide affinity), since
// they're spawned lazily by gRPC's runtime. This is fine for now, since all
// RPCs are "lightweight". (In the future, when we define "heavyweight" RPCs
// for e.g. injections, then we may want to look at gRPC's ResourceQuota and
// custom thread pool mechanisms.)


struct FrbServer : public std::enable_shared_from_this<FrbServer>
{
    struct Params {
        // Dedispersion config specified at construction.
        // Some members (zone_nfreq, zone_freq_edges, time_sample_ms, beams_per_gpu)
        // will be overwritten ("filled") later. (The "filled" DedispersionConfig
        // is available as FrbServer::plan->config.)
        DedispersionConfig config_prefilled;

        std::shared_ptr<BumpAllocator> host_allocator;
        std::shared_ptr<BumpAllocator> gpu_allocator;
        int cuda_device_id = -1;
        
        std::vector<std::shared_ptr<Receiver>> receivers;
        std::shared_ptr<FileWriter> file_writer;
        std::string rpc_server_address;
        int ringbuf_nchunks = 0;  // logical ring buffer length in time chunks

        // Minimum data-NIC MTU expected by the receiver. Used (only) on
        // the sender side: surfaced via the GetConfig RPC so a fake X-engine
        // sender can check that its outgoing NIC has MTU >= this value.
        int min_data_mtu = 0;

        // Artificial per-frame delay injected by the processing thread,
        // in seconds. Used to simulate slow GPU processing for testing
        // the FakeXEngine pacing path; defaults to 0 (no delay).
        // Applied off-lock so worker threads can keep advancing
        // rb_assembled during the "work".
        double processing_delay_sec = 0.0;

        // gRPC address ("ip:port") of the FrbGrouper this server feeds. Empty
        // string => no grouper (no Session RPC; GpuDedisperser built with
        // num_consumers=0). Must be a loopback address (CUDA IPC requires the
        // grouper to be on the same physical GPU); the constructor enforces this.
        std::string grouper_ip_addr;

        // Weight-ring depth (GpuDedisperser::Params::nbatches_wt) of the internal
        // dedisperser. 0 (default) = num_active_batches. Must be >= num_active_batches
        // if nonzero (enforced by the GpuDedisperser constructor). Only used by unit
        // tests (e.g. 'test --serv'), which randomize the weight-ring depth.
        long nbatches_wt = 0;

        // If true, the processing thread skips ALL GPU work: data is not even
        // copied host->device, and no dequantization / dedispersion kernels are
        // launched. The receive/assemble/ringbuf/reaper pipeline still runs in
        // full (the processing thread still consumes frames from the ring buffer
        // so rb_processed advances normally). The dedisperser is still fully
        // constructed and allocated -- it is just never fed. This is an
        // infrequently used corner case (e.g. exercising the network/assembly
        // path with no GPU). --no-dedispersion implies --no-grouper, so
        // grouper_ip_addr must be empty when this is set (constructor asserts).
        bool no_dedispersion = false;

        // If true, suppress the per-chunk "FrbServer: beamset=..." stdout line
        // (printed once per fully-assembled chunk). Everything else is unaffected.
        // Set true by callers that would otherwise be swamped by per-chunk output
        // (e.g. the ephemeral test server, which assembles hundreds of chunks).
        bool quiet = false;
    };

    // Factory method (constructor is private).
    static std::shared_ptr<FrbServer> create(const Params &params);

    ~FrbServer();  // calls stop(), joins workers/reaper, then rpc_server->Wait()

    // Start/stop the Receivers and the RPC service.
    // Asynchronous: neither start() nor stop() calls rpc_server->Wait().
    void start();  // entry point
    void stop(std::exception_ptr e = nullptr);  // idempotent

    Params params;
    std::shared_ptr<AssembledFrameAllocator> frame_allocator;

    std::unique_ptr<FrbRpcService> rpc_service;
    std::unique_ptr<grpc::Server> rpc_server;

    // Thread-backed class state (protected by mutex).
    std::mutex mutex;
    std::condition_variable cv;  // signaled on: stop, metadata available
    bool is_started = false;
    bool is_stopped = false;
    std::exception_ptr error;

    // Cached metadata pointer (canonical copy lives in the
    // AssembledFrameAllocator; this is the snapshot the FIRST worker thread
    // captured when it sized frame_ringbuf). Set under 'mutex' by the first
    // worker that completes its init block; null before that. Used by the
    // worker's "is this the first init?" check; the reaper and RPC handlers
    // read metadata via frame_allocator->get_metadata() instead.
    //
    // freq_channels: always FREQUENCY-SCRUBBED (empty), inherited from the
    // frame_allocator's canonical copy.
    std::shared_ptr<const XEngineMetadata> metadata;

    // Maps (beam id) -> (position in metadata.beam_ids).
    std::unordered_map<long,int> beam_id_to_index;

    // The frame_ringbuf is initialized at the same time as the metadata.
    // Ring buffer has length (params.ringbuf_nchunks * metadata.get_nbeams()).
    std::vector<std::shared_ptr<AssembledFrame>> frame_ringbuf;

    // ----- Streams (StartStream / ShowStreams / CancelStream RPCs) -----

    // A "stream" (pirate::FileStream, defined in FileWriter.hpp): frames
    // matching (beam_ids x chunk range) are queued for disk writing by the
    // frame_finalizing_thread, at the moment each frame transitions from
    // "assembled" to "processed" (i.e. just before rb_processed advances
    // past it -- see the capture-hook comment in
    // _frame_finalizing_thread_main). Chunks already processed when the
    // stream was registered are NOT captured retroactively (that's
    // WriteFiles' job).

    // Protected by 'mutex'. Expired streams (chunk_last fully processed)
    // are deactivated lazily, by the frame_finalizing_thread's capture hook
    // and by the ShowStreams handler; CancelStream deactivates eagerly.
    std::vector<std::shared_ptr<FileStream>> active_streams;

    // Ring of the last inactive_file_stream_capacity deactivated streams
    // (expired or cancelled), retained for ShowStreams history. Protected
    // by 'mutex'. Uses the same absolute-counter + modular-slot idiom as
    // frame_ringbuf/rb_*: deactivated stream number i (0-based, in
    // deactivation order) lives in slot i % capacity, so the ring holds
    // entries [max(0, n - capacity), n) where n = num_deactivated_streams
    // (the monotone total, exported by ShowStreams so clients can detect
    // dropped history). Eviction just drops the shared_ptr; ring residency
    // is display-only history -- cancellation semantics (drain,
    // notifications, counters) never depend on it.
    std::array<std::shared_ptr<FileStream>,
               constants::inactive_file_stream_capacity> inactive_streams;
    long num_deactivated_streams = 0;

    // Deactivate one stream: stamp the deactivation fields (cancelled,
    // deactivated_at_unix_ns) and append it to the inactive ring, evicting
    // the oldest entry beyond capacity. Caller must hold 'mutex' and is
    // responsible for removing 'st' from active_streams. Never retracts
    // queued writes -- the FileWriter drains them regardless.
    void _deactivate_stream(const std::shared_ptr<FileStream> &st, bool cancelled);

    // Queue one file write on 'frame': push {relpath, stream} onto
    // frame->save_paths and hand the frame to the FileWriter (which does
    // all disk I/O on its own threads; process_frame only enqueues).
    // Returns false if the frame was skipped because it has been reaped
    // without ever being written. Shared by the WriteFiles RPC handler
    // (stream = nullptr) and the frame_finalizing_thread's stream-capture
    // hook. Caller must NOT hold 'mutex' or frame->mutex.
    bool _queue_frame_write(const std::shared_ptr<AssembledFrame> &frame,
                            const std::string &relpath,
                            const std::shared_ptr<FileStream> &stream);

    // Dedispersion plan built by the processing thread once X-engine metadata
    // is available. Null before that; non-null once the plan is announced.
    // Protected by 'mutex'.
    std::shared_ptr<DedispersionPlan> plan;

    // GpuDedisperser built by the processing thread immediately after 'plan'.
    // Published atomically with 'plan' (same critical section), so both
    // members transition from null to non-null together. Protected by 'mutex'.
    std::shared_ptr<GpuDedisperser> dedisperser;

    // Lock-protected. Published by processing_thread once
    // GpuDedisperser::allocate() returns and the post-allocate setup
    // is complete (along with dedisperser_is_initialized = true).

    // evrb_dq: produced by the dequantization kernel on compute_stream,
    //   consumed (with lag nstreams) by the H2G copies on h2g_stream.
    //   Gates per-stream-slot scratch reuse.
    std::shared_ptr<CudaEventRingbuf> evrb_dq;

    // evrb_h2g: produced by processing_thread on h2g_stream after the
    //   per-batch H2G copies complete. Two consumers: the dequantization
    //   kernel (h2g -> compute barrier on compute_stream) and the
    //   frame_finalizing_thread.
    std::shared_ptr<CudaEventRingbuf> evrb_h2g;

    // Lock-protected. Set to true by processing_thread once
    // { plan, dedisperser, evrb_dq, evrb_h2g } are
    // all assigned. Used by frame_finalizing_thread to know it's safe
    // to read evrb_h2g.
    bool dedisperser_is_initialized = false;

    // "Frame ids" are defined as (time_chunk_index * nbeams + ibeam).
    // See below for invariants.
    long rb_start        = 0;   // (first frame_id in ringbuf)
    long rb_reaped       = 0;   // (last reaped frame_id) + 1
    long rb_processed    = 0;   // (last GPU-processed frame_id) + 1
    long rb_assembled    = 0;   // (last fully-assembled frame_id) + 1
    long rb_end          = 0;   // (last frame_id in ringbuf) + 1
    bool rb_initialized  = false;

    // Worker threads (one per receiver).
    std::vector<std::thread> workers;

    // Reaper thread (only spawned in non-dummy mode).
    std::thread reaper_thread;

    // Processing thread: builds a DedispersionPlan and GpuDedisperser
    // once X-engine metadata has arrived, then runs the dedispersion
    // ingest pipeline (H2G copies + dequantization + dedispersion
    // kernel launches). See top-of-file comment.
    std::thread processing_thread;

    // Frame-finalizing thread: advances rb_processed by beams_per_batch
    // each time the processing_thread's per-batch H2G copies complete
    // (signaled via evrb_h2g). See top-of-file comment.
    std::thread frame_finalizing_thread;

    // ----- FrbGrouper client state (only used when params.grouper_ip_addr is set) -----

    // gRPC client state for the FrbGrouper Session stream. Defined in
    // FrbServer.cpp (holds channel, stub, ClientContext, ClientReaderWriter).
    // Hidden behind a pImpl so this header does not pull in grpc++/protobuf.
    struct GrouperClient;
    std::unique_ptr<GrouperClient> grouper_client;   // null unless grouper enabled

    // Set true (under mutex) by grouper_send_thread once the handshake has been
    // sent and HandshakeReply received; wakes grouper_receive_thread.
    bool grouper_handshake_done = false;

    // Grouper threads (only spawned when params.grouper_ip_addr is non-empty).
    std::thread grouper_send_thread;
    std::thread grouper_receive_thread;

    // ----- Noncopyable, nonmoveable -----

    FrbServer(const FrbServer &) = delete;
    FrbServer &operator=(const FrbServer &) = delete;
    FrbServer(FrbServer &&) = delete;
    FrbServer &operator=(FrbServer &&) = delete;

    // Call with lock held.
    inline void _check_rb_invariants()
    {
        xassert(rb_start >= 0);
        xassert(rb_start     <= rb_reaped);
        xassert(rb_reaped    <= rb_processed);
        xassert(rb_processed <= rb_assembled);
        xassert(rb_assembled <= rb_end);
        xassert(rb_end <= rb_start + long(frame_ringbuf.size()));
    }

private:
    // Private constructor (use create() instead).
    FrbServer(const Params &params);

    // Helper for entry points. Caller must hold mutex.
    void _throw_if_stopped(const char *method_name);

    // Worker thread functions.
    void _worker_main(int receiver_index);
    void worker_main(int receiver_index);

    // Reaper thread functions.
    void _reaper_thread_main();
    void reaper_thread_main();

    // Processing thread functions.
    void _processing_thread_main();
    void processing_thread_main();

    // Frame-finalizing thread functions.
    void _frame_finalizing_thread_main();
    void frame_finalizing_thread_main();

    // Grouper send/receive thread functions (mirroring the existing pairs).
    void _grouper_send_thread_main();
    void grouper_send_thread_main();
    void _grouper_receive_thread_main();
    void grouper_receive_thread_main();

    // Fills 'hs' from the (initialized) dedisperser + plan. See FrbServer.cpp.
    void _fill_handshake(frb::grouper::v1::Handshake *hs,
                         const std::shared_ptr<GpuDedisperser> &dd,
                         const std::shared_ptr<DedispersionPlan> &plan_snap);
};


}  // namespace pirate

#endif // _PIRATE_FRB_SERVER_HPP
