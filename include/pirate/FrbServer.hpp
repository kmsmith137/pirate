#ifndef _PIRATE_FRB_SERVER_HPP
#define _PIRATE_FRB_SERVER_HPP

#include "Receiver.hpp"
#include "AssembledFrame.hpp"
#include "DedispersionConfig.hpp"
#include "XEngineMetadata.hpp"

#include <ksgpu/xassert.hpp>

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

namespace pirate {
#if 0
}  // editor auto-indent
#endif


struct FileWriter;        // FileWriter.hpp
struct DedispersionPlan;  // DedispersionPlan.hpp
struct GpuDedisperser;    // Dedisperser.hpp
struct CudaStreamPool;    // CudaStreamPool.hpp
struct FrbRpcService;     // defined in FrbServer.cpp


// FrbServer: a thread-backed class that manages Receivers and an RPC service.
//
// Backing threads: one worker thread per Receiver, a reaper thread, and a
// processing thread (which builds a DedispersionPlan once X-engine metadata
// has arrived). These threads inherit their vcpu affinity from the caller
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
// CudaStreamPool) once the plan exists. Both 'plan' and 'dedisperser'
// become non-null atomically (published in the same critical section).
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
    };

    // Factory method (constructor is private).
    static std::shared_ptr<FrbServer> create(const Params &params);

    ~FrbServer();  // calls stop(), joins workers/reaper, then rpc_server->Wait()

    // Start/stop the Receivers and the RPC service.
    // Asynchronous: neither start() nor stop() calls rpc_server->Wait().
    void start();  // entry point
    void stop(std::exception_ptr e = nullptr);  // idempotent

    Params params;
    std::shared_ptr<AssembledFrameAllocator> allocator;

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
    // read metadata via allocator->get_metadata() instead.
    //
    // freq_channels: always FREQUENCY-SCRUBBED (empty), inherited from the
    // allocator's canonical copy.
    std::shared_ptr<const XEngineMetadata> metadata;

    // Maps (beam id) -> (position in metadata.beam_ids).
    std::unordered_map<long,int> beam_id_to_index;

    // The frame_ringbuf is initialized at the same time as the metadata.
    // Ring buffer has length (params.ringbuf_nchunks * metadata.get_nbeams()).
    std::vector<std::shared_ptr<AssembledFrame>> frame_ringbuf;

    // Dedispersion plan built by the processing thread once X-engine metadata
    // is available. Null before that; non-null once the plan is announced.
    // Protected by 'mutex'.
    std::shared_ptr<DedispersionPlan> plan;

    // GpuDedisperser built by the processing thread immediately after 'plan'.
    // Published atomically with 'plan' (same critical section), so both
    // members transition from null to non-null together. Protected by 'mutex'.
    std::shared_ptr<GpuDedisperser> dedisperser;

    // "Frame ids" are defined as (time_chunk_index * nbeams + ibeam).
    // See below for invariants.
    long rb_start = 0;       // (first frame_id in ringbuf)
    long rb_reaped = 0;      // (last reaped frame_id in ringbuf) + 1
    long rb_finalized = 0;   // (last finalized frame_id in ringbuf) + 1
    long rb_end = 0;         // (last frame_id in ringbuf) + 1

    // Worker threads (one per receiver).
    std::vector<std::thread> workers;

    // Reaper thread (only spawned in non-dummy mode).
    std::thread reaper_thread;

    // Processing thread: builds a DedispersionPlan and GpuDedisperser
    // once X-engine metadata has arrived, then exits (more steps to be
    // added later).
    std::thread processing_thread;

    // ----- Noncopyable, nonmoveable -----

    FrbServer(const FrbServer &) = delete;
    FrbServer &operator=(const FrbServer &) = delete;
    FrbServer(FrbServer &&) = delete;
    FrbServer &operator=(FrbServer &&) = delete;

    // Call with lock held.
    inline void _check_rb_invariants()
    {
        xassert(rb_start >= 0);
        xassert(rb_start <= rb_reaped);
        xassert(rb_reaped <= rb_finalized);
        xassert(rb_finalized <= rb_end);
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
};


}  // namespace pirate

#endif // _PIRATE_FRB_SERVER_HPP
