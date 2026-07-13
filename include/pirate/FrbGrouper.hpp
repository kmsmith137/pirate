#ifndef _PIRATE_FRB_GROUPER_HPP
#define _PIRATE_FRB_GROUPER_HPP

#include "Dedisperser.hpp"          // GpuDedisperser::Outputs (+ ksgpu Array/Dtype)
#include "XEngineMetadata.hpp"
#include "DedispersionConfig.hpp"
#include "constants.hpp"            // FrbGrouperClient default timeouts

#include <yaml-cpp/yaml.h>          // YAML::Node

#include <chrono>
#include <condition_variable>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// Forward-declare the generated proto types used below (by FrbGrouper's
// _process_handshake() and by FrbGrouperClient's write()/read()), so this header
// (included by pybind) does not pull in grpc++/protobuf. The .cpp includes the
// generated headers.
namespace frb::grouper::v1 { class Handshake; class ProducerMessage; class ConsumerMessage; }

namespace pirate {
#if 0
}  // editor auto-indent
#endif

struct DedispersionPlan;   // defined in DedispersionPlan.hpp


// -------------------------------------------------------------------------------------------------
//
// FrbGrouperClient: gRPC *client* side of the FrbGrouper service -- the producer
// (FrbServer) end of the connection. Owns the channel / stub / Session stream and
// the connect/ping/cancel logic; FrbServer drives the protocol (Handshake +
// produced/consumed loops) over it, on threads FrbServer owns.
//
// Usage: run_server constructs one per server, calls ping() EARLY (before bump
// allocation) to fail fast if the grouper isn't running, then passes it into the
// FrbServer constructor (the way Receivers / FileWriter are passed in). Later,
// FrbServer's grouper_send_thread calls connect() just before the Handshake.


struct FrbGrouperClient
{
    explicit FrbGrouperClient(const std::string &grouper_ip_addr);
    ~FrbGrouperClient();

    // The grouper's "ip:port" (loopback -- CUDA IPC requires it on the same node).
    std::string grouper_ip_addr;

    // Channel-level connectivity check: bring a throwaway channel to READY, then
    // drop it. NO Session RPC, NO Handshake -- so it does not touch the grouper's
    // single-session state. Throws runtime_error if the grouper is not reachable
    // within timeout_ms. Meant to be called early in server startup.
    void ping(int timeout_ms = constants::grouper_ping_timeout_ms);

    // Open the real connection: build a fresh channel + stub, wait for READY
    // (throws runtime_error on timeout), and open the Session stream. Called by
    // FrbServer::grouper_send_thread just before the Handshake.
    void connect(int timeout_ms = constants::grouper_connect_timeout_ms);

    // Session-stream I/O forwarded to the ClientReaderWriter (valid only after
    // connect()). Return false when the stream is closed (mirrors grpc Write/Read).
    // gRPC permits one concurrent Write + one concurrent Read (send vs receive
    // thread), which is how FrbServer uses these.
    bool write(const frb::grouper::v1::ProducerMessage &msg);
    bool read(frb::grouper::v1::ConsumerMessage *msg);

    // Unblock any in-flight write()/read() (idempotent; safe from any thread).
    // ClientContext::TryCancel() makes the pending Write/Read return false. Used
    // by FrbServer::stop().
    void cancel();

    // Noncopyable / nonmovable (owns gRPC state).
    FrbGrouperClient(const FrbGrouperClient&) = delete;
    FrbGrouperClient& operator=(const FrbGrouperClient&) = delete;

private:
    // pImpl: channel / stub / context / Session stream. Defined in FrbGrouper.cpp
    // so this header stays free of grpc++/protobuf.
    struct GrpcState;
    std::unique_ptr<GrpcState> grpc_state;
};


// -------------------------------------------------------------------------------------------------
//
// FrbGrouper: gRPC *server* side of the FrbGrouper service (frb_grouper.proto).
//
// A downstream Python consumer uses FrbGrouper to receive an FrbServer's
// GpuDedisperser::output_ringbuf over CUDA IPC. The producer (FrbServer, the
// gRPC *client*) connects, sends a Handshake (CUDA IPC mem handle + array
// descriptors + run-context YAML), then streams produced_seq_id as each output
// batch is written; the consumer streams back consumed_seq_id as it finishes
// each batch. FrbGrouper opens the IPC handle, reconstructs the ring-buffer
// arrays as views into the shared GPU memory, and exposes per-batch slices to
// Python via acquire_output() / release_output().
//
// Thread-backed class. Threads per session: the gRPC Session handler (a gRPC
// pool thread) IS the "receive thread" (reads produced_seq_id), plus one
// spawned "send thread" (writes HandshakeReply + consumed_seq_id), plus the
// Python consumer thread (acquire_output / release_output). stop() unblocks the
// handler's blocking Read via ServerContext::TryCancel(); see FrbGrouper.cpp
// for the full threading / teardown rationale.
//
// Single session only: open() may be called once, and a second concurrent
// Session RPC is rejected. There is no reconnect support (the intended
// deployment is one producer per consumer-process lifetime).

struct FrbGrouper
{
    // Factory method (constructor is private). Constructs the RPC service but
    // does NOT listen; call open() to start listening + wait for a client.
    static std::shared_ptr<FrbGrouper> create(const std::string &ip_addr);

    ~FrbGrouper();   // close(): stop(); join send thread; server shutdown

    // ----- Metadata sent by the RPC client (populated at handshake) -----
    std::shared_ptr<XEngineMetadata> xengine_metadata;
    DedispersionConfig dedispersion_config;
    YAML::Node dedispersion_plan_yaml;      // NOT pybind-wrapped (see injections)

    // "Incomplete" DedispersionPlan, deserialized from the handshake yamls -- see
    // DedispersionPlan::make_incomplete_plan_from_yaml(). Supports decode_argmax*() and
    // compute_steady_state_it0() (with the PRODUCER's per-tree Dcore values); its low-level
    // members (MegaRingbuf, kernel/buffer params) are uninitialized. Internal hack --
    // deliberately not pybind-wrapped.
    std::shared_ptr<DedispersionPlan> incomplete_plan;

    std::string xengine_metadata_yaml_string;
    std::string dedispersion_config_yaml_string;
    std::string dedispersion_plan_yaml_string;

    // The grouper's own listen address ("ip:port"), specified at construction.
    std::string grouper_ip_addr;

    // The producer FrbServer's FrbSearch RPC endpoint ("ip:port"), from the handshake
    // (Handshake::rpc_ip_addr == FrbServer::Params::rpc_server_address). Lets a
    // consumer reach back to the producing server over its frb_search RPC.
    std::string search_ip_addr;

    // ----- Convenience accessors (derived at handshake) -----
    int cuda_device_id = -1;      // from handshake
    ksgpu::Dtype dtype;           // = dedispersion_config.dtype
    long nt_in = 0;               // = dedispersion_config.time_samples_per_chunk
    long total_beams = 0;         // = dedispersion_config.beams_per_gpu. The TOTAL beams
                                  //   per chunk; used to map a seq_id to a physical beam
                                  //   range. NOT the output-ringbuf leading axis (see below).
    long beams_per_batch = 0;     // = dedispersion_config.beams_per_batch
    long nbatches = 0;            // = total_beams / beams_per_batch (beam-batches per time
                                  //   chunk, covering all beams). NOT num_batch_slots (the
                                  //   ring depth); producer seq_id = ichunk*nbatches + ibatch.
    long num_batch_slots = 0;     // = handshake num_batch_slots (== producer nbatches_out).
                                  //   The output_ringbuf leading (beam) axis has length
                                  //   num_batch_slots * beams_per_batch, which is <=
                                  //   total_beams (NOT equal to total_beams in general).
    long initial_chunk = 0;       // = handshake initial_chunk (producer's
                                  //   GpuDedisperser::Params::initial_chunk). Time-chunk
                                  //   index of the first dedispersion output relative to
                                  //   FPGA seq 0; used to set Outputs::ichunk_fpga_based.

    long ntrees = 0;              // = dedispersion_plan_yaml['ntrees']
    std::vector<long> ndm_out;    // length ntrees, from dedispersion_plan_yaml['trees'][:]['ndm_out']
    std::vector<long> nt_out;     // length ntrees, from dedispersion_plan_yaml['trees'][:]['nt_out']

    // ----- Lifecycle (entry points) -----
    //
    // Teardown is RAII-safe: the Session handler holds a bare back-pointer,
    // never an owning reference (see the FrbGrouperService comment in
    // FrbGrouper.cpp), so dropping the last reference runs ~FrbGrouper on
    // the dropping thread. Its close() cancels any active session and shuts
    // the gRPC server down -- waiting for the handler -- before members are
    // destroyed. An explicit close() (e.g. from the Python context manager)
    // remains good practice for prompt, predictable teardown, but is not
    // required for correctness.
    void open();    // start listening + block until client connects + handshake processed
    void close();   // stop() + join + server shutdown (deterministic teardown)
    void stop(std::exception_ptr e = nullptr) const;   // idempotent

    // open() decomposed into two steps, so the pybind binding can drive the wait
    // in 0.5s increments and poll for Ctrl-C between them (see the .cpp + binding):
    void start_listening();                   // bind port + spawn send thread (once); non-blocking
    bool wait_for_handshake(int timeout_ms);  // true once handshake done; else waits timeout_ms

    // Blocks until produced_seq_id has been received for 'seq_id'; returns a
    // per-batch slice (nbeams == beams_per_batch) of output_ringbuf.
    GpuDedisperser::Outputs acquire_output(long seq_id);

    // Non-blocking: records that the caller is done with 'seq_id'; the send
    // thread will emit CONSUMED(seq_id).
    void release_output(long seq_id);

    // Forwards to DedispersionPlan::compute_steady_state_it0() on the producer's
    // (incomplete) plan from the handshake -- see that method for the meaning of the
    // returned array. (A forwarder is needed since 'incomplete_plan' is deliberately
    // not pybind-wrapped.) Valid only after the handshake.
    ksgpu::Array<long> _compute_steady_state_it0(long itree) const;

    bool is_stopped_pub();   // lock-protected read, for Python polling

    // Noncopyable / nonmovable.
    FrbGrouper(const FrbGrouper&) = delete;
    FrbGrouper& operator=(const FrbGrouper&) = delete;

private:
    // The gRPC service (defined in FrbGrouper.cpp) forwards Session() into the
    // private _run_session(); it needs access to that private entry point.
    friend class FrbGrouperService;

    explicit FrbGrouper(const std::string &ip_addr);

    // pImpl: gRPC service + server + the active session's context/stream.
    // Defined in FrbGrouper.cpp so this header stays free of grpc++/protobuf.
    struct GrpcState;
    std::unique_ptr<GrpcState> grpc_state;

    // Thread-backed state.
    mutable std::mutex mutex;

    // One condition variable per wait-predicate (see the "Concurrency"
    // section of notes/cpp.md):
    //
    // handshake_cv -- waiters: wait_for_handshake() (timed poll) and
    //   _send_loop()'s startup wait (predicate: handshake_done, or stopped).
    //   Signaled on: the handshake latch in _run_session() (notify_all --
    //   one-shot latch), and stop().
    //
    // produced_cv -- waiter: acquire_output() (predicate: handshake_done &&
    //   rb_produced > seq_id, or stopped). At most one thread can be parked
    //   here: the consecutive-seq_id cursor xassert in acquire_output() makes
    //   a second concurrent caller throw instead of waiting -- so the
    //   rb_produced advance uses notify_one. Signaled on: rb_produced advance
    //   in _receive_loop() (notify_one), the handshake latch (notify_all --
    //   the predicate's handshake_done conjunct), and stop().
    //
    // consumed_cv -- waiter: the send thread's drain wait in _send_loop()
    //   (predicate: rb_consumed_sent < rb_consumed, or stopped). Signaled on:
    //   rb_consumed advance in release_output() (notify_one -- single
    //   waiter), and stop().
    //
    // send_io_cv -- waiter: the Session handler's step-5 teardown wait in
    //   _run_session() (predicate: send_io_done; deliberately NOT
    //   stop-sensitive -- the handler must wait for the send thread to stop
    //   touching the stream even when stopped). Signaled on: send-thread exit
    //   in send_thread_main() (notify_one -- single waiter), the vacuous
    //   send_io_done set in start_listening()'s failure path, and stop()
    //   (harmless spurious wake; predicate re-checked).
    mutable std::condition_variable handshake_cv;
    mutable std::condition_variable produced_cv;
    mutable std::condition_variable consumed_cv;
    mutable std::condition_variable send_io_cv;

    mutable bool is_stopped = false;
    mutable std::exception_ptr error;

    // Session coordination flags (all under mutex).
    bool session_active  = false;   // single-client guard
    bool handshake_done  = false;   // handshake processed; output_ringbuf valid
    bool send_io_done    = false;   // send thread has stopped touching the stream
    bool opened          = false;   // start_listening() called-once guard (single session)
    bool closed          = false;   // close() idempotency guard

    // Serializes concurrent close() calls: the second caller blocks until
    // the first has fully torn down (see FrbGrouper::close()). Leaf-level:
    // never acquired while 'mutex' is held.
    std::mutex close_mutex;

    // Guards the one-time "waiting for FrbServer to connect" stdout message in
    // wait_for_handshake() (which is polled every ~0.5s). Under mutex; set true
    // after the message is printed once, so it does not repeat.
    bool waiting_print_done = false;

    // Progress counters (under mutex).
    long rb_produced      = 0;      // (last produced_seq_id) + 1   [updated by receive loop]
    long rb_acquired      = 0;      // (last seq_id passed to acquire_output) + 1
    long rb_consumed      = 0;      // (last seq_id passed to release_output) + 1
    long rb_consumed_sent = 0;      // (last consumed_seq_id written to wire) + 1

    // Shared GPU ring buffer: views into the IPC-mapped producer memory. NOT
    // initialized until the handshake is processed (see _process_handshake); it
    // is an empty Outputs before a client connects. Full ring buffer:
    // nbeams == num_batch_slots * beams_per_batch (== producer nbatches_out *
    // beams_per_batch); this is <= total_beams (beams_per_gpu), NOT equal to it
    // in general. Private: consumers reach per-batch slices only via acquire_output().
    GpuDedisperser::Outputs output_ringbuf;

    // IPC mapping base; deleter calls cudaIpcCloseMemHandle. Every output
    // array's Array::base aliases this, so the mapping outlives all views.
    std::shared_ptr<void> ipc_base;

    std::thread send_thread;        // write loop; the receive loop runs on the
                                    // gRPC handler (pool) thread, not a member thread

    // Result of _run_session(), mapped to a grpc::Status by the .cpp service
    // handler. (Kept grpc-free so this header doesn't pull in grpc++.)
    enum class SessionResult { ok, stopped, busy };

    // Internals.
    void _throw_if_stopped(const char *method_name);   // caller holds mutex

    // Called from the .cpp service handler. 'ctx' / 'stream' are actually
    // grpc::ServerContext* / grpc::ServerReaderWriter<...>* passed as void* to
    // keep grpc out of this header; the .cpp casts them back.
    SessionResult _run_session(void *ctx, void *stream);

    void _process_handshake(const frb::grouper::v1::Handshake &hs);  // fwd-declared proto type
    void _receive_loop();           // read produced_seq_id (handler thread)
    void _send_loop();              // write HandshakeReply + consumed_seq_id (send thread)
    void send_thread_main();
};


}  // namespace pirate

#endif // _PIRATE_FRB_GROUPER_HPP
