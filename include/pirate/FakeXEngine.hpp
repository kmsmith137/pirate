#ifndef _PIRATE_FAKE_XENGINE_HPP
#define _PIRATE_FAKE_XENGINE_HPP

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "AssembledFrame.hpp"   // AssembledFrameSet (Command holds a shared_ptr<>)
#include "XEngineMetadata.hpp"
#include "network_utils.hpp"   // Socket (Worker holds one by value)


// Forward-declare grpc::ClientContext so we can hold a
// std::unique_ptr<grpc::ClientContext> as a FakeXEngine member without
// pulling the heavy grpc++ headers into every translation unit that
// includes FakeXEngine.hpp. (The FrbSearch::Stub used by the pacing
// thread is nested -- not forward-declarable in standard C++ -- so
// the Stub is created as a local in _pacing_thread_main rather than
// held as a member; see the paced-mode section below.)
namespace grpc { class ClientContext; }


namespace pirate {
#if 0
}   // pacify editor auto-indent
#endif


// FakeXEngine: simulates multiple upstream X-engine nodes sending data to a
// single receiver. Driven externally by a controller thread that submits
// Command objects to per-worker queues.
//
// This class follows the "thread-backed class" pattern (see
// notes/thread_backed_class.md), but with the synchronization state moved
// *into* each Worker rather than shared across workers. FakeXEngine itself
// holds no mutex / cv / is_stopped bool / error slot of its own. The only
// FakeXEngine-level "stopped" representation is the atomic
// 'is_stopped_cache', which is purely an O(1) cache for the pybind11
// 'is_stopped' property reader and a re-entry guard for stop(); workers
// and entry points synchronize through each Worker's own mutex+cv+flag.
//
// Each worker thread:
//   - Waits on its per-worker command queue for SEND_JUNK commands.
//   - On the first SEND_JUNK, opens a TCP connection to its assigned
//     receiver IP and sends the protocol header (magic + flags + YAML
//     metadata).
//   - On every SEND_JUNK, sends one minichunk: a 12-byte header (uint32
//     magic + uint64 seq) followed by a shape-(nbeams, nfreq, 256) int4
//     data array (all zeros -- "junk"). The wire seq is derived from
//     Command::minichunk_index.
//
// Workers are assigned round-robin to IP addresses (nworkers must be a
// multiple of ip_addrs.size()). Frequency channels are assigned round-
// robin to worker threads. There is NO internal cross-worker barrier --
// the controller thread is responsible for any "minichunk N waits for
// (N-2)" style serialization by interleaving wait_until_processed() and
// enqueue_send_junk() calls. See plans/fake_xengine_command_queue.md
// for the canonical controller pseudocode.
//
// Worker threads inherit the vcpu affinity of the thread that calls the
// FakeXEngine constructor. Python callers MUST call the constructor
// inside a ThreadAffinity context manager.
//
// Paced mode (paced=true, the default): the FakeXEngine spawns one
// extra "pacing thread" that holds a streaming MonitorRingbuf RPC to
// the FrbServer and broadcasts each pushed rb_processed value to all
// workers (Worker::rb_processed under each Worker::mutex). Worker
// threads gate their sends so the sender stays at most 5 time chunks
// ahead of the server: before each SEND_JUNK / SEND_MINICHUNK, the
// worker blocks on its cv until Worker::rb_processed >= (ichunk - 5)
// * nbeams, where ichunk = cmd.minichunk_index / minichunks_per_chunk.
// A bootstrap floor (FakeXEngine::rb_processed_floor) covers the
// gap before the server has received enough data to publish its
// first rb_processed value. See plans/fake_xengine_pacing.md.
//
// Usage:
//   with ThreadAffinity(vcpu_list):
//       fxe = FakeXEngine(xmd, {"10.0.0.2:5000", "10.0.1.2:5000"}, 64,
//                         /*time_samples_per_chunk=*/32768)
//       # Spawn a controller thread (under the same affinity) that calls
//       # fxe.enqueue_send_junk / fxe.wait_until_processed in a loop.
//   ...
//   fxe.stop()   # signals workers and any in-flight entry points to exit.

struct FakeXEngine
{
    // Protocol magic number (little-endian): 0xf4bf4b02 where 02 is the version number.
    // Used both for the initial handshake AND for the header of every minichunk.
    static constexpr uint32_t protocol_magic = 0xf4bf4b02;
    // Timeout for send operations (milliseconds).
    static constexpr int send_timeout_ms = 10;

    // Bit set in the 32-bit "flags" field of the connection header
    // when the FakeXEngine is constructed with debug=true. Mirror
    // of the receiver-side FLAG_ACK in Receiver.cpp.
    static constexpr uint32_t FLAG_ACK = 0x1;

    // Per-minichunk status codes stored in Worker::minichunk_status
    // (one byte per state-advancing command queued on the worker,
    // in queue order). The codes for "wire-acked" outcomes
    // (DROPPED=0, ASSEMBLED=1) match the 1-byte ack values sent
    // by the receiver on the wire -- see notes/network_protocol.md.
    //
    // minichunk_status is appended-only and grows without bound,
    // which is one of the reasons debug mode is not for production.
    static constexpr unsigned char STATUS_DROPPED   = 0;  // sent, ack=0 received
    static constexpr unsigned char STATUS_ASSEMBLED = 1;  // sent, ack=1 received
    static constexpr unsigned char STATUS_SENT      = 2;  // sent, ack not received yet
    static constexpr unsigned char STATUS_QUEUED    = 3;  // in command_queue, not processed yet
    static constexpr unsigned char STATUS_SKIPPED   = 4;  // SKIP_MINICHUNK processed

    // ----- Nested types -----

    // Command: a unit of work submitted by an external controller thread
    // to a worker's queue.
    struct Command
    {
        enum class Kind : uint32_t {
            UNINITIALIZED  = 0,
            SEND_JUNK      = 1,   // send minichunk_buf with all-zero ("junk") data
            SKIP_MINICHUNK = 2,   // advance state, send nothing on the wire
            SEND_MINICHUNK = 3,   // gather data from frame_set, then send
            DISCONNECT     = 4,   // close the worker's TCP socket
            WAIT_FOR_ACKS  = 5,   // worker calls _read_acks(blocking=true). Only
                                  // enqueued via FakeXEngine::enqueue_wait_for_acks().
        } kind = Kind::UNINITIALIZED;

        // Used by all of {SEND_JUNK, SKIP_MINICHUNK, SEND_MINICHUNK}.
        // Ignored by DISCONNECT (which does not touch
        // last_processed_minichunk or last_queued_minichunk).
        // Wire seq = minichunk_index * 256 * xmd.seq_per_frb_time_sample.
        // Successive state-advancing commands on a worker must have
        // strictly +1 monotonic minichunk_index, with one exception:
        // the very FIRST such command may pick any minichunk_index >= 0
        // -- this is what makes NOTE-2-style nonzero-initial-chunk tests
        // work. The check fires at *queue time* in FakeXEngine::_enqueue
        // (raising runtime_error to the caller); a redundant defense-
        // in-depth check also fires at *processing time* in
        // _skip_or_send.
        long minichunk_index = -1;

        // Required for SEND_MINICHUNK; an empty pointer otherwise. The
        // set's frames supply the int4 data to gather into the wire
        // minichunk; its time_chunk_index and ntime determine which
        // minichunk-within-the-chunk we're sending. The set's metadata
        // is expected to be consistent with the worker's xmd (same
        // total_nfreq, same nbeams, same ntime).
        //
        // WARNING: the SEND_MINICHUNK gather loop reads frame.data WITHOUT
        // acquiring frame.mutex. This races the AssembledFrame reaper.
        // The reaper currently runs in FrbServer (not FakeXEngine), and
        // the assumption is that FrbServer and FakeXEngine are NEVER
        // colocated in the same process -- so the race doesn't occur in
        // practice. A defensive xassert_gt(frame.data.size, 0) catches
        // gross misuse (calling SEND_MINICHUNK on a reaped frame) but is
        // not safe against an actively-running reaper. See also the
        // identical warning on the send_minichunk() entry point below.
        std::shared_ptr<AssembledFrameSet> frame_set;
    };

    // PendingAck: one entry per in-flight wire minichunk awaiting an
    // ack byte from the receiver. Pushed into Worker::ack_queue by
    // _skip_or_send after a successful SEND_*; popped by _read_acks
    // when the receiver's ack byte arrives. Used by the FLAG_ACK
    // back-channel only.
    struct PendingAck
    {
        // The wire minichunk_index this ack corresponds to.
        long minichunk_index;
        // Snapshot of FakeXEngine::max_acked_minichunk[w.receiver_id]
        // taken JUST BEFORE _send_all() for this minichunk. Used in
        // _read_acks to lower-bound the receiver's curr_base_chunk
        // at the time it processed this minichunk: any ack-arrival
        // observed at snapshot time implies the receiver had already
        // processed (and emitted that ack for) some minichunk by then,
        // so its curr_base_chunk is at least one chunk-index below the
        // acked minichunk's chunk-index.
        long max_ack_at_submission;
    };

    // Worker: per-worker state and synchronization. Each Worker is an
    // independent "thread-backed" unit -- mutex, cv, and is_stopped/error
    // are all per-worker, so there is no cross-worker contention on the
    // hot path.
    struct Worker
    {
        // ---- All of these are protected by 'mutex'. ----

        mutable std::mutex mutex;
        // Notified by: enqueue_send_junk / enqueue_skip_minichunk /
        // enqueue_send_minichunk / enqueue_disconnect (after enqueue),
        // the worker thread (after a successful command processed-step
        // updates last_processed_minichunk), and stop() (when
        // is_stopped transitions to true).
        std::condition_variable cv;

        // Commands waiting to be processed by this worker, FIFO.
        std::deque<Command> command_queue;

        // Latest minichunk_index this worker has finished *processing*
        // for a state-advancing command (SEND_JUNK / SKIP_MINICHUNK /
        // SEND_MINICHUNK), or -1 if no such command has completed yet.
        // Only the worker thread writes this; external threads read it
        // (under 'mutex') via wait_until_processed(). (Name reflects
        // that SKIP_MINICHUNK *processes* a command without actually
        // sending anything.)
        long last_processed_minichunk = -1;

        // Latest minichunk_index that has been *enqueued* on this
        // worker for a state-advancing command, or -1 if no such
        // command has been queued yet. DISCONNECT commands do not
        // touch this field. Used in FakeXEngine::_enqueue to enforce
        // strict +1 monotonic submission ordering at queue time
        // (analogous to the processing-time check in _skip_or_send,
        // but with the friendlier semantics of throwing
        // runtime_error to the offending caller rather than tripping
        // an internal xassert + stop()).
        //
        // Protected by 'mutex'. _enqueue holds the mutex continuously
        // across (a) the sequentiality check, (b) this counter's
        // update, and (c) command_queue.push_back -- so the recorded
        // counter is always consistent with the FIFO order of the
        // queue.
        long last_queued_minichunk = -1;

        // The minichunk_index of the very first state-advancing
        // command queued on this worker, or -1 if no such command has
        // ever been queued. Set exactly once (when
        // last_queued_minichunk transitions from -1 to its first
        // value) and immutable thereafter. DISCONNECT commands do
        // not touch this field.
        //
        // Protected by 'mutex' (set inside _enqueue under the same
        // lock that holds last_queued_minichunk consistent).
        long first_minichunk = -1;

        // Per-worker stopped flag. Workers and entry points synchronize
        // exclusively through this flag (NOT FakeXEngine::is_stopped_cache).
        // Set by stop()'s per-worker sweep.
        bool is_stopped = false;

        // Per-worker error slot. On a successful sweep from
        // FakeXEngine::stop(e), every worker's error is set to e; on a
        // clean shutdown (e == nullptr) every worker's error stays null
        // and entry-point throws are "called on stopped instance"
        // runtime_errors.
        std::exception_ptr error;

        // Debug-mode back-channel state. Both empty unless
        // FakeXEngine::debug == true.
        //
        // Per-minichunk status code (one of FakeXEngine::STATUS_X).
        // Indexed by (minichunk_index - first_minichunk). Invariant
        // (held whenever 'mutex' is dropped):
        //     minichunk_status.size() == (last_queued_minichunk >= 0)
        //         ? (last_queued_minichunk - first_minichunk + 1)
        //         : 0
        //
        // Both the vector AND ITS CONTENTS are protected by 'mutex'
        // (single-byte reads + writes go through the lock).
        // Appended-only and grows without bound; one of several
        // reasons debug mode is not for production.
        std::vector<unsigned char> minichunk_status;

        // Pending acks (one entry per in-flight wire minichunk).
        // When a SEND_JUNK or SEND_MINICHUNK is sent successfully on
        // the wire, the worker pushes a PendingAck here (under
        // 'mutex'); when an ack byte arrives, _read_acks pops the
        // front and uses (minichunk_index - first_minichunk) to find
        // the right minichunk_status slot.
        //
        // Each PendingAck also carries 'max_ack_at_submission' -- a
        // snapshot of FakeXEngine::max_acked_minichunk[receiver_id]
        // taken just before the wire send. The ack-prediction
        // assertion in _read_acks uses this snapshot to lower-bound
        // the receiver's curr_base_chunk at the moment it processed
        // this minichunk. See PendingAck doc + _skip_or_send.
        std::deque<PendingAck> ack_queue;

        // Latest "effective" rb_processed observed for this worker
        // (paced mode only). Written by (a) the pacing thread on each
        // server-pushed update, and (b) the worker thread itself on
        // its first SEND_*, seeded from FakeXEngine::rb_processed_floor.
        // Read by the worker thread in the paced-mode gate inside
        // _skip_or_send. Monotone-nondecreasing -- both writers use
        // std::max on update. Zero (and unused) in non-paced mode.
        //
        // Protected by 'mutex'.
        long rb_processed = 0;

        // ---- Constant after construction, not lock-protected. ----

        std::thread worker_thread;

        // ---- Worker-thread-only state. Initialized in _initialize()
        // (called at the top of _worker_main) and modified only by
        // the worker thread. No other thread reads or writes these,
        // so they need no synchronization. ----

        // Index into FakeXEngine::max_acked_minichunk that maps this
        // worker to its Receiver on the server side. Set in
        // _initialize() to (worker_id % ip_addrs.size()), matching
        // the round-robin assignment of worker -> ip_addrs.
        long receiver_id = -1;

        // False until this worker has performed its very first SEND_JUNK
        // or SEND_MINICHUNK; flipped to true inside _skip_or_send the
        // first time the need_send branch is taken. Used to apply the
        // "+minichunks_per_chunk" shift to max_sent_minichunk on the
        // first send only (see the long comment in _skip_or_send).
        // Persists across DISCONNECT/reconnect cycles.
        bool first_send_done = false;

        // Per-worker XEngineMetadata, MEANINGFUL: freq_channels holds the
        // round-robin subset of channels this worker sends on the wire.
        // Built by make_worker_metadata() from FakeXEngine::xmd at worker
        // init time -- this is the form transmitted in the per-connection
        // YAML handshake (the receiver treats each TCP connection as one
        // X-engine node with a specific freq_channels subset).
        XEngineMetadata xmd;

        // Send buffer laid out as:
        //   [ 12-byte conn header + padded YAML ][ 12-byte mc header + (nbeams,nfreq,256) int4 data ]
        //   ^                                    ^
        //   send_buf.data()                      send_buf.data() + header_nbytes
        //
        // The first SEND_JUNK sends the whole buffer in one _send_all()
        // call; subsequent SEND_JUNKs send only the minichunk portion.
        // The minichunk's magic byte is stamped once in _initialize();
        // only the 8-byte seq field at offset (header_nbytes + 4) is
        // rewritten per SEND_JUNK.
        std::vector<char> send_buf;

        // Byte offset of the minichunk portion of send_buf.
        long header_nbytes = 0;

        // Size in bytes of the minichunk portion (= send_buf.size() - header_nbytes).
        long mc_nbytes = 0;

        // Parsed destination address (round-robin: worker_id % ip_addrs.size()).
        std::string ip_addr;
        uint16_t port = 0;

        // TCP socket. Default-constructed (fd == -1) at Worker creation;
        // reassigned via sock = Socket(PF_INET, SOCK_STREAM) on the first
        // SEND_JUNK before sock.connect().
        Socket sock;

        // False until sock.connect() has succeeded; flipped back to false
        // by _disconnect() when a DISCONNECT command is processed. The
        // discriminator between "first send-after-connect" (open + send
        // conn header + first mc) and "subsequent send" (just send the
        // mc).
        //
        // Atomic so external threads can sample it via the
        // FakeXEngine::is_connected() entry point without locking. The
        // worker thread is the sole writer; all accesses (worker and
        // external) use memory_order_relaxed -- the flag has no
        // synchronization relationship with any other Worker state, so
        // there's nothing to acquire/release. A stale snapshot from an
        // external reader is acceptable (this is an informational query).
        std::atomic<bool> connected{false};

        // Worker is neither copyable nor movable -- std::mutex and
        // std::condition_variable are non-copyable AND non-movable.
        // FakeXEngine therefore holds workers as
        // std::vector<std::unique_ptr<Worker>> rather than
        // std::vector<Worker>, so each Worker lives at a stable heap
        // address.
        Worker() = default;
        Worker(const Worker &) = delete;
        Worker &operator=(const Worker &) = delete;
        Worker(Worker &&) = delete;
        Worker &operator=(Worker &&) = delete;
    };

    // ----- Constructor args -----

    // Top-level metadata template; the constructor stores the shared_ptr
    // handed in by the caller. freq_channels: typically FREQUENCY-SCRUBBED
    // (empty) on the way in -- it's IGNORED here regardless, because
    // make_worker_metadata() builds each Worker's xmd by copying this
    // template and overwriting freq_channels with the per-worker
    // round-robin subset (see Worker::xmd above). Non-null (the
    // constructor throws if the caller passes nullptr).
    const std::shared_ptr<const XEngineMetadata> xmd;
    const std::vector<std::string> ip_addrs;  // each element is "ip:port"
    const int nworkers;

    // Receiver-side time_samples_per_chunk (from the FrbServer's
    // AssembledFrameAllocator). Required for the ack-prediction
    // assertion in _read_acks: it maps wire minichunk_index ->
    // chunk index via
    //   minichunks_per_chunk = time_samples_per_chunk / 256.
    // Non-optional. Validated in the constructor to be > 0 and a
    // multiple of 256.
    const long time_samples_per_chunk;

    // Adds real-time debugging checks with nontrivial cpu/network
    // cost, and unbounded memory usage. Useful for unit tests, but
    // don't use in production!
    const bool debug;

    // ----- Paced-mode config (constants after ctor) -----
    //
    // When true, FakeXEngine spawns a pacing thread that opens a
    // MonitorRingbuf streaming RPC to the FrbServer and gates each
    // worker's sends to stay <=5 chunks ahead of server-side
    // rb_processed. See class doc-comment + plans/fake_xengine_pacing.md.
    const bool paced;

    // gRPC address ("ip:port") of the FrbServer's RPC endpoint. Required
    // (non-empty) when paced=true; ignored when paced=false (a non-empty
    // value is silently accepted).
    const std::string rpc_address;

    // = xmd->get_nbeams(). Cached at construction for use in the
    // paced-mode gate (which computes 5-chunks-ahead in units of
    // frames = chunks * nbeams).
    const long nbeams;

    // ----- Derived from time_samples_per_chunk -----

    // = time_samples_per_chunk / 256. Number of wire minichunks per
    // assembled chunk on the receiver side. Used by the ack-prediction
    // assertion to convert minichunk_index -> chunk_index.
    const long minichunks_per_chunk;

    // ----- Derived from ip_addrs -----

    // = long(ip_addrs.size()). Each ip_addr corresponds to one
    // Receiver on the FrbServer; each worker is assigned to one
    // Receiver via round-robin (worker.receiver_id = worker_id %
    // num_receivers, set in _initialize).
    const long num_receivers;

    // ----- Stop-state cache -----

    // O(1) cache for the pybind11 'is_stopped' property reader, and the
    // re-entry guard for stop() (first compare_exchange_strong wins; later
    // callers return immediately). Set by FakeXEngine::stop() *before*
    // the per-worker sweep, so any property reader between the atomic
    // store and the last per-worker notify observes "stopped" correctly.
    //
    // NOT load-bearing for synchronization with worker threads -- those
    // synchronize through each Worker::is_stopped under its own mutex.
    std::atomic<bool> is_stopped_cache{false};

    // ----- Worker state -----

    // Length nworkers, but exposed as a vector of unique_ptr<Worker> so
    // that the Worker objects (which embed non-movable std::mutex and
    // std::condition_variable) are heap-allocated and stable in memory.
    // This avoids any need to make Worker movable.
    std::vector<std::unique_ptr<Worker>> workers;

    // ----- Debug-mode back-channel state (used iff debug=true) -----

    // Per-Receiver max minichunk_index acked, over all workers
    // connected to that Receiver. Vector of length num_receivers;
    // each entry init to -1 in the constructor body. Updated in
    // _read_acks via fetch_max(max_acked_minichunk[w.receiver_id],
    // done.minichunk_index) after each ack byte is processed.
    //
    // PER-RECEIVER (not per-server-aggregate) for correctness of the
    // lower bound on the receiver's cbc_B: an ack from Receiver A for
    // minichunk N tells us about A's cbc_A, NOT B's cbc_B. The
    // cross-Receiver evict mechanism (Receiver::evict from FrbServer
    // worker threads when frame_sets are pulled from sibling
    // Receivers) is an async thread-handoff that lags, so a global
    // "max ack" doesn't bound any specific Receiver's cbc.
    //
    // Read via fetch_max + load with memory_order_relaxed (the
    // happens-before chain that makes the bound hold goes through
    // the network, which the C++ memory model doesn't capture).
    std::vector<std::atomic<long>> max_acked_minichunk;

    // Max (over all workers, all Receivers) "shifted" minichunk_index
    // contribution. Updated in _skip_or_send via fetch_max JUST
    // BEFORE the call to _send_all (before, not after, to ensure
    // every receiver-processable minichunk is captured by the time
    // any later reader observes max_sent_minichunk -- otherwise an
    // ack-back-to-FakeXEngine could race the local atomic update).
    //
    // SHIFT: a worker's VERY FIRST SEND contributes
    //    cmd.minichunk_index + minichunks_per_chunk
    // (one full chunk ahead); subsequent sends contribute just
    // cmd.minichunk_index. This folds the receiver's
    // initial_time_chunk (which equals the chunk-index of whichever
    // worker's first send arrived first) into max_sent_minichunk,
    // so the upper bound on the receiver's cbc_B doesn't need a
    // separate init bracket. See the long comment in _skip_or_send.
    //
    // GLOBAL (not per-Receiver) is REQUIRED for the upper bound to
    // hold in multi-Receiver setups: cross-Receiver evict() from
    // FrbServer can advance cbc_B based on a minichunk sent to a
    // SIBLING Receiver, not to B itself. A per-Receiver max_sent[B]
    // would miss those sends. The global aggregate captures them all.
    std::atomic<long> max_sent_minichunk{-1};

    // Per-outcome counters for the three-way ack-prediction check in
    // _read_acks. Incremented once per ack byte processed when
    // debug=true (always zero when debug=false, since no acks arrive).
    // Indices:
    //   0 = unambiguous, DROPPED    (ii <  cbcp_lower; ack was DROPPED)
    //   1 = unambiguous, ASSEMBLED  (ii >= cbcp_upper; ack was ASSEMBLED)
    //   2 = ambiguous, DROPPED      (ambiguous band; ack == DROPPED)
    //   3 = ambiguous, ASSEMBLED    (ambiguous band; ack == ASSEMBLED)
    //
    // Initialized to zero in the constructor body. Updated with
    // memory_order_relaxed; read back via get_debug_counters() for
    // diagnostic purposes only (no synchronization relationship with
    // anything else).
    std::array<std::atomic<long>, 4> debug_counters;

    // ----- Paced-mode state -----

    // Bootstrap floor for paced mode. Initialized to -1; CAS-set ONCE by
    // whichever worker performs its first SEND_*, to (ichunk * nbeams)
    // where ichunk = cmd.minichunk_index / minichunks_per_chunk. Every
    // worker reads this on its own first SEND_* (whether or not it won
    // the CAS) and seeds its Worker::rb_processed to at least this
    // value. Covers the latency gap before the server has received
    // enough data to publish its first rb_processed via MonitorRingbuf.
    std::atomic<long> rb_processed_floor{-1};

    // MonitorRingbuf context. Owned here so FakeXEngine::stop() can
    // TryCancel() the streaming RPC to unblock the pacing thread's
    // Read(). The corresponding gRPC Stub is not a member -- it lives
    // as a local in _pacing_thread_main, since FrbSearch::Stub is a
    // nested class that can't be forward-declared in C++ and we don't
    // want grpc++ headers in this .hpp. Constructed only when paced=true.
    std::unique_ptr<grpc::ClientContext> pacing_ctx;

    // Pacing thread. Joinable only when paced=true.
    std::thread pacing_thread;

    // ----- Public interface -----

    // Constructor: validates args, then spawns nworkers worker threads.
    // Each worker thread inherits the vcpu affinity of the caller. Each
    // element of 'ip_addrs' is "ip:port" format. nworkers must be a
    // multiple of ip_addrs.size().
    //
    // time_samples_per_chunk: receiver-side chunk size (in samples).
    // Must equal the FrbServer's
    // AssembledFrameAllocator::time_samples_per_chunk. Must be
    // positive and a multiple of 256.
    //
    // debug (default false): Adds real-time debugging checks with
    // nontrivial cpu/network cost, and unbounded memory usage. Useful
    // for unit tests, but don't use in production!
    //
    // Python callers MUST call the constructor inside a ThreadAffinity
    // context manager so the spawned worker threads are pinned to the
    // intended vcpus.
    // xmd: required, non-null. xmd->freq_channels is IGNORED --
    // make_worker_metadata() overwrites freq_channels per-worker, so the
    // caller can pass either a meaningful or frequency-scrubbed xmd (the
    // latter is the typical case, e.g. from XEngineMetadata::make_fiducial).
    //
    // paced (default true): if true, also spawns a pacing thread that
    // holds a MonitorRingbuf streaming RPC to the FrbServer at
    // rpc_address. Worker threads gate sends to stay <=5 chunks ahead
    // of server-side rb_processed. rpc_address must be non-empty when
    // paced=true; ignored (silently accepted) when paced=false.
    //
    // rpc_address: "ip:port" of the FrbServer's gRPC endpoint. Required
    // when paced=true; pass empty string when paced=false.
    FakeXEngine(const std::shared_ptr<const XEngineMetadata> &xmd,
                const std::vector<std::string> &ip_addrs,
                int nworkers, long time_samples_per_chunk,
                bool debug = false,
                bool paced = true,
                const std::string &rpc_address = "");

    // Destructor calls stop() and joins worker threads.
    ~FakeXEngine();

    // Entry point: submit a SEND_JUNK(minichunk_index) command to
    // workers[worker_id]->command_queue. Non-blocking. Throws if stopped
    // or worker_id is out of range. The pybind11 wrapper releases the GIL.
    //
    // Wire effect: the worker sends one minichunk worth of all-zero data.
    // (The protocol handshake is sent ahead of the first SEND_JUNK or
    // SEND_MINICHUNK on this worker; SKIP_MINICHUNK never triggers
    // connect/handshake.)
    void enqueue_send_junk(long worker_id, long minichunk_index);

    // Entry point: submit a SKIP_MINICHUNK(minichunk_index) command.
    // Non-blocking; same lock discipline as enqueue_send_junk.
    //
    // Wire effect: NONE. Advances last_processed_minichunk past
    // minichunk_index (per the usual +1 monotonicity rule, with the
    // first-command exception, enforced at queue time -- see
    // enqueue_send_junk). A worker whose only commands are SKIPs
    // never opens its TCP connection -- useful for "silent peer"
    // tests.
    void enqueue_skip_minichunk(long worker_id, long minichunk_index);

    // Entry point: submit a SEND_MINICHUNK(minichunk_index, frame_set)
    // command. Non-blocking; same lock discipline as enqueue_send_junk.
    // Throws if frame_set is null.
    //
    // Wire effect: gather the per-(beam, freq) int4 data for the
    // minichunk at offset (minichunk_index - frame_set->time_chunk_index
    // * minichunks_per_chunk) within the set, then send one minichunk.
    //
    // The caller is responsible for keeping the AssembledFrameSet (and
    // its frames' data buffers) alive throughout SEND_MINICHUNK
    // processing -- the worker holds a shared_ptr while the Command sits
    // in the queue + is being processed, so the typical pattern (hold
    // the same Python reference until wait_until_processed returns for
    // this minichunk_index) is sufficient.
    //
    // WARNING: the worker reads the frames' data buffers WITHOUT taking
    // frame.mutex, so it races the AssembledFrame reaper. The reaper
    // currently lives in FrbServer (not FakeXEngine), and the assumption
    // is that FakeXEngine and FrbServer never run in the same process,
    // so the race doesn't actually occur. A defensive
    // xassert_gt(frame.data.size, 0) inside the gather loop catches
    // calls against an already-reaped frame; it does NOT protect against
    // an actively-running reaper. If we ever want to colocate
    // FakeXEngine with a reaper, the gather loop needs lock acquisition
    // (one-per-beam) -- see plans/fake_xengine_skip_and_send_minichunk.md.
    void enqueue_send_minichunk(long worker_id, long minichunk_index,
                                std::shared_ptr<AssembledFrameSet> frame_set);

    // Entry point: submit a DISCONNECT command. Non-blocking
    // (fire-and-forget); the worker closes its TCP socket on receipt.
    // last_processed_minichunk and last_queued_minichunk are NOT
    // touched. The next SEND_JUNK or SEND_MINICHUNK on this worker
    // transparently reopens the connection AND re-sends the protocol
    // handshake.
    //
    // SKIP_MINICHUNK commands continue to work normally while
    // disconnected. If the caller wants the next reconnect to start at
    // a higher minichunk_index, it's the caller's responsibility to
    // bridge the gap with SKIP_MINICHUNK commands. (DISCONNECT itself
    // is socket-level only.)
    //
    // DISCONNECT on an already-disconnected (or never-connected) worker
    // is a no-op. The pybind11 wrapper releases the GIL.
    void enqueue_disconnect(long worker_id);

    // Entry point: returns true iff workers[worker_id] has an open TCP
    // connection to its receiver right now. The result is a snapshot --
    // by the time the caller observes it, the worker thread may have
    // already flipped the flag. is_connected() does NOT throw on a
    // stopped FakeXEngine (the last-known per-worker state is still
    // meaningful for diagnostics). Throws only on out-of-range
    // worker_id.
    //
    // O(1) atomic load. The pybind11 wrapper does NOT release the GIL
    // (the load is faster than the GIL ops that would protect it).
    bool is_connected(long worker_id) const;

    // Entry point: block until
    // workers[worker_id]->last_processed_minichunk >= minichunk_index,
    // or throw if stopped. Returns immediately for minichunk_index < 0
    // (since last_processed_minichunk starts at -1) -- this is what
    // makes the controller's "wait_until_processed(w, n-2)" call work
    // for n in {0, 1}. The pybind11 wrapper releases the GIL.
    void wait_until_processed(long worker_id, long minichunk_index);

    // Entry point: enqueue a WAIT_FOR_ACKS command on
    // workers[worker_id]'s queue. Non-blocking (fire-and-forget):
    // when the worker eventually pops the command, it calls
    // _read_acks(blocking=true) -- which drains all outstanding
    // acks (or throws after the 1-second per-call deadline).
    //
    // Throws if FakeXEngine was constructed with debug=false (a
    // programming error to ask for acks when there are none), or
    // if worker_id is out of range, or the FakeXEngine is stopped.
    // The pybind11 wrapper releases the GIL.
    void enqueue_wait_for_acks(long worker_id);

    // Entry point: block the CALLING thread until
    // workers[worker_id]->command_queue has been fully drained.
    // If debug is true, additionally enqueue a WAIT_FOR_ACKS (via
    // enqueue_wait_for_acks() above) BEFORE waiting -- so the wait
    // also covers all outstanding acks.
    //
    // SEMANTICS: this is a "drain everything in the queue"
    // barrier, NOT a "drain everything that was in the queue at
    // the moment I was called" barrier. If another thread is
    // concurrently enqueueing commands, synchronize() waits for
    // THEIR commands too -- the wait predicate is
    // `command_queue.empty()`, which is sensitive to any future
    // pushes. Document this in your test code.
    //
    // Throws on stopped FakeXEngine and on out-of-range worker_id.
    // The pybind11 wrapper releases the GIL.
    void synchronize(long worker_id);

    // Entry point: snapshot the per-minichunk status byte for
    // workers[worker_id]->minichunk_status[minichunk_index -
    // first_minichunk]. Returns one of the STATUS_X constants.
    //
    // Throws if debug is false, if worker_id is out of range,
    // if no state-advancing commands have been enqueued yet on
    // this worker, or if minichunk_index is outside the range
    // [first_minichunk, first_minichunk + minichunk_status.size()).
    //
    // Snapshot semantic: the returned byte may already be stale
    // by the time the caller observes it (the worker thread or
    // an ack may have advanced the state). Does NOT throw on a
    // stopped FakeXEngine. No GIL release (cheap atomic load
    // + mutex acquire + array read).
    unsigned char get_minichunk_status(long worker_id, long minichunk_index) const;

    // Entry point: snapshot the four debug_counters (see the member
    // declaration above for index meanings). Cheap atomic loads with
    // relaxed ordering. The snapshot is "consistent" only insofar as
    // nothing is actively being acked concurrently -- callers typically
    // call this after synchronize() has drained all in-flight acks.
    // Does NOT throw, even on a stopped FakeXEngine or when debug=false
    // (in which case all four entries are zero).
    std::array<long, 4> get_debug_counters() const;

    // Return the round-robin subset of total frequency channels assigned
    // to workers[worker_id]: { worker_id, worker_id + nworkers,
    // worker_id + 2*nworkers, ... } intersected with [0, total_nfreq).
    // This is the same content that _initialize() writes into
    // Worker::xmd.freq_channels; we compute it from FakeXEngine::xmd
    // and nworkers so the answer is well-defined whether or not the
    // worker thread has reached _initialize() yet. Throws on
    // out-of-range worker_id.
    std::vector<long> get_worker_freq_channels(long worker_id) const;

    // Put FakeXEngine into stopped state. First caller's compare-exchange
    // on is_stopped_cache wins; subsequent concurrent calls return
    // immediately. The winner sweeps every worker, locking each one's
    // mutex briefly to set is_stopped + error and notify its cv. Any
    // in-flight entry-point calls (wait_until_processed /
    // enqueue_send_junk) then throw on their next predicate re-check.
    // If 'e' is non-null, it represents an error; otherwise normal
    // termination.
    void stop(std::exception_ptr e = nullptr);

    // ----- Noncopyable, nonmoveable -----

    FakeXEngine(const FakeXEngine &) = delete;
    FakeXEngine &operator=(const FakeXEngine &) = delete;
    FakeXEngine(FakeXEngine &&) = delete;
    FakeXEngine &operator=(FakeXEngine &&) = delete;

private:
    // Helper: create XEngineMetadata for a specific worker (with subset of freq channels).
    XEngineMetadata make_worker_metadata(int worker_id) const;

    // Helper: check if the given Worker is stopped; throw if so. Caller
    // must hold w.mutex. Rethrows w.error if non-null; otherwise throws
    // a runtime_error including method_name.
    void _throw_if_stopped(Worker &w, const char *method_name);

    // Worker thread main function. Calls _initialize() once, then loops
    // popping commands off the worker's queue and dispatching them.
    void _worker_main(int worker_id);

    // Wrapper that catches exceptions and calls stop().
    void worker_main(int worker_id);

    // One-time setup at the top of the worker thread: builds workers[id]'s
    // xmd, send_buf (with the connection header stamped and the per-
    // minichunk magic stamped), header_nbytes, mc_nbytes, ip_addr, port.
    // Does NOT open the TCP connection -- that happens lazily on the
    // first SEND_JUNK or SEND_MINICHUNK inside _skip_or_send().
    void _initialize(int worker_id);

    // Combined handler for SEND_JUNK / SKIP_MINICHUNK / SEND_MINICHUNK.
    // Performs a defense-in-depth strict-+1 monotonicity check
    // against last_processed_minichunk (redundant with the queue-time
    // check in _enqueue), lazily connects on the first SEND_*, gathers
    // data into the minichunk_buf for SEND_MINICHUNK, stamps the
    // wire-seq, calls _send_all(), and finally publishes
    // last_processed_minichunk + notifies the worker's cv.
    //
    // Returns false if _send_all() bailed out (stop or peer connreset);
    // the caller (_worker_main) should treat that as a signal to exit
    // the worker loop. Returns true on success.
    bool _skip_or_send(Worker &w, const Command &cmd);

    // Handler for DISCONNECT. Idempotent: closes the worker's socket
    // and flips w.connected to false (no-op if already disconnected).
    // Does NOT touch w.last_processed_minichunk or
    // w.last_queued_minichunk. Does NOT call cv.notify_all (no state
    // that wait_until_processed waiters care about has changed).
    void _disconnect(Worker &w);

    // Helper called by _skip_or_send for SEND_MINICHUNK: gather one
    // minichunk's worth of int4 data from the set's per-beam frames
    // into the worker's send_buf. Outer loop over beams, inner loop
    // over freqs, with memcpy(dst, src, 128) per (beam, freq) so the
    // compiler can lower to SIMD load/store pairs.
    //
    // WARNING (reaper race): reads each frame's data without taking
    // frame.mutex. Safe today only because the AssembledFrame reaper
    // runs exclusively in FrbServer, never in the same process as
    // FakeXEngine. See the warning block on enqueue_send_minichunk().
    void _populate_minichunk_buf(Worker &w, const AssembledFrameSet &fset,
                                 long minichunk_index);

    // Helper for the enqueue_send_junk / enqueue_skip_minichunk /
    // enqueue_send_minichunk / enqueue_disconnect entry points.
    // Validates worker_id, takes workers[worker_id]->mutex,
    // throws-if-stopped, performs the queue-time sequentiality check
    // (only if is_state_advancing is true), updates
    // last_queued_minichunk and (on the very first call)
    // first_minichunk, pushes the Command, drops the lock, notifies
    // the worker's cv. method_name is passed through to
    // _throw_if_stopped and to the sequentiality-check error
    // message for diagnostics.
    //
    // Callers MUST pass is_state_advancing = true for SEND_JUNK,
    // SKIP_MINICHUNK, SEND_MINICHUNK; and false for DISCONNECT.
    // (We pass this as a separate argument rather than inspecting
    // cmd.kind, so it's explicit at every call site.)
    //
    // CRITICAL: the mutex is held continuously across the check,
    // the counter updates, and the queue push -- so the recorded
    // last_queued_minichunk is always consistent with the FIFO
    // ordering of command_queue.
    void _enqueue(long worker_id, Command &&cmd, bool is_state_advancing,
                  const char *method_name);

    // Helper: send all bytes from buffer, using short send_with_timeout
    // calls so we can periodically re-check w.is_stopped under w.mutex
    // and bail out promptly. Returns false on stop or peer connection
    // reset. On connection reset, also calls FakeXEngine::stop() so that
    // sibling workers exit too.
    bool _send_all(Worker &w, Socket &sock, const void *buf, long nbytes);

    // Drain ack bytes from the worker's socket and update
    // Worker::minichunk_status / Worker::ack_queue accordingly.
    // Runs on the worker thread. xassert(debug) at top -- never
    // call this method when debug is false.
    //
    // blocking=false: a single non-blocking recv attempt (one
    // syscall). Returns whatever's available (possibly zero bytes
    // -- not an error).
    //
    // blocking=true: loops with 10ms inner timeouts and a 1-second
    // outer deadline. Throws on (a) peer EOF before ack_queue is
    // drained, (b) invalid ack byte (not 0 or 1), (c) the deadline.
    // The throw propagates via worker_main's catch -> stop(e).
    void _read_acks(Worker &w, bool blocking);

    // Pacing thread (paced=true only). Holds a MonitorRingbuf
    // streaming RPC to the FrbServer; for each pushed rb_processed
    // value, sweeps workers and updates Worker::rb_processed under
    // each Worker::mutex.
    void _pacing_thread_main();
    void pacing_thread_main();   // try/catch wrapper, calls stop() on throw
};


}  // namespace pirate

#endif // _PIRATE_FAKE_XENGINE_HPP
