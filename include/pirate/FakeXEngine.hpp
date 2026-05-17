#ifndef _PIRATE_FAKE_XENGINE_HPP
#define _PIRATE_FAKE_XENGINE_HPP

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
//     metadata). FakeXEngine always sends flags=0; the FLAG_ACK back-
//     channel is NOT supported here (a real test client that wants
//     FLAG_ACK lives outside FakeXEngine).
//   - On every SEND_JUNK, sends one minichunk: a 12-byte header (uint32
//     magic + uint64 seq) followed by a shape-(nbeams, nfreq, 256) int4
//     data array (all zeros -- "junk"). The wire seq is derived from
//     Command::minichunk_index.
//
// Threads are assigned round-robin to IP addresses (nthreads must be a
// multiple of ip_addrs.size()). Frequency channels are assigned round-
// robin to worker threads. There is NO internal cross-worker barrier --
// the controller thread is responsible for any "minichunk N waits for
// (N-2)" style serialization by interleaving wait_for_send() and
// send_junk() calls. See plans/fake_xengine_command_queue.md for the
// canonical controller pseudocode.
//
// Worker threads inherit the vcpu affinity of the thread that calls the
// FakeXEngine constructor. Python callers MUST call the constructor
// inside a ThreadAffinity context manager.
//
// Usage:
//   with ThreadAffinity(vcpu_list):
//       fxe = FakeXEngine(xmd, ["10.0.0.2:5000", "10.0.1.2:5000"], 64)
//       # Spawn a controller thread (under the same affinity) that calls
//       # fxe.send_junk / fxe.wait_for_send in a loop.
//   ...
//   fxe.stop()   # signals workers and any in-flight entry points to exit.

struct FakeXEngine
{
    // Protocol magic number (little-endian): 0xf4bf4b02 where 02 is the version number.
    // Used both for the initial handshake AND for the header of every minichunk.
    static constexpr uint32_t protocol_magic = 0xf4bf4b02;
    // Timeout for send operations (milliseconds).
    static constexpr int send_timeout_ms = 10;

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
        } kind = Kind::UNINITIALIZED;

        // Used by all of {SEND_JUNK, SKIP_MINICHUNK, SEND_MINICHUNK}.
        // Ignored by DISCONNECT (which does not touch last_minichunk_sent).
        // Wire seq = minichunk_index * 256 * xmd.seq_per_frb_time_sample.
        // Successive state-advancing commands on a worker must have
        // strictly +1 monotonic minichunk_index, with one exception:
        // the very FIRST such command may pick any minichunk_index >= 0
        // -- this is what makes NOTE-2-style nonzero-initial-chunk tests
        // work.
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

    // Worker: per-worker state and synchronization. Each Worker is an
    // independent "thread-backed" unit -- mutex, cv, and is_stopped/error
    // are all per-worker, so there is no cross-worker contention on the
    // hot path.
    struct Worker
    {
        // ---- All of these are protected by 'mutex'. ----

        std::mutex mutex;
        // Notified by: send_junk (after enqueue), the worker thread (after
        // a successful send updates last_minichunk_sent), and stop() (when
        // is_stopped transitions to true).
        std::condition_variable cv;

        // Commands waiting to be processed by this worker, FIFO.
        std::deque<Command> command_queue;

        // Latest minichunk_index this worker has finished sending, or -1
        // if no SEND_JUNK has completed yet. Only the worker thread
        // writes this; external threads read it (under 'mutex') via
        // wait_for_send().
        long last_minichunk_sent = -1;

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

        // ---- Constant after construction, not lock-protected. ----

        std::thread worker_thread;

        // ---- Worker-thread-only state. Initialized in _initialize()
        // (called at the top of _worker_main) and modified only by
        // _send_junk(). No other thread reads or writes these, so they
        // need no synchronization. ----

        // Per-worker XEngineMetadata, with the round-robin subset of
        // freq_channels assigned to this thread.
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

        // Parsed destination address (round-robin: thread_id % ip_addrs.size()).
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

    const XEngineMetadata xmd;
    const std::vector<std::string> ip_addrs;  // each element is "ip:port"
    const int nthreads;

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

    // Length nthreads, but exposed as a vector of unique_ptr<Worker> so
    // that the Worker objects (which embed non-movable std::mutex and
    // std::condition_variable) are heap-allocated and stable in memory.
    // This avoids any need to make Worker movable.
    std::vector<std::unique_ptr<Worker>> workers;

    // ----- Public interface -----

    // Constructor: validates args, then spawns nthreads worker threads.
    // Each worker thread inherits the vcpu affinity of the caller. Each
    // element of 'ip_addrs' is "ip:port" format. nthreads must be a
    // multiple of ip_addrs.size().
    //
    // Python callers MUST call the constructor inside a ThreadAffinity
    // context manager so the spawned worker threads are pinned to the
    // intended vcpus.
    FakeXEngine(const XEngineMetadata &xmd, const std::vector<std::string> &ip_addrs, int nthreads);

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
    void send_junk(long worker_id, long minichunk_index);

    // Entry point: submit a SKIP_MINICHUNK(minichunk_index) command.
    // Non-blocking; same lock discipline as send_junk.
    //
    // Wire effect: NONE. Advances last_minichunk_sent past minichunk_index
    // (per the usual +1 monotonicity rule, with the first-command
    // exception). A worker whose only commands are SKIPs never opens its
    // TCP connection -- useful for "silent peer" tests.
    void skip_minichunk(long worker_id, long minichunk_index);

    // Entry point: submit a SEND_MINICHUNK(minichunk_index, frame_set)
    // command. Non-blocking; same lock discipline as send_junk. Throws
    // if frame_set is null.
    //
    // Wire effect: gather the per-(beam, freq) int4 data for the
    // minichunk at offset (minichunk_index - frame_set->time_chunk_index
    // * minichunks_per_chunk) within the set, then send one minichunk.
    //
    // The caller is responsible for keeping the AssembledFrameSet (and
    // its frames' data buffers) alive throughout SEND_MINICHUNK
    // processing -- the worker holds a shared_ptr while the Command sits
    // in the queue + is being processed, so the typical pattern (hold
    // the same Python reference until wait_for_send returns for this
    // minichunk_index) is sufficient.
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
    void send_minichunk(long worker_id, long minichunk_index,
                        std::shared_ptr<AssembledFrameSet> frame_set);

    // Entry point: submit a DISCONNECT command. Non-blocking
    // (fire-and-forget); the worker closes its TCP socket on receipt.
    // last_minichunk_sent is NOT touched. The next SEND_JUNK or
    // SEND_MINICHUNK on this worker transparently reopens the connection
    // AND re-sends the protocol handshake.
    //
    // SKIP_MINICHUNK commands continue to work normally while
    // disconnected. If the caller wants the next reconnect to start at
    // a higher minichunk_index, it's the caller's responsibility to
    // bridge the gap with SKIP_MINICHUNK commands. (DISCONNECT itself
    // is socket-level only.)
    //
    // DISCONNECT on an already-disconnected (or never-connected) worker
    // is a no-op. The pybind11 wrapper releases the GIL.
    void disconnect(long worker_id);

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

    // Entry point: block until workers[worker_id]->last_minichunk_sent >=
    // minichunk_index, or throw if stopped. Returns immediately for
    // minichunk_index < 0 (since last_minichunk_sent starts at -1) -- this
    // is what makes the controller's "wait_for_send(w, n-2)" call work for
    // n in {0, 1}. The pybind11 wrapper releases the GIL.
    void wait_for_send(long worker_id, long minichunk_index);

    // Put FakeXEngine into stopped state. First caller's compare-exchange
    // on is_stopped_cache wins; subsequent concurrent calls return
    // immediately. The winner sweeps every worker, locking each one's
    // mutex briefly to set is_stopped + error and notify its cv. Any
    // in-flight entry-point calls (wait_for_send / send_junk) then throw
    // on their next predicate re-check. If 'e' is non-null, it represents
    // an error; otherwise normal termination.
    void stop(std::exception_ptr e = nullptr);

    // ----- Noncopyable, nonmoveable -----

    FakeXEngine(const FakeXEngine &) = delete;
    FakeXEngine &operator=(const FakeXEngine &) = delete;
    FakeXEngine(FakeXEngine &&) = delete;
    FakeXEngine &operator=(FakeXEngine &&) = delete;

private:
    // Helper: create XEngineMetadata for a specific worker thread (with subset of freq channels).
    XEngineMetadata make_worker_metadata(int thread_id) const;

    // Helper: check if the given Worker is stopped; throw if so. Caller
    // must hold w.mutex. Rethrows w.error if non-null; otherwise throws
    // a runtime_error including method_name.
    void _throw_if_stopped(Worker &w, const char *method_name);

    // Worker thread main function. Calls _initialize() once, then loops
    // popping commands off the worker's queue and dispatching them.
    void _worker_main(int thread_id);

    // Wrapper that catches exceptions and calls stop().
    void worker_main(int thread_id);

    // One-time setup at the top of the worker thread: builds workers[id]'s
    // xmd, send_buf (with the connection header stamped and the per-
    // minichunk magic stamped), header_nbytes, mc_nbytes, ip_addr, port.
    // Does NOT open the TCP connection -- that happens lazily on the
    // first SEND_JUNK or SEND_MINICHUNK inside _skip_or_send().
    void _initialize(int thread_id);

    // Combined handler for SEND_JUNK / SKIP_MINICHUNK / SEND_MINICHUNK.
    // Performs the strict-+1 monotonicity check (with the first-command
    // exception), lazily connects on the first SEND_*, gathers data
    // into the minichunk_buf for SEND_MINICHUNK, stamps the wire-seq,
    // calls _send_all(), and finally publishes last_minichunk_sent +
    // notifies the worker's cv.
    //
    // Returns false if _send_all() bailed out (stop or peer connreset);
    // the caller (_worker_main) should treat that as a signal to exit
    // the worker loop. Returns true on success.
    bool _skip_or_send(int thread_id, const Command &cmd);

    // Handler for DISCONNECT. Idempotent: closes the worker's socket
    // and flips w.connected to false (no-op if already disconnected).
    // Does NOT touch w.last_minichunk_sent. Does NOT call cv.notify_all
    // (no state that wait_for_send waiters care about has changed).
    void _disconnect(int thread_id);

    // Helper called by _skip_or_send for SEND_MINICHUNK: gather one
    // minichunk's worth of int4 data from the set's per-beam frames
    // into the worker's send_buf. Outer loop over beams, inner loop
    // over freqs, with memcpy(dst, src, 128) per (beam, freq) so the
    // compiler can lower to SIMD load/store pairs.
    //
    // WARNING (reaper race): reads each frame's data without taking
    // frame.mutex. Safe today only because the AssembledFrame reaper
    // runs exclusively in FrbServer, never in the same process as
    // FakeXEngine. See the warning block on send_minichunk().
    void _populate_minichunk_buf(Worker &w, const AssembledFrameSet &fset,
                                 long minichunk_index);

    // Helper for the send_junk / skip_minichunk / send_minichunk entry
    // points. Validates worker_id, takes workers[worker_id]->mutex,
    // throws-if-stopped, pushes the Command, drops the lock, notifies
    // the worker's cv. method_name is passed through to
    // _throw_if_stopped for diagnostics.
    void _enqueue(long worker_id, Command &&cmd, const char *method_name);

    // Helper: send all bytes from buffer, using short send_with_timeout
    // calls so we can periodically re-check w.is_stopped under w.mutex
    // and bail out promptly. Returns false on stop or peer connection
    // reset. On connection reset, also calls FakeXEngine::stop() so that
    // sibling workers exit too.
    bool _send_all(Worker &w, Socket &sock, const void *buf, long nbytes);
};


}  // namespace pirate

#endif // _PIRATE_FAKE_XENGINE_HPP
