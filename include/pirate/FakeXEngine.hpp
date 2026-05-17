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

#include "XEngineMetadata.hpp"


namespace pirate {
#if 0
}   // pacify editor auto-indent
#endif


struct Socket;  // forward declaration (defined in network_utils.hpp)


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
    // to a worker's queue. Room for future kinds (SEND_DATA, RECONNECT, ...).
    struct Command
    {
        enum class Kind : uint32_t {
            UNINITIALIZED = 0,
            SEND_JUNK     = 1,   // sends current contents of the worker's minichunk_buf
        } kind = Kind::UNINITIALIZED;

        // Used by SEND_JUNK: index in "minichunk units" (256 time samples).
        // Wire seq = minichunk_index * 256 * xmd.seq_per_frb_time_sample.
        // The worker asserts that successive SEND_JUNK commands have
        // exactly-monotonic minichunk_index (last_minichunk_sent + 1).
        long minichunk_index = -1;
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
    void send_junk(long worker_id, long minichunk_index);

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

    // Worker thread main function.
    void _worker_main(int thread_id);

    // Wrapper that catches exceptions and calls stop().
    void worker_main(int thread_id);

    // Helper: send all bytes from buffer, using short send_with_timeout
    // calls so we can periodically re-check w.is_stopped under w.mutex
    // and bail out promptly. Returns false on stop or peer connection
    // reset. On connection reset, also calls FakeXEngine::stop() so that
    // sibling workers exit too.
    bool _send_all(Worker &w, Socket &sock, const void *buf, long nbytes);
};


}  // namespace pirate

#endif // _PIRATE_FAKE_XENGINE_HPP
