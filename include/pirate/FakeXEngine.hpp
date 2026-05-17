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
// This class is a "thread-backed class" (see notes/thread_backed_class.md)
// that spawns 'nthreads' worker threads in its constructor. Each worker:
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

    // Worker: per-worker state. command_queue and last_minichunk_sent are
    // protected by FakeXEngine::mutex; worker_thread is constant after
    // construction.
    struct Worker
    {
        // Protected by FakeXEngine::mutex.
        std::deque<Command> command_queue;

        // Latest minichunk_index this worker has finished sending, or -1
        // if no SEND_JUNK has completed yet on this worker. Only the
        // worker thread writes this; external threads read it (under
        // FakeXEngine::mutex) via wait_for_send().
        long last_minichunk_sent = -1;

        // Constant after construction, not lock-protected.
        std::thread worker_thread;

        // Move-only (std::thread is move-only). std::vector<Worker>::resize(n)
        // default-constructs Worker; the thread is assigned by move in the
        // FakeXEngine constructor body.
        Worker() = default;
        Worker(const Worker &) = delete;
        Worker &operator=(const Worker &) = delete;
        Worker(Worker &&) = default;
        Worker &operator=(Worker &&) = default;
    };

    // ----- Constructor args -----

    const XEngineMetadata xmd;
    const std::vector<std::string> ip_addrs;  // each element is "ip:port"
    const int nthreads;

    // ----- Thread-backed class state (protected by 'mutex') -----

    std::mutex mutex;
    std::condition_variable cv;   // notified on: enqueue, last_minichunk_sent update, stop
    bool is_stopped = false;
    std::exception_ptr error;

    // ----- Worker state -----

    // Length nthreads. Resized once in the constructor; each worker_thread
    // is then move-assigned in.
    std::vector<Worker> workers;

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
    // workers[worker_id].command_queue. Non-blocking. Throws if stopped
    // or worker_id is out of range. The pybind11 wrapper releases the GIL.
    void send_junk(long worker_id, long minichunk_index);

    // Entry point: block until workers[worker_id].last_minichunk_sent >=
    // minichunk_index, or throw if stopped. Returns immediately for
    // minichunk_index < 0 (since last_minichunk_sent starts at -1) -- this
    // is what makes the controller's "wait_for_send(w, n-2)" call work for
    // n in {0, 1}. The pybind11 wrapper releases the GIL.
    void wait_for_send(long worker_id, long minichunk_index);

    // Put FakeXEngine into stopped state. Worker threads exit promptly.
    // Any in-flight entry-point calls (wait_for_send / send_junk) throw.
    // If 'e' is non-null, it represents an error; otherwise normal termination.
    void stop(std::exception_ptr e = nullptr);

    // ----- Noncopyable, nonmoveable -----

    FakeXEngine(const FakeXEngine &) = delete;
    FakeXEngine &operator=(const FakeXEngine &) = delete;
    FakeXEngine(FakeXEngine &&) = delete;
    FakeXEngine &operator=(FakeXEngine &&) = delete;

private:
    // Helper: create XEngineMetadata for a specific worker thread (with subset of freq channels).
    XEngineMetadata make_worker_metadata(int thread_id) const;

    // Helper: check if stopped, with lock held by caller.
    void _throw_if_stopped(const char *method_name);

    // Worker thread main function.
    void _worker_main(int thread_id);

    // Wrapper that catches exceptions and calls stop().
    void worker_main(int thread_id);

    // Helper: send all bytes from buffer, using short timeouts to allow prompt exit.
    // Returns false if stopped or connection reset.
    bool _send_all(Socket &sock, const void *buf, long nbytes);
};


}  // namespace pirate

#endif // _PIRATE_FAKE_XENGINE_HPP
