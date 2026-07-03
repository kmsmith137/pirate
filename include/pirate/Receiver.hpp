#ifndef _PIRATE_RECEIVER_HPP
#define _PIRATE_RECEIVER_HPP

#include <array>
#include <atomic>
#include <condition_variable>
#include <exception>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>
#include "network_utils.hpp"    // Socket, Epoll
#include "XEngineMetadata.hpp"  // XEngineMetadata


namespace pirate {
#if 0
}   // pacify editor auto-indent
#endif

struct AssembledFrame;           // AssembledFrame.hpp
struct AssembledFrameSet;        // AssembledFrame.hpp
struct AssembledFrameAllocator;  // AssembledFrame.hpp

// Receiver: a thread-backed class that listens for TCP connections
// and reads data as quickly as possible from all open connections.
//
// The Receiver has three worker threads:
//   - listener: calls accept() in a loop, hands off new connections to reader
//   - reader: uses epoll() to read data from all open connections
//   - assembler: copies data minichunks from per-peer ring buffers into AssembledFrames
//
// NOTE: worker threads inherit the same vcpu affinity as the caller of start()
// (not the Receiver constructor!). Python callers should call Receiver.start()
// within a ThreadAffinity context manager.
//
// External-thread cross-coordination: evict(K) can be called from a non-
// Receiver thread (in practice, a FrbServer worker) to ask the assembler to
// force-advance its 2-chunk window past chunk K. Used by FrbServer to keep
// multiple Receivers synchronized -- see FrbServer::_worker_main.
//
// XEngineMetadata ownership: the Receiver does NOT own the consensus
// metadata. The reader thread hands each peer's parsed YAML to
// AssembledFrameAllocator::initialize_metadata(); other callers
// (assembler, FrbServer, RPC) read it back via
// AssembledFrameAllocator::get_metadata().
//
// See notes/network_protocol.md for the network protocol parsed by the Receiver.

struct Receiver
{
    static constexpr int accept_timeout_ms = 10;
    static constexpr int epoll_timeout_ms = 1;

    // ----- Public interface -----

    // Constructor args.
    struct Params {
        std::string address;               // "ip:port" format, e.g. "127.0.0.1:5000"
        std::shared_ptr<AssembledFrameAllocator> allocator;
        long consumer_id = -1;
        // If true, every peer socket accepted by the listener thread
        // has Socket::set_misbehaving_reads() called on it before
        // being handed to the reader thread. Test-only -- never set
        // in production.
        bool misbehaving_reads = false;
    };

    // Constructor initializes state but does not start worker threads.
    Receiver(const Params &params);

    // Destructor calls stop() and joins worker threads.
    ~Receiver();

    // Spawn worker threads. Throws exception if called twice or after stop().
    void start();

    // Thread-safe: returns current number of active TCP connections, and total bytes read.
    void get_status(long &num_connections, long &nbytes_cumul);

    // Entry point: wait until the listener thread has bound the listening
    // socket (i.e. a client's connect() will succeed). Returns true once it is
    // listening; throws if the Receiver is stopped first. If timeout_sec >= 0,
    // give up after that many seconds and return false (still listening == false);
    // a negative timeout (the default) waits forever. Useful when the caller must
    // not attempt a connection before the Receiver is accepting (e.g. FakeXEngine,
    // whose lazy connect throws on ECONNREFUSED). A finite timeout lets a Python
    // caller poll so it stays responsive to signals / can detect a dead peer.
    // Requires start() to have been called (otherwise waits until stop()/timeout).
    bool wait_until_listening(double timeout_sec = -1.0);

    // Entry point: retrieve an assembled frame set (= one time chunk, all
    // beams) from the queue (blocking). Blocks until a set is available,
    // or throws if the Receiver is stopped. The returned set corresponds
    // to one AssembledFrameAllocator::get_frame_set() call on the
    // assembler side.
    std::shared_ptr<AssembledFrameSet> get_frame_set();

    // Put Receiver into stopped state. Worker threads exit promptly.
    // If 'e' is non-null, it represents an error; otherwise normal termination.
    void stop(std::exception_ptr e = nullptr);

    // Entry point: schedule the assembler thread to evict all chunks with
    // chunk_index <= evicted_chunk (i.e., advance curr_base_chunk past
    // evicted_chunk).
    //
    // Non-blocking -- returns immediately after setting state and notifying.
    // Thread-safe; intended to be called from non-Receiver threads (in
    // practice, FrbServer worker threads). Idempotent / monotone: only
    // ratchets the internal target upward. No-op (silently) if the Receiver
    // is already stopped.
    void evict(long evicted_chunk);

    // ----- Noncopyable, nonmoveable -----

    Receiver(const Receiver &) = delete;
    Receiver &operator=(const Receiver &) = delete;
    Receiver(Receiver &&) = delete;
    Receiver &operator=(Receiver &&) = delete;

    // Constructor args.
    Params params;
    std::string ip_addr;    // parsed from params.address
    uint16_t tcp_port = 0;  // parsed from params.address

    // Worker threads.
    std::thread listener_thread;
    std::thread reader_thread;
    std::thread assembler_thread;

    // Lock-free counters.
    std::atomic<long> num_connections{0};
    std::atomic<long> nbytes_cumul{0};

    // Thread-backed class state (protected by 'mutex').
    mutable std::mutex mutex;
    mutable std::condition_variable cv;
    bool is_started = false;
    bool is_listening = false;   // set by listener thread once the listening socket is bound
    bool is_stopped = false;
    std::exception_ptr error;

    // All public members after this point are protected by 'mutex'.

    // Queue of completed frame sets ready for retrieval via get_frame_set().
    std::queue<std::shared_ptr<AssembledFrameSet>> completed_frame_sets;

    // Peer: per-sender state for an active TCP connection.
    // Parses the network protocol described in notes/network_protocol.md.
    // (Full definition is in Receiver.cpp.)
    struct Peer;

    std::vector<std::shared_ptr<Peer>> reader_peer_queue;     // handoff listener -> reader
    std::deque<std::shared_ptr<Peer>> assembler_peer_queue;   // handoff reader -> assembler

    // Eviction coordination: target chunk index for external-thread-
    // initiated eviction. Written by evict() (from external threads),
    // read by the assembler thread to decide how far to advance
    // curr_base_chunk. Sentinel value -1 = "no eviction has been
    // requested". Protected by 'mutex'.
    long evicted_chunk = -1;

private:
    // Worker thread main functions.
    void _listener_main();
    void _reader_main();
    void _assembler_main();

    // Wrappers that catch exceptions and call stop().
    void listener_main();
    void reader_main();
    void assembler_main();

    // Helpers called by reader thread.
    void _read_ini(const std::shared_ptr<Peer> &peer);
    void _read_yaml(const std::shared_ptr<Peer> &peer);
    void _read_data(const std::shared_ptr<Peer> &peer);

    // Helpers called by assembler thread.
    void _process_data(const std::shared_ptr<Peer> &peer);

    // Advance curr_base_chunk by 1: evict the AssembledFrameSet in the
    // "old" slot to completed_frame_sets, replace it with a fresh set
    // pulled from the allocator. Caller (assembler thread) must NOT hold
    // Receiver::mutex.
    //
    // CRITICAL INVARIANT: after std::move(curr_frame_sets[slot]) into
    // completed_frame_sets, the assembler thread must not perform any
    // further write to any frame inside the just-evicted set. See
    // Receiver.cpp for the full discussion.
    void _advance_one_chunk();

    // Send a 1-byte FLAG_ACK reply (0 = dropped, 1 = assembled) on the
    // peer's TCP socket. Used only when the peer enabled FLAG_ACK in the
    // handshake; called from the assembler thread per minichunk in
    // _process_data. Loops with a 10ms inner timeout (for prompt
    // is_stopped checks) and throws if the client doesn't drain the
    // byte within 1 second.
    void _send_ack(const std::shared_ptr<Peer> &peer, char ack_byte);

    // Helper for entry points. Caller must hold mutex.
    void _throw_if_stopped(const char *method_name) const;

    // curr_frame_sets, curr_base_chunk: not lock-protected -- only
    // modified by the assembler thread. The assembler does read
    // curr_base_chunk while holding 'mutex' (for its wait predicate);
    // this is safe because the assembler itself is the only writer.
    //
    // The Receiver incrementally receives a data "cube" indexed by (beam, freq, time).
    // Time indices are divided into "chunks" (of size Params::time_samples_per_chunk),
    // and further subdivided into 256-sample "minichunks".
    //
    // At any time, the Receiver holds a two-chunk subset of the data cube.
    // When the first bit of data is received for chunk N, it evicts chunk (N-2).
    // (As a second trigger, the assembler also evicts chunks in response to
    //  external evict() calls -- see the 'evicted_chunk' member above.)
    // Each of the two live chunks is represented by one AssembledFrameSet
    // (one frame per beam), so curr_frame_sets has length 2:
    // curr_frame_sets[k] is the live set for ichunk = curr_base_chunk + k,
    // where k in {0,1}. Rotation in _advance_one_chunk evicts slot 0, shifts
    // slot 1 -> slot 0, and pulls a fresh allocator set into slot 1.

    std::array<std::shared_ptr<AssembledFrameSet>, 2> curr_frame_sets;
    long curr_base_chunk = 0;

};


}  // namespace pirate

#endif // _PIRATE_RECEIVER_HPP
