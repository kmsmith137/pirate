#ifndef _PIRATE_RECEIVER_HPP
#define _PIRATE_RECEIVER_HPP

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
struct AssembledFrameAllocator;  // AssembledFrame.hpp

// Receiver: a thread-backed class that listens for TCP connections
// and reads data as quickly as possible from all open connections.
//
// The Receiver has two worker threads:
//   - listener: calls accept() in a loop, hands off new connections to reader
//   - reader: uses epoll() to read data from all open connections
//
// See notes/network_protocol.md for the network protocol parsed by the Receiver.

struct Receiver
{
    static constexpr long initial_recv_bufsize = 256 * 1024;
    static constexpr int accept_timeout_ms = 10;
    static constexpr int epoll_timeout_ms = 1;

    // ----- Public interface -----

    // Constructor args.
    struct Params {
        std::string address;               // "ip:port" format, e.g. "127.0.0.1:5000"
        long time_samples_per_chunk = 0;   // must be a multiple of 256
        std::shared_ptr<AssembledFrameAllocator> allocator;
        long consumer_id = -1;
    };

    // Constructor initializes state but does not start worker threads.
    Receiver(const Params &params);

    // Destructor calls stop() and joins worker threads.
    ~Receiver();

    // Spawn worker threads. Throws exception if called twice or after stop().
    void start();

    // Thread-safe: returns current number of active TCP connections, and total bytes read.
    void get_status(long &num_connections, long &num_bytes);

    // Entry point: retrieve an assembled frame from the queue (blocking).
    // This blocks until a frame is available, or throws if Receiver is stopped.
    std::shared_ptr<AssembledFrame> get_frame();

    // Put Receiver into stopped state. Worker threads exit promptly.
    // If 'e' is non-null, it represents an error; otherwise normal termination.
    void stop(std::exception_ptr e = nullptr);

    // If blocking=false and metadata has not been initialized, return a default-constructed XEngineMetadata.
    XEngineMetadata get_metadata(bool blocking) const;

    // ----- Noncopyable, nonmoveable -----

    Receiver(const Receiver &) = delete;
    Receiver &operator=(const Receiver &) = delete;
    Receiver(Receiver &&) = delete;
    Receiver &operator=(Receiver &&) = delete;

    // Constructor args.
    Params params;
    std::string ip_addr;    // parsed from params.address
    uint16_t tcp_port = 0;  // parsed from params.address

    // Thread-backed class state (protected by 'mutex').
    mutable std::mutex mutex;
    mutable std::condition_variable cv;
    bool is_started = false;
    bool is_stopped = false;
    std::exception_ptr error;

    // Reference metadata from first peer (protected by 'mutex').
    // Used to check that all peers send consistent metadata.
    // (Checks zone_nfreq, zone_freq_edges, nbeams, beam_ids. Does NOT check freq_channels.)

    bool has_metadata = false;
    XEngineMetadata metadata;

    // Queue of completed frames ready for retrieval via get_frame().
    // Protected by 'mutex'. Frames are pushed from _find_frame() (reader thread)
    // and popped by get_frame() (caller thread).
    std::queue<std::shared_ptr<AssembledFrame>> completed_frames;

    // The Receiver incrementally receives a data "cube" indexed by (beam, freq, time).
    // Time indices are divided into "chunks" (of size Params::time_samples_per_chunk),
    // and further subdivided into 256-sample "segments".
    //
    // At any time, the Receiver holds a two-chunk subset of the data cube.
    // When the first bit of data is received for chunk N, it evicts chunk (N-2).
    // This is represented by a ring buffer 'curr_frames' of length (2 * nbeams).
    // (Recall that an AssembledFrame represetns one time chunk and one beam).
    //
    // 'curr_frames' is a ring buffer of length (2 * nbeams).
    // It contains data for (curr_base_chunk) <= ichunk < (curr_base_chunk + 2).
    // Accessed exclusively by reader thread -- not protected by lock.

    std::vector<std::shared_ptr<AssembledFrame>> curr_frames;  // length (2 * metadata.nbeams)
    long curr_base_chunk = 0;

    // Worker threads.
    std::thread listener_thread;
    std::thread reader_thread;

    // Peer: per-sender state for an active TCP connection.
    // Parses the network protocol described in notes/network_protocol.md.
    // (Full definition is in Receiver.cpp.)
    struct Peer;

    // Pending peers: handed off from listener to reader.
    // Protected by 'mutex'.
    std::vector<std::shared_ptr<Peer>> pending_peers;

    // Statistics (atomic for lock-free reads).
    std::atomic<long> num_connections{0};
    std::atomic<long> num_bytes{0};


private:
    // Worker thread main functions.
    void _listener_main();
    void _reader_main();

    // Wrappers that catch exceptions and call stop().
    void listener_main();
    void reader_main();

    // Helpers for reading data.
    void _read_data(Peer *peer);
    void _post_metadata(Peer *peer);
    void _process_4bit_data(Peer *peer, const char *buf, long nsegments);
    char *_find_frame(long ichunk, long ibeam);

    // Helper for entry points. Caller must hold mutex.
    void _throw_if_stopped(const char *method_name) const;
};


}  // namespace pirate

#endif // _PIRATE_RECEIVER_HPP
