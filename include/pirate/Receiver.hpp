#ifndef _PIRATE_RECEIVER_HPP
#define _PIRATE_RECEIVER_HPP

#include <atomic>
#include <condition_variable>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "network_utils.hpp"  // Socket, Epoll


namespace pirate {
#if 0
}   // pacify editor auto-indent
#endif


// Receiver: a thread-backed class that listens for TCP connections
// and reads data as quickly as possible from all open connections.
//
// The Receiver has two worker threads:
//   - listener: calls accept() in a loop, hands off new connections to reader
//   - reader: uses epoll() to read data from all open connections
//
// Usage:
//   auto receiver = std::make_shared<Receiver> (ip_addr, port);
//   // ... wait for data to be received ...
//   long num_conn, num_bytes;
//   receiver->get_status(num_conn, num_bytes);
//   receiver->stop();

struct Receiver
{
    // Constructor args.
    const std::string ip_addr;
    const uint16_t tcp_port;

    // Thread-backed class state (protected by 'mutex').
    std::mutex mutex;
    std::condition_variable cv;
    bool is_stopped = false;
    std::exception_ptr error;

    // Worker threads.
    std::thread listener_thread;
    std::thread reader_thread;

    // Pending sockets: handed off from listener to reader.
    // Protected by 'mutex'.
    std::vector<std::shared_ptr<Socket>> pending_sockets;

    // Statistics (atomic for lock-free reads).
    std::atomic<long> num_connections{0};
    std::atomic<long> num_bytes{0};

    // Timeouts (milliseconds).
    static constexpr int accept_timeout_ms = 10;
    static constexpr int epoll_timeout_ms = 1;

    // Receive buffer size.
    static constexpr long recv_bufsize = 256 * 1024;

    // ----- Public interface -----

    // Constructor spawns both worker threads.
    Receiver(const std::string &ip_addr, uint16_t tcp_port);

    // Destructor calls stop() and joins worker threads.
    ~Receiver();

    // Thread-safe: returns current number of active TCP connections, and total bytes read.
    void get_status(long &num_connections, long &num_bytes);

    // Put Receiver into stopped state. Worker threads exit promptly.
    // If 'e' is non-null, it represents an error; otherwise normal termination.
    void stop(std::exception_ptr e = nullptr);

    // ----- Noncopyable, nonmoveable -----

    Receiver(const Receiver &) = delete;
    Receiver &operator=(const Receiver &) = delete;
    Receiver(Receiver &&) = delete;
    Receiver &operator=(Receiver &&) = delete;

private:
    // Worker thread main functions.
    void _listener_main();
    void _reader_main();

    // Wrappers that catch exceptions and call stop().
    void listener_main();
    void reader_main();
};


}  // namespace pirate

#endif // _PIRATE_RECEIVER_HPP
