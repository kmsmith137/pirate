#include "../include/pirate/Receiver.hpp"

#include <algorithm>
#include <functional>
#include <sstream>
#include <stdexcept>

#include <ksgpu/xassert.hpp>

using namespace std;


namespace pirate {
#if 0
}  // editor auto-indent
#endif


Receiver::Receiver(const std::string &ip_addr_, uint16_t tcp_port_) :
    ip_addr(ip_addr_),
    tcp_port(tcp_port_),
    epoll(false)  // uninitialized; will be initialized in reader thread
{
    xassert(ip_addr.size() > 0);
    xassert(tcp_port > 0);

    // Spawn both worker threads.
    listener_thread = std::thread(&Receiver::listener_main, this);
    reader_thread = std::thread(&Receiver::reader_main, this);
}


Receiver::~Receiver()
{
    this->stop();

    if (listener_thread.joinable())
        listener_thread.join();
    if (reader_thread.joinable())
        reader_thread.join();
}


void Receiver::get_status(long &out_num_connections, long &out_num_bytes)
{
    out_num_connections = num_connections.load();
    out_num_bytes = num_bytes.load();
}


void Receiver::stop(std::exception_ptr e)
{
    std::lock_guard<std::mutex> lock(mutex);

    if (is_stopped)
        return;

    is_stopped = true;
    error = e;
    cv.notify_all();
}


// -------------------------------------------------------------------------------------------------
//
// Listener thread


void Receiver::listener_main()
{
    try {
        _listener_main();
    } catch (...) {
        stop(std::current_exception());
    }
}


void Receiver::_listener_main()
{
    // Create and configure listening socket.
    Socket listening_socket(PF_INET, SOCK_STREAM);
    listening_socket.set_reuseaddr();
    listening_socket.bind(ip_addr, tcp_port);
    listening_socket.listen();

    while (true) {
        // Check if stopped.
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (is_stopped)
                return;
        }

        // Accept with timeout, so we can check is_stopped frequently.
        Socket new_socket = listening_socket.accept(accept_timeout_ms);

        // Timeout expired, loop again.
        if (new_socket.fd < 0)
            continue;

        // Create shared_ptr and hand off to reader thread.
        auto sp = std::make_shared<Socket>(std::move(new_socket));

        {
            std::lock_guard<std::mutex> lock(mutex);
            pending_sockets.push_back(sp);
            cv.notify_all();
        }
    }
}


// -------------------------------------------------------------------------------------------------
//
// Reader thread


void Receiver::reader_main()
{
    try {
        _reader_main();
    } catch (...) {
        stop(std::current_exception());
    }
}


void Receiver::_reader_main()
{
    // Initialize epoll and receive buffer.
    epoll.initialize();
    recv_buf.resize(recv_bufsize);

    while (true) {
        // Check if stopped, and receive pending sockets from listener thread.
        {
            std::lock_guard<std::mutex> lock(mutex);

            if (is_stopped)
                return;

            // Receive pending sockets from listener.
            for (auto &sp : pending_sockets) {
                sp->set_nonblocking();

                epoll_event ev;
                ev.events = EPOLLIN | EPOLLRDHUP | EPOLLHUP;
                ev.data.u64 = (uint64_t) active_sockets.size();  // index into active_sockets
                epoll.add_fd(sp->fd, ev);

                active_sockets.push_back(sp);
                num_connections.fetch_add(1);
            }
            pending_sockets.clear();
        }

        // If no active sockets, sleep briefly and loop again.
        if (active_sockets.empty()) {
            // Sleep 1ms to avoid busy-waiting.
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        // Wait for events with timeout.
        int num_events = epoll.wait(epoll_timeout_ms);

        // Process events.
        // Note: we must be careful when removing sockets, since indices in epoll events
        // reference positions in active_sockets. We mark sockets to remove, then remove them.

        vector<long> sockets_to_remove;

        for (int i = 0; i < num_events; i++) {
            long idx = (long) epoll.events[i].data.u64;
            uint32_t ev_flags = epoll.events[i].events;

            xassert((idx >= 0) && (idx < (long)active_sockets.size()));
            auto &sp = active_sockets[idx];

            // Connection closed by peer.
            if ((ev_flags & EPOLLRDHUP) || (ev_flags & EPOLLHUP) || (ev_flags & EPOLLERR)) {
                sockets_to_remove.push_back(idx);
                continue;
            }

            // Data available to read.
            if (ev_flags & EPOLLIN) {
                // Nonblocking read.
                long nbytes = sp->read(recv_buf.data(), recv_bufsize);

                if (nbytes > 0) {
                    // Data received (and thrown away).
                    num_bytes.fetch_add(nbytes);
                }
                else if (nbytes == 0) {
                    // Connection closed (read returned 0 on nonblocking socket means EOF).
                    sockets_to_remove.push_back(idx);
                }
            }
        }

        // Remove closed sockets.
        // We must remove in reverse order to avoid invalidating indices.
        // Also, we must rebuild epoll since indices will change.

        if (!sockets_to_remove.empty()) {
            // Sort in reverse order.
            std::sort(sockets_to_remove.begin(), sockets_to_remove.end(), std::greater<long>());

            // Remove duplicates.
            auto last = std::unique(sockets_to_remove.begin(), sockets_to_remove.end());
            sockets_to_remove.erase(last, sockets_to_remove.end());

            for (long idx : sockets_to_remove) {
                // Remove from epoll.
                epoll.delete_fd(active_sockets[idx]->fd);

                // Remove from active_sockets (swap with last, then pop).
                long last_idx = (long)active_sockets.size() - 1;

                if (idx != last_idx) {
                    // Update epoll for the swapped socket.
                    // We need to modify the existing fd in epoll to update its data.u64.
                    auto &swapped = active_sockets.back();

                    epoll_event ev;
                    ev.events = EPOLLIN | EPOLLRDHUP | EPOLLHUP;
                    ev.data.u64 = (uint64_t) idx;

                    int err = epoll_ctl(epoll.epfd, EPOLL_CTL_MOD, swapped->fd, &ev);
                    if (err < 0) {
                        stringstream ss;
                        ss << "epoll_ctl(EPOLL_CTL_MOD) failed: " << strerror(errno);
                        throw runtime_error(ss.str());
                    }

                    std::swap(active_sockets[idx], active_sockets.back());
                }

                active_sockets.pop_back();
                num_connections.fetch_sub(1);
            }
        }
    }
}


}  // namespace pirate
