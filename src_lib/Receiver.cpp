#include "../include/pirate/Receiver.hpp"

#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include <ksgpu/xassert.hpp>

using namespace std;


namespace pirate {
#if 0
}  // editor auto-indent
#endif


Receiver::Receiver(const std::string &ip_addr_, uint16_t tcp_port_) :
    ip_addr(ip_addr_),
    tcp_port(tcp_port_)
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
    // Local state for reader thread.
    Epoll epoll;
    vector<char> recv_buf(recv_bufsize);
    unordered_map<Socket *, shared_ptr<Socket>> active_sockets;

    while (true) {
        // Check if stopped, and receive pending sockets from listener thread.
        {
            std::lock_guard<std::mutex> lock(mutex);

            if (is_stopped)
                return;

            // Receive pending sockets from listener.
            for (auto &sp : pending_sockets) {
                Socket *sock = sp.get();
                xassert(sock != nullptr);
                xassert(sock->fd >= 0);
                xassert(active_sockets.find(sock) == active_sockets.end());

                sock->set_nonblocking();

                epoll_event ev;
                ev.events = EPOLLIN | EPOLLRDHUP | EPOLLHUP;
                ev.data.ptr = sock;  // store Socket* pointer in epoll context
                epoll.add_fd(sock->fd, ev);

                active_sockets[sock] = sp;
                num_connections.fetch_add(1);
            }
            pending_sockets.clear();
        }

        // If no active sockets, sleep briefly and loop again.
        if (active_sockets.empty()) {
            // Sleep 1ms to avoid busy-waiting.
            std::this_thread::sleep_for(std::chrono::milliseconds(epoll_timeout_ms));
            continue;
        }

        // Wait for events with timeout.
        int num_events = epoll.wait(epoll_timeout_ms);

        // Process events and collect sockets to remove.
        vector<Socket *> sockets_to_remove;

        for (int i = 0; i < num_events; i++) {
            Socket *sock = static_cast<Socket *>(epoll.events[i].data.ptr);
            uint32_t ev_flags = epoll.events[i].events;

            xassert(sock != nullptr);
            xassert(active_sockets.find(sock) != active_sockets.end());

            // Connection closed by peer (via epoll flags).
            if ((ev_flags & EPOLLRDHUP) || (ev_flags & EPOLLHUP) || (ev_flags & EPOLLERR)) {
                sockets_to_remove.push_back(sock);
                continue;
            }

            // Data available to read.
            if (ev_flags & EPOLLIN) {
                // Nonblocking read.
                long nbytes = sock->read(recv_buf.data(), recv_bufsize);

                if (nbytes > 0) {
                    // Data received (and thrown away).
                    num_bytes.fetch_add(nbytes);
                }
                else if ((nbytes == 0) && sock->eof) {
                    // Connection closed (EOF).
                    sockets_to_remove.push_back(sock);
                }
                // If nbytes == 0 and !sock->eof, it's just "would block" - do nothing.
            }
        }

        // Remove closed sockets.
        // With unordered_map + Socket* pointers, no index updates needed.
        for (Socket *sock : sockets_to_remove) {
            auto it = active_sockets.find(sock);
            if (it == active_sockets.end())
                continue;  // already removed (e.g., duplicate in sockets_to_remove)

            epoll.delete_fd(sock->fd);
            active_sockets.erase(it);
            num_connections.fetch_sub(1);
        }
    }
}


}  // namespace pirate
