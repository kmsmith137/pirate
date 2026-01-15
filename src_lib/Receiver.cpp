#include "../include/pirate/Receiver.hpp"
#include "../include/pirate/XEngineMetadata.hpp"

#include <cstring>    // memcpy
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include <ksgpu/xassert.hpp>

using namespace std;


namespace pirate {
#if 0
}  // editor auto-indent
#endif


// -------------------------------------------------------------------------------------------------
//
// Receiver::Peer


// Peer: per-sender state for an active TCP connection.
// Parses the network protocol described in notes/network_protocol.md.
struct Receiver::Peer
{
    // Parsing state machine.
    enum class State {
        ReadMagic,       // Reading 4-byte magic number 0xf4bf4b01
        ReadStringLen,   // Reading 4-byte YAML string length
        ReadYamlString,  // Reading zero-terminated YAML string
        ReadData,        // Reading (nbeams, nfreq, 256) data arrays
        Error            // Protocol error occurred
    };

    Socket socket;
    State state = State::ReadMagic;

    // Receive buffer for accumulating partial data.
    std::vector<char> recv_buf;

    // Size of pending write (set by get_recv_buffer, used by process_received).
    long pending_write_size = 0;

    // After ReadStringLen, this is the expected YAML string length (including null terminator).
    uint32_t yaml_string_len = 0;

    // After ReadYamlString, this contains the parsed metadata.
    XEngineMetadata metadata;

    // Size of each data chunk: (nbeams * nfreq * 256) / 2 bytes (int4 pairs).
    // Set after metadata is parsed.
    long data_chunk_bytes = 0;

    // Number of bytes read in the current data chunk.
    long data_chunk_pos = 0;

    // Magic number for protocol version 1.
    static constexpr uint32_t magic_v1 = 0xf4bf4b01;

    // Constructor.
    explicit Peer(Socket sock);

    // Returns pointer to buffer where new data should be written.
    // Ensures recv_buf has space for at least 'nbytes' more bytes.
    char *get_recv_buffer(long nbytes);

    // Called after 'nbytes' have been written to the buffer returned by get_recv_buffer().
    // Processes the data and updates state machine.
    void process_received(long nbytes);

private:
    // Process as much data as possible from recv_buf. Called by process_received().
    void _process_recv_buf();
};


Receiver::Peer::Peer(Socket sock)
    : socket(std::move(sock))
{
    xassert(socket.fd >= 0);
}


char *Receiver::Peer::get_recv_buffer(long nbytes)
{
    xassert(pending_write_size == 0);  // Must call process_received() before next get_recv_buffer().
    size_t old_size = recv_buf.size();
    recv_buf.resize(old_size + nbytes);
    pending_write_size = nbytes;
    return recv_buf.data() + old_size;
}


void Receiver::Peer::process_received(long nbytes)
{
    xassert(pending_write_size > 0);
    xassert(nbytes >= 0);
    xassert(nbytes <= pending_write_size);

    // Shrink recv_buf if we received fewer bytes than requested.
    if (nbytes < pending_write_size) {
        recv_buf.resize(recv_buf.size() - (pending_write_size - nbytes));
    }
    pending_write_size = 0;

    _process_recv_buf();
}


void Receiver::Peer::_process_recv_buf()
{
    // Process as much as possible from recv_buf.
    while (true) {
        if (state == State::ReadMagic) {
            if (recv_buf.size() < 4)
                break;

            // Read 4-byte little-endian magic number.
            uint32_t magic = 0;
            memcpy(&magic, recv_buf.data(), 4);

            if (magic != magic_v1) {
                stringstream ss;
                ss << "Receiver::Peer: invalid magic number 0x" << hex << magic
                   << " (expected 0x" << magic_v1 << ")";
                throw runtime_error(ss.str());
            }

            // Consume 4 bytes from recv_buf.
            recv_buf.erase(recv_buf.begin(), recv_buf.begin() + 4);
            state = State::ReadStringLen;
        }
        else if (state == State::ReadStringLen) {
            if (recv_buf.size() < 4)
                break;

            // Read 4-byte little-endian string length.
            memcpy(&yaml_string_len, recv_buf.data(), 4);

            if (yaml_string_len == 0) {
                throw runtime_error("Receiver::Peer: YAML string length is zero");
            }
            if (yaml_string_len > 1024 * 1024) {
                stringstream ss;
                ss << "Receiver::Peer: YAML string length " << yaml_string_len
                   << " exceeds maximum (1 MB)";
                throw runtime_error(ss.str());
            }

            // Consume 4 bytes from recv_buf.
            recv_buf.erase(recv_buf.begin(), recv_buf.begin() + 4);
            state = State::ReadYamlString;
        }
        else if (state == State::ReadYamlString) {
            if (recv_buf.size() < yaml_string_len)
                break;

            // Verify null terminator.
            if (recv_buf[yaml_string_len - 1] != '\0') {
                throw runtime_error("Receiver::Peer: YAML string is not null-terminated");
            }

            // Parse YAML string into XEngineMetadata.
            string yaml_str(recv_buf.data(), yaml_string_len - 1);  // Exclude null terminator.
            metadata = XEngineMetadata::from_yaml_string(yaml_str);

            // Compute data chunk size: (nbeams * nfreq * 256) int4 values, packed 2 per byte.
            long nfreq = static_cast<long>(metadata.freq_channels.size());
            if (nfreq == 0) {
                // If freq_channels is empty, use total_nfreq.
                nfreq = metadata.get_total_nfreq();
            }
            data_chunk_bytes = (metadata.nbeams * nfreq * 256) / 2;

            // Consume yaml_string_len bytes from recv_buf.
            recv_buf.erase(recv_buf.begin(), recv_buf.begin() + yaml_string_len);
            state = State::ReadData;
        }
        else if (state == State::ReadData) {
            // Discard data (placeholder for future processing).
            // Just consume all remaining bytes in recv_buf.
            if (recv_buf.empty())
                break;

            long bytes_to_consume = recv_buf.size();

            // Track position in current data chunk (for future use).
            data_chunk_pos += bytes_to_consume;
            while (data_chunk_pos >= data_chunk_bytes) {
                data_chunk_pos -= data_chunk_bytes;
            }

            recv_buf.clear();
        }
        else if (state == State::Error) {
            // In error state, discard all data.
            recv_buf.clear();
            break;
        }
        else {
            throw runtime_error("Receiver::Peer: unknown state");
        }
    }
}


// -------------------------------------------------------------------------------------------------


Receiver::Receiver(const std::string &ip_addr_, uint16_t tcp_port_) :
    ip_addr(ip_addr_),
    tcp_port(tcp_port_)
{
    xassert(ip_addr.size() > 0);
    xassert(tcp_port > 0);
}


void Receiver::start()
{
    std::lock_guard<std::mutex> lock(mutex);

    if (is_started)
        throw std::runtime_error("Receiver::start() called twice");
    if (is_stopped)
        throw std::runtime_error("Receiver::start() called after stop()");

    is_started = true;

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

        // Create Peer (which owns the Socket) and hand off to reader thread.
        auto peer_ptr = std::make_shared<Peer>(std::move(new_socket));

        {
            std::lock_guard<std::mutex> lock(mutex);
            pending_peers.push_back(peer_ptr);
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
    unordered_map<Peer *, shared_ptr<Peer>> active_peers;

    while (true) {
        // Check if stopped, and receive pending peers from listener thread.
        {
            std::lock_guard<std::mutex> lock(mutex);

            if (is_stopped)
                return;

            // Receive pending peers from listener.
            for (auto &peer_ptr : pending_peers) {
                Peer *peer = peer_ptr.get();
                Socket &sock = peer->socket;
                xassert(peer != nullptr);
                xassert(sock.fd >= 0);
                xassert(active_peers.find(peer) == active_peers.end());

                sock.set_nonblocking();

                epoll_event ev;
                ev.events = EPOLLIN | EPOLLRDHUP | EPOLLHUP;
                ev.data.ptr = peer;  // store Peer* pointer in epoll context
                epoll.add_fd(sock.fd, ev);

                active_peers[peer] = peer_ptr;
                num_connections.fetch_add(1);
            }
            pending_peers.clear();
        }

        // If no active peers, sleep briefly and loop again.
        if (active_peers.empty()) {
            // Sleep 1ms to avoid busy-waiting.
            std::this_thread::sleep_for(std::chrono::milliseconds(epoll_timeout_ms));
            continue;
        }

        // Wait for events with timeout.
        int num_events = epoll.wait(epoll_timeout_ms);

        // Process events and collect peers to remove.
        vector<Peer *> peers_to_remove;

        for (int i = 0; i < num_events; i++) {
            Peer *peer = static_cast<Peer *>(epoll.events[i].data.ptr);
            Socket &sock = peer->socket;
            uint32_t ev_flags = epoll.events[i].events;

            xassert(peer != nullptr);
            xassert(active_peers.find(peer) != active_peers.end());

            // Connection closed by peer (via epoll flags).
            if ((ev_flags & EPOLLRDHUP) || (ev_flags & EPOLLHUP) || (ev_flags & EPOLLERR)) {
                peers_to_remove.push_back(peer);
                continue;
            }

            // Data available to read.
            if (ev_flags & EPOLLIN) {
                // Get buffer from Peer to read directly into.
                char *buf = peer->get_recv_buffer(recv_bufsize);

                // Nonblocking read directly into Peer's buffer.
                long nbytes = sock.read(buf, recv_bufsize);

                if (nbytes > 0) {
                    // Process data through the Peer state machine.
                    // This parses the protocol (magic, string length, YAML, data).
                    peer->process_received(nbytes);
                    num_bytes.fetch_add(nbytes);
                }
                else {
                    // No data read - must still call process_received(0) to reset state.
                    peer->process_received(0);

                    if ((nbytes == 0) && sock.eof) {
                        // Connection closed (EOF).
                        peers_to_remove.push_back(peer);
                    }
                }
                // If nbytes == 0 and !sock.eof, it's just "would block" - do nothing.
            }
        }

        // Remove closed peers.
        // With unordered_map + Peer* pointers, no index updates needed.
        for (Peer *peer : peers_to_remove) {
            auto it = active_peers.find(peer);
            if (it == active_peers.end())
                continue;  // already removed (e.g., duplicate in peers_to_remove)

            epoll.delete_fd(peer->socket.fd);
            active_peers.erase(it);
            num_connections.fetch_sub(1);
        }
    }
}


}  // namespace pirate
