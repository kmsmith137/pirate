#include "../include/pirate/Receiver.hpp"
#include "../include/pirate/XEngineMetadata.hpp"

#include <cstring>    // memcpy
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include <ksgpu/Array.hpp>
#include <ksgpu/xassert.hpp>

using namespace std;
using namespace ksgpu;


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
    Socket socket;

    // Parsing state machine.
    enum class State {
        ReadMagic,       // Reading 4-byte magic number 0xf4bf4b01
        ReadStringLen,   // Reading 4-byte YAML string length
        ReadYamlString,  // Reading zero-terminated YAML string
        ReadData,        // Reading (nbeams, nfreq, 256) data arrays
    };

    State state = State::ReadMagic;
    Array<char> recv_buf;           // only used while state < ReadData
    long recv_nbytes = 0;           // 0 <= recv_nbytes < recv_buf.size

    // If (state > ReadStringLen), this is the expected YAML string length (including null terminator).
    int32_t yaml_string_len = 0;

    // If (state > ReadYamlString), this contains the parsed metadata.
    XEngineMetadata metadata;

    // Constructor.
    explicit Peer(Socket sock);

    // Caller reads data into 'recv_buf', updates 'recv_nbytes', then calls process_received().
    void process_received();
};


Receiver::Peer::Peer(Socket sock)
    : socket(std::move(sock))
{
    xassert(socket.fd >= 0);
    this->recv_buf = Array<char> ({10*1024}, af_uhost);
}


void Receiver::Peer::process_received()
{
    // Magic number for protocol version 1.
    static constexpr uint32_t magic_v1 = 0xf4bf4b01;

    // Process as much as possible from recv_buf.
    if (state == State::ReadMagic) {
        if (recv_nbytes < 4)
            return;

        // Read 4-byte little-endian magic number.
        uint32_t magic = *((uint32_t *) recv_buf.data);

        if (magic != magic_v1) {
            stringstream ss;
            ss << "Receiver::Peer: invalid magic number 0x" << hex << magic
                << " (expected 0x" << magic_v1 << ")";
            throw runtime_error(ss.str());
        }

        state = State::ReadStringLen;
        // fall through...
    }

    if (state == State::ReadStringLen) {
        if (recv_nbytes < 8)
            return;

        // Read 4-byte little-endian string length.
        this->yaml_string_len = *((int32_t *) (recv_buf.data + 4));
        xassert_gt(yaml_string_len, 0);
        xassert_le(yaml_string_len, 1024*1024);

        // Enlarge recv_buf if needed.
        long target_size = yaml_string_len + 8;

        if (recv_buf.size < target_size) {
            Array<char> new_buf({target_size}, af_uhost);
            memcpy(new_buf.data, recv_buf.data, recv_nbytes);
            recv_buf = new_buf;
        }

        state = State::ReadYamlString;
        // fall through...
    }

    if (state == State::ReadYamlString) {
        if (recv_nbytes < yaml_string_len + 8)
            return;

        const char *yaml_string = recv_buf.data + 8;

        // Verify null terminator.
        if (yaml_string[yaml_string_len - 1] != '\0')
            throw runtime_error("Receiver::Peer: YAML string is not null-terminated");

        // Parse YAML string into XEngineMetadata.
        this->metadata = XEngineMetadata::from_yaml_string(std::string(yaml_string));
        this->metadata.validate();
        state = State::ReadData;
        return;  // don't fall through
    }

    throw runtime_error("Receiver::Peer::process_received(): should never get here");
}


// -------------------------------------------------------------------------------------------------


Receiver::Receiver(const string &ip_addr_, uint16_t tcp_port_) :
    ip_addr(ip_addr_),
    tcp_port(tcp_port_)
{
    xassert(ip_addr.size() > 0);
    xassert(tcp_port > 0);
}


void Receiver::start()
{
    lock_guard<std::mutex> lock(mutex);

    if (is_started)
        throw runtime_error("Receiver::start() called twice");
    if (is_stopped)
        throw runtime_error("Receiver::start() called after stop()");

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
    lock_guard<std::mutex> lock(mutex);

    if (is_stopped)
        return;

    is_stopped = true;
    error = e;
    cv.notify_all();
}


void Receiver::_throw_if_stopped(const char *method_name) const
{
    if (error)
        std::rethrow_exception(error);

    if (is_stopped)
        throw runtime_error(string(method_name) + " called on stopped instance");
}


XEngineMetadata Receiver::get_metadata(bool blocking) const
{
    unique_lock<std::mutex> lock(mutex);

    for (;;) {
        _throw_if_stopped("Receiver::get_metadata");

        if (has_metadata)
            return metadata;
        if (!blocking)
            return XEngineMetadata();

        cv.wait(lock);
    }
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
            lock_guard<std::mutex> lock(mutex);
            if (is_stopped)
                return;
        }

        // Accept with timeout, so we can check is_stopped frequently.
        Socket new_socket = listening_socket.accept(accept_timeout_ms);

        // Timeout expired, loop again.
        if (new_socket.fd < 0)
            continue;

        // Create Peer (which owns the Socket) and hand off to reader thread.
        auto peer_ptr = make_shared<Peer> (std::move(new_socket));

        {
            lock_guard<std::mutex> lock(mutex);
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

    // Currently, this buffer is just used for reading data that we plan to throw away.
    Array<char> recv_buf({recv_bufsize}, af_uhost);

    while (true) {
        // Check if stopped, and receive pending peers from listener thread.
        {
            lock_guard<std::mutex> lock(mutex);

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
                bool use_peer_recv_buf = (peer->state < Peer::State::ReadData);
                char *buf = use_peer_recv_buf ? (peer->recv_buf.data + peer->recv_nbytes) : (recv_buf.data);
                long bufsize = use_peer_recv_buf ? (peer->recv_buf.size - peer->recv_nbytes) : (recv_buf.size);

                long nbytes = sock.read(buf, bufsize);

                if (nbytes > 0) {
                    if (use_peer_recv_buf) {
                        peer->recv_nbytes += nbytes;
                        peer->process_received();

                        // If peer just finished parsing metadata, validate consistency.
                        if (peer->state == Peer::State::ReadData) {
                            lock_guard<std::mutex> lock(mutex);
                            if (!has_metadata) {
                                // We clear metadata.freq_channels, since it would otherwise contain the frequency channels
                                // sent by one arbitrarily selected peer, which would be more confusing than helpful.
                                metadata = peer->metadata;
                                metadata.freq_channels.clear();
                                has_metadata = true;
                                cv.notify_all();
                            } else {
                                XEngineMetadata::check_sender_consistency(metadata, peer->metadata);
                            }
                        }
                    }
                    this->num_bytes.fetch_add(nbytes);
                }
                
                if (sock.eof)
                    peers_to_remove.push_back(peer);

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
