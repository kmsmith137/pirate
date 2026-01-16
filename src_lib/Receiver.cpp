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
        ReadData         // Reading (nbeams, nfreq, 256) data arrays
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
};


Receiver::Peer::Peer(Socket sock)
    : socket(std::move(sock))
{
    xassert(socket.fd >= 0);
    this->recv_buf = Array<char> ({recv_bufsize}, af_uhost);
}


// -------------------------------------------------------------------------------------------------


// Helper: parse "ip:port" address string.
static void parse_address(const string &address, string &ip_addr, uint16_t &tcp_port)
{
    size_t colon_pos = address.rfind(':');
    if (colon_pos == string::npos) {
        throw runtime_error("Receiver: invalid address '" + address + "' (expected 'ip:port' format)");
    }

    ip_addr = address.substr(0, colon_pos);
    string port_str = address.substr(colon_pos + 1);

    if (ip_addr.empty()) {
        throw runtime_error("Receiver: invalid address '" + address + "' (empty IP)");
    }
    if (port_str.empty()) {
        throw runtime_error("Receiver: invalid address '" + address + "' (empty port)");
    }

    try {
        int port = std::stoi(port_str);
        if ((port <= 0) || (port > 65535)) {
            throw runtime_error("Receiver: invalid port in address '" + address + "'");
        }
        tcp_port = static_cast<uint16_t>(port);
    } catch (const std::exception &) {
        throw runtime_error("Receiver: invalid port in address '" + address + "'");
    }
}


Receiver::Receiver(const Params &p) : params(p)
{
    parse_address(params.address, this->ip_addr, this->tcp_port);

    xassert(params.allocator);
    xassert(params.consumer_id >= 0);
    xassert(params.time_samples_per_chunk > 0);
    xassert_divisible(params.time_samples_per_chunk, 256);
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
                char *buf = peer->recv_buf.data + peer->recv_nbytes;
                long bufsize = peer->recv_buf.size - peer->recv_nbytes;
                long nbytes = sock.read(buf, bufsize);

                if (nbytes > 0) {
                    peer->recv_nbytes += nbytes;
                    this->num_bytes.fetch_add(nbytes);
                    this->_post_receive(peer);
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


void Receiver::_post_receive(Peer *peer)
{
    char *recv_buf = peer->recv_buf.data;
    long recv_nbytes = peer->recv_nbytes;
    
    // Process as much as possible from recv_buf.
    if (peer->state == Peer::State::ReadMagic) {
        if (recv_nbytes < 4)
            return;
        
        // Read 4-byte little-endian magic number.
        uint32_t magic = *((uint32_t *) recv_buf);
        static constexpr uint32_t magic_v1 = 0xf4bf4b01;
        
        if (magic != magic_v1) {
            stringstream ss;
            ss << "Receiver::Peer: invalid magic number 0x" << hex << magic
               << " (expected 0x" << magic_v1 << ")";
            throw runtime_error(ss.str());
        }
        
        peer->state = Peer::State::ReadStringLen;
        // fall through...
    }
    
    if (peer->state == Peer::State::ReadStringLen) {
        if (recv_nbytes < 8)
            return;
        
        // Read 4-byte little-endian string length.
        peer->yaml_string_len = *((int32_t *) (recv_buf + 4));
        xassert_gt(peer->yaml_string_len, 0);
        xassert_le(peer->yaml_string_len, 1024*1024);
        
        // Enlarge recv_buf if needed.
        long target_size = peer->yaml_string_len + 8;
        
        if (peer->recv_buf.size < target_size) {
            Array<char> new_buf({target_size}, af_uhost);
            memcpy(new_buf.data, recv_buf, recv_nbytes);
            peer->recv_buf = new_buf;
            recv_buf = new_buf.data;
        }
        
        peer->state = Peer::State::ReadYamlString;
        // fall through...
    }
    
    if (peer->state == Peer::State::ReadYamlString) {
        if (recv_nbytes < peer->yaml_string_len + 8)
           return;
        
        const char *yaml_string = recv_buf+8;
        
        // Verify null terminator.
        if (yaml_string[peer->yaml_string_len - 1] != '\0')
            throw runtime_error("Receiver::Peer: YAML string is not null-terminated");
        
        // Parse YAML string into XEngineMetadata.
        peer->metadata = XEngineMetadata::from_yaml_string(std::string(yaml_string));
        peer->metadata.validate();

        unique_lock<std::mutex> lock(mutex);

        if (!this->has_metadata) {
            // We clear metadata.freq_channels, since it would otherwise contain the frequency channels
            // sent by one arbitrarily selected peer, which would be more confusing than helpful.
            this->metadata = peer->metadata;
            this->metadata.freq_channels.clear();
            this->has_metadata = true;
            this->cv.notify_all();
        } 
        else
            XEngineMetadata::check_sender_consistency(this->metadata, peer->metadata);
        
        peer->state = Peer::State::ReadData;
        // fall through...
    }
    
    if (peer->state == Peer::State::ReadData) {
        // For now, throw data away!
        peer->recv_nbytes = 0;
        return;
    }

    throw runtime_error("Should never get here");
}


}  // namespace pirate
