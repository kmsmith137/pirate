#include "../include/pirate/Receiver.hpp"
#include "../include/pirate/AssembledFrame.hpp"

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
    
    Array<char> recv_buf;
    long recv_nbytes = 0;   // 0 <= recv_nbytes < recv_buf.size

    // If (state > ReadStringLen), this is the expected YAML string length (including null terminator).
    int32_t yaml_string_len = 0;

    // If (state > ReadYamlString), this contains the parsed metadata.
    XEngineMetadata metadata;

    // The Receiver incrementally receives a data "cube" indexed by (beam, freq, time).
    // Time indices are divided into "chunks" (of size Params::time_samples_per_chunk),
    // and further subdivided into 256-sample "segments".
    // These members keep track of the per-Peer current position in the data cube.
    // They are ordered from slowest varying to fastest varying (see notes/network_protocol.md).

    long curr_chunk = 0;
    long curr_segment = 0;
    long curr_ibeam = 0;
    long curr_ifreq = 0;

    // Constructor.
    explicit Peer(Socket sock);
};


Receiver::Peer::Peer(Socket sock)
    : socket(std::move(sock))
{
    xassert(socket.fd >= 0);
    this->recv_buf = Array<char> ({initial_recv_bufsize}, af_uhost);
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

    params.allocator->stop();
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


// Entry point: retrieve an assembled frame from the queue (blocking).
shared_ptr<AssembledFrame> Receiver::get_frame()
{
    unique_lock<std::mutex> lock(mutex);

    for (;;) {
        _throw_if_stopped("Receiver::get_frame");

        if (!completed_frames.empty()) {
            shared_ptr<AssembledFrame> frame = completed_frames.front();
            completed_frames.pop();
            return frame;
        }

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
                this->_read_data(peer);

                if (peer->socket.eof)
                    peers_to_remove.push_back(peer);
            }
        }

        // Remove closed peers.
        // With unordered_map + Peer* pointers, no index updates needed.
        // Note: if a peer closes the connection with < 128 bytes of residual data
        // (incomplete segment), then the data is silently lost. This is okay.

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


// Called after when peer->socket is ready for reading.
// Reference for network protocol: notes/network_protocol.md.

void Receiver::_read_data(Peer *peer)
{
    // Receive bufsize must always be a multiple of 128 (see below).
    static_assert((initial_recv_bufsize % 128) == 0);

    char *recv_buf = peer->recv_buf.data;
    long recv_nbytes = peer->recv_nbytes;
    long recv_capacity = peer->recv_buf.size;

    long nbytes = peer->socket.read(recv_buf + recv_nbytes, recv_capacity - recv_nbytes);

    // If nbytes == 0 and !sock.eof, it's just "would block" - do nothing.
    if (nbytes <= 0)
        return;

    recv_nbytes += nbytes;
    peer->recv_nbytes = recv_nbytes;
    this->num_bytes.fetch_add(nbytes);
    
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
        // Receive bufsize must always be a multiple of 128 (see below).
        long target_size = peer->yaml_string_len + 8;
        target_size = (target_size + 127) & ~127;  // align to multiple of 128
        
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
        
        const char *yaml_string = recv_buf + 8;

        // Verify null terminator.
        if (yaml_string[peer->yaml_string_len - 1] != '\0')
            throw runtime_error("Receiver::Peer: YAML string is not null-terminated");
        
        // Parse YAML string into XEngineMetadata.
        peer->metadata = XEngineMetadata::from_yaml_string(std::string(yaml_string));
        peer->metadata.validate();

        this->_post_metadata(peer);

        // After processing the metadata, there may be some int4 data in the buffer.

        const char *data = recv_buf + peer->yaml_string_len + 8;
        long data_nbytes = recv_nbytes - peer->yaml_string_len - 8;
        long nsegments = data_nbytes >> 7;
        long residual = data_nbytes - (nsegments << 7);

        this->_process_4bit_data(peer, data, nsegments);

        // Move residual data to front of buffer.
        memmove(recv_buf, data + (nsegments << 7), residual);
        peer->recv_nbytes = residual;

        peer->state = Peer::State::ReadData;
        return;  // don't fall through
    }
    
    if (peer->state == Peer::State::ReadData) {
        long nsegments = recv_nbytes >> 7;
        long residual = recv_nbytes - (nsegments << 7);

        if (nsegments > 0)
            this->_process_4bit_data(peer, recv_buf, nsegments);

        // Move residual data to front of buffer.
        // We use memcpy(..., 128) instead of memmove(..., residual) so that the compiler
        // will emit a few "unrolled" simd instructions. This is safe because the receive
        // bufsize is always a multiple of 128, and "if ((nsegments > 0) && (residual > 0))"
        // protects against read-past-end-of-buffer or overlapping-dst-src.

        if ((nsegments > 0) && (residual > 0))
            memcpy(recv_buf, recv_buf + (nsegments << 7), 128);

        peer->recv_nbytes = residual;
        return;
    }

    throw runtime_error("Should never get here");
}


// Called after peer->metadata is initialized.
void Receiver::_post_metadata(Peer *peer)
{
    xassert(peer->metadata.freq_channels.size() > 0);
    xassert(peer->metadata.beam_ids.size() > 0);

    unique_lock<std::mutex> lock(mutex);

    if (this->has_metadata) {
        // It's okay to access this->metadata after dropping the lock, since this->metadata
        // is constant after initialization.
        lock.unlock();
        XEngineMetadata::check_sender_consistency(this->metadata, peer->metadata);
        return;
    }

    // If we get here, then we're seeing the X-engine metadata for the first time.
    // This triggers a lot of initialization logic!

    this->metadata = peer->metadata;
    this->has_metadata = true;

    // We clear metadata.freq_channels, since it would otherwise contain the frequency channels
    // sent by one arbitrarily selected peer, which would be more confusing than helpful.
    this->metadata.freq_channels.clear();

    // Okay to drop lock at this point.
    // (It's okay to access this->metadata after dropping the lock, see above.)
    lock.unlock();
    this->cv.notify_all();

    // Initialize AssembledFrameAllocator and curr_frames.
    // (No lock held here, but there's only one reader thread, so no possibility of race conditions.)

    this->params.allocator->initialize(
        this->params.consumer_id,
        this->metadata.get_total_nfreq(),
        this->params.time_samples_per_chunk, 
        this->metadata.beam_ids
    );
    
    long nbeams = this->metadata.nbeams;
    this->curr_frames.resize(2*nbeams);

    for (long ichunk = 0; ichunk < 2; ichunk++) {
        for (long ibeam = 0; ibeam < nbeams; ibeam++) {
            auto frame = params.allocator->get_frame(params.consumer_id);
            xassert(frame->time_chunk_index == ichunk);
            xassert(frame->beam_id == metadata.beam_ids.at(ibeam));

            this->curr_frames[ichunk*nbeams + ibeam] = frame;
        }
    }
}


// Called after 4-bit data is received over the network, to copy the data 
// into the AssembledFrames.

void Receiver::_process_4bit_data(Peer *peer, const char *buf, long n)
{
    if (n == 0)
        return;

    const long *freq_channels = &peer->metadata.freq_channels[0];
    long nfreq = peer->metadata.freq_channels.size();
    long nt_chunk = params.time_samples_per_chunk;

    long ichunk = peer->curr_chunk;
    long iseg = peer->curr_segment;
    long ibeam = peer->curr_ibeam;
    long ifreq = peer->curr_ifreq;

    // Points to an int4 array of shape (nfreq_tot, time_samples_per_chunk)
    char *frame = this->_find_frame(ichunk, ibeam);

    for (long i = 0; i < n; i++) {
        if (frame != nullptr) {
            // Logical array location (freq_index, 256 * iseg).
            // Note factors of 2, since the array is int4, but the pointer is (char *).
            long freq_index = freq_channels[ifreq];
            char *dst = frame + freq_index * (nt_chunk >> 1) + (iseg << 7);
            memcpy(dst, buf, 128);
        }

        // Advance to the next value (ichunk, iseg, ibeam, ifreq).
        // If (ichunk, ibeam) change, then we call _find_frame() again.

        buf += 128;
        ifreq++;

        if (ifreq < nfreq)
            continue;

        ifreq = 0;
        ibeam++;

        if (ibeam == metadata.nbeams) {
            ibeam = 0;
            iseg++;
        }

        if (iseg == (nt_chunk >> 8)) {
            iseg = 0;
            ichunk++;
        }

        frame = this->_find_frame(ichunk, ibeam);
    }

    peer->curr_chunk = ichunk;
    peer->curr_segment = iseg;
    peer->curr_ibeam = ibeam;
    peer->curr_ifreq = ifreq;
}


// Returns a pointer to the specified frame in the 'curr_frames' ringbuf,
// advancing the ringbuf if necessary. Returns NULL if the ring buffer has
// already advanced beyond the target frame.

char *Receiver::_find_frame(long ichunk, long ibeam)
{
    long nbeams = metadata.nbeams;

    // Target chunk is no longer in buffer (peer is running slow).
    // In this case, we siletly drop the data.
    if (ichunk < curr_base_chunk)
        return nullptr;

    if (ichunk >= curr_base_chunk + 2) {
        // Since chunks are processed sequentially, we should never need to advance
        // the ring buffer by more than one chunk. If this ever fails, it's an internal
        // server error, so throw an exception to flag it for debugging.
        xassert(ichunk == curr_base_chunk + 2);
        
        // Advance the ring buffer. Evicted frames are transferred to completed_frames queue.

        for (long b = 0; b < nbeams; b++) {
            long i = (ichunk & 1) * nbeams + b;
            
            // Transfer evicted frame to completed_frames queue.
            {
                lock_guard<std::mutex> lock(mutex);
                this->completed_frames.push(std::move(this->curr_frames[i]));
            }
            this->cv.notify_all();
            
            // Replace with new frame from allocator.
            shared_ptr<AssembledFrame> frame = params.allocator->get_frame(params.consumer_id);
            xassert(frame->time_chunk_index == ichunk);
            xassert(frame->beam_id == metadata.beam_ids.at(b));
            this->curr_frames[i] = std::move(frame);
        }

        this->curr_base_chunk++;
    }

    long i = (ichunk & 1) * nbeams + ibeam;
    return (char *) (curr_frames[i]->data.data);
}


}  // namespace pirate
