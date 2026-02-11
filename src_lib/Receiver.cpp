#include "../include/pirate/Receiver.hpp"
#include "../include/pirate/AssembledFrame.hpp"
#include "../include/pirate/inlines.hpp"  // xdiv()

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
struct Receiver::Peer
{
    Socket socket;

    // Parsing state machine.
    // Network protocol reference: notes/network_protocol.md.
    enum class State {
        ReadIni = 0,      // Reading first 8 bytes (magic + yaml_len)
        ReadYaml = 1,     // Reading zero-terminated YAML string
        ReadData = 2      // Reading (nbeams, nfreq, 256) data arrays
    };

    State state = State::ReadIni;
    
    // First 8 bytes (read during State::ReadIni)
    char ini_buf[8];
    long ini_nbytes = 0;  // bytes read so far

    // These members are initialized when state ReadIni -> ReadYaml.
    Array<char> yaml_buf;         // length (yaml_string_len + 1)
    long yaml_nbytes = 0;         // bytes read so far (0 <= yaml_nbytes < yaml_string_len)
    long yaml_string_len = 0;     // total bytes expected from sender

    // All following members are initialized when state ReadYaml -> ReadData.
    XEngineMetadata metadata;
    long bytes_per_segment = 0;   // = nbeams * nfreq * 128
    long segments_per_chunk = 0;  // = Receiver::Params::time_samples_per_chunk / 256

    Array<char> rb_buf;           // length (rb_capacity)
    long rb_capacity = 0;

    // Ring buffer state.
    long rb_start = 0;
    long rb_end = 0;

    explicit Peer(Socket sock);
};


Receiver::Peer::Peer(Socket sock)
    : socket(std::move(sock))
{
    xassert(socket.fd >= 0);
}


// -------------------------------------------------------------------------------------------------


// Helper for Receiver constructor: parse "ip:port" address string.
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


void Receiver::get_status(long &out_num_connections, long &out_nbytes_cumul)
{
    out_num_connections = num_connections.load();
    out_nbytes_cumul = nbytes_cumul.load();
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
            Peer *raw_peer = static_cast<Peer *> (epoll.events[i].data.ptr);
            uint32_t ev_flags = epoll.events[i].events;

            auto it = active_peers.find(raw_peer);
            xassert(raw_peer != nullptr);
            xassert(it != active_peers.end());

            shared_ptr<Peer> sh_peer = it->second;
            xassert(sh_peer.get() == raw_peer);

            // Connection closed by peer (via epoll flags).
            if ((ev_flags & EPOLLRDHUP) || (ev_flags & EPOLLHUP) || (ev_flags & EPOLLERR)) {
                peers_to_remove.push_back(raw_peer);
                continue;
            }

            // Data available to read.
            if (ev_flags & EPOLLIN) {
                // Note: lack of "else if" is intentional -- can "fall through" from one state to the next.
                if (sh_peer->state == Peer::State::ReadIni)
                    this->_read_ini(sh_peer);
                if (sh_peer->state == Peer::State::ReadYaml)
                    this->_read_yaml(sh_peer);
                if (sh_peer->state == Peer::State::ReadData)
                    this->_read_data(sh_peer);

                if (sh_peer->socket.eof)
                    peers_to_remove.push_back(raw_peer);
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


void Receiver::_read_ini(const shared_ptr<Peer> &peer)
{
    xassert(peer->ini_nbytes < 8);

    char *buf = peer->ini_buf + peer->ini_nbytes;
    long bufsize = 8 - peer->ini_nbytes;
    long nbytes_read = peer->socket.read(buf, bufsize);

    // If nbytes_read == 0 and !sock.eof, it's just "would block" - do nothing.
    if (nbytes_read <= 0)
        return;

    peer->ini_nbytes += nbytes_read;
    this->nbytes_cumul.fetch_add(nbytes_read);

    if (peer->ini_nbytes < 8)
        return;

    // Read 4-byte little-endian magic number.
    uint32_t magic = *((uint32_t *) peer->ini_buf);
    static constexpr uint32_t magic_v1 = 0xf4bf4b01;
    
    if (magic != magic_v1) {
        stringstream ss;
        ss << "Receiver::Peer: invalid magic number 0x" << hex << magic
            << " (expected 0x" << magic_v1 << ")";
        throw runtime_error(ss.str());
    }

    // Read 4-byte little-endian string length.
    int32_t yaml_len = *((int32_t *) (peer->ini_buf + 4));
    xassert_gt(yaml_len, 0);
    xassert_le(yaml_len, 1024*1024);
    
    peer->yaml_string_len = yaml_len;
    peer->yaml_buf = Array<char> ({yaml_len+1}, af_uhost);
    peer->yaml_buf.data[yaml_len] = 0;
    peer->state = Peer::State::ReadYaml;
}


void Receiver::_read_yaml(const shared_ptr<Peer> &peer)
{
    xassert(peer->yaml_string_len > 0);
    xassert(peer->yaml_nbytes < peer->yaml_string_len);

    char *yaml_str = peer->yaml_buf.data;
    char *buf = yaml_str + peer->yaml_nbytes;
    long bufsize = peer->yaml_string_len - peer->yaml_nbytes;
    long nbytes_read = peer->socket.read(buf, bufsize);    

    // If nbytes_read == 0 and !sock.eof, it's just "would block" - do nothing.
    if (nbytes_read <= 0)
        return;

    peer->yaml_nbytes += nbytes_read;
    this->nbytes_cumul.fetch_add(nbytes_read);

    if (peer->yaml_nbytes < peer->yaml_string_len)
        return;

    // Verify null terminator.
    if (yaml_str[peer->yaml_string_len - 1] != '\0')
        throw runtime_error("Receiver::Peer: YAML string is not null-terminated");
    
    // Parse YAML string into XEngineMetadata.
    peer->metadata = XEngineMetadata::from_yaml_string(std::string(yaml_str));
    peer->metadata.validate();

    long nfreq = peer->metadata.freq_channels.size();
    long nbeams = peer->metadata.nbeams;

    xassert(nfreq > 0);
    xassert(nbeams > 0);
    xassert(peer->metadata.initial_time_sample == 0);  // FIXME for now!

    peer->bytes_per_segment = nbeams * nfreq * 128;
    peer->segments_per_chunk = xdiv(params.time_samples_per_chunk, 256);

    // Receive buffer size is either 64KB or two segments, whichever is larger.
    long nseg = max((64*1024) / peer->bytes_per_segment, 2L);

    peer->rb_capacity = nseg * peer->bytes_per_segment;
    peer->rb_buf = Array<char> ({peer->rb_capacity}, af_uhost);
    peer->state = Peer::State::ReadData;

    // Lock this->mutex, not peer->mutex!
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


void Receiver::_read_data(const shared_ptr<Peer> &peer)
{
    long rb_capacity = peer->rb_capacity;    
    long rb_start = peer->rb_start;
    long rb_end = peer->rb_end;

    xassert(rb_capacity > 0);
    xassert(rb_start <= rb_end);
    xassert(rb_end < rb_start + rb_capacity);

    // Read new data at rb_end.
    // Compute (buf, bufsize) accounting for wraparound.
    long pos = rb_end % rb_capacity;
    char *buf = peer->rb_buf.data + pos;
    long bufsize = min(rb_capacity - pos, rb_start + rb_capacity - rb_end);

    xassert(bufsize > 0);
    long nbytes_read = peer->socket.read(buf, bufsize);

    // If nbytes_read == 0 and !sock.eof, it's just "would block" - do nothing.
    if (nbytes_read <= 0)
        return;

    rb_end += nbytes_read;
    peer->rb_end = rb_end;
    this->nbytes_cumul.fetch_add(nbytes_read);

    // Currently, we process data in the reader thread.
    // (In the future, this will happen in an assembler thread.)

    while (peer->rb_end >= peer->rb_start + peer->bytes_per_segment)
        this->_process_data(peer);
}


void Receiver::_process_data(const shared_ptr<Peer> &peer)
{
    long nt_chunk = params.time_samples_per_chunk;
    long nfreq = peer->metadata.freq_channels.size();
    long nbeams = peer->metadata.nbeams;
    
    long bs = peer->bytes_per_segment;
    long rb_capacity = peer->rb_capacity;
    long rb_start = peer->rb_start;
    long rb_end = peer->rb_end;

    xassert(bs > 0);
    xassert(rb_capacity > 0);

    long rb_pos = rb_start % rb_capacity;
    xassert(rb_end >= rb_start + bs);
    xassert(rb_capacity >= rb_pos + bs);

    long iseg = rb_start / bs;
    xassert(rb_start == iseg * bs);

    long ichunk = iseg / peer->segments_per_chunk;
    iseg -= ichunk * peer->segments_per_chunk;

    // Target chunk is no longer in buffer (peer is running slow).
    // In this case, we silently drop the data.
    if (ichunk < curr_base_chunk) {
        peer->rb_start += bs;
        return;
    }

    if (ichunk >= curr_base_chunk + 2) {
        // Since chunks are processed sequentially, we should never need to advance
        // the ring buffer by more than one chunk. If this ever fails, it's an internal
        // server error, so throw an exception to flag it for debugging.

        xassert(ichunk == curr_base_chunk + 2);
        
        // Advance the ring buffer. Evicted frames are transferred to completed_frames queue.

        for (long b = 0; b < nbeams; b++) {
            long i = (ichunk & 1) * nbeams + b;
            
            // Transfer evicted frame to completed_frames queue.
            unique_lock<std::mutex> lock(mutex);
            this->completed_frames.push(std::move(this->curr_frames[i]));
            lock.unlock();
            cv.notify_all();
            
            // Replace with new frame from allocator.
            shared_ptr<AssembledFrame> frame = params.allocator->get_frame(params.consumer_id);
            xassert(frame->time_chunk_index == ichunk);
            xassert(frame->beam_id == metadata.beam_ids.at(b));
            this->curr_frames[i] = std::move(frame);
        }

        this->curr_base_chunk++;
    }

    // The rest of _process_data() copies one "segment" of data
    // (all beams, per-sender frequency subset, 256 time samples)
    // into the AssembledFrames.

    const char *src = peer->rb_buf.data + rb_pos;
    const long *freq_channels = &peer->metadata.freq_channels[0];

    for (long b = 0; b < nbeams; b++) {
        long i = (ichunk & 1) * nbeams + b;
        shared_ptr<AssembledFrame> frame = this->curr_frames[i];
        xassert(frame);

        // The destination "base" pointer includes a time offset
        // (128 bytes per segment), but no (beam, frequency) offset.
        char *dst_base = (char *)frame->data.data + (iseg << 7);

        for (long ifreq = 0; ifreq < nfreq; ifreq++) {
            long freq_index = freq_channels[ifreq];
            char *dst = dst_base + freq_index * (nt_chunk >> 1);
            memcpy(dst, src, 128);
            src += 128;
        }
    }

    peer->rb_start += bs;
}


}  // namespace pirate
