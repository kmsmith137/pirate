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
    long bytes_per_minichunk = 0;   // = nbeams * nfreq * 128
    long minichunks_per_chunk = 0;  // = Receiver::Params::time_samples_per_chunk / 256

    Array<char> rb_buf;           // length (rb_capacity)
    long rb_capacity = 0;

    // Ring buffer counters (rb_start, rb_end). Shared between reader/assembler threads.
    // The mutex protects these counters, but no other Receiver::Peer members.
    std::mutex mutex;
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
    unique_lock<std::mutex> lock(mutex);

    if (is_started)
        throw runtime_error("Receiver::start() called twice");
    if (is_stopped)
        throw runtime_error("Receiver::start() called after stop()");

    is_started = true;
    lock.unlock();

    // Spawn worker threads.
    listener_thread = std::thread(&Receiver::listener_main, this);
    reader_thread = std::thread(&Receiver::reader_main, this);
    assembler_thread = std::thread(&Receiver::assembler_main, this);
}


Receiver::~Receiver()
{
    this->stop();

    if (listener_thread.joinable())
        listener_thread.join();

    if (reader_thread.joinable())
        reader_thread.join();

    if (assembler_thread.joinable())
        assembler_thread.join();
}


void Receiver::get_status(long &out_num_connections, long &out_nbytes_cumul)
{
    out_num_connections = num_connections.load();
    out_nbytes_cumul = nbytes_cumul.load();
}


void Receiver::stop(std::exception_ptr e)
{
    unique_lock<std::mutex> lock(mutex);

    if (is_stopped)
        return;

    is_stopped = true;
    error = e;
    lock.unlock();

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

    for (;;) {
        unique_lock<std::mutex> lock(mutex);
        if (is_stopped) return;
        lock.unlock();

        // Call accept() without lock held, and with a ~10ms timeout so that we
        // regularly check for calls to Receiver::stop() (via is_stopped).
        Socket new_socket = listening_socket.accept(accept_timeout_ms);

        // Timeout expired, loop again.
        if (new_socket.fd < 0)
            continue;

        // Create Peer (which owns the Socket) and hand off to reader thread.
        auto peer_ptr = make_shared<Peer> (std::move(new_socket));

        lock.lock();
        if (is_stopped) return;
        reader_peer_queue.push_back(peer_ptr);
        lock.unlock();

        // Wake up reader thread.
        cv.notify_all();
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
    Epoll epoll;
    unordered_map<Peer *, shared_ptr<Peer>> active_peers;

    // Each iteration of the outer loop corresponds to one call to epoll.wait().

    for (;;) {
        // Acquire lock to check is_stopped and drain reader_peer_queue.
        unique_lock<std::mutex> lock(mutex);

        for (;;) {
            if (is_stopped)
                return;

            if (!reader_peer_queue.empty())
                break;
            if (!active_peers.empty())
                break;

            // No active peers and no pending peers -- wait for listener thread.
            cv.wait(lock);
        }

        // Move pending peers from reader_peer_queue to local vector
        // (so that we can drop the lock before calling set_nonblocking/epoll).
        vector<shared_ptr<Peer>> new_peers = std::move(reader_peer_queue);
        reader_peer_queue.clear();
        lock.unlock();

        // Add new peers to epoll and active_peers (no lock needed).
        for (const shared_ptr<Peer> &peer_ptr: new_peers) {
            Peer *peer = peer_ptr.get();
            xassert(peer != nullptr);

            Socket &sock = peer->socket;
            xassert(sock.fd >= 0);
            sock.set_nonblocking();

            epoll_event ev;
            ev.events = EPOLLIN | EPOLLRDHUP | EPOLLHUP;
            ev.data.ptr = peer;  // store Peer* pointer in epoll context
            epoll.add_fd(sock.fd, ev);

            num_connections.fetch_add(1);

            xassert(active_peers.find(peer) == active_peers.end());
            active_peers[peer] = peer_ptr;
        }

        if (active_peers.empty())
            continue;  // back to top, re-acquire lock

        // Lock is not held here.
        // Wait for events. We use a ~1ms timeout here, rather than a blocking call,
        // so that we check for new connections (via reader_peer_queue), and check
        // for calls to Receiver::stop() (via is_stopped).
        int num_events = epoll.wait(epoll_timeout_ms);

        // Process events and collect peers to remove.
        vector<Peer *> peers_to_remove;

        for (int i = 0; i < num_events; i++) {
            Peer *raw_peer = static_cast<Peer *> (epoll.events[i].data.ptr);
            uint32_t ev_flags = epoll.events[i].events;
            xassert(raw_peer != nullptr);

            auto it = active_peers.find(raw_peer);
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

        // Remove closed peers (no lock needed -- active_peers is local, num_connections is atomic).
        // Note: if a peer closes the connection with residual data (incomplete minichunk),
        // then data is silently lost -- this is okay.

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

    peer->bytes_per_minichunk = nbeams * nfreq * 128;
    peer->minichunks_per_chunk = xdiv(params.time_samples_per_chunk, 256);

    // Receive buffer size is either 64KB or two minichunks, whichever is larger.
    long nmc = max((64*1024) / peer->bytes_per_minichunk, 2L);

    peer->rb_capacity = nmc * peer->bytes_per_minichunk;
    peer->rb_buf = Array<char> ({peer->rb_capacity}, af_uhost);
    peer->state = Peer::State::ReadData;

    // Lock this->mutex, not peer->mutex!
    unique_lock<std::mutex> lock(mutex);
    _throw_if_stopped("Receiver::_read_yaml");

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

    lock.unlock();
    cv.notify_all();  // wake up any threads waiting for metadata (including assembler thread)
}


void Receiver::_read_data(const shared_ptr<Peer> &peer)
{
    long rb_capacity = peer->rb_capacity;
    xassert(rb_capacity > 0);

    // Note peer->mutex here, not this->mutex!
    unique_lock<std::mutex> plock(peer->mutex);
    long rb_start = peer->rb_start;
    long rb_end = peer->rb_end;
    plock.unlock();

    // If there is no room in the ring buffer, then return without calling Socket::read().
    // This happens if the assembler thread is running slow, and will put the reader thread
    // into a busy-wait state, making repeated nonblocking calls to Epoll::wait().
    // FIXME is there a better alternative? Or is this not worth worrying about?

    if (rb_end >= rb_start + rb_capacity)
        return;

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

    this->nbytes_cumul.fetch_add(nbytes_read);

    // Reacquire peer->mutex, update peer->rb_end, decide whether to queue for assembler thread.
    plock.lock();
    xassert(peer->rb_end == rb_end);
    peer->rb_end += nbytes_read;
    bool enqueue = (peer->rb_end >= peer->rb_start + peer->bytes_per_minichunk);
    plock.unlock();

    if (enqueue) {
        // Enqueue 'peer' for processing by assembler thread.
        // Note this->mutex here, not peer->mutex!
        unique_lock<std::mutex> rlock(this->mutex);
        _throw_if_stopped("Receiver::_read_data");
        this->assembler_peer_queue.push_back(peer);
        rlock.unlock();
        cv.notify_all();
    }
}


// -------------------------------------------------------------------------------------------------
//
// Assembler thread


void Receiver::assembler_main()
{
    try {
        _assembler_main();
    } catch (...) {
        stop(std::current_exception());
    }
}


void Receiver::_assembler_main()
{
    // Wait for yaml metadata to be parsed, by calling get_metadata(blocking=true) .
    XEngineMetadata metadata = this->get_metadata(true);

    // The assembler thread is responsible for calling AssembledFrameAllocator::initialize().
    this->params.allocator->initialize(
        this->params.consumer_id,
        metadata.get_total_nfreq(),
        this->params.time_samples_per_chunk, 
        metadata.beam_ids
    );
    
    // Initialize 'curr_frames' (not lock-protected).

    long nbeams = metadata.nbeams;
    this->curr_frames.resize(2*nbeams);

    for (long ichunk = 0; ichunk < 2; ichunk++) {
        for (long ibeam = 0; ibeam < nbeams; ibeam++) {
            auto frame = params.allocator->get_frame(params.consumer_id);
            xassert(frame->time_chunk_index == ichunk);
            xassert(frame->beam_id == metadata.beam_ids.at(ibeam));
            this->curr_frames[ichunk*nbeams + ibeam] = frame;
        }
    }

    // Receive Peers from reader thread (via assembler_peer_queue),
    // and call this->_process_data().

    unique_lock<std::mutex> lock(mutex);

    for (;;) {
        if (is_stopped)
            return;
        
        if (assembler_peer_queue.empty()) {
            cv.wait(lock);
            continue;  // back to top of loop, with lock held.
        }

        shared_ptr<Peer> peer = assembler_peer_queue.front();
        assembler_peer_queue.pop_front();
        lock.unlock();

        this->_process_data(peer);
        
        // Back to top of loop with lock held.
        lock.lock();
    }
}


void Receiver::_process_data(const shared_ptr<Peer> &peer)
{
    const long *freq_channels = &peer->metadata.freq_channels[0];
    long nfreq = peer->metadata.freq_channels.size();
    long nbeams = peer->metadata.nbeams;
    long nt_chunk = params.time_samples_per_chunk;
    long rb_capacity = peer->rb_capacity;
    long bmc = peer->bytes_per_minichunk;

    xassert(bmc > 0);
    xassert(rb_capacity > 0);

    // Note peer->lock here, not this->lock.
    unique_lock<std::mutex> plock(peer->mutex);
    long rb_start = peer->rb_start;
    long rb_end = peer->rb_end;
    plock.unlock();

    // Loop over minichunks (256 time samples).
    while (rb_end >= rb_start + bmc) {
        long rb_pos = rb_start % rb_capacity;
        const char *src = peer->rb_buf.data + rb_pos;
        xassert(rb_end >= rb_start + bmc);
        xassert(rb_capacity >= rb_pos + bmc);

        long imc = rb_start / bmc;
        xassert(rb_start == imc * bmc);

        long ichunk = imc / peer->minichunks_per_chunk;
        imc -= ichunk * peer->minichunks_per_chunk;

        // Note: this->curr_base_chunk and this->curr_frame are not lock-protected.
        // (These members are only accessed by the assembler thread.)

        if (ichunk < curr_base_chunk) {
            // Target chunk is no longer in buffer (peer is running slow).
            // In this case, we silently drop the data.
            goto minichunk_done;  // hmmm
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
                // Note this->lock (protects completed_frames) here, not peer->lock
                unique_lock<std::mutex> rlock(mutex);
                _throw_if_stopped("Receiver::_process_data");
                this->completed_frames.push(std::move(this->curr_frames[i]));
                rlock.unlock();
                this->cv.notify_all();
                
                // Replace with new frame from allocator.
                shared_ptr<AssembledFrame> frame = params.allocator->get_frame(params.consumer_id);
                xassert(frame->time_chunk_index == ichunk);
                xassert(frame->beam_id == metadata.beam_ids.at(b));
                this->curr_frames[i] = std::move(frame);
            }

            this->curr_base_chunk++;
        }

        // The rest of _process_data() copies one minichunk of data
        // (all beams, per-sender frequency subset, 256 time samples)
        // into the AssembledFrames.

        for (long b = 0; b < nbeams; b++) {
            long i = (ichunk & 1) * nbeams + b;
            shared_ptr<AssembledFrame> frame = this->curr_frames[i];
            xassert(frame);

            // The destination "base" pointer includes a time offset
            // (128 bytes per minichunk), but no (beam, frequency) offset.
            char *dst_base = (char *)frame->data.data + (imc << 7);

            for (long ifreq = 0; ifreq < nfreq; ifreq++) {
                long freq_index = freq_channels[ifreq];
                char *dst = dst_base + freq_index * (nt_chunk >> 1);
                memcpy(dst, src, 128);
                src += 128;
            }
        }

    minichunk_done:
        plock.lock();
        xassert(peer->rb_start == rb_start);
        peer->rb_start += bmc;
        rb_start = peer->rb_start;
        rb_end = peer->rb_end;
        plock.unlock();
    }
}


}  // namespace pirate
