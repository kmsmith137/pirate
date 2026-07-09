#include "../include/pirate/Receiver.hpp"
#include "../include/pirate/AssembledFrame.hpp"
#include "../include/pirate/inlines.hpp"  // xdiv()

#include <chrono>
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
// Protocol constants (v2). See notes/network_protocol.md.
//
// magic_v2 is used twice: as the 4-byte handshake magic at the start of each
// TCP connection (read by _read_ini), and as the per-minichunk magic at the
// start of every minichunk's 12-byte header (read by _process_data).


static constexpr uint32_t magic_v2 = 0xf4bf4b02;
static constexpr long minichunk_header_nbytes = 12;   // uint32 magic + uint64 seq
static constexpr long handshake_nbytes = 12;          // uint32 magic + uint32 flags + uint32 yaml_len

// v2 handshake flag bits. Currently one defined:
//   FLAG_ACK: server sends a 1-byte ack per minichunk back to the sender.
//     For testing/debugging only.
// Any flag bit set in the handshake's flags field but not in
// flags_supported_mask causes _read_ini to throw (fail-fast on
// protocol drift).
static constexpr uint32_t FLAG_ACK = 0x1;
static constexpr uint32_t flags_supported_mask = FLAG_ACK;


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

    // First 12 bytes (read during State::ReadIni):
    //   bytes 0-3:  uint32 magic
    //   bytes 4-7:  uint32 flags
    //   bytes 8-11: uint32 yaml_len
    char ini_buf[handshake_nbytes];
    long ini_nbytes = 0;  // bytes read so far

    // Parsed handshake flags. Written by the reader thread in _read_ini
    // (single thread until the peer transitions to ReadData); after that
    // the assembler thread tests (flags & FLAG_ACK) inline in _process_data.
    // Publication to the assembler happens via the enqueue under
    // Receiver::mutex (same pattern as other ReadYaml->ReadData state).
    uint32_t flags = 0;

    // These members are initialized when state ReadIni -> ReadYaml.
    Array<char> yaml_buf;         // length (yaml_string_len + 1)
    long yaml_nbytes = 0;         // bytes read so far (0 <= yaml_nbytes < yaml_string_len)
    long yaml_string_len = 0;     // total bytes expected from sender

    // All following members are initialized when state ReadYaml -> ReadData.
    // metadata: MEANINGFUL freq_channels here -- this Peer represents one
    // specific X-engine sender, and freq_channels is the channel subset it
    // sends. (Contrast with the allocator's canonical metadata, which is
    // frequency-scrubbed; see AssembledFrameAllocator::metadata.)
    XEngineMetadata metadata;
    long bytes_per_minichunk = 0;   // = 12 + nbeams * nfreq * 128 (v2)
    long minichunks_per_chunk = 0;  // = allocator->time_samples_per_chunk / 256

    // Minichunk-sequence tracking. Touched only by the assembler thread (in
    // _process_data); no lock needed.
    //
    // next_expected_imc: smallest minichunk index (mc_seq / seq_per_minichunk)
    //   the assembler will accept on the next minichunk. Initialized to 0;
    //   bumped to (imc + 1) after each accepted minichunk. Strictly monotonic
    //   -- senders cannot rewind the stream, but may skip ahead (NOTE 1).
    long next_expected_imc = 0;

    // Reader-thread flag: set to true once the first 12-byte minichunk
    // header has been observed on this peer's TCP stream and forwarded to
    // AssembledFrameAllocator::initialize_initial_chunk(). Touched only by
    // the reader thread; no lock needed.
    bool reader_seen_first_mc_header = false;

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


// parse_ip_address() is defined in network_utils.hpp.


Receiver::Receiver(const Params &p) : params(p)
{
    parse_ip_address(params.address, this->ip_addr, this->tcp_port);

    xassert(params.allocator);
    xassert(params.consumer_id >= 0);
    // time_samples_per_chunk lives on the allocator (validated > 0 at
    // allocator construction). The divisibility-by-256 check is specific
    // to the network protocol's minichunk size, so we enforce it here at
    // the Receiver rather than at the allocator.
    xassert_divisible(params.allocator->time_samples_per_chunk, 256);
}


void Receiver::start()
{
    unique_lock<std::mutex> lock(mutex);

    if (is_started)
        throw runtime_error("Receiver::start() called twice");

    // Rethrows the saved error if stop(e) was called, else throws a generic
    // "called on stopped instance" message.
    _throw_if_stopped("Receiver::start()");

    is_started = true;
    lock.unlock();

    // Spawn worker threads.
    //
    // If thread creation fails partway (e.g. std::system_error), stop the
    // already-spawned threads before rethrowing; otherwise they would keep
    // running on a not-stopped object. (The destructor also stops+joins, but
    // a caller that catches the exception should see a stopped object.)
    try {
        listener_thread = std::thread(&Receiver::listener_main, this);
        reader_thread = std::thread(&Receiver::reader_main, this);
        assembler_thread = std::thread(&Receiver::assembler_main, this);
    } catch (...) {
        this->stop(std::current_exception());
        throw;
    }
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

    // Forward 'e' so that threads blocked in allocator entry points rethrow
    // the root cause (see "Error reporting" in notes/stoppable_class.md).
    params.allocator->stop(e);
    cv.notify_all();
}


void Receiver::_throw_if_stopped(const char *method_name) const
{
    if (error)
        std::rethrow_exception(error);

    if (is_stopped)
        throw runtime_error(string(method_name) + " called on stopped instance");
}


// Entry point: retrieve an assembled frame set (one time chunk, all beams)
// from the queue (blocking).
shared_ptr<AssembledFrameSet> Receiver::get_frame_set()
{
    unique_lock<std::mutex> lock(mutex);

    for (;;) {
        _throw_if_stopped("Receiver::get_frame_set");

        if (!completed_frame_sets.empty()) {
            shared_ptr<AssembledFrameSet> set = std::move(completed_frame_sets.front());
            completed_frame_sets.pop();
            return set;
        }

        cv.wait(lock);
    }
}


// Entry point: wait until the listener thread has bound the listening socket.
// Returns true if now listening, false on timeout; throws if stopped first.
// See header comment for details.
bool Receiver::wait_until_listening(double timeout_sec)
{
    unique_lock<std::mutex> lock(mutex);

    // Predicate: wake on either "listening" or "stopped" (the throw happens
    // outside the wait). Checked under the lock.
    auto ready = [this] { return is_listening || is_stopped; };

    if (timeout_sec < 0.0)
        cv.wait(lock, ready);
    else
        cv.wait_for(lock, std::chrono::duration<double>(timeout_sec), ready);

    _throw_if_stopped("Receiver::wait_until_listening");
    return is_listening;
}


// Entry point: schedule eviction of all chunks with chunk_index <= evicted_chunk.
// Asynchronous: sets the target and wakes the assembler. See header comment for
// details.
//
// Note: the parameter 'evicted_chunk' shadows the member of the same name.
// Inside this function we refer to the member via 'this->evicted_chunk'.
void Receiver::evict(long evicted_chunk)
{
    unique_lock<std::mutex> lock(mutex);

    if (is_stopped)
        return;  // tolerate post-stop calls

    if (evicted_chunk > this->evicted_chunk) {
        this->evicted_chunk = evicted_chunk;
        lock.unlock();
        cv.notify_all();
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

    // Announce that a client's connect() will now succeed (see wait_until_listening()).
    {
        lock_guard<std::mutex> lock(mutex);
        is_listening = true;
        cv.notify_all();
    }

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

        // Test-only short-read injection on every peer socket.
        if (params.misbehaving_reads)
            new_socket.set_misbehaving_reads();

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
    xassert(peer->ini_nbytes < handshake_nbytes);

    char *buf = peer->ini_buf + peer->ini_nbytes;
    long bufsize = handshake_nbytes - peer->ini_nbytes;
    long nbytes_read = peer->socket.read(buf, bufsize);

    // If nbytes_read == 0 and !sock.eof, it's just "would block" - do nothing.
    if (nbytes_read <= 0)
        return;

    peer->ini_nbytes += nbytes_read;
    this->nbytes_cumul.fetch_add(nbytes_read);

    if (peer->ini_nbytes < handshake_nbytes)
        return;

    // Parse the 12-byte handshake:
    //   bytes 0-3:  uint32 magic
    //   bytes 4-7:  uint32 flags
    //   bytes 8-11: uint32 yaml_len
    uint32_t magic;
    uint32_t flags;
    int32_t yaml_len;
    memcpy(&magic,    peer->ini_buf,     4);
    memcpy(&flags,    peer->ini_buf + 4, 4);
    memcpy(&yaml_len, peer->ini_buf + 8, 4);

    if (magic != magic_v2) {
        stringstream ss;
        ss << "Receiver::Peer: invalid handshake magic 0x" << hex << magic
            << " (expected 0x" << magic_v2 << ")";
        throw runtime_error(ss.str());
    }

    // Reject any flag bits we don't understand. Forward-compat / fail-fast
    // on protocol drift.
    if (flags & ~flags_supported_mask) {
        stringstream ss;
        ss << "Receiver::Peer: handshake flags 0x" << hex << flags
           << " contain unknown bits 0x" << (flags & ~flags_supported_mask)
           << " (supported mask: 0x" << flags_supported_mask << ")";
        throw runtime_error(ss.str());
    }
    peer->flags = flags;

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
    long nbeams = peer->metadata.get_nbeams();

    xassert(nfreq > 0);
    xassert(nbeams > 0);

    // v2 per-minichunk layout: 12-byte header (uint32 magic + uint64 seq), then
    //   (nbeams, nfreq, 2) float16  = 4*nbeams*nfreq bytes scales_offsets, then
    //   (nbeams, nfreq, 256) int4   = 128*nbeams*nfreq bytes data.
    peer->bytes_per_minichunk = minichunk_header_nbytes
                              + nbeams * nfreq * 4
                              + nbeams * nfreq * 128;
    peer->minichunks_per_chunk = xdiv(params.allocator->time_samples_per_chunk, 256);

    // Receive buffer size is either 64KB or two minichunks, whichever is larger.
    long nmc = max((64*1024) / peer->bytes_per_minichunk, 2L);

    peer->rb_capacity = nmc * peer->bytes_per_minichunk;
    peer->rb_buf = Array<char> ({peer->rb_capacity}, af_uhost);
    peer->state = Peer::State::ReadData;

    // The reader thread (this one) is responsible for handing the parsed
    // YAML to the AssembledFrameAllocator. The allocator's initialize()
    // method does first-vs-subsequent branching internally:
    //   - first call (from any consumer): stores the metadata, projects
    //     away freq_channels, fills nfreq / beam_ids / etc.
    //   - subsequent calls: validates via check_sender_consistency() and
    //     errors if time_samples_per_chunk doesn't match.

    params.allocator->initialize_metadata(peer->metadata);
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
    long updated_rb_end = peer->rb_end;
    long updated_rb_start = peer->rb_start;
    bool enqueue = (updated_rb_end >= updated_rb_start + peer->bytes_per_minichunk);
    plock.unlock();

    // First-minichunk-header notification to the allocator. As soon as the
    // first 12 bytes are buffered we can parse the per-minichunk header and
    // tell the allocator the canonical initial_time_chunk. The assembler
    // hasn't run yet on this peer (it's still blocked on get_metadata() /
    // wait_for_initial_chunk()), so rb_start is guaranteed to be 0 here and
    // the header lives at rb_buf[0..11].
    if (!peer->reader_seen_first_mc_header
        && updated_rb_end >= updated_rb_start + minichunk_header_nbytes)
    {
        xassert(updated_rb_start == 0);
        uint32_t mc_magic;
        uint64_t mc_seq;
        memcpy(&mc_magic, peer->rb_buf.data,     4);
        memcpy(&mc_seq,   peer->rb_buf.data + 4, 8);

        if (mc_magic != magic_v2) {
            stringstream ss;
            ss << "Receiver::Peer: bad first-minichunk magic 0x" << hex << mc_magic
               << " (expected 0x" << magic_v2 << ")";
            throw runtime_error(ss.str());
        }

        long seq_per_mc = 256L * peer->metadata.seq_per_frb_time_sample;
        xassert_divisible(long(mc_seq), seq_per_mc);
        long mc_index = long(mc_seq) / seq_per_mc;
        long received_chunk = mc_index / peer->minichunks_per_chunk;

        params.allocator->initialize_initial_chunk(received_chunk);

        peer->reader_seen_first_mc_header = true;
    }

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
    // Wait for the canonical metadata to be parsed by SOME reader thread
    // (which calls allocator->initialize_metadata() once it has parsed a
    // peer's YAML). The allocator handles the cross-thread / cross-receiver
    // coordination internally. We don't need the metadata ourselves here
    // (the assembler accesses metadata via per-peer state in _process_data,
    // and via the AssembledFrameSet's own metadata field), but we DO need
    // metadata to be initialized before calling get_frame_set() below,
    // which throws if metadata is unset.
    (void) params.allocator->get_metadata(true);

    // Wait for the canonical initial_time_chunk to be established (the
    // first 12 bytes of a minichunk arriving on ANY peer of ANY Receiver
    // calls allocator->initialize_initial_chunk()). The 2-chunk window
    // 'curr_frame_sets' is anchored at this value, so a Receiver that never
    // gets data of its own still has a coherent starting point and is
    // dragged forward by FrbServer's evict() chain.
    long initial_time_chunk = params.allocator->wait_for_initial_chunk();
    this->curr_base_chunk = initial_time_chunk;

    // Initialize 'curr_frame_sets' (not lock-protected). One set per live
    // chunk; the set's frames[] vector is in beam_ids order with internal
    // (metadata, time_chunk_index) consistency validated by the allocator.
    //
    // Straight slot indexing: curr_frame_sets[k] holds the set for ichunk =
    // curr_base_chunk + k, where k in {0,1}. Slot lookup in _process_data
    // uses (ichunk - curr_base_chunk); rotation in _advance_one_chunk shifts
    // slot 1 -> slot 0 and pulls a fresh set into slot 1.

    for (long offset = 0; offset < 2; offset++) {
        long chunk_idx = initial_time_chunk + offset;
        auto set = params.allocator->get_frame_set(params.consumer_id);
        xassert(set->time_chunk_index == chunk_idx);
        this->curr_frame_sets[offset] = std::move(set);
    }

    // Main loop. We have two kinds of work:
    //  - process peers handed off from the reader thread (via assembler_peer_queue),
    //  - advance curr_base_chunk in response to evict() calls from external threads.
    // The cv.wait predicate covers both, plus the is_stopped exit.

    unique_lock<std::mutex> lock(mutex);

    for (;;) {
        cv.wait(lock, [&]() {
            return is_stopped
                || !assembler_peer_queue.empty()
                || (evicted_chunk >= curr_base_chunk);
        });

        if (is_stopped)
            return;

        // Snapshot this iteration's work under the lock.
        long target = evicted_chunk;
        shared_ptr<Peer> peer;
        if (!assembler_peer_queue.empty()) {
            peer = assembler_peer_queue.front();
            assembler_peer_queue.pop_front();
        }
        lock.unlock();

        // Eviction first: advance curr_base_chunk past any chunks that
        // evict() has scheduled. Loop condition: while there is still some
        // chunk <= target sitting in curr_frame_sets (i.e. curr_base_chunk has
        // not yet advanced past target), evict one more.
        while (curr_base_chunk <= target)
            _advance_one_chunk();

        // Then process peer data, if any.
        if (peer)
            this->_process_data(peer);

        // Back to top of loop with lock held.
        lock.lock();
    }
}


void Receiver::_process_data(const shared_ptr<Peer> &peer)
{
    const long *freq_channels = &peer->metadata.freq_channels[0];
    long nfreq = peer->metadata.freq_channels.size();
    long nbeams = peer->metadata.get_nbeams();
    long nt_chunk = params.allocator->time_samples_per_chunk;
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

        xassert_divisible(rb_start, bmc);   // rb-position alignment (defensive)

        // v2 per-minichunk header (12 bytes): uint32 magic + uint64 seq.
        // After the header is parsed and validated, advance `src` past it
        // so the data-copy loop below sees only the (nbeams, nfreq, 256)
        // int4 payload (pirate is little-endian -- no byteswap needed).
        uint32_t mc_magic;
        uint64_t mc_seq;
        memcpy(&mc_magic, src,     4);
        memcpy(&mc_seq,   src + 4, 8);
        src += minichunk_header_nbytes;

        if (mc_magic != magic_v2) {
            stringstream ss;
            ss << "Receiver::Peer: bad per-minichunk magic 0x" << hex << mc_magic
               << " at rb minichunk index " << dec << (rb_start / bmc)
               << " (expected 0x" << hex << magic_v2 << ")";
            throw runtime_error(ss.str());
        }

        // Decode the minichunk index from seq. The seq must lie on a
        // minichunk boundary; the wire spec guarantees this.
        long seq_per_minichunk = 256L * peer->metadata.seq_per_frb_time_sample;
        xassert_divisible(long(mc_seq), seq_per_minichunk);
        long imc = long(mc_seq) / seq_per_minichunk;

        // Strict monotonicity (skips OK -- NOTE 1 -- rewinds not).
        // The NOTE-2 "first seq must be 0" deferral is gone: the canonical
        // initial_time_chunk is set by the reader thread on the first
        // 12-byte header (see _read_data) and propagated to the assembler
        // via allocator->wait_for_initial_chunk().
        // next_expected_imc was set to (prev_imc + 1) after the previous
        // accepted minichunk, so (imc == next_expected_imc) means "no
        // skip" and (imc > next_expected_imc) means "sender skipped some".
        xassert(imc >= peer->next_expected_imc);
        peer->next_expected_imc = imc + 1;

        // Derive chunk index and within-chunk minichunk index. (NOT from
        // rb_start anymore -- the two decouple once skipping is allowed.)
        // After this point 'imc' is the within-chunk minichunk index,
        // matching the v1/v2-strict pattern this block replaces.
        long ichunk = imc / peer->minichunks_per_chunk;
        imc -= ichunk * peer->minichunks_per_chunk;

        // Note: this->curr_base_chunk and this->curr_frame_sets are not lock-protected.
        // (These members are only accessed by the assembler thread.)

        // 'assembled' tracks whether this minichunk's data made it into a
        // curr_frame_sets AssembledFrame (true) or was dropped (false). Used
        // by the FLAG_ACK back-channel at the bottom of the loop.
        bool assembled = false;

        if (ichunk < curr_base_chunk) {
            // Target chunk is no longer in buffer (peer is running slow).
            // In this case, we silently drop the data.
            goto minichunk_done;  // hmmm
        }

        // If the sender skipped one or more entire chunks (NOTE 1), advance
        // the 2-chunk window enough to bring the current minichunk's chunk
        // into the top of the window. Each _advance_one_chunk call pushes
        // an AssembledFrameSet to completed_frame_sets whose frames' data is the
        // default-allocator 0x88 = masked int4 (-8) -- which is exactly
        // what NOTE 1 says missing samples should look like at the output.
        while (ichunk >= curr_base_chunk + 2)
            _advance_one_chunk();

        // The rest of _process_data() copies one minichunk (scales_offsets
        // and int4 data, in that order to match the wire layout) for all
        // beams and the per-sender frequency subset into the AssembledFrames
        // inside curr_frame_sets[ichunk - curr_base_chunk].
        //
        // Wrapped in a block so the local variables don't trip the
        // 'goto minichunk_done' jump above (which is taken when the
        // minichunk is dropped for ichunk < curr_base_chunk).
        {
            long slot = ichunk - curr_base_chunk;
            xassert((slot == 0) || (slot == 1));
            xassert(this->curr_frame_sets[slot]);
            const auto &frames_vec = this->curr_frame_sets[slot]->frames;
            xassert(long(frames_vec.size()) == nbeams);
            long mpc = peer->minichunks_per_chunk;

            // First pass: scales_offsets (4 bytes per (beam, freq)).
            // Frame buffer layout is (nfreq, mpc, 2) float16, so per-freq row
            // stride is (mpc * 4) bytes and per-minichunk slice is 4 bytes.
            for (long b = 0; b < nbeams; b++) {
                // `frame` here is the post-eviction frame for `ichunk`. Only
                // the assembler thread writes to curr_frame_sets, so we don't
                // race with eviction; the slot always holds the current live
                // set for ichunk. (See _advance_one_chunk's invariant comment.)
                shared_ptr<AssembledFrame> frame = frames_vec[b];
                xassert(frame);

                char *so_base = (char *)frame->scales_offsets.data + imc * 4;

                for (long ifreq = 0; ifreq < nfreq; ifreq++) {
                    long freq_index = freq_channels[ifreq];
                    char *dst = so_base + freq_index * (mpc * 4);
                    memcpy(dst, src, 4);
                    src += 4;
                }
            }

            // Second pass: int4 data (128 bytes per (beam, freq)).
            for (long b = 0; b < nbeams; b++) {
                shared_ptr<AssembledFrame> frame = frames_vec[b];

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
            assembled = true;
        }

    minichunk_done:
        plock.lock();
        xassert(peer->rb_start == rb_start);
        peer->rb_start += bmc;
        rb_start = peer->rb_start;
        rb_end = peer->rb_end;
        plock.unlock();

        // If the peer opted into FLAG_ACK in the handshake, send a 1-byte
        // ack per minichunk: 1 = data was assembled (memcpy'd into a frame
        // that will eventually flow to the ringbuf), 0 = received but
        // dropped (e.g. ichunk < curr_base_chunk).
        if (peer->flags & FLAG_ACK) {
            char ack = assembled ? char(1) : char(0);
            this->_send_ack(peer, ack);
        }
    }
}


// -------------------------------------------------------------------------------------------------
//
// _advance_one_chunk: factored out of _process_data, also called by _assembler_main
// to service external evict() requests.
//
// Pull a fresh AssembledFrameSet for the chunk that is about to enter the
// top of the 2-chunk window. Before this call:
//     curr_frame_sets holds chunks [curr_base_chunk, curr_base_chunk+1].
// After this call:
//     curr_frame_sets holds chunks [curr_base_chunk+1, curr_base_chunk+2],
//     curr_base_chunk has been incremented by 1.
//
// CRITICAL INVARIANT: once an AssembledFrameSet has been std::move'd out
// of curr_frame_sets[0] into completed_frame_sets, the assembler thread
// MUST NOT perform any further write (in particular, no memcpy) to any
// frame inside the just-evicted set. After the push, the set's frames may
// be concurrently read by FrbServer workers / gRPC / FileWriter, or reaped
// (data cleared) by the FrbServer reaper thread.
//
// We satisfy this invariant by:
//   (1) std::move into completed_frame_sets clears curr_frame_sets[0]
//       (the source shared_ptr becomes null).
//   (2) The shift curr_frame_sets[0] = std::move(curr_frame_sets[1])
//       immediately replaces slot 0 with the set that was already live in
//       slot 1 (still safe to write to -- not yet evicted).
//   (3) A fresh allocator set replaces the now-empty curr_frame_sets[1].
// After this and curr_base_chunk++, _process_data's slot lookup
// (ichunk - curr_base_chunk) only resolves to the shifted-down or freshly-
// pulled sets, never the just-evicted one. The assembler holds no other
// shared_ptr to the moved-out set (only function-local references inside
// _process_data and _advance_one_chunk, both of which go out of scope
// before the next assembler-main iteration).


void Receiver::_advance_one_chunk()
{
    long ichunk_new = curr_base_chunk + 2;

    // Step 1: transfer the evicted set (chunk curr_base_chunk, always in
    // slot 0 under straight indexing) to the completed-set queue. After this
    // push the assembler thread will not write to any frame in the
    // just-evicted set -- see the critical invariant in the comment block
    // above.
    {
        unique_lock<std::mutex> lock(mutex);
        _throw_if_stopped("Receiver::_advance_one_chunk");
        this->completed_frame_sets.push(std::move(this->curr_frame_sets[0]));
    }
    xassert(!this->curr_frame_sets[0]);   // defense in depth: std::move did clear the slot
    this->cv.notify_all();

    // Step 2: shift slot 1 down to slot 0 (the now-vacant slot). The set
    // moving down was already live -- still safe to write to.
    this->curr_frame_sets[0] = std::move(this->curr_frame_sets[1]);
    xassert(!this->curr_frame_sets[1]);

    // Step 3: pull a fresh set for the chunk newly entering the top of the
    // 2-chunk window. allocator->get_frame_set() may block briefly while
    // the allocator's worker thread refills the slab pool.
    shared_ptr<AssembledFrameSet> fresh = params.allocator->get_frame_set(params.consumer_id);
    xassert(fresh->time_chunk_index == ichunk_new);
    this->curr_frame_sets[1] = std::move(fresh);

    this->curr_base_chunk++;
}


// -------------------------------------------------------------------------------------------------
//
// _send_ack: FLAG_ACK back-channel write.
//
// Sends a single byte (0 = dropped, 1 = assembled) on the peer's TCP socket.
// Loops calling Socket::send_with_timeout(10ms) so Receiver::stop() propagates
// promptly. If the client doesn't drain the byte within 1 second total wall-
// clock, throws -- the assembler thread can stall for up to 1 second per peer,
// but that's acceptable because FLAG_ACK is opted into only by test clients.

void Receiver::_send_ack(const shared_ptr<Peer> &peer, char ack_byte)
{
    static constexpr int ack_inner_timeout_ms = 10;
    static constexpr auto ack_total_timeout = std::chrono::seconds(1);

    auto deadline = std::chrono::steady_clock::now() + ack_total_timeout;

    while (true) {
        {
            unique_lock<std::mutex> lock(mutex);
            _throw_if_stopped("Receiver::_send_ack");
        }

        long n = peer->socket.send_with_timeout(&ack_byte, 1, ack_inner_timeout_ms);
        if (n == 1)
            return;
        if (peer->socket.connreset)
            return;  // client gave up reading acks; let the peer drop naturally

        // n == 0 (poll timeout or EAGAIN); retry until deadline.
        if (std::chrono::steady_clock::now() >= deadline) {
            throw runtime_error(
                "Receiver::_send_ack: FLAG_ACK client did not drain ack byte within 1 second");
        }
    }
}


}  // namespace pirate
