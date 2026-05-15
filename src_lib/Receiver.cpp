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
// Protocol constants (v2). See notes/network_protocol.md.
//
// magic_v2 is used twice: as the 4-byte handshake magic at the start of each
// TCP connection (read by _read_ini), and as the per-minichunk magic at the
// start of every minichunk's 12-byte header (read by _process_data).


static constexpr uint32_t magic_v2 = 0xf4bf4b02;
static constexpr long minichunk_header_nbytes = 12;   // uint32 magic + uint64 seq


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
    long bytes_per_minichunk = 0;   // = 12 + nbeams * nfreq * 128 (v2)
    long minichunks_per_chunk = 0;  // = allocator->time_samples_per_chunk / 256

    // Minichunk-sequence tracking. Touched only by the assembler thread (in
    // _process_data); no lock needed.
    //
    // next_expected_imc: smallest minichunk index (mc_seq / seq_per_minichunk)
    //   the assembler will accept on the next minichunk. Initialized to 0;
    //   bumped to (imc + 1) after each accepted minichunk. Strictly monotonic
    //   -- senders cannot rewind the stream, but may skip ahead (NOTE 1).
    //
    // seen_first_minichunk: false until the first minichunk has been processed.
    //   Used to enforce the NOTE-2-deferred "first seq must be 0" invariant
    //   exactly once per connection.
    long next_expected_imc = 0;
    bool seen_first_minichunk = false;

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

    if (magic != magic_v2) {
        stringstream ss;
        ss << "Receiver::Peer: invalid handshake magic 0x" << hex << magic
            << " (expected 0x" << magic_v2 << ")";
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
    long nbeams = peer->metadata.get_nbeams();

    xassert(nfreq > 0);
    xassert(nbeams > 0);

    // v2: each minichunk is prefixed with a 12-byte header
    // (uint32 magic + uint64 seq), then (nbeams, nfreq, 256) int4 = 128*nbeams*nfreq bytes.
    peer->bytes_per_minichunk = minichunk_header_nbytes + nbeams * nfreq * 128;
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
    // It also throws if the allocator is stopped (which is cascaded from
    // Receiver::stop()), so the _throw_if_stopped() below is logically
    // redundant -- we keep it for clarity in the reader-thread path.
    {
        unique_lock<std::mutex> lock(mutex);
        _throw_if_stopped("Receiver::_read_yaml");
    }

    params.allocator->initialize(peer->metadata);
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
    // Wait for the canonical metadata to be parsed by SOME reader thread
    // (which calls allocator->initialize() once it has parsed a peer's
    // YAML). The allocator handles the cross-thread/cross-receiver
    // coordination internally. We cache the shared_ptr in this->metadata
    // so _advance_one_chunk doesn't have to re-acquire the allocator lock.
    this->metadata = params.allocator->get_metadata(true);

    // Initialize 'curr_frames' (not lock-protected).

    long nbeams = this->metadata->get_nbeams();
    this->curr_frames.resize(2*nbeams);

    for (long ichunk = 0; ichunk < 2; ichunk++) {
        for (long ibeam = 0; ibeam < nbeams; ibeam++) {
            auto frame = params.allocator->get_frame(params.consumer_id);
            xassert(frame->time_chunk_index == ichunk);
            xassert(frame->beam_id == this->metadata->beam_ids.at(ibeam));
            this->curr_frames[ichunk*nbeams + ibeam] = frame;
        }
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
        // chunk <= target sitting in curr_frames (i.e. curr_base_chunk has
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

        // NOTE 2 is still deferred -- first minichunk on the connection
        // must have seq = 0 (i.e. imc = 0). When NOTE 2 is implemented,
        // replace this with a per-Peer "first imc" capture and downstream
        // offset.
        if (!peer->seen_first_minichunk) {
            xassert_eq(imc, 0L);
            peer->seen_first_minichunk = true;
        }

        // Strict monotonicity (skips OK -- NOTE 1 -- rewinds not).
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

        // Note: this->curr_base_chunk and this->curr_frame are not lock-protected.
        // (These members are only accessed by the assembler thread.)

        if (ichunk < curr_base_chunk) {
            // Target chunk is no longer in buffer (peer is running slow).
            // In this case, we silently drop the data.
            goto minichunk_done;  // hmmm
        }

        // If the sender skipped one or more entire chunks (NOTE 1), advance
        // the 2-chunk window enough to bring the current minichunk's chunk
        // into the top of the window. Each _advance_one_chunk call pushes
        // nbeams AssembledFrames to completed_frames whose data is the
        // default-allocator 0x88 = masked int4 (-8) -- which is exactly
        // what NOTE 1 says missing samples should look like at the output.
        while (ichunk >= curr_base_chunk + 2)
            _advance_one_chunk();

        // The rest of _process_data() copies one minichunk of data
        // (all beams, per-sender frequency subset, 256 time samples)
        // into the AssembledFrames.

        for (long b = 0; b < nbeams; b++) {
            long i = (ichunk & 1) * nbeams + b;
            shared_ptr<AssembledFrame> frame = this->curr_frames[i];
            xassert(frame);

            // Note: `frame` here is the post-eviction frame for `ichunk`.
            // The critical invariant (see _advance_one_chunk's comment) is
            // that the assembler never memcpy's into an evicted frame -- the
            // eviction logic moves the old frame out of curr_frames[i] before
            // we get here, so this read always returns the current/replacement
            // frame.

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


// -------------------------------------------------------------------------------------------------
//
// _advance_one_chunk: factored out of _process_data, also called by _assembler_main
// to service external evict() requests.
//
// Pull fresh frames for the chunk that is about to enter the top of the 2-chunk
// window. Before this call:
//     curr_frames holds chunks [curr_base_chunk, curr_base_chunk+1].
// After this call:
//     curr_frames holds chunks [curr_base_chunk+1, curr_base_chunk+2],
//     curr_base_chunk has been incremented by 1.
//
// CRITICAL INVARIANT: once a frame has been std::move'd out of curr_frames[i]
// into completed_frames, the assembler thread MUST NOT perform any further
// write (in particular, no memcpy) to that frame's data buffer. After the
// push, the frame may be concurrently read by FrbServer workers / gRPC /
// FileWriter, or reaped (data cleared) by the FrbServer reaper thread.
//
// We satisfy this invariant by:
//   (1) std::move clears curr_frames[i] (the source shared_ptr becomes null).
//   (2) The fresh allocator frame pulled below immediately replaces
//       curr_frames[i]. Any subsequent assembler write reads curr_frames[i]
//       again -- which now refers to the new frame, not the just-evicted one.
// The assembler holds no other shared_ptr to the moved-out frame (only
// function-local references inside _process_data and _advance_one_chunk,
// both of which go out of scope before the next assembler-main iteration).


void Receiver::_advance_one_chunk()
{
    long ichunk_new = curr_base_chunk + 2;
    long nbeams = metadata->get_nbeams();

    for (long b = 0; b < nbeams; b++) {
        long i = (curr_base_chunk & 1) * nbeams + b;

        // Transfer evicted frame to completed_frames queue. After this push
        // the assembler thread will not write to the just-evicted frame
        // again -- see the critical invariant in the comment block above.
        {
            unique_lock<std::mutex> lock(mutex);
            _throw_if_stopped("Receiver::_advance_one_chunk");
            this->completed_frames.push(std::move(this->curr_frames[i]));
        }
        xassert(!this->curr_frames[i]);   // defense in depth: std::move did clear the slot
        this->cv.notify_all();

        // Pull a fresh frame for the chunk newly entering the top of the
        // 2-chunk window. allocator->get_frame() may block briefly while
        // the allocator's worker thread refills the slab pool.
        shared_ptr<AssembledFrame> frame = params.allocator->get_frame(params.consumer_id);
        xassert(frame->time_chunk_index == ichunk_new);
        xassert(frame->beam_id == metadata->beam_ids.at(b));
        this->curr_frames[i] = std::move(frame);
    }

    this->curr_base_chunk++;
}


}  // namespace pirate
