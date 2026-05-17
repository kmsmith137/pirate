#include "../include/pirate/FakeXEngine.hpp"
#include "../include/pirate/network_utils.hpp"

#include <cstring>
#include <memory>
#include <sstream>
#include <stdexcept>

#include <ksgpu/xassert.hpp>

using namespace std;


namespace pirate {
#if 0
}  // editor auto-indent
#endif


// File-scope helper: sets w.is_stopped, w.error, notifies w.cv. Idempotent
// (no-op if w.is_stopped is already true). Used by FakeXEngine::stop()'s
// per-worker sweep; never called from anywhere else.
static void _stop_one_worker(FakeXEngine::Worker &w, std::exception_ptr e)
{
    {
        std::lock_guard<std::mutex> lock(w.mutex);
        if (w.is_stopped) return;
        w.is_stopped = true;
        w.error = e;
    }
    w.cv.notify_all();
}


FakeXEngine::FakeXEngine(const XEngineMetadata &xmd_, const std::vector<std::string> &ip_addrs_, int nthreads_) :
    xmd(xmd_),
    ip_addrs(ip_addrs_),
    nthreads(nthreads_)
{
    if (ip_addrs.empty())
        throw runtime_error("FakeXEngine: ip_addrs is empty");
    xassert(nthreads > 0);

    long naddrs = ip_addrs.size();
    if (nthreads % naddrs != 0) {
        stringstream ss;
        ss << "FakeXEngine: nthreads=" << nthreads
           << " is not a multiple of ip_addrs.size()=" << naddrs;
        throw runtime_error(ss.str());
    }

    // Validate all addresses (parse early to catch errors before threads spawn).
    for (const auto &addr : ip_addrs) {
        string ip;
        uint16_t port;
        parse_ip_address(addr, ip, port);
    }

    // Validate that XEngineMetadata has enough frequency channels for all threads.
    long total_nfreq = xmd.get_total_nfreq();
    if (total_nfreq < nthreads) {
        stringstream ss;
        ss << "FakeXEngine: nthreads=" << nthreads
           << " but total frequency channels=" << total_nfreq;
        throw runtime_error(ss.str());
    }

    xmd.validate();

    // Allocate Worker objects first (heap-allocated, stable address --
    // Worker embeds std::mutex / std::condition_variable, which are
    // neither copyable nor movable). Then spawn the worker threads.
    // Workers inherit the vcpu affinity of the caller (the documented
    // constructor contract).
    workers.reserve(nthreads);
    for (int i = 0; i < nthreads; i++)
        workers.push_back(std::make_unique<Worker>());

    try {
        for (int i = 0; i < nthreads; i++)
            workers[i]->worker_thread = std::thread(&FakeXEngine::worker_main, this, i);
    } catch (...) {
        // Partial-spawn cleanup: signal stop so any workers that did start
        // exit promptly, then join whichever ones are joinable. stop()'s
        // atomic CAS ensures only the first call sweeps; the destructor
        // (which also calls stop()) will then be a no-op for the atomic
        // and the per-worker sweep is idempotent.
        stop(std::current_exception());
        for (auto &wp : workers) {
            if (wp->worker_thread.joinable())
                wp->worker_thread.join();
        }
        throw;
    }
}


FakeXEngine::~FakeXEngine()
{
    this->stop();

    for (auto &wp : workers) {
        if (wp->worker_thread.joinable())
            wp->worker_thread.join();
    }
}


void FakeXEngine::_throw_if_stopped(Worker &w, const char *method_name)
{
    // Caller must hold w.mutex.
    if (w.error)
        std::rethrow_exception(w.error);

    if (w.is_stopped)
        throw runtime_error(string(method_name) + " called on stopped instance");
}


void FakeXEngine::stop(std::exception_ptr e)
{
    // Re-entry guard: only the first caller's compare-exchange wins.
    // Subsequent concurrent or later callers see the cache already true
    // and return immediately, so 'e' from the first caller is the one
    // propagated to every Worker::error.
    bool expected = false;
    if (!is_stopped_cache.compare_exchange_strong(expected, true))
        return;

    // Per-worker sweep. Each iteration takes one Worker's mutex briefly
    // to set is_stopped+error and notify its cv. No nested locks; no
    // possibility of deadlock with anything else in this class.
    for (auto &wp : workers)
        _stop_one_worker(*wp, e);
}


XEngineMetadata FakeXEngine::make_worker_metadata(int thread_id) const
{
    xassert(thread_id >= 0);
    xassert(thread_id < nthreads);

    long total_nfreq = xmd.get_total_nfreq();

    // Assign frequency channels round-robin to this thread.
    // Thread 'thread_id' gets channels: thread_id, thread_id + nthreads, thread_id + 2*nthreads, ...
    vector<long> freq_channels;
    for (long ch = thread_id; ch < total_nfreq; ch += nthreads)
        freq_channels.push_back(ch);

    XEngineMetadata ret = xmd;
    ret.freq_channels = std::move(freq_channels);
    return ret;
}


void FakeXEngine::worker_main(int thread_id)
{
    try {
        _worker_main(thread_id);
    } catch (...) {
        stop(std::current_exception());
    }
}


bool FakeXEngine::_send_all(Worker &w, Socket &sock, const void *buf, long nbytes)
{
    const char *ptr = static_cast<const char *>(buf);
    long pos = 0;

    while (pos < nbytes) {
        // Check this worker's stopped flag periodically (bounded by
        // send_timeout_ms = 10ms granularity). Locks only w.mutex.
        {
            std::lock_guard<std::mutex> lock(w.mutex);
            if (w.is_stopped) return false;
        }

        long n = sock.send_with_timeout(ptr + pos, nbytes - pos, send_timeout_ms);

        if (sock.connreset) {
            // Receiver hung up. Surface to the whole FakeXEngine so
            // sibling workers exit too. stop()'s atomic CAS makes
            // repeated connreset->stop() calls cheap (only the first
            // one actually sweeps).
            stop();
            return false;
        }

        pos += n;
    }

    return true;
}


// -------------------------------------------------------------------------------------------------
//
// External-thread entry points: send_junk(), skip_minichunk(),
// send_minichunk(), wait_for_send(). The first three share an
// _enqueue() helper that does the worker_id range check + lock +
// throw-if-stopped + push + notify pattern.


void FakeXEngine::_enqueue(long worker_id, Command &&cmd, const char *method_name)
{
    if (worker_id < 0 || worker_id >= long(nthreads)) {
        stringstream ss;
        ss << method_name << ": worker_id=" << worker_id
           << " out of range [0, " << nthreads << ")";
        throw runtime_error(ss.str());
    }

    Worker &w = *workers[worker_id];

    std::unique_lock<std::mutex> lock(w.mutex);
    _throw_if_stopped(w, method_name);
    w.command_queue.push_back(std::move(cmd));
    lock.unlock();
    w.cv.notify_all();   // wakes only this worker
}


void FakeXEngine::send_junk(long worker_id, long minichunk_index)
{
    xassert_ge(minichunk_index, 0L);

    Command cmd;
    cmd.kind = Command::Kind::SEND_JUNK;
    cmd.minichunk_index = minichunk_index;
    _enqueue(worker_id, std::move(cmd), "FakeXEngine::send_junk");
}


void FakeXEngine::skip_minichunk(long worker_id, long minichunk_index)
{
    xassert_ge(minichunk_index, 0L);

    Command cmd;
    cmd.kind = Command::Kind::SKIP_MINICHUNK;
    cmd.minichunk_index = minichunk_index;
    _enqueue(worker_id, std::move(cmd), "FakeXEngine::skip_minichunk");
}


void FakeXEngine::send_minichunk(long worker_id, long minichunk_index,
                                 std::shared_ptr<AssembledFrameSet> frame_set)
{
    xassert_ge(minichunk_index, 0L);
    if (!frame_set)
        throw runtime_error("FakeXEngine::send_minichunk: frame_set is null");

    Command cmd;
    cmd.kind = Command::Kind::SEND_MINICHUNK;
    cmd.minichunk_index = minichunk_index;
    cmd.frame_set = std::move(frame_set);
    _enqueue(worker_id, std::move(cmd), "FakeXEngine::send_minichunk");
}


void FakeXEngine::disconnect(long worker_id)
{
    Command cmd;
    cmd.kind = Command::Kind::DISCONNECT;
    // minichunk_index stays -1; frame_set stays null. Neither is read
    // by _disconnect.
    _enqueue(worker_id, std::move(cmd), "FakeXEngine::disconnect");
}


bool FakeXEngine::is_connected(long worker_id) const
{
    if (worker_id < 0 || worker_id >= long(nthreads)) {
        stringstream ss;
        ss << "FakeXEngine::is_connected: worker_id=" << worker_id
           << " out of range [0, " << nthreads << ")";
        throw runtime_error(ss.str());
    }
    // No _throw_if_stopped: this is an informational query, and the
    // last-known per-worker connected state is meaningful even after
    // FakeXEngine::stop(). Same semantics as the is_stopped property.
    return workers[worker_id]->connected.load(std::memory_order_relaxed);
}


void FakeXEngine::wait_for_send(long worker_id, long minichunk_index)
{
    if (worker_id < 0 || worker_id >= long(nthreads)) {
        stringstream ss;
        ss << "FakeXEngine::wait_for_send: worker_id=" << worker_id
           << " out of range [0, " << nthreads << ")";
        throw runtime_error(ss.str());
    }

    Worker &w = *workers[worker_id];

    std::unique_lock<std::mutex> lock(w.mutex);
    for (;;) {
        _throw_if_stopped(w, "FakeXEngine::wait_for_send");
        if (w.last_minichunk_sent >= minichunk_index)
            return;
        w.cv.wait(lock);
    }
}


// -------------------------------------------------------------------------------------------------
//
// _initialize: one-time setup, called from the top of _worker_main.
//
// Populates the worker's per-thread state (xmd, send_buf, header_nbytes,
// mc_nbytes, ip_addr, port). The TCP socket is NOT opened here -- that
// happens lazily on the first SEND_JUNK inside _send_junk().
//
// Combined "send buffer" laid out as:
//
//   [ 12-byte conn header + padded YAML ][ 12-byte mc header + (nbeams,nfreq,256) int4 data ]
//   ^                                    ^
//   send_buf.data()                      send_buf.data() + header_nbytes
//
// The first SEND_JUNK sends the whole buffer in one _send_all() call
// (connection header + first minichunk); subsequent SEND_JUNKs send only
// the minichunk portion. Combining the two saves a _send_all() call on
// the first send and avoids a TCP-level Nagle round-trip between the
// header and the first minichunk.
//
// The data is "junk" -- all zeros -- and stays that way; only the
// minichunk's seq field (at offset header_nbytes + 4) is rewritten per
// SEND_JUNK. See notes/network_protocol.md for the v2 wire format.


void FakeXEngine::_initialize(int thread_id)
{
    Worker &w = *workers[thread_id];

    // Per-worker metadata (round-robin freq-channel subset for this thread).
    w.xmd = make_worker_metadata(thread_id);
    long nfreq = w.xmd.freq_channels.size();
    long nbeams = w.xmd.get_nbeams();

    static constexpr long mc_header_nbytes = 12;
    long data_nbytes = nbeams * nfreq * 128;
    w.mc_nbytes = mc_header_nbytes + data_nbytes;

    string yaml_str = w.xmd.to_yaml_string();
    long str_len = long(yaml_str.size()) + 1;     // +1 for the null terminator
    long padded_len = ((str_len + 3) / 4) * 4;    // 4-byte align
    w.header_nbytes = 12 + padded_len;

    w.send_buf.assign(w.header_nbytes + w.mc_nbytes, 0);

    // Stamp the connection header (one-time fields: protocol magic +
    // flags=0 + yaml_len + padded YAML).
    {
        uint32_t magic = protocol_magic;
        uint32_t flags = 0;
        uint32_t len32 = static_cast<uint32_t>(padded_len);
        std::memcpy(w.send_buf.data() + 0, &magic, 4);
        std::memcpy(w.send_buf.data() + 4, &flags, 4);
        std::memcpy(w.send_buf.data() + 8, &len32, 4);
        std::memcpy(w.send_buf.data() + 12, yaml_str.data(), yaml_str.size());
    }

    // Stamp the per-minichunk magic. Constant for this thread's lifetime;
    // only the seq field at offset (header_nbytes + 4) is rewritten per
    // SEND_JUNK.
    {
        uint32_t mc_magic = protocol_magic;
        std::memcpy(w.send_buf.data() + w.header_nbytes, &mc_magic, 4);
    }

    // Parsed destination (ip:port). Round-robin: threads share IPs cyclically.
    parse_ip_address(ip_addrs[thread_id % ip_addrs.size()], w.ip_addr, w.port);

    // w.sock stays default-constructed (fd == -1); w.connected stays false.
    // _send_junk's first call performs the connect().
}


// -------------------------------------------------------------------------------------------------
//
// _populate_minichunk_buf: gather one minichunk's worth of (beam, freq,
// 256-time-sample) int4 data from an AssembledFrameSet into the
// worker's send_buf. Hot path.
//
// Loop structure: outer over beams, inner over freqs, with
// memcpy(dst, src, 128) per (beam, freq). The 128 is a literal so the
// compiler lowers it to a small fixed sequence of SIMD load/store
// pairs (4x 32-byte on AVX2; 2x 64-byte on AVX-512). The destination
// pointer is contiguous within each beam's slab, which is friendly to
// the hardware prefetcher.
//
// REAPER RACE WARNING (see header): this function reads each frame's
// data buffer WITHOUT taking frame.mutex. The AssembledFrame reaper
// can concurrently zero data.size and release the underlying slab,
// which would make the const-char-* read here a use-after-free. We
// rely on the assumption that the reaper (which lives in FrbServer)
// is NEVER colocated with FakeXEngine in the same process. The
// defensive xassert_gt(frame.data.size, 0) below catches "frame was
// reaped before this call started", but is NOT a guard against an
// actively-running reaper that races us mid-loop. If we ever want to
// colocate FakeXEngine with a reaper, replace the unlocked read with
// a per-beam acquire of frame.mutex.


void FakeXEngine::_populate_minichunk_buf(Worker &w,
                                          const AssembledFrameSet &fset,
                                          long minichunk_index)
{
    // Number of bytes on the wire per (beam, freq) for one minichunk:
    // 256 int4 samples = 128 bytes. Compile-time constant for the
    // memcpy lowering.
    static constexpr long mc_time_bytes = 128;

    long worker_nfreq = long(w.xmd.freq_channels.size());
    long nbeams = fset.nbeams;
    long ntime = fset.ntime;

    // Cheap defensive consistency checks.
    xassert_eq(nbeams, w.xmd.get_nbeams());
    xassert((ntime % 256) == 0);

    long minichunks_per_chunk = ntime / 256;
    long imc_within_chunk = minichunk_index
                          - fset.time_chunk_index * minichunks_per_chunk;
    xassert_ge(imc_within_chunk, 0L);
    xassert_lt(imc_within_chunk, minichunks_per_chunk);

    // Source-side layout: each AssembledFrame's data is int4 with shape
    // (nfreq_total, ntime), packed as (nfreq_total, ntime/2) bytes. Per-
    // freq row stride is ntime/2 bytes. Per-minichunk offset within a
    // freq row is imc_within_chunk * 128 bytes.
    long src_freq_stride = ntime / 2;
    long src_time_offset = imc_within_chunk * mc_time_bytes;

    // Destination-side layout in send_buf: wire data starts at
    // (send_buf + header_nbytes + 12), with shape
    // (nbeams, worker_nfreq, 128 bytes). The outer loop over beams
    // matches that layout, so each beam's destination is a contiguous
    // (worker_nfreq * 128)-byte slab.
    char *dst_data = w.send_buf.data() + w.header_nbytes + 12;

    const long *freq_channels = w.xmd.freq_channels.data();

    for (long ibeam = 0; ibeam < nbeams; ibeam++) {
        const AssembledFrame &frame = *fset.frames[ibeam];

        // Defense-in-depth: a reaped frame has data.size == 0. NOT a
        // safe guard against a concurrently-running reaper -- see the
        // race warning at the top of this function -- but cheap and
        // catches "frame was already reaped before this call started".
        xassert_gt(frame.data.size, 0L);

        const char *frame_data = static_cast<const char *>(frame.data.data);
        char *dst_beam = dst_data + ibeam * worker_nfreq * mc_time_bytes;

        for (long ifreq_local = 0; ifreq_local < worker_nfreq; ifreq_local++) {
            long freq_global = freq_channels[ifreq_local];
            const char *src = frame_data
                            + freq_global * src_freq_stride
                            + src_time_offset;
            char *dst = dst_beam + ifreq_local * mc_time_bytes;
            std::memcpy(dst, src, 128);
        }
    }
}


// -------------------------------------------------------------------------------------------------
//
// _disconnect: handler for DISCONNECT.
//
// Closes the worker's TCP socket (Socket::close() is public, idempotent,
// and resets fd back to -1) and flips the atomic flag. Idempotent --
// repeated DISCONNECTs on an already-disconnected worker are no-ops.
//
// Crucially, does NOT touch w.last_minichunk_sent; the next SEND_* on
// this worker will hit the !w.connected branch in _skip_or_send and
// transparently reopen the connection + re-send the protocol header
// (bundled with the next minichunk in one _send_all). Also does NOT
// call cv.notify_all: nothing in any wait_for_send predicate depends
// on the connected flag.


void FakeXEngine::_disconnect(int thread_id)
{
    Worker &w = *workers[thread_id];

    if (w.connected.load(std::memory_order_relaxed)) {
        w.sock.close();
        w.connected.store(false, std::memory_order_relaxed);
    }
}


// -------------------------------------------------------------------------------------------------
//
// _skip_or_send: handler for SEND_JUNK / SKIP_MINICHUNK / SEND_MINICHUNK.
// See header for contract. Three discriminators:
//
//   - first_command (last_minichunk_sent == -1): the very first
//     state-advancing command on this worker may pick any
//     minichunk_index >= 0 (for NOTE-2 nonzero-initial-chunk tests).
//     Subsequent state-advancing commands must advance by exactly +1.
//     (DISCONNECT is not state-advancing -- it lives in _disconnect.)
//
//   - need_send (kind in {SEND_JUNK, SEND_MINICHUNK}): does this
//     command put bytes on the wire?
//
//   - first_send_after_connect: did this _skip_or_send call open the
//     TCP socket? If so, the conn header rides along in the same
//     _send_all call as the first minichunk. Note this branch is
//     taken both on the very first SEND_* AND after a DISCONNECT (in
//     which case it opens a fresh TCP connection + re-sends the
//     handshake transparently to the controller).


bool FakeXEngine::_skip_or_send(int thread_id, const Command &cmd)
{
    Worker &w = *workers[thread_id];

    // Monotonicity check. Safe to read w.last_minichunk_sent without
    // w.mutex: the worker thread is the sole writer.
    bool first_command = (w.last_minichunk_sent == -1);
    if (!first_command)
        xassert_eq(cmd.minichunk_index, w.last_minichunk_sent + 1L);
    xassert_ge(cmd.minichunk_index, 0L);

    bool need_send = (cmd.kind == Command::Kind::SEND_JUNK ||
                      cmd.kind == Command::Kind::SEND_MINICHUNK);

    if (need_send) {
        // Lazy connect: the first SEND_* opens the socket. SKIP_MINICHUNK
        // never opens the socket -- so a worker that only ever sees
        // SKIPs leaves its TCP connection un-established.
        bool first_send_after_connect = false;
        if (!w.connected.load(std::memory_order_relaxed)) {
            // connect() may throw on ECONNREFUSED etc.; worker_main's
            // catch handler then calls FakeXEngine::stop().
            //
            // This branch is also entered after a DISCONNECT command
            // has flipped w.connected back to false -- in that case
            // the same lazy-connect path opens a fresh TCP connection
            // and re-sends the protocol handshake (since the
            // first_send_after_connect=true branch packs both the
            // connection header and this minichunk into one _send_all).
            w.sock = Socket(PF_INET, SOCK_STREAM);
            w.sock.connect(w.ip_addr, w.port);
            w.connected.store(true, std::memory_order_relaxed);
            first_send_after_connect = true;
        }

        // SEND_MINICHUNK: gather real data into the minichunk_buf.
        // SEND_JUNK: leave the data area as the all-zero initialization
        // from _initialize().
        if (cmd.kind == Command::Kind::SEND_MINICHUNK) {
            // _enqueue() already rejected null frame_set, but
            // defense-in-depth in case a future caller bypasses it.
            xassert(cmd.frame_set);
            _populate_minichunk_buf(w, *cmd.frame_set, cmd.minichunk_index);
        }

        // Stamp the wire-seq for this minichunk.
        char *mc_ptr = w.send_buf.data() + w.header_nbytes;
        uint64_t mc_seq = uint64_t(cmd.minichunk_index) * 256ULL
                        * uint64_t(w.xmd.seq_per_frb_time_sample);
        std::memcpy(mc_ptr + 4, &mc_seq, 8);

        // First send after connect: conn header + minichunk in one shot.
        // Subsequent sends: just the minichunk portion of send_buf.
        const char *send_ptr = first_send_after_connect ? w.send_buf.data() : mc_ptr;
        long send_nbytes = first_send_after_connect ? long(w.send_buf.size()) : w.mc_nbytes;

        if (!_send_all(w, w.sock, send_ptr, send_nbytes))
            return false;
    }
    // SKIP_MINICHUNK falls through here -- no wire activity, just
    // advance state below.

    // Publish last_minichunk_sent and wake any wait_for_send waiters
    // for this worker. (No cross-worker notify is needed -- each
    // wait_for_send is bound to a specific worker_id and waits on that
    // worker's own cv.)
    {
        std::lock_guard<std::mutex> lock(w.mutex);
        w.last_minichunk_sent = cmd.minichunk_index;
    }
    w.cv.notify_all();

    return true;
}


// -------------------------------------------------------------------------------------------------
//
// Worker main loop. Drained by external controller thread(s) via send_junk().


void FakeXEngine::_worker_main(int thread_id)
{
    Worker &w = *workers[thread_id];

    _initialize(thread_id);

    for (;;) {
        Command cmd;
        {
            std::unique_lock<std::mutex> lock(w.mutex);
            while (!w.is_stopped && w.command_queue.empty())
                w.cv.wait(lock);
            if (w.is_stopped) return;

            cmd = w.command_queue.front();
            w.command_queue.pop_front();
        }

        switch (cmd.kind) {
        case Command::Kind::SEND_JUNK:
        case Command::Kind::SKIP_MINICHUNK:
        case Command::Kind::SEND_MINICHUNK:
            if (!_skip_or_send(thread_id, cmd))
                return;
            break;
        case Command::Kind::DISCONNECT:
            _disconnect(thread_id);
            break;
        default: {
            // Defensive -- UNINITIALIZED Commands should never be enqueued.
            stringstream ss;
            ss << "FakeXEngine worker " << thread_id
               << ": got Command with kind=" << uint32_t(cmd.kind)
               << " (expected SEND_JUNK)";
            throw runtime_error(ss.str());
        }
        }
    }
}


}  // namespace pirate
