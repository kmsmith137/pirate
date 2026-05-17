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


// File-scope helper: atomically sets dst = max(dst, src). Portable
// C++17 CAS-loop equivalent of C++26 std::atomic::fetch_max. Used by
// the FLAG_ACK ack-prediction tracking (max_sent_minichunk and
// max_acked_minichunk).
//
// Memory ordering: relaxed throughout. These atomics are running
// counters with no synchronization relationship to other state. The
// happens-before relation that makes the ack-prediction bounds hold
// goes THROUGH the network (worker A sends bytes -> server processes
// -> worker B receives ack), which the C++ memory model doesn't
// capture anyway -- any reasonable ordering works in practice.
static inline void fetch_max(std::atomic<long> &dst, long src)
{
    long expected = dst.load(std::memory_order_relaxed);
    while (expected < src) {
        if (dst.compare_exchange_weak(expected, src,
                                      std::memory_order_relaxed,
                                      std::memory_order_relaxed))
            return;
        // CAS failed; `expected` was updated to the current value of dst.
    }
}


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


FakeXEngine::FakeXEngine(const XEngineMetadata &xmd_, const std::vector<std::string> &ip_addrs_,
                         int nthreads_, long time_samples_per_chunk_, bool flag_ack_) :
    xmd(xmd_),
    ip_addrs(ip_addrs_),
    nthreads(nthreads_),
    time_samples_per_chunk(time_samples_per_chunk_),
    flag_ack(flag_ack_),
    minichunks_per_chunk(time_samples_per_chunk_ / 256),
    num_receivers(long(ip_addrs_.size()))
{
    if (ip_addrs.empty())
        throw runtime_error("FakeXEngine: ip_addrs is empty");
    xassert(nthreads > 0);

    if (time_samples_per_chunk <= 0) {
        stringstream ss;
        ss << "FakeXEngine: time_samples_per_chunk=" << time_samples_per_chunk
           << " must be positive";
        throw runtime_error(ss.str());
    }
    if ((time_samples_per_chunk % 256) != 0) {
        stringstream ss;
        ss << "FakeXEngine: time_samples_per_chunk=" << time_samples_per_chunk
           << " must be a multiple of 256";
        throw runtime_error(ss.str());
    }

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

    // Size the per-Receiver ack high-water-mark vector and seed each
    // entry to -1. Can't use std::vector(n, atomic<long>{-1}) because
    // std::atomic is not copyable; have to default-construct then
    // explicitly store.
    max_acked_minichunk = std::vector<std::atomic<long>>(num_receivers);
    for (long i = 0; i < num_receivers; i++)
        max_acked_minichunk[i].store(-1, std::memory_order_relaxed);

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
// send_minichunk(), disconnect(), wait_until_processed(). All but the
// last share an _enqueue() helper that does the worker_id range
// check + lock + throw-if-stopped + (for state-advancing commands)
// queue-time sequentiality check + push + notify pattern. The
// is_state_advancing argument is passed explicitly per call rather
// than inspected from cmd.kind, so each call site documents its
// intent.


void FakeXEngine::_enqueue(long worker_id, Command &&cmd, bool is_state_advancing,
                           const char *method_name)
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

    // Queue-time sequentiality check for state-advancing commands
    // (SEND_JUNK / SKIP_MINICHUNK / SEND_MINICHUNK). DISCONNECT does
    // not participate -- its minichunk_index is unused.
    if (is_state_advancing) {
        if (w.last_queued_minichunk < 0) {
            // First state-advancing command on this worker. Any
            // minichunk_index >= 0 is allowed (supports NOTE-2-style
            // nonzero-initial-chunk tests). The entry-point
            // xassert_ge already enforced cmd.minichunk_index >= 0.
            w.first_minichunk = cmd.minichunk_index;
        } else if (cmd.minichunk_index != w.last_queued_minichunk + 1L) {
            stringstream ss;
            ss << method_name << ": worker_id=" << worker_id
               << " expected minichunk_index="
               << (w.last_queued_minichunk + 1L)
               << " (strict +1 monotonicity), got "
               << cmd.minichunk_index;
            throw runtime_error(ss.str());
        }
        w.last_queued_minichunk = cmd.minichunk_index;

        // FLAG_ACK back-channel: append STATUS_QUEUED. This
        // push_back may reallocate w.minichunk_status WITH THE LOCK
        // HELD -- normally a performance smell, but acceptable here
        // because FLAG_ACK is testing-only and the worker thread
        // isn't on the latency-critical path. After the push, the
        // invariant
        //   minichunk_status.size() == last_queued_minichunk - first_minichunk + 1
        // holds.
        if (flag_ack)
            w.minichunk_status.push_back(STATUS_QUEUED);
    }

    w.command_queue.push_back(std::move(cmd));

    // The lock has been held continuously from is_stopped check
    // through the queue-time check, the counter updates, and the
    // push -- so the recorded last_queued_minichunk is consistent
    // with the FIFO position of this command in command_queue. Now
    // drop it and wake the worker.
    lock.unlock();
    w.cv.notify_all();   // wakes only this worker
}


void FakeXEngine::send_junk(long worker_id, long minichunk_index)
{
    xassert_ge(minichunk_index, 0L);

    Command cmd;
    cmd.kind = Command::Kind::SEND_JUNK;
    cmd.minichunk_index = minichunk_index;
    _enqueue(worker_id, std::move(cmd), /*is_state_advancing=*/true,
             "FakeXEngine::send_junk");
}


void FakeXEngine::skip_minichunk(long worker_id, long minichunk_index)
{
    xassert_ge(minichunk_index, 0L);

    Command cmd;
    cmd.kind = Command::Kind::SKIP_MINICHUNK;
    cmd.minichunk_index = minichunk_index;
    _enqueue(worker_id, std::move(cmd), /*is_state_advancing=*/true,
             "FakeXEngine::skip_minichunk");
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
    _enqueue(worker_id, std::move(cmd), /*is_state_advancing=*/true,
             "FakeXEngine::send_minichunk");
}


void FakeXEngine::disconnect(long worker_id)
{
    Command cmd;
    cmd.kind = Command::Kind::DISCONNECT;
    // minichunk_index stays -1; frame_set stays null. Neither is read
    // by _disconnect.
    _enqueue(worker_id, std::move(cmd), /*is_state_advancing=*/false,
             "FakeXEngine::disconnect");
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


void FakeXEngine::wait_until_processed(long worker_id, long minichunk_index)
{
    if (worker_id < 0 || worker_id >= long(nthreads)) {
        stringstream ss;
        ss << "FakeXEngine::wait_until_processed: worker_id=" << worker_id
           << " out of range [0, " << nthreads << ")";
        throw runtime_error(ss.str());
    }

    Worker &w = *workers[worker_id];

    std::unique_lock<std::mutex> lock(w.mutex);
    for (;;) {
        _throw_if_stopped(w, "FakeXEngine::wait_until_processed");
        if (w.last_minichunk_processed >= minichunk_index)
            return;
        w.cv.wait(lock);
    }
}


void FakeXEngine::wait_for_acks(long worker_id)
{
    if (!flag_ack)
        throw runtime_error(
            "FakeXEngine::wait_for_acks: FakeXEngine was not constructed"
            " with flag_ack=true (so there are no acks to wait for)");

    Command cmd;
    cmd.kind = Command::Kind::WAIT_FOR_ACKS;
    _enqueue(worker_id, std::move(cmd), /*is_state_advancing=*/false,
             "FakeXEngine::wait_for_acks");
}


void FakeXEngine::synchronize(long worker_id)
{
    if (worker_id < 0 || worker_id >= long(nthreads)) {
        stringstream ss;
        ss << "FakeXEngine::synchronize: worker_id=" << worker_id
           << " out of range [0, " << nthreads << ")";
        throw runtime_error(ss.str());
    }

    // If flag_ack is enabled, enqueue a WAIT_FOR_ACKS first. The
    // worker eventually pops it and calls _read_acks(blocking=true);
    // when that finishes the centralized cv.notify_all in
    // _worker_main fires, which wakes us from cv.wait below.
    if (flag_ack)
        wait_for_acks(worker_id);

    // Wait for the queue to drain. Predicate is (command_queue.empty()
    // && ack_queue.empty()):
    //
    //   - command_queue.empty() is the basic "all commands processed"
    //     barrier. Sensitive to ANY commands in the queue, including
    //     ones enqueued by other threads AFTER we called wait_for_acks
    //     above (synchronize is a "drain everything in the queue"
    //     barrier, not "drain everything as of when I was called").
    //
    //   - ack_queue.empty() is required for the flag_ack case to
    //     close the small window where the worker has popped
    //     WAIT_FOR_ACKS (making command_queue.empty() == true) but
    //     is still inside its blocking _read_acks call (so ack_queue
    //     is not yet empty). Without the second clause, a spurious
    //     cv wakeup in that window would let synchronize observe the
    //     transient empty-but-not-drained state.
    //
    //   - When flag_ack=false, ack_queue is always empty, so the
    //     predicate reduces to command_queue.empty().
    Worker &w = *workers[worker_id];
    std::unique_lock<std::mutex> lock(w.mutex);
    for (;;) {
        _throw_if_stopped(w, "FakeXEngine::synchronize");
        if (w.command_queue.empty() && w.ack_queue.empty())
            return;
        w.cv.wait(lock);
    }
}


unsigned char FakeXEngine::get_minichunk_status(long worker_id, long minichunk_index) const
{
    if (worker_id < 0 || worker_id >= long(nthreads)) {
        stringstream ss;
        ss << "FakeXEngine::get_minichunk_status: worker_id=" << worker_id
           << " out of range [0, " << nthreads << ")";
        throw runtime_error(ss.str());
    }
    if (!flag_ack)
        throw runtime_error(
            "FakeXEngine::get_minichunk_status: FakeXEngine was not"
            " constructed with flag_ack=true");

    const Worker &w = *workers[worker_id];

    std::lock_guard<std::mutex> lock(w.mutex);

    if (w.first_minichunk < 0)
        throw runtime_error(
            "FakeXEngine::get_minichunk_status: no state-advancing"
            " commands have been enqueued yet on this worker");

    long idx = minichunk_index - w.first_minichunk;
    if (idx < 0 || idx >= long(w.minichunk_status.size())) {
        stringstream ss;
        ss << "FakeXEngine::get_minichunk_status: minichunk_index="
           << minichunk_index << " out of range ["
           << w.first_minichunk << ", "
           << (w.first_minichunk + long(w.minichunk_status.size()))
           << ") for worker_id=" << worker_id;
        throw runtime_error(ss.str());
    }
    return w.minichunk_status[idx];
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

    // Round-robin assignment of worker -> Receiver, matching the
    // round-robin ip_addr assignment below. Indexes
    // max_acked_minichunk[] from _skip_or_send / _read_acks.
    w.receiver_id = long(thread_id) % num_receivers;

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
    // flags + yaml_len + padded YAML). 'flags' carries FLAG_ACK if
    // the FakeXEngine was constructed with flag_ack=true.
    {
        uint32_t magic = protocol_magic;
        uint32_t flags = flag_ack ? FLAG_ACK : 0u;
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
// Crucially, does NOT touch w.last_minichunk_processed or
// w.last_queued_minichunk; the next SEND_* on this worker will hit
// the !w.connected branch in _skip_or_send and transparently reopen
// the connection + re-send the protocol header (bundled with the
// next minichunk in one _send_all). Also does NOT call
// cv.notify_all: nothing in any wait_until_processed predicate
// depends on the connected flag.


// -------------------------------------------------------------------------------------------------
//
// _read_acks: drain FLAG_ACK ack bytes from the worker's socket and
// update Worker::minichunk_status / Worker::ack_queue. Runs on the
// worker thread.
//
// blocking=false: a single non-blocking recv attempt (one syscall).
// Returns whatever's available -- zero bytes is fine.
//
// blocking=true: loops with 10ms inner timeouts (so is_stopped is
// observed within ~10ms of being set) and a 1-second outer deadline.
// On any of:
//   - peer EOF before ack_queue is drained
//   - invalid ack byte (not 0 or 1)
//   - the 1-second deadline expires
// _read_acks THROWS rather than recovering gracefully -- worker_main's
// catch handler calls stop(e) and the existing cascade tears down all
// controllers + workers with a descriptive error. This matches the
// "throw on anything wrong" policy for the FLAG_ACK test-only path.


void FakeXEngine::_read_acks(int thread_id, bool blocking)
{
    // ensures that we never call Socket.read() in production (when
    // no acks will be sent)
    xassert(flag_ack);

    Worker &w = *workers[thread_id];

    if (!w.connected.load(std::memory_order_relaxed))
        return;

    // Snapshot the number of acks we need under the lock.
    long need;
    {
        std::lock_guard<std::mutex> lock(w.mutex);
        need = long(w.ack_queue.size());
    }
    if (need == 0) return;

    using clock = std::chrono::steady_clock;
    static constexpr auto blocking_deadline = std::chrono::seconds(1);
    static constexpr int blocking_inner_ms = 10;   // is_stopped check granularity
    auto deadline = clock::now() + blocking_deadline;

    char buf[256];

    while (need > 0) {
        // Stop-responsiveness: short-circuit if a sibling worker
        // or external thread has called stop().
        {
            std::lock_guard<std::mutex> lock(w.mutex);
            if (w.is_stopped) return;
        }

        long max_read = std::min<long>(long(sizeof(buf)), need);
        int inner_timeout = blocking ? blocking_inner_ms : 0;
        long n = w.sock.read_with_timeout(buf, max_read, inner_timeout);

        if (w.sock.eof) {
            // Peer closed before we received all expected acks.
            // Per the FLAG_ACK policy, throw rather than try to
            // recover -- worker_main's catch will call stop(e) and
            // propagate via the existing cascade.
            std::stringstream ss;
            ss << "FakeXEngine::_read_acks: receiver closed connection"
               << " with " << need << " ack(s) still expected";
            throw runtime_error(ss.str());
        }

        if (n == 0) {
            if (!blocking) return;   // single-shot non-blocking: empty is fine
            if (clock::now() >= deadline)
                throw runtime_error(
                    "FakeXEngine::_read_acks: timeout (1s) waiting for FLAG_ACK acks");
            continue;
        }

        // Process the n bytes under the lock.
        {
            std::lock_guard<std::mutex> lock(w.mutex);
            // Defense-in-depth invariants. Should never trip under
            // correct locking discipline (only the worker thread
            // pops ack_queue and changes minichunk_status), but
            // they're cheap and would catch a queue/status drift.
            xassert(long(w.ack_queue.size()) == need);
            xassert(long(w.minichunk_status.size()) ==
                    w.last_queued_minichunk - w.first_minichunk + 1);

            for (long i = 0; i < n; i++) {
                if (w.ack_queue.empty())
                    throw runtime_error(
                        "FakeXEngine::_read_acks: received more ack bytes than expected");
                PendingAck done = w.ack_queue.front();
                unsigned char ack = static_cast<unsigned char>(buf[i]);
                if (ack != STATUS_DROPPED && ack != STATUS_ASSEMBLED) {
                    std::stringstream ss;
                    ss << "FakeXEngine::_read_acks: invalid ack byte "
                       << int(ack) << " from receiver (expected 0 or 1)";
                    throw runtime_error(ss.str());
                }
                long idx = done.minichunk_index - w.first_minichunk;
                xassert_ge(idx, 0L);
                xassert_lt(idx, long(w.minichunk_status.size()));
                xassert_eq(w.minichunk_status[idx], STATUS_SENT);

                // The STATUS_DROPPED == 0 and STATUS_ASSEMBLED == 1
                // numeric coincidence with the wire's ack-byte
                // values lets us assign minichunk_status[idx] = ack
                // directly.
                w.minichunk_status[idx] = ack;
                w.ack_queue.pop_front();

                // Update per-Receiver ack high-water mark: an ack for
                // done.minichunk_index from THIS worker's Receiver B
                // implies B has processed that minichunk.
                fetch_max(max_acked_minichunk[w.receiver_id],
                          done.minichunk_index);

                // Three-way ack-prediction assertion. Bounds the
                // receiver-side curr_base_chunk just BEFORE B
                // processed done.minichunk_index, then checks ack
                // against the prediction:
                //
                //   cbcp_lower = ichunk(done.max_ack_at_submission) - 1
                //     LOWER BOUND on B's cbc at processing time. Since
                //     B had processed done.max_ack_at_submission BEFORE
                //     this worker sent done.minichunk_index (the
                //     snapshot was taken before the send), B's cbc was
                //     >= ichunk(snapshot) - 1 by the time it processed
                //     done.minichunk_index.
                //   cbcp_upper = ichunk(max_sent_minichunk_now) - 1
                //     UPPER BOUND. Every advance to B's cbc (via B's
                //     own _process_data OR via cross-Receiver evict
                //     cascade from FrbServer) traces back to some
                //     sent minichunk m_trigger, and
                //     fetch_max(max_sent_minichunk, m_trigger + shift)
                //     ran BEFORE m_trigger left the kernel. The +mpc
                //     shift on first sends folds the receiver's
                //     initial_time_chunk into max_sent so this bound
                //     covers the init state too (see _skip_or_send).
                //
                // Three cases:
                //   i = ichunk(done.minichunk_index)
                //   i >= cbcp_upper -> B definitely processed with
                //                       cbc <= i -> must be ASSEMBLED.
                //   i < cbcp_lower  -> B definitely processed with
                //                       cbc > i  -> must be DROPPED.
                //   otherwise        -> ambiguous, no assertion.
                long imc = done.minichunk_index;
                long max_sent_now = max_sent_minichunk.load(std::memory_order_relaxed);

                auto ichunk_of = [this](long x) -> long {
                    return (x >= 0) ? (x / minichunks_per_chunk) : -1L;
                };

                long ii         = imc / minichunks_per_chunk;
                long cbcp_lower = ichunk_of(done.max_ack_at_submission) - 1L;
                long cbcp_upper = ichunk_of(max_sent_now) - 1L;

                if (ii >= cbcp_upper) {
                    if (ack != STATUS_ASSEMBLED) {
                        std::stringstream ss;
                        ss << "FakeXEngine::_read_acks: ack-prediction"
                           << " mismatch: minichunk_index=" << imc
                           << " predicted ASSEMBLED (ichunk=" << ii
                           << " >= cbcp_upper=" << cbcp_upper
                           << ", max_ack_at_submission=" << done.max_ack_at_submission
                           << ", max_sent=" << max_sent_now
                           << "), got ack=" << int(ack);
                        throw runtime_error(ss.str());
                    }
                } else if (ii < cbcp_lower) {
                    if (ack != STATUS_DROPPED) {
                        std::stringstream ss;
                        ss << "FakeXEngine::_read_acks: ack-prediction"
                           << " mismatch: minichunk_index=" << imc
                           << " predicted DROPPED (ichunk=" << ii
                           << " < cbcp_lower=" << cbcp_lower
                           << ", max_ack_at_submission=" << done.max_ack_at_submission
                           << ", max_sent=" << max_sent_now
                           << "), got ack=" << int(ack);
                        throw runtime_error(ss.str());
                    }
                }
                // else: ambiguous, no assertion fires.
            }
        }
        need -= n;
    }
}


void FakeXEngine::_disconnect(int thread_id)
{
    Worker &w = *workers[thread_id];

    // Drain remaining acks BEFORE closing. If anything goes wrong
    // during the drain (peer EOF / invalid ack byte / 1-second
    // deadline), _read_acks throws -- worker_main's catch handler
    // calls stop(e) and the cascade tears down. No defensive
    // ack_queue.clear() is needed: either we drained successfully
    // (ack_queue is empty) or we threw (the worker is exiting).
    if (flag_ack)
        _read_acks(thread_id, /*blocking=*/true);

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
//   - first_command (last_minichunk_processed == -1): the very first
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

    // Defense-in-depth monotonicity check. In principle this is
    // redundant with the queue-time sequentiality check that
    // _enqueue has already performed on every state-advancing
    // command (queue-time + FIFO ordering of command_queue together
    // guarantee that the order observed here matches the order
    // submitted), but we keep the processing-time check out of
    // caution -- the cost is negligible, and a violation would
    // immediately surface a queue-time logic bug as a worker-thread
    // exception via the existing stop() cascade.
    //
    // Safe to read w.last_minichunk_processed without w.mutex: the
    // worker thread is the sole writer.
    bool first_command = (w.last_minichunk_processed == -1);
    if (!first_command)
        xassert_eq(cmd.minichunk_index, w.last_minichunk_processed + 1L);
    xassert_ge(cmd.minichunk_index, 0L);

    bool need_send = (cmd.kind == Command::Kind::SEND_JUNK ||
                      cmd.kind == Command::Kind::SEND_MINICHUNK);

    // Snapshot of max_acked_minichunk[receiver_id] taken just before
    // _send_all. Attached to the PendingAck pushed at the bottom of
    // this function. Only set in the need_send branch below; the
    // SKIP_MINICHUNK path leaves it at -1 (which is never read,
    // since SKIP doesn't push to ack_queue).
    long max_ack_at_submission = -1;

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

        // FLAG_ACK ack-prediction bookkeeping. Snapshot the per-Receiver
        // ack high-water mark BEFORE the wire send, then update the
        // global "max sent" mark (also BEFORE the send). Both timings
        // are load-bearing for the bounds used by _read_acks's
        // ack-prediction assertion (see _read_acks for the full
        // assertion logic).
        max_ack_at_submission =
            max_acked_minichunk[w.receiver_id].load(std::memory_order_relaxed);

        // SHIFT TRICK for first send: a worker's very first SEND
        // contributes (cmd.minichunk_index + minichunks_per_chunk) to
        // max_sent_minichunk; subsequent sends contribute just
        // cmd.minichunk_index. Non-obvious; here's why:
        //
        // The receiver-side Receiver::curr_base_chunk (cbc) starts at
        // `initial_time_chunk`, which is set on the first per-minichunk
        // header to arrive at ANY peer of ANY Receiver -- i.e. by the
        // chunk-index of WHICHEVER worker's first SEND lands first. We
        // (the FakeXEngine) don't know which worker that is.
        //
        // The ack-prediction upper bound on cbc_B uses
        //    cbcp_upper = ichunk(max_sent_minichunk) - 1
        // (where ichunk(x) = x / minichunks_per_chunk). For this to be
        // a valid upper bound at the moment the receiver's cbc is just
        // sitting at init (no minichunks processed yet), we need
        //    ichunk(max_sent_minichunk) - 1 >= init.
        // Without the shift, max_sent_minichunk >= m_init_setter and
        // ichunk(max_sent) - 1 = init - 1 < init -- BOUND BREAKS, can
        // produce false-positive ASSEMBLED assertions.
        //
        // With the shift, the init-setter worker's contribution is
        //    m_init_setter + minichunks_per_chunk,
        // so ichunk(max_sent) - 1 >= ichunk(m_init_setter) = init.
        // Bound holds.
        //
        // We apply the shift to EVERY worker's first send because we
        // don't know at runtime which worker won the init race -- the
        // shift is the same whether it's the init-setter or not, and
        // overestimating max_sent_minichunk only makes the upper bound
        // looser (safe direction).
        //
        // first_send_done persists across DISCONNECT/reconnect cycles
        // -- the receiver's initial_time_chunk is set ONCE per server
        // process and never reset, so re-shifting on reconnect would
        // be incorrect bookkeeping (still safe but pointless).
        long shift = w.first_send_done ? 0L : minichunks_per_chunk;
        fetch_max(max_sent_minichunk, cmd.minichunk_index + shift);
        w.first_send_done = true;

        if (!_send_all(w, w.sock, send_ptr, send_nbytes))
            return false;
    }
    // SKIP_MINICHUNK falls through here -- no wire activity, just
    // advance state below. SKIPs do NOT touch max_sent_minichunk,
    // max_ack_at_submission, or first_send_done: nothing went on
    // the wire, so the server cannot have seen this minichunk from
    // us and cannot ever ack it.

    // Determine the new minichunk_status entry for this cmd.
    // Successful SEND_* -> STATUS_SENT (the wire bytes went out;
    // the ack hasn't arrived yet). SKIP_MINICHUNK -> STATUS_SKIPPED.
    unsigned char new_status;
    bool push_ack;
    if (cmd.kind == Command::Kind::SKIP_MINICHUNK) {
        new_status = STATUS_SKIPPED;
        push_ack = false;
    } else {
        new_status = STATUS_SENT;
        push_ack = true;
    }

    // Publish last_minichunk_processed + (if flag_ack) update
    // status / push ack_queue under lock. (cv.notify_all is
    // centralized in _worker_main, fired once per command dispatched
    // -- so wait_until_processed AND synchronize waiters wake
    // together after each command is fully processed.)
    {
        std::lock_guard<std::mutex> lock(w.mutex);
        if (flag_ack) {
            long idx = cmd.minichunk_index - w.first_minichunk;
            xassert_ge(idx, 0L);
            xassert_lt(idx, long(w.minichunk_status.size()));
            xassert_eq(w.minichunk_status[idx], STATUS_QUEUED);
            w.minichunk_status[idx] = new_status;
            if (push_ack)
                w.ack_queue.push_back({cmd.minichunk_index, max_ack_at_submission});
        }
        w.last_minichunk_processed = cmd.minichunk_index;
    }

    // Drain any acks that arrived while we were sending. Cheap
    // single-syscall non-blocking attempt -- if no bytes are
    // queued, the recv returns 0 and we proceed.
    if (flag_ack)
        _read_acks(thread_id, /*blocking=*/false);

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
            for (;;) {
                if (w.is_stopped) return;
                if (!w.command_queue.empty()) break;

                // When ack_queue is empty (always so when flag_ack=
                // false), wait unbounded -- no spurious wakeups.
                // When ack_queue is non-empty, periodically drain
                // acks so we don't let the receiver's send buffer
                // for FLAG_ACK bytes fill up.
                if (w.ack_queue.empty()) {
                    w.cv.wait(lock);
                } else {
                    auto status = w.cv.wait_for(lock, std::chrono::milliseconds(1));
                    if (status == std::cv_status::timeout) {
                        lock.unlock();
                        _read_acks(thread_id, /*blocking=*/false);
                        lock.lock();
                    }
                }
            }

            cmd = w.command_queue.front();
            w.command_queue.pop_front();
        }

        // Dispatch.
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
        case Command::Kind::WAIT_FOR_ACKS:
            // WAIT_FOR_ACKS is enqueued only by wait_for_acks() /
            // synchronize(), both of which check flag_ack first.
            // xassert here as a defense-in-depth invariant.
            xassert(flag_ack);
            _read_acks(thread_id, /*blocking=*/true);
            break;
        default: {
            // Defensive -- UNINITIALIZED Commands should never be enqueued.
            stringstream ss;
            ss << "FakeXEngine worker " << thread_id
               << ": got Command with kind=" << uint32_t(cmd.kind);
            throw runtime_error(ss.str());
        }
        }

        // Centralized notify, one per command dispatched. Wakes both
        // wait_until_processed waiters (whenever last_minichunk_
        // processed advanced) AND synchronize() waiters (whenever
        // command_queue empties or shrinks). Notifying *after* the
        // dispatch returns is what makes synchronize() wait for
        // WAIT_FOR_ACKS's blocking drain to finish before returning.
        w.cv.notify_all();
    }
}


}  // namespace pirate
