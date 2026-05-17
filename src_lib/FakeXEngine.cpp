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
// send_junk() / wait_for_send() - external-thread entry points.


void FakeXEngine::send_junk(long worker_id, long minichunk_index)
{
    if (worker_id < 0 || worker_id >= long(nthreads)) {
        stringstream ss;
        ss << "FakeXEngine::send_junk: worker_id=" << worker_id
           << " out of range [0, " << nthreads << ")";
        throw runtime_error(ss.str());
    }
    xassert_ge(minichunk_index, 0L);

    Worker &w = *workers[worker_id];

    std::unique_lock<std::mutex> lock(w.mutex);
    _throw_if_stopped(w, "FakeXEngine::send_junk");

    Command cmd;
    cmd.kind = Command::Kind::SEND_JUNK;
    cmd.minichunk_index = minichunk_index;
    w.command_queue.push_back(cmd);

    lock.unlock();
    w.cv.notify_all();   // wakes only this worker
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
// Worker main loop. Drained by external controller thread(s) via send_junk().


void FakeXEngine::_worker_main(int thread_id)
{
    Worker &w = *workers[thread_id];

    // -- One-time setup (constants for the lifetime of this worker thread). --

    // Per-worker metadata (with the round-robin freq-channel subset for this thread).
    XEngineMetadata worker_xmd = make_worker_metadata(thread_id);
    long nfreq = worker_xmd.freq_channels.size();
    long nbeams = worker_xmd.get_nbeams();

    // Each v2 minichunk is a 12-byte header (uint32 magic + uint64 seq)
    // followed by an (nbeams, nfreq, 256) int4 data array (packed 2 per
    // byte = nbeams*nfreq*128 bytes). The data is "junk" -- all zeros --
    // and stays that way; only the seq field is rewritten per minichunk.
    // See notes/network_protocol.md.
    static constexpr long mc_header_nbytes = 12;
    long data_nbytes = nbeams * nfreq * 128;
    long mc_nbytes = mc_header_nbytes + data_nbytes;
    vector<char> minichunk_buf(mc_nbytes, 0);

    {
        uint32_t mc_magic = protocol_magic;
        std::memcpy(minichunk_buf.data(), &mc_magic, 4);
    }

    // Protocol header buffer (sent once on the first SEND_JUNK):
    //   uint32 magic + uint32 flags (always 0) + uint32 yaml_len + padded yaml.
    string yaml_str = worker_xmd.to_yaml_string();
    long str_len = yaml_str.size() + 1;           // +1 for the null terminator
    long padded_len = ((str_len + 3) / 4) * 4;    // 4-byte align

    vector<char> header_buf(12 + padded_len, '\0');
    {
        uint32_t magic = protocol_magic;
        uint32_t flags = 0;
        uint32_t len32 = static_cast<uint32_t>(padded_len);
        std::memcpy(header_buf.data() + 0, &magic, 4);
        std::memcpy(header_buf.data() + 4, &flags, 4);
        std::memcpy(header_buf.data() + 8, &len32, 4);
        std::memcpy(header_buf.data() + 12, yaml_str.data(), yaml_str.size());
    }

    // Parsed destination (ip:port). Round-robin: threads share IPs cyclically.
    string ip_addr;
    uint16_t port;
    parse_ip_address(ip_addrs[thread_id % ip_addrs.size()], ip_addr, port);

    // TCP socket -- opened lazily on the first SEND_JUNK. Socket is
    // default-constructible and move-only, so we hold one slot and
    // overwrite it once on first send.
    Socket sock;
    bool connected = false;

    // -- Main command-processing loop. --
    for (;;) {
        Command cmd;
        long prev_minichunk;

        {
            std::unique_lock<std::mutex> lock(w.mutex);
            while (!w.is_stopped && w.command_queue.empty())
                w.cv.wait(lock);
            if (w.is_stopped) return;

            cmd = w.command_queue.front();
            w.command_queue.pop_front();
            prev_minichunk = w.last_minichunk_sent;
        }

        if (cmd.kind != Command::Kind::SEND_JUNK) {
            // Defensive -- UNINITIALIZED Commands should never be enqueued.
            stringstream ss;
            ss << "FakeXEngine worker " << thread_id
               << ": got Command with kind=" << uint32_t(cmd.kind)
               << " (expected SEND_JUNK)";
            throw runtime_error(ss.str());
        }

        if (!connected) {
            // First SEND_JUNK on this worker: open the TCP connection and
            // send the protocol header. connect() may throw on
            // ECONNREFUSED etc.; the wrapping worker_main catches and
            // calls stop() which surfaces the failure through the
            // FakeXEngine's normal exception path.
            xassert_eq(prev_minichunk, -1L);
            sock = Socket(PF_INET, SOCK_STREAM);
            sock.connect(ip_addr, port);
            connected = true;

            if (!_send_all(w, sock, header_buf.data(), header_buf.size()))
                return;
        } else {
            // Subsequent SEND_JUNK: strict monotonic advance by 1.
            xassert_eq(cmd.minichunk_index, prev_minichunk + 1L);
        }

        // Stamp the wire-seq for this minichunk and send. (The first
        // SEND_JUNK falls through here too -- it sends the minichunk for
        // whatever minichunk_index the controller chose, typically 0 or
        // the initial_time_chunk * minichunks_per_chunk offset for NOTE-2
        // tests.)
        uint64_t mc_seq = uint64_t(cmd.minichunk_index) * 256ULL
                        * uint64_t(worker_xmd.seq_per_frb_time_sample);
        std::memcpy(minichunk_buf.data() + 4, &mc_seq, 8);

        if (!_send_all(w, sock, minichunk_buf.data(), mc_nbytes))
            return;

        // Publish last_minichunk_sent and wake any wait_for_send waiters
        // for this worker. (No cross-worker notify is needed -- each
        // wait_for_send is bound to a specific worker_id and waits on
        // that worker's own cv.)
        {
            std::lock_guard<std::mutex> lock(w.mutex);
            w.last_minichunk_sent = cmd.minichunk_index;
        }
        w.cv.notify_all();
    }
}


}  // namespace pirate
