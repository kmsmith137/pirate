#include "../include/pirate/FakeXEngine.hpp"
#include "../include/pirate/network_utils.hpp"

#include <cstring>
#include <sstream>
#include <stdexcept>

#include <ksgpu/xassert.hpp>

using namespace std;


namespace pirate {
#if 0
}  // editor auto-indent
#endif



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

    // Spawn worker threads. workers.resize default-constructs nthreads Worker
    // objects (with non-joinable std::thread members); the per-worker thread
    // is move-assigned in the loop below. Workers inherit the vcpu affinity
    // of the caller (the documented constructor contract).
    workers.resize(nthreads);

    try {
        for (int i = 0; i < nthreads; i++)
            workers[i].worker_thread = std::thread(&FakeXEngine::worker_main, this, i);
    } catch (...) {
        // Partial-spawn cleanup: signal stop, then join whichever workers
        // did start, so the destructor's joinable-thread invariant holds
        // even if std::thread construction throws partway through.
        stop(std::current_exception());
        for (auto &w : workers) {
            if (w.worker_thread.joinable())
                w.worker_thread.join();
        }
        throw;
    }
}


FakeXEngine::~FakeXEngine()
{
    this->stop();

    for (auto &w : workers) {
        if (w.worker_thread.joinable())
            w.worker_thread.join();
    }
}


void FakeXEngine::_throw_if_stopped(const char *method_name)
{
    // Caller must hold mutex.
    if (error)
        std::rethrow_exception(error);

    if (is_stopped)
        throw runtime_error(string(method_name) + " called on stopped instance");
}


void FakeXEngine::stop(std::exception_ptr e)
{
    std::lock_guard<std::mutex> lock(mutex);

    if (is_stopped)
        return;

    is_stopped = true;
    error = e;
    cv.notify_all();
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


bool FakeXEngine::_send_all(Socket &sock, const void *buf, long nbytes)
{
    const char *ptr = static_cast<const char *>(buf);
    long pos = 0;

    while (pos < nbytes) {
        // Check if stopped.
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (is_stopped)
                return false;
        }

        // Try to send with short timeout.
        long n = sock.send_with_timeout(ptr + pos, nbytes - pos, send_timeout_ms);

        if (sock.connreset) {
            stop();  // wake up other workers
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

    std::unique_lock<std::mutex> lock(mutex);
    _throw_if_stopped("FakeXEngine::send_junk");

    Command cmd;
    cmd.kind = Command::Kind::SEND_JUNK;
    cmd.minichunk_index = minichunk_index;
    workers[worker_id].command_queue.push_back(cmd);

    lock.unlock();
    // Single shared cv: notify_all wakes every worker (only the target
    // one has new work). This is wasteful at high nthreads; a future
    // commit splits to per-worker cv to fix it.
    cv.notify_all();
}


void FakeXEngine::wait_for_send(long worker_id, long minichunk_index)
{
    if (worker_id < 0 || worker_id >= long(nthreads)) {
        stringstream ss;
        ss << "FakeXEngine::wait_for_send: worker_id=" << worker_id
           << " out of range [0, " << nthreads << ")";
        throw runtime_error(ss.str());
    }

    std::unique_lock<std::mutex> lock(mutex);
    for (;;) {
        _throw_if_stopped("FakeXEngine::wait_for_send");
        if (workers[worker_id].last_minichunk_sent >= minichunk_index)
            return;
        cv.wait(lock);
    }
}


// -------------------------------------------------------------------------------------------------
//
// Worker main loop. Drained by external controller thread(s) via send_junk().


void FakeXEngine::_worker_main(int thread_id)
{
    Worker &w = workers[thread_id];

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
            std::unique_lock<std::mutex> lock(mutex);
            while (!is_stopped && w.command_queue.empty())
                cv.wait(lock);
            if (is_stopped) return;

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

            if (!_send_all(sock, header_buf.data(), header_buf.size()))
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

        if (!_send_all(sock, minichunk_buf.data(), mc_nbytes))
            return;

        // Publish last_minichunk_sent and wake any wait_for_send waiters.
        {
            std::lock_guard<std::mutex> lock(mutex);
            w.last_minichunk_sent = cmd.minichunk_index;
        }
        cv.notify_all();
    }
}


}  // namespace pirate
