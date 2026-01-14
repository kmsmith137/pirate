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



FakeXEngine::FakeXEngine(const XEngineMetadata &xmd_, const std::string &ip_addr_, uint16_t port_, int nthreads_) :
    xmd(xmd_),
    ip_addr(ip_addr_),
    port(port_),
    nthreads(nthreads_)
{
    xassert(ip_addr.size() > 0);
    xassert(port > 0);
    xassert(nthreads > 0);

    // Validate that XEngineMetadata has enough frequency channels for all threads.
    long total_nfreq = xmd.get_total_nfreq();
    if (total_nfreq < nthreads) {
        stringstream ss;
        ss << "FakeXEngine: nthreads=" << nthreads
           << " but total frequency channels=" << total_nfreq;
        throw runtime_error(ss.str());
    }

    xmd.validate();
}


FakeXEngine::~FakeXEngine()
{
    this->stop();

    for (auto &w : workers) {
        if (w.joinable())
            w.join();
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


void FakeXEngine::start()
{
    std::unique_lock<std::mutex> lock(mutex);
    _throw_if_stopped("FakeXEngine::start");

    if (!workers.empty())
        throw runtime_error("FakeXEngine::start() called but workers already exist");

    lock.unlock();

    // Create worker threads.
    workers.resize(nthreads);
    for (int i = 0; i < nthreads; i++)
        workers[i] = std::thread(&FakeXEngine::worker_main, this, i);
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

    XEngineMetadata ret;
    ret.version = xmd.version;
    ret.zone_nfreq = xmd.zone_nfreq;
    ret.zone_freq_edges = xmd.zone_freq_edges;
    ret.freq_channels = freq_channels;
    ret.nbeams = xmd.nbeams;
    ret.beam_ids = xmd.beam_ids;

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

        if (sock.connreset)
            return false;

        pos += n;
    }

    return true;
}


void FakeXEngine::_worker_main(int thread_id)
{
    // Create worker-specific metadata.
    XEngineMetadata worker_xmd = make_worker_metadata(thread_id);
    long nfreq = worker_xmd.freq_channels.size();
    long nbeams = worker_xmd.nbeams;

    // Data array size: shape (nbeams, nfreq, 256) int4, packed 2 per byte.
    // = nbeams * nfreq * 256 / 2 = nbeams * nfreq * 128 bytes.
    long data_nbytes = nbeams * nfreq * 128;
    vector<char> data_buf(data_nbytes, 0);  // all zeros for now

    // Open TCP connection.
    Socket sock(PF_INET, SOCK_STREAM);
    sock.connect(ip_addr, port);

    // Build protocol header: magic (4 bytes) + string length (4 bytes) + YAML string.
    string yaml_str = worker_xmd.to_yaml_string();

    // Pad YAML string to include null terminator and 4-byte alignment.
    long str_len = yaml_str.size() + 1;
    long padded_len = ((str_len + 3) / 4) * 4;

    // Build contiguous header buffer.
    vector<char> header_buf(8 + padded_len, '\0');
    uint32_t magic = protocol_magic;
    uint32_t len32 = static_cast<uint32_t>(padded_len);
    memcpy(header_buf.data(), &magic, 4);
    memcpy(header_buf.data() + 4, &len32, 4);
    memcpy(header_buf.data() + 8, yaml_str.data(), yaml_str.size());

    // Send protocol header in a single call.
    if (!_send_all(sock, header_buf.data(), header_buf.size()))
        return;

    // Send data arrays in a loop.
    long array_index = 0;

    while (true) {
        // Synchronization: before sending array N, wait until all threads
        // have finished sending array N-2. Uses even/odd barrier counters.
        // barrier[slot] counts completions of arrays with (index % 2 == slot).
        // "All threads finished array K" means barrier[K%2] >= nthreads * (K/2 + 1).
        // For array N-2: barrier[N%2] >= nthreads * ((N-2)/2 + 1) = nthreads * (N/2).

        int slot = array_index % 2;
        long required = long(nthreads) * (array_index / 2);

        std::unique_lock<std::mutex> lock(mutex);

        while (barrier[slot] < required) {
            if (is_stopped)
                return;
            cv.wait(lock);
        }

        lock.unlock();

        // Send data array. Checks 'is_stopped'.
        if (!_send_all(sock, data_buf.data(), data_nbytes))
            return;

        // Increment barrier and notify waiting threads.
        lock.lock();
        barrier[slot]++;
        lock.unlock();
        
        cv.notify_all();
        array_index++;
    }
}


}  // namespace pirate
