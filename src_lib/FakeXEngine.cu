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


// Protocol magic number (little-endian): 0xf4bf4b01 where 01 is the version number.
static constexpr uint32_t protocol_magic = 0xf4bf4b01;


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


bool FakeXEngine::wait_for_sync(long array_index)
{
    // Before sending array N, wait until all threads have finished array N-2.
    // "All threads finished array K" means arrays_completed >= nthreads * (K + 1).
    // So we need arrays_completed >= nthreads * (N - 2 + 1) = nthreads * (N - 1).

    if (array_index < 2)
        return true;  // no waiting needed for first two arrays

    long required = long(nthreads) * (array_index - 1);

    while (arrays_completed.load() < required) {
        // Check if stopped.
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (is_stopped)
                return false;
        }

        // Brief sleep to avoid busy-waiting.
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }

    return true;
}


void FakeXEngine::worker_main(int thread_id)
{
    try {
        _worker_main(thread_id);
    } catch (...) {
        stop(std::current_exception());
    }
}


// Helper: send all bytes from buffer, using short timeouts to allow prompt exit.
// Returns false if stopped or connection reset.
static bool send_all(FakeXEngine *fxe, Socket &sock, const void *buf, long nbytes)
{
    const char *ptr = static_cast<const char *>(buf);
    long pos = 0;

    while (pos < nbytes) {
        // Check if stopped.
        {
            std::lock_guard<std::mutex> lock(fxe->mutex);
            if (fxe->is_stopped)
                return false;
        }

        // Try to send with short timeout.
        long n = sock.send_with_timeout(ptr + pos, nbytes - pos, FakeXEngine::send_timeout_ms);

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

    // Send protocol header: magic number (4 bytes).
    uint32_t magic = protocol_magic;
    if (!send_all(this, sock, &magic, sizeof(magic)))
        return;

    // Send YAML metadata.
    string yaml_str = worker_xmd.to_yaml_string();

    // Pad string to include null terminator, and pad to 4-byte alignment.
    long str_len = yaml_str.size() + 1;  // include null terminator
    long padded_len = ((str_len + 3) / 4) * 4;
    yaml_str.resize(padded_len, '\0');

    // Send string length (4 bytes).
    uint32_t len32 = static_cast<uint32_t>(padded_len);
    if (!send_all(this, sock, &len32, sizeof(len32)))
        return;

    // Send the YAML string (null-terminated and padded).
    if (!send_all(this, sock, yaml_str.data(), padded_len))
        return;

    // Send data arrays in a loop.
    long array_index = 0;

    while (true) {
        // Wait for synchronization barrier.
        if (!wait_for_sync(array_index))
            return;

        // Check if stopped before sending.
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (is_stopped)
                return;
        }

        // Send data array.
        if (!send_all(this, sock, data_buf.data(), data_nbytes))
            return;

        // Increment completion counter.
        arrays_completed.fetch_add(1);
        array_index++;
    }
}


}  // namespace pirate
