#include "../include/pirate/HwtestSender.hpp"
#include "../include/pirate/Hwtest.hpp"         // Hwtest::tcp_port
#include "../include/pirate/network_utils.hpp"  // Socket
#include "../include/pirate/system_utils.hpp"   // set_thread_affinity()
#include "../include/pirate/constants.hpp"      // default_poll_cadence_ms

#include <sstream>
#include <iostream>

#include <ksgpu/mem_utils.hpp>
#include <ksgpu/string_utils.hpp>  // nbytes_to_str()
#include <ksgpu/xassert.hpp>


using namespace std;
using namespace ksgpu;


namespace pirate {
#if 0
}  // editor auto-indent
#endif


HwtestSender::HwtestSender(long send_bufsize_, bool use_zerocopy_, bool use_mmap_, bool use_hugepages_)
{
    xassert(send_bufsize_ > 0);

    this->send_bufsize = send_bufsize_;
    this->use_zerocopy = use_zerocopy_;
    this->use_mmap = use_mmap_;
    this->use_hugepages = use_hugepages_;
}


HwtestSender::~HwtestSender()
{
    this->stop();
    this->join();
}


void HwtestSender::_throw_if_stopped(const char *method_name)
{
    // Caller must hold mutex.
    if (error)
        std::rethrow_exception(error);

    if (is_stopped)
        throw runtime_error(string(method_name) + " called on stopped instance");
}


// Note: per the strict stoppable-class policy (notes/stoppable_class.md),
// ANY exception thrown from an entry point stops the HwtestSender --
// including precondition errors ("called twice", "called after start()").

void HwtestSender::add_endpoint(const string &ip_addr, long num_tcp_connections, double total_gbps, const vector<int> &vcpu_list)
{
    try {
        std::unique_lock<std::mutex> lock(mutex);
        _throw_if_stopped("HwtestSender::add_endpoint");

        if (is_started)
            throw runtime_error("HwtestSender::add_endpoint() called after start()");

        Endpoint e;
        e.ip_addr = ip_addr;
        e.num_tcp_connections = num_tcp_connections;
        e.total_gbps = total_gbps;
        e.vcpu_list = vcpu_list;

        endpoints.push_back(e);
    } catch (...) {
        stop(std::current_exception());
        throw;
    }
}


void HwtestSender::start()
{
    // The try/catch wrapper also covers thread-creation failure partway
    // through the spawns: the already-spawned workers see the stop and exit
    // promptly, instead of running on a not-stopped object.
    try {
        std::unique_lock<std::mutex> lock(mutex);
        _throw_if_stopped("HwtestSender::start");

        if (is_started)
            throw runtime_error("HwtestSender::start() called twice");

        if (endpoints.empty())
            throw runtime_error("HwtestSender::start() called with no endpoints");

        is_started = true;

        // Publish 'workers' under the mutex (join()/wait() synchronize on
        // it). Holding the lock across the spawns is safe: freshly-spawned
        // workers block briefly in _stopped() until we release, and start()
        // never waits on them.
        long num_endpoints = endpoints.size();
        workers.resize(num_endpoints);

        for (long i = 0; i < num_endpoints; i++)
            workers[i] = std::thread(&HwtestSender::worker_main, this, i);
    } catch (...) {
        stop(std::current_exception());
        throw;
    }
}


void HwtestSender::stop(std::exception_ptr e) const
{
    std::unique_lock<std::mutex> lock(mutex);

    if (is_stopped)
        return;

    is_stopped = true;
    error = e;

    // Notify after releasing the mutex, so woken threads aren't
    // immediately blocked re-acquiring it.
    lock.unlock();
    cv.notify_all();
}


void HwtestSender::join()
{
    // Briefly take the mutex to synchronize with start(), which publishes
    // 'workers' under it -- this guarantees we observe either the
    // fully-spawned vector or an empty one, never a mid-resize state. The
    // joins themselves run with the mutex RELEASED: worker threads take
    // the mutex on their exit path, so joining under it would deadlock.
    // (Not safe to call join() concurrently with itself.)
    long nworkers;
    {
        std::lock_guard<std::mutex> lock(mutex);
        nworkers = long(workers.size());
    }

    for (long i = 0; i < nworkers; i++) {
        if (workers[i].joinable())
            workers[i].join();
    }
}


bool HwtestSender::_stopped()
{
    std::lock_guard<std::mutex> lock(mutex);
    return is_stopped;
}


bool HwtestSender::wait(int timeout_ms)
{
    std::unique_lock<std::mutex> lock(mutex);
    long nworkers = long(workers.size());
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);

    while (num_workers_exited < nworkers) {
        if (cv.wait_until(lock, deadline) == std::cv_status::timeout)
            return false;
    }

    return true;
}


void HwtestSender::worker_main(long endpoint_index)
{
    // Everything that could conceivably throw -- endpoints.at() and the
    // exit-summary print (stringstream allocation) -- lives inside the
    // try/catch, so no exception can escape the thread (std::terminate).
    // The exit-counter block stays outside the try: wait() depends on it
    // running on every path, and it cannot throw.
    try {
        const string &ip_addr = endpoints.at(endpoint_index).ip_addr;
        string errmsg;
        long nbytes_sent = 0;

        try {
            nbytes_sent = _worker_main(endpoint_index);
        } catch (const exception &exc) {
            errmsg = exc.what();
            stop(std::current_exception());
        } catch (...) {
            errmsg = "unknown exception";
            stop(std::current_exception());
        }

        stringstream ss;
        ss << ip_addr << ": exiting, sent " << nbytes_to_str(nbytes_sent);
        if (!errmsg.empty())
            ss << " [error: " << errmsg << "]";
        ss << "\n";
        cout << ss.str() << flush;
    } catch (...) {
        stop(std::current_exception());
    }

    {
        std::lock_guard<std::mutex> lock(mutex);
        num_workers_exited++;
    }
    cv.notify_all();
}


bool HwtestSender::_send_all(Socket &sock, const void *buf, long nbytes)
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


long HwtestSender::_worker_main(long endpoint_index)
{
    xassert(endpoint_index >= 0);
    xassert(endpoint_index < long(endpoints.size()));
    Endpoint &e = endpoints.at(endpoint_index);

    // Early check: covers the window where stop() lands between start()
    // releasing the mutex and thread creation. (Later blocking points --
    // connect and send -- recheck is_stopped themselves.)
    if (_stopped())
        return 0;

    long nconn = e.num_tcp_connections;
    set_thread_affinity(e.vcpu_list);

    {
        stringstream ss;
        ss << e.ip_addr << ": creating " << nconn << " TCP connection(s)\n";
        cout << ss.str() << flush;
    }

    int aflags = ksgpu::af_uhost;
    if (use_mmap)
        aflags |= (use_hugepages ? ksgpu::af_mmap_huge : ksgpu::af_mmap_small);

    shared_ptr<char> buf = ksgpu::af_alloc<char> (send_bufsize, aflags);
    vector<Socket> sockets;

    for (long i = 0; i < nconn; i++) {
        sockets.push_back(Socket(PF_INET, SOCK_STREAM));
        Socket &socket = sockets[i];

        // Non-blocking connect + poll, rechecking is_stopped every
        // constants::default_poll_cadence_ms. (A plain blocking connect() could
        // stall for the kernel's SYN-retry timeout, ~2 minutes, if the receiver
        // is not running -- blocking stop() and the destructor for that long.)
        socket.start_connect(e.ip_addr, Hwtest::tcp_port);

        while (!socket.wait_for_connect(constants::default_poll_cadence_ms)) {
            if (_stopped())
                return 0;
        }

        if (e.total_gbps > 0.0) {
            double nbytes_per_sec = e.total_gbps / nconn / 8.0e-9;
            socket.set_pacing_rate(nbytes_per_sec);
        }

        if (use_zerocopy)
            socket.set_zerocopy();
    }

    {
        stringstream ss;
        ss << e.ip_addr << ": " << nconn << " TCP connection(s) active, sending data\n";
        cout << ss.str() << flush;
    }

    long nbytes_sent = 0;

    for (;;) {
        for (long i = 0; i < nconn; i++) {
            if (!_send_all(sockets[i], buf.get(), send_bufsize))
                return nbytes_sent;
            nbytes_sent += send_bufsize;
        }
    }
}


}  // namespace pirate
