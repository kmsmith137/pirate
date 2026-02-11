#include "../include/pirate/FakeCorrelator.hpp"
#include "../include/pirate/network_utils.hpp"  // Socket
#include "../include/pirate/system_utils.hpp"   // set_thread_affinity()

#include <sstream>
#include <iostream>

#include <ksgpu/mem_utils.hpp>
#include <ksgpu/xassert.hpp>


using namespace std;
using namespace ksgpu;


namespace pirate {
#if 0
}  // editor auto-indent
#endif


FakeCorrelator::FakeCorrelator(long send_bufsize_, bool use_zerocopy_, bool use_mmap_, bool use_hugepages_)
{
    xassert(send_bufsize_ > 0);

    this->send_bufsize = send_bufsize_;
    this->use_zerocopy = use_zerocopy_;
    this->use_mmap = use_mmap_;
    this->use_hugepages = use_hugepages_;
}


FakeCorrelator::~FakeCorrelator()
{
    this->stop();
    this->join();
}


void FakeCorrelator::_throw_if_stopped(const char *method_name)
{
    // Caller must hold mutex.
    if (error)
        std::rethrow_exception(error);

    if (is_stopped)
        throw runtime_error(string(method_name) + " called on stopped instance");
}


void FakeCorrelator::add_endpoint(const string &ip_addr, long num_tcp_connections, double total_gbps, const vector<int> &vcpu_list)
{
    std::lock_guard<std::mutex> lock(mutex);
    _throw_if_stopped("FakeCorrelator::add_endpoint");

    if (is_started)
        throw runtime_error("FakeCorrelator::add_endpoint() called after start()");

    Endpoint e;
    e.ip_addr = ip_addr;
    e.num_tcp_connections = num_tcp_connections;
    e.total_gbps = total_gbps;
    e.vcpu_list = vcpu_list;

    endpoints.push_back(e);
}


void FakeCorrelator::start()
{
    std::unique_lock<std::mutex> lock(mutex);
    _throw_if_stopped("FakeCorrelator::start");

    if (is_started)
        throw runtime_error("FakeCorrelator::start() called twice");

    if (endpoints.empty())
        throw runtime_error("FakeCorrelator::start() called with no endpoints");

    is_started = true;
    lock.unlock();

    long num_endpoints = endpoints.size();
    workers.resize(num_endpoints);

    for (long i = 0; i < num_endpoints; i++)
        workers[i] = std::thread(&FakeCorrelator::worker_main, this, i);
}


void FakeCorrelator::stop(std::exception_ptr e)
{
    std::lock_guard<std::mutex> lock(mutex);

    if (is_stopped)
        return;

    is_stopped = true;
    error = e;
    cv.notify_all();
}


void FakeCorrelator::join()
{
    for (auto &w : workers) {
        if (w.joinable())
            w.join();
    }
}


void FakeCorrelator::worker_main(long endpoint_index)
{
    try {
        _worker_main(endpoint_index);
    } catch (...) {
        stop(std::current_exception());
    }
}


bool FakeCorrelator::_send_all(Socket &sock, const void *buf, long nbytes)
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


void FakeCorrelator::_worker_main(long endpoint_index)
{
    xassert(endpoint_index >= 0);
    xassert(endpoint_index < long(endpoints.size()));
    Endpoint &e = endpoints.at(endpoint_index);

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

        // FIXME connect() can block for a long time if receiver is not running.
        socket.connect(e.ip_addr, 8787);  // TCP port 8787

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

    for (;;) {
        for (long i = 0; i < nconn; i++) {
            if (!_send_all(sockets[i], buf.get(), send_bufsize))
                return;
        }
    }
}


}  // namespace pirate
