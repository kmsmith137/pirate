#include "../include/pirate/FakeCorrelator.hpp"
#include "../include/pirate/network_utils.hpp"  // Socket
#include "../include/pirate/system_utils.hpp"   // pin_thread_to_vcpus()

#include <thread>
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


FakeCorrelator::FakeCorrelator(long send_bufsize_, bool use_zerocopy_, bool use_mmap_, bool use_hugepages_) :
    barrier(0)  // Barrier will be initialized in run()
{
    this->send_bufsize = send_bufsize_;
    this->use_zerocopy = use_zerocopy_;
    this->use_mmap = use_mmap_;
    this->use_hugepages = use_hugepages_;
    
    xassert(send_bufsize > 0);
}


void FakeCorrelator::add_endpoint(const string &ip_addr, long num_tcp_connections, double total_gbps, const vector<int> &vcpu_list)
{
    Endpoint e;
    e.ip_addr = ip_addr;
    e.num_tcp_connections = num_tcp_connections;
    e.total_gbps = total_gbps;
    e.vcpu_list = vcpu_list;

    this->endpoints.push_back(e);
}


static void sender_thread_main(FakeCorrelator *correlator, long endpoint_index)
{
    try {
	correlator->sender_main(endpoint_index);
    } catch (const exception &exc) {
	correlator->abort(exc.what());
    }
}


void FakeCorrelator::run()
{
    long num_endpoints = this->endpoints.size();
    xassert(num_endpoints >= 0);

    cout << "\n"
	 << "Warning: this half-finished program will \"hang\" unless you wait to\n"
	 << "start it until after the receiver is fully initialized. The receiver is\n"
	 << "fully initialized when every TcpReceiver thread prints a line like this:\n"
	 << "\n"
	 << "  TcpReceiver(10.1.1.2, 1 connections, use_epoll=1): listening for TCP connections\n"
	 << endl;
    
    barrier.initialize(num_endpoints+1);
    vector<std::thread> threads(num_endpoints);

    for (long i = 0; i < num_endpoints; i++)
	threads.at(i) = std::thread(sender_thread_main, this, i);

    barrier.wait();
    
    stringstream ss;
    ss << "All TCP connections active, sending data.\n"
       << "Data will be sent until the receiver closes the connections.\n";
    cout << ss.str() << flush;

    barrier.wait();
    
    for (int ithread = 0; ithread < num_endpoints; ithread++)
	threads[ithread].join();
}


void FakeCorrelator::abort(const string &msg)
{
    std::unique_lock<mutex> lk(lock);
	
    if (aborted)
	return;
	
    this->aborted = true;
    this->abort_msg = msg;
    cout << msg << endl;
}


void FakeCorrelator::throw_exception_if_aborted()
{
    std::unique_lock<mutex> lk(lock);
    if (aborted)
	throw runtime_error(abort_msg);
}

    
void FakeCorrelator::sender_main(long endpoint_index)
{
    xassert(endpoint_index >= 0);
    xassert(endpoint_index < long(endpoints.size()));
    Endpoint &e = endpoints.at(endpoint_index);

    long nconn = e.num_tcp_connections;
    pin_thread_to_vcpus(e.vcpu_list);
    
    stringstream ss;
    ss << e.ip_addr << ": creating " << nconn << " TCP connections\n";
    cout << ss.str() << flush;
    
    int aflags = ksgpu::af_uhost;
    if (use_mmap)
	aflags |= (use_hugepages ? ksgpu::af_mmap_huge : ksgpu::af_mmap_small);
    
    shared_ptr<char> buf = ksgpu::af_alloc<char> (send_bufsize, aflags);
    vector<Socket> sockets;

    for (long i = 0; i < nconn; i++) {
	sockets.push_back(Socket(PF_INET, SOCK_STREAM));
	Socket &socket = sockets[i];
	
	// later: consider setsockopt(SO_RCVBUF), setsockopt(SO_SNDBUF), setsockopt(TCP_MAXSEG)
	socket.connect(e.ip_addr, 8787);  // TCP port 8787

	if (e.total_gbps > 0.0) {
	    double nbytes_per_sec = e.total_gbps / nconn / 8.0e-9;
	    socket.set_pacing_rate(nbytes_per_sec);
	}
    
	if (use_zerocopy)
	    socket.set_zerocopy();
    }

    this->barrier.wait();    
    this->barrier.wait();
    
    long nbytes_total = 0;

    for (;;) {
	for (long i = 0; i < nconn; i++) {
	    Socket &socket = sockets[i];

	    // FIXME should have a flag in socket.send() to enable this loop automatically
	    long pos = 0;
	    while (pos < send_bufsize) {
		long nbytes_sent = socket.send(buf.get() + pos, send_bufsize - pos);
		throw_exception_if_aborted();
		if (socket.connreset)
		    return;
		pos += nbytes_sent;
	    }
	    
	    nbytes_total += send_bufsize;
	}
    }
}


}  // namespace pirate
