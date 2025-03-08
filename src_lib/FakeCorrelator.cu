#include "../include/pirate/FakeCorrelator.hpp"
#include "../include/pirate/network_utils.hpp"  // Socket

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


FakeCorrelator::FakeCorrelator(const Params &params_)
    : params(params_)
{
    xassert(params.nconn_per_ipaddr > 0);
    xassert(params.ipaddr_list.size() > 0);
    xassert(params.send_bufsize > 0);
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

    
void FakeCorrelator::sender_main(const string &ipaddr)
{
    long nbytes_total = 0;
    
    int aflags = ksgpu::af_uhost;
    if (params.use_mmap)
	aflags |= (params.use_hugepages ? ksgpu::af_mmap_huge : ksgpu::af_mmap_small);
    
    shared_ptr<char> buf = ksgpu::af_alloc<char> (params.send_bufsize, aflags);
    
    Socket socket(PF_INET, SOCK_STREAM);
    // later: consider setsockopt(SO_RCVBUF), setsockopt(SO_SNDBUF), setsockopt(TCP_MAXSEG)
    socket.connect(ipaddr, 8787);  // TCP port 8787
    
    if (params.gbps_per_ipaddr > 0.0) {
	double nbytes_per_sec = params.gbps_per_ipaddr / params.nconn_per_ipaddr / 8.0e-9;
	socket.set_pacing_rate(nbytes_per_sec);
    }
    
    if (params.use_zerocopy)
	socket.set_zerocopy();
    
    do {
	long nbytes_sent = socket.send(buf.get(), params.send_bufsize);
	nbytes_total += nbytes_sent;
	throw_exception_if_aborted();
    } while (!socket.connreset);
}


static void sender_thread_main(shared_ptr<FakeCorrelator> correlator, string ipaddr)
{
    try {
	correlator->sender_main(ipaddr);
    } catch (const exception &exc) {
	correlator->abort(exc.what());
    }
}


// Static member function
void FakeCorrelator::run(const Params &params)
{
    shared_ptr<FakeCorrelator> correlator = make_shared<FakeCorrelator> (params);
    
    int naddr = params.ipaddr_list.size();
    int nconn = params.nconn_per_ipaddr;
    int nthreads = naddr * nconn;
    
    vector<std::thread> threads(nthreads);

    for (int iaddr = 0; iaddr < naddr; iaddr++)
	for (int iconn = 0; iconn < nconn; iconn++)
	    threads[iaddr*nconn + iconn] = std::thread(sender_thread_main, correlator, params.ipaddr_list[iaddr]);

    for (int ithread = 0; ithread < nthreads; ithread++)
	threads[ithread].join();
}


}  // namespace pirate
