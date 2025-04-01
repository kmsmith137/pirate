#include "../include/pirate/FakeServer.hpp"

#include <mutex>
#include <thread>
#include <sstream>
#include <iostream>

#include <ksgpu/Barrier.hpp>
#include <ksgpu/mem_utils.hpp>
#include <ksgpu/cuda_utils.hpp>
#include <ksgpu/time_utils.hpp>
#include <ksgpu/string_utils.hpp>
#include <ksgpu/memcpy_kernels.hpp>
#include <ksgpu/xassert.hpp>

#include "../include/pirate/inlines.hpp"
#include "../include/pirate/file_utils.hpp"
#include "../include/pirate/system_utils.hpp"
#include "../include/pirate/network_utils.hpp"  // Socket, Epoll
#include "../include/pirate/loose_ends/cpu_downsample.hpp"


using namespace std;
using namespace ksgpu;


namespace pirate {
#if 0
}  // editor auto-indent
#endif


static int get_cuda_device_count()
{
    int count = 0;
    CUDA_CALL(cudaGetDeviceCount(&count));
    return count;
}


// -------------------------------------------------------------------------------------------------
//
// FakeServer::Stats


// Note: if new members are added, update _show_stats() and Stats::operator+=() below.
struct FakeServer::Stats
{
    // Bandwidth.
    long nbytes_hmem = 0;    // total host memory bandwidth (for memcpy, include factor 2)
    long nbytes_gmem = 0;    // total GPU memory bandwidth
    long nbytes_h2g = 0;     // host -> GPU
    long nbytes_g2h = 0;     // GPU -> host
    long nbytes_ssd = 0;

    // Network.
    long nbytes_received = 0;
    long num_read_calls = 0;
    long num_epoll_calls = 0;

    // FRB-specific.
    long nsamp_downsampled = 0;

    Stats &operator+=(const Stats &s);
};


FakeServer::Stats &FakeServer::Stats::operator+=(const Stats &s)
{
    nbytes_hmem += s.nbytes_hmem;
    nbytes_gmem += s.nbytes_gmem;
    nbytes_h2g += s.nbytes_h2g;
    nbytes_g2h += s.nbytes_g2h;
    nbytes_ssd += s.nbytes_ssd;

    nbytes_received += s.nbytes_received;
    num_read_calls += s.num_read_calls;
    num_epoll_calls += s.num_epoll_calls;

    nsamp_downsampled += s.nsamp_downsampled;

    return *this;
}


static double _show_stats(const vector<FakeServer::Stats> &stats, const vector<double> &dt)
{
    xassert_eq(stats.size(), dt.size());

    double hmem_bw = 0.0;
    double gmem_bw = 0.0;
    double h2g_bw = 0.0;
    double g2h_bw = 0.0;
    double ssd_bw = 0.0;
    double network_bw = 0.0;
    double read_call_rate = 0.0;
    double epoll_call_rate = 0.0;
    double ds_rate = 0.0;
    double dt_max = 0.0;

    for (ulong i = 0; i < stats.size(); i++) {
	dt_max = std::max(dt_max, dt[i]);
	
	// Don't trust rate estimates below 0.5 sec.
	if (dt[i] < 0.5)
	    continue;
	
	hmem_bw += (stats[i].nbytes_hmem / dt[i]);
	gmem_bw += (stats[i].nbytes_gmem / dt[i]);
	h2g_bw += (stats[i].nbytes_h2g / dt[i]);
	g2h_bw += (stats[i].nbytes_g2h / dt[i]);
	ssd_bw += (stats[i].nbytes_ssd / dt[i]);
	network_bw += (stats[i].nbytes_received / dt[i]);
	read_call_rate += (stats[i].num_read_calls / dt[i]);
	epoll_call_rate += (stats[i].num_epoll_calls / dt[i]);
	ds_rate += (stats[i].nsamp_downsampled / dt[i]);
    }

    stringstream ss;
    
    if (hmem_bw > 0.0)
	ss << "   Host memory bandwidth: " << (1.0e-9 * hmem_bw) << " GB/s\n";
    if (gmem_bw > 0.0)
	ss << "   GPU global memory bandwidth: " << (1.0e-9 * gmem_bw) << " GB/s\n";
    if (h2g_bw > 0.0)
	ss << "   Host->GPU bandwidth: " << (1.0e-9 * h2g_bw) << " GB/s\n";
    if (g2h_bw > 0.0)
	ss << "   GPU->Host bandwidth: " << (1.0e-9 * g2h_bw) << " GB/s\n";
    if (ssd_bw > 0.0)
	ss << "   SSD bandwidth: " << (1.0e-9 * ssd_bw) << " GB/s\n";
    if (network_bw > 0.0)
	ss << "   Network bandwidth: " << (8.0e-9 * network_bw) << " Gbps (not GB/s)\n";
    if (read_call_rate > 0.0)
	ss << "   read() calls/sec: " << (read_call_rate) << "\n";
    if (epoll_call_rate > 0.0)
	ss << "   epoll() calls/sec: " << (epoll_call_rate) << "\n";
    if (ds_rate > 0.0)
	ss << "   Downsampling throughput: " << (1.0e-9 * ds_rate) << " Gsamp/s\n";

    string s = ss.str();
    
    if (s.size() > 0)
	cout << "Elapsed time: " << dt_max << " sec\n" << s << flush;

    return dt_max;
}


// -------------------------------------------------------------------------------------------------
//
// FakeServer::State


struct FakeServer::State
{
    // Values of 'code' member below.
    static constexpr int initializing = 0;
    static constexpr int running = 1;
    static constexpr int stopped = 2;
    static constexpr int aborted = 3;
    
    // Worker threads wait at the barrier twice.
    //  - After returning from worker_initialize(), before calling worker_accept_connections()
    //  - After returning from worker_accept_connections(), before calling worker_body().

    State(bool use_hugepages);
    
    int code = initializing;
    std::string abort_msg;
    std::mutex lock;
    
    bool use_hugepages;
    ksgpu::Barrier barrier;

    void abort(const string &msg);
};


FakeServer::State::State(bool use_hugepages_) :
    use_hugepages(use_hugepages_),
    barrier(0)  // will be initialized in FakeServer::start(), after worker thread count is known
{ }


void FakeServer::State::abort(const string &msg)
{
    this->barrier.abort(msg);

    std::unique_lock lk(lock);

    if (code != aborted) {
	this->code = aborted;
	this->abort_msg = msg;
	// cout << msg << endl;
    }
}


// -------------------------------------------------------------------------------------------------
//
// FakeServer::Worker


struct FakeServer::Worker
{
    using State = FakeServer::State;
    using Stats = FakeServer::Stats;
    
    shared_ptr<FakeServer::State> state;
    
    string worker_name = "Anonymous thread";
    vector<int> vcpu_list;

    std::mutex lock;
    struct timeval tv0;
    struct timeval tv1;
    bool tv_initialized = false;
    Stats cumulative_stats;

    Worker(const shared_ptr<FakeServer::State> &state_, const vector<int> vcpu_list_);
    virtual ~Worker() { }
    
    // Called by worker thread.
    virtual void worker_initialize() = 0;
    virtual void worker_accept_connections() { }   // only Receiver subclass defines this
    virtual Stats worker_body() = 0;

    shared_ptr<char> worker_alloc(long nbytes, bool on_gpu=false);
    string devstr(int device);
};


FakeServer::Worker::Worker(const shared_ptr<FakeServer::State> &state_, const vector<int> vcpu_list_) :
    state(state_), vcpu_list(vcpu_list_)
{
    xassert(state);
}


shared_ptr<char> FakeServer::Worker::worker_alloc(long nbytes, bool on_gpu)
{
    xassert(nbytes > 0);
    
    int aflags = on_gpu ? ksgpu::af_gpu : ksgpu::af_rhost;
    // aflags |= af_verbose;

    if (state->use_hugepages && !on_gpu)
	aflags |= ksgpu::af_mmap_huge;
	
    return ksgpu::af_alloc<char> (nbytes, aflags | af_zero);
}


string FakeServer::Worker::devstr(int device)
{
    if (device < 0)
	return "host";
	
    stringstream ss;
    ss << "gpu" << device;
    return ss.str();
}


// -------------------------------------------------------------------------------------------------
//
// FakeServer core logic. This is the difficult part!


FakeServer::FakeServer(const std::string &server_name_, bool use_hugepages) :
    server_name(server_name_),
    state(make_shared<FakeServer::State> (use_hugepages))
{ }


void FakeServer::_add_worker(const shared_ptr<Worker> &wp, const string &caller)
{
    std::unique_lock lk(state->lock);

    if (state->code != State::initializing)
	throw runtime_error(caller + "() called on server which has either been started or aborted");;

    // Reminder: this->workers is protected by state->lock.
    this->workers.push_back(wp);
}    


// Called by FakeServer::start().
static void worker_thread_main(shared_ptr<FakeServer::Worker> worker)
{
    using State = FakeServer::State;
    using Stats = FakeServer::Stats;
    
    shared_ptr<State> state = worker->state;
    
    try {
	// No-ops if 'vcpu_list' is empty.
	pin_thread_to_vcpus(worker->vcpu_list);
	
	// Reminder: worker threads wait at the barrier twice.
	//  - After returning from worker_initialize(), before calling worker_accept_connections()
	//  - After returning from worker_accept_connections(), before calling worker_body().

	worker->worker_initialize();
	state->barrier.wait();
	
	worker->worker_accept_connections();
	state->barrier.wait();

	for (;;) {
	    std::unique_lock lk(state->lock);

	    if (state->code == State::aborted)
		throw runtime_error(state->abort_msg);
	    if (state->code == State::stopped)
		return;   // normal exit (triggered asychronously by FakeServer::stop())
	    
	    xassert(state->code == State::running);
	    lk.unlock();  // okay to drop lock after checking state->code.
	    
	    Stats stats = worker->worker_body();
	    struct timeval tv = ksgpu::get_time();

	    std::unique_lock lk2(worker->lock);

	    // Skip stats from first call to worker_body().
	    if (!worker->tv_initialized) {
		worker->tv0 = tv;
		worker->tv1 = tv;
		worker->tv_initialized = true;
	    }
	    else {
		worker->tv1 = tv;
		worker->cumulative_stats += stats;
	    }
	}
    }
    catch (const exception &exc) {
	state->abort(exc.what());
    }
}


// Called from python, to start server.
void FakeServer::start()
{
    std::unique_lock lk(state->lock);
    
    long num_workers = workers.size();
    
    if (num_workers == 0)
	throw runtime_error("FakeServer does not contain any worker threads");
    
    if (state->code != State::initializing)
	throw runtime_error("FakeServer::start() called on server which has either been started or aborted");;

    state->code = State::running;
    state->barrier.initialize(num_workers);

    std::unique_lock lk2(this->thread_lock);
    lk.unlock();  // okay to drop state->lock

    try {
	xassert(threads.size() == 0);
	threads.resize(num_workers);

	// Reminder: after server is started, 'workers' is immutable, so we don't need to hold state->lock here.
	for (long i = 0; i < num_workers; i++) {
	    auto wp = workers[i];
	    stringstream ss;
	    ss << wp->worker_name << ": vcpu_list=" << ksgpu::tuple_str(wp->vcpu_list) << "\n";
	    cout << ss.str() << flush;
	    threads[i] = std::thread(worker_thread_main, wp);
	}
    }
    catch (const exception &exc) {
	state->abort(exc.what());
    }		     
}


double FakeServer::show_stats()
{
    std::unique_lock lk(state->lock);

    // If a C++ worker thread throws an exception, this allows the exception text to show up in python.
    if (state->code == State::aborted)
	throw runtime_error(state->abort_msg);
    
    if (state->code == State::initializing)
	throw runtime_error("FakeSever::show_stats() called on server which has not been started");

    lk.unlock();

    // Reminder: after server is started, 'workers' is immutable, so we don't need to hold state->lock here...
    long num_workers = workers.size();
    vector<Stats> stats(num_workers);
    vector<double> dt(num_workers);

    for (long i = 0; i < num_workers; i++) {
	// ...however, we do need to hold the per-worker lock, before reading stats from each worker.
	Worker *w = workers[i].get();
	std::unique_lock lk2(w->lock);
	stats[i] = w->cumulative_stats;
	dt[i] = w->tv_initialized ? ksgpu::time_diff(w->tv0, w->tv1) : 0.0;
    }
    
    return _show_stats(stats, dt);
}


// Called from python, to stop server.
void FakeServer::stop()
{
    std::unique_lock lk(state->lock);

    // If a C++ worker thread throws an exception, this allows the exception text to show up in python.
    if (state->code == State::aborted)
	throw runtime_error(state->abort_msg);
    
    if (state->code == State::initializing)
	throw runtime_error("FakeSever::stop() called on server which has not been started");

    // No-ops if stop() called multiple times.
    state->code = State::stopped;
}


// Called from python, to join threads.
void FakeServer::join_threads()
{
    std::unique_lock lk(state->lock);

    if (state->code < State::stopped)
	throw runtime_error("FakeServer::join_threads() called on server which has not been stopped (or aborted)");

    std::unique_lock lk2(thread_lock);
    lk.unlock();
    
    for (ulong i = 0; i < threads.size(); i++)
	threads[i].join();
}


// Called from python (C++ callers can call State::abort())
void FakeServer::abort(const string &abort_msg)
{
    // Call without acquiring state->lock.
    state->abort(abort_msg);
}


// -------------------------------------------------------------------------------------------------
//
// Receiver


struct Receiver : FakeServer::Worker
{
    // Initialized in constructor.
    string ip_addr;
    long num_tcp_connections = 0;
    long recv_bufsize = 0;
    bool use_epoll = true;
    long network_sync_cadence = 0;

    // Initialized in worker_initialize().
    shared_ptr<char> recv_buf;
    Socket listening_socket;
    Epoll epoll;

    // Initialized in worker_accept_connections().
    vector<Socket> data_sockets;

    
    Receiver(const shared_ptr<FakeServer::State> state_, const vector<int> &vcpu_list_, const string &ip_addr_, long num_tcp_connections_, long recv_bufsize_, bool use_epoll_, long network_sync_cadence_) :
	Worker(state_, vcpu_list_),
	ip_addr(ip_addr_),
	num_tcp_connections(num_tcp_connections_),
	recv_bufsize(recv_bufsize_),
	use_epoll(use_epoll_),
	network_sync_cadence(network_sync_cadence_),
	epoll(false)  // 'epoll' is constructed in an uninitialized state, and gets initialized in worker_initialize()
    {
	xassert(ip_addr.size() > 0);
	xassert(num_tcp_connections > 0);
        xassert(use_epoll || (num_tcp_connections == 1));
        xassert(recv_bufsize > 0);
        xassert(network_sync_cadence > 0);

	stringstream ss;
	ss << "TCP receive thread(" << ip_addr << ", " << num_tcp_connections << " connections)";
	this->worker_name = ss.str();
    }
	
    
    virtual void worker_initialize() override
    {
	this->recv_buf = worker_alloc(recv_bufsize);

	this->listening_socket = Socket(PF_INET, SOCK_STREAM);
	this->listening_socket.set_reuseaddr();
	this->listening_socket.bind(ip_addr, 8787);  // TCP port 8787

	if (use_epoll)
	    this->epoll.initialize();
    }

    
    virtual void worker_accept_connections()
    {
	stringstream ss;
	ss << worker_name << ": listening for TCP connections\n";
	cout << ss.str() << flush;

	this->data_sockets.resize(num_tcp_connections);
	this->listening_socket.listen();

	// Later: setsockopt(SO_RCVBUF), setsockopt(SO_SNDBUF) here?
	// I tried SO_RECVLOWAT here, but uProf reported significantly higher memory bandwidth!

	for (unsigned int ids = 0; ids < data_sockets.size(); ids++) {
	    this->data_sockets[ids] = listening_socket.accept();

	    if (!use_epoll)
		continue;
	    
	    this->data_sockets[ids].set_nonblocking();

	    epoll_event ev;
	    ev.events = EPOLLIN | EPOLLRDHUP | EPOLLHUP;
	    ev.data.u32 = ids;
	    epoll.add_fd(data_sockets[ids].fd, ev);
	}

	stringstream ss2;
	ss2 << worker_name << ": receiving data\n";
	cout << ss2.str() << flush;
    }


    // Receives (network_sync_cadence) bytes of data, and updates this->cumulative_stats.
    virtual Stats worker_body() override
    {
	return use_epoll ? _receive_epoll() : _receive_no_epoll();
    }

    
    // Helper for receive_data(). (Called if use_epoll=false.)
    Stats _receive_no_epoll()
    {
	xassert(data_sockets.size() == 1);
	Stats stats;
	
	while (stats.nbytes_received < network_sync_cadence) {
	    // Blocking read()
	    // FIXME in hindsight, not a fan of how max_read is computed.
	    long max_read = std::min(recv_bufsize, network_sync_cadence - stats.nbytes_received);
	    long nbytes_read = data_sockets[0].read(recv_buf.get(), max_read);
	    xassert(nbytes_read >= 0);
	    
	    stats.nbytes_received += nbytes_read;
	    stats.num_read_calls++;
	    
	    if (nbytes_read <= 0)
		throw runtime_error("TCP connection ended prematurely");
	}

	return stats;
    }


    // Helper for receive_iteration(). (Called if use_epoll=true.)
    Stats _receive_epoll()
    {
	Stats stats;
	
	while (stats.nbytes_received < network_sync_cadence) {
	    long nbytes_prev = stats.nbytes_received;
	    
	    int num_events = epoll.wait();  // blocking
	    stats.num_epoll_calls++;

	    for (int iev = 0; iev < num_events; iev++) {
		uint32_t ev_flags = epoll.events[iev].events;

		if (!(ev_flags & EPOLLIN))
		    continue;

		unsigned int ids = epoll.events[iev].data.u32;
		xassert(ids < data_sockets.size());
		
		// Non-blocking read()
		// FIXME in hindsight, not a fan of how max_read is computed.
		long max_read = std::min(recv_bufsize, network_sync_cadence - stats.nbytes_received);
		long nbytes_read = data_sockets[ids].read(recv_buf.get(), max_read);
		xassert(nbytes_read >= 0);
	    
		stats.nbytes_received += nbytes_read;
		stats.num_read_calls++;

		// FIXME read() can return zero, even with EPOLLIN set?!
		// I'd like to revisit this, just for the sake of general understanding.
		// It seems to be harmless, since the reported values of "bytes/read()" and "bytes/epoll()"
		// look reasonable.
		
		if (nbytes_read == 0)
		    continue;
		
		if (stats.nbytes_received >= network_sync_cadence)
		    return stats;
	    }

	    if ((stats.nbytes_received == nbytes_prev) && (num_events == int(data_sockets.size())))
		throw runtime_error("TCP connection(s) ended premaurely");
	}

	return stats;
    }
};
    

// -------------------------------------------------------------------------------------------------
//
// MemcpyWorker (either host->host, host->device, or device->host)
//
//  - Currently use host->host MemcpyWorker as crude placeholder for packet assembler.
//  - Currently use (host->device + device->host) pair as crude placeholder for dedisperion.


struct MemcpyWorker : public FakeServer::Worker
{
    // string worker_name;    // inherited from 'Worker' base class

    // Initialized in constructor.
    int src_device = -1;   // -1 for host
    int dst_device = -1;   // -1 for host
    long nbytes = 0;
    long blocksize = 0;
    
    shared_ptr<char> psrc;
    shared_ptr<char> pdst;

    // Reminder: cudaStream_t is a typedef for (CUstream_st *)
    shared_ptr<CUstream_st> stream;

    
    MemcpyWorker(const shared_ptr<FakeServer::State> &state_, const vector<int> &vcpu_list_, int src_device_, int dst_device_, long nbytes_, long blocksize_):
	Worker(state_, vcpu_list_)
    {
	this->src_device = src_device_;
	this->dst_device = dst_device_;
	this->nbytes = nbytes_;
	this->blocksize = (blocksize_ > 0) ? blocksize_ : nbytes;

	int num_gpus_in_machine = get_cuda_device_count();

	if (src_device >= 0)
	    xassert_lt(src_device, num_gpus_in_machine);
	if (dst_device >= 0)
	    xassert_lt(dst_device, num_gpus_in_machine);
	if ((src_device >= 0) && (dst_device >= 0) && (src_device != dst_device))
	    throw runtime_error("MemcpyWorker: copy between different GPUs is currently unsupported");				  

	stringstream ss;
	ss << "Memcpy thread (src=" << devstr(src_device)
	   << ", dst=" << devstr(dst_device)
	   << ", nbytes=" << ksgpu::nbytes_to_str(nbytes)
	   << ", blocksize=" << ksgpu::nbytes_to_str(blocksize)
	   << ")";
	    
	this->worker_name = ss.str();
    }


    virtual void worker_initialize() override
    {
	int dev = max(src_device, dst_device);
	int cuda_stream_priority = -1;  // lower numerical value = higher priority
	
	if (dev >= 0) {
	    CUDA_CALL(cudaSetDevice(dev));  // initialize CUDA device in worker thread
	    this->stream = ksgpu::CudaStreamWrapper(cuda_stream_priority).p;  // RAII stream
	}

	this->psrc = worker_alloc(blocksize, (src_device >= 0));  // last argument is boolean 'on_gpu'
	this->pdst = worker_alloc(blocksize, (dst_device >= 0));  // last argument is boolean 'on_gpu'
    }
    

    virtual Stats worker_body() override
    {
	long pos = 0;
	
	while (pos < nbytes) {
	    long n = min(nbytes-pos, blocksize);
	    pos += n;

	    if (stream)
		CUDA_CALL(cudaMemcpyAsync(pdst.get(), psrc.get(), n, cudaMemcpyDefault, stream.get()));
	    else
		memcpy(pdst.get(), psrc.get(), n);
	}

	Stats stats;

	if (src_device < 0)
	    stats.nbytes_hmem += nbytes;
	if (dst_device < 0)
	    stats.nbytes_hmem += nbytes;
	
	if (src_device >= 0)
	    stats.nbytes_gmem += nbytes;
	if (dst_device >= 0)
	    stats.nbytes_gmem += nbytes;

	if ((src_device < 0) && (dst_device >= 0))
	    stats.nbytes_h2g += nbytes;
	if ((src_device >= 0) && (dst_device < 0))
	    stats.nbytes_g2h += nbytes;

	if (stream)
	    CUDA_CALL(cudaStreamSynchronize(stream.get()));

	return stats;
    }
    
    virtual ~MemcpyWorker() { }
};


// -------------------------------------------------------------------------------------------------
//
// GpuCopyKernel


struct GpuCopyKernel : public FakeServer::Worker
{
    // string worker_name;    // inherited from 'Worker' base class

    int device = -1;
    long nbytes = 0;
    long blocksize = 0;
    
    shared_ptr<char> psrc;
    shared_ptr<char> pdst;

    // Reminder: cudaStream_t is a typedef for (CUstream_st *)
    shared_ptr<CUstream_st> stream;

    GpuCopyKernel(const shared_ptr<FakeServer::State> &state_,  const vector<int> &vcpu_list_, int device_, long nbytes_, long blocksize_):
	Worker(state_, vcpu_list_),
	device(device_),
	nbytes(nbytes_),
	blocksize(blocksize_)
    {
	xassert(nbytes > 0);
	xassert(blocksize > 0);
	xassert((nbytes % 128) == 0);
	xassert((blocksize % 128) == 0);
	
	stringstream ss;
	ss << "GpuCopyKernel(device= " << devstr(device)
	   << ", nbytes=" << ksgpu::nbytes_to_str(nbytes)
	   << ", blocksize=" << ksgpu::nbytes_to_str(blocksize)
	   << ")";

	this->worker_name = ss.str();
    }


    virtual void worker_initialize() override
    {
	CUDA_CALL(cudaSetDevice(this->device));  // initialize CUDA device in worker thread
	this->stream = ksgpu::CudaStreamWrapper().p;  // RAII stream

	this->psrc = worker_alloc(blocksize, true);  // on_gpu=true
	this->pdst = worker_alloc(blocksize, true);  // on_gpu=true
    }
    

    virtual Stats worker_body() override
    {
	long pos = 0;

	while (pos < nbytes) {
	    long n = min(nbytes - pos, blocksize);
	    ksgpu::launch_memcpy_kernel(pdst.get(), psrc.get(), n, stream.get());
	    pos += n;
	}

	CUDA_CALL(cudaStreamSynchronize(stream.get()));
	
	Stats stats;
	stats.nbytes_gmem = 2 * nbytes;
	return stats;
    }
    
    virtual ~GpuCopyKernel() { }
};


// -------------------------------------------------------------------------------------------------
//
// SsdWorker


struct SsdWorker : public FakeServer::Worker
{
    // string worker_name;    // inherited from 'Worker' base class

    string root_dir;
    long nfiles_per_iteration = 0;
    long nbytes_per_file = 0;
    long nbytes_per_write = 0;
    long ifile = 0;
    
    shared_ptr<char> data;

    SsdWorker(const shared_ptr<State> &state_, const vector<int> &vcpu_list_, const string &root_dir_, long nfiles_per_iteration_, long nbytes_per_file_, long nbytes_per_write_) :
	Worker(state_, vcpu_list_),
	root_dir(root_dir_),
	nfiles_per_iteration(nfiles_per_iteration_),
	nbytes_per_file(nbytes_per_file_),
	nbytes_per_write(nbytes_per_write_)
    {
	xassert(root_dir.size() > 0);
	xassert(nfiles_per_iteration > 0);
	xassert(nbytes_per_file > 0);
	xassert(nbytes_per_write > 0);
	
	stringstream ss;
	ss << "Ssd(dir=" << root_dir
	   << ", nfiles=" << nfiles_per_iteration
	   << ", nbytes_per_file=" << ksgpu::nbytes_to_str(nbytes_per_file)
	   << ", nbytes_per_write=" << ksgpu::nbytes_to_str(nbytes_per_write)
	   << ")";
	
	this->worker_name = ss.str();
	
	pirate::makedir(root_dir, false);  // throw_exception_if_directory_exists = false
    }

    virtual void worker_initialize() override
    {
	// FIXME should be random data, to avoid confusion from compressed filesystems
	data = worker_alloc(nbytes_per_file);
    }
    
    virtual Stats worker_body() override
    {
	// FIXME try writev(), if (nwrites_per_file > 1).
	// For now, I'm just calling write() in a loop, since this seems to give decent performance.

	long jfile = ifile + nfiles_per_iteration;
	
	while (ifile < jfile) {
	    stringstream ss;
	    ss << root_dir << "/file_" << ifile << "_" << ifile;
	    string filename = ss.str();

	    File f(filename, O_WRONLY | O_CREAT | O_TRUNC | O_DIRECT | O_SYNC);
	    long pos = 0;

	    while (pos < nbytes_per_file) {
		long n = min(nbytes_per_write, nbytes_per_file - pos);
		f.write(data.get() + pos, n);
		pos += n;
	    }

	    ifile++;
	}

	Stats stats;
	stats.nbytes_hmem = nfiles_per_iteration * nbytes_per_file;
	stats.nbytes_ssd = nfiles_per_iteration * nbytes_per_file;
	return stats;
    }

    virtual ~SsdWorker() { }
};


// -------------------------------------------------------------------------------------------------
//
// DownsamplingWorker


struct DownsamplingWorker : public FakeServer::Worker
{
    // string worker_name;    // inherited from 'Worker' base class

    int src_bit_depth = 0;
    long src_nelts = 0;
    long src_nbytes = 0;
    long dst_nbytes = 0;

    shared_ptr<char> psrc;
    shared_ptr<char> pdst;
    
    DownsamplingWorker(const shared_ptr<State> &state_, const vector<int> &vcpu_list_, int src_bit_depth_, long src_nelts_) :
	Worker(state_, vcpu_list_),
	src_bit_depth(src_bit_depth_),
	src_nelts(src_nelts_)
    {
	xassert((src_bit_depth >= 4) && (src_bit_depth <= 7));
	xassert((src_nelts % 16) == 0);

	this->src_nbytes = xdiv(src_nelts,8) * (src_bit_depth);
	this->dst_nbytes = xdiv(src_nelts,16) * (src_bit_depth+1);

	long nbytes_per_chunk = cpu_downsample_src_bytes_per_chunk(src_bit_depth);
	xassert((src_nbytes % nbytes_per_chunk) == 0);

	stringstream ss;
	ss << "Downsampling thread (bit depth " << src_bit_depth
	   << ", src=" << nbytes_to_str(src_nbytes)
	   << ", dst=" << nbytes_to_str(dst_nbytes)
	   << ")";

	this->worker_name = ss.str();
    }

    virtual void worker_initialize() override
    {
	this->psrc = worker_alloc(src_nbytes);
	this->pdst = worker_alloc(dst_nbytes);
    }
    
    virtual Stats worker_body() override
    {
	// cpu_downsample(int src_bit_depth, const uint8_t *src, uint8_t *dst, long src_nbytes, long dst_nbytes);
	cpu_downsample(src_bit_depth, reinterpret_cast<const uint8_t *> (psrc.get()), reinterpret_cast<uint8_t *> (pdst.get()), src_nbytes, dst_nbytes);

	Stats stats;
	stats.nbytes_hmem = src_nbytes + dst_nbytes;
	stats.nsamp_downsampled = src_nelts;
	return stats;
    }

    virtual ~DownsamplingWorker() { }
};


// -------------------------------------------------------------------------------------------------
//
// Add workers.


void FakeServer::add_receiver(const string &ip_addr, long num_tcp_connections, long recv_bufsize, bool use_epoll, long network_sync_cadence, const vector<int> &vcpu_list)
{
    auto wp = make_shared<Receiver> (state, vcpu_list, ip_addr, num_tcp_connections, recv_bufsize, use_epoll, network_sync_cadence);
    this->_add_worker(wp, "add_receiver");
}


void FakeServer::add_memcpy_worker(int src_device, int dst_device, long nbytes_per_iteration, long blocksize, const vector<int> &vcpu_list)
{
    auto wp = make_shared<MemcpyWorker> (state, vcpu_list, src_device, dst_device, nbytes_per_iteration, blocksize);
    this->_add_worker(wp, "add_memcpy_worker");
}

void FakeServer::add_gpu_copy_kernel(int device, long nbytes, long blocksize, const vector<int> &vcpu_list)
{
    auto wp = make_shared<GpuCopyKernel> (state, vcpu_list, device, nbytes, blocksize);
    this->_add_worker(wp, "add_gpu_copy_kernel");
}

void FakeServer::add_ssd_worker(const string &root_dir, long nfiles_per_iteration, long nbytes_per_file, long nbytes_per_write, const vector<int> &vcpu_list)
{
    auto wp = make_shared<SsdWorker> (state, vcpu_list, root_dir, nfiles_per_iteration, nbytes_per_file, nbytes_per_write);
    this->_add_worker(wp, "add_ssd_worker");
}


void FakeServer::add_downsampling_worker(int src_bit_depth, long src_nelts, const vector<int> &vcpu_list)
{
    auto wp = make_shared<DownsamplingWorker> (state, vcpu_list, src_bit_depth, src_nelts);
    this->_add_worker(wp, "add_downsampling_worker");
}


}  // namespace pirate
