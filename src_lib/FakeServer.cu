#include "../include/pirate/FakeServer.hpp"

#include <thread>
#include <sstream>
#include <iostream>
#include <condition_variable>

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


inline int vmin(const vector<int> &v)
{
    xassert(v.size() > 0);
    int ret = v[0];
    for (unsigned i = 1; i < v.size(); i++)
	ret = std::min(ret, v[i]);
    return ret;
}


shared_ptr<char> server_alloc(long nbytes, bool use_hugepages, bool on_gpu=false)
{
    xassert(nbytes > 0);
    
    int aflags = on_gpu ? ksgpu::af_gpu : ksgpu::af_rhost;
    // aflags |= af_verbose;

    if (use_hugepages && !on_gpu)
	aflags |= ksgpu::af_mmap_huge;
	
    return ksgpu::af_alloc<char> (nbytes, aflags | af_zero);
}

static int get_cuda_device_count()
{
    int count = 0;
    CUDA_CALL(cudaGetDeviceCount(&count));
    return count;
}


// -------------------------------------------------------------------------------------------------
//
// FakeServer::Receiver


struct ReceiverStats
{
    double elapsed_time = 0.0;
    long nbytes_read = 0;
    long num_read_calls = 0;
    long num_epoll_calls = 0;
};


struct FakeServer::Receiver
{
    string ip_addr;
    long num_tcp_connections = 0;
    long recv_bufsize = 0;
    bool use_epoll = true;
    long network_sync_cadence = 0;
    vector<int> vcpu_list;

    // Initialized in initialize().
    shared_ptr<char> recv_buf;
    Socket listening_socket;
    Epoll epoll;

    // Initialized in accept().
    vector<Socket> data_sockets;

    // Used to print summary statistics in announcer thread.
    std::mutex lock;
    ReceiverStats cumulative_stats;

    // Not protected by lock -- only accessed from receiver thread.
    bool tv_initialized = false;
    struct timeval tv_start;

    
    Receiver(const string &ip_addr_, long num_tcp_connections_, long recv_bufsize_, bool use_epoll_, long network_sync_cadence_, const vector<int> &vcpu_list_) :
	epoll(false)  // 'epoll' is constructed in an uninitialized state, and gets initialized in Receiver::initialize()
    {
	this->ip_addr = ip_addr_;
	this->num_tcp_connections = num_tcp_connections_;
	this->recv_bufsize = recv_bufsize_;
	this->use_epoll = use_epoll_;
	this->network_sync_cadence = network_sync_cadence_;
	this->vcpu_list = vcpu_list_;

	xassert(ip_addr.size() > 0);
	xassert(num_tcp_connections > 0);
        xassert(use_epoll || (num_tcp_connections == 1));
        xassert(recv_bufsize > 0);
        xassert(network_sync_cadence > 0);
    }
	
    
    void initialize(bool use_hugepages)
    {
	xassert(!recv_buf);  // detect double call to initialize().
	this->recv_buf = server_alloc(recv_bufsize, use_hugepages);

	this->listening_socket = Socket(PF_INET, SOCK_STREAM);
	this->listening_socket.set_reuseaddr();
	this->listening_socket.bind(ip_addr, 8787);  // TCP port 8787

	if (use_epoll)
	    this->epoll.initialize();
    }

    
    void accept()
    {
	xassert(recv_buf);    // detect call to accept() without initialize()
	xassert(data_sockets.size() == 0);  // detect double call to accept()
	xassert(num_tcp_connections > 0);
	
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
    }


    // Receives (network_sync_cadence) bytes of data, and updates this->cumulative_stats.
    void receive_data()
    {
	xassert(recv_buf);
	xassert(data_sockets.size() > 0);

	if (!tv_initialized) {
	    this->tv_start = ksgpu::get_time();
	    this->tv_initialized = true;
	}
	
	ReceiverStats stats;

	if (use_epoll)
	    _receive_epoll(stats);
	else
	    _receive_no_epoll(stats);

	std::unique_lock ul(lock);
	this->cumulative_stats.elapsed_time = ksgpu::time_since(tv_start);
	this->cumulative_stats.nbytes_read += stats.nbytes_read;
	this->cumulative_stats.num_read_calls += stats.num_read_calls;
	this->cumulative_stats.num_epoll_calls += stats.num_epoll_calls;
    }


    inline long _read_socket(uint32_t ids, ReceiverStats &stats)
    {
	xassert(ids < data_sockets.size());
	
	long max_read = std::min(recv_bufsize, network_sync_cadence - stats.nbytes_read);
	long nbytes_read = data_sockets[ids].read(recv_buf.get(), max_read);
	    
	stats.num_read_calls++;
	stats.nbytes_read += nbytes_read;

	return nbytes_read;
    }
    
    
    // Helper for receive_data(). (Called if use_epoll=false.)
    void _receive_no_epoll(ReceiverStats &stats)
    {
	xassert(data_sockets.size() == 1);
	
	while (stats.nbytes_read < network_sync_cadence) {
	    // Blocking read()
	    long nbytes_read = _read_socket(0, stats);
	    
	    if (nbytes_read == 0)
		throw runtime_error("TCP connection ended prematurely");
	}
    }


    // Helper for receive_iteration(). (Called if use_epoll=true.)
    void _receive_epoll(ReceiverStats &stats)
    {
	while (stats.nbytes_read < network_sync_cadence) {
	    long nbytes_prev = stats.nbytes_read;
	    
	    int num_events = epoll.wait();  // blocking
	    stats.num_epoll_calls++;

	    for (int iev = 0; iev < num_events; iev++) {
		uint32_t ev_flags = epoll.events[iev].events;

		if (!(ev_flags & EPOLLIN))
		    continue;

		// Non-blocking read()
		unsigned int ids = epoll.events[iev].data.u32;
		long nbytes_read = _read_socket(ids, stats);

		// FIXME read() can return zero, even with EPOLLIN set?!
		// I'd like to revisit this, just for the sake of general understanding.
		// It seems to be harmless, since the reported values of "bytes/read()" and "bytes/epoll()"
		// look reasonable.
		
		if (nbytes_read == 0)
		    continue;
		
		if (stats.nbytes_read >= network_sync_cadence)
		    return;
	    }

	    if ((stats.nbytes_read == nbytes_prev) && (num_events == int(data_sockets.size())))
		throw runtime_error("TCP connection(s) ended premaurely");
	}
    }


    // Called by announcer thread
    void show(int irecv, bool show_vcpus, bool show_stats)
    {
	ReceiverStats rs;
	
	cout << "    Receiver thread " << irecv << " [" << ip_addr << ", " << num_tcp_connections << " connections]";

	if (show_vcpus)
	    cout << ", vcpu_list=" << ksgpu::tuple_str(vcpu_list);

	if (show_stats) {
	    std::unique_lock ul(lock);
	    rs = this->cumulative_stats;
	    ul.unlock();
	}

	if (show_stats && (rs.nbytes_read == 0))
	    cout << ": insufficient data received" << endl;

	if (show_stats && (rs.nbytes_read > 0)) {
	    xassert(rs.elapsed_time > 0.0);
	    xassert(rs.num_read_calls > 0);

	    cout << ": Gbps=" << (8.0e-9 * rs.nbytes_read / rs.elapsed_time)
		 << ", bytes/read()=" << (rs.nbytes_read / double(rs.num_read_calls));
	    
	    if (rs.num_epoll_calls > 0)
		cout << ", bytes/epoll()=" << (rs.nbytes_read / double(rs.num_epoll_calls));
	}
	
	cout << endl;
    }
};
    

// -------------------------------------------------------------------------------------------------
//
// Workers start here


struct WorkerStats
{
    double total_time = 0.0;
    double active_time = 0.0;
    long num_iterations = 0;
};


struct FakeServer::Worker
{
    string worker_name = "Anonymous worker";
    vector<int> vcpu_list;

    // Per-iteration
    long nbytes_h2g = 0;
    long nbytes_g2h = 0;
    long nbytes_h2h = 0;
    long nbytes_gmem = 0;
    long nbytes_ssd = 0;

    Worker(const vector<int> &vcpu_list_) : vcpu_list(vcpu_list_) { }
    virtual ~Worker() { }
    
    // Called by worker thread.
    virtual void worker_initialize(bool use_hugepages) { }
    virtual void worker_body(int iter) = 0;
    
    // Noncopyable (use shared_ptr<Worker>)
    Worker(const Worker &) = delete;
    Worker &operator=(const Worker &) = delete;

    // Used to print summary statistics in announcer thread.
    std::mutex lock;
    WorkerStats cumulative_stats;


    // Helper function for show().
    static void _show_bandwidth(const string &label, const WorkerStats &ws, long nbytes_per_iter)
    {
	if (nbytes_per_iter <= 0)
	    return;

	double gb = 1.0e-9 * ws.num_iterations * nbytes_per_iter;	
	cout << ", " << label << " (active,net) = (" << (gb/ws.active_time) << ", " << (gb/ws.total_time) << ") GB/s";
    }

    void show(int iworker, bool show_vcpus, bool show_stats)
    {
	WorkerStats ws;
	
	cout << "    Worker thread " << iworker << ": " << worker_name;

	if (show_vcpus)
	    cout << ", vcpu_list=" << ksgpu::tuple_str(vcpu_list);

	if (show_stats) {
	    std::unique_lock ul(lock);
	    ws = this->cumulative_stats;
	    ul.unlock();
	    cout << ": " << ws.num_iterations << " iterations";
	}

	if (show_stats && (ws.num_iterations > 0)) {
	    xassert(ws.active_time > 0.0);
	    xassert(ws.total_time > 0.0);

	    cout << ", loadfrac=" << (ws.active_time / ws.total_time);
	    _show_bandwidth("host->gpu", ws, this->nbytes_h2g);
	    _show_bandwidth("gpu->host", ws, this->nbytes_g2h);
	    _show_bandwidth("host->host", ws, this->nbytes_h2h);
	    _show_bandwidth("gmem", ws, this->nbytes_gmem);
	    _show_bandwidth("ssd", ws, this->nbytes_ssd);
	}
	
	cout << endl;
    }
};


// -------------------------------------------------------------------------------------------------
//
// SleepyWorker: in each iteration, sleep for (sleep_usec)


struct SleepyWorker : public FakeServer::Worker
{
    long sleep_usec = 0;
    
    SleepyWorker(long sleep_usec_) :
	Worker({}),   // empty vcpu_list (no core-pinning)
	sleep_usec(sleep_usec_)
    {
	xassert(sleep_usec > 0);
	this->worker_name = "SleepyWorker(usec=" + to_string(sleep_usec) + ")";
    }

    virtual void worker_body(int iter) override
    {
	sys_usleep(sleep_usec);
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
    
    int src_device;   // -1 for host
    int dst_device;   // -1 for host

    long nbytes = 0;
    long blocksize = 0;
    
    shared_ptr<char> psrc;
    shared_ptr<char> pdst;

    // Reminder: cudaStream_t is a typedef for (CUstream_st *)
    shared_ptr<CUstream_st> stream;

    
    MemcpyWorker(int src_device_, int dst_device_, long nbytes_, long blocksize_, const vector<int> &vcpu_list) :
	Worker(vcpu_list)
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
	
	if ((src_device < 0) && (dst_device < 0))
	    this->nbytes_h2h = 2 * nbytes;
	else if (src_device < 0)
	    this->nbytes_h2g = this->nbytes_gmem = nbytes;
	else if (dst_device < 0)
	    this->nbytes_g2h = this->nbytes_gmem = nbytes;
	else
	    throw runtime_error("MemcpyWorker: device->device is currrently unsupported");
	
	stringstream ss;
	ss << "Memcpy(src=" << devstr(src_device)
	   << ", dst=" << devstr(dst_device)
	   << ", nbytes=" << ksgpu::nbytes_to_str(nbytes)
	   << ", blocksize=" << ksgpu::nbytes_to_str(blocksize)
	   << ")";
	    
	this->worker_name = ss.str();
    }


    // Helper for constructor
    static string devstr(int device)
    {
	if (device < 0)
	    return "host";
	
	stringstream ss;
	ss << "gpu" << device;
	return ss.str();
    }


    virtual void worker_initialize(bool use_hugepages) override
    {
	int dev = max(src_device, dst_device);
	int cuda_stream_priority = -1;  // lower numerical value = higher priority
	
	if (dev >= 0) {
	    CUDA_CALL(cudaSetDevice(dev));  // initialize CUDA device in worker thread
	    this->stream = ksgpu::CudaStreamWrapper(cuda_stream_priority).p;  // RAII stream
	    
#if 0
	    // Just curious what the allowed priority range is.
	    // (On an A40 it turned out to be [-5,0].)
	    int leastPriority = 137;
	    int greatestPriority = 137;
	    CUDA_CALL(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
	    cout << "XXX " << leastPriority << " " << greatestPriority << endl;
#endif

#if 0
	    int pri = 137;
	    CUDA_CALL(cudaStreamGetPriority(this->stream.get(), &pri));
	    cout << "XXX pri=" << pri << endl;
#endif
	}

	this->psrc = server_alloc(blocksize, use_hugepages, (src_device >= 0));
	this->pdst = server_alloc(blocksize, use_hugepages, (dst_device >= 0));
    }
    

    virtual void worker_body(int iter) override
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

	if (stream)
	    CUDA_CALL(cudaStreamSynchronize(stream.get()));
    }
    
    virtual ~MemcpyWorker() { }
};


// -------------------------------------------------------------------------------------------------
//
// GmemWorker: uses memory bandwidth on GPU, but from a kernel instead of cudaMemcpyDeviceToDevice().


struct GmemWorker : public FakeServer::Worker
{
    // string worker_name;    // inherited from 'Worker' base class

    int device = -1;
    long nbytes_copy = 0;   // = nbytes_gmem/2
    long blocksize = 0;
    
    shared_ptr<char> psrc;
    shared_ptr<char> pdst;

    // Reminder: cudaStream_t is a typedef for (CUstream_st *)
    shared_ptr<CUstream_st> stream;

    
    GmemWorker(int device_, long nbytes_per_iteration, long blocksize_, const vector<int> &vcpu_list) :
	Worker(vcpu_list)
    {
	this->device = device_;
	this->nbytes_copy = nbytes_per_iteration / 2;
	this->blocksize = (blocksize_ > 0) ? blocksize_ : nbytes_copy;
	this->blocksize = min(blocksize, nbytes_copy);

	xassert(nbytes_per_iteration > 0);
	xassert((nbytes_per_iteration % 256) == 0);
	xassert((blocksize % 128) == 0);
	
	stringstream ss;
	ss << "GmemKernel(device= " << device
	   << ", nbytes_per_iteration=" << ksgpu::nbytes_to_str(nbytes_per_iteration)
	   << ", blocksize=" << ksgpu::nbytes_to_str(blocksize)
	   << ")";

	// inherited from Worker base class.
	this->nbytes_gmem = nbytes_per_iteration;
	this->worker_name = ss.str();
    }


    virtual void worker_initialize(bool use_hugepages) override
    {
	CUDA_CALL(cudaSetDevice(this->device));  // initialize CUDA device in worker thread
	this->stream = ksgpu::CudaStreamWrapper().p;  // RAII stream

	int flags = ksgpu::af_gpu | ksgpu::af_zero;

	this->psrc = ksgpu::af_alloc<char> (blocksize, flags);
	this->pdst = ksgpu::af_alloc<char> (blocksize, flags);
    }
    

    virtual void worker_body(int iter) override
    {
	long nbytes_cumul = 0;

	while (nbytes_cumul < nbytes_copy) {
	    long nbytes_block = min(nbytes_copy - nbytes_cumul, blocksize);
	    ksgpu::launch_memcpy_kernel(pdst.get(), psrc.get(), nbytes_block, stream.get());
	    nbytes_cumul += nbytes_block;
	}

	CUDA_CALL(cudaStreamSynchronize(stream.get()));
    }
    
    virtual ~GmemWorker() { }
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
    
    shared_ptr<char> data;

    SsdWorker(const string &root_dir_, long nfiles_per_iteration_, long nbytes_per_file_, long nbytes_per_write_, const vector<int> &vcpu_list) :
	Worker(vcpu_list)
    {
	this->root_dir = root_dir_;
	this->nfiles_per_iteration = nfiles_per_iteration_;
	this->nbytes_per_file = nbytes_per_file_;
	this->nbytes_per_write = nbytes_per_write_;

	xassert(nfiles_per_iteration > 0);
	xassert(nbytes_per_file > 0);
	xassert(nbytes_per_write > 0);
	
	stringstream ss2;
	ss2 << "Ssd(dir=" << root_dir
	    << ", nfiles=" << nfiles_per_iteration
	    << ", nbytes_per_file=" << ksgpu::nbytes_to_str(nbytes_per_file)
	    << ", nbytes_per_write=" << ksgpu::nbytes_to_str(nbytes_per_write)
	    << ")";
	
	this->worker_name = ss2.str();
	
	pirate::makedir(root_dir, false);  // throw_exception_if_directory_exists = false
    }

    virtual void worker_initialize(bool use_hugepages) override
    {
	// FIXME should be random data, to avoid confusion from compressed filesystems
	data = server_alloc(nbytes_per_file, use_hugepages);
    }
    
    virtual void worker_body(int iter) override
    {
	// FIXME try writev(), if (nwrites_per_file > 1).
	// For now, I'm just calling write() in a loop, since this seems to give decent performance.
	
	for (int ifile = 0; ifile < nfiles_per_iteration; ifile++) {
	    stringstream ss;
	    ss << root_dir << "/file_" << iter << "_" << ifile;
	    string filename = ss.str();

	    File f(filename, O_WRONLY | O_CREAT | O_TRUNC | O_DIRECT | O_SYNC);
	    long pos = 0;

	    while (pos < nbytes_per_file) {
		long n = min(nbytes_per_write, nbytes_per_file - pos);
		f.write(data.get() + pos, n);
		pos += n;
	    }
	}
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
    long src_nbytes = 0;
    long dst_nbytes = 0;

    shared_ptr<char> psrc;
    shared_ptr<char> pdst;
    
    DownsamplingWorker(int src_bit_depth_, long src_nelts, const vector<int> &vcpu_list) :
	Worker(vcpu_list)
    {
	this->src_bit_depth = src_bit_depth_;
	xassert((src_bit_depth >= 4) && (src_bit_depth <= 7));
	xassert((src_nelts % 16) == 0);

	this->src_nbytes = xdiv(src_nelts, 8) * src_bit_depth;
	this->dst_nbytes = xdiv(src_nelts, 16) * (src_bit_depth+1);

	long nbytes_per_chunk = cpu_downsample_src_bytes_per_chunk(src_bit_depth);
	xassert((src_nbytes % nbytes_per_chunk) == 0);

	stringstream ss;
	ss << "Downsample(bit depth " << src_bit_depth
	   << ", src=" << nbytes_to_str(src_nbytes)
	   << ", dst=" << nbytes_to_str(dst_nbytes)
	   << ")";

	this->worker_name = ss.str();
    }

    virtual void worker_initialize(bool use_hugepages) override
    {
	this->psrc = server_alloc(src_nbytes, use_hugepages);
	this->pdst = server_alloc(dst_nbytes, use_hugepages);
    }
    
    virtual void worker_body(int iter) override
    {
	// cpu_downsample(int src_bit_depth, const uint8_t *src, uint8_t *dst, long src_nbytes, long dst_nbytes);
	cpu_downsample(src_bit_depth, reinterpret_cast<const uint8_t *> (psrc.get()), reinterpret_cast<uint8_t *> (pdst.get()), src_nbytes, dst_nbytes);
    }

    virtual ~DownsamplingWorker() { }
};


// -------------------------------------------------------------------------------------------------
//
// FakeServer


FakeServer::FakeServer(const string &server_name_, bool use_hugepages_) :
    server_name(server_name_), use_hugepages(use_hugepages_),
    barrier(0)   // will be initialized in run(), after number of threads is known
{ }


void FakeServer::add_receiver(const string &ip_addr, long num_tcp_connections, long recv_bufsize, bool use_epoll, long network_sync_cadence, const vector<int> &vcpu_list)
{
    auto rp = make_shared<Receiver> (ip_addr, num_tcp_connections, recv_bufsize, use_epoll, network_sync_cadence, vcpu_list);

    std::unique_lock lk(this->lock);
    
    if (running)
	throw runtime_error("FakeServer::add_receiver() called on running server");

    this->receivers.push_back(rp);
}


void FakeServer::add_memcpy_worker(int src_device, int dst_device, long nbytes_per_iteration, long blocksize, const vector<int> &vcpu_list)
{
    auto wp = make_shared<MemcpyWorker> (src_device, dst_device, nbytes_per_iteration, blocksize, vcpu_list);
    this->_add_worker(wp, "add_memcpy_worker");
}


void FakeServer::add_gmem_worker(int device, long nbytes_per_iteration, long blocksize, const vector<int> &vcpu_list)
{
    auto wp = make_shared<GmemWorker> (device, nbytes_per_iteration, blocksize, vcpu_list);
    this->_add_worker(wp, "add_gmem_worker");
}


void FakeServer::add_ssd_worker(const string &root_dir, long nfiles_per_iteration, long nbytes_per_file, long nbytes_per_write, const vector<int> &vcpu_list)
{
    auto wp = make_shared<SsdWorker> (root_dir, nfiles_per_iteration, nbytes_per_file, nbytes_per_write, vcpu_list);
    this->_add_worker(wp, "add_ssd_worker");
}


void FakeServer::add_downsampling_worker(int src_bit_depth, long src_nelts, const vector<int> &vcpu_list)
{
    auto wp = make_shared<DownsamplingWorker> (src_bit_depth, src_nelts, vcpu_list);
    this->_add_worker(wp, "add_downsampling_worker");
}


void FakeServer::add_sleepy_worker(long sleep_usec)
{
    auto wp = make_shared<SleepyWorker> (sleep_usec);
    this->_add_worker(wp, "add_sleepy_worker");
}


static void announcer_thread_main(FakeServer *server, long num_iterations)
{
    try {
	server->announcer_main(num_iterations);
    } catch (const exception &exc) {
	server->abort(exc.what());
    }
}


static void receiver_thread_main(FakeServer *server, int irecv, long num_iterations)
{
    try {
	server->receiver_main(irecv, num_iterations);
    } catch (const exception &exc) {
	server->abort(exc.what());
    }
}


static void worker_thread_main(FakeServer *server, int iworker, long num_iterations)
{
    try {
	server->worker_main(iworker, num_iterations);
    } catch (const exception &exc) {
	server->abort(exc.what());
    }
}


void FakeServer::run(long num_iterations)
{
    std::unique_lock lk(this->lock);
    
    if (running)
	throw runtime_error("FakeServer::run() called on running server");

    this->running = true;
    lk.unlock();
    
    int num_receivers = this->receivers.size();
    int num_workers = this->workers.size();
    int num_threads = num_receivers + num_workers + 1;

    if (num_workers == 0)
	throw runtime_error("FakeServer: No workers! You may need to add a SleepyWorker");
    
    this->barrier.initialize(num_threads);
    this->counters = vector<int> (workers.size(), 0);
    
    vector<std::thread> threads(num_threads);
    threads[0] = std::thread(announcer_thread_main, this, num_iterations);

    for (int i = 0; i < num_receivers; i++)
	threads[i+1] = std::thread(receiver_thread_main, this, i, num_iterations);
    
    for (int i = 0; i < num_workers; i++)
	threads[num_receivers+i+1] = std::thread(worker_thread_main, this, i, num_iterations);

    for (int i = 0; i < num_threads; i++)
	threads[i].join();

    lk.lock();
    this->running = false;
}


// -------------------------------------------------------------------------------------------------


void FakeServer::_add_worker(const shared_ptr<Worker> &wp, const string &caller)
{
    std::unique_lock lk(this->lock);

    if (running)
	throw runtime_error("FakeServer::" + caller + "() called on running server");

    this->workers.push_back(wp);
}


// Increments counter[ix] from (expected_value) -> (expected_value+1).
void FakeServer::increment_counter(int ix, int expected_value)
{
    std::unique_lock<mutex> lk(lock);
    xassert((ix >= 0) && (ix < int(counters.size())));
    xassert(counters[ix] == expected_value);
    
    if (aborted)
	throw runtime_error(abort_msg);
    
    this->counters[ix] = expected_value+1;
    this->min_counter = vmin(counters);
    
    bool notify = (min_counter == expected_value+1);
    lk.unlock();
    
    if (notify)
	cv.notify_all();
}


// Waits until all counters are >= threshold. Returns actual minimum of all counters (must be >= threshold)
int FakeServer::wait_for_counters(int threshold)
{
    std::unique_lock<mutex> lk(lock);
    
    auto f = [this,threshold] { return this->aborted || (this->min_counter >= threshold); };
    cv.wait(lk, f);
    
    if (aborted)
	throw runtime_error(abort_msg);
    
    return min_counter;
}


int FakeServer::peek_at_counter()
{
    std::unique_lock ul(lock);
    return this->min_counter;
}
    
    
void FakeServer::abort(const string &msg)
{
    this->barrier.abort(msg);
    
    std::unique_lock<mutex> lk(lock);
	
    if (aborted)
	return;
    
    this->aborted = true;
    this->abort_msg = msg;
    cout << msg << endl;
    
    lk.unlock();
    cv.notify_all();
}


void FakeServer::receiver_main(int irecv, long num_iterations)
{
    int num_receivers = receivers.size();
    xassert((irecv >= 0) && (irecv < num_receivers));
    
    shared_ptr<Receiver> receiver = receivers[irecv];
    xassert(receiver);

    // No-ops if 'vcpu_list' is empty.
    pin_thread_to_vcpus(receiver->vcpu_list);
    
    receiver->initialize(use_hugepages);
    barrier.wait();  // wait at barrier [1/3]
    
    receiver->accept();
    barrier.wait();  // wait at barrier [2/3]
    
    while (this->peek_at_counter() < num_iterations)
	receiver->receive_data();
    
    barrier.wait();  // wait at barrier [3/3]
}


void FakeServer::worker_main(int iworker, long num_iterations)
{
    int num_workers = workers.size();
    xassert((iworker >= 0) && (iworker < num_workers));
    
    shared_ptr<Worker> worker = workers[iworker];
    xassert(worker);

    // No-ops if 'vcpu_list' is empty.
    pin_thread_to_vcpus(worker->vcpu_list);
    
    worker->worker_initialize(use_hugepages);
    barrier.wait();  // wait at barrier [1/3]
    barrier.wait();  // wait at barrier [2/3]
    
    struct timeval tv_start = ksgpu::get_time();
	
    for (int niter = 0; niter < num_iterations; niter++) {
	struct timeval tv0 = ksgpu::get_time();
	worker->worker_body(niter);
	struct timeval tv1 = ksgpu::get_time();
	
	std::unique_lock ul(worker->lock);
	worker->cumulative_stats.total_time = ksgpu::time_diff(tv_start, tv1);
	worker->cumulative_stats.active_time += ksgpu::time_diff(tv0, tv1);
	worker->cumulative_stats.num_iterations = niter+1;
	ul.unlock();
	
	// Advance counter niter -> (niter+1).
	increment_counter(iworker, niter);
	
	// Wait until all counters reach (niter+1).
	wait_for_counters(niter+1);
    }
    
    barrier.wait();  // wait at barrier [3/3]
}


// Helper for announcer_main()
void FakeServer::_show_all(bool show_vcpus, bool show_stats)
{
    for (unsigned int irecv = 0; irecv < receivers.size(); irecv++)
	receivers[irecv]->show(irecv, show_vcpus, show_stats);
    
    for (unsigned int iworker = 0; iworker < workers.size(); iworker++)
	workers[iworker]->show(iworker, show_vcpus, show_stats);
}
    

void FakeServer::announcer_main(long num_iterations)
{
    _show_all(true, false);   // show_vcpus=true, show_stats=false
    
    cout << server_name << ": initializing (use_hugepages=" << use_hugepages << ")" << endl;
    barrier.wait();  // wait at barrier [1/3]
    
    if (receivers.size() > 0)
	cout << server_name << ": receiver threads accepting connections" << endl;
    
    barrier.wait();  // wait at barrier [2/3]

    struct timeval tv0 = get_time();
    cout << server_name << ": running!" << endl;
    
    for (int niter = 1; niter <= num_iterations; niter++) {
	wait_for_counters(niter);
	double dt = time_since(tv0);
	cout << "Iteration " << niter << " done, average time/iteration = " << (dt/niter) << " seconds" << endl;
	_show_all(false, true);   // show_vcpus=false, show_stats=true
    }
    
    barrier.wait();  // wait at barrier [3/3]
}



}  // namespace pirate
