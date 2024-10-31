#include "../include/pirate/internals/FakeServer.hpp"

#include <thread>
#include <sstream>
#include <iostream>
#include <condition_variable>

#include <gputils/Barrier.hpp>
#include <gputils/mem_utils.hpp>
#include <gputils/cuda_utils.hpp>
#include <gputils/time_utils.hpp>
#include <gputils/string_utils.hpp>
#include <gputils/system_utils.hpp>
#include <gputils/memcpy_kernels.hpp>

#include "../include/pirate/internals/File.hpp"
#include "../include/pirate/internals/Epoll.hpp"
#include "../include/pirate/internals/Socket.hpp"
#include "../include/pirate/internals/inlines.hpp"
#include "../include/pirate/internals/cpu_downsample.hpp"


using namespace std;
using namespace gputils;


namespace pirate {
#if 0
}  // editor auto-indent
#endif


inline int vmin(const vector<int> &v)
{
    assert(v.size() > 0);
    int ret = v[0];
    for (unsigned i = 1; i < v.size(); i++)
	ret = std::min(ret, v[i]);
    return ret;
}


shared_ptr<char> server_alloc(ssize_t nbytes, bool use_hugepages, bool on_gpu=false)
{
    assert(nbytes > 0);
    
    int aflags = on_gpu ? gputils::af_gpu : gputils::af_rhost;
    // aflags |= af_verbose;

    if (use_hugepages && !on_gpu)
	aflags |= gputils::af_mmap_huge;
	
    return gputils::af_alloc<char> (nbytes, aflags);
}


// -------------------------------------------------------------------------------------------------
//
// FakeServer::Receiver


struct ReceiverStats
{
    double elapsed_time = 0.0;
    ssize_t nbytes_read = 0;
    ssize_t num_read_calls = 0;
    ssize_t num_epoll_calls = 0;
};


struct FakeServer::Receiver
{
    const FakeServer::Params params;
    string ip_addr;

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

    
    Receiver(const FakeServer::Params &params_, int irecv) :
	params(params_),
	epoll(false)  // 'epoll' is constructed in an uninitialized state, and gets initialized in Receiver::initialize()
    {
	int num_ipaddr = params.ipaddr_list.size();
	assert((irecv >= 0) && (irecv < num_ipaddr));
	assert(params.nconn_per_ipaddr > 0);
	assert(params.recv_bufsize > 0);
	assert(params.network_sync_cadence > 0);
	
	this->ip_addr = params.ipaddr_list[irecv];
    }
	
    
    void initialize()
    {
	assert(!recv_buf);  // detect double call to initialize().
	this->recv_buf = server_alloc(params.recv_bufsize, params.use_hugepages);

	this->listening_socket = Socket(PF_INET, SOCK_STREAM);
	this->listening_socket.set_reuseaddr();
	this->listening_socket.bind(this->ip_addr, 8787);  // TCP port 8787

	if (params.use_epoll)
	    this->epoll.initialize();
    }

    
    void accept()
    {
	assert(recv_buf);    // detect call to accept() without initialize()
	assert(data_sockets.size() == 0);  // detect double call to accept()
	assert(params.nconn_per_ipaddr > 0);
	
	this->data_sockets.resize(params.nconn_per_ipaddr);
	this->listening_socket.listen();

	// Later: setsockopt(SO_RCVBUF), setsockopt(SO_SNDBUF) here?
	// I tried SO_RECVLOWAT here, but uProf reported significantly higher memory bandwidth!

	for (unsigned int ids = 0; ids < data_sockets.size(); ids++) {
	    this->data_sockets[ids] = listening_socket.accept();

	    if (!params.use_epoll)
		continue;
	    
	    this->data_sockets[ids].set_nonblocking();

	    epoll_event ev;
	    ev.events = EPOLLIN | EPOLLRDHUP | EPOLLHUP;
	    ev.data.u32 = ids;
	    epoll.add_fd(data_sockets[ids].fd, ev);
	}
    }


    // Receives (params.network_sync_cadence) bytes of data, and updates this->cumulative_stats.
    void receive_data()
    {
	assert(recv_buf);
	assert(data_sockets.size() > 0);

	if (!tv_initialized) {
	    this->tv_start = gputils::get_time();
	    this->tv_initialized = true;
	}
	
	ReceiverStats stats;

	if (params.use_epoll)
	    _receive_epoll(stats);
	else
	    _receive_no_epoll(stats);

	std::unique_lock ul(lock);
	this->cumulative_stats.elapsed_time = gputils::time_since(tv_start);
	this->cumulative_stats.nbytes_read += stats.nbytes_read;
	this->cumulative_stats.num_read_calls += stats.num_read_calls;
	this->cumulative_stats.num_epoll_calls += stats.num_epoll_calls;
    }


    inline ssize_t _read_socket(uint32_t ids, ReceiverStats &stats)
    {
	assert(ids < data_sockets.size());
	
	ssize_t max_read = std::min(params.recv_bufsize, params.network_sync_cadence - stats.nbytes_read);
	ssize_t nbytes_read = data_sockets[ids].read(recv_buf.get(), max_read);
	    
	stats.num_read_calls++;
	stats.nbytes_read += nbytes_read;

	return nbytes_read;
    }
    
    
    // Helper for receive_data(). (Called if use_epoll=false.)
    void _receive_no_epoll(ReceiverStats &stats)
    {
	assert(data_sockets.size() == 1);
	
	while (stats.nbytes_read < params.network_sync_cadence) {
	    // Blocking read()
	    ssize_t nbytes_read = _read_socket(0, stats);
	    
	    if (nbytes_read == 0)
		throw runtime_error("TCP connection ended prematurely");
	}
    }


    // Helper for receive_iteration(). (Called if use_epoll=true.)
    void _receive_epoll(ReceiverStats &stats)
    {
	while (stats.nbytes_read < params.network_sync_cadence) {
	    ssize_t nbytes_prev = stats.nbytes_read;
	    
	    int num_events = epoll.wait();  // blocking
	    stats.num_epoll_calls++;

	    for (int iev = 0; iev < num_events; iev++) {
		uint32_t ev_flags = epoll.events[iev].events;

		if (!(ev_flags & EPOLLIN))
		    continue;

		// Non-blocking read()
		unsigned int ids = epoll.events[iev].data.u32;
		ssize_t nbytes_read = _read_socket(ids, stats);

		// FIXME read() can return zero, even with EPOLLIN set?!
		// I'd like to revisit this, just for the sake of general understanding.
		// It seems to be harmless, since the reported values of "bytes/read()" and "bytes/epoll()"
		// look reasonable.
		
		if (nbytes_read == 0)
		    continue;
		
		if (stats.nbytes_read >= params.network_sync_cadence)
		    return;
	    }

	    if ((stats.nbytes_read == nbytes_prev) && (num_events == int(data_sockets.size())))
		throw runtime_error("TCP connection(s) ended premaurely");
	}
    }


    // Called by announcer thread
    void show(int irecv, bool show_stats=true)
    {
	cout << "    Receiver thread " << irecv << " [" << ip_addr << ", " << params.nconn_per_ipaddr << " connections]";

	if (!show_stats) {
	    cout << endl;
	    return;
	}

	std::unique_lock ul(lock);
	ReceiverStats rs = this->cumulative_stats;
	ul.unlock();

	if (rs.nbytes_read == 0) {
	    cout << ": insufficient data received" << endl;
	    return;
	}

	assert(rs.elapsed_time > 0.0);
	assert(rs.num_read_calls > 0);

	cout << ": Gbps=" << (8.0e-9 * rs.nbytes_read / rs.elapsed_time)
	     << ", bytes/read()=" << (rs.nbytes_read / double(rs.num_read_calls));
	
	if (rs.num_epoll_calls > 0)
	    cout << ", bytes/epoll()=" << (rs.nbytes_read / double(rs.num_epoll_calls));

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
    ssize_t num_iterations = 0;
};


struct FakeServer::Worker
{
    FakeServer::Params params;
    string worker_name = "Anonymous worker";

    // Per-iteration
    ssize_t nbytes_h2g = 0;
    ssize_t nbytes_g2h = 0;
    ssize_t nbytes_h2h = 0;
    ssize_t nbytes_gmem = 0;
    ssize_t nbytes_ssd = 0;
    
    Worker(const FakeServer::Params &params_) : params(params_) { }
    virtual ~Worker() { }
    
    // Called by worker thread.
    virtual void worker_initialize() { }
    virtual void worker_body(int iter) = 0;
    
    // Noncopyable (use shared_ptr<Worker>)
    Worker(const Worker &) = delete;
    Worker &operator=(const Worker &) = delete;

    // Used to print summary statistics in announcer thread.
    std::mutex lock;
    WorkerStats cumulative_stats;


    // Helper function for show().
    static void _show_bandwidth(const string &label, const WorkerStats &ws, ssize_t nbytes_per_iter)
    {
	if (nbytes_per_iter <= 0)
	    return;

	double gb = 1.0e-9 * ws.num_iterations * nbytes_per_iter;	
	cout << ", " << label << " (active,net) = (" << (gb/ws.active_time) << ", " << (gb/ws.total_time) << ") GB/s";
    }

    void show(int iworker, bool show_stats=true)
    {
	cout << "    Worker thread " << iworker << ": " << worker_name;

	if (!show_stats) {
	    cout << endl;
	    return;
	}

	std::unique_lock ul(lock);
	WorkerStats ws = this->cumulative_stats;
	ul.unlock();

	cout << ": " << ws.num_iterations << " iterations";
	
	if (ws.num_iterations == 0) {
	    cout << endl;
	    return;
	}

	assert(ws.active_time > 0.0);
	assert(ws.total_time > 0.0);

	cout << ", loadfrac=" << (ws.active_time / ws.total_time);
	_show_bandwidth("host->gpu", ws, this->nbytes_h2g);
	_show_bandwidth("gpu->host", ws, this->nbytes_g2h);
	_show_bandwidth("host->host", ws, this->nbytes_h2h);
	_show_bandwidth("gmem", ws, this->nbytes_gmem);
	_show_bandwidth("ssd", ws, this->nbytes_ssd);
	
	cout << endl;
    }
};


// -------------------------------------------------------------------------------------------------
//
// SleepyWorker: in each iteration, sleep for (params.sleep_usec)


struct SleepyWorker : public FakeServer::Worker
{
    SleepyWorker(const FakeServer::Params &params_)
	: FakeServer::Worker(params_)
    {
	assert(params.sleep_usec > 0);
	this->worker_name = "SleepyWorker(usec=" + to_string(params.sleep_usec) + ")";
    }

    virtual void worker_body(int iter) override
    {
	usleep_x(params.sleep_usec);
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
    
    const int src_device;   // -1 for host
    const int dst_device;   // -1 for host

    ssize_t nbytes = 0;
    ssize_t blocksize = 0;
    ssize_t nblocks = 0;
    
    shared_ptr<char> psrc;
    shared_ptr<char> pdst;

    // Reminder: cudaStream_t is a typedef for (CUstream_st *)
    shared_ptr<CUstream_st> stream;

    
    MemcpyWorker(const FakeServer::Params &params_, int src_device_, int dst_device_) :
	FakeServer::Worker(params_), src_device(src_device_), dst_device(dst_device_)
    {
	if ((src_device < 0) && (dst_device < 0))
	    this->nbytes_h2h = this->nbytes = params.nbytes_h2h;
	else if (src_device < 0)
	    this->nbytes_h2g = this->nbytes_gmem = this->nbytes = params.nbytes_h2g;
	else if (dst_device < 0)
	    this->nbytes_g2h = this->nbytes_gmem = this->nbytes = params.nbytes_g2h;
	else
	    throw runtime_error("MemcpyWorker: device->device is currrently unsupported");

	this->blocksize = (params.memcpy_blocksize > 0) ? params.memcpy_blocksize : nbytes;
	this->nblocks = xdiv(nbytes, blocksize);
	
	stringstream ss;
	ss << "Memcpy(src=" << devstr(src_device)
	   << ", dst=" << devstr(dst_device)
	   << ", nbytes=" << gputils::nbytes_to_str(nbytes)
	   << ", blocksize=" << gputils::nbytes_to_str(blocksize)
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


    virtual void worker_initialize() override
    {
	int dev = max(src_device, dst_device);
	int cuda_stream_priority = -1;  // lower numerical value = higher priority
	
	if (dev >= 0) {
	    CUDA_CALL(cudaSetDevice(dev));  // initialize CUDA device in worker thread
	    this->stream = gputils::CudaStreamWrapper(cuda_stream_priority).p;  // RAII stream
	    
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

	this->psrc = server_alloc(nbytes, params.use_hugepages, (src_device >= 0));
	this->pdst = server_alloc(nbytes, params.use_hugepages, (dst_device >= 0));
    }
    

    virtual void worker_body(int iter) override
    {
	for (ssize_t i = 0; i < nblocks; i++) {
	    char *dp = pdst.get() + i * blocksize;
	    const char *sp = psrc.get() + i * blocksize;
	    
	    if (stream)
		CUDA_CALL(cudaMemcpyAsync(dp, sp, blocksize, cudaMemcpyDefault, stream.get()));
	    else
		memcpy(dp, sp, blocksize);
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

    const int device;

    ssize_t nbytes_copy = 0;
    ssize_t blocksize = 0;
    
    shared_ptr<char> psrc;
    shared_ptr<char> pdst;

    // Reminder: cudaStream_t is a typedef for (CUstream_st *)
    shared_ptr<CUstream_st> stream;

    
    GmemWorker(const FakeServer::Params &params_, int device_) :
	FakeServer::Worker(params_), device(device_)
    {
	this->nbytes_copy = xdiv(params.nbytes_gmem_kernel, 2);
	this->blocksize = params.gmem_kernel_blocksize;

	if (blocksize <= 0)
	    blocksize = 4L * 1024L * 1024L * 1024L;

	blocksize = min(blocksize, nbytes_copy);
	
	stringstream ss;
	ss << "GmemKernel(device= " << device
	   << ", nbytes=" << gputils::nbytes_to_str(2 * nbytes_copy)
	   << ", blocksize=" << gputils::nbytes_to_str(blocksize)
	   << ")";

	this->nbytes_gmem = 2 * this->nbytes_copy;
	this->worker_name = ss.str();
    }


    virtual void worker_initialize() override
    {
	CUDA_CALL(cudaSetDevice(this->device));  // initialize CUDA device in worker thread
	this->stream = gputils::CudaStreamWrapper().p;  // RAII stream

	int flags = gputils::af_gpu | gputils::af_zero;

	this->psrc = gputils::af_alloc<char> (blocksize, flags);
	this->pdst = gputils::af_alloc<char> (blocksize, flags);
    }
    

    virtual void worker_body(int iter) override
    {
	ssize_t nbytes_cumul = 0;

	while (nbytes_cumul < nbytes_copy) {
	    ssize_t nbytes_block = min(nbytes_copy - nbytes_cumul, blocksize);
	    gputils::launch_memcpy_kernel(pdst.get(), psrc.get(), nbytes_block, stream.get());
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
    shared_ptr<char> data;
    ssize_t nfiles_per_iteration = 0;
    ssize_t nbytes_per_write = 0;

    SsdWorker(const FakeServer::Params &params_, const string &ssd, int ithread) :
	FakeServer::Worker(params_)
    {
	this->nfiles_per_iteration = xdiv(params.nbytes_per_ssd, params.nthreads_per_ssd * params.nbytes_per_file);
	this->nbytes_per_write = xdiv(params.nbytes_per_file, params.nwrites_per_file);
	this->nbytes_ssd = nfiles_per_iteration * params.nbytes_per_file;

	stringstream ss;
	ss << ssd << "/thread_" << ithread;
	this->root_dir = ss.str();

	stringstream ss2;
	ss2 << "Ssd(dir=" << root_dir
	    << ", nfiles=" << nfiles_per_iteration
	    << ", nbytes_per_file=" << gputils::nbytes_to_str(params.nbytes_per_file)
	    << ", nwrites_per_file=" << params.nwrites_per_file
	    << ")";
	
	this->worker_name = ss2.str();
	
	// Create directory
	gputils::mkdir_x(root_dir);
    }

    virtual void worker_initialize() override
    {
	// FIXME should be random data, to avoid confusion from compressed filesystems
	data = server_alloc(params.nbytes_per_file, params.use_hugepages);
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

	    for (int iwrite = 0; iwrite < params.nwrites_per_file; iwrite++)
		f.write(data.get() + iwrite * nbytes_per_write, nbytes_per_write);
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

    const int src_bit_depth;
    
    ssize_t src_nbytes = 0;
    ssize_t dst_nbytes = 0;

    shared_ptr<char> psrc;
    shared_ptr<char> pdst;

    
    DownsamplingWorker(const FakeServer::Params &params_, int src_bit_depth_) :
	FakeServer::Worker(params_), src_bit_depth(src_bit_depth_)
    {
	assert((src_bit_depth >= 4) && (src_bit_depth <= 7));

	ssize_t src_nelts = xdiv(params.nbytes_downsample, (1 << (src_bit_depth-4)));
	this->src_nbytes = xdiv(src_nelts, 8) * src_bit_depth;
	this->dst_nbytes = xdiv(src_nelts, 16) * (src_bit_depth+1);

	ssize_t nbytes_per_chunk = cpu_downsample_src_bytes_per_chunk(src_bit_depth);
	assert((src_nbytes % nbytes_per_chunk) == 0);

	stringstream ss;
	ss << "Downsample(bit depth " << src_bit_depth
	   << ", src=" << nbytes_to_str(src_nbytes)
	   << ", dst=" << nbytes_to_str(dst_nbytes)
	   << ")";

	this->worker_name = ss.str();
    }

    virtual void worker_initialize() override
    {
	this->psrc = server_alloc(src_nbytes, params.use_hugepages);
	this->pdst = server_alloc(dst_nbytes, params.use_hugepages);
    }
    
    virtual void worker_body(int iter) override
    {
	// cpu_downsample(int src_bit_depth, const uint8_t *src, uint8_t *dst, ssize_t src_nbytes, ssize_t dst_nbytes);
	cpu_downsample(src_bit_depth, reinterpret_cast<const uint8_t *> (psrc.get()), reinterpret_cast<uint8_t *> (pdst.get()), src_nbytes, dst_nbytes);
    }

    virtual ~DownsamplingWorker() { }
};


// -------------------------------------------------------------------------------------------------
//
// FakeServer


FakeServer::FakeServer(const Params &params_)
    : params(params_), barrier(0)
{
    // ------------------------------  Error checking  ------------------------------
    
    assert(params.num_iterations > 0);
    
    bool gpus_requested = (params.nbytes_h2g > 0) || (params.nbytes_g2h > 0) || (params.nbytes_gmem_kernel > 0);
    bool nics_requested = (params.nconn_per_ipaddr > 0) || (params.ipaddr_list.size() > 0);
    bool ssds_requested = (params.nbytes_per_ssd > 0);
    
    if (gpus_requested) {
	if (params.ngpu < 0) {
	    CUDA_CALL(cudaGetDeviceCount(&params.ngpu));
	    cout << params.server_name << ": " << params.ngpu << " GPU(s) detected in cudaGetDeviceCount()" << endl;
	    assert(params.ngpu >= 0);
	}
	
	if (params.ngpu == 0)
	    throw runtime_error("GPUs requested, but either params.ngpu=0, or no GPUs were detected");
    }
    
    if (nics_requested) {
	// FIXME are these fully asserted in Receiver methods?
	assert(params.ipaddr_list.size() > 0);
	assert(params.nconn_per_ipaddr > 0);
	assert(params.use_epoll || (params.nconn_per_ipaddr == 1));
	assert(params.recv_bufsize > 0);
	assert(params.network_sync_cadence > 0);
    }
    
    if (ssds_requested) {
	assert(params.nthreads_per_ssd > 0);
	assert(params.nbytes_per_file > 0);
	assert(params.nwrites_per_file > 0);
	assert(params.ssd_list.size() > 0);
	assert((params.nbytes_per_ssd % (params.nthreads_per_ssd * params.nbytes_per_file)) == 0);
	assert((params.nbytes_per_file % params.nwrites_per_file) == 0);
    }
    
    if (params.memcpy_blocksize > 0) {
	assert((params.nbytes_h2h % params.memcpy_blocksize) == 0);
	assert((params.nbytes_g2h % params.memcpy_blocksize) == 0);
	assert((params.nbytes_h2g % params.memcpy_blocksize) == 0);
    }

    if (params.nbytes_gmem_kernel > 0) {
	assert((params.nbytes_gmem_kernel % 256) == 0);
	assert(params.gmem_kernel_blocksize >= 0);
	assert((params.gmem_kernel_blocksize % 128) == 0);
    }
    
    // ------------------------------  Receivers and workers  ------------------------------
    
    if (params.ipaddr_list.size() > 0) {
	for (unsigned int irecv = 0; irecv < params.ipaddr_list.size(); irecv++)
	    receivers.push_back(make_shared<Receiver> (params, irecv));
    }

    if (params.nbytes_h2h > 0)
	workers.push_back(make_shared<MemcpyWorker> (params, -1, -1));
    
    if (params.nbytes_h2g > 0) {
	for (int igpu = 0; igpu < params.ngpu; igpu++)
	    workers.push_back(make_shared<MemcpyWorker> (params, -1, igpu));
    }
    
    if (params.nbytes_g2h > 0) {
	for (int igpu = 0; igpu < params.ngpu; igpu++)	    
	    workers.push_back(make_shared<MemcpyWorker> (params, igpu, -1));
    }

    if (params.nbytes_gmem_kernel > 0) {
	for (int igpu = 0; igpu < params.ngpu; igpu++)	    
	    workers.push_back(make_shared<GmemWorker> (params, igpu));
    }
    
    if (params.nbytes_downsample > 0) {
	for (int src_bit_depth = 4; src_bit_depth <= 7; src_bit_depth++)
	    workers.push_back(make_shared<DownsamplingWorker> (params, src_bit_depth));
    }
    
    if (params.nbytes_per_ssd > 0) {
	for (const string &ssd: params.ssd_list)
	    for (int ithread = 0; ithread < params.nthreads_per_ssd; ithread++)
		workers.push_back(make_shared<SsdWorker> (params, ssd, ithread));
    }
    
    if (params.sleep_usec > 0)
	workers.push_back(make_shared<SleepyWorker> (params));
    
    if (workers.size() == 0)
	throw runtime_error("FakeServer: No workers! You may need to add a SleepyWorker, by setting 'params.sleep_usec'");
    
    // ------------------------------  Synchronization  ------------------------------
    
    int num_threads = receivers.size() + workers.size() + 1;
    
    this->barrier.initialize(num_threads);
    this->counters = vector<int> (workers.size(), 0);
}


// Increments counter[ix] from (expected_value) -> (expected_value+1).
void FakeServer::increment_counter(int ix, int expected_value)
{
    std::unique_lock<mutex> lk(lock);
    assert((ix >= 0) && (ix < int(counters.size())));
    assert(counters[ix] == expected_value);
    
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


void FakeServer::receiver_main(int irecv)
{
    int num_receivers = receivers.size();
    assert((irecv >= 0) && (irecv < num_receivers));
    
    shared_ptr<Receiver> receiver = receivers[irecv];
    assert(receiver);
    
    receiver->initialize();
    barrier.wait();  // wait at barrier [1/3]
    
    receiver->accept();
    barrier.wait();  // wait at barrier [2/3]
    
    while (this->peek_at_counter() < params.num_iterations)
	receiver->receive_data();
    
    barrier.wait();  // wait at barrier [3/3]
}


void FakeServer::worker_main(int iworker)
{
    int num_workers = workers.size();;
    assert((iworker >= 0) && (iworker < num_workers));
    
    shared_ptr<Worker> worker = workers[iworker];
    assert(worker);
    
    worker->worker_initialize();
    barrier.wait();  // wait at barrier [1/3]
    barrier.wait();  // wait at barrier [2/3]
    
    struct timeval tv_start = gputils::get_time();
	
    for (int niter = 0; niter < params.num_iterations; niter++) {
	struct timeval tv0 = gputils::get_time();
	worker->worker_body(niter);
	struct timeval tv1 = gputils::get_time();
	
	std::unique_lock ul(worker->lock);
	worker->cumulative_stats.total_time = gputils::time_diff(tv_start, tv1);
	worker->cumulative_stats.active_time += gputils::time_diff(tv0, tv1);
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
void FakeServer::_show_all(bool show_stats)
{
    for (unsigned int irecv = 0; irecv < receivers.size(); irecv++)
	receivers[irecv]->show(irecv, show_stats);
    
    for (unsigned int iworker = 0; iworker < workers.size(); iworker++)
	workers[iworker]->show(iworker, show_stats);
}
    

void FakeServer::announcer_main()
{
    _show_all(false);   // show_stats=false
    
    cout << params.server_name << ": initializing" << endl;
    barrier.wait();  // wait at barrier [1/3]
    
    if (receivers.size() > 0)
	cout << params.server_name << ": receiver threads accepting connections" << endl;
    
    barrier.wait();  // wait at barrier [2/3]

    struct timeval tv0 = get_time();
    cout << params.server_name << ": running!" << endl;
    
    for (int niter = 1; niter <= params.num_iterations; niter++) {
	wait_for_counters(niter);
	double dt = time_since(tv0);
	cout << "Iteration " << niter << " done, average time/iteration = " << (dt/niter) << " seconds" << endl;
	_show_all();
    }
    
    barrier.wait();  // wait at barrier [3/3]
}


static void announcer_thread_main(shared_ptr<FakeServer> server)
{
    try {
	server->announcer_main();
    } catch (const exception &exc) {
	server->abort(exc.what());
    }
}


static void receiver_thread_main(shared_ptr<FakeServer> server, int irecv)
{
    try {
	server->receiver_main(irecv);
    } catch (const exception &exc) {
	server->abort(exc.what());
    }
}


static void worker_thread_main(shared_ptr<FakeServer> server, int iworker)
{
    try {
	server->worker_main(iworker);
    } catch (const exception &exc) {
	server->abort(exc.what());
    }
}


// Static member function.
void FakeServer::run(const FakeServer::Params &params)
{
    shared_ptr<FakeServer> server = make_shared<FakeServer> (params);
    int num_receivers = server->receivers.size();
    int num_workers = server->workers.size();
    int num_threads = num_receivers + num_workers + 1;
    
    vector<std::thread> threads(num_threads);
    threads[0] = std::thread(announcer_thread_main, server);

    for (int i = 0; i < num_receivers; i++)
	threads[i+1] = std::thread(receiver_thread_main, server, i);
    
    for (int i = 0; i < num_workers; i++)
	threads[num_receivers+i+1] = std::thread(worker_thread_main, server, i);

    for (int i = 0; i < num_threads; i++)
	threads[i].join();
}


}  // namespace pirate
