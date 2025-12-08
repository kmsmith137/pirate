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
#include "../include/pirate/trackers.hpp"       // BandwidthTracker
#include "../include/pirate/file_utils.hpp"     // File, listdir()
#include "../include/pirate/system_utils.hpp"
#include "../include/pirate/network_utils.hpp"  // Socket, Epoll
#include "../include/pirate/loose_ends/cpu_downsample.hpp"
#include "../include/pirate/Dedisperser.hpp"


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


struct FakeServer::Stats
{
    static constexpr int max_cpu = 2;
    static constexpr int max_gpu = 8;
    static constexpr int max_ssd = 8;
    static constexpr int max_nic = 8;
    
    // Hardare bandwidths.
    double hmem_floating = 0;     // host memory bandwidth for "floating" threads (not pinned to a specific CPU)
    double hmem_pinned[max_cpu];  // host memory bandwidth for pinned threads
    double gmem[max_gpu];         // GPU memory bandwidth
    double h2g[max_gpu];          // host -> GPU
    double g2h[max_gpu];          // GPU -> host
    double ssd[max_ssd];
    double nic[max_nic];

    // FRB-specific.
    double ds_kernel = 0;
    double chime_beams = 0.0;

    Stats();
    Stats &operator+=(const Stats &s);

    double &hmem(int cpu);
};


FakeServer::Stats::Stats()
{
    memset(this, 0, sizeof(*this));
}


FakeServer::Stats &FakeServer::Stats::operator+=(const Stats &s)
{
    double *dst = reinterpret_cast<double *> (this);
    const double *src = reinterpret_cast<const double *> (&s);

    for (uint i = 0; (i * sizeof(double)) < sizeof(*this); i++)
        dst[i] += src[i];
    
    return *this;
}


FakeServer::Stats operator/(const FakeServer::Stats &s, double x)
{
    FakeServer::Stats ret;
    xassert(x != 0);

    double *dst = reinterpret_cast<double *> (&ret);
    const double *src = reinterpret_cast<const double *> (&s);

    for (uint i = 0; (i * sizeof(double)) < sizeof(ret); i++)
        dst[i] = src[i] / x;

    return ret;
}


double &FakeServer::Stats::hmem(int icpu)
{
    xassert(icpu < max_cpu);
    return (icpu >= 0) ? hmem_pinned[icpu] : hmem_floating;
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

    // Only used for updating Stats::hmem[].
    // A negative value indicates a "floating" worker thread, which can run on multiple CPUs.
    int cpu = -1;

    std::mutex lock;
    struct timeval tv0;
    struct timeval tv1;
    bool tv_initialized = false;
    Stats cumulative_stats;

    Worker(const shared_ptr<FakeServer::State> &state_, const vector<int> vcpu_list_, int cpu_);
    virtual ~Worker() { }
    
    // Called by worker thread.
    virtual void worker_initialize() = 0;
    virtual void worker_accept_connections() { }   // only Receiver subclass defines this
    virtual Stats worker_body() = 0;

    shared_ptr<char> worker_alloc(long nbytes, bool on_gpu=false);
    string devstr(int device);
};


FakeServer::Worker::Worker(const shared_ptr<FakeServer::State> &state_, const vector<int> vcpu_list_, int cpu_) :
    state(state_), vcpu_list(vcpu_list_), cpu(cpu_)
{
    xassert(state);
    xassert(cpu < FakeServer::Stats::max_cpu);  // negative value is allowed (see above)
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
            ss << "  [Thread " << i << "] " << wp->worker_name << ": ";
            if (wp->cpu >= 0)
                ss << "cpu=" << wp->cpu;
            else
                ss << "cpu=None";
            ss << ", vcpu_list=" << ksgpu::tuple_str(wp->vcpu_list) << "\n";
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
    using Stats = FakeServer::Stats;

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
    
    Stats bw;
    double dt_max = 0.0;

    for (long i = 0; i < num_workers; i++) {
        dt_max = std::max(dt_max, dt[i]);
        
        // Don't trust rate estimates below 0.5 sec.
        if (dt[i] >= 0.5)
            bw += stats[i] / dt[i];
    }

    stringstream ss;

    for (int cpu = 0; cpu < FakeServer::Stats::max_cpu; cpu++)
        if (bw.hmem_pinned[cpu] > 0.0)
            ss << "  Host memory bandwidth (CPU " << cpu << ", estimated not measured): " << (1.0e-9 * bw.hmem_pinned[cpu]) << " GB/s\n";

    if (bw.hmem_floating > 0.0)
        ss << "  Host memory bandwidth (floating): " << (1.0e-9 * bw.hmem_floating) << " GB/s\n";
        
    for (int gpu = 0; gpu < FakeServer::Stats::max_gpu; gpu++)
        if (bw.gmem[gpu] > 0.0)
            ss << "  GPU global memory bandwidth (GPU " << gpu << "): " << (1.0e-9 * bw.gmem[gpu]) << " GB/s\n";

    for (int gpu = 0; gpu < FakeServer::Stats::max_gpu; gpu++)
        if (bw.h2g[gpu] > 0.0)
            ss << "  Host->GPU bandwidth (GPU " << gpu << "): " << (1.0e-9 * bw.h2g[gpu]) << " GB/s\n";

    for (int gpu = 0; gpu < FakeServer::Stats::max_gpu; gpu++)
        if (bw.g2h[gpu] > 0.0)
            ss << "  GPU->Host bandwidth (GPU " << gpu << "): " << (1.0e-9 * bw.g2h[gpu]) << " GB/s\n";

    for (int issd = 0; issd < FakeServer::Stats::max_ssd; issd++)
        if (bw.ssd[issd] > 0.0)
            ss << "  SSD bandwidth (SSD " << issd << "): " << (1.0e-9 * bw.ssd[issd]) << " GB/s\n";

    for (int inic = 0; inic < FakeServer::Stats::max_nic; inic++)
        if (bw.nic[inic] > 0.0)
            ss << "  Network bandwidth (NIC " << inic << "): " << (8.0e-9 * bw.nic[inic]) << " Gbps (not GB/s)\n";
    
    if (bw.ds_kernel > 0.0)
        ss << "  AVX2 kernel throughput: " << (1.0e-9 * bw.ds_kernel) << " Gsamp/s\n";
    if (bw.chime_beams > 0.0)
        ss << "  Real-time CHIME beams: " << bw.chime_beams << "\n";

    string s = ss.str();
    
    if (s.size() > 0)
        cout << "Elapsed time: " << dt_max << " sec\n" << s << flush;

    return dt_max;
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
    int inic = -1;
    
    long nbytes_per_iteration = 300 * 1024 * 1024;
    
    // Initialized in worker_initialize().
    shared_ptr<char> recv_buf;
    Socket listening_socket;
    Epoll epoll;

    // Initialized in worker_accept_connections().
    vector<Socket> data_sockets;

    
    Receiver(const shared_ptr<FakeServer::State> state_, const vector<int> &vcpu_list_, int cpu_, int inic_, const string &ip_addr_, long num_tcp_connections_, long recv_bufsize_, bool use_epoll_) :
        Worker(state_, vcpu_list_, cpu_),
        ip_addr(ip_addr_),
        num_tcp_connections(num_tcp_connections_),
        recv_bufsize(recv_bufsize_),
        use_epoll(use_epoll_),
        inic(inic_),
        epoll(false)  // 'epoll' is constructed in an uninitialized state, and gets initialized in worker_initialize()
    {
        xassert(ip_addr.size() > 0);
        xassert(num_tcp_connections > 0);
        xassert(use_epoll || (num_tcp_connections == 1));
        xassert(recv_bufsize > 0);
        xassert((inic >= 0) && (inic < FakeServer::Stats::max_nic));

        stringstream ss;
        ss << "TcpReceiver(" << ip_addr << ", " << num_tcp_connections << " connections, use_epoll=" << use_epoll << ")";
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


    virtual Stats worker_body() override
    {
        return use_epoll ? _receive_epoll() : _receive_no_epoll();
    }

    
    // Helper for receive_data(). (Called if use_epoll=false.)
    Stats _receive_no_epoll()
    {
        xassert(data_sockets.size() == 1);      
        long nbytes_cumul = 0;
        
        while (nbytes_cumul < nbytes_per_iteration) {
            // Blocking read()
            long nbytes_read = data_sockets[0].read(recv_buf.get(), recv_bufsize);
            xassert(nbytes_read >= 0);
            
            if (nbytes_read <= 0)
                throw runtime_error("TCP connection ended prematurely");
            
            nbytes_cumul += nbytes_read;            
        }
        
        Stats stats;
        stats.nic[inic] = nbytes_cumul;
        stats.hmem(cpu) += 3 * nbytes_cumul;  // note factor 3 from "non-zerocopy" TCP
        return stats;
    }


    // Helper for receive_iteration(). (Called if use_epoll=true.)
    Stats _receive_epoll()
    {
        long nbytes_cumul = 0;
        
        while (nbytes_cumul < nbytes_per_iteration) {
            long nbytes_save = nbytes_cumul;
            int num_events = epoll.wait();  // blocking

            for (int iev = 0; iev < num_events; iev++) {
                uint32_t ev_flags = epoll.events[iev].events;

                if (!(ev_flags & EPOLLIN))
                    continue;

                unsigned int ids = epoll.events[iev].data.u32;
                xassert(ids < data_sockets.size());
                
                // Non-blocking read()
                long nbytes_read = data_sockets[ids].read(recv_buf.get(), recv_bufsize);
                xassert(nbytes_read >= 0);

                // FIXME read() can return zero, even with EPOLLIN set?!
                // I'd like to revisit this, just for the sake of general understanding.
                // It seems to be harmless, since the reported values of "bytes/read()" and "bytes/epoll()"
                // look reasonable.
                
                if (nbytes_read == 0)
                    continue;

                nbytes_cumul += nbytes_read;
            }

            if ((nbytes_cumul == nbytes_save) && (num_events == int(data_sockets.size())))
                throw runtime_error("TCP connection(s) ended prematurely");
        }
        
        Stats stats;
        stats.nic[inic] += nbytes_cumul;
        stats.hmem(cpu) += 3 * nbytes_cumul;  // note factor 3 from "non-zerocopy" TCP
        return stats;
    }
};
    

// -------------------------------------------------------------------------------------------------
//
// ChimeWorker


struct ChimeWorker : public FakeServer::Worker
{
    ChimeDedisperser dedisperser;
    long niter = 0;
    long ichunk = 0;
    int device = -1;

    
    ChimeWorker(const shared_ptr<FakeServer::State> state_, const vector<int> &vcpu_list_, int cpu_, int device_, int beams_per_gpu, int num_active_batches, int beams_per_batch, bool use_copy_engine) :
        Worker(state_, vcpu_list_, cpu_),
        dedisperser(beams_per_gpu, num_active_batches, beams_per_batch, use_copy_engine),
        device(device_)
    {
        xassert(device >= 0);
        xassert(device < FakeServer::Stats::max_gpu);
        xassert(device < get_cuda_device_count());

        xassert(beams_per_gpu > 0);  // paranoid -- also checked in ChimeDedisperser constructor
        this->niter = (beams_per_gpu + 99) / beams_per_gpu;
        
        stringstream ss;
        ss << "ChimeDedisperser(" << devstr(device) << ", use_copy_engine=" << use_copy_engine << ")";
        this->worker_name = ss.str();
    }

    virtual void worker_initialize() override
    {
        // Must precede ChimeDedisperser::initialize().
        CUDA_CALL(cudaSetDevice(this->device));
        dedisperser.initialize();
    }

    virtual Stats worker_body() override
    {
        for (long i = 0; i < niter; i++)
            dedisperser.run(ichunk++);
        
        Stats stats;
        stats.chime_beams = niter * dedisperser.config.beams_per_gpu * (1.0e-3 * dedisperser.config.time_samples_per_chunk);
        stats.gmem[device] += niter * dedisperser.bw_per_run_call.nbytes_gmem;
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
    // Initialized in constructor.
    int src_device = -1;   // -1 for host
    int dst_device = -1;   // -1 for host
    long blocksize = 0;    
    long nbytes_per_iteration = 0;
    bool use_copy_engine = false;

    // Initialized in worker_initialize().
    shared_ptr<char> psrc;
    shared_ptr<char> pdst;

    // Reminder: cudaStream_t is a typedef for (CUstream_st *)
    CudaStreamWrapper stream;

    
    MemcpyWorker(const shared_ptr<FakeServer::State> &state_, const vector<int> &vcpu_list_, int cpu_, int src_device_, int dst_device_, long blocksize_, bool use_copy_engine_) :
        Worker(state_, vcpu_list_, cpu_),
        src_device(src_device_),
        dst_device(dst_device_),
        blocksize(blocksize_),
        use_copy_engine(use_copy_engine_)
    {
        int num_gpus_in_machine = get_cuda_device_count();
        xassert(num_gpus_in_machine < FakeServer::Stats::max_gpu);

        if (src_device >= 0)
            xassert_lt(src_device, num_gpus_in_machine);
        if (dst_device >= 0)
            xassert_lt(dst_device, num_gpus_in_machine);
        if ((src_device >= 0) && (dst_device >= 0) && (src_device != dst_device))
            throw runtime_error("MemcpyWorker: copy between different GPUs is currently unsupported");

        xassert(blocksize > 0);
        xassert((blocksize % 128) == 0);

        if ((src_device < 0) && (dst_device < 0))
            this->nbytes_per_iteration = 1024L * 1024L * 1024L;        // host->host
        else if ((src_device >= 0) && (dst_device >= 0))
            this->nbytes_per_iteration = 100L * 1024L * 1024L * 1024L;  // device->device
        else
            this->nbytes_per_iteration = 4L * 1024L * 1024L * 1024L;

        stringstream ss;
        ss << "MemcpyThread(src=" << devstr(src_device)
           << ", dst=" << devstr(dst_device)
           << ", blocksize=" << ksgpu::nbytes_to_str(blocksize);

        if ((src_device >= 0) && (dst_device >= 0))
            ss << ", use_copy_engine=" << use_copy_engine;
        
        ss << ")";
        this->worker_name = ss.str();
    }


    virtual void worker_initialize() override
    {
        bool uses_gpu = (src_device >= 0) || (dst_device >= 0);
        bool uses_host = (src_device < 0) || (dst_device < 0);
        
        if (uses_gpu) {
            int gpu = max(src_device, dst_device);
            int cuda_stream_priority = uses_host ? -1 : 0;  // lower numerical value = higher priority
            CUDA_CALL(cudaSetDevice(gpu));                  // initialize CUDA device in worker thread
            this->stream = CudaStreamWrapper::create(cuda_stream_priority);             // RAII stream
        }

        this->psrc = worker_alloc(blocksize, (src_device >= 0));  // last argument is boolean 'on_gpu'
        this->pdst = worker_alloc(blocksize, (dst_device >= 0));  // last argument is boolean 'on_gpu'
    }
    

    virtual Stats worker_body() override
    {
        long pos = 0;
        
        while (pos < nbytes_per_iteration) {
            long n = min(nbytes_per_iteration - pos, blocksize);
            pos += n;

            if (!stream)
                memcpy(pdst.get(), psrc.get(), n);
            else if (use_copy_engine || (src_device < 0) || (dst_device < 0))
                CUDA_CALL(cudaMemcpyAsync(pdst.get(), psrc.get(), n, cudaMemcpyDefault, stream));
            else
                ksgpu::launch_memcpy_kernel(pdst.get(), psrc.get(), n, stream);
        }

        Stats stats;

        if (src_device < 0)
            stats.hmem(cpu) += nbytes_per_iteration;
        if (dst_device < 0)
            stats.hmem(cpu) += nbytes_per_iteration;
        
        if (src_device >= 0)
            stats.gmem[src_device] += nbytes_per_iteration;
        if (dst_device >= 0)
            stats.gmem[dst_device] += nbytes_per_iteration;

        if ((src_device < 0) && (dst_device >= 0))
            stats.h2g[dst_device] += nbytes_per_iteration;
        if ((src_device >= 0) && (dst_device < 0))
            stats.g2h[src_device] += nbytes_per_iteration;

        if (stream)
            CUDA_CALL(cudaStreamSynchronize(stream));

        return stats;
    }
    
    virtual ~MemcpyWorker() { }
};


// -------------------------------------------------------------------------------------------------
//
// SsdWorker


struct SsdWorker : public FakeServer::Worker
{
    // string worker_name;    // inherited from 'Worker' base class

    string root_dir;
    long nbytes_per_file = 0;
    int issd = -1;
    
    long nfiles_per_iteration = 0;
    long curr_file = 0;
    
    shared_ptr<char> data;

    SsdWorker(const shared_ptr<State> &state_, const vector<int> &vcpu_list_, int cpu_, int issd_, const string &root_dir_, long nbytes_per_file_) :
        Worker(state_, vcpu_list_, cpu_),
        root_dir(root_dir_),
        nbytes_per_file(nbytes_per_file_),
        issd(issd_)
    {
        xassert(root_dir.size() > 0);
        xassert(nbytes_per_file > 0);
        xassert((issd >= 0) && (issd < FakeServer::Stats::max_ssd));

        this->nfiles_per_iteration = (nbytes_per_file + 256L*1024L*1024L - 1) / nbytes_per_file;
        
        stringstream ss;
        ss << "SsdWriter(dir=" << root_dir
           << ", nbytes_per_file=" << ksgpu::nbytes_to_str(nbytes_per_file)
           << ", nfiles_per_iteration=" << nfiles_per_iteration
           << ")";
        
        this->worker_name = ss.str();
    }

    virtual void worker_initialize() override
    {
        // FIXME should be random data, to avoid confusion from compressed filesystems
        data = worker_alloc(nbytes_per_file);

        // listdir() is defined in pirate/file_utils.cu
        vector<string> all_files = listdir(root_dir);
        vector<string> files_to_delete;

        for (const string &filename: all_files) {
            if (is_stale_file(filename))
                files_to_delete.push_back(filename);
        }

        if (files_to_delete.size() == 0)
            return;

        stringstream ss;
        ss << "SsdWriter(" << root_dir << "): deleting " << files_to_delete.size() << " stale files from previous run\n";
        cout << ss.str() << flush;

        for (const string &filename: files_to_delete) {
            // delete_file() is defined in pirate/file_utils.cu
            stringstream ss2;
            ss2 << root_dir << "/" << filename;
            delete_file(ss2.str());
        }
    }


    // Helper for worker_initialize().
    // Returns 'true' if filename is of the form 'file_NNN'
    bool is_stale_file(const string &filename)
    {
        const char *s = filename.c_str();
        int len = strlen(s);

        if (len < 6)
            return false;
        
        if (memcmp(s, "file_", 5))
            return false;
        
        for (int i = 5; i < len; i++)
            if (!isdigit(s[i]))
                return false;

        return true;
    }
        
        
    virtual Stats worker_body() override
    {
        // FIXME try writev(), if (nwrites_per_file > 1).
        // For now, I'm just calling write() in a loop, since this seems to give decent performance.

        for (int i = 0; i < nfiles_per_iteration; i++) {
            stringstream ss;
            ss << root_dir << "/file_" << (curr_file++);
            string filename = ss.str();

            File f(filename, O_WRONLY | O_CREAT | O_TRUNC | O_DIRECT | O_SYNC);
            f.write(data.get(), nbytes_per_file);
        }

        Stats stats;
        stats.hmem(cpu) = nfiles_per_iteration * nbytes_per_file;
        stats.ssd[issd] = nfiles_per_iteration * nbytes_per_file;
        return stats;
    }

    virtual ~SsdWorker() { }
};


// -------------------------------------------------------------------------------------------------
//
// DownsamplingWorker


struct DownsamplingWorker : public FakeServer::Worker
{
    int src_bit_depth = 0;
    long src_nelts = 0;
    long src_nbytes = 0;
    long dst_nbytes = 0;

    shared_ptr<char> psrc;
    shared_ptr<char> pdst;
    
    DownsamplingWorker(const shared_ptr<State> &state_, const vector<int> &vcpu_list_, int cpu_, int src_bit_depth_, long src_nelts_) :
        Worker(state_, vcpu_list_, cpu_),
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
        ss << "Avx2Downsampler(bit_depth=" << src_bit_depth
           << ", src=" << nbytes_to_str(src_nbytes)
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
        stats.hmem(cpu) = src_nbytes + dst_nbytes;
        stats.ds_kernel = src_nelts;
        return stats;
    }

    virtual ~DownsamplingWorker() { }
};


// -------------------------------------------------------------------------------------------------
//
// Add workers.


void FakeServer::add_tcp_receiver(const string &ip_addr, long num_tcp_connections, long recv_bufsize, bool use_epoll, const vector<int> &vcpu_list, int cpu, int inic)
{
    auto wp = make_shared<Receiver> (state, vcpu_list, cpu, inic, ip_addr, num_tcp_connections, recv_bufsize, use_epoll);
    this->_add_worker(wp, "add_tcp_receiver");
}

void FakeServer::add_chime_dedisperser(int device, int beams_per_gpu, int num_active_batches, int beams_per_batch, bool use_copy_engine, const vector<int> &vcpu_list, int cpu)
{
    auto wp = make_shared<ChimeWorker> (state, vcpu_list, cpu, device, beams_per_gpu, num_active_batches, beams_per_batch, use_copy_engine);
    this->_add_worker(wp, "add_chime_dedisperser");
}

void FakeServer::add_memcpy_thread(int src_device, int dst_device, long blocksize, bool use_copy_engine, const vector<int> &vcpu_list, int cpu)
{
    auto wp = make_shared<MemcpyWorker> (state, vcpu_list, cpu, src_device, dst_device, blocksize, use_copy_engine);
    this->_add_worker(wp, "add_memcpy_thread");
}

void FakeServer::add_ssd_writer(const string &root_dir, long nbytes_per_file, const vector<int> &vcpu_list, int cpu, int issd)
{
    auto wp = make_shared<SsdWorker> (state, vcpu_list, cpu, issd, root_dir, nbytes_per_file);
    this->_add_worker(wp, "add_ssd_writer");
}

void FakeServer::add_downsampling_thread(int src_bit_depth, long src_nelts, const vector<int> &vcpu_list, int cpu)
{
    auto wp = make_shared<DownsamplingWorker> (state, vcpu_list, cpu, src_bit_depth, src_nelts);
    this->_add_worker(wp, "add_downsampling_thread");
}


}  // namespace pirate
