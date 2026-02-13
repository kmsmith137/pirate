#include "../include/pirate/Hwtest.hpp"

#include <mutex>
#include <thread>
#include <sstream>
#include <iostream>

#include <ksgpu/mem_utils.hpp>
#include <ksgpu/cuda_utils.hpp>
#include <ksgpu/time_utils.hpp>
#include <ksgpu/string_utils.hpp>
#include <ksgpu/memcpy_kernels.hpp>
#include <ksgpu/xassert.hpp>

#include "../include/pirate/inlines.hpp"
#include "../include/pirate/file_utils.hpp"     // File, remove_file()
#include "../include/pirate/system_utils.hpp"
#include "../include/pirate/network_utils.hpp"  // Socket, Epoll
#include "../include/pirate/loose_ends/cpu_downsample.hpp"
#include "../include/pirate/Dedisperser.hpp"
#include "../include/pirate/DedispersionPlan.hpp"
#include "../include/pirate/BumpAllocator.hpp"
#include "../include/pirate/CudaStreamPool.hpp"
#include "../include/pirate/Barrier.hpp"


namespace fs = std::filesystem;

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
// Hwtest::Stats


struct Hwtest::Stats
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


Hwtest::Stats::Stats()
{
    memset(this, 0, sizeof(*this));
}


Hwtest::Stats &Hwtest::Stats::operator+=(const Stats &s)
{
    double *dst = reinterpret_cast<double *> (this);
    const double *src = reinterpret_cast<const double *> (&s);

    for (uint i = 0; (i * sizeof(double)) < sizeof(*this); i++)
        dst[i] += src[i];
    
    return *this;
}


Hwtest::Stats operator/(const Hwtest::Stats &s, double x)
{
    Hwtest::Stats ret;
    xassert(x != 0);

    double *dst = reinterpret_cast<double *> (&ret);
    const double *src = reinterpret_cast<const double *> (&s);

    for (uint i = 0; (i * sizeof(double)) < sizeof(ret); i++)
        dst[i] = src[i] / x;

    return ret;
}


double &Hwtest::Stats::hmem(int icpu)
{
    xassert(icpu < max_cpu);
    return (icpu >= 0) ? hmem_pinned[icpu] : hmem_floating;
}
        

// -------------------------------------------------------------------------------------------------
//
// Hwtest::Worker


struct Hwtest::Worker
{
    using Stats = Hwtest::Stats;

    // Safe to hold bare pointer: Hwtest destructor joins all worker threads.
    Hwtest *hwtest = nullptr;

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

    Worker(Hwtest *hwtest_, const vector<int> vcpu_list_, int cpu_);
    virtual ~Worker() { }

    // Called by worker thread.
    virtual void worker_initialize() = 0;
    virtual void worker_accept_connections() { }   // only TcpReceiver subclass defines this
    virtual Stats worker_body() = 0;

    shared_ptr<char> worker_alloc(long nbytes, bool on_gpu=false);
    string devstr(int device);
};


Hwtest::Worker::Worker(Hwtest *hwtest_, const vector<int> vcpu_list_, int cpu_) :
    hwtest(hwtest_), vcpu_list(vcpu_list_), cpu(cpu_)
{
    xassert(hwtest);
    xassert(cpu < Hwtest::Stats::max_cpu);  // negative value is allowed (see above)
}


shared_ptr<char> Hwtest::Worker::worker_alloc(long nbytes, bool on_gpu)
{
    xassert(nbytes > 0);

    int aflags = on_gpu ? ksgpu::af_gpu : ksgpu::af_rhost;
    // aflags |= af_verbose;

    if (hwtest->use_hugepages && !on_gpu)
        aflags |= ksgpu::af_mmap_huge;

    return ksgpu::af_alloc<char> (nbytes, aflags | af_zero);
}


string Hwtest::Worker::devstr(int device)
{
    if (device < 0)
        return "host";
        
    stringstream ss;
    ss << "gpu" << device;
    return ss.str();
}


// -------------------------------------------------------------------------------------------------
//
// Hwtest core logic. This is the difficult part!


Hwtest::Hwtest(const std::string &server_name_, bool use_hugepages_) :
    server_name(server_name_),
    use_hugepages(use_hugepages_),
    barrier(0)  // will be initialized in Hwtest::start(), after worker thread count is known
{ }


shared_ptr<Hwtest> Hwtest::create(const std::string &server_name, bool use_hugepages)
{
    return shared_ptr<Hwtest> (new Hwtest(server_name, use_hugepages));
}


void Hwtest::_throw_if_stopped(const char *method_name)
{
    if (error)
        std::rethrow_exception(error);

    if (is_stopped)
        throw runtime_error(string(method_name) + " called on stopped instance");
}


void Hwtest::_add_worker(const shared_ptr<Worker> &wp, const string &caller)
{
    std::lock_guard lk(this->mutex);
    _throw_if_stopped(caller.c_str());

    if (is_started)
        throw runtime_error(caller + " called after start()");

    this->workers.push_back(wp);
}    


// Called by Hwtest::start().
static void worker_thread_main(Hwtest *hwtest, shared_ptr<Hwtest::Worker> worker)
{
    using Stats = Hwtest::Stats;

    try {
        // No-ops if 'vcpu_list' is empty.
        set_thread_affinity(worker->vcpu_list);

        // Reminder: worker threads wait at the barrier twice.
        //  - After returning from worker_initialize(), before calling worker_accept_connections()
        //  - After returning from worker_accept_connections(), before calling worker_body().

        worker->worker_initialize();
        hwtest->barrier.wait();

        worker->worker_accept_connections();
        hwtest->barrier.wait();

        for (;;) {
            std::unique_lock lk(hwtest->mutex);

            if (hwtest->is_stopped)
                return;

            lk.unlock();

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
    catch (...) {
        hwtest->stop(std::current_exception());
    }
}


// Called from python, to start server.
void Hwtest::start()
{
    {
        std::lock_guard lk(this->mutex);
        _throw_if_stopped("Hwtest::start");

        if (is_started)
            throw runtime_error("Hwtest::start() called twice");
        if (workers.empty())
            throw runtime_error("Hwtest does not contain any worker threads");

        is_started = true;
        barrier.initialize(workers.size());
    }

    try {
        long num_workers = workers.size();
        threads.resize(num_workers);

        // Reminder: after server is started, 'workers' is immutable, so we don't need to hold the lock here.
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

            threads[i] = std::thread(worker_thread_main, this, wp);
        }
    }
    catch (...) {
        this->stop(std::current_exception());
        throw;
    }
}


double Hwtest::show_stats()
{
    unique_lock lock(this->mutex);
    _throw_if_stopped("Hwtest::show_stats");
    
    if (!is_started)
        throw runtime_error("Hwtest::show_stats() called on server which has not been started");

    lock.unlock();

    try {
        return _show_stats();
    } catch (...) {
        this->stop(std::current_exception());
        throw;
    }
}


double Hwtest::_show_stats()
{
    using Stats = Hwtest::Stats;

    // Reminder: after server is started, 'workers' is immutable, so we don't need to hold the lock here...
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

    for (int cpu = 0; cpu < Hwtest::Stats::max_cpu; cpu++)
        if (bw.hmem_pinned[cpu] > 0.0)
            ss << "  Host memory bandwidth (CPU " << cpu << ", estimated not measured): " << (1.0e-9 * bw.hmem_pinned[cpu]) << " GB/s\n";

    if (bw.hmem_floating > 0.0)
        ss << "  Host memory bandwidth (floating): " << (1.0e-9 * bw.hmem_floating) << " GB/s\n";

    for (int gpu = 0; gpu < Hwtest::Stats::max_gpu; gpu++)
        if (bw.gmem[gpu] > 0.0)
            ss << "  GPU global memory bandwidth (GPU " << gpu << "): " << (1.0e-9 * bw.gmem[gpu]) << " GB/s\n";

    for (int gpu = 0; gpu < Hwtest::Stats::max_gpu; gpu++)
        if (bw.h2g[gpu] > 0.0)
            ss << "  Host->GPU bandwidth (GPU " << gpu << "): " << (1.0e-9 * bw.h2g[gpu]) << " GB/s\n";

    for (int gpu = 0; gpu < Hwtest::Stats::max_gpu; gpu++)
        if (bw.g2h[gpu] > 0.0)
            ss << "  GPU->Host bandwidth (GPU " << gpu << "): " << (1.0e-9 * bw.g2h[gpu]) << " GB/s\n";

    for (int issd = 0; issd < Hwtest::Stats::max_ssd; issd++)
        if (bw.ssd[issd] > 0.0)
            ss << "  SSD bandwidth (SSD " << issd << "): " << (1.0e-9 * bw.ssd[issd]) << " GB/s\n";

    for (int inic = 0; inic < Hwtest::Stats::max_nic; inic++)
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


void Hwtest::stop(std::exception_ptr e)
{
    std::lock_guard lk(this->mutex);
    if (is_stopped)
        return;
    is_stopped = true;
    error = e;
    barrier.stop(e);
}


Hwtest::~Hwtest()
{
    stop();

    for (auto &t : threads)
        if (t.joinable())
            t.join();
}


void Hwtest::join()
{
    std::exception_ptr e;

    {
        std::lock_guard lk(this->mutex);
        if (!is_stopped)
            throw runtime_error("Hwtest::join() called before stop()");
        e = error;
    }

    for (auto &t : threads)
        if (t.joinable())
            t.join();

    // Rethrow error so that python caller sees the exception (after joining all threads).
    if (e)
        std::rethrow_exception(e);
}


// -------------------------------------------------------------------------------------------------
//
// TcpReceiver


struct TcpReceiver : Hwtest::Worker
{
    // Initialized in constructor.
    string ip_addr;
    long num_tcp_connections = 0;
    long recv_bufsize = 0;
    int inic = -1;

    long nbytes_per_iteration = 300 * 1024 * 1024;

    // Initialized in worker_initialize().
    shared_ptr<char> recv_buf;
    Socket listening_socket;
    Epoll epoll;

    // Initialized in worker_accept_connections().
    vector<Socket> data_sockets;


    TcpReceiver(Hwtest *hwtest_, const vector<int> &vcpu_list_, int cpu_, int inic_, const string &ip_addr_, long num_tcp_connections_, long recv_bufsize_) :
        Worker(hwtest_, vcpu_list_, cpu_),
        ip_addr(ip_addr_),
        num_tcp_connections(num_tcp_connections_),
        recv_bufsize(recv_bufsize_),
        inic(inic_),
        epoll(false)  // 'epoll' is constructed in an uninitialized state, and gets initialized in worker_initialize()
    {
        xassert(ip_addr.size() > 0);
        xassert(num_tcp_connections > 0);
        xassert(recv_bufsize > 0);
        xassert((inic >= 0) && (inic < Hwtest::Stats::max_nic));

        stringstream ss;
        ss << "TcpReceiver(" << ip_addr << ", " << num_tcp_connections << " connections)";
        this->worker_name = ss.str();
    }


    virtual void worker_initialize() override
    {
        this->recv_buf = worker_alloc(recv_bufsize);

        this->listening_socket = Socket(PF_INET, SOCK_STREAM);
        this->listening_socket.set_reuseaddr();
        this->listening_socket.bind(ip_addr, 8787);  // TCP port 8787
        this->epoll.initialize();
    }


    virtual void worker_accept_connections()
    {
        stringstream ss;
        ss << worker_name << ": listening for TCP connections. Reminder: use 'pirate_frb hwtest -s <config.yml>' to send data\n";
        cout << ss.str() << flush;

        this->data_sockets.resize(num_tcp_connections);
        this->listening_socket.listen();

        // Later: setsockopt(SO_RCVBUF), setsockopt(SO_SNDBUF) here?
        // I tried SO_RECVLOWAT here, but uProf reported significantly higher memory bandwidth!

        for (unsigned int ids = 0; ids < data_sockets.size(); ids++) {
            this->data_sockets[ids] = listening_socket.accept();
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
        long nbytes_cumul = 0;

        while (nbytes_cumul < nbytes_per_iteration) {
            long nbytes_save = nbytes_cumul;
            int num_events = epoll.wait(100);  // 100ms timeout, to recheck is_stopped periodically

            if (num_events == 0) {
                std::unique_lock lk(hwtest->mutex);
                if (hwtest->is_stopped)
                    return Stats();
                continue;
            }

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


struct ChimeWorker : public Hwtest::Worker
{
    DedispersionConfig dedispersion_config;
    shared_ptr<DedispersionPlan> dedispersion_plan;
    shared_ptr<CudaStreamPool> cuda_stream_pool;
    shared_ptr<GpuDedisperser> gpu_dedisperser;
    
    int device = -1;
    long seq_id = 0;

    
    ChimeWorker(Hwtest *hwtest_, const vector<int> &vcpu_list_, int cpu_, int device_) :
        Worker(hwtest_, vcpu_list_, cpu_),
        device(device_)
    {
        xassert(device >= 0);
        xassert(device < Hwtest::Stats::max_gpu);
        xassert(device < get_cuda_device_count());

        stringstream ss;
        ss << "ChimeDedisperser(" << devstr(device) << ")";
        this->worker_name = ss.str();
    }

    virtual void worker_initialize() override
    {
        CUDA_CALL(cudaSetDevice(this->device));

        // Hardcoded CHIME dedispersion config (equivalent to configs/dedispersion/chime.yml,
        // except beams_per_gpu=16 and max_gpu_clag=1000. GPU memory usage is ~30 GB.
        
        dedispersion_config.zone_nfreq = { 16384 };
        dedispersion_config.zone_freq_edges = { 400, 800 };
        dedispersion_config.time_sample_ms = 1.0;
        dedispersion_config.tree_rank = 15;
        dedispersion_config.num_downsampling_levels = 4;
        dedispersion_config.time_samples_per_chunk = 2048;
        dedispersion_config.dtype = Dtype::from_str("float16");
        dedispersion_config.beams_per_gpu = 16;
        dedispersion_config.beams_per_batch = 2;
        dedispersion_config.num_active_batches = 2;
        dedispersion_config.max_gpu_clag = 1000;

        // No early triggers or frequency subbands.
        dedispersion_config.early_triggers = { };
        dedispersion_config.frequency_subband_counts = { 0, 0, 0, 0, 1 };

        // FIXME peak_finding_params are not quite right (same params at each level).
        // (max_width, dm_downsampling, time_downsampling, wt_dm_downsampling, wt_time_downsampling)
        dedispersion_config.peak_finding_params = {
            { 16, 0, 0, 64, 64 },
            { 16, 0, 0, 64, 64 },
            { 16, 0, 0, 64, 64 },
            { 16, 0, 0, 64, 64 }
        };
        
        dedispersion_config.validate();

        // Create DedispersionPlan and GpuDedisperser.
        dedispersion_plan = make_shared<DedispersionPlan> (dedispersion_config);
        cuda_stream_pool = CudaStreamPool::create(dedispersion_config.num_active_batches);

        GpuDedisperser::Params gdd_params;
        gdd_params.plan = dedispersion_plan;
        gdd_params.stream_pool = cuda_stream_pool;
        gpu_dedisperser = GpuDedisperser::create(gdd_params);

        // Allocate GpuDedisperser using dummy-mode BumpAllocators.
        int host_aflags = af_rhost | af_zero;
        if (hwtest->use_hugepages)
            host_aflags |= af_mmap_huge;
        
        BumpAllocator host_allocator(host_aflags, -1);
        BumpAllocator gpu_allocator(af_gpu | af_zero, -1);
        gpu_dedisperser->allocate(gpu_allocator, host_allocator);
    }

    virtual Stats worker_body() override
    {
        const long nstreams = gpu_dedisperser->nstreams;
        const long nbatches = gpu_dedisperser->nbatches;
        const long beams_per_batch = gpu_dedisperser->beams_per_batch;
        const long nouter = max(32/beams_per_batch, 1L);

        for (long iouter = 0; iouter < nouter; iouter++) {
            long istream = seq_id % nstreams;
            long ichunk = seq_id / nbatches;
            long ibatch = seq_id % nbatches;
            this->seq_id++;
            
            cudaStream_t compute_stream = cuda_stream_pool->compute_streams.at(istream);

            gpu_dedisperser->acquire_input(ichunk, ibatch, compute_stream);
            gpu_dedisperser->release_input(ichunk, ibatch, compute_stream);
            
            gpu_dedisperser->acquire_output(ichunk, ibatch, compute_stream);
            gpu_dedisperser->release_output(ichunk, ibatch, compute_stream);
        }

        // Reminder: ResourceTracker counts are "per batch".
        const ResourceTracker &rt = gpu_dedisperser->resource_tracker;
        double tchunk = 1.0e-3 * dedispersion_config.time_sample_ms * dedispersion_config.time_samples_per_chunk;

        Stats stats;
        stats.hmem(cpu) = nouter * rt.get_hmem_bw();
        stats.gmem[device] = nouter * rt.get_gmem_bw();
        stats.h2g[device] = nouter * rt.get_h2g_bw();
        stats.g2h[device] = nouter * rt.get_g2h_bw();
        stats.chime_beams += nouter * beams_per_batch * tchunk;  // beam-seconds
        return stats;
    }
};


// -------------------------------------------------------------------------------------------------
//
// MemcpyWorker (either host->host, host->device, or device->host)
//
//  - Currently use host->host MemcpyWorker as crude placeholder for packet assembler.
//  - Currently use (host->device + device->host) pair as crude placeholder for dedisperion.


struct MemcpyWorker : public Hwtest::Worker
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

    
    MemcpyWorker(Hwtest *hwtest_, const vector<int> &vcpu_list_, int cpu_, int src_device_, int dst_device_, long blocksize_, bool use_copy_engine_) :
        Worker(hwtest_, vcpu_list_, cpu_),
        src_device(src_device_),
        dst_device(dst_device_),
        blocksize(blocksize_),
        use_copy_engine(use_copy_engine_)
    {
        int num_gpus_in_machine = get_cuda_device_count();
        xassert(num_gpus_in_machine < Hwtest::Stats::max_gpu);

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


struct SsdWorker : public Hwtest::Worker
{
    // string worker_name;    // inherited from 'Worker' base class

    fs::path root_dir;
    long nbytes_per_file = 0;
    int issd = -1;
    
    long nfiles_per_iteration = 0;
    long curr_file = 0;
    
    shared_ptr<char> data;

    SsdWorker(Hwtest *hwtest_, const vector<int> &vcpu_list_, int cpu_, int issd_, const string &root_dir_, long nbytes_per_file_) :
        Worker(hwtest_, vcpu_list_, cpu_),
        root_dir(root_dir_),
        nbytes_per_file(nbytes_per_file_),
        issd(issd_)
    {
        xassert(!root_dir.empty());
        xassert(nbytes_per_file > 0);
        xassert((issd >= 0) && (issd < Hwtest::Stats::max_ssd));

        this->nfiles_per_iteration = (nbytes_per_file + 256L*1024L*1024L - 1) / nbytes_per_file;

        stringstream ss;
        ss << "SsdWriter(dir=" << root_dir.string()
           << ", nbytes_per_file=" << ksgpu::nbytes_to_str(nbytes_per_file)
           << ", nfiles_per_iteration=" << nfiles_per_iteration
           << ")";

        this->worker_name = ss.str();
    }

    virtual void worker_initialize() override
    {
        // FIXME should be random data, to avoid confusion from compressed filesystems
        data = worker_alloc(nbytes_per_file);

        vector<fs::path> files_to_delete;

        for (const auto &entry : fs::directory_iterator(root_dir))
            if (is_stale_file(entry.path().filename().string()))
                files_to_delete.push_back(entry.path());

        if (files_to_delete.empty())
            return;

        stringstream ss;
        ss << "SsdWriter(" << root_dir.string() << "): deleting " << files_to_delete.size() << " stale files from previous run\n";
        cout << ss.str() << flush;

        for (const auto &path : files_to_delete)
            remove_file(path);
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
            string filename = (root_dir / ("file_" + to_string(curr_file++))).string();
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


struct DownsamplingWorker : public Hwtest::Worker
{
    int src_bit_depth = 0;
    long src_nelts = 0;
    long src_nbytes = 0;
    long dst_nbytes = 0;

    shared_ptr<char> psrc;
    shared_ptr<char> pdst;
    
    DownsamplingWorker(Hwtest *hwtest_, const vector<int> &vcpu_list_, int cpu_, int src_bit_depth_, long src_nelts_) :
        Worker(hwtest_, vcpu_list_, cpu_),
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


void Hwtest::add_tcp_receiver(const string &ip_addr, long num_tcp_connections, long recv_bufsize, const vector<int> &vcpu_list, int cpu, int inic)
{
    auto wp = make_shared<TcpReceiver> (this, vcpu_list, cpu, inic, ip_addr, num_tcp_connections, recv_bufsize);
    this->_add_worker(wp, "add_tcp_receiver");
}

void Hwtest::add_chime_dedisperser(int device, const vector<int> &vcpu_list, int cpu)
{
    auto wp = make_shared<ChimeWorker> (this, vcpu_list, cpu, device);
    this->_add_worker(wp, "add_chime_dedisperser");
}

void Hwtest::add_memcpy_thread(int src_device, int dst_device, long blocksize, bool use_copy_engine, const vector<int> &vcpu_list, int cpu)
{
    auto wp = make_shared<MemcpyWorker> (this, vcpu_list, cpu, src_device, dst_device, blocksize, use_copy_engine);
    this->_add_worker(wp, "add_memcpy_thread");
}

void Hwtest::add_ssd_writer(const string &root_dir, long nbytes_per_file, const vector<int> &vcpu_list, int cpu, int issd)
{
    auto wp = make_shared<SsdWorker> (this, vcpu_list, cpu, issd, root_dir, nbytes_per_file);
    this->_add_worker(wp, "add_ssd_writer");
}

void Hwtest::add_downsampling_thread(int src_bit_depth, long src_nelts, const vector<int> &vcpu_list, int cpu)
{
    auto wp = make_shared<DownsamplingWorker> (this, vcpu_list, cpu, src_bit_depth, src_nelts);
    this->_add_worker(wp, "add_downsampling_thread");
}


}  // namespace pirate
