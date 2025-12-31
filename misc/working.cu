

struct GpuDedisperser
{
    struct Params 
    {
        std::shared_ptr<DedispersionPlan> plan;
        std::shared_ptr<CudaStreamPool> stream_pool;

        // detect_deadlocks=true: assumes that {acquire,release}_input() is called
        // on the same thread as {acquire,release}_output(), and detect deadlocks
        // accordingly.
        //
        // detect_deadlocks=false: assumes that {acquire,release}_input() is called
        // on a different thread as {acquire,release}_output(). In this case, the
        // deadlock-checking logic is disabled.

        bool detect_deadlocks = true;

        // fixed_weights=false: acquire_input() and release_input() also acquire
        // and release the pf_weights. (This makes sense if the caller dynamically
        // changes the weights, e.g. in a unit test.)
        //
        // fixed_weights=true: weights are assumed constant throughout the lifetime
        // of the GpuDedisperser. (Currently used in timing, since it should be more
        // representative of the final system.)
        //
        // Both of these ways of handling the weights are temporary hacks, that I'll
        // revisit later!

        bool fixed_weights = false;
    };

    // acquire_input): after call, 'stream' sees empty input buffer.
    // release_input(): before call, 'stream' must see full input buffer.
    // acquire_output(): after call, 'stream' sees full output buffer.
    // release_output(): before call, 'stream' must see empty output buffer.
    // FIXME: I may rethink this API for input/output buffers later.

    void acquire_input(long ichunk, long ibatch, cudaStream_t stream);
    void release_input(long ichunk, long ibatch, cudaEvent_t stream);
    void acquire_output(long ichunk, long ibatch, cudaStream_t stream);
    void release_output(long ichunk, long ibatch, cudaStream_t stream);


    // --------------------------------------------------------------------------------

    void _launch_tree_gridding(long ichunk, long ibatch);
    void _launch_lagged_downsampler(long ichunk, long ibatch);
    void _launch_dd_stage1(long ichunk, long ibatch);
    void _launch_et_g2g(long ichunk, long ibatch);   // copies from 'gpu' zones to 'g2h' zones
    void _launch_g2h(long ichunk, long ibatch);      // this is the main gpu->host copy
    void _launch_h2g(long ichunk, long ibatch);      // this is the main host->gpu copy
    void _launch_cdd2(long ichunk, long ibatch);

    void _run_et_h2h(long ichunk, long ibatch);      // on CPU! copies from 'host' zones to 'et_host' zone
    void _launch_et_h2g(long ichuink, long ibatch);  // copies from 'et_host' zone to 'et_gpu' zone

    void _launch_dedispersion_kernels(long ichunk, long ibatch, cudaStream_t stream);
    void _worker_thread_main();

    // The CudaEventRingbufs keep track of lagged dependencies between kernels.
    //
    // Synchronization between main thread and et_h2h worker thread works as follows:
    //   - evrb_g2h: produced in main thread, consumed in worker
    //   - evrb_cdd2: produced in main thread, consumed in worker
    //   - evrb_et_h2g: produced in worker, consumed in main thread
    //
    // Synchronization of input/output buffers works as follows:
    //   - (evrb_tree_gridding or evrb_cdd2): consumed in acquire_input()
    //   - (evrb_output): produced in release_output()

    std::shared_ptr<CudaEventRingbuf> evrb_tree_gridding;
    std::shared_ptr<CudaEventRingbuf> evrb_g2g;
    std::shared_ptr<CudaEventRingbuf> evrb_g2h;
    std::shared_ptr<CudaEventRingbuf> evrb_h2g;
    std::shared_ptr<CudaEventRingbuf> evrb_cdd2;
    std::shared_ptr<CudaEventRingbuf> evrb_et_h2g;
    std::shared_ptr<CudaEventRingbuf> evrb_output;

    std::mutex mutex;
    std::condition_variable cv;

    long curr_input_ichunk = 0;
    long curr_input_ibatch = 0;
    bool curr_input_acquired = false;

    long curr_output_ichunk = 0;
    long curr_output_ibatch = 0;
    bool curr_output_acquired = false;

    // For now, we always use one et_h2h worker thread.
    // FIXME(?) dynamically assign the number of et_h2h worker threads later?
    std::thread worker;
};


GpuDedisperser::GpuDedisperser() :
    params(params_)
{
    xassert(plan);
    xassert(stream_pool);
    xassert(plan->nstreams == stream_pool->nstreams);

    this->plan = params.plan;

    // Initialize CudaEventRingbufs.
    //
    //   tg: consumers=[acq_input]
    //   g2g: consumers=[g2h]
    //   g2h: consumers=[dd1,h2g,et_h2h]
    //   h2g: consumers=[g2h,cdd2]
    //   cdd2: consumers=[tg,h2g,et_h2g,acq_input,acq_output]
    //   et_h2g: consumers=[g2h,et_h2h,cdd2]
    //   output: consumers=[cdd2]

    long tg_nconsumers = params.fixed_weights ? 1 : 0;
    long cdd2_nconsumers = params.fixed_weights ? 4 : 5;

    // FIXME: using huge CudaEventRingbuf capacity for now!
    // Not sure if this will work, but let's try.
    long capacity = (plan->mega_ringbuf->max_clag * nbatches) + nstreams;

    this->evrb_tree_gridding = make_shared<CudaEventRingbuf> ("tree_gridding", tg_nconsumers, capacity);
    this->evrb_g2g = make_shared<CudaEventRingbuf> ("g2g", 1, capacity);
    this->evrb_g2h = make_shared<CudaEventRingbuf> ("g2h", 3, capacity);
    this->evrb_h2g = make_shared<CudaEventRingbuf> ("h2g", 2, capacity);
    this->evrb_cdd2 = make_shared<CudaEventRingbuf> ("cdd2", cdd2_nconsumers, capacity);
    this->evrb_et_h2g = make_shared<CudaEventRingbuf> ("et_h2g", 3, capacity);
    this->evrb_output = make_shared<CudaEventRingbuf> ("output", 1, capacity);

    // Note special "prefetch" logic for h2g copies!!
    // I can't decide if this way of doing it is elegant or a hack :)
    //
    // When gpu kernels are queued (in GpuDedisperser::release_input()),
    // instead of queueing h2g(seq_id), we queue h2g(seq_id + nstreams).
    // This should improve throughput, by "prefetching" the h2g data for
    // the next batch of gpu kernels.

    // For h2g prefectching to work, the following assert must be satisfied.
    long min_host_clag = plan->mega_ringbuf->min_host_clag;
    xassert_ge(min_host_clag * nbatches, nstreams);

    // To implement h2g prefectching, we pretend that the first (nstreams) batches
    // of h2g data have already been copied, by calling evrb_h2g.record(). Note
    // that this data is all-zeroes anyway, by the previous assert.
    for (long seq_id = 0; seq_id < nstreams; seq_id++) {
        cudaStream_t s = plan->stream_pool->low_priority_h2g;
        evrb_h2g->record(s, seq_id);
    }

    // Note that we don't implement any sort of prefetching for the et_h2g
    // copy, since it's launched asychronously in a worker thread anyway.
}


GpuDedisperser::allocate()
{
    create thread;
};


// ------------------------------------------------------------------------------


void GpuDedisperser::acquire_input(long ichunk, long ibatch, long stream)
{
    xassert(ichunk >= 0);
    xassert((ibatch >= 0) && (ibatch < nbatches));
    xassert(is_allocated);

    std::unique_lock<std::mutex> lock(mutex);

    if ((ichunk != curr_input_ichunk) || (ibatch != curr_input_ibatch)) {
        stringstream ss;
        ss << "GpuDedisperser::acquire_input(): expected (ichunk,ibatch)=(" 
           << curr_input_ichunk << "," << curr_input_ibatch << "), got ("
           << ichunk << "," << ibatch << ")";
        throw runtime_error(ss.str());
    }

    if (curr_input_acquired) {
        stringstream ss;
        ss << "GpuDedisperser::acquire_input(): double call to acquire_input()"
           << " with (ichunk,ibatch)=(" << ichunk << "," << ibatch << ")";
        throw runtime_error(ss.str());
    }

    if (params.detect_deadlocks) {
        long input_seq_id = ichunk * nbatches + ibatch;
        long output_seq_id = curr_output_ichunk * nbatches + curr_output_ibatch;

        if (input_seq_id >= output_seq_id + nstreams) {
            throw runtime_error("GpuDedisperser: deadlock detected (calls to acquire_input()"
                " are too far ahead of calls to release_output(). If the input/output arrays"
                " are handled in different threads, then this error is a false alarm, and you"
                " can suppress it by setting GpuDedisperser::Params::detect_deadlocks = false."
        }
    }

    curr_input_acquired = true;
    lock.unlock();

    // Argument-checking ends here.
    // If fixed_weights=true, then 'stream' should wait for the tree gridding
    // kernel (consumer of input array). If fixed_weights=false, then 'stream'
    // should wait for cdd2 (consumer of pf_weights array).

    shared_ptr<CudaEventRingbuf> &evrb = params.fixed_weights ? evrb_tree_gridding : evrb_cdd2;

    // This call to wait() can be nonblocking, since we know that the tree_gridding/cdd2
    // kernel was successfully launched by a previous call to release_input().
    evrb->wait(stream, seq_id - nstreams);
}


void GpuDeDedispeser::release_input(long ichunk, long ibatch, cudaStream_t stream)
{
    xassert(ichunk >= 0);
    xassert((ibatch >= 0) && (ibatch < nbatches));

    std::unique_lock<std::mutex> lock(mutex);

    if ((ichunk != curr_input_ichunk) || (ibatch != curr_input_ibatch)) {
        stringstream ss;
        ss << "GpuDedisperser::release_input(): expected (ichunk,ibatch)=(" 
           << curr_input_ichunk << "," << curr_input_ibatch << "), got ("
           << ichunk << "," << ibatch << ")";
        throw runtime_error(ss.str());
    }

    if (!curr_input_acquired) {
        stringstream ss;
        ss << "GpuDedisperser::acquire_input(): release_input() called without preceding"
           << " acquire_input(), (ichunk,ibatch)=(" << ichunk << "," << ibatch << ")";
        throw runtime_error(ss.str());
    }

    curr_input_ibatch++;
    curr_input_acquired = false;

    if (curr_input_ibatch == nbatches) {
        curr_input_ichunk++;
        curr_input_ibatch = 0;
    }

    lock.unlock();

    // Argument-checking ends here. The rest of release_input() is in its own
    // method _launch_dedispersion_kernels().
    _launch_dedispersion_kernels(ichunk, ibatch, stream);
}


void GpuDedisperser::acquire_output(long ichunk, long ibatch, cudaStream_t stream)
{
    xassert(ichunk >= 0);
    xassert((ibatch >= 0) && (ibatch < nbatches));
    xassert(is_allocated);

    std::unique_lock<std::mutex> lock(mutex);

    if ((ichunk != curr_output_ichunk) || (ibatch != curr_output_ibatch)) {
        stringstream ss;
        ss << "GpuDedisperser::acquire_output(): expected (ichunk,ibatch)=(" 
           << curr_output_ichunk << "," << curr_output_ibatch << "), got ("
           << ichunk << "," << ibatch << ")";
        throw runtime_error(ss.str());
    }

    if (curr_output_acquired) {
        stringstream ss;
        ss << "GpuDedisperser::acquire_output(): double call to acquire_output()"
           << " with (ichunk,ibatch)=(" << ichunk << "," << ibatch << ")";
        throw runtime_error(ss.str());
    }

    if (params.detect_deadlocks) {
        long input_seq_id = curr_input_ichunk * nbatches + curr_input_ibatch;
        long output_seq_id = ichunk * nbatches + ibatch;

        if (output_seq_id >= input_seq_id) {
            throw runtime_error("GpuDedisperser: deadlock detected (calls to acquire_output()"
                " are ahead of calls to release_input(). If the input/output arrays"
                " are handled in different threads, then this error is a false alarm, and you"
                " can suppress it by setting GpuDedisperser::Params::detect_deadlocks = false."
        }
    }

    curr_output_acquired = true;
    lock.unlock();

    // Argument checking ends here. The caller-specified stream waits for 'cdd2' to produce outputs.
    long seq_id = ichunk * nbatches + ibatch;
    bool blocking = !params.detect_deadlocks;
    evrb_cdd2->wait(stream, seq_id, blocking);
}


void GpuDedisperser::release_output(long ichunk, long ibatch, cudaStream_t stream)
{
    xassert(ichunk >= 0);
    xassert((ibatch >= 0) && (ibatch < nbatches));

    long seq_id = ichunk * nbatches + ibatch;
    cudaStream_t g2h_stream = params.stream_pool->low_priority_g2h_stream;
    cudaStream_t h2g_stream = params.stream_pool->low_priority_h2g_stream;
    cudaStream_t compute_stream = params.stream_pool->compute_streams.at(seq_id % nstreams);

    std::unique_lock<std::mutex> lock(mutex);

    if ((ichunk != curr_output_ichunk) || (ibatch != curr_output_ibatch)) {
        stringstream ss;
        ss << "GpuDedisperser::release_output(): expected (ichunk,ibatch)=(" 
           << curr_output_ichunk << "," << curr_output_ibatch << "), got ("
           << ichunk << "," << ibatch << ")";
        throw runtime_error(ss.str());
    }

    if (!curr_output_acquired) {
        stringstream ss;
        ss << "GpuDedisperser::acquire_output(): release_output() called without preceding"
           << " acquire_output(), (ichunk,ibatch)=(" << ichunk << "," << ibatch << ")";
        throw runtime_error(ss.str());
    }

    curr_output_ibatch++;
    curr_output_acquired = false;

    if (curr_output_ibatch == nbatches) {
        curr_output_ichunk++;
        curr_output_ibatch = 0;
    }

    lock.unlock();
    
    // Argument-checking ends here. We record an event from the caller-specified stream,
    // and put it in 'evrb_output' (a CudaEventRingbuf). The ccd2 kernel will wait for
    // this event later.
    evrb_output->record(stream, seq_id);
}


// ------------------------------------------------------------------------------------
//
// Difficult code starts here!
//
// Dependency graph:
//
//  - 6 MegaRingbuf zones: gpu, host, g2h, h2g, et_host, et_gpu
//
//  - 5+2 MegaRingbuf-adjacent kernels: dd1, g2g, g2h, h2g, cdd2, et_h2h, et_h2g
//
//  - Per-kernel inbufs and outbufs:
//
//      dd1 kernel:     inbufs []                 outbufs [gpu,g2h]
//      g2g kernel:     inbufs [gpu]              outbufs [g2h]
//      g2h kernel:     inbufs [g2h]              outbufs [host]
//      h2g kernel:     inbufs [host]             outbufs [h2g]
//      cdd2 kernel:    inbufs [gpu,h2g,et_gpu]   outbufs []
//      et_h2h kernel:  inbufs [host]             outbufs [et_host]
//      et_h2g kernel:  inbufs [et_host]          outbufs [et_gpu]
//
//  - Per-ringbuf producers and consumers, produced by a mechanical process:
//
//      gpu ringbuf:      producers [dd1]      consumers [g2g,cdd2]
//      host ringbuf:     producers [g2h]      consumers [h2g,et_h2h]
//      g2h ringbuf:      producers [dd1,g2g]  consumers [g2h]
//      h2g ringbuf:      producers [h2g]      consumers [cdd2]
//      et_host ringbuf:  producers [et_h2h]   consumers [et_h2g]
//      et_gpu ringbuf:   producers [et_h2g]   consumers [cdd2]
//
//  - Dependency analysis, produced by very mechanical cut-and-paste:
//
//      dd1 kernel: inbufs=[], outbufs=[gpu,g2h]
//        gpu outbuf: consumers=[g2g,cdd2]
//        g2h outbuf: consumers=[g2h]
//
//      g2g kernel: inbufs=[gpu], outbufs=[g2h]
//        gpu ringbuf: producers=[dd1]
//        g2h ringbuf: consumers=[g2h]
//
//      g2h kernel: inbufs=[g2h], outbufs=[host]
//        g2h ringbuf:  producers=[dd1,g2g]  
//        host ringbuf: consumers=[h2g,et_h2h]
//
//      h2g kernel: inbufs=[host], outbufs=[h2g]
//        host ringbuf: producers=[g2h]
//        h2g ringbuf:  consumers=[cdd2]
//
//      cdd2 kernel: inbufs=[gpu,h2g,et_gpu], outbufs=[]
//        gpu ringbuf:    producers=[dd1]
//        h2g ringbuf:    producers=[h2g]
//        et_gpu ringbuf: producers=[et_h2g]
//
//      et_h2h kernel: inbufs=[host], outbufs=[et_host]
//        host ringbuf:     producers=[g2h]
//        et_host ringbuf:  consumers=[et_h2g]
//
//      et_h2g kernel: inbufs=[et_host], outbufs=[et_gpu]
//        et_host ringbuf:  producers=[et_h2h]
//        et_gpu ringbuf:   consumers=[cdd2]
//
// FIXME now that the dust has settled, this synchronization logic is mechanical
// enough that I could capture it in a KernelGraph helper class, which keeps track
// of lagged dependencies between kernels. A KernelGraph::Node could represent
// a kernel, and contain shared_ptr<CudaEventRingbuf. A KernelGraph::Edge could
// represent a kernel dependency with lags. (There may also be a KernelGraph::Buffer
// used temporarily when building the graph.) Note that initializing CudaEventRingbuf
// capacities may be a sticking point. Don't forget KernelGraph::to_yaml()!
//
// Note that the synchronization between main thread and et_h2h thread
// is entirely via CudaEventRingbufs:
//
//   - evrb_g2h: produced in main thread, consumed in worker
//   - evrb_cdd2: produced in main thread, consumed in worker
//   - evrb_et_h2g: produced in worker, consumed in main thread


void GpuDedispersionKernel::_launch_dedispersion_kernels(long ichunk, long ibatch, cudaStream_t stream)
{
    // Argument-checking has already been done in release_input().
    long seq_id = ichunk * nbatches + ibatch;
    cudaStream_t g2h_stream = params.stream_pool->low_priority_g2h_stream;
    cudaStream_t h2g_stream = params.stream_pool->low_priority_h2g_stream;
    cudaStream_t compute_stream = params.stream_pool->compute_streams.at(seq_id % nstreams);

    // Compute kernel waits on the caller-specified stream.
    cudaEvent_t *event = nullptr;
    CUDA_CALL(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    CUDA_CALL(cudaEventRecord(&event, stream));
    CUDA_CALL(cudaStreamWaitEvent(compute_stream, event, 0));
    CUDA_CALL(cudaEventDestroy(event));

    // This call to wait() ensures that all pstate is up-to-date.
    // (Note that the the previous chunk may have run on a different compute stream.)
    // In principle, one can do better by inserting per-kernel wait() calls below, for
    // every kernel which has pstate. I might revisit this later.
    evrb_cdd2->wait(compute_stream, seq_id - nbatches);

    // Now we can start launching kernels!

    _launch_tree_gridding(ichunk, ibatch, compute_stream);
    evrb_tree_gridding->record(compute_stream, seq_id);

    _launch_lagged_downsampler(ichunk, ibatch, compute_stream);

    // dd1 kernel: inbufs=[], outbufs=[gpu,g2h]
    //   gpu outbuf: consumers=[g2g,cdd2]
    //   g2h outbuf: consumers=[g2h]
    //
    // The [g2g,cdd2] consumers are automatic, since they run earlier on the same stream. 
    // but we need to wait for g2h, before launching dd1.

    evrb_g2h->wait(compute_stream, seq_id - nstreams);    // consumer (g2h)
    _launch_dd_stage1(ichunk, ibatch, compute_stream);

    // g2g kernel: inbufs=[gpu], outbufs=[g2h]
    //   gpu inbuf: producers=[dd1]
    //   g2h outbuf: consumers=[g2h]
    //
    // The dd1 producer is okay, since it's on the same stream.
    // The g2h consumer is okay, since dd1 waited on it (a few lines of code ago).

    _launch_g2g(ichunk, ibatch);
    evrb_g2g->record(compute_stream, seq_id);

    // g2h kernel: inbufs=[g2h], outbufs=[host]
    //   g2h inbuf: producers=[dd1,g2g]  
    //   host outbuf: consumers=[h2g,et_h2h]
    //
    // The dd1 producer is okay, since it's (earlier) on the same stream as g2g.
    // Need to wait/synchronize with et_h2h consumer, h2g consumer, g2g producer.
    //
    // To synchronize with the et_h2h consumer, we use a hack: wait on et_h2g producer
    // instead. This is slightly suboptimal, but convenient since we can use CudaEventRingbuf
    // as the only synchronization mechanism. (In practice, I doubt that the suboptimality
    // matters at all.)

    long et_h2h_headroom = plan->mega_ringbuf->et_h2h_headroom;
    long et_seq_id = seq_id - nstreams - (et_h2h_headroom * nbatches);
    evrb_et_h2g->synchronize_with_producer(et_seq_id);   // consumer (et_h2g)
    evrb_h2g->wait(g2h_stream, seq_id - nstreams);       // consumer (h2g)
    evrb_g2g->wait(g2h_stream, seq_id);                  // producer (g2g)
    _launch_g2h(ichunk, ibatch, g2h_stream);
    evrb_g2h->record(g2h_stream, seq_id);

    // Note: h2g kernel postponed until the end -- see below!

    // cdd2 kernel: inbufs=[gpu,h2g,et_gpu], outbufs=[]
    //   gpu inbuf:    producers=[dd1]
    //   h2g inbuf:    producers=[h2g]
    //   et_gpu inbuf: producers=[et_h2g]
    //
    // The dd1 producer is okay, since it's earlier on the same stream.
    // We need to wait on the producers h2g and et_h2g. 
    //
    // Note also that we need to wait on the 'evrb_output' events.
    // These are produced in release_output().
    
    bool out_blocking = !params.detect_deadlocks;
    evrb_h2g->wait(compute_stream, seq_id);                               // producer (h2g)
    evrb_et_h2g->wait(compute_stream, seq_id, true);                      // producer (et_h2g), blocking=true
    evrb_output->wait(compute_stream, seq_id - nstreams, out_blocking);   // consumer (output)
    _launch_cdd2(ichunk, ibatch, compute_stream);
    evrb_cdd2->record(compute_stream, seq_id);

    // Now the h2g kernel. As explained in the constructor above,
    // instead of queueing h2g(seq_id), we queue h2g(seq_id + nstreams).
    // This should improve throughput, by "prefetching" the h2g data for
    // the next batch of gpu kernels.
    //
    // h2g kernel: inbufs=[host], outbufs=[h2g]
    //   host inbuf: producers=[g2h]    (need to wait on this)
    //   h2g outbuf: consumers=[cdd2]   (need to wait on this)

    long prefetch_ichunk = (seq_id + nstreams) / nbatches;
    long prefetch_ibatch = (seq_id + nstreams) % nbatches;
    long min_host_clag = plan->mega_ringbuf->min_host_clag;
    long producer_seq_id = seq_id + nstreams - (min_host_clag * nbatches);

    evrb_g2h->wait(h2g_stream, producer_seq_id);  // producer (g2h)
    evrb_cdd2->wait(h2g_stream, seq_id);          // consumer (cdd2)
    _launch_h2g(prefetch_ichunk, prefetch_ibatch, h2g_stream);
    evrb_h2g->record(h2g_stream, seq_id);
}


void GpuDedisperser::_worker_main()
{
    xassert(is_allocated);

    cudaStream_t h2g_stream = params.stream_pool->low_priority_h2g_stream;
    long min_et_clag = plan->mega_ringbuf->min_et_clag;
    long seq_id = 0;

    for (;;) {
        long ichunk = seq_id / nstreams;
        long ibatch = seq_id % nstreams;

        // et_h2h kernel: inbufs=[host], outbufs=[et_host]
        //   host inbuf: producers=[g2h]
        //   et_host outbuf: consumers=[et_h2g]

        evrb_g2h->synchronize(seq_id - min_et_clag, true);   // producer (blocking=true but farfetched)
        evrb_et_h2g->synchronize(seq_id - nstreams);         // consumer
        _do_et_h2h(ichunk, ibatch);

        // et_h2g kernel: inbufs=[et_host], outbufs=[et_gpu]
        //   et_host inbuf: producers=[et_h2h]
        //   et_gpu outbuf: consumers=[cdd2]
        //
        // The et_h2h producer is okay, since it runs synchronously in the same thread,
        // but we need to wait for the cdd2 consumer, before launching et_h2g.

        evrb_cdd2->wait(h2g_stream, seq_id - nstreams, true);  // consumer (blocking=true)
        _launch_et_h2g(ichunk, ibatch, h2g_stream);
        evrb_et_h2g->record(h2g_stream, seq_id);
        
        seq_id++;
    }
}


// ------------------------------------------------------------------------------------


void GpuDedisperser::time(long num_iterations, bool use_hugepages)
{
    xassert(!is_allocated);  // ensures that hugepage flag

    GpuDequantizationKernel dequantization_kernel(dtype, beams_per_batch, nfreq, nt_in);

    ResourceTracker rt = this->resource_tracker;  // copy
    rt += dequantization_kernel.resource_tracker;
    rt.add_memcpy_h2g("raw_data", raw_nbytes);

    Dtype dt_int4 = Dtype::from_str("int4");
    int cpu_aflags = af_rhost | af_zero | (use_hugepages ? af_mmap_huge : 0);
    int gpu_aflags = af_gpu | af_zero;

    // The "multi_" prefix means "one array per stream".
    Array<void> multi_raw_cpu(dt_int4, {nstreams, beams_per_batch, nfreq, nt_in}, cpu_flags);
    Array<void> multi_raw_gpu(dt_int4, {nstreams, beams_per_batch, nfreq, nt_in}, gpu_aflags);
    long raw_nbytes = xdiv(beams_per_batch * nfreq * nt_in, 2);

    BumpAllocator dummy_cpu_allocator(cpu_aflags, -1);
    BumpAllocator dummy_gpu_allocator(gpu_aflags, -1);
    this->allocate(dummy_gpu_allocator, dummy_cpu_allocator);

    // We use a ksgpu::KernelTimer in the timing loop, for the sake of tradition,
    // but it's a little awkward since the KernelTimer defines its own stream pool.
    // In the loop below, we keep the KernelTimer streams synchronized with the 
    // GpuDedisperser streams.

    KernelTimer kt(niterations, nstreams);

    while (kt.next()) {
        long ichunk = kt.curr_iteration / nbatches;
        long ibatch = kt.curr_iteration % nbatches;

        cudaStream_t h2g_stream = this->params.stream_pool.high_priority_h2g;
        cudaStream_t compute_stream = this->params.stream_pool.compute_streams.at(kt.istream);

        Array<void> raw_cpu = multi_raw_cpu.slice(0, kt.istream);
        Array<void> raw_gpu = multi_raw_gpu.slice(0, kt.istream);
        Array<void> dd_in = 

        CUDA_CALL(cudaMemcpyAsync(raw_gpu.data, raw_cpu.data, raw_nbytes, cudaMemcpyHostToDevice, h2g_stream));

        // Run dequantization kernel on compute stream.
        xxx;  // synchronize h2g_stream -> compute_stream
        this->acquire_input(ichunk, ibatch, compute_stream);
        dequantization_kernel.launch(this->input_arrays., in, compute_stream);

        // Launches all the dedispersion kernels.
        this->release_input(ichunk, ibatch, compute_stream);

        this->acquire_output(ichunk, ibatch, compute_stream);
        this->release_output(ichunk, ibatch, compute_stream);

        if (kt.warmed_up) {
            cout << xxx;
        }
    }
}