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

    // acquire_input_sync() - blocks calling thread until input buffer is empty.
    // acquire_output_async() - queues "input buffer is empty" event to specified stream.
    void acquire_input_sync(long ichunk, long ibatch);
    void acquire_input_async(long ichunk, long ibatch, cudaStream_t stream);

    // Caller passes "input buffer is full" event, when done with the input buffer.
    // The release_input() function queues all the compute kernels.
    // NOTE: release_input() can block!!
    void release_input(cudaEvent_t event);

    // acquire_input_sync() - blocks calling thread until output buffer is full.
    // acquire_output_async() - queues "output buffer is full" event to specified stream.
    void acquire_output_sync(long ichunk, long ibatch);
    void acquire_output_async(long ichunk, long ibatch, cudaStream_t stream);

    // Caller passes "output buffer is empty" event, when done with the output buffer.
    void release_output(cudaEvent_t event);

    // From main thread.
    void _launch_tree_gridding(long ichunk, long ibatch);
    void _launch_lagged_downsampler(long ichunk, long ibatch);
    void _launch_dd_stage1(long ichunk, long ibatch);
    void _launch_et_g2g(long ichunk, long ibatch);   // copies from 'gpu' zones to 'g2h' zones
    void _launch_g2h(long ichunk, long ibatch);      // this is the main gpu->host copy
    void _launch_h2g(long ichunk, long ibatch);      // this is the main host->gpu copy
    void _launch_cdd2(long ichunk, long ibatch);

    // From et_h2h thread.
    void _run_et_h2h(long ichunk, long ibatch);      // on CPU! copies from 'host' zones to 'et_host' zone
    void _launch_et_h2g(long ichuink, long ibatch);  // copies from 'et_host' zone to 'et_gpu' zone

    std::shared_ptr<CudaEventRingbuf> evrb_tree_gridding;
    std::shared_ptr<CudaEventRingbuf> evrb_g2g;
    std::shared_ptr<CudaEventRingbuf> evrb_g2h;
    std::shared_ptr<CudaEventRingbuf> evrb_h2g;     // produced by main thread, consumed by both threads
    std::shared_ptr<CudaEventRingbuf> evrb_cdd2;
    std::shared_ptr<CudaEventRingbuf> evrb_et_h2g;  // produced by worker thread, consumed by main thread
    std::shared_ptr<CudaEventRingbuf> evrb_output;  // produced in release_output()

    std::mutex mutex;
    bool is_allocated = false;

    long curr_input_ichunk = 0;
    long curr_input_ibatch = 0;
    bool curr_input_acquired = false;

    long curr_output_ichunk = 0;
    long curr_output_ibatch = 0;
    bool curr_output_acquired = false;
};


GpuDedispeser::GpuDedisperser() :
    params(params_)
{
    xassert(plan);
    xassert(stream_pool);
    xassert(plan->nstreams == stream_pool->nstreams);

    this->plan = params.plan;

    long tg_nconsumers = params.fixed_weights ? 1 : 0;
    long cdd2_nconsumers = params.fixed_weights ? 5 : 6;
    this->evrb_tree_gridding = make_shared<CudaEventRingbuf> ("tree_gridding", tg_nconsumers, xx);
    this->evrb_g2g = make_shared<CudaEventRingbuf> ("g2g", 1, xxx);
    this->evrb_g2h = make_shared<CudaEventRingbuf> ("g2h", 3, xxx);
    this->evrb_h2g = make_shared<CudaEventRingbuf> ("h2g", 2, xxx);
    this->evrb_cdd2 = make_shared<CudaEventRingbuf> ("cdd2", cdd2_nconsumers, xxx);
    this->evrb_et_h2g = make_shared<CudaEventRingbuf> ("et_h2g", 3, xxx);
    this->evrb_output = make_shared<CudaEventRingbuf> ("output", 1, xxx);
}


// Dependency graph (in words):
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
// Note that the synchronization between main thread and et_h2h thread
// is entirely via CudaEventRingbufs:
//
//   - evrb_g2h: produced in main thread, consumed in worker
//   - evrb_cdd2: produced in main thread, consumed in worker
//   - evrb_et_h2g: produced in worker, consumed in main thread
//
// Similarly, the synchronization logic in {acquire}

CudaEventRingbuf &GpuDedisperser::_acquire_input(long ichunk, long ibatch)
{
    xassert(ichunk >= 0);
    xassert((ibatch >= 0) && (ibatch < nbatches));

    std::lock_guard<std::mutex> lock(mutex);
    xassert(is_allocated);

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

    // If fixed_weights=true, then we just need to weight for the tree gridding
    // kernel (consumer of input array). If fixed_weights=false, then we also need
    // to wait for cdd2 (consumer of pf_weights array).

    return params.fixed_weights ? evrb_tree_gridding : evrb_cdd2;
}


void GpuDedisperser::acquire_input_sync(long ichunk, long ibatch)
{
    long seq_id = ichunk * nbatches + ibatch;
    CudaEventRingbuf &evrb = _acquire_input(ichunk, ibatch);

    // blocking=true here, in case we need to wait for et_host thread to queue events.
    evrb.synchronize(seq_id - nstreams, true);
}

void GpuDedisperser::acquire_input_async(long ichunk, long ibatch, cudaStream_t stream)
{
    long seq_id = ichunk * nbatches + ibatch;
    CudaEventRingbuf &evrb = _acquire_input(ichunk, ibatch);

    // blocking=true here, in case we need to wait for et_host thread to queue events.
    evrb.wait(stream, seq_id - nstreams, true);
}


void GpuDeDedispeser::release_input(long ichunk, long ibatch, cudaEvent_t event)
{
    xassert(ichunk >= 0);
    xassert((ibatch >= 0) && (ibatch < nbatches));

    long seq_id = ichunk * nbatches + ibatch;
    cudaStream_t g2h_stream = params.stream_pool->low_priority_g2h_stream;
    cudaStream_t h2g_stream = params.stream_pool->low_priority_h2g_stream;
    cudaStream_t compute_stream = params.stream_pool->compute_streams.at(seq_id % nstreams);

    std::unique_lock<std::mutex> lock(mutex);
    xassert(is_allocated);

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

    // Before launching the first kernel (tree_gridding), wait on the caller-specified
    // "input buffer is full" event.

    if (event != nullptr)
        CUDA_CALL(cudaStreamWaitEvent(compute_stream, event, 0));

    // This call to wait() ensures that all pstate is up-to-date.
    // (Note that the the previous chunk may have run on a different compute stream.)
    // In principle, one can do better by inserting per-kernel wait() calls below, for
    // every kernel which has pstate. I might revisit this later.
    evrb_cdd2.wait(compute_stream, seq_id - nbatches);

    // Now we can start launching kernels!

    _launch_tree_gridding(ichunk, ibatch, compute_stream);
    evrb_tree_gridding.record(compute_stream, seq_id);

    _launch_lagged_downsampler(ichunk, ibatch, compute_stream);

    // dd1 kernel: inbufs=[], outbufs=[gpu,g2h]
    //   gpu outbuf: consumers=[g2g,cdd2]
    //   g2h outbuf: consumers=[g2h]
    //
    // The [g2g,cdd2] consumers are automatic, since they run earlier on the same stream. 
    // but we need to wait for g2h, before launching dd1.

    evrb_g2h.wait(compute_stream, seq_id - nstreams);
    _launch_dd_stage1(ichunk, ibatch, compute_stream);

    // g2g kernel: inbufs=[gpu], outbufs=[g2h]
    //   gpu inbuf: producers=[dd1]
    //   g2h outbuf: consumers=[g2h]
    //
    // The dd1 producer is okay, since it's on the same stream.
    // The g2h consumer is okay, since dd1 waited on it (a few lines of code ago).

    _launch_g2g(ichunk, ibatch);
    evrb_g2g.record(compute_stream, seq_id);

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

    evrb_et_h2g.synchronize_with_producer(seq_id - xxxxx);  // can block!
    evrb_h2g.wait(g2h_stream, seq_id - nstreams);
    evrb_g2g.wait(g2h_stream, seq_id);
    _launch_g2h(ichunk, ibatch, g2h_stream);
    evrb_g2h.record(g2h_stream, seq_id);

    // h2g kernel: inbufs=[host], outbufs=[h2g]
    //   host inbuf: producers=[g2h]
    //   h2g outbuf: consumers=[cdd2]

    evrb_g2h.wait(h2g_stream, seq_id - xxxxxxx);
    evrb_cdd2.wait(h2g_stream, seq_id - nstreams);
    _launch_h2g(ichunk, ibatch, h2g_stream);
    evrb_h2g.record(h2g_stream, seq_id);

    // cdd2 kernel: inbufs=[gpu,h2g,et_gpu], outbufs=[]
    //   gpu inbuf:    producers=[dd1]
    //   h2g inbuf:    producers=[h2g]
    //   et_gpu inbuf: producers=[et_h2g]
    //
    // The dd1 producer is okay, since it's earlier on the same stream.
    // We need to wait on the producers h2g and et_h2g. 
    //
    // Note also that we need to wait on the 'evrb_output' events, that are
    // produced by release_output().
    
    evrb_h2g.wait(compute_stream, seq_id);
    evrb_et_h2g.wait(compute_stream, seq_id, true);  // can block!
    evrb_output.wait(compute_stream, seq_id - nstreams);
    _launch_cdd2(ichunk, ibatch, compute_stream);
    evrb_cdd2.record(compute_stream, seq_id);
}


void GpuDedisperser::_worker_main()
{
    cudaStream_t h2g_stream = params.stream_pool->low_priority_h2g_stream;
    long seq_id = 0;

    for (;;) {
        long ichunk = seq_id / nstreams;
        long ibatch = seq_id % nstreams;

        // et_h2h kernel: inbufs=[host], outbufs=[et_host]
        //   host inbuf: producers=[g2h]
        //   et_host outbuf: consumers=[et_h2g]

        evrb_g2h.synchronize(seq_id - xxxxxx, true);       // blocking=true
        evrb_et_h2g.synchronize(seq_id - nstreams, true);  // blocking=true
        _run_et_h2h(ichunk, ibatch);

        // et_h2g kernel: inbufs=[et_host], outbufs=[et_gpu]
        //   et_host inbuf: producers=[et_h2h]
        //   et_gpu outbuf: consumers=[cdd2]
        //
        // The et_h2h producer is okay, since it runs synchronously in the same thread,
        // but we need to wait for the cdd2 consumer, before launching et_h2g.

        evrb_cdd2.wait(h2g_stream, seq_id - nstreams, true);  // blocking=true
        _launch_et_h2g(ichunk, ibatch, h2g_stream);
        evrb_et_h2g.record(h2g_stream, seq_id);
        
        seq_id++;
    }
}


void GpuDedisperser::_acquire_output(long ichunk, long ibatch)
{
    xassert(ichunk >= 0);
    xassert((ibatch >= 0) && (ibatch < nbatches));

    std::lock_guard<std::mutex> lock(mutex);
    xassert(is_allocated);

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
}


void GpuDedisperser::acquire_output_sync(long ichunk, long ibatch)
{
     _acquire_output(ichunk, ibatch);

    long seq_id = ichunk * nbatches + ibatch;
    evrb_cdd2.synchronize(seq_id, xxx);  // blocking?
}


void GpuDedisperser::acquire_output_async(long ichunk, long ibatch, )
{
    _acquire_output(ichunk, ibatch);

    long seq_id = ichunk * nbatches + ibatch;
    evrb_cdd2.synchronize(seq_id, xxx);  // blocking?
}

void GpuDedisperser::release_output()
{
    xassert(ichunk >= 0);
    xassert((ibatch >= 0) && (ibatch < nbatches));

    long seq_id = ichunk * nbatches + ibatch;
    cudaStream_t g2h_stream = params.stream_pool->low_priority_g2h_stream;
    cudaStream_t h2g_stream = params.stream_pool->low_priority_h2g_stream;
    cudaStream_t compute_stream = params.stream_pool->compute_streams.at(seq_id % nstreams);

    std::unique_lock<std::mutex> lock(mutex);
    xassert(is_allocated);

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

    // XXX doesn't quite work at the end -- need a stream!
    evrb_output.record(seq_id);
}

