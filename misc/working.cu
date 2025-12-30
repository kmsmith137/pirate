struct GpuDedisperser
{
    void acquire_input_sync(long ichunk, long ibatch);
    void acquire_input_async(long ichunk, long ibatch, cudaStream_t stream);

    // This function queues all the compute kernels.
    // 'event' can be NULL, if appropriate compute stream has already released the input buffer.
    void release_input(cudaEvent_t event);  // NOTE can block!!


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

    std::shared_ptr<CudaStreamPool> stream_pool;

    CudaEventRingbuf evrb_tree_gridding;
    CudaEventRingbuf evrb_g2g;
    CudaEventRingbuf evrb_g2h;
    CudaEventRingbuf evrb_h2g;     // produced by main thread, consumed by both threads
    CudaEventRingbuf evrb_et_h2g;  // produced by worker thread, consumed by main thread
    CudaEventRingbuf evrb_cdd2;

    long input_curr_ichunk = 0;
    long input_curr_ibatch = 0;
    bool input_acquired = false;
};



void GpuDedisperser::acquire_input_sync(long ichunk, long ibatch)
{
    xxx;  // kill worker thread on exception
    xxx;  // check for ichunk/ibatch mismatch
    xxx;  // assert allocated
    evrb_tree_gridding.synchronize(iseq - nstreams);
}

void GpuDedisperser::acquire_input_async(long ichunk, long ibatch, cudaStream_t stream)
{
    xxx;  // kill worker thread on exception
    xxx;  // check for ichunk/ibatch mismatch
    xxx;  // assert allocated
    evrb_tree_gridding.wait(stream, iseq - nstreams);
}

void GpuDeDeispeser::release_input(long ichunk, long ibatch, cudaEvent_t event)
{
    xxx;  // kill worker thread on exception
    xxx;  // check for ichunk/ibatch mismatch

    cudaStream_t g2h_stream = stream_pool->low_priority_g2h_stream;
    cudaStream_t h2g_stream = stream_pool->low_priority_h2g_stream;
    cudaStream_t compute_stream = stream_pool->compute_streams.at(istream);

    if (event != nullptr)
        CUDA_CALL(cudaStreamWaitEvent(compute_stream, event, 0));

    // Here we make a big simplification! The first wait() ensures that all pstate
    // is up-to-date, and the second wait() ensures that all intermediate buffers
    // are free. In principle, one can do better by waiting incrementally, one kernel
    // at a time, but this requires an obnoxious number of wait() calls in the code below.
    // I might revisit this later.

    evrb_cdd2.wait(compute_stream, iseq - nbatches);
    evrb_cdd2.wait(compute_stream, iseq - nstreams);

    _launch_tree_gridding(ichunk, ibatch);
    evrb_tree_gridding.record(compute_stream, iseq);

    _launch_lagged_downsampler(ichunk, ibatch);

    // dd1 needs to worry about three consumers: g2g, cdd2, and g2h.
    // First two are handled by "big simplification" above, but we need to wait for g2h.

    evrb_g2h.wait(compute_stream, iseq - nstreams);
    _launch_dd_stage1(ichunk, ibatch);

    // g2g needs to worry about one consumer: g2h.
    // Can omit the wait() since we already waited on g2h a few lines of code ago.

    _launch_g2g(ichunk, ibatch);
    evrb_g2g.record(compute_stream, iseq);

    // g2h needs to worry about two consumers: et_h2h and h2g.
    // g2h needs to worry about one producer: g2g (since dd1 goes along for the ride).

    evrb_et_h2g.synchronize_with_producer(xx);  // can block!
    evrb_h2g.wait(g2h_stream, iseq - nstreams);
    evrb_g2g.wait(g2h_stream, iseq);
    _launch_g2h(ichunk, ibatch, g2h_stream);
    evrb_g2h.record(g2h_stream, iseq);
    
    // h2g needs to worry about one producer: g2h.
    // h2g needs to worry about one consumer: cdd2.

    evrb_g2h.wait(h2g_stream, iseq - xxx)
    evrb_cdd2.wait(h2g_stream, iseq - S);
    _launch_h2g(ichunk, ibatch, h2g_stream);
    evrb_h2g.record(h2g_stream, iseq);

    // cdd2 needs to worry about two producers: h2g and et_h2g.
    evrb_h2g.wait();
    evrb_et_h2g.wait(compute_stream, xxx, true);  //can block!
    _launch_cdd2(ichunk, ibatch);
}

