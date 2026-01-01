// Static member function.
void GpuDedisperser::time(long num_iterations, bool use_hugepages)
{
    GpuDequantizationKernel dequantization_kernel(dtype, beams_per_batch, nfreq, nt_in);

    ResourceTracker rt = this->resource_tracker;  // copy
    rt += dequantization_kernel.resource_tracker;
    rt.add_memcpy_h2g("raw_data", raw_nbytes);

    double h2g_bw = rt.get_h2g_bw();
    double g2h_bw = rt.get_g2h_bw();
    double gmem_bw = rt.get_gmem_bw();

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

    CudaEventRingbuf evrb_raw;  // copy raw data from cpu->gpu
    CudaEventRingbuf evrb_dq;   // dequantization kernel

    // We use a ksgpu::KernelTimer in the timing loop

    KernelTimer kt(niterations, nstreams);

    while (kt.next()) {
        long seq_id = kt.curr_iteration;
        long ichunk = seq_id / nbatches;
        long ibatch = seq_id % nbatches;

        cudaStream_t h2g_stream = this->stream_pool.high_priority_h2g;
        cudaStream_t compute_stream = this->stream_pool.compute_streams.at(kt.istream);

        Array<void> raw_cpu = multi_raw_cpu.slice(0, kt.istream);
        Array<void> raw_gpu = multi_raw_gpu.slice(0, kt.istream);

        // Copy raw data cpu->gpu. 
        // First, wait on the dequantization consumer.
        evrb_dq.wait(h2g_stream, seq_id - nstreams);
        CUDA_CALL(cudaMemcpyAsync(raw_gpu.data, raw_cpu.data, raw_nbytes, cudaMemcpyHostToDevice, h2g_stream));
        evrb_raw.record(h2g_stream, seq_id);

        // I decided to synchronize the KernelTimer stream at this point.
        // This is done by making the KernelTimer stream a consumer of 'evrb_raw' (a CudaEventRingbuf).
        evrb_raw.wait(kt.stream, seq_id);

        // Run dequantization kernel on compute stream. 
        // First, wait on the producer (the cpu->gpu copy).
        // First, wait on the consumer (the dedisperser), by calling this->acquire_input().
        evrb_raw.wait(compute_stream, );
        Array<void> dd_in = this->acquire_input(ichunk, ibatch, compute_stream);
        dequantization_kernel.launch(dd_in, raw_gpu, compute_stream);

        // Launch all the dedispersion kernels.
        // (They will wait for the dequantization kernel.)
        this->release_input(ichunk, ibatch, compute_stream);

        // Throw away the dedispersion output.
        this->acquire_output(ichunk, ibatch, compute_stream);
        this->release_output(ichunk, ibatch, compute_stream);

        if (kt.warmed_up) {
            cout << "iteration " << kt.curr_iteration
                 << ": beams/sec = " << (beams_per_batch / kt.dt)
                 << ": gmem_bw = " << (1.0e-9 * gmem_bw / kt.dt)
                 << ", g2h_bw = " << (1.0e-9 * g2h_bw / kt.dt)
                 << ", h2g_bw = " << (1.0e-9 * h2g_bw / kt.dt)
                 << " GB/s" << endl;
        }
    }
}