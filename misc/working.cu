
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

    xxx;  // sleep and synchronize, so that et thread is not behind

    // We use a ksgpu::KernelTimer in the timing loop, for the sake of tradition,
    // but it's a little awkward since the KernelTimer defines its own stream pool.
    // In the loop below, we keep the KernelTimer streams synchronized with the 
    // GpuDedisperser streams.

    KernelTimer kt(niterations, nstreams);

    while (kt.next()) {
        long ichunk = kt.curr_iteration / nbatches;
        long ibatch = kt.curr_iteration % nbatches;

        cudaStream_t h2g_stream = this->stream_pool.high_priority_h2g;
        cudaStream_t compute_stream = this->stream_pool.compute_streams.at(kt.istream);

        Array<void> raw_cpu = multi_raw_cpu.slice(0, kt.istream);
        Array<void> raw_gpu = multi_raw_gpu.slice(0, kt.istream);

        CUDA_CALL(cudaMemcpyAsync(raw_gpu.data, raw_cpu.data, raw_nbytes, cudaMemcpyHostToDevice, h2g_stream));

        // Run dequantization kernel on compute stream.
        xxx;  // synchronize h2g_stream -> compute_stream
        Array<void> dd_in = this->acquire_input(ichunk, ibatch, compute_stream);
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