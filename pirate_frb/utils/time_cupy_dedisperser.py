"""Timing benchmark for GpuDedisperser using Python/cupy."""

import time

import numpy as np
import cupy as cp

from ..pirate_pybind11 import GpuDequantizationKernel
from .safe_memcpy import safe_h2g_copy


def time_cupy_dedisperser(dedisperser, gpu_allocator, cpu_allocator, niterations):
    """
    Time the GpuDedisperser using Python/cupy, similar to C++ GpuDedisperser::time().

    To run from command line:  'python -m pirate_frb time_dedisperser config.yml --python'.
    (Note that omitting the --python flag will run the C++ version of the timing benchmark,
    which is in GpuDedisperser::time().)

    This function reimplements the timing logic from C++ using cupy for array/stream
    management. It:

    1. Creates the GpuDequantizationKernel
    2. Allocates raw data + scales_offsets arrays using the provided allocators
    3. Runs a timing loop that:
       - Copies raw data (int4) and scales_offsets (float16) from host to device
         on h2g_stream (back-to-back; the stream sequences them)
       - Uses CUDA events to synchronize between h2g_stream and compute_stream
       - Runs dequantization kernel on compute_stream
       - Runs dedispersion kernels via get_input() context manager
       - Measures timing per iteration

    Args:
        dedisperser: An allocated GpuDedisperser instance
        gpu_allocator: BumpAllocator for GPU memory
        cpu_allocator: BumpAllocator for CPU (pinned) memory
        niterations: Number of timing iterations to run
    """

    # Extract key parameters from dedisperser
    config = dedisperser.config
    plan = dedisperser.plan
    stream_pool = dedisperser.stream_pool

    dtype = plan.dtype
    B = plan.beams_per_batch          # beams per batch
    F = plan.nfreq                    # total frequency channels
    T = plan.nt_in                    # time samples per chunk
    S = plan.num_active_batches       # number of streams/active batches
    Tc = 1.0e-3 * T * config.time_sample_ms  # chunk duration in seconds

    assert niterations > 2 * S, f"niterations ({niterations}) must be > 2*num_active_batches ({2*S})"
    assert T % 256 == 0, f"T ({T}) must be divisible by 256"

    print(f"time_cupy_dedisperser: B={B}, F={F}, T={T}, S={S}, Tc={Tc:.3f}s")
    print()

    # Create dequantization kernel (int4 -> float16/float32, with affine transform).
    dequantization_kernel = GpuDequantizationKernel(dtype, B, F, T)

    # Resource tracking for bandwidth calculations
    rt = dedisperser.resource_tracker.clone()
    rt += dequantization_kernel.resource_tracker
    raw_nbytes   = (B * F * T) // 2                  # int4: T elements = T/2 bytes
    scoff_nbytes = B * F * (T // 256) * 2 * 2        # fp16: (scale, offset) per 256 samples
    rt.add_memcpy_h2g("raw_data",       raw_nbytes)
    rt.add_memcpy_h2g("scales_offsets", scoff_nbytes)

    h2g_bw = rt.get_h2g_bw()
    g2h_bw = rt.get_g2h_bw()
    gmem_bw = rt.get_gmem_bw()

    print(f"Expected bandwidth per iteration: h2g={h2g_bw/1e9:.2f} GB, gmem={gmem_bw/1e9:.2f} GB")
    print()

    # Create raw data + scales_offsets arrays.
    # int4 is represented as uint8 with half the elements (two int4 values per uint8 byte),
    # so the raw data shape is (S, B, F, T//2). scales_offsets is fp16 with shape
    # (S, B, F, T//256, 2); last axis is (scale, offset).
    # Note that cpu_allocator returns pinned memory.
    print("time_cupy_dedisperser: allocating raw data + scales_offsets arrays")

    multi_raw_shape   = (S, B, F, T // 2)
    multi_scoff_shape = (S, B, F, T // 256, 2)
    multi_raw_cpu   = cpu_allocator.allocate_array(np.uint8,   multi_raw_shape)
    multi_raw_gpu   = gpu_allocator.allocate_array(cp.uint8,   multi_raw_shape)
    multi_scoff_cpu = cpu_allocator.allocate_array(np.float16, multi_scoff_shape)
    multi_scoff_gpu = gpu_allocator.allocate_array(cp.float16, multi_scoff_shape)

    # Timing loop
    print(f"time_cupy_dedisperser: running {niterations} iterations...")
    print()

    # Warmup and drain any pending work
    cp.cuda.Device().synchronize()

    timestamps = [time.perf_counter()]
    event = cp.cuda.Event(disable_timing=True)

    for iteration in range(niterations):
        # iteration is the seq_id (global batch index).
        istream = iteration % S

        # Setup for current iteration.
        # Use h2g_stream for host->GPU copies, compute_stream for kernels.
        h2g_stream = stream_pool.high_priority_h2g_stream
        compute_stream = stream_pool.compute_streams[istream]
        raw_cpu   = multi_raw_cpu[istream]
        raw_gpu   = multi_raw_gpu[istream]
        scoff_cpu = multi_scoff_cpu[istream]
        scoff_gpu = multi_scoff_gpu[istream]

        # Copy raw data + scales_offsets from CPU to GPU on h2g_stream
        # (back-to-back; the stream sequences them). Use safe_h2g_copy
        # instead of cupy's .set() because raw_cpu / scoff_cpu may live in
        # hugepage-backed BumpAllocator memory whose chunked
        # cudaHostRegister layout breaks an unsplit cudaMemcpyAsync.
        # See plans/python_h2g_chunking.md.
        safe_h2g_copy(raw_gpu,   raw_cpu,   h2g_stream)
        safe_h2g_copy(scoff_gpu, scoff_cpu, h2g_stream)

        # Synchronize compute_stream before recording timestamp.
        # This ensures we measure wall-clock time for the previous iteration's work.
        compute_stream.synchronize()
        timestamps.append(time.perf_counter())

        # Use CUDA event to synchronize: compute_stream waits for h2g_stream.
        # This ensures dequantization kernel doesn't start until H2G copies complete.
        event.record(h2g_stream)
        compute_stream.wait_event(event)

        # Run dequantization and dedispersion kernels.
        # The get_input() context manager handles synchronization with dedisperser.
        with dedisperser.get_input(iteration, stream=compute_stream) as dd_in:
            # The kernel expects uint8 data which it interprets as int4.
            dequantization_kernel.launch(dd_in, scoff_gpu, raw_gpu, compute_stream)
            # Note: exiting the context manager triggers all the dedispersion kernels.
        
        # We're throwing away the output for timing purposes, but we still call get_output()
        # since it performs important synchronization. (get_output() yields an Outputs
        # object with .out_max / .out_argmax attributes; we don't use them here.)
        with dedisperser.get_output(iteration, stream=compute_stream) as outputs:
            pass
        
        # Calculate and print timing after warmup
        # Use the same averaging logic as C++ KernelTimer
        it_min = S + 1
        if iteration > it_min:
            i0 = it_min + (iteration - it_min) // 2
            dt = (timestamps[iteration + 1] - timestamps[i0 + 1]) / (iteration - i0)
            
            real_time_beams = B * Tc / dt
            gmem_bw_achieved = 1.0e-9 * gmem_bw / dt
            g2h_bw_achieved = 1.0e-9 * g2h_bw / dt
            h2g_bw_achieved = 1.0e-9 * h2g_bw / dt
            
            print(f"  iteration {iteration}: real-time beams = {real_time_beams:.2f}, "
                  f"gmem_bw = {gmem_bw_achieved:.2f}, "
                  f"g2h_bw = {g2h_bw_achieved:.2f}, "
                  f"h2g_bw = {h2g_bw_achieved:.2f} GB/s")
    
    print()
    print("time_cupy_dedisperser: timing complete!")
