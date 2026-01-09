"""Utility functions and context managers for pirate_frb."""

import time

import numpy as np
import cupy as cp

from ..pirate_pybind11 import get_thread_affinity, set_thread_affinity
from ..pirate_pybind11 import GpuDequantizationKernel


class ThreadAffinity:
    """Context manager for temporarily setting thread CPU affinity.
    
    On entry, sets the calling thread's affinity to the specified vCPUs.
    On exit, restores the original affinity.
    
    Example:
        with ThreadAffinity([2, 3]):
            # Thread is pinned to vCPUs 2 and 3
            do_work()
        # Original affinity is restored
    
    Args:
        vcpu_list: List of vCPU indices to pin the thread to.
    """
    
    def __init__(self, vcpu_list):
        self.vcpu_list = vcpu_list
        self.saved_affinity = None
    
    def __enter__(self):
        self.saved_affinity = get_thread_affinity()
        set_thread_affinity(self.vcpu_list)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        set_thread_affinity(self.saved_affinity)
        return False  # Don't suppress exceptions


def time_cupy_dedisperser(dedisperser, gpu_allocator, cpu_allocator, niterations):
    """
    Time the GpuDedisperser using Python/cupy, similar to C++ GpuDedisperser::time().
    
    This function reimplements the timing logic from C++ using cupy for array/stream
    management. It:
    1. Creates the GpuDequantizationKernel
    2. Allocates raw data arrays using the provided allocators
    3. Runs a timing loop that:
       - Copies raw data (int4) from host to device on h2g_stream
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
    
    print(f"time_cupy_dedisperser: B={B}, F={F}, T={T}, S={S}, Tc={Tc:.3f}s")
    print()
    
    # Create dequantization kernel (int4 -> float16/float32)
    dequantization_kernel = GpuDequantizationKernel(dtype, B, F, T)
    
    # Resource tracking for bandwidth calculations
    rt = dedisperser.resource_tracker.clone()
    rt += dequantization_kernel.resource_tracker
    raw_nbytes = (B * F * T) // 2  # int4 = 4 bits, so T elements = T/2 bytes
    rt.add_memcpy_h2g("raw_data", raw_nbytes)
    
    h2g_bw = rt.get_h2g_bw()
    g2h_bw = rt.get_g2h_bw()
    gmem_bw = rt.get_gmem_bw()
    
    print(f"Expected bandwidth per iteration: h2g={h2g_bw/1e9:.2f} GB, gmem={gmem_bw/1e9:.2f} GB")
    print()
    
    # Create raw data arrays
    # int4 is represented as uint8 with half the elements (two int4 values per uint8 byte)
    # Shape is (S, B, F, T//2) for uint8 representation.
    # Note that cpu_allocator returns pinned memory.
    print("time_cupy_dedisperser: allocating raw data arrays")
    
    multi_raw_shape = (S, B, F, T // 2)
    multi_raw_cpu = cpu_allocator.allocate_array(np.uint8, multi_raw_shape)
    multi_raw_gpu = gpu_allocator.allocate_array(cp.uint8, multi_raw_shape)
    
    # Timing loop
    print(f"time_cupy_dedisperser: running {niterations} iterations...")
    print()
    
    # Warmup and drain any pending work
    cp.cuda.Device().synchronize()
    
    timestamps = [time.perf_counter()]
    event = cp.cuda.Event(disable_timing=True)
    
    for iteration in range(niterations):
        # Compute chunk/batch indices
        ichunk = iteration // dedisperser.nbatches
        ibatch = iteration % dedisperser.nbatches
        istream = iteration % S

        # Setup for current iteration.
        # Use h2g_stream for host->GPU copies, compute_stream for kernels.
        h2g_stream = stream_pool.high_priority_h2g_stream
        compute_stream = stream_pool.compute_streams[istream]
        raw_cpu = multi_raw_cpu[istream]
        raw_gpu = multi_raw_gpu[istream]

        # Copy raw data from CPU to GPU on h2g_stream.
        raw_gpu.set(raw_cpu, stream=h2g_stream)
        
        # Synchronize compute_stream before recording timestamp.
        # This ensures we measure wall-clock time for the previous iteration's work.
        compute_stream.synchronize()
        timestamps.append(time.perf_counter())

        # Use CUDA event to synchronize: compute_stream waits for h2g_stream.
        # This ensures dequantization kernel doesn't start until H2G copy completes.
        event.record(h2g_stream)
        compute_stream.wait_event(event)

        # Run dequantization and dedispersion kernels.
        # The get_input() context manager handles synchronization with dedisperser.
        with dedisperser.get_input(ichunk, ibatch, stream=compute_stream) as dd_in:
            # The kernel expects uint8 input which it interprets as int4.
            dequantization_kernel.launch(dd_in, raw_gpu, compute_stream)
            # Note: exiting the context manager triggers all the dedispersion kernels.
        
        # We're throwing away the output for timing purposes, but we still call get_output()
        # since it performs important synchronization.
        with dedisperser.get_output(ichunk, ibatch, stream=compute_stream) as (out_max, out_argmax):
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


__all__ = ['ThreadAffinity', 'get_thread_affinity', 'set_thread_affinity', 'time_cupy_dedisperser']
