#!/usr/bin/env python
"""
Python reimplementation of GpuDedisperser::time_one() for testing cupy/pybind11 interface.

This script times the GpuDedisperser using Python/cupy, demonstrating how to:
  - Load a DedispersionConfig from YAML
  - Create GpuDedisperser and GpuDequantizationKernel
  - Use acquire_input/release_input/acquire_output/release_output with cupy streams
  - Run and time the dedispersion pipeline

Usage:
    python time_dedisperser.py [config_file] [-n niterations]

Example:
    python time_dedisperser.py configs/dedispersion/chord_sb2_et.yml -n 50

Compare with C++ version:
    pirate_frb show_dedisperser --time configs/dedispersion/chord_sb2_et.yml -n 50
"""

import time
import argparse

import numpy as np
import cupy as cp

from pirate_frb.pirate_pybind11 import (
    DedispersionConfig,
    DedispersionPlan,
    GpuDedisperser,
    GpuDequantizationKernel,
    BumpAllocator,
    CudaStreamPool,
)


def time_dedisperser(config_file: str, niterations: int):
    """
    Time the GpuDedisperser using Python/cupy, reimplementing the C++ time_one() logic.
    
    This function:
    1. Loads a dedispersion config from YAML
    2. Creates the GpuDedisperser and GpuDequantizationKernel
    3. Allocates all necessary buffers
    4. Runs a timing loop that:
       - Copies raw data (int4) from host to device
       - Runs dequantization kernel
       - Runs dedispersion kernels
       - Measures timing per iteration
    """
    
    # Load and validate config
    config = DedispersionConfig.from_yaml(config_file)
    config.validate()
    
    print(config.to_yaml_string())
    print()
    
    # Extract key parameters
    dtype = config.dtype
    B = config.beams_per_batch         # beams per batch
    F = config.get_total_nfreq()       # total frequency channels
    T = config.time_samples_per_chunk  # time samples per chunk
    S = config.num_active_batches      # number of streams/active batches
    Tc = 1.0e-3 * T * config.time_sample_ms  # chunk duration in seconds
    
    assert niterations > 2 * S, f"niterations ({niterations}) must be > 2*num_active_batches ({2*S})"
    
    print(f"Parameters: B={B}, F={F}, T={T}, S={S}, Tc={Tc:.3f}s")
    print()
    
    # Create dequantization kernel (int4 -> float16/float32)
    dequantization_kernel = GpuDequantizationKernel(dtype, B, F, T)
    
    # Create GpuDedisperser
    print("Creating GpuDedisperser...")
    plan = DedispersionPlan(config)
    stream_pool = CudaStreamPool(S)
    gdd = GpuDedisperser(plan, stream_pool, detect_deadlocks=True)
    
    # Resource tracking for bandwidth calculations
    rt = gdd.resource_tracker.clone()
    rt += dequantization_kernel.resource_tracker
    raw_nbytes = (B * F * T) // 2  # int4 = 4 bits, so T elements = T/2 bytes
    rt.add_memcpy_h2g("raw_data", raw_nbytes)
    
    h2g_bw = rt.get_h2g_bw()
    g2h_bw = rt.get_g2h_bw()
    gmem_bw = rt.get_gmem_bw()
    
    print(f"Expected bandwidth per iteration: h2g={h2g_bw/1e9:.2f} GB, gmem={gmem_bw/1e9:.2f} GB")
    print()
    
    # Allocate GpuDedisperser buffers using dummy allocators
    print("Allocating buffers...")
    gpu_allocator = BumpAllocator('af_gpu | af_zero', -1)  # -1 = dummy mode
    host_allocator = BumpAllocator('af_rhost | af_zero', -1)
    gdd.allocate(gpu_allocator, host_allocator)
    
    # Create raw data arrays
    # int4 is represented as uint8 with half the elements (two int4 values per uint8 byte)
    # Shape is (S, B, F, T//2) for uint8 representation.
    # Note that host_allocator returns pinned memory.

    multi_raw_shape = (S, B, F, T // 2)
    multi_raw_cpu = host_allocator.allocate_array(np.uint8, multi_raw_shape)
    multi_raw_gpu = gpu_allocator.allocate_array(cp.uint8, multi_raw_shape)
    
    # Timing loop
    print(f"Running {niterations} iterations...")
    print()
    
    # Warmup and drain any pending work
    cp.cuda.Device().synchronize()
    
    timestamps = [time.perf_counter()]
    event = cp.cuda.Event(disable_timing=True)
    
    for iteration in range(niterations):
        # Compute chunk/batch indices
        ichunk = iteration // gdd.nbatches
        ibatch = iteration % gdd.nbatches
        istream = iteration % S

        # Setup for current iteration.
        h2g_stream = stream_pool.high_priority_h2g_stream
        compute_stream = stream_pool.compute_streams[istream]
        raw_cpu = multi_raw_cpu[istream]
        raw_gpu = multi_raw_gpu[istream]

        # Copy raw data from CPU to GPU.
        raw_gpu.set(raw_cpu, stream = h2g_stream)
        
        # Synchronize here for timing measurement.
        compute_stream.synchronize()
        timestamps.append(time.perf_counter())

        # Next step is dequantization kernel.
        # Before launching, need to wait on producer (host->gpu copy) using a cuda event.
        # Need to wait on consumer (dedisperser) using the with-statement below.

        event.record(h2g_stream)
        compute_stream.wait_event(event)

        with gdd.get_input(ichunk, ibatch, stream = compute_stream) as dd_in:
            # The kernel expects uint8 input which it interprets as int4.
            # Note: launch() takes a raw stream pointer (int), not a CudaStreamWrapper.
            dequantization_kernel.launch(dd_in, raw_gpu, compute_stream)
            # Note that exiting the context manager triggers all the dedispersion kernels.
        
        # Acquire and release output (we're throwing away the output for timing purposes)
        gdd.acquire_output(ichunk, ibatch, compute_stream)
        gdd.release_output(ichunk, ibatch, compute_stream)
        
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
    print("Timing complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Time GpuDedisperser using Python/cupy interface",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'config_file',
        nargs='?',
        default='configs/dedispersion/chord_sb2_et.yml',
        help="Path to YAML config file"
    )
    parser.add_argument(
        '-n', '--niter',
        type=int,
        default=50,
        help="Number of timing iterations"
    )
    
    args = parser.parse_args()
    time_dedisperser(args.config_file, args.niter)


if __name__ == "__main__":
    main()
