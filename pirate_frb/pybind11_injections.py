"""
Method injections for pybind11-wrapped pirate classes.
Extends C++ classes with Python convenience methods.
"""

from contextlib import contextmanager

import numpy as np
import ksgpu
from . import pirate_pybind11

# Most pybind11 injections are in this file, with the following exceptions:
#  class FrbGrouper -> pirate_frb/rpc/_FrbGrouper.py


########################################################################################################


@ksgpu.inject_methods(pirate_pybind11.BumpAllocator)
class BumpAllocatorInjections:
    # No class docstring here: BumpAllocator's docstring lives in the pybind11
    # binding (option 1 in notes/docstrings.md). inject_methods would otherwise
    # copy a docstring written here onto the class, overriding the pybind11 one.

    # Save original C++ constructor
    _cpp_init = pirate_pybind11.BumpAllocator.__init__
    
    def __init__(self, aflags, capacity, is_async=False, nthreads=0, cuda_device=-1):
        """
        Create a BumpAllocator.

        Parameters
        ----------
        aflags : int, str, or ksgpu flags
            Memory allocation flags. Can be:
            - int: raw flags (e.g., af_gpu | af_zero)
            - str: 'gpu', 'rhost', 'uhost', etc.
            - Result of ksgpu.parse_aflags()
        capacity : int
            Capacity in bytes.
            - If >= 0: pre-allocates this many bytes
            - If < 0: dummy mode (each array gets independent allocation)
        is_async : bool
            If True, constructor returns immediately; allocation/zeroing
            happens on worker threads. Public methods (allocate_array,
            get_base) block until init complete; failures rethrow.
            Async mode supports exactly these aflag combinations:
              - af_mmap_huge | af_rhost | af_zero (mmap + chunked register)
              - af_rhost | af_zero (cudaHostAlloc + parallel zero)
              - af_gpu | af_zero (cudaMalloc + cudaMemset)
            cuda_device is required (>= 0). nthreads >= 2 for cases 1 and 2;
            ignored for case 3 (af_gpu).
        nthreads : int
            Number of worker threads (async mode only).
        cuda_device : int
            CUDA device id (async mode only; required >= 0).

        Examples
        --------
        >>> # GPU allocator with 1 GB capacity (sync)
        >>> alloc = BumpAllocator('gpu', 1024**3)
        >>>
        >>> # Async host allocator with hugepages
        >>> alloc = BumpAllocator(ksgpu.af_mmap_huge | ksgpu.af_rhost | ksgpu.af_zero,
        ...                       100 * 1024**3, is_async=True, nthreads=8, cuda_device=0)
        >>> alloc.wait_until_initialized()
        """
        aflags = ksgpu.parse_aflags(aflags)
        self._cpp_init(aflags, capacity, is_async, nthreads, cuda_device)
    
    def allocate_array(self, dtype, shape):
        """
        Allocate an array from this allocator.
        
        The array's lifetime is tied to the allocator's base memory region.
        All allocations are aligned to 128-byte cache lines.
        
        Parameters
        ----------
        dtype : str, numpy.dtype, cupy.dtype, or ksgpu.Dtype
            Data type. Examples: 'float32', np.int64, cp.complex64
        shape : list or tuple of int
            Array dimensions
            
        Returns
        -------
        ksgpu.Array
            Allocated array backed by this allocator's memory
            
        Raises
        ------
        RuntimeError
            If allocation would exceed capacity (in normal mode)
            
        Examples
        --------
        >>> alloc = BumpAllocator('gpu', 1024**3)
        >>> arr1 = alloc.allocate_array('float32', (1024, 1024))
        >>> arr2 = alloc.allocate_array(np.int16, (512, 512, 4))
        >>> print(alloc.nbytes_allocated)  # Shows total allocated
        """
        dtype = ksgpu.Dtype(dtype)
        return self._allocate_array_raw(dtype, list(shape))


@ksgpu.inject_methods(pirate_pybind11.CudaStreamPool)
class CudaStreamPoolInjections:
    """Pool of CUDA streams used by the dedispersion pipeline.

    Holds a set of compute streams plus dedicated high/low-priority GPU-to-host
    and host-to-GPU transfer streams.

    The stream accessors (``low_priority_g2h_stream``, ``high_priority_h2g_stream``,
    ``compute_streams``, ...) return ``ksgpu.CudaStreamWrapper`` objects, which can be
    used as cupy stream context managers.
    """
    # This class docstring (above) is the CudaStreamPool docstring: the pybind11
    # binding deliberately sets none, and inject_methods copies this one onto the
    # class (option 2 in notes/docstrings.md).

    # Save references to C++ properties
    _cpp_low_priority_g2h_stream = pirate_pybind11.CudaStreamPool.low_priority_g2h_stream
    _cpp_low_priority_h2g_stream = pirate_pybind11.CudaStreamPool.low_priority_h2g_stream
    _cpp_high_priority_g2h_stream = pirate_pybind11.CudaStreamPool.high_priority_g2h_stream
    _cpp_high_priority_h2g_stream = pirate_pybind11.CudaStreamPool.high_priority_h2g_stream
    _cpp_compute_streams = pirate_pybind11.CudaStreamPool.compute_streams
    
    @property
    def low_priority_g2h_stream(self):
        """Low-priority GPU-to-host transfer stream.
        
        Returns
        -------
        ksgpu.CudaStreamWrapper
            Stream wrapper that can be used with cupy context managers.
            
        Examples
        --------
        >>> pool = CudaStreamPool(num_compute_streams=4)
        >>> with pool.low_priority_g2h_stream:
        ...     # cupy operations will use this stream
        ...     arr = cp.arange(1000)
        """
        return ksgpu.CudaStreamWrapper(self._cpp_low_priority_g2h_stream)
    
    @property
    def low_priority_h2g_stream(self):
        """Low-priority host-to-GPU transfer stream.
        
        Returns
        -------
        ksgpu.CudaStreamWrapper
            Stream wrapper that can be used with cupy context managers.
        """
        return ksgpu.CudaStreamWrapper(self._cpp_low_priority_h2g_stream)
    
    @property
    def high_priority_g2h_stream(self):
        """High-priority GPU-to-host transfer stream.
        
        Returns
        -------
        ksgpu.CudaStreamWrapper
            Stream wrapper that can be used with cupy context managers.
        """
        return ksgpu.CudaStreamWrapper(self._cpp_high_priority_g2h_stream)
    
    @property
    def high_priority_h2g_stream(self):
        """High-priority host-to-GPU transfer stream.
        
        Returns
        -------
        ksgpu.CudaStreamWrapper
            Stream wrapper that can be used with cupy context managers.
        """
        return ksgpu.CudaStreamWrapper(self._cpp_high_priority_h2g_stream)
    
    @property
    def compute_streams(self):
        """List of all compute streams, as ksgpu.CudaStreamWrapper objects.

        Returns
        -------
        list of ksgpu.CudaStreamWrapper
            One wrapper per compute stream; each can be used with cupy
            context managers.
        """
        return [ ksgpu.CudaStreamWrapper(x) for x in self._cpp_compute_streams ]


@ksgpu.inject_methods(pirate_pybind11.CasmBeamformer)
class CasmBeamformerInjections:
    # No class docstring here: CasmBeamformer's docstring lives in the pybind11
    # binding (option 1 in notes/docstrings.md); this injector only adds a stream
    # argument to launch_beamformer().

    # Save reference to C++ method
    _cpp_launch_beamformer = pirate_pybind11.CasmBeamformer.launch_beamformer
    
    def launch_beamformer(self, e_in, feed_weights, i_out, stream=None):
        """Launch beamformer kernel on specified CUDA stream.
        
        Parameters
        ----------
        e_in : ksgpu.Array
            Input electric field array, shape (T, F, 2, 256)
            where T = time samples, F = frequency channels,
            2 = polarizations, 256 = antennas
        feed_weights : ksgpu.Array
            Per-feed beamforming weights, shape (F, 2, 256, 2)
            where last axis is (real, imag)
        i_out : ksgpu.Array
            Output beamformed intensities, shape (Tout, F, B)
            where Tout = T / downsampling_factor, B = number of beams
        stream : cupy.cuda.Stream or None, optional
            CUDA stream to use. If None, uses current cupy stream.
            
        Examples
        --------
        >>> import cupy as cp
        >>> # Use default stream
        >>> bf.launch_beamformer(e_in, feed_weights, i_out)
        >>> 
        >>> # Use explicit stream
        >>> with cp.cuda.Stream() as s:
        ...     bf.launch_beamformer(e_in, feed_weights, i_out, stream=s)
        """
        import cupy as cp
        
        if stream is None:
            stream = cp.cuda.get_current_stream()
        
        # Get raw cudaStream_t pointer from cupy stream
        stream_ptr = stream.ptr
        
        # Call C++ method with stream pointer
        return self._cpp_launch_beamformer(e_in, feed_weights, i_out, stream_ptr)


@ksgpu.inject_methods(pirate_pybind11.GpuDedisperser)
class GpuDedisperserInjections:
    """Low-level C++ GPU dedisperser class, with a python context-manager interface.

    This is probably not the class you want -- you probably want OfflineDedisperser!
    The OfflineDedisperser is a wrapper which handles details like memory allocation
    and parsing AssembledFrames.

    GpuDedisperser has two interfaces: a high-level context manager interface::
    
        self.get_input(seq_id) -> (context manager)
        self.get_output(seq_id) -> (context manager)

    and a lower-level interface which is closer to C++::

        self._acquire_input(seq_id, stream_ptr) -> (cupy array)
        self._release_input_and_launch_dd_kernels(seq_id, stream_ptr) -> None
        self._acquire_output(consumer_id, seq_id, stream_ptr) -> GpuDedisperserOutputs
        self._release_output(consumer_id, seq_id, stream_ptr) -> None

    Processing is one seq_id at a time (seq_id = ichunk*nbatches + ibatch).

    WARNING: don't use input/output arrays outside their context managers -- otherwise
    you'll get a silent race condition!! (The input/output arrays are views into a
    GPU memory ring buffer, and will be overwritten soon after context manager exit.)

    Example code::

        dd = GpuDedisperser(plan, stream_pool, cuda_device_id=0, num_consumers=1)
        dd.allocate(gpu_allocator, host_allocator)
        dd.fill_analytic_weights(freq_variances)   # so out_max comes out as an SNR

        for seq_id in range(nchunks * dd.nbatches):
            # Write this beam-batch's input; leaving the block launches the
            # dedispersion kernels. in_arr has shape (beams_per_batch, nfreq, nt_in).
            with dd.get_input(seq_id) as in_arr:
                in_arr[:] = chunk_data             # e.g. a cupy array

            # Read this seq_id's outputs (only valid inside the block).
            with dd.get_output(seq_id) as outputs:
                for itree in range(dd.ntrees):
                    out_max = outputs.out_max[itree]        # (beams_per_batch, ndm, nt)
                    out_argmax = outputs.out_argmax[itree]
                    ...

    Stream semantics. All synchronization is via CUDA events on the stream you pass
    to get_input()/get_output() (default: cupy's current stream, captured at context
    manager entry). The context managers never synchronize the host with the GPU:

    - get_input(): entry makes 'stream' wait until the input slot is free; exit
      records an event on 'stream' which the dedispersion kernels wait on. So all
      writes to the input array must be ENQUEUED ON 'stream', inside the block.
      They need only be enqueued (not finished) by exit. The buffer holds stale
      data from an earlier chunk -- overwrite it completely.

    - get_output(): entry makes 'stream' wait until the outputs for seq_id are
      ready. (The host may block until the producing kernel has been *launched*,
      but never waits for it to *finish* -- so at entry the outputs are visible
      only to work enqueued on 'stream', not to the host or other streams.) Exit
      records an event on 'stream'; the ring slot is reused only after that event,
      so reads also need only be enqueued (not finished) by exit.

    Gotchas which follow from the above:

    - GPU work submitted on any OTHER stream inside a block is unordered in both
      directions: it can read outputs before the kernels finish, and the ring slot
      can be recycled while it is still running. With default cupy usage, just
      don't change the current stream inside the block.

    - A GPU->host copy of the outputs enqueued on the context manager's stream is
      safe with no host sync (slot reuse is stream-ordered behind it), but the
      HOST buffer is not valid until the copy actually finishes. Blocking copies
      (cupy's arr.get() / cp.asnumpy(), blocking=True by default) handle this;
      after arr.get(blocking=False) you must stream.synchronize() before reading
      the numpy array.

    - Back-pressure: exiting the get_input(seq_id) block blocks the host until
      output (seq_id - nbatches_out) has been released. A single thread driving
      both inputs and outputs (as in the example below) must therefore not let
      output releases lag input submissions by more than nbatches_out batches,
      or it will deadlock.
    """
    # This class docstring (above) is the GpuDedisperser docstring: the pybind11
    # binding deliberately sets none, and inject_methods copies this one onto the
    # class (option 2 in notes/docstrings.md).

    # The get_input()/get_output() context managers below are the Python interface.
    # They wrap the low-level C++ acquire/release methods, which are bound (with a
    # leading underscore, to mark them internal) in pirate_pybind11.cpp:
    #     self._acquire_input(seq_id, stream_ptr)
    #     self._release_input_and_launch_dd_kernels(seq_id, stream_ptr)
    #     self._acquire_output(consumer_id, seq_id, stream_ptr)
    #     self._release_output(consumer_id, seq_id, stream_ptr)
    # These take a raw cudaStream_t pointer (not a cupy stream); call them directly
    # only if you need low-level acquire/release control outside the context managers.

    @contextmanager
    def get_input(self, seq_id, stream=None):
        """Context manager for acquiring and releasing input buffer.

        Acquires the input buffer on entry and releases it on exit. Entry makes
        'stream' wait until the slot is free; exit records an event on 'stream'
        and launches the dedispersion kernels behind it. All writes to the buffer
        must be enqueued on 'stream' inside the block (enqueued suffices -- they
        need not have completed by exit). The buffer holds stale data from an
        earlier chunk: overwrite it completely. See the class docstring ("Stream
        semantics") for gotchas.

        Note: the release-and-launch happens in a 'finally', so leaving the block
        by RAISING an exception still launches the dedispersion kernels on whatever
        is in the input buffer (partially-written or stale). This keeps the internal
        seq_id cursors consistent (the alternative -- skipping the launch -- would
        desync them), at the cost of producing garbage outputs for this seq_id
        rather than an error. So don't rely on an exception here to abort the chunk.

        Parameters
        ----------
        seq_id : int
            Global batch index 0, 1, 2, ... (= ichunk*nbatches + ibatch).
        stream : cupy.cuda.Stream or None, optional
            CUDA stream to use. If None, uses current cupy stream.

        Yields
        ------
        ksgpu.Array
            Input buffer array that can be used as a cupy array.
            
        Examples
        --------
        >>> g = GpuDedisperser(plan, stream_pool)
        >>> g.allocate(gpu_alloc, host_alloc)
        >>> with g.get_input(seq_id=0) as arr:
        ...     arr[:] = input_data  # write to the buffer
        """
        import cupy as cp
        if stream is None:
            stream = cp.cuda.get_current_stream()
        arr = self._acquire_input(seq_id, stream.ptr)
        try:
            yield arr
        finally:
            self._release_input_and_launch_dd_kernels(seq_id, stream.ptr)
    
    @contextmanager
    def get_output(self, seq_id, stream=None, consumer_id=0):
        """Context manager for acquiring and releasing output buffer.

        Acquires the output buffer on entry and yields the Outputs object;
        releases the buffer on exit. Entry makes 'stream' wait until the outputs
        are ready (the host does not wait for the producing kernel to finish, so
        at entry the outputs are visible only to work enqueued on 'stream'). Exit
        records an event on 'stream'; the ring slot is reused only after that
        event, so reads need only be enqueued (not completed) by exit. A GPU->host
        copy enqueued on 'stream' is therefore race-free, but its host buffer is
        only valid once the copy finishes (cupy's arr.get() blocks by default;
        after a non-blocking copy, synchronize the stream before reading it). See
        the class docstring ("Stream semantics") for gotchas.

        Parameters
        ----------
        seq_id : int
            Global batch index 0, 1, 2, ... (= ichunk*nbatches + ibatch).
        stream : cupy.cuda.Stream or None, optional
            CUDA stream to use. If None, uses current cupy stream.
        consumer_id : int, optional
            Output consumer id in [0, num_consumers). Defaults to 0, which
            is correct for the typical num_consumers=1 case.

        Yields
        ------
        GpuDedisperserOutputs
            Object with out_max and out_argmax attributes (see docstring)

        Examples
        --------
        >>> g = GpuDedisperser(plan, stream_pool, cuda_device_id=0, num_consumers=1)
        >>> g.allocate(gpu_alloc, host_alloc)
        >>> with g.get_output(seq_id=0) as outputs:
        ...     for itree in range(g.ntrees):
        ...         process_tree(outputs.out_max[itree], outputs.out_argmax[itree])
        """
        import cupy as cp
        if stream is None:
            stream = cp.cuda.get_current_stream()
        outputs = self._acquire_output(consumer_id, seq_id, stream.ptr)
        try:
            yield outputs
        finally:
            self._release_output(consumer_id, seq_id, stream.ptr)


@ksgpu.inject_methods(pirate_pybind11.SlabAllocator)
class SlabAllocatorInjections:
    # No class docstring here: SlabAllocator's docstring lives in the pybind11
    # binding (option 1 in notes/docstrings.md). inject_methods would otherwise
    # copy a docstring written here onto the class, overriding the pybind11 one.

    # Save original C++ constructor
    _cpp_init = pirate_pybind11.SlabAllocator.__init__
    
    def __init__(self, aflags_or_bump_allocator, nbytes):
        """
        Create a SlabAllocator.
        
        Parameters
        ----------
        aflags_or_bump_allocator : int, str, ksgpu flags, or BumpAllocator
            Either memory allocation flags or a BumpAllocator to get memory from.
            If aflags:
            - int: raw flags (e.g., af_gpu | af_zero)
            - str: 'gpu', 'rhost', 'uhost', etc.
            - Result of ksgpu.parse_aflags()
            If BumpAllocator: memory is allocated from the BumpAllocator.
        nbytes : int
            Capacity in bytes.
            - If aflags and >= 0: pre-allocates this many bytes, subdivided into slabs
            - If aflags and < 0: dummy mode (each get_slab() allocates fresh memory)
            - If BumpAllocator: must be positive
        
        Examples
        --------
        >>> # Host allocator with 100 MB capacity
        >>> alloc = SlabAllocator('rhost', 100 * 1024**2)
        >>> 
        >>> # GPU allocator with 1 GB capacity
        >>> alloc = SlabAllocator('gpu', 1024**3)
        >>> 
        >>> # Dummy mode (no pre-allocation)
        >>> alloc = SlabAllocator('rhost', -1)
        >>>
        >>> # From a BumpAllocator
        >>> bump = BumpAllocator('rhost', 1024**3)
        >>> slab = SlabAllocator(bump, 100 * 1024**2)
        """
        if isinstance(aflags_or_bump_allocator, pirate_pybind11.BumpAllocator):
            self._cpp_init(aflags_or_bump_allocator, nbytes)
        else:
            aflags = ksgpu.parse_aflags(aflags_or_bump_allocator)
            self._cpp_init(aflags, nbytes)


@ksgpu.inject_methods(pirate_pybind11.GpuDequantizationKernel)
class GpuDequantizationKernelInjections:
    # No class docstring here: GpuDequantizationKernel's docstring lives in the
    # pybind11 binding (option 1 in notes/docstrings.md); this injector adds dtype
    # conversion for the constructor and a stream argument for launch().

# Save references to C++ methods
    _cpp_init = pirate_pybind11.GpuDequantizationKernel.__init__
    _cpp_launch = pirate_pybind11.GpuDequantizationKernel.launch

    def __init__(self, dtype, nbeams, nfreq, ntime):
        """Create a GpuDequantizationKernel.

        Applies the per-(beam, freq, minichunk) affine transform

            out[b,f,t] = 0                                       if data[b,f,t] == -8
            out[b,f,t] = scales_offsets[b,f,t//256,0] * data[b,f,t]
                       + scales_offsets[b,f,t//256,1]            otherwise

        during int4 -> float32/float16 conversion. data == -8 is the
        "missing sample" sentinel (matching the convention of
        AssembledFrame.data) and is always mapped to 0 in the output,
        regardless of scale and offset.

        Parameters
        ----------
        dtype : str, numpy.dtype, cupy.dtype, or ksgpu.Dtype
            Output dtype (must be float32 or float16)
        nbeams : int
            Number of beams
        nfreq : int
            Number of frequency channels
        ntime : int
            Number of time samples (must be divisible by 256)
        """
        dtype = ksgpu.Dtype(dtype)
        self._cpp_init(dtype, nbeams, nfreq, ntime)

    def launch(self, out, scales_offsets, data_uint8, stream=None):
        """GPU kernel launch (async, does not sync stream).

        Parameters
        ----------
        out : ksgpu.Array
            Output array, shape (nbeams, nfreq, ntime), dtype matches
            kernel's dtype (float32 or float16), fully contiguous, on GPU.
        scales_offsets : ksgpu.Array
            Shape (nbeams, nfreq, ntime//256, 2), dtype float16, fully
            contiguous, on GPU. Last axis is (scale, offset); one pair is
            applied to every int4 sample in the matching (beam, freq,
            minichunk) slice of data_uint8.
        data_uint8 : ksgpu.Array
            Shape (nbeams, nfreq, ntime//2), dtype uint8, fully contiguous,
            on GPU. Reinterpreted as int4 with shape (nbeams, nfreq, ntime).
        stream : cupy.cuda.Stream or None, optional
            CUDA stream to use. If None, uses current cupy stream.

        Note
        ----
        The data array is passed as uint8 because numpy/cupy don't support int4
        (all dtypes must be at least 8 bits). Each uint8 element contains two
        int4 values: low nibble = even index, high nibble = odd index.

        data == -8 is mapped to 0 in the output regardless of scale and
        offset; see the __init__ docstring above.
        """
        import cupy as cp

        if stream is None:
            stream = cp.cuda.get_current_stream()

        self._cpp_launch(out, scales_offsets, data_uint8, stream.ptr)


@ksgpu.inject_methods(pirate_pybind11.DedispersionConfig)
class DedispersionConfigInjections:
    # No class docstring here: DedispersionConfig's docstring lives in the pybind11
    # binding (option 1 in notes/docstrings.md); this injector adds a flexible dtype
    # setter that accepts strings, numpy/cupy dtypes, etc.

    # Save reference to C++ dtype attribute
    _cpp_dtype = pirate_pybind11.DedispersionConfig.dtype
    
    @property
    def dtype(self):
        """Data type for dedispersion.
        
        Returns
        -------
        ksgpu.Dtype
            The current dtype setting.
        """
        return self._cpp_dtype
    
    @dtype.setter
    def dtype(self, value):
        """Set the data type for dedispersion.
        
        Parameters
        ----------
        value : str, numpy.dtype, cupy.dtype, or ksgpu.Dtype
            Data type specification. Examples:
            - 'float32', 'float16'
            - np.float32, cp.float16
            - ksgpu.Dtype('float32')
            
        Examples
        --------
        >>> config = DedispersionConfig()
        >>> config.dtype = 'float32'
        >>> config.dtype = np.float16
        >>> config.dtype = ksgpu.Dtype('float32')
        """
        self._cpp_dtype = ksgpu.Dtype(value)


@ksgpu.inject_methods(pirate_pybind11.SimulatedFrameFactory)
class SimulatedFrameFactoryInjections:
    # No class docstring here: SimulatedFrameFactory's docstring lives in the pybind11
    # binding (option 1 in notes/docstrings.md); this injector only adds pop_events().

    def pop_events(self, chunk_fpga_start, chunk_fpga_end):
        """Return the FRB-injection events recorded since the last call, as an FrbSifterEvents.

        Wraps the C++ _pop_events() (which returns a list of SimulatedFrameFactoryEvent and clears
        the internal list, so each event is returned exactly once), converting it to a
        pirate_frb.rpc.FrbSifterEvents with rfi_probs = 0 and the given FPGA-counter window.

        Parameters
        ----------
        chunk_fpga_start, chunk_fpga_end : int
            Absolute FPGA-counter window (start/end of the time chunk) for this batch of events.

        Returns
        -------
        pirate_frb.rpc.FrbSifterEvents
            One entry per injected FRB (rfi_prob = 0). Empty (length-0 arrays) if no FRBs were
            injected since the last call.
        """

        # Lazy import to avoid an import cycle (rpc -> ... at module-load time).
        from .rpc import FrbSifterEvents

        # One awkward aspect of the current code: we have two very similar ways to represent an event
        # list, either a python FrbSifterEvents object, or a C++ vector<SimulatedFrameFactory::Event>.
        # This function converts between the two, at the "python/C++ boundary".
        #
        # This design is awkward but I think it's the least bad option. (The only real alternative seems
        # to be rewriting the FrbSifterClient in C++, and I prefer the current python implementation.)
        
        events = self._pop_events()
        n = len(events)
        return FrbSifterEvents(
            beam_ids             = np.array([e.beam_id             for e in events], dtype=np.int32),
            fpga_timestamps      = np.array([e.fpga_timestamp      for e in events], dtype=np.int64),
            dms                  = np.array([e.dm                  for e in events], dtype=np.float32),
            snrs                 = np.array([e.snr                 for e in events], dtype=np.float32),
            rfi_probs            = np.zeros(n, dtype=np.float32),
            widths_ms            = np.array([e.width_ms            for e in events], dtype=np.float32),
            subband_freqs_lo_MHz = np.array([e.subband_freq_lo_MHz for e in events], dtype=np.float32),
            subband_freqs_hi_MHz = np.array([e.subband_freq_hi_MHz for e in events], dtype=np.float32),
            chunk_fpga_start     = chunk_fpga_start,
            chunk_fpga_end       = chunk_fpga_end)
