"""
Method injections for pybind11-wrapped pirate classes.
Extends C++ classes with Python convenience methods.
"""

from contextlib import contextmanager

import ksgpu
from . import pirate_pybind11


########################################################################################################


@ksgpu.inject_methods(pirate_pybind11.BumpAllocator)
class BumpAllocatorInjections:
    """Python extensions for BumpAllocator."""
    
    # Save original C++ constructor
    _cpp_init = pirate_pybind11.BumpAllocator.__init__
    
    def __init__(self, aflags, capacity):
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
        
        Examples
        --------
        >>> # GPU allocator with 1 GB capacity
        >>> alloc = BumpAllocator('gpu', 1024**3)
        >>> 
        >>> # Host allocator (registered) with 100 MB
        >>> alloc = BumpAllocator('rhost', 100 * 1024**2)
        >>> 
        >>> # Dummy mode
        >>> alloc = BumpAllocator('gpu', -1)
        """
        aflags = ksgpu.parse_aflags(aflags)
        self._cpp_init(aflags, capacity)
    
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
    """Python extensions for CudaStreamPool.
    
    Wraps C++ stream members as Python-friendly ksgpu.CudaStreamWrapper objects
    that can be used with cupy context managers.
    """
    
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
        """First compute stream (compute_streams[0]).
        
        Returns
        -------
        ksgpu.CudaStreamWrapper
            Stream wrapper that can be used with cupy context managers.
        """
        return [ ksgpu.CudaStreamWrapper(x) for x in self._cpp_compute_streams ]


@ksgpu.inject_methods(pirate_pybind11.CasmBeamformer)
class CasmBeamformerInjections:
    """Python extensions for CasmBeamformer.
    
    Adds stream argument support to launch_beamformer.
    """
    
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
    """Python extensions for GpuDedisperser.
    
    Adds stream argument support to acquire/release methods.
    """
    
    # Save references to C++ methods
    _cpp_acquire_input = pirate_pybind11.GpuDedisperser.acquire_input
    _cpp_release_input = pirate_pybind11.GpuDedisperser.release_input
    _cpp_acquire_output = pirate_pybind11.GpuDedisperser.acquire_output
    _cpp_release_output = pirate_pybind11.GpuDedisperser.release_output
    
    def acquire_input(self, ichunk, ibatch, stream=None):
        """Acquire input buffer for writing.
        
        After this call returns, 'stream' sees an empty input buffer ready for writing.
        Use view_input() to get the input buffer array.
        
        Parameters
        ----------
        ichunk : int
            Chunk index
        ibatch : int
            Batch index
        stream : cupy.cuda.Stream or None, optional
            CUDA stream to use. If None, uses current cupy stream.
        """
        import cupy as cp
        if stream is None:
            stream = cp.cuda.get_current_stream()
        self._cpp_acquire_input(ichunk, ibatch, stream.ptr)
    
    def release_input(self, ichunk, ibatch, stream=None):
        """Release input buffer after writing.
        
        Before calling this, 'stream' must see a full input buffer.
        
        Parameters
        ----------
        ichunk : int
            Chunk index
        ibatch : int
            Batch index
        stream : cupy.cuda.Stream or None, optional
            CUDA stream to use. If None, uses current cupy stream.
        """
        import cupy as cp
        if stream is None:
            stream = cp.cuda.get_current_stream()
        self._cpp_release_input(ichunk, ibatch, stream.ptr)
    
    def acquire_output(self, ichunk, ibatch, stream=None):
        """Acquire output buffer for reading.
        
        After this call returns, 'stream' sees a full output buffer ready for reading.
        
        Parameters
        ----------
        ichunk : int
            Chunk index
        ibatch : int
            Batch index
        stream : cupy.cuda.Stream or None, optional
            CUDA stream to use. If None, uses current cupy stream.
        """
        import cupy as cp
        if stream is None:
            stream = cp.cuda.get_current_stream()
        self._cpp_acquire_output(ichunk, ibatch, stream.ptr)
    
    def release_output(self, ichunk, ibatch, stream=None):
        """Release output buffer after reading.
        
        Before calling this, 'stream' must see an empty output buffer.
        
        Parameters
        ----------
        ichunk : int
            Chunk index
        ibatch : int
            Batch index
        stream : cupy.cuda.Stream or None, optional
            CUDA stream to use. If None, uses current cupy stream.
        """
        import cupy as cp
        if stream is None:
            stream = cp.cuda.get_current_stream()
        self._cpp_release_output(ichunk, ibatch, stream.ptr)
    
    @contextmanager
    def get_input(self, ichunk, ibatch, stream=None):
        """Context manager for acquiring and releasing input buffer.
        
        Acquires the input buffer on entry and releases it on exit.
        
        Parameters
        ----------
        ichunk : int
            Chunk index
        ibatch : int
            Batch index
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
        >>> with g.get_input(ichunk=0, ibatch=0) as arr:
        ...     arr[:] = input_data  # write to the buffer
        """
        import cupy as cp
        if stream is None:
            stream = cp.cuda.get_current_stream()
        self._cpp_acquire_input(ichunk, ibatch, stream.ptr)
        arr = self.view_input(ichunk, ibatch)
        try:
            yield arr
        finally:
            self._cpp_release_input(ichunk, ibatch, stream.ptr)
    
    @contextmanager
    def get_output(self, ichunk, ibatch, stream=None):
        """Context manager for acquiring and releasing output buffer.
        
        Acquires the output buffer on entry and releases it on exit.
        
        Parameters
        ----------
        ichunk : int
            Chunk index
        ibatch : int
            Batch index
        stream : cupy.cuda.Stream or None, optional
            CUDA stream to use. If None, uses current cupy stream.
            
        Yields
        ------
        tuple (out_max, out_argmax)
            out_max : list of ksgpu.Array
                Peak-finding maximum values, one array per tree.
            out_argmax : list of ksgpu.Array
                Peak-finding argmax tokens, one array per tree.
            
        Examples
        --------
        >>> g = GpuDedisperser(plan, stream_pool)
        >>> g.allocate(gpu_alloc, host_alloc)
        >>> with g.get_output(ichunk=0, ibatch=0) as (out_max, out_argmax):
        ...     for itree in range(g.ntrees):
        ...         process_tree(out_max[itree], out_argmax[itree])
        """
        import cupy as cp
        if stream is None:
            stream = cp.cuda.get_current_stream()
        self._cpp_acquire_output(ichunk, ibatch, stream.ptr)
        out_max = self.view_out_max(ichunk, ibatch)
        out_argmax = self.view_out_argmax(ichunk, ibatch)
        try:
            yield (out_max, out_argmax)
        finally:
            self._cpp_release_output(ichunk, ibatch, stream.ptr)


@ksgpu.inject_methods(pirate_pybind11.SlabAllocator)
class SlabAllocatorInjections:
    """Python extensions for SlabAllocator."""
    
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


@ksgpu.inject_methods(pirate_pybind11.DedispersionConfig)
class DedispersionConfigInjections:
    """Python extensions for DedispersionConfig.
    
    Adds flexible dtype setter that accepts strings, numpy/cupy dtypes, etc.
    """
    
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


