"""Host<->device cudaMemcpy* wrappers that handle BumpAllocator chunked
hugepage registration.

cupy's `ndarray.set()` / `.get()` call cudaMemcpyAsync directly with no
splitting at cudaHostRegister chunk boundaries. When the host buffer
lives in a pirate hugepage-backed BumpAllocator
(`af_rhost | af_zero | af_mmap_huge`), such a copy can fail with
cudaErrorInvalidValue. The wrappers below delegate to pirate's
`safe_memcpy_*` C++ helpers, which split the copy at
cuda_host_register_chunk_size-aligned host addresses. See
`plans/python_h2g_chunking.md` and the doc-comment block at the top of
`include/pirate/utils.hpp`.
"""

from ..pirate_pybind11 import (
    safe_memcpy_h2g_async as _safe_memcpy_h2g_async_cpp,
    safe_memcpy_g2h_async as _safe_memcpy_g2h_async_cpp,
)


def _host_ptr(arr):
    """Extract a host data pointer from a numpy ndarray."""
    return int(arr.__array_interface__['data'][0])


def _check_h2g(gpu_arr, cpu_arr):
    if cpu_arr.nbytes != gpu_arr.nbytes:
        raise ValueError(
            f"nbytes mismatch: cpu={cpu_arr.nbytes}, gpu={gpu_arr.nbytes}")
    if not cpu_arr.flags.c_contiguous:
        raise ValueError("cpu_arr must be C-contiguous")
    if not gpu_arr.flags.c_contiguous:
        raise ValueError("gpu_arr must be C-contiguous")


def safe_h2g_copy(gpu_arr, cpu_arr, stream):
    """Async host->device copy with BumpAllocator chunk-aware splitting.

    Drop-in replacement for ``gpu_arr.set(cpu_arr, stream=stream)`` when
    ``cpu_arr`` may live in hugepage-backed pinned memory from a pirate
    BumpAllocator.

    Parameters
    ----------
    gpu_arr : cupy.ndarray
        Contiguous destination.
    cpu_arr : numpy.ndarray
        Contiguous source with matching nbytes.
    stream : ksgpu.CudaStreamWrapper or cupy.cuda.Stream
        Stream to issue the async copy on; must expose ``.ptr``.
    """
    _check_h2g(gpu_arr, cpu_arr)
    _safe_memcpy_h2g_async_cpp(
        int(gpu_arr.data.ptr),
        _host_ptr(cpu_arr),
        int(cpu_arr.nbytes),
        int(stream.ptr),
    )


def safe_g2h_copy(cpu_arr, gpu_arr, stream):
    """Async device->host copy with BumpAllocator chunk-aware splitting.

    Drop-in replacement for ``gpu_arr.get(out=cpu_arr, stream=stream)``
    when ``cpu_arr`` may live in hugepage-backed pinned memory from a
    pirate BumpAllocator.

    Parameters
    ----------
    cpu_arr : numpy.ndarray
        Contiguous destination.
    gpu_arr : cupy.ndarray
        Contiguous source with matching nbytes.
    stream : ksgpu.CudaStreamWrapper or cupy.cuda.Stream
        Stream to issue the async copy on; must expose ``.ptr``.
    """
    _check_h2g(gpu_arr, cpu_arr)  # same checks, names swapped in callers
    _safe_memcpy_g2h_async_cpp(
        _host_ptr(cpu_arr),
        int(gpu_arr.data.ptr),
        int(cpu_arr.nbytes),
        int(stream.ptr),
    )
