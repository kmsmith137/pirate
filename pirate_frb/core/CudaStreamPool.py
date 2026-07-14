"""CudaStreamPool method injections (+ re-export of the pybind11 class)."""

import ksgpu
from ..pirate_pybind11 import CudaStreamPool


@ksgpu.inject_methods(CudaStreamPool)
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
    _cpp_low_priority_g2h_stream = CudaStreamPool.low_priority_g2h_stream
    _cpp_low_priority_h2g_stream = CudaStreamPool.low_priority_h2g_stream
    _cpp_high_priority_g2h_stream = CudaStreamPool.high_priority_g2h_stream
    _cpp_high_priority_h2g_stream = CudaStreamPool.high_priority_h2g_stream
    _cpp_compute_streams = CudaStreamPool.compute_streams

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
