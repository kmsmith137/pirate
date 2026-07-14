"""CasmBeamformer method injections (+ re-export of the pybind11 class)."""

import ksgpu
from ..pirate_pybind11 import CasmBeamformer


@ksgpu.inject_methods(CasmBeamformer)
class CasmBeamformerInjections:
    # No class docstring here: CasmBeamformer's docstring lives in the pybind11
    # binding (option 1 in notes/docstrings.md); this injector only adds a stream
    # argument to launch_beamformer().

    # Save reference to C++ method
    _cpp_launch_beamformer = CasmBeamformer.launch_beamformer

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
