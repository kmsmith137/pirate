"""Python method injections for ChimeDequantizationKernel (a C++ class, see
src_lib/chimefrb/ChimeDequantizationKernel.cu).

In a file of its own, rather than in cpp_transforms.py with the other kernels, because this
is the step BEFORE the transforms -- it produces the (intensity, weights) pair a chain runs
on -- rather than part of the transform interface. Same reasoning as AssembledChunkReader.py.
"""

import ksgpu

from ..pirate_pybind11 import ChimeDequantizationKernel


@ksgpu.inject_methods(ChimeDequantizationKernel)
class ChimeDequantizationKernelInjections:
    # No class docstring here: ChimeDequantizationKernel's docstring lives in the pybind11
    # binding (option 1 in notes/docstrings.md); this injector only adds default arguments
    # to launch().

    # Save reference to C++ method
    _cpp_launch = ChimeDequantizationKernel.launch

    def launch(self, intensity, weights, scales, offsets, data,
               rfi_mask=None, apply_rfimask=False, scale=1.0, stream=None):
        """GPU kernel launch (async, does not sync stream).

        Parameters
        ----------
        intensity : cupy.ndarray
            Shape ``(nfreq, nt)``, float32, on GPU. Fully overwritten. PARTIALLY CONTIGUOUS:
            the time stride must be 1, but the frequency stride is free (``>= nt``), so this
            may be a column slice of a larger block.
        weights : cupy.ndarray
            Same rules as ``intensity``, and its own frequency stride. Fully overwritten.
            Must not be the same array as ``intensity``.
        scales, offsets : cupy.ndarray
            Shape ``(nfreq_coarse, nt_coarse)``, float32, contiguous, on GPU. Read only.
        data : cupy.ndarray
            Shape ``(nfreq, nt)``, uint8, contiguous, on GPU. Read only.
        rfi_mask : cupy.ndarray or None, optional
            Shape ``(nrfifreq, nt/8)``, uint8, contiguous, on GPU, bit-packed LSB-first with
            a SET bit meaning GOOD data. ``nrfifreq`` must divide ``nfreq``. Ignored when
            ``apply_rfimask`` is False, which is what None means here.
        apply_rfimask : bool, optional
            If True, both outputs are +0.0 wherever the mask marks the sample bad. Defaults
            to False, so that a caller who passes no mask gets the unmasked decode rather
            than an error.
        scale : float, optional
            Multiplies the scales and offsets before the decode, the way the CHIME L1
            server's ``intensity_prescale`` (1e-4 in production) did; see the class
            docstring. Defaults to 1.
        stream : cupy.cuda.Stream or None, optional
            CUDA stream to use. If None, uses current cupy stream.
        """
        import cupy as cp

        if stream is None:
            stream = cp.cuda.get_current_stream()
        if rfi_mask is None:
            # The C++ side ignores the mask when apply_rfimask is False, but it still has to
            # BE an array: the ksgpu type caster has no conversion for None.
            rfi_mask = cp.empty((0, 0), dtype=cp.uint8)

        self._cpp_launch(intensity, weights, scales, offsets, data, rfi_mask,
                         apply_rfimask, float(scale), stream.ptr)
