"""GpuDedisperserOutputs method injections (+ re-export of the pybind11 class)."""

import functools

import ksgpu
from ..pirate_pybind11 import GpuDedisperserOutputs


@ksgpu.inject_methods(GpuDedisperserOutputs)
class GpuDedisperserOutputsInjections:
    """Helper class, representing dedispersion outputs for one beam batch.
    Return value from FrbGrouper.get_output(ichunk, ibatch)

    Example usage::

        with grouper.get_output(ichunk, ibatch) as outputs:
            # 'outputs' has type GpuDedisperserOutputs
            # Loop over dedispersion trees.
            for itree, tree_out in enumerate(outputs.out_max):
                # 'tree_out' has shape (beams_per_batch, coarse_ndm, coarse_ntime)

    Attributes (all read-only):

    - ``ichunk_zero_based`` (int) -- chunk index of this output, relative to the first dedisperser output.
    - ``ichunk_fpga_based`` (int) -- chunk index relative to FPGA seq 0 (= ichunk_zero_based + the producer's initial_chunk).
    - ``ibeam`` (int) -- beam index (NOT beam_id) of the first beam in this output.
    - ``out_max`` (list) -- length-ntrees list of peak-finding maximum-value arrays.
    - ``out_argmax`` (list) -- length-ntrees list of peak-finding argmax-token arrays.
    """
    # This class docstring (above) is the GpuDedisperserOutputs docstring: the
    # pybind11 binding deliberately sets none, and inject_methods copies this one
    # onto the class (option 2 in notes/docstrings.md).
    #
    # Implementation note (deliberately NOT in the docstring -- it's a low-level
    # detail): this injector wraps the C++ out_max/out_argmax members (exposed by
    # the binding under the underscore names _out_max/_out_argmax) in
    # cached_property accessors, so the vector<Array> -> list-of-arrays conversion
    # runs ONCE per Outputs object instead of on every attribute read. Repeated
    # access -- e.g. 'outputs.out_max[itree]' in a for-loop -- was previously
    # O(ntrees) full conversions per access (O(ntrees^2) over the loop); it is now
    # a cheap list index after the first access.
    #
    # WHY THIS CACHE IS SAFE -- and the invariant it depends on:
    #
    # GpuDedisperser.acquire_output() and FrbGrouper.acquire_output() return a
    # fresh Outputs BY VALUE, so pybind creates a NEW Python object (with a fresh,
    # empty __dict__) for every get_output() call. The cache therefore lives and
    # dies with a single batch and can never serve a later batch's arrays.
    #
    # DANGER: this correctness rests ENTIRELY on the by-value return. If
    # acquire_output ever returns a reference/pointer into a persistent Outputs
    # (e.g. GpuDedisperser::output_ringbuf), pybind would hand back the SAME
    # Python object across batches, and this cache would then return stale views
    # into recycled GPU ring-buffer memory -- a silent wrong-answer bug. The C++
    # declarations (Dedisperser.hpp, FrbGrouper.hpp) carry a matching warning; do
    # not switch them to return by reference without removing this cache.

    @functools.cached_property
    def out_max(self):
        """Length-ntrees list of peak-finding maximum-value arrays. Valid only
        inside the get_output() context manager (see the class docstring)."""
        return self._out_max

    @functools.cached_property
    def out_argmax(self):
        """Length-ntrees list of peak-finding argmax-token arrays. Valid only
        inside the get_output() context manager (see the class docstring)."""
        return self._out_argmax
