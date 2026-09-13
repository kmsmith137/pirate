"""GpuTransform: the base class of every chimefrb transform, and its python side.

The class itself is C++ (include/pirate/chimefrb/Transform.hpp), bound with pybind11. This
module adds, with ksgpu.inject_methods, the python-side methods that every transform, C++ or
python, shares -- launch() with its stream=None / scratch=None conventions, the
check_yaml_keys() classmethod, and __repr__ -- and re-exports the class. A transform written
in python subclasses GpuPythonTransform (GpuPythonTransform.py), not this class directly.
"""

import ksgpu

from ..pirate_pybind11 import GpuTransform


@ksgpu.inject_methods(GpuTransform)
class GpuTransformInjections:
    """Base class of every chimefrb transform: anything a :class:`Pipeline` or
    :class:`RfiMaskPipeline` can run.

    A transform processes one (nbeams, nfreq, ntime) block of intensity and weights, in place,
    on the GPU, through :meth:`launch`, which checks its arguments and then runs the
    transform's computation. Which of the two arrays a transform modifies is stated in its
    class docstring.

    The class is C++ (``include/pirate/chimefrb/Transform.hpp``), and its direct subclasses
    are the five ported RFI transforms (GpuBadChannelMask, GpuIntensityClipper,
    GpuStdDevClipper, GpuPolynomialDetrender, GpuSplineDetrender) and
    :class:`GpuPythonTransform`. To write a transform in python, subclass GpuPythonTransform,
    not this class; its docstring is the how-to.

    Attributes (read-only):

    - ``nbeams``, ``nfreq``, ``ntime`` -- the block shape, from the constructor.
    - ``scratch_nelts`` -- float32 scratch elements ``launch()`` needs, from the constructor.
    """

    # Save reference to C++ method
    _cpp_launch = GpuTransform.launch

    def launch(self, intensity, weights, scratch, stream=None):
        """Run the transform on one block (async; does not sync the stream).

        Checks the arguments (see the class docstring for what must hold), then runs the
        transform's ``launch_checked()`` on ``stream``. Which of the two arrays a transform
        modifies is stated in its class docstring.

        Parameters
        ----------
        intensity, weights : cupy.ndarray
            Shape (nbeams, nfreq, ntime), float32, C-contiguous, distinct. Modified in place
            as the transform sees fit.
        scratch : cupy.ndarray or None
            1-d float32 with at least ``scratch_nelts`` elements (any array when that is 0),
            or None to allocate one -- convenient interactively, wasteful in a loop, since the
            point of the argument is to share one allocation across a whole chain.
        stream : cupy.cuda.Stream or None, optional
            CUDA stream to use. If None, uses current cupy stream.

        Raises
        ------
        RuntimeError
            If an argument has the wrong shape, is not C-contiguous, is not on the GPU, or
            aliases another; or if ``scratch`` is too small.
        TypeError
            If an argument is not a cupy array, or not float32.
        """
        import cupy as cp

        if stream is None:
            stream = cp.cuda.get_current_stream()
        if scratch is None:
            scratch = cp.empty(self.scratch_nelts, dtype=cp.float32)

        self._cpp_launch(intensity, weights, scratch, stream.ptr)

    @classmethod
    def check_yaml_keys(cls, d, keys):
        """The check every ``from_yaml_dict`` starts with: ``d`` is a dict whose ``class_name``
        is this class's name, and whose other keys are exactly ``keys`` -- a missing key and
        an unexpected key are both errors, with a message naming them."""
        name = cls.__name__

        if not isinstance(d, dict):
            raise ValueError(f"{name}.from_yaml_dict: expected a dict, got {type(d).__name__}")
        if d.get('class_name') != name:
            raise ValueError(f"{name}.from_yaml_dict: expected class_name {name!r},"
                             f" got {d.get('class_name')!r}")

        got = set(d) - {'class_name'}
        want = set(keys)
        if got != want:
            parts = []
            if want - got:
                parts.append(f"missing key(s) {sorted(want - got)}")
            if got - want:
                parts.append(f"unexpected key(s) {sorted(got - want)}")
            raise ValueError(f"{name}.from_yaml_dict: " + ", ".join(parts))

    def __repr__(self):
        return f'{type(self).__name__}(nbeams={self.nbeams}, nfreq={self.nfreq}, ntime={self.ntime})'
