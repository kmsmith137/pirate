"""GpuTransformBase: the base class of every chimefrb transform, and the file to read before
writing one in python. See the class docstring; ExampleCupyTransform.py is a complete example.

The class itself is C++ (include/pirate/chimefrb/TransformBase.hpp), bound with pybind11. This
module adds its python-side methods with ksgpu.inject_methods, and re-exports it.
"""

import ksgpu

from ..pirate_pybind11 import GpuTransformBase


@ksgpu.inject_methods(GpuTransformBase)
class GpuTransformBaseInjections:
    """Base class of every chimefrb transform: anything a :class:`WiPipeline` or
    :class:`RfiMaskPipeline` can run.

    A transform processes one (nbeams, nfreq, ntime) block of intensity and weights, in place,
    on the GPU. The five ported RFI transforms (GpuBadChannelMask, GpuIntensityClipper,
    GpuStdDevClipper, GpuPolynomialDetrender, GpuSplineDetrender) are C++ subclasses; the two
    pipeline classes and :class:`ExampleCupyTransform` are python subclasses; yours can be
    either. To write one in python::

        class MyTransform(GpuTransformBase):
            def __init__(self, nbeams, nfreq, ntime, sigma=3.0):
                super().__init__(nbeams, nfreq, ntime)      # scratch_nelts=... if you need scratch
                self.sigma = float(sigma)

            def launch_checked(self, intensity, weights, scratch):
                ...   # cupy code, in place on 'intensity' and 'weights'

            def to_yaml_dict(self):
                return {'class_name': 'MyTransform', 'sigma': self.sigma}

            @classmethod
            def from_yaml_dict(cls, d, nbeams, nfreq, ntime):
                check_yaml_keys(d, 'MyTransform', ['sigma'])
                return cls(nbeams, nfreq, ntime, sigma=d['sigma'])

    :class:`ExampleCupyTransform` is this, complete, in forty lines. Everything python code
    can see of this class is in this file (``pirate_frb/chimefrb/GpuTransformBase.py``). The
    one part written in C++ is the argument checking that ``launch()`` does before calling your
    ``launch_checked()``; what it checks is listed under :meth:`launch_checked`, and a failed
    check raises ``RuntimeError`` with a message that starts with your class's name.

    What ``launch_checked()`` may assume, and must respect:

    - ``intensity`` and ``weights`` are cupy float32 arrays of shape (nbeams, nfreq, ntime),
      C-contiguous, distinct, and to be modified IN PLACE (either, both, or neither). Weights
      are nonnegative, and a zero weight means "ignore this sample". Two footguns: a NaN
      intensity at a zero-weight sample must not poison its row -- select with
      ``cp.where(weights != 0, weights*intensity, 0)``, never multiply -- and a row with no
      weight at all must not divide by zero.
    - The arrays are VIEWS of the caller's arrays (new python objects on the same memory), so
      write through them -- ``weights[...] = 0``, ``intensity += c`` -- and never rebind them.
    - ``scratch`` is a 1-d cupy float32 array of exactly ``scratch_nelts`` elements, garbage
      on entry and on exit. Most transforms should leave ``scratch_nelts`` at 0 and ignore it:
      cupy's memory pool makes ordinary temporaries cheap. The mechanism exists for a
      transform that calls a raw kernel needing a workspace, or that must not allocate; such
      a transform passes its ``scratch_nelts`` to this constructor and carves what it needs
      out of the array it is handed.
    - It runs with the pipeline's CUDA stream made current, so cupy puts its kernels on that
      stream, in order with everything else in the chain. Do not synchronize.

    The yaml methods are the subclass's own, so that what a file contains is visible in the
    subclass rather than assembled by machinery elsewhere. ``class_name`` is the class's
    python name. When a file is read, a class that is not part of ``pirate_frb.chimefrb`` must
    be handed to the reader::

        WiPipeline.read_yaml_file(path, nbeams=1, nfreq=16384, ntime=4096, classes=[MyTransform])

    There is no ``from_json_dict``: the legacy rf_pipelines json describes only the ported
    transforms. ``pirate_frb.chimefrb.transform_io`` states the whole interface.

    Subclassing one of the five C++ transforms in python works for python methods (a
    different ``to_yaml_dict``, say), but a ``launch_checked()`` defined there is NOT called:
    those classes' computation is C++, and ``launch()`` runs the C++ one.

    Attributes (read-only):

    - ``nbeams``, ``nfreq``, ``ntime`` -- the block shape, from the constructor.
    - ``scratch_nelts`` -- float32 scratch elements ``launch()`` needs, from the constructor.
    """

    # Save references to C++ methods
    _cpp_init = GpuTransformBase.__init__
    _cpp_launch = GpuTransformBase.launch

    def __init__(self, nbeams, nfreq, ntime, scratch_nelts=0):
        """Create the base of a transform. A subclass's ``__init__`` calls this first.

        Parameters
        ----------
        nbeams, nfreq, ntime : int
            The block shape ``launch()`` will be given; each >= 1.
        scratch_nelts : int, optional
            Float32 scratch elements ``launch_checked()`` needs; 0 for most transforms.

        Raises
        ------
        RuntimeError
            On a shape value < 1 or a negative ``scratch_nelts``.
        """
        # The C++ constructor takes the class name too, for its messages.
        self._cpp_init(type(self).__name__, nbeams, nfreq, ntime, scratch_nelts)

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

    def launch_checked(self, intensity, weights, scratch):
        """The computation, defined by the subclass. Called by :meth:`launch` after the
        arguments have been validated, with the pipeline's stream made current and
        ``scratch`` cut to exactly ``scratch_nelts`` elements. See the class docstring for the
        contract."""
        raise NotImplementedError(f'{type(self).__name__} must define launch_checked(); see GpuTransformBase')

    def _dispatch_launch_checked(self, intensity, weights, scratch, stream_ptr):
        # Called from C++ -- GpuTransformBase::launch(), through the pybind11 trampoline in
        # src_pybind11/pirate_pybind11_chimefrb.cpp -- when the transform's launch_checked()
        # is written in python. The arguments have already been checked, and 'scratch' is
        # None when scratch_nelts is 0. Making the stream current is what lets cupy code in
        # launch_checked() run in order with the rest of the chain.
        import cupy as cp

        if scratch is None:
            scratch = cp.empty(0, dtype=cp.float32)

        with cp.cuda.ExternalStream(stream_ptr):
            self.launch_checked(intensity, weights, scratch)

    def to_yaml_dict(self):
        """The transform's yaml form: ``{'class_name': <python class name>, **parameters}``.
        Defined by the subclass (see the class docstring)."""
        raise NotImplementedError(f'{type(self).__name__} must define to_yaml_dict(); see GpuTransformBase')

    @classmethod
    def from_yaml_dict(cls, d, nbeams, nfreq, ntime):
        """The inverse of :meth:`to_yaml_dict`, at the given geometry. Defined by the subclass
        (see the class docstring)."""
        raise NotImplementedError(f'{cls.__name__} must define from_yaml_dict(); see GpuTransformBase')

    def __repr__(self):
        return f'{type(self).__name__}(nbeams={self.nbeams}, nfreq={self.nfreq}, ntime={self.ntime})'
