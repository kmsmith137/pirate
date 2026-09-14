"""GpuPythonTransform: the base class of every chimefrb transform written in python, and the
file to read before writing one. See the class docstring; ExamplePythonTransform.py is a
complete example.

Its own base class, GpuTransform, is C++ (bound with pybind11): it owns the array
geometry and the checked launch(), and its python-side methods are in cpp_transforms.py.
This class is plain python. Everything a python transform inherits that is specific to
being written in python -- the constructor, the methods a subclass defines, and the hook
the C++ side calls -- is in this file.
"""

from .cpp_transforms import GpuTransform


class GpuPythonTransform(GpuTransform):
    """Base class of a chimefrb transform written in python.

    A transform processes one (nbeams, nfreq, ntime) block of intensity and weights, in place,
    on the GPU, and is anything a :class:`Pipeline` or :class:`RfiMaskPipeline` can run. The
    five ported RFI transforms (GpuBadChannelMask, GpuIntensityClipper, GpuStdDevClipper,
    GpuPolynomialDetrender, GpuSplineDetrender) are C++, on :class:`GpuTransform` directly;
    the two pipeline classes, :class:`ExamplePythonTransform`, and anything you write in python
    are subclasses of this class. To write one::

        class MyTransform(GpuPythonTransform):
            def __init__(self, nbeams, nfreq, ntime, sigma=3.0):
                super().__init__(nbeams, nfreq, ntime)      # scratch_nelts=... if you need scratch
                self.sigma = float(sigma)

            def launch_checked(self, intensity, weights, scratch):
                ...   # cupy code, in place on 'intensity' and 'weights'

            def to_yaml_dict(self):
                return {'class_name': 'MyTransform', 'sigma': self.sigma}

            @classmethod
            def from_yaml_dict(cls, d, nbeams, nfreq, ntime):
                cls.check_yaml_keys(d, ['sigma'])
                return cls(nbeams, nfreq, ntime, sigma=d['sigma'])

    :class:`ExamplePythonTransform` is this, complete, in forty lines. This class is plain
    python, and this file is all of it. What it inherits from GpuTransform is python too
    (``launch()``, ``check_yaml_keys()`` and ``__repr__``, in
    ``pirate_frb/chimefrb/cpp_transforms.py``), except the argument checking that
    ``launch()`` does before calling your ``launch_checked()``, which is C++; what it checks is
    listed under :meth:`launch_checked`, and a failed check raises ``RuntimeError`` with a
    message that starts with your class's name.

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

        Pipeline.read_yaml_file(path, nbeams=1, nfreq=16384, ntime=4096, classes=[MyTransform])

    There is no ``from_json_dict`` unless the transform has a legacy counterpart, which only
    :class:`RfiMaskExtractor` (the old ``mask_counter``) does. ``pirate_frb.chimefrb.utils``
    states the whole interface.

    Three things to know about the class hierarchy. A pipeline accepts any
    :class:`GpuTransform`, C++ or python, so the ``isinstance`` checks in the package test
    that class, not this one. Subclassing one of the five C++ transforms in python works for
    python methods (a different ``to_yaml_dict``, say), but a ``launch_checked()`` defined
    there is NOT called: those classes' computation is C++, and ``launch()`` runs the C++ one.
    And a subclass that forgets ``super().__init__()`` gets pybind11's ``TypeError:
    ...GpuTransform.__init__() must be called when overriding __init__``, which names the
    C++ base rather than this class.

    Attributes (read-only):

    - ``nbeams``, ``nfreq``, ``ntime`` -- the block shape, from the constructor.
    - ``scratch_nelts`` -- float32 scratch elements ``launch()`` needs, from the constructor.
    """

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
        # The C++ base class also takes the class name, which prefixes its messages.
        super().__init__(type(self).__name__, nbeams, nfreq, ntime, scratch_nelts)

    def launch_checked(self, intensity, weights, scratch):
        """The computation, defined by the subclass. Called by :meth:`launch` after the
        arguments have been validated, with the pipeline's stream made current and
        ``scratch`` cut to exactly ``scratch_nelts`` elements. See the class docstring for the
        contract."""
        raise NotImplementedError(f'{type(self).__name__} must define launch_checked(); see GpuPythonTransform')

    def _dispatch_launch_checked(self, intensity, weights, scratch, stream_ptr):
        # Called from C++ -- GpuTransform::launch(), through the pybind11 trampoline in
        # src_pybind11/pirate_pybind11_chimefrb.cpp -- after the arguments have been checked.
        # 'scratch' is None when scratch_nelts is 0. Making the stream current is what lets
        # cupy code in launch_checked() run in order with the rest of the chain.
        import cupy as cp

        if scratch is None:
            scratch = cp.empty(0, dtype=cp.float32)

        with cp.cuda.ExternalStream(stream_ptr):
            self.launch_checked(intensity, weights, scratch)

    def to_yaml_dict(self):
        """The transform's yaml form: ``{'class_name': <python class name>, **parameters}``.
        Defined by the subclass (see the class docstring)."""
        raise NotImplementedError(f'{type(self).__name__} must define to_yaml_dict(); see GpuPythonTransform')

    @classmethod
    def from_yaml_dict(cls, d, nbeams, nfreq, ntime):
        """The inverse of :meth:`to_yaml_dict`, at the given geometry. Defined by the subclass
        (see the class docstring)."""
        raise NotImplementedError(f'{cls.__name__} must define from_yaml_dict(); see GpuPythonTransform')
