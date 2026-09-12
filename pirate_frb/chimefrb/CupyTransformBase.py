"""CupyTransformBase: the base class for a chimefrb transform written in cupy, without C++.

See the class docstring for what a subclass supplies; ExampleCupyTransform.py is a worked
example, and pirate_frb/chimefrb/transform_io.py states the interface every transform
follows.
"""

import numpy as np

from .transform_io import check_launch_args, default_scratch_and_stream


class CupyTransformBase:
    """Base class for a transform written in cupy.

    A transform is anything a :class:`WiPipeline` or :class:`RfiMaskPipeline` can run: it
    processes one (nbeams, nfreq, ntime) block of intensity and weights, in place, on the
    GPU. This class supplies the geometry attributes and a checked ``launch()``; a subclass
    supplies the computation, and its own yaml methods::

        class MyTransform(CupyTransformBase):
            def __init__(self, nbeams, nfreq, ntime, sigma=3.0):
                super().__init__(nbeams, nfreq, ntime)
                self.sigma = float(sigma)

            def launch_checked(self, intensity, weights, scratch):
                ...   # cupy code, in place on 'intensity' and 'weights'

            def to_yaml_dict(self):
                return {'class_name': 'MyTransform', 'sigma': self.sigma}

            @classmethod
            def from_yaml_dict(cls, d, nbeams, nfreq, ntime):
                check_yaml_keys(d, 'MyTransform', ['sigma'])
                return cls(nbeams, nfreq, ntime, sigma=d['sigma'])

    What ``launch_checked()`` may assume, and must respect:

    - ``intensity`` and ``weights`` are cupy float32 arrays of shape (nbeams, nfreq, ntime),
      C-contiguous, distinct, and to be modified IN PLACE (either, both, or neither).
      Weights are nonnegative, and a zero weight means "ignore this sample". Two footguns:
      a NaN intensity at a zero-weight sample must not poison its row -- select with
      ``cp.where(weights != 0, weights*intensity, 0)``, never multiply -- and a row with no
      weight at all must not divide by zero.
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
    transforms.

    Attributes (read-only by convention):

    - ``nbeams``, ``nfreq``, ``ntime`` -- the block shape, from the constructor.
    - ``scratch_nelts`` -- float32 scratch elements ``launch()`` needs, from the constructor.
    """

    def __init__(self, nbeams, nfreq, ntime, scratch_nelts=0):
        for (name, x) in (('nbeams', nbeams), ('nfreq', nfreq), ('ntime', ntime)):
            if not (isinstance(x, (int, np.integer)) and (x >= 1)):
                raise ValueError(f'{type(self).__name__}: expected {name} to be an integer >= 1, got {x!r}')
        if not (isinstance(scratch_nelts, (int, np.integer)) and (scratch_nelts >= 0)):
            raise ValueError(f'{type(self).__name__}: expected scratch_nelts to be an integer >= 0, got {scratch_nelts!r}')

        self.nbeams = int(nbeams)
        self.nfreq = int(nfreq)
        self.ntime = int(ntime)
        self.scratch_nelts = int(scratch_nelts)

    def launch(self, intensity, weights, scratch, stream=None):
        """Run the transform on one block (async; does not sync the stream).

        Parameters
        ----------
        intensity, weights : cupy.ndarray
            Shape (nbeams, nfreq, ntime), float32, C-contiguous, distinct. Modified in place
            as the subclass sees fit.
        scratch : cupy.ndarray or None
            1-d float32 with at least ``scratch_nelts`` elements (any array when that is 0),
            or None to allocate one.
        stream : cupy.cuda.Stream or None, optional
            CUDA stream to use. If None, uses current cupy stream.
        """
        (scratch, stream) = default_scratch_and_stream(scratch, stream, self.scratch_nelts)
        check_launch_args(intensity, weights, scratch, (self.nbeams, self.nfreq, self.ntime),
                          self.scratch_nelts, type(self).__name__)
        with stream:
            self.launch_checked(intensity, weights, scratch[:self.scratch_nelts])

    def launch_checked(self, intensity, weights, scratch):
        """The computation. Called by :meth:`launch` after the arguments have been validated,
        the pipeline's stream made current, and ``scratch`` cut to exactly ``scratch_nelts``
        elements. See the class docstring for the contract."""
        raise NotImplementedError(f'{type(self).__name__} must define launch_checked()')

    def to_yaml_dict(self):
        """The transform's yaml form: ``{'class_name': <python class name>, **parameters}``.
        Defined by the subclass (see the class docstring)."""
        raise NotImplementedError(f'{type(self).__name__} must define to_yaml_dict()')

    @classmethod
    def from_yaml_dict(cls, d, nbeams, nfreq, ntime):
        """The inverse of :meth:`to_yaml_dict`, at the given geometry. Defined by the subclass
        (see the class docstring)."""
        raise NotImplementedError(f'{cls.__name__} must define from_yaml_dict()')

    def __repr__(self):
        return f'{type(self).__name__}(nbeams={self.nbeams}, nfreq={self.nfreq}, ntime={self.ntime})'
