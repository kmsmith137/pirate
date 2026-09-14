"""The python side of every pybind11-bound class in pirate_frb.chimefrb, in one file.

The classes themselves are C++ (include/pirate/chimefrb/*.hpp), bound with pybind11 in
src_pybind11/pirate_pybind11_chimefrb.cpp. This module adds the python-side methods each one
needs, with ksgpu.inject_methods, and re-exports them. Three kinds of injection appear below,
one section each: the python side shared by EVERY transform (GpuTransform), a stream=None
default on the three compute kernels' launch(), and the yaml / legacy-json methods of the five
ported RFI transforms.

The injections are small and all follow one pattern, so they live together here rather than
one per class. The numpy reference for each class -- which is the interesting code, and much
longer -- stays in its own Reference<ClassName>.py.

pirate_frb/chimefrb/__init__.py imports this module before anything else in the subpackage, so
the injections are in place whichever submodule a caller imports first (importing any submodule
runs __init__.py first). Import these classes from pirate_frb.chimefrb, or from here -- never
from pirate_frb.pirate_pybind11, which hands back the same class objects but does not guarantee
that the injections have run.

Not every bound class in the subpackage is here. AssembledChunk and GpuClipperBase have no
injections at all, and __init__.py imports them straight from pirate_frb.pirate_pybind11;
AssembledChunkReader has its own file, since it is the step before the transforms rather
than part of the transform interface.
"""

import ksgpu

from ..pirate_pybind11 import (GpuBadChannelMask, GpuIntensityClipper, GpuPolynomialDetrender,
                               GpuSplineDetrender, GpuStdDevClipper, GpuTransform,
                               GpuWiDownsamplingKernel, GpuWrmsKernel, GpuWtUpsamplingKernel)

# Note the two different 'utils': '..utils' is pirate_frb/utils.py (the package-level one),
# and '.utils' is pirate_frb/chimefrb/utils.py (this subpackage's).
from ..utils import atomic_print
from .utils import CHIME_FREQ_RANGE, axis_from_json, check_json_keys


# -------------------------------------------------------------------------------------------------
#
# GpuTransform: the base class of every chimefrb transform, C++ or python. Everything it
# gains here is inherited by every class further down.


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
            point of the argument is to share one allocation across a whole chain. The caller
            is responsible for 128-byte alignment, which is not checked (a cupy allocation
            always is): a transform's internal sub-arrays are aligned relative to the base.
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


# -------------------------------------------------------------------------------------------------
#
# The three compute kernels. Unlike everything else in this file these are NOT transforms:
# they do not subclass GpuTransform and a Pipeline cannot run them. They are the pieces the
# clippers are built from, exposed to python because the unit tests rebuild a clipper out of
# them and compare. Each injection does one thing -- give launch() a stream=None default
# meaning "the current cupy stream", since the C++ signature takes a raw stream pointer.


@ksgpu.inject_methods(GpuWiDownsamplingKernel)
class GpuWiDownsamplingKernelInjections:
    # No class docstring here: GpuWiDownsamplingKernel's docstring lives in the pybind11
    # binding (option 1 in notes/docstrings.md); this injector adds a stream
    # argument for launch().

    # Save reference to C++ method
    _cpp_launch = GpuWiDownsamplingKernel.launch

    def launch(self, out_i, out_w, in_i, in_w, stream=None):
        """GPU kernel launch (async, does not sync stream).

        Parameters
        ----------
        out_i, out_w : cupy.ndarray
            Shape (B, F//Df, T//Dt), or (B, T//Dt, F//Df) if ``transpose``.
            Float32, fully contiguous, on GPU. Fully overwritten.
        in_i, in_w : cupy.ndarray
            Shape (B, F, T), float32, fully contiguous, on GPU. Read only, and
            must not alias the output arrays. ``in_w`` must be >= 0.
        stream : cupy.cuda.Stream or None, optional
            CUDA stream to use. If None, uses current cupy stream.
        """
        import cupy as cp

        if stream is None:
            stream = cp.cuda.get_current_stream()

        self._cpp_launch(out_i, out_w, in_i, in_w, stream.ptr)


@ksgpu.inject_methods(GpuWrmsKernel)
class GpuWrmsKernelInjections:
    # No class docstring here: GpuWrmsKernel's docstring lives in the pybind11 binding
    # (option 1 in notes/docstrings.md); this injector adds a stream argument for
    # launch(), and lets the caller omit the scratch array.

    # Save reference to C++ method
    _cpp_launch = GpuWrmsKernel.launch

    def launch(self, mean, var, in_i, in_w, scratch=None, stream=None):
        """GPU kernel launch (async, does not sync stream).

        Parameters
        ----------
        mean, var : cupy.ndarray
            Shape (R,), float32, contiguous, on GPU. Fully overwritten.
        in_i, in_w : cupy.ndarray
            Shape (R, L), float32, fully contiguous, on GPU. Read only, and must not
            alias the outputs. ``in_w`` must be >= 0.
        scratch : cupy.ndarray or None, optional
            Shape ``(self.scratch_nelts(R),)``, float32, on GPU. If None, one is
            allocated here -- convenient for tests, wasteful in a loop, since the
            whole point of the argument is to reuse one allocation across chunks.
        stream : cupy.cuda.Stream or None, optional
            CUDA stream to use. If None, uses current cupy stream.
        """
        import cupy as cp

        if stream is None:
            stream = cp.cuda.get_current_stream()
        if scratch is None:
            scratch = cp.empty(self.scratch_nelts(in_i.shape[0]), dtype=cp.float32)

        self._cpp_launch(mean, var, in_i, in_w, scratch, stream.ptr)


@ksgpu.inject_methods(GpuWtUpsamplingKernel)
class GpuWtUpsamplingKernelInjections:
    # No class docstring here: GpuWtUpsamplingKernel's docstring lives in the pybind11 binding
    # (option 1 in notes/docstrings.md); this injector adds a stream argument for launch().

    # Save reference to C++ method
    _cpp_launch = GpuWtUpsamplingKernel.launch

    def launch(self, w_hires, w_lores, stream=None):
        """GPU kernel launch (async, does not sync stream).

        Note the order: the array that is modified comes first, as in the old code. At
        ``(Df, Dt) = (1, 1)`` the two shapes agree, so a swapped call is not caught.

        Parameters
        ----------
        w_hires : cupy.ndarray
            Shape ``(B, F_lo*Df, T_lo*Dt)``, float32, fully contiguous, on GPU. MODIFIED IN
            PLACE: every weight in a masked cell becomes +0.0, and every other weight is
            left bit-identical. Never read.
        w_lores : cupy.ndarray
            Shape ``(B, F_lo, T_lo)``, float32, fully contiguous, on GPU, with any ``B``,
            ``F_lo`` and ``T_lo``. Read only. Must not be the same array as ``w_hires``.
        stream : cupy.cuda.Stream or None, optional
            CUDA stream to use. If None, uses current cupy stream.
        """
        import cupy as cp

        if stream is None:
            stream = cp.cuda.get_current_stream()

        self._cpp_launch(w_hires, w_lores, stream.ptr)


# -------------------------------------------------------------------------------------------------
#
# The five ported RFI transforms. launch() is inherited from GpuTransform; what each injection
# adds is the yaml pair every transform has (to_yaml_dict / from_yaml_dict) plus, since all
# five have a legacy rf_pipelines form, from_json_dict(). utils.py states the interface those
# methods follow. GpuBadChannelMask also wraps its constructor, to normalize the MHz ranges.


# The yaml keys of GpuBadChannelMask, which are also its constructor's argument names after
# the geometry.
BADCHANNEL_MASK_YAML_KEYS = ('mask_ranges', 'freq_range')


def _as_range_list(mask_ranges):
    """A list of (float, float) pairs, from any sequence of pairs (a numpy (n, 2) array
    included), with a clear error for anything else."""
    ranges = []
    for r in mask_ranges:
        r = tuple(r)
        if len(r) != 2:
            raise ValueError(f'GpuBadChannelMask: expected each mask range to be a (lo, hi) pair, got {r!r}')
        ranges.append((float(r[0]), float(r[1])))
    return ranges


@ksgpu.inject_methods(GpuBadChannelMask)
class GpuBadChannelMaskInjections:
    # No class docstring here: GpuBadChannelMask's docstring lives in the pybind11 binding
    # (option 1 in notes/docstrings.md). launch() is inherited from GpuTransform; this
    # injector normalizes the constructor's range arguments, and adds the yaml and
    # legacy-json methods (utils.py).

    # Save references to C++ methods
    _cpp_init = GpuBadChannelMask.__init__

    def __init__(self, nbeams, nfreq, ntime, mask_ranges, freq_range, warps_per_block=4):
        """Create a GpuBadChannelMask.

        Parameters
        ----------
        nbeams, nfreq, ntime : int
            The array shape launch() will be given.
        mask_ranges : sequence of (lo, hi) pairs
            Frequency ranges to mask, in MHz, each with lo < hi, in any order. A numpy array
            of shape (n, 2) is fine.
        freq_range : (lo, hi)
            The band in MHz, channel 0 at the top; (400, 800) for CHIME.
        warps_per_block : int, optional
            Performance knob, 4, 8, 16 or 32, which must not change the result. See
            :meth:`time_selected`.
        """
        band = _as_range_list([freq_range])[0]
        self._cpp_init(int(nbeams), int(nfreq), int(ntime), _as_range_list(mask_ranges), band,
                       int(warps_per_block))

    def to_yaml_dict(self):
        """The yaml form (see ``chimefrb.utils``): the class name, ``freq_range`` and
        ``mask_ranges``, in MHz as given to the constructor."""
        return {'class_name': 'GpuBadChannelMask',
                'freq_range': [float(self.freq_range[0]), float(self.freq_range[1])],
                'mask_ranges': [[float(lo), float(hi)] for (lo, hi) in self.mask_ranges]}

    @classmethod
    def from_yaml_dict(cls, d, nbeams, nfreq, ntime):
        """The inverse of :meth:`to_yaml_dict`, at the given geometry."""
        cls.check_yaml_keys(d, BADCHANNEL_MASK_YAML_KEYS)
        return cls(nbeams, nfreq, ntime, d['mask_ranges'], d['freq_range'])

    @classmethod
    def from_json_dict(cls, d, nbeams, nfreq, ntime):
        """From the legacy rf_pipelines json element (``class_name: badchannel_mask``).

        The legacy json carries the MHz ranges but NOT the band, which rf_pipelines read
        from the stream at bind time; the CHIME band (400, 800) is assumed, and a line saying
        so is printed to stderr. A nonempty ``mask_path`` (a file of extra ranges) is not
        supported.
        """
        check_json_keys(d, 'badchannel_mask', ['mask_ranges', 'mask_path'])
        if d['mask_path']:
            raise NotImplementedError(f"GpuBadChannelMask.from_json_dict: mask_path={d['mask_path']!r}"
                                      f" (a file of extra ranges) is not supported; only 'mask_ranges' is")
        atomic_print(f'GpuBadChannelMask.from_json_dict: assuming freq_range = {CHIME_FREQ_RANGE} MHz'
                     f' (the legacy json does not record the band; rf_pipelines took it from the stream)',
                     fd=2)   # stderr, so that a converter writing yaml to stdout stays clean
        return cls(nbeams, nfreq, ntime, d['mask_ranges'], CHIME_FREQ_RANGE)


# The yaml keys of GpuIntensityClipper, which are also its constructor's argument names after
# the geometry.
INTENSITY_CLIPPER_YAML_KEYS = ('nt_chunk', 'axis', 'sigma', 'niter', 'iter_sigma', 'Df', 'Dt', 'two_pass')


@ksgpu.inject_methods(GpuIntensityClipper)
class GpuIntensityClipperInjections:
    # No class docstring here: GpuIntensityClipper's docstring lives in the pybind11
    # binding (option 1 in notes/docstrings.md). launch() is inherited from GpuTransform;
    # this injector adds the yaml and legacy-json methods (utils.py).

    def to_yaml_dict(self):
        """The yaml form (see ``chimefrb.utils``): the class name and the semantic parameters
        (``nt_chunk``, ``axis`` as 'freq'/'time'/'none', ``sigma``, ``niter``,
        ``iter_sigma``, ``Df``, ``Dt``, ``two_pass``)."""
        return {'class_name': 'GpuIntensityClipper', 'nt_chunk': int(self.nt_chunk),
                'axis': self.axis, 'sigma': float(self.sigma),
                'niter': int(self.niter), 'iter_sigma': float(self.iter_sigma),
                'Df': int(self.Df), 'Dt': int(self.Dt), 'two_pass': bool(self.two_pass)}

    @classmethod
    def from_yaml_dict(cls, d, nbeams, nfreq, ntime):
        """The inverse of :meth:`to_yaml_dict`, at the given geometry."""
        cls.check_yaml_keys(d, INTENSITY_CLIPPER_YAML_KEYS)
        return cls(nbeams, nfreq, ntime, d['nt_chunk'], d['axis'], d['sigma'],
                   d['Df'], d['Dt'], d['niter'], d['iter_sigma'], d['two_pass'])

    @classmethod
    def from_json_dict(cls, d, nbeams, nfreq, ntime):
        """From the legacy rf_pipelines json element (``class_name: intensity_clipper``),
        applying two of its conventions: ``nt_chunk == 0`` means the whole block (here
        ``ntime``), and ``iter_sigma == 0`` means ``sigma``."""
        check_json_keys(d, 'intensity_clipper',
                        ['axis', 'sigma', 'niter', 'iter_sigma', 'Df', 'Dt', 'two_pass', 'nt_chunk'])
        nt_chunk = d['nt_chunk'] if d['nt_chunk'] else ntime
        iter_sigma = d['iter_sigma'] if d['iter_sigma'] else d['sigma']
        return cls(nbeams, nfreq, ntime, nt_chunk, axis_from_json(d['axis']), d['sigma'],
                   d['Df'], d['Dt'], d['niter'], iter_sigma, d['two_pass'])


# The yaml keys of GpuStdDevClipper, which are also its constructor's argument names after
# the geometry.
STD_DEV_CLIPPER_YAML_KEYS = ('nt_chunk', 'axis', 'sigma', 'Df', 'Dt', 'two_pass')


@ksgpu.inject_methods(GpuStdDevClipper)
class GpuStdDevClipperInjections:
    # No class docstring here: GpuStdDevClipper's docstring lives in the pybind11 binding
    # (option 1 in notes/docstrings.md). launch() is inherited from GpuTransform; this
    # injector adds the yaml and legacy-json methods (utils.py).

    def to_yaml_dict(self):
        """The yaml form (see ``chimefrb.utils``): the class name and the semantic parameters
        (``nt_chunk``, ``axis`` as 'freq'/'time', ``sigma``, ``Df``, ``Dt``, ``two_pass``)."""
        return {'class_name': 'GpuStdDevClipper', 'nt_chunk': int(self.nt_chunk),
                'axis': self.axis, 'sigma': float(self.sigma),
                'Df': int(self.Df), 'Dt': int(self.Dt), 'two_pass': bool(self.two_pass)}

    @classmethod
    def from_yaml_dict(cls, d, nbeams, nfreq, ntime):
        """The inverse of :meth:`to_yaml_dict`, at the given geometry."""
        cls.check_yaml_keys(d, STD_DEV_CLIPPER_YAML_KEYS)
        return cls(nbeams, nfreq, ntime, d['nt_chunk'], d['axis'], d['sigma'],
                   d['Df'], d['Dt'], d['two_pass'])

    @classmethod
    def from_json_dict(cls, d, nbeams, nfreq, ntime):
        """From the legacy rf_pipelines json element (``class_name: std_dev_clipper``);
        ``nt_chunk == 0`` means the whole block (here ``ntime``)."""
        check_json_keys(d, 'std_dev_clipper', ['axis', 'sigma', 'Df', 'Dt', 'two_pass', 'nt_chunk'])
        nt_chunk = d['nt_chunk'] if d['nt_chunk'] else ntime
        return cls(nbeams, nfreq, ntime, nt_chunk, axis_from_json(d['axis']), d['sigma'],
                   d['Df'], d['Dt'], d['two_pass'])


# The yaml keys of GpuPolynomialDetrender: its constructor's argument names after the
# geometry, plus 'axis', which the old json carries and which must be 'time' (the only axis
# this class implements; ReferencePolynomialDetrender also does 'freq').
POLYNOMIAL_DETRENDER_YAML_KEYS = ('polydeg', 'epsilon', 'nt_chunk', 'axis')


@ksgpu.inject_methods(GpuPolynomialDetrender)
class GpuPolynomialDetrenderInjections:
    # No class docstring here: GpuPolynomialDetrender's docstring lives in the pybind11
    # binding (option 1 in notes/docstrings.md). launch() is inherited from GpuTransform;
    # this injector adds the yaml and legacy-json methods (utils.py).

    def to_yaml_dict(self):
        """The yaml form (see ``chimefrb.utils``): the class name, ``polydeg``, ``epsilon``,
        ``nt_chunk``, and ``axis: time``."""
        return {'class_name': 'GpuPolynomialDetrender', 'polydeg': int(self.polydeg),
                'epsilon': float(self.epsilon), 'nt_chunk': int(self.nt_chunk), 'axis': 'time'}

    @classmethod
    def from_yaml_dict(cls, d, nbeams, nfreq, ntime):
        """The inverse of :meth:`to_yaml_dict`, at the given geometry."""
        cls.check_yaml_keys(d, POLYNOMIAL_DETRENDER_YAML_KEYS)
        if d['axis'] != 'time':
            raise ValueError(f"GpuPolynomialDetrender.from_yaml_dict: axis={d['axis']!r}, but this class"
                             f" implements axis 'time' only (ReferencePolynomialDetrender also does 'freq')")
        return cls(nbeams, nfreq, ntime, d['polydeg'], d['epsilon'], d['nt_chunk'])

    @classmethod
    def from_json_dict(cls, d, nbeams, nfreq, ntime):
        """From the legacy rf_pipelines json element (``class_name: polynomial_detrender``),
        whose ``polydeg`` is written as a double and whose ``nt_chunk == 0`` means the whole
        block (here ``ntime``)."""
        check_json_keys(d, 'polynomial_detrender', ['axis', 'polydeg', 'epsilon', 'nt_chunk'])
        if axis_from_json(d['axis']) != 'time':
            raise ValueError(f"GpuPolynomialDetrender.from_json_dict: axis={d['axis']!r}, but this class"
                             f" implements axis 'time' only (ReferencePolynomialDetrender also does 'freq')")
        polydeg = d['polydeg']
        if int(polydeg) != polydeg:
            raise ValueError(f"GpuPolynomialDetrender.from_json_dict: polydeg={polydeg!r} is not an integer")
        nt_chunk = d['nt_chunk'] if d['nt_chunk'] else ntime
        return cls(nbeams, nfreq, ntime, int(polydeg), d['epsilon'], nt_chunk)


# The yaml keys of GpuSplineDetrender: its constructor's argument names after the geometry,
# plus 'axis', which the old json carries and which must be 'freq' (the only axis this class
# implements).
SPLINE_DETRENDER_YAML_KEYS = ('nbins', 'epsilon', 'axis')


@ksgpu.inject_methods(GpuSplineDetrender)
class GpuSplineDetrenderInjections:
    # No class docstring here: GpuSplineDetrender's docstring lives in the pybind11
    # binding (option 1 in notes/docstrings.md). launch() is inherited from GpuTransform;
    # this injector adds the yaml and legacy-json methods (utils.py).

    def to_yaml_dict(self):
        """The yaml form (see ``chimefrb.utils``): the class name, ``nbins``, ``epsilon``, and
        ``axis: freq``."""
        return {'class_name': 'GpuSplineDetrender', 'nbins': int(self.nbins),
                'epsilon': float(self.epsilon), 'axis': 'freq'}

    @classmethod
    def from_yaml_dict(cls, d, nbeams, nfreq, ntime):
        """The inverse of :meth:`to_yaml_dict`, at the given geometry."""
        cls.check_yaml_keys(d, SPLINE_DETRENDER_YAML_KEYS)
        if d['axis'] != 'freq':
            raise ValueError(f"GpuSplineDetrender.from_yaml_dict: axis={d['axis']!r}, but this class"
                             f" implements axis 'freq' only (as did the old code)")
        return cls(nbeams, nfreq, ntime, d['nbins'], d['epsilon'])

    @classmethod
    def from_json_dict(cls, d, nbeams, nfreq, ntime):
        """From the legacy rf_pipelines json element (``class_name: spline_detrender``). Its
        ``nt_chunk`` is a processing granularity with no effect on the result (the fit is per
        time sample), and is ignored."""
        check_json_keys(d, 'spline_detrender', ['axis', 'nbins', 'epsilon', 'nt_chunk'])
        if axis_from_json(d['axis']) != 'freq':
            raise ValueError(f"GpuSplineDetrender.from_json_dict: axis={d['axis']!r}, but this class"
                             f" implements axis 'freq' only (as did the old code)")
        if d['nt_chunk'] < 0:
            raise ValueError(f"GpuSplineDetrender.from_json_dict: nt_chunk={d['nt_chunk']} < 0")
        return cls(nbeams, nfreq, ntime, d['nbins'], d['epsilon'])
