"""Tests for Pipeline and RfiMaskPipeline (the containers of the chimefrb port), and for
GpuPythonTransform (the base class of every transform written in python) together with the
python side of GpuTransform underneath it.

Dispatched from ``python -m pirate_frb test --cfrb``.

WHAT IS TESTED, AND HOW. The transforms themselves are tested by their own files. The
containers are a few dozen lines of glue, and a mistake in that glue is GROSS: transforms
run in the wrong order or on the wrong arrays, a scratch region too small or overlapping
another, a bracket that touches the full-resolution intensity or masks the wrong cells, a
yaml or json field dropped or misread. So this file uses TOY transforms whose effect is
exactly predictable -- add a constant to the intensity, multiply it by a constant, zero every
k-th channel's weights -- on integer-valued data, and compares the containers' output against
the closed-form prediction with np.array_equal. Nothing here compares one GPU kernel's output
against another run of itself, and nothing depends on the real transforms being
bit-reproducible.

Every iteration builds one random pipeline in the production chain's shape -- a Pipeline
holding an RfiMaskPipeline and two leaves, in a random order -- launches it, checks the
prediction, then round-trips it through yaml and launches the reconstruction. At iteration 0
only: a hardcoded legacy json (the old rf_pipelines format) exercising every branch of the
reader, a list of arguments that must be refused, the base class's contract for a python
subclass (the arguments launch_checked() receives, the stream, the error messages), one
smoke launch of a chain of the REAL transforms (which catches a scratch accounting error,
since the C++ transforms assert their scratch size), and ExamplePythonTransform on planted
outliers.
"""

import os
import tempfile

import numpy as np
import yaml

from . import (ExamplePythonTransform, GpuBadChannelMask, GpuContainerBase,
               GpuIntensityClipper, GpuPolynomialDetrender, GpuPythonTransform,
               GpuSplineDetrender, GpuStdDevClipper, GpuTransform, RfiMaskPipeline,
               Pipeline)
from .transform_io import (transform_from_json_dict, transform_from_yaml_dict,
                           yaml_string)
from ..utils import atomic_print
from .testutils import default_rng as _default_rng


# -------------------------------------------------------------------------------------------------
#
# Toy transforms. Each is a complete transform (GpuPythonTransform supplies the checked launch),
# so they also exercise the base class and the 'classes=' mechanism of the yaml reader, which
# is how the reader finds classes that are not part of pirate_frb.chimefrb.


class _ToyAdd(GpuPythonTransform):
    """intensity += c. Asks for scratch: inside an RfiMaskPipeline, a wrong scratch offset
    shows up as a too-small or overlapping scratch, which the base class's checks refuse."""

    SCRATCH_NELTS = 1024

    def __init__(self, nbeams, nfreq, ntime, c):
        super().__init__(nbeams, nfreq, ntime, scratch_nelts=self.SCRATCH_NELTS)
        self.c = float(c)

    def launch_checked(self, intensity, weights, scratch):
        assert scratch.shape == (self.SCRATCH_NELTS,)
        intensity += self.c

    def to_yaml_dict(self):
        return {'class_name': '_ToyAdd', 'c': self.c}

    @classmethod
    def from_yaml_dict(cls, d, nbeams, nfreq, ntime):
        cls.check_yaml_keys(d, ['c'])
        return cls(nbeams, nfreq, ntime, d['c'])


class _ToyScale(GpuPythonTransform):
    """intensity *= m."""

    def __init__(self, nbeams, nfreq, ntime, m):
        super().__init__(nbeams, nfreq, ntime)
        self.m = float(m)

    def launch_checked(self, intensity, weights, scratch):
        intensity *= self.m

    def to_yaml_dict(self):
        return {'class_name': '_ToyScale', 'm': self.m}

    @classmethod
    def from_yaml_dict(cls, d, nbeams, nfreq, ntime):
        cls.check_yaml_keys(d, ['m'])
        return cls(nbeams, nfreq, ntime, d['m'])


class _ToyZeroChannels(GpuPythonTransform):
    """weights[:, ::step, :] = 0."""

    def __init__(self, nbeams, nfreq, ntime, step):
        super().__init__(nbeams, nfreq, ntime)
        self.step = int(step)

    def launch_checked(self, intensity, weights, scratch):
        weights[:, ::self.step, :] = 0

    def to_yaml_dict(self):
        return {'class_name': '_ToyZeroChannels', 'step': self.step}

    @classmethod
    def from_yaml_dict(cls, d, nbeams, nfreq, ntime):
        cls.check_yaml_keys(d, ['step'])
        return cls(nbeams, nfreq, ntime, d['step'])


class _ToyContainer(GpuContainerBase):
    """The smallest possible container: runs its elements in order, at its own geometry.

    It exercises GpuContainerBase's helpers, and it stands in for a container written
    OUTSIDE pirate_frb.chimefrb, which is the case transform_io has to get right (see
    _check_container_base)."""

    def __init__(self, transforms):
        (transforms, (nbeams, nfreq, ntime)) = self.check_transforms(transforms)
        super().__init__(nbeams, nfreq, ntime, self.max_scratch_nelts(transforms))
        self.transforms = transforms

    def launch_checked(self, intensity, weights, scratch):
        self.launch_transforms(intensity, weights, scratch)

    def to_yaml_dict(self):
        return {'class_name': '_ToyContainer', 'transforms': self.transforms_yaml_list()}

    @classmethod
    def from_yaml_dict(cls, d, nbeams, nfreq, ntime, classes=None):
        cls.check_yaml_keys(d, ['transforms'])
        return cls(cls.transforms_from_yaml_list(d, nbeams, nfreq, ntime, classes))


class _DuckTransform:
    """Has every attribute and method of a transform, but subclasses neither GpuPythonTransform
    nor GpuTransform. A pipeline must refuse it: the base class is where the launch
    checks live."""

    def __init__(self, nbeams, nfreq, ntime):
        (self.nbeams, self.nfreq, self.ntime, self.scratch_nelts) = (nbeams, nfreq, ntime, 0)

    def launch(self, intensity, weights, scratch, stream=None):
        pass

    def to_yaml_dict(self):
        return {'class_name': '_DuckTransform'}


class _ToySpy(GpuPythonTransform):
    """Records what its launch_checked() was handed -- the current stream, the scratch array's
    shape and dtype, and the three arrays' device pointers -- and modifies nothing. This is
    the check of the base class's contract for a python subclass."""

    def __init__(self, nbeams, nfreq, ntime, scratch_nelts):
        super().__init__(nbeams, nfreq, ntime, scratch_nelts=scratch_nelts)
        self.seen = None

    def launch_checked(self, intensity, weights, scratch):
        import cupy as cp
        self.seen = dict(stream_ptr=cp.cuda.get_current_stream().ptr,
                         scratch_shape=scratch.shape, scratch_dtype=scratch.dtype,
                         ptrs=(intensity.data.ptr, weights.data.ptr, scratch.data.ptr))


TOY_CLASSES = [_ToyAdd, _ToyScale, _ToyZeroChannels, _ToyContainer]


def predict(pipeline, intensity, weights):
    """The closed-form effect of a pipeline of toys on numpy (intensity, weights): what the
    toys do, in order, with an RfiMaskPipeline's bracket written out -- downsampled weights
    are the SUM of a cell's weights, the inner toys act on the downsampled copy (so an inner
    _ToyAdd changes nothing visible), and every full-resolution weight under a cell whose
    downsampled weight is <= w_cutoff is zeroed. Exact on integer data."""

    i = intensity.astype(np.float64)
    w = weights.astype(np.float64)

    for t in pipeline.transforms:
        if isinstance(t, _ToyAdd):
            i = i + t.c
        elif isinstance(t, _ToyScale):
            i = i * t.m
        elif isinstance(t, _ToyZeroChannels):
            w = w.copy()
            w[:, ::t.step, :] = 0
        elif isinstance(t, RfiMaskPipeline):
            (B, F, T) = w.shape
            (Df, Dt) = (t.Df, t.Dt)
            w_ds = w.reshape(B, F // Df, Df, T // Dt, Dt).sum(axis=(2, 4))
            for s in t.transforms:
                if isinstance(s, _ToyZeroChannels):
                    w_ds[:, ::s.step, :] = 0
                else:
                    assert isinstance(s, (_ToyAdd, _ToyScale)), 'predict(): unknown inner toy'
            masked = np.repeat(np.repeat(w_ds <= t.w_cutoff, Df, axis=1), Dt, axis=2)
            w = np.where(masked, 0.0, w)
        else:
            raise AssertionError(f'predict(): unknown transform {type(t).__name__}')

    return (i.astype(np.float32), w.astype(np.float32))


# -------------------------------------------------------------------------------------------------


def random_pipeline(rng):
    """One random pipeline in the production's shape: a Pipeline holding, in a random order,
    a _ToyAdd, a _ToyScale, and an RfiMaskPipeline of [_ToyAdd, _ToyZeroChannels].

    Small integer constants and unit weights, so that predict() is exact in float32. The
    w_cutoff is 0 (only the zeroed channels are masked) or Df*Dt, which every surviving cell's
    summed weight equals exactly, so the upsampler's '<=' then masks everything.
    """

    nbeams = int(rng.integers(1, 3))
    (nfreq_ds, ntime_ds) = (int(rng.choice([32, 64])), int(rng.choice([32, 64])))
    (Df, Dt) = (int(rng.choice([2, 4])), int(rng.choice([1, 2])))
    (nfreq, ntime) = (nfreq_ds * Df, ntime_ds * Dt)
    w_cutoff = float(rng.choice([0.0, Df * Dt]))

    inner = [_ToyAdd(nbeams, nfreq_ds, ntime_ds, int(rng.integers(1, 6))),
             _ToyZeroChannels(nbeams, nfreq_ds, ntime_ds, int(rng.integers(1, 4)))]
    outer = [_ToyAdd(nbeams, nfreq, ntime, int(rng.integers(1, 6))),
             _ToyScale(nbeams, nfreq, ntime, int(rng.integers(2, 5))),
             RfiMaskPipeline(inner, Df, Dt, w_cutoff)]
    order = rng.permutation(3)
    return Pipeline([outer[k] for k in order])


def random_data(rng, shape):
    """Integer-valued float32 intensity in 0..100, and unit weights."""
    intensity = rng.integers(0, 101, size=shape).astype(np.float32)
    weights = np.ones(shape, dtype=np.float32)
    return (intensity, weights)


def _run(cp, pipeline, intensity, weights, scratch):
    gi = cp.asarray(intensity)
    gw = cp.asarray(weights)
    pipeline.launch(gi, gw, scratch)
    cp.cuda.get_current_stream().synchronize()
    return (cp.asnumpy(gi), cp.asnumpy(gw))


def _assert_equal(got, want, what):
    if not np.array_equal(got, want):
        bad = np.argwhere(got != want)
        (b, f, t) = bad[0]
        raise AssertionError(f'test_pipeline: {what}: {len(bad)} of {got.size} elements differ;'
                             f' first at (b,f,t) = ({b},{f},{t}): got {got[b,f,t]}, want {want[b,f,t]}')


def _count_transforms(d):
    """Nodes in a pipeline's yaml dict: the transform itself, plus its elements, recursively.
    Used to check that a chain came out with the nesting it should have."""
    return 1 + sum(_count_transforms(e) for e in d.get('transforms', []))


def _expect_raise(exc, f, *args, **kwargs):
    try:
        f(*args, **kwargs)
    except exc:
        return
    raise AssertionError(f'test_pipeline: expected {exc.__name__} from {getattr(f, "__name__", f)}')


# -------------------------------------------------------------------------------------------------
#
# Iteration-0 checks.


# A legacy rf_pipelines json, just long enough to reach every branch of the reader, at
# (nbeams, nfreq, ntime) = (1, 512, 1024): an ignored element in each pipeline; a
# badchannel_mask (whose band is assumed); a wi_sub_pipeline in the nfreq_out/nds_out spelling
# wrapping a 'pipeline' whose clippers use nt_chunk = 0 (-> ntime) and iter_sigma = 0
# (-> sigma), and whose polydeg is a double; a wi_sub_pipeline in the Df/Dt spelling wrapping
# a bare transform; and a full-resolution detrender.
LEGACY_JSON = {
    'class_name': 'pipeline', 'name': 'pipeline',
    'elements': [
        {'class_name': 'mask_counter', 'where': 'before_rfi', 'nt_chunk': 1024},
        {'class_name': 'badchannel_mask', 'mask_ranges': [[440.2099609375, 440.4052734375]], 'mask_path': ''},
        {'class_name': 'wi_sub_pipeline',
         'sub_pipeline': {
             'class_name': 'pipeline', 'name': 'pipeline',
             'elements': [
                 {'class_name': 'std_dev_clipper', 'Df': 1, 'Dt': 1, 'sigma': 3, 'two_pass': True,
                  'axis': 'AXIS_TIME', 'nt_chunk': 0},
                 {'class_name': 'intensity_clipper', 'niter': 9, 'Df': 2, 'Dt': 16, 'iter_sigma': 0,
                  'sigma': 5, 'two_pass': False, 'axis': 'AXIS_NONE', 'nt_chunk': 1024},
                 {'class_name': 'polynomial_detrender', 'epsilon': 0.01, 'polydeg': 4.0,
                  'nt_chunk': 512, 'axis': 'AXIS_TIME'},
                 {'class_name': 'chime_slow_pulsar_writer', 'nt_chunk': 1024, 'name': 'slow'},
             ]},
         'Df': 0, 'Dt': 0, 'nfreq_out': 128, 'nds_out': 1, 'w_cutoff': 0},
        {'class_name': 'wi_sub_pipeline',
         'sub_pipeline': {'class_name': 'spline_detrender', 'epsilon': 0.0003, 'nt_chunk': 0,
                          'nbins': 6, 'axis': 'AXIS_FREQ'},
         'Df': 2, 'Dt': 2, 'nfreq_out': 0, 'nds_out': 0, 'w_cutoff': 1.5},
        {'class_name': 'polynomial_detrender', 'epsilon': 0.01, 'polydeg': 4, 'nt_chunk': 1024,
         'axis': 'AXIS_TIME'},
    ]}


def _check_legacy_json():
    (nbeams, nfreq, ntime) = (1, 512, 1024)
    p = Pipeline.from_json_dict(LEGACY_JSON, nbeams, nfreq, ntime)
    assert (p.nbeams, p.nfreq, p.ntime) == (nbeams, nfreq, ntime)
    assert len(p.transforms) == 4, 'mask_counter should have been skipped'

    (bcm, sub1, sub2, poly) = p.transforms
    assert isinstance(bcm, GpuBadChannelMask)
    assert bcm.freq_range == (400.0, 800.0) and bcm.mask_ranges == [(440.2099609375, 440.4052734375)]
    assert bcm.nmasked > 0

    assert isinstance(sub1, RfiMaskPipeline)
    assert (sub1.Df, sub1.Dt, sub1.w_cutoff) == (4, 1, 0.0), 'nfreq_out=128 at nfreq=512 is Df=4; nds_out=1 is Dt=1'
    assert (sub1.nbeams, sub1.nfreq, sub1.ntime) == (nbeams, nfreq, ntime)
    assert len(sub1.transforms) == 3, 'chime_slow_pulsar_writer should have been skipped'
    (sd, ic, pd) = sub1.transforms
    assert isinstance(sd, GpuStdDevClipper) and (sd.nfreq, sd.ntime, sd.nt_chunk) == (128, 1024, 1024), 'nt_chunk 0 means ntime'
    assert sd.axis == 'time' and sd.sigma == 3.0 and sd.two_pass
    assert isinstance(ic, GpuIntensityClipper) and ic.iter_sigma == 5.0, 'iter_sigma 0 means sigma'
    assert (ic.axis, ic.Df, ic.Dt, ic.niter, ic.two_pass, ic.nt_chunk) == ('none', 2, 16, 9, False, 1024)
    assert isinstance(pd, GpuPolynomialDetrender) and (pd.polydeg, pd.nt_chunk, pd.ntime) == (4, 512, 1024)

    assert isinstance(sub2, RfiMaskPipeline) and (sub2.Df, sub2.Dt, sub2.w_cutoff) == (2, 2, 1.5)
    assert len(sub2.transforms) == 1 and isinstance(sub2.transforms[0], GpuSplineDetrender)
    assert (sub2.transforms[0].nfreq, sub2.transforms[0].ntime, sub2.transforms[0].nbins) == (256, 512, 6)

    assert isinstance(poly, GpuPolynomialDetrender) and (poly.polydeg, poly.nt_chunk, poly.epsilon) == (4, 1024, 0.01)

    # The real transforms' yaml methods: a dict round trip, and the file-level string.
    d = p.to_yaml_dict()
    d2 = yaml.safe_load(yaml.safe_dump(d, sort_keys=False))
    assert d2 == d, 'yaml.safe_dump/safe_load changed the dict (a non-plain type in to_yaml_dict?)'
    p2 = Pipeline.from_yaml_dict(d2, nbeams, nfreq, ntime)
    assert p2.to_yaml_dict() == d
    # The file-level string's layout (transform_io._YamlDumper): a pipeline stays in block
    # style, since it holds a list of transforms, while a leaf transform's parameters go
    # inline -- which is what keeps a long chain readable (the production chain is 183 lines
    # this way and 893 with every parameter on its own line).
    text = yaml_string(d)
    assert text.startswith('class_name: Pipeline\n'), text[:200]
    assert any(line.lstrip().startswith('- {class_name: Gpu') for line in text.splitlines()), text

    # The file-level pair, through a temporary file.
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, 'chain.yml')
        p.write_yaml_file(fname)
        p3 = Pipeline.read_yaml_file(fname, nbeams=nbeams, nfreq=nfreq, ntime=ntime)
        assert p3.to_yaml_dict() == d, 'write_yaml_file/read_yaml_file changed the pipeline'
        assert open(fname).readline().startswith('#'), 'write_yaml_file should start with a comment header'
    assert _count_transforms(d) == 1 + 4 + 3 + 1, 'the legacy json produced the wrong nesting'

    # Two more legacy conventions: the consistency check when both spellings are given, and
    # the explicit-nds spelling inside a nested wi_sub_pipeline.
    both = dict(LEGACY_JSON['elements'][2], Df=4, Dt=1)
    assert RfiMaskPipeline.from_json_dict(both, nbeams, nfreq, ntime).Df == 4
    _expect_raise(ValueError, RfiMaskPipeline.from_json_dict, dict(both, Df=2), nbeams, nfreq, ntime)
    _expect_raise(ValueError, RfiMaskPipeline.from_json_dict, dict(both, nfreq_out=100, Df=0), nbeams, nfreq, ntime)
    _expect_raise(ValueError, RfiMaskPipeline.from_json_dict, dict(both, nds_out=3, Dt=0), nbeams, nfreq, ntime)


def _check_arguments():
    (B, F, T) = (1, 64, 64)
    add = _ToyAdd(B, F, T, 1)

    _expect_raise(ValueError, Pipeline, [])
    _expect_raise(ValueError, Pipeline, [add, _ToyAdd(B, 2*F, T, 1)])                   # mixed geometry
    _expect_raise(TypeError, Pipeline, [add, 'not a transform'])
    _expect_raise(ValueError, RfiMaskPipeline, [add], 1, 1)                                # (1, 1)
    _expect_raise(ValueError, RfiMaskPipeline, [add], 2, 1, -1.0)                          # w_cutoff < 0
    _expect_raise(ValueError, RfiMaskPipeline, [_ToyAdd(B, 48, T, 1)], 2, 1)              # inner nfreq % 32
    _expect_raise(RuntimeError, GpuPythonTransform, 0, F, T)                               # nbeams < 1
    _expect_raise(RuntimeError, GpuPythonTransform, B, F, T, -1)                           # scratch_nelts < 0
    _expect_raise(TypeError, Pipeline, [add, _DuckTransform(B, F, T)])                    # not a GpuTransform

    _expect_raise(ValueError, transform_from_yaml_dict, {'class_name': 'NoSuchTransform'}, B, F, T)
    _expect_raise(ValueError, transform_from_yaml_dict, {'class_name': '_ToyAdd', 'c': 1.0}, B, F, T)  # no classes=
    _expect_raise(ValueError, _ToyAdd.from_yaml_dict, {'class_name': '_ToyAdd', 'c': 1.0, 'd': 2}, B, F, T)  # extra key
    _expect_raise(ValueError, _ToyAdd.from_yaml_dict, {'class_name': '_ToyAdd'}, B, F, T)              # missing key
    _expect_raise(ValueError, _ToyAdd.from_yaml_dict, {'class_name': '_ToyScale', 'c': 1.0}, B, F, T)  # wrong class
    _expect_raise(ValueError, Pipeline.from_yaml_dict, {'class_name': 'Pipeline', 'transforms': []}, B, F, T)
    _expect_raise(ValueError, RfiMaskPipeline.from_yaml_dict,
                  {'class_name': 'RfiMaskPipeline', 'Df': 3, 'Dt': 1, 'w_cutoff': 0.0,
                   'transforms': [{'class_name': 'ExamplePythonTransform', 'sigma': 3.0}]}, B, F, T)     # 64 % 3
    _expect_raise(ValueError, GpuPolynomialDetrender.from_yaml_dict,
                  {'class_name': 'GpuPolynomialDetrender', 'polydeg': 4, 'epsilon': 0.01,
                   'nt_chunk': 64, 'axis': 'freq'}, B, F, T)                                          # wrong axis

    _expect_raise(ValueError, transform_from_json_dict, {'class_name': 'mask_expander'}, B, F, T)      # unported
    _expect_raise(RuntimeError, GpuStdDevClipper.from_json_dict,
                  {'class_name': 'std_dev_clipper', 'Df': 1, 'Dt': 1, 'sigma': 3, 'two_pass': True,
                   'axis': 'AXIS_TIME', 'nt_chunk': 32}, B, F, T)                                     # nt_chunk != ntime
    _expect_raise(ValueError, GpuPolynomialDetrender.from_json_dict,
                  {'class_name': 'polynomial_detrender', 'epsilon': 0.01, 'polydeg': 4.5,
                   'nt_chunk': 64, 'axis': 'AXIS_TIME'}, B, F, T)                                     # non-integer polydeg


def _check_base_class(cp):
    """GpuPythonTransform's contract for a python transform, which goes through the C++ base
    class and back: launch_checked() runs on the launch stream, sees exactly scratch_nelts
    elements of scratch, and sees the caller's arrays (not copies); a failed check names the
    subclass; a missing launch_checked() is a NotImplementedError; a python class that skips
    GpuPythonTransform is refused at its first launch; check_yaml_keys() checks by class name;
    and every transform, C++ or python, is a GpuTransform."""

    (B, F, T) = (2, 64, 128)
    intensity = cp.zeros((B, F, T), dtype=cp.float32)
    weights = cp.ones((B, F, T), dtype=cp.float32)

    # The stream and the arrays. The scratch is oversized on purpose: launch_checked() must
    # see exactly scratch_nelts elements of it.
    spy = _ToySpy(B, F, T, 100)
    scratch = cp.empty(137, dtype=cp.float32)
    stream = cp.cuda.Stream()
    spy.launch(intensity, weights, scratch, stream=stream)
    stream.synchronize()
    assert spy.seen is not None, 'a python launch_checked() was not called'
    assert spy.seen['stream_ptr'] == stream.ptr, 'launch_checked() did not run with the launch stream current'
    assert spy.seen['scratch_shape'] == (100,) and spy.seen['scratch_dtype'] == cp.float32
    assert spy.seen['ptrs'] == (intensity.data.ptr, weights.data.ptr, scratch.data.ptr), \
        'launch_checked() did not receive views of the caller\'s arrays'

    spy0 = _ToySpy(B, F, T, 0)
    spy0.launch(intensity, weights, None)
    assert spy0.seen['scratch_shape'] == (0,) and spy0.seen['scratch_dtype'] == cp.float32

    # A failed check is a RuntimeError from C++, worded for a python reader, naming the class.
    try:
        spy.launch(intensity, cp.ones((B, F, T // 2), dtype=cp.float32), None)
    except RuntimeError as e:
        msg = str(e)
        assert msg.startswith('_ToySpy.launch():') and "'weights'" in msg and '(2, 64, 64)' in msg, msg
    else:
        raise AssertionError('a wrong-shaped weights array was accepted')
    _expect_raise(RuntimeError, spy.launch, intensity, weights, cp.empty(5, dtype=cp.float32))   # too small
    _expect_raise(RuntimeError, spy.launch, intensity, intensity, None)                            # aliased
    _expect_raise(RuntimeError, spy.launch, intensity, np.ones((B, F, T), dtype=np.float32), None) # host array
    _expect_raise(TypeError, spy.launch, intensity, cp.ones((B, F, T), dtype=cp.float64), None)   # wrong dtype

    # The stubs a subclass must replace, reached through the trampoline (launch_checked) or
    # directly (to_yaml_dict).
    bare = GpuPythonTransform(B, F, T)
    _expect_raise(NotImplementedError, bare.launch, intensity, weights, None)
    _expect_raise(NotImplementedError, bare.to_yaml_dict)

    # A python class that subclasses the C++ base directly, skipping GpuPythonTransform, is
    # constructible but has no dispatcher for the trampoline to call: its first launch is a
    # RuntimeError that says where to derive from, and its launch_checked() is never reached.
    class _Direct(GpuTransform):
        def __init__(self):
            super().__init__('_Direct', B, F, T, 0)

        def launch_checked(self, intensity, weights, scratch):
            raise AssertionError('a launch_checked() defined outside GpuPythonTransform was called')

    for t in (GpuTransform('Bare', B, F, T, 0), _Direct()):
        try:
            t.launch(intensity, weights, None)
        except RuntimeError as e:
            assert 'GpuPythonTransform' in str(e), str(e)
        else:
            raise AssertionError(f'{type(t).__name__}.launch() was accepted without a dispatcher')

    # check_yaml_keys() by name: the class's own name, exactly the given keys.
    _ToyAdd.check_yaml_keys({'class_name': '_ToyAdd', 'c': 1.0}, ['c'])
    _expect_raise(ValueError, _ToyAdd.check_yaml_keys, {'class_name': '_ToyScale', 'c': 1.0}, ['c'])
    _expect_raise(ValueError, _ToyAdd.check_yaml_keys, {'class_name': '_ToyAdd', 'c': 1.0, 'd': 2}, ['c'])
    _expect_raise(ValueError, _ToyAdd.check_yaml_keys, 'not a dict', ['c'])

    for t in (GpuBadChannelMask(B, F, T, [(500.0, 520.0)], (400.0, 800.0)), spy,
              Pipeline([spy]), RfiMaskPipeline([_ToyAdd(B, F // 2, T, 1.0)], 2, 1)):
        assert isinstance(t, GpuTransform), f'{type(t).__name__} is not a GpuTransform'
    assert isinstance(spy, GpuPythonTransform) and isinstance(Pipeline([spy]), GpuPythonTransform)
    assert not isinstance(GpuBadChannelMask(B, F, T, [(500.0, 520.0)], (400.0, 800.0)), GpuPythonTransform)


def _check_container_base(cp):
    """GpuContainerBase: the base class of a transform that runs other transforms.

    The case that matters is a container defined OUTSIDE pirate_frb.chimefrb.
    transform_io decides whether to pass 'classes' down to a factory by testing
    issubclass(cls, GpuContainerBase), so such a container must have its own elements
    resolved from the caller's list. When the two pipeline classes were hardcoded instead,
    this raised "unknown transform class_name '_ToyAdd'" -- telling the caller to pass a
    class they had already passed."""

    (B, F, T) = (1, 64, 64)
    p = Pipeline([_ToyContainer([_ToyAdd(B, F, T, 2.0), _ToyScale(B, F, T, 3.0)])])
    d = p.to_yaml_dict()
    assert d['transforms'][0]['class_name'] == '_ToyContainer'
    assert _count_transforms(d) == 1 + 1 + 2

    back = Pipeline.from_yaml_dict(d, B, F, T, classes=TOY_CLASSES)
    assert back.to_yaml_dict() == d, "a caller's own container did not round-trip through yaml"
    # ... and the classes= list really is what finds the nested toys.
    _expect_raise(ValueError, Pipeline.from_yaml_dict, d, B, F, T)

    # It runs: (i + 2) * 3, through the container, on the caller's arrays.
    intensity = np.ones((B, F, T), dtype=np.float32)
    weights = np.ones((B, F, T), dtype=np.float32)
    (gi, gw) = _run(cp, back, intensity, weights, None)
    _assert_equal(gi, 9.0 * np.ones((B, F, T), dtype=np.float32), 'intensity through _ToyContainer')

    # Containers are GpuContainerBase; leaf transforms are not.
    assert isinstance(p, GpuContainerBase) and isinstance(back.transforms[0], GpuContainerBase)
    assert isinstance(RfiMaskPipeline([_ToyAdd(B, F // 2, T, 1.0)], 2, 1), GpuContainerBase)
    assert not isinstance(_ToyAdd(B, F, T, 1.0), GpuContainerBase)
    assert not isinstance(GpuBadChannelMask(B, F, T, [(500.0, 520.0)], (400.0, 800.0)), GpuContainerBase)

    # check_transforms(), called by name: it reports the class it was called on.
    (ts, geom) = _ToyContainer.check_transforms([_ToyAdd(B, F, T, 1.0)])
    assert (geom == (B, F, T)) and (len(ts) == 1) and isinstance(ts, tuple)
    _expect_raise(ValueError, _ToyContainer.check_transforms, [])
    _expect_raise(TypeError, _ToyContainer.check_transforms, [_DuckTransform(B, F, T)])
    try:
        _ToyContainer.check_transforms([_ToyAdd(B, F, T, 1.0), _ToyAdd(B, 2 * F, T, 1.0)])
    except ValueError as e:
        assert str(e).startswith('_ToyContainer:'), str(e)
    else:
        raise AssertionError('check_transforms accepted a mixed geometry')


def _check_real_transforms(cp, rng):
    """One smoke launch of a chain of the real transforms, in the production's shape. Not a
    correctness check: the transforms have their own tests. It catches a scratch accounting
    error (the C++ transforms assert their scratch size) or a geometry mistake, which the
    toys, having no C++ side, could not."""

    (B, F, T) = (1, 128, 128)
    (Fd, Td) = (F // 2, T)
    inner = [GpuStdDevClipper(B, Fd, Td, Td, 'time', 3.0, 1, 1, True),
             GpuIntensityClipper(B, Fd, Td, Td, 'freq', 5.0, 1, 1, 2, 3.0, True),
             GpuPolynomialDetrender(B, Fd, Td, 2, 0.01, 64),
             GpuSplineDetrender(B, Fd, Td, 3, 3.0e-4),
             ExamplePythonTransform(B, Fd, Td, 3.0)]
    p = Pipeline([GpuBadChannelMask(B, F, T, [(500.0, 520.0)], (400.0, 800.0)),
                  RfiMaskPipeline(inner, 2, 1, 0.0),
                  GpuPolynomialDetrender(B, F, T, 2, 0.01, T)])

    assert p.scratch_nelts >= max(t.scratch_nelts for t in inner)
    assert _count_transforms(p.to_yaml_dict()) == 1 + 1 + 1 + 5 + 1

    intensity = (100.0 + rng.standard_normal((B, F, T))).astype(np.float32)
    intensity[0, 7, 3] = 1.0e4                                   # a spike for the clippers
    weights = (rng.uniform(size=(B, F, T)) < 0.9).astype(np.float32)

    (gi, gw) = _run(cp, p, intensity, weights, None)
    assert np.isfinite(gi).all() and np.isfinite(gw).all(), 'the real chain produced non-finite output'
    assert (gw == 0).sum() > (weights == 0).sum(), 'the real chain masked nothing'
    assert np.all(gw <= weights), 'a weight increased'


def _check_example_transform(cp, rng):
    """ExamplePythonTransform on Gaussian data with eight planted 10-sigma outliers: the eight
    are masked, nothing within 2.5 sigma of its row mean is, the intensity is untouched, and a
    NaN at a zero-weight sample changes nothing."""

    (B, F, T) = (2, 16, 256)
    intensity = rng.standard_normal((B, F, T)).astype(np.float32)
    weights = np.ones((B, F, T), dtype=np.float32)

    flat = rng.choice(B * F * T, size=8, replace=False)
    planted = np.zeros(B * F * T, dtype=bool)
    planted[flat] = True
    planted = planted.reshape(B, F, T)
    intensity[planted] = 10.0 * rng.choice([-1.0, 1.0], size=8)

    ex = ExamplePythonTransform(B, F, T, sigma=3.0)
    (gi, gw) = _run(cp, ex, intensity, weights, None)

    assert np.array_equal(gi, intensity), 'ExamplePythonTransform modified the intensity'
    assert (gw[planted] == 0).all(), 'a planted 10-sigma outlier was not masked'
    d = np.abs(intensity - intensity.mean(axis=2, keepdims=True))
    assert (gw[d < 2.5] == 1).all(), 'a sample within 2.5 sigma of its row mean was masked'

    weights[0, 0, 5] = 0.0
    (_, gw0) = _run(cp, ex, intensity, weights, None)
    poisoned = intensity.copy()
    poisoned[0, 0, 5] = np.nan
    (_, gw1) = _run(cp, ex, poisoned, weights, None)
    assert np.array_equal(gw0, gw1), 'a NaN at a zero-weight sample changed the result'

    d = ex.to_yaml_dict()
    assert transform_from_yaml_dict(d, B, F, T).to_yaml_dict() == d


# -------------------------------------------------------------------------------------------------


def test_pipeline(iteration=0, rng=None, verbose=False):
    """One random pipeline of toy transforms, launched and checked against its closed-form
    result, then round-tripped through yaml; on iteration 0, also the legacy json reader, the
    refused arguments, GpuPythonTransform's contract for a python transform, a smoke launch of
    the real transforms, and ExamplePythonTransform."""

    try:
        import cupy as cp
    except ImportError:
        if verbose:
            atomic_print('    test_pipeline: cupy not available, skipped')
        return

    rng = _default_rng(rng)

    p = random_pipeline(rng)
    shape = (p.nbeams, p.nfreq, p.ntime)
    (intensity, weights) = random_data(rng, shape)
    (want_i, want_w) = predict(p, intensity, weights)
    tag = f'({p!r})'

    # (a) The pipeline against its prediction, with an allocated scratch.
    (gi, gw) = _run(cp, p, intensity, weights, None)
    _assert_equal(gi, want_i, f'intensity {tag}')
    _assert_equal(gw, want_w, f'weights {tag}')

    # (b) Scratch is really scratch: an oversized one, prefilled with NaN.
    scratch = cp.full(p.scratch_nelts + 37, np.nan, dtype=cp.float32)
    (gi, gw) = _run(cp, p, intensity, weights, scratch)
    _assert_equal(gi, want_i, f'intensity with NaN scratch {tag}')
    _assert_equal(gw, want_w, f'weights with NaN scratch {tag}')

    # (c) yaml round trip: the dict, and the reconstruction's output. The toys are not part of
    # pirate_frb.chimefrb, so the reader needs classes=; without it, it must say so.
    d = p.to_yaml_dict()
    d2 = yaml.safe_load(yaml.safe_dump(d, sort_keys=False))
    assert d2 == d, f'yaml.safe_dump/safe_load changed the dict {tag}'
    p2 = Pipeline.from_yaml_dict(d2, *shape, classes=TOY_CLASSES)
    assert p2.to_yaml_dict() == d, f'yaml round trip changed the pipeline {tag}'
    assert p2.scratch_nelts == p.scratch_nelts
    (gi, gw) = _run(cp, p2, intensity, weights, None)
    _assert_equal(gi, want_i, f'intensity after yaml round trip {tag}')
    _assert_equal(gw, want_w, f'weights after yaml round trip {tag}')
    _expect_raise(ValueError, Pipeline.from_yaml_dict, d2, *shape)

    if iteration == 0:
        _check_legacy_json()
        _check_arguments()
        _check_base_class(cp)
        _check_container_base(cp)
        _check_real_transforms(cp, rng)
        _check_example_transform(cp, rng)

    if verbose:
        atomic_print(f'    test_pipeline: {p!r}: ok')
