"""Tests for WiPipeline, RfiMaskPipeline and CupyTransformBase: the containers of the
chimefrb port.

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

Every iteration builds one random pipeline in the production chain's shape -- a WiPipeline
holding an RfiMaskPipeline and two leaves, in a random order -- launches it, checks the
prediction, then round-trips it through yaml and launches the reconstruction. At iteration 0
only: a hardcoded legacy json (the old rf_pipelines format) exercising every branch of the
reader, a list of arguments that must be refused, one smoke launch of a chain of the REAL
transforms (which catches a scratch accounting error, since the C++ transforms assert their
scratch size), and ExampleCupyTransform on planted outliers.
"""

import numpy as np
import yaml

from . import (ClipperAxis, CupyTransformBase, ExampleCupyTransform, GpuBadChannelMask,
               GpuIntensityClipper, GpuPolynomialDetrender, GpuSplineDetrender,
               GpuStdDevClipper, RfiMaskPipeline, WiPipeline)
from .transform_io import check_yaml_keys, transform_from_json_dict, transform_from_yaml_dict
from ..utils import atomic_print
from .testutils import default_rng as _default_rng


# -------------------------------------------------------------------------------------------------
#
# Toy transforms. Each is a complete transform (CupyTransformBase supplies the checked launch),
# so they also exercise the base class and the 'classes=' mechanism of the yaml reader, which
# is how the reader finds classes that are not part of pirate_frb.chimefrb.


class _ToyAdd(CupyTransformBase):
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
        check_yaml_keys(d, '_ToyAdd', ['c'])
        return cls(nbeams, nfreq, ntime, d['c'])


class _ToyScale(CupyTransformBase):
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
        check_yaml_keys(d, '_ToyScale', ['m'])
        return cls(nbeams, nfreq, ntime, d['m'])


class _ToyZeroChannels(CupyTransformBase):
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
        check_yaml_keys(d, '_ToyZeroChannels', ['step'])
        return cls(nbeams, nfreq, ntime, d['step'])


TOY_CLASSES = [_ToyAdd, _ToyScale, _ToyZeroChannels]


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
    """One random pipeline in the production's shape: a WiPipeline holding, in a random order,
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
    return WiPipeline([outer[k] for k in order])


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
        raise AssertionError(f'test_wi_pipeline: {what}: {len(bad)} of {got.size} elements differ;'
                             f' first at (b,f,t) = ({b},{f},{t}): got {got[b,f,t]}, want {want[b,f,t]}')


def _expect_raise(exc, f, *args, **kwargs):
    try:
        f(*args, **kwargs)
    except exc:
        return
    raise AssertionError(f'test_wi_pipeline: expected {exc.__name__} from {getattr(f, "__name__", f)}')


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
    p = WiPipeline.from_json_dict(LEGACY_JSON, nbeams, nfreq, ntime)
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
    assert sd.axis == ClipperAxis.TIME and sd.sigma == 3.0 and sd.two_pass
    assert isinstance(ic, GpuIntensityClipper) and ic.iter_sigma == 5.0, 'iter_sigma 0 means sigma'
    assert (ic.axis, ic.Df, ic.Dt, ic.niter, ic.two_pass, ic.nt_chunk) == (ClipperAxis.NONE, 2, 16, 9, False, 1024)
    assert isinstance(pd, GpuPolynomialDetrender) and (pd.polydeg, pd.nt_chunk, pd.ntime) == (4, 512, 1024)

    assert isinstance(sub2, RfiMaskPipeline) and (sub2.Df, sub2.Dt, sub2.w_cutoff) == (2, 2, 1.5)
    assert len(sub2.transforms) == 1 and isinstance(sub2.transforms[0], GpuSplineDetrender)
    assert (sub2.transforms[0].nfreq, sub2.transforms[0].ntime, sub2.transforms[0].nbins) == (256, 512, 6)

    assert isinstance(poly, GpuPolynomialDetrender) and (poly.polydeg, poly.nt_chunk, poly.epsilon) == (4, 1024, 0.01)

    # The real transforms' yaml methods: a dict round trip, and the file-level string.
    d = p.to_yaml_dict()
    d2 = yaml.safe_load(yaml.safe_dump(d, sort_keys=False))
    assert d2 == d, 'yaml.safe_dump/safe_load changed the dict (a non-plain type in to_yaml_dict?)'
    p2 = WiPipeline.from_yaml_dict(d2, nbeams, nfreq, ntime)
    assert p2.to_yaml_dict() == d
    assert isinstance(p.yaml_string(), str) and ('class_name: WiPipeline' in p.yaml_string())
    assert len(p.describe().splitlines()) == 1 + 4 + 3 + 1

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

    _expect_raise(ValueError, WiPipeline, [])
    _expect_raise(ValueError, WiPipeline, [add, _ToyAdd(B, 2*F, T, 1)])                   # mixed geometry
    _expect_raise(TypeError, WiPipeline, [add, 'not a transform'])
    _expect_raise(ValueError, RfiMaskPipeline, [add], 1, 1)                                # (1, 1)
    _expect_raise(ValueError, RfiMaskPipeline, [add], 2, 1, -1.0)                          # w_cutoff < 0
    _expect_raise(ValueError, RfiMaskPipeline, [_ToyAdd(B, 48, T, 1)], 2, 1)              # inner nfreq % 32
    _expect_raise(ValueError, CupyTransformBase, 0, F, T)

    _expect_raise(ValueError, transform_from_yaml_dict, {'class_name': 'NoSuchTransform'}, B, F, T)
    _expect_raise(ValueError, transform_from_yaml_dict, {'class_name': '_ToyAdd', 'c': 1.0}, B, F, T)  # no classes=
    _expect_raise(ValueError, _ToyAdd.from_yaml_dict, {'class_name': '_ToyAdd', 'c': 1.0, 'd': 2}, B, F, T)  # extra key
    _expect_raise(ValueError, _ToyAdd.from_yaml_dict, {'class_name': '_ToyAdd'}, B, F, T)              # missing key
    _expect_raise(ValueError, _ToyAdd.from_yaml_dict, {'class_name': '_ToyScale', 'c': 1.0}, B, F, T)  # wrong class
    _expect_raise(ValueError, WiPipeline.from_yaml_dict, {'class_name': 'WiPipeline', 'transforms': []}, B, F, T)
    _expect_raise(ValueError, RfiMaskPipeline.from_yaml_dict,
                  {'class_name': 'RfiMaskPipeline', 'Df': 3, 'Dt': 1, 'w_cutoff': 0.0,
                   'transforms': [{'class_name': 'ExampleCupyTransform', 'sigma': 3.0}]}, B, F, T)     # 64 % 3
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


def _check_real_transforms(cp, rng):
    """One smoke launch of a chain of the real transforms, in the production's shape. Not a
    correctness check: the transforms have their own tests. It catches a scratch accounting
    error (the C++ transforms assert their scratch size) or a geometry mistake, which the
    toys, having no C++ side, could not."""

    (B, F, T) = (1, 128, 128)
    (Fd, Td) = (F // 2, T)
    inner = [GpuStdDevClipper(B, Fd, Td, Td, ClipperAxis.TIME, 3.0, 1, 1, True),
             GpuIntensityClipper(B, Fd, Td, Td, ClipperAxis.FREQ, 5.0, 1, 1, 2, 3.0, True),
             GpuPolynomialDetrender(B, Fd, Td, 2, 0.01, 64),
             GpuSplineDetrender(B, Fd, Td, 3, 3.0e-4),
             ExampleCupyTransform(B, Fd, Td, 3.0)]
    p = WiPipeline([GpuBadChannelMask(B, F, T, [(500.0, 520.0)], (400.0, 800.0)),
                    RfiMaskPipeline(inner, 2, 1, 0.0),
                    GpuPolynomialDetrender(B, F, T, 2, 0.01, T)])

    assert p.scratch_nelts >= max(t.scratch_nelts for t in inner)
    assert len(p.describe().splitlines()) == 1 + 1 + 1 + 5 + 1

    intensity = (100.0 + rng.standard_normal((B, F, T))).astype(np.float32)
    intensity[0, 7, 3] = 1.0e4                                   # a spike for the clippers
    weights = (rng.uniform(size=(B, F, T)) < 0.9).astype(np.float32)

    (gi, gw) = _run(cp, p, intensity, weights, None)
    assert np.isfinite(gi).all() and np.isfinite(gw).all(), 'the real chain produced non-finite output'
    assert (gw == 0).sum() > (weights == 0).sum(), 'the real chain masked nothing'
    assert np.all(gw <= weights), 'a weight increased'


def _check_example_transform(cp, rng):
    """ExampleCupyTransform on Gaussian data with eight planted 10-sigma outliers: the eight
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

    ex = ExampleCupyTransform(B, F, T, sigma=3.0)
    (gi, gw) = _run(cp, ex, intensity, weights, None)

    assert np.array_equal(gi, intensity), 'ExampleCupyTransform modified the intensity'
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


def test_wi_pipeline(iteration=0, rng=None, verbose=False):
    """One random pipeline of toy transforms, launched and checked against its closed-form
    result, then round-tripped through yaml; on iteration 0, also the legacy json reader, the
    refused arguments, a smoke launch of the real transforms, and ExampleCupyTransform."""

    try:
        import cupy as cp
    except ImportError:
        if verbose:
            atomic_print('    test_wi_pipeline: cupy not available, skipped')
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
    p2 = WiPipeline.from_yaml_dict(d2, *shape, classes=TOY_CLASSES)
    assert p2.to_yaml_dict() == d, f'yaml round trip changed the pipeline {tag}'
    assert p2.scratch_nelts == p.scratch_nelts
    (gi, gw) = _run(cp, p2, intensity, weights, None)
    _assert_equal(gi, want_i, f'intensity after yaml round trip {tag}')
    _assert_equal(gw, want_w, f'weights after yaml round trip {tag}')
    _expect_raise(ValueError, WiPipeline.from_yaml_dict, d2, *shape)

    if iteration == 0:
        _check_legacy_json()
        _check_arguments()
        _check_real_transforms(cp, rng)
        _check_example_transform(cp, rng)

    if verbose:
        atomic_print(f'    test_wi_pipeline: {p!r}: ok')
