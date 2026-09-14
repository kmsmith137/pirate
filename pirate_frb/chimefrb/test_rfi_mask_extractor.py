"""Randomized unit tests for RfiMaskExtractor, GpuTransform.get_mask_extractor(), and the
legacy json rule that turns a chain's last mask_counter into an extractor.

Dispatched from ``python -m pirate_frb test --cfrb``.

Two oracles, neither of which re-implements anything. In a plain Pipeline the extractor sees
the weights the pipeline returns, so the mask must be ``np.packbits(weights_out > 0,
bitorder='little')``. Inside an RfiMaskPipeline the weights it sees are discarded with the
scratch, but the pipeline's full-resolution output determines them: with ``w_cutoff = 0`` a
downsampled cell is kept iff its full-resolution weights survive, so ``mask[cell] ==
any(weights_out[cell] > 0)`` exactly (the transforms only ever zero weights, and a cell whose
inputs were all zero is masked either way). That identity is also why a chain's mask can be
recovered from its full-resolution output when nothing after the sub-pipeline touches the
weights.
"""

import numpy as np

from . import (ExamplePythonTransform, GpuBadChannelMask, Pipeline, RfiMaskExtractor,
               RfiMaskPipeline)
from ..utils import atomic_print
from .utils import LEGACY_EXTRACT_KEY, legacy_chain_from_json
from .testutils import default_rng as _default_rng


def _expect_raise(exc, f, *args, **kwds):
    try:
        f(*args, **kwds)
    except exc:
        return
    raise AssertionError(f'test_rfi_mask_extractor: expected {exc.__name__} from {f}')


def packed_reference(weights, m):
    """The mask an extractor planted with 'm' windows must produce from (nbeams, nfreq, ntime)
    weights: shape (m, nbeams, nfreq, ntime//(8*m)), LSB-first, set bit = weight > 0."""

    (B, F, T) = weights.shape
    w = T // m
    packed = np.packbits((weights > 0).astype(np.uint8), axis=-1, bitorder='little')   # (B, F, T/8)
    return np.ascontiguousarray(packed.reshape(B, F, m, w // 8).transpose(2, 0, 1, 3))


def random_weights(rng, shape):
    """{0, positive} weights with a random masked fraction, some all-zero rows, and no
    denormals (the packing kernel flushes those; its own test says why)."""

    p = np.clip(rng.uniform(-0.1, 1.1), 0.0, 1.0)
    w = (rng.uniform(size=shape) < p) * rng.uniform(0.5, 1.5, size=shape)
    dead = rng.uniform(size=shape[:-1]) < 0.1
    w[dead] = 0.0
    return w.astype(np.float32)


def _run(cp, pipeline, intensity, weights):
    gi = cp.asarray(intensity)
    gw = cp.asarray(weights)
    pipeline.launch(gi, gw, None)
    cp.cuda.get_current_stream().synchronize()
    return (cp.asnumpy(gi), cp.asnumpy(gw))


def _planted(cp, ext, m):
    """A destination for 'm' windows, prefilled so that an unwritten byte is visible."""
    dst = cp.full((m, ext.nbeams, ext.nfreq, ext.ntime // (8 * m)), 0xAA, dtype=cp.uint8)
    ext.set_rfi_mask(dst)
    return dst


def _check_plain(cp, rng, verbose):
    """An extractor after a clipper in a plain Pipeline: the mask is the packed output
    weights, and the extractor changes nothing."""

    B = int(rng.integers(1, 3))
    F = int(rng.integers(1, 40))
    m = int(rng.choice([1, 2, 3, 4]))
    T = 1024 * m * int(rng.integers(1, 3))

    intensity = (100.0 + rng.standard_normal((B, F, T))).astype(np.float32)
    intensity[rng.uniform(size=(B, F, T)) < 0.01] += 50.0          # outliers for the clipper
    weights = random_weights(rng, (B, F, T))

    ext = RfiMaskExtractor(B, F, T)
    clip = ExamplePythonTransform(B, F, T, sigma=3.0)
    dst = _planted(cp, ext, m)
    tag = f'plain (B,F,T)=({B},{F},{T}) m={m}'

    (gi, gw) = _run(cp, Pipeline([clip, ext]), intensity, weights)
    (gi2, gw2) = _run(cp, Pipeline([clip]), intensity, weights)
    assert np.array_equal(gi, gi2) and np.array_equal(gw, gw2), f'{tag}: the extractor modified the data'
    assert (gw == 0).sum() > (weights == 0).sum() or (weights == 0).all(), f'{tag}: the clipper masked nothing'

    got = cp.asnumpy(dst)
    want = packed_reference(gw, m)
    bad = (got != want)
    assert not bad.any(), f'{tag}: {int(bad.sum())} of {bad.size} mask bytes differ'
    assert ext.rfi_mask is dst

    if verbose:
        atomic_print(f'    test_rfi_mask_extractor: {tag}: ok')


def _check_sub_pipeline(cp, rng, verbose):
    """An extractor at the end of an RfiMaskPipeline: the mask is determined by the
    full-resolution output (see the module docstring)."""

    B = int(rng.integers(1, 3))
    Df = int(rng.choice([1, 2, 4]))
    Dt = int(rng.choice([1, 2])) if (Df == 1) else int(rng.choice([1, 2]))
    if (Df, Dt) == (1, 1):
        Dt = 2
    Fd = 32 * int(rng.integers(1, 3))
    m = int(rng.choice([1, 2, 4]))
    Td = 1024 * m
    (F, T) = (Fd * Df, Td * Dt)

    intensity = (100.0 + rng.standard_normal((B, F, T))).astype(np.float32)
    intensity[rng.uniform(size=(B, F, T)) < 0.01] += 50.0
    weights = random_weights(rng, (B, F, T))

    ext = RfiMaskExtractor(B, Fd, Td)
    inner = [ExamplePythonTransform(B, Fd, Td, sigma=3.0), ext]
    p = Pipeline([RfiMaskPipeline(inner, Df, Dt, 0.0)])
    assert p.get_mask_extractor() is ext
    dst = _planted(cp, ext, m)
    tag = f'sub-pipeline (B,F,T)=({B},{F},{T}) (Df,Dt)=({Df},{Dt}) m={m}'

    (gi, gw) = _run(cp, p, intensity, weights)
    assert np.array_equal(gi, intensity), f'{tag}: RfiMaskPipeline modified the full-resolution intensity'

    cells = (gw > 0).reshape(B, Fd, Df, Td, Dt).any(axis=(2, 4)).astype(np.float32)   # (B, Fd, Td)
    want = packed_reference(cells, m)
    got = cp.asnumpy(dst)
    bad = (got != want)
    assert not bad.any(), f'{tag}: {int(bad.sum())} of {bad.size} mask bytes differ'

    if verbose:
        atomic_print(f'    test_rfi_mask_extractor: {tag}: ok')


def _check_arguments(cp, rng):
    """Planting rules, the unplanted launch, and replanting with a different window count."""

    (B, F, T) = (1, 8, 2048)
    ext = RfiMaskExtractor(B, F, T)
    intensity = cp.zeros((B, F, T), dtype=cp.float32)
    weights = cp.asarray(random_weights(rng, (B, F, T)))

    _expect_raise(ValueError, RfiMaskExtractor, B, F, 1000)                       # ntime % 1024
    _expect_raise(RuntimeError, ext.launch, intensity, weights, None)              # nothing planted
    assert ext.rfi_mask is None

    ok = lambda shape: cp.zeros(shape, dtype=cp.uint8)
    _expect_raise(TypeError, ext.set_rfi_mask, np.zeros((1, B, F, T // 8), dtype=np.uint8))   # host
    _expect_raise(TypeError, ext.set_rfi_mask, cp.zeros((1, B, F, T // 8), dtype=cp.int8))    # dtype
    _expect_raise(ValueError, ext.set_rfi_mask, ok((B, F, T // 8)))                            # 3-d
    _expect_raise(ValueError, ext.set_rfi_mask, ok((1, B, F, T // 8 + 1)))                     # shape
    _expect_raise(ValueError, ext.set_rfi_mask, ok((3, B, F, T // 24)))                        # 2048 % 3
    _expect_raise(ValueError, ext.set_rfi_mask, ok((4, B, F, T // 32)))                        # 512-sample windows
    _expect_raise(ValueError, ext.set_rfi_mask, ok((1, B, F, T // 4))[:, :, :, ::2])           # not contiguous
    assert ext.rfi_mask is None

    # Plant one window, launch; replant as two windows, launch: same bytes, rearranged.
    d1 = _planted(cp, ext, 1)
    ext.launch(intensity, weights, None)
    d2 = _planted(cp, ext, 2)
    ext.launch(intensity, weights, None)
    cp.cuda.get_current_stream().synchronize()
    w = cp.asnumpy(weights)
    assert np.array_equal(cp.asnumpy(d1), packed_reference(w, 1))
    assert np.array_equal(cp.asnumpy(d2), packed_reference(w, 2))

    ext.set_rfi_mask(None)
    assert ext.rfi_mask is None
    _expect_raise(RuntimeError, ext.launch, intensity, weights, None)


def _check_get_mask_extractor():
    """None for a leaf that is not one, the unique one for a container at any depth, self for
    the extractor, ValueError for two."""

    (B, F, T) = (1, 64, 2048)
    ex = lambda: ExamplePythonTransform(B, F, T)
    ext = RfiMaskExtractor(B, F, T)

    assert ex().get_mask_extractor() is None
    assert GpuBadChannelMask(B, F, T, [(500.0, 520.0)], (400.0, 800.0)).get_mask_extractor() is None
    assert ext.get_mask_extractor() is ext
    assert Pipeline([ex(), ex()]).get_mask_extractor() is None
    assert Pipeline([ex(), ext]).get_mask_extractor() is ext
    assert Pipeline([Pipeline([ex(), ext]), ex()]).get_mask_extractor() is ext

    inner_ext = RfiMaskExtractor(B, F // 2, T)
    sub = RfiMaskPipeline([ExamplePythonTransform(B, F // 2, T), inner_ext], 2, 1)
    assert sub.get_mask_extractor() is inner_ext
    assert Pipeline([ex(), sub, ex()]).get_mask_extractor() is inner_ext

    _expect_raise(ValueError, Pipeline([ext, Pipeline([RfiMaskExtractor(B, F, T)])]).get_mask_extractor)
    _expect_raise(ValueError, Pipeline([sub, ext]).get_mask_extractor)


def _check_yaml():
    """The yaml form is a bare class_name, and survives a round trip inside a sub-pipeline."""

    (B, F, T) = (1, 64, 2048)
    inner = [ExamplePythonTransform(B, F // 2, T), RfiMaskExtractor(B, F // 2, T)]
    p = Pipeline([RfiMaskPipeline(inner, 2, 1, 0.0), ExamplePythonTransform(B, F, T)])

    d = p.to_yaml_dict()
    assert d['transforms'][0]['transforms'][-1] == {'class_name': 'RfiMaskExtractor'}, d

    p2 = Pipeline.from_yaml_dict(d, B, F, T)
    assert p2.to_yaml_dict() == d
    e2 = p2.get_mask_extractor()
    assert isinstance(e2, RfiMaskExtractor) and (e2.nbeams, e2.nfreq, e2.ntime) == (B, F // 2, T)

    _expect_raise(ValueError, RfiMaskExtractor.from_yaml_dict, {'class_name': 'RfiMaskExtractor', 'x': 1}, B, F, T)
    _expect_raise(ValueError, RfiMaskExtractor.from_yaml_dict, {'class_name': 'Pipeline'}, B, F, T)


def _legacy_clipper():
    return {'class_name': 'std_dev_clipper', 'nt_chunk': 1024, 'axis': 'AXIS_TIME',
            'sigma': 3.0, 'Df': 1, 'Dt': 1, 'two_pass': True}


def _legacy_counter(where):
    return {'class_name': 'mask_counter', 'nt_chunk': 1024, 'where': where}


def _legacy_sub(elements):
    return {'class_name': 'wi_sub_pipeline', 'Df': 2, 'Dt': 0, 'nfreq_out': 0, 'nds_out': 1,
            'w_cutoff': 0.0, 'sub_pipeline': {'class_name': 'pipeline', 'name': 'p', 'elements': elements}}


def _check_legacy_json():
    """The last mask_counter in document order becomes the extractor, at its own level; the
    others are skipped; a chain without one has no extractor; the caller's dict is untouched."""

    (B, F, T) = (1, 64, 1024)     # T == the clippers' nt_chunk, which GpuStdDevClipper requires

    # The production shape: a counter at the head of the inner chain and one at its tail.
    j = {'class_name': 'pipeline', 'name': 'p',
         'elements': [_legacy_sub([_legacy_counter('before'), _legacy_clipper(), _legacy_counter('after')]),
                      _legacy_clipper()]}
    before = repr(j)
    chain = legacy_chain_from_json(j, B, F, T)
    assert repr(j) == before, 'legacy_chain_from_json modified its argument'
    ext = chain.get_mask_extractor()
    assert isinstance(ext, RfiMaskExtractor) and (ext.nbeams, ext.nfreq, ext.ntime) == (B, F // 2, T)
    sub = chain.transforms[0]
    assert isinstance(sub, RfiMaskPipeline) and (sub.transforms[-1] is ext)
    assert len(sub.transforms) == 2, 'the leading mask_counter should have been skipped'
    assert LEGACY_EXTRACT_KEY not in j['elements'][0]['sub_pipeline']['elements'][-1]

    # The last one at the top level, after the sub-pipeline: full geometry.
    j = {'class_name': 'pipeline', 'name': 'p',
         'elements': [_legacy_sub([_legacy_clipper(), _legacy_counter('inner')]), _legacy_counter('outer')]}
    ext = legacy_chain_from_json(j, B, F, T).get_mask_extractor()
    assert isinstance(ext, RfiMaskExtractor) and (ext.nbeams, ext.nfreq, ext.ntime) == (B, F, T)

    # None at all.
    j = {'class_name': 'pipeline', 'name': 'p', 'elements': [_legacy_clipper()]}
    assert legacy_chain_from_json(j, B, F, T).get_mask_extractor() is None

    # A lone mask_counter at the top level is the whole chain.
    assert isinstance(legacy_chain_from_json(_legacy_counter('x'), B, F, T), RfiMaskExtractor)


# -------------------------------------------------------------------------------------------------


def test_rfi_mask_extractor(iteration=0, rng=None, verbose=False):
    """One extractor in a plain Pipeline and one inside an RfiMaskPipeline, both at random
    geometry, against the two oracles in the module docstring; on iteration 0, also the
    planting rules, get_mask_extractor(), the yaml form, and the legacy json rule."""

    try:
        import cupy as cp
    except ImportError:
        if verbose:
            atomic_print('    test_rfi_mask_extractor: cupy not available, skipped')
        return

    rng = _default_rng(rng)

    _check_plain(cp, rng, verbose)
    _check_sub_pipeline(cp, rng, verbose)

    if iteration == 0:
        _check_arguments(cp, rng)
        _check_get_mask_extractor()
        _check_yaml()
        _check_legacy_json()
