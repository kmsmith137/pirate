#!/usr/bin/env python3
"""Spot test: badchannel_mask, pirate vs rf_pipelines.

Compares pirate_frb.chimefrb.badchannel_keep() -- the conversion from MHz ranges to channels
that GpuBadChannelMask.from_mask_ranges() uses -- against rf_pipelines::badchannel_mask, run
through the real pipeline machinery.

WHY THIS TEST MATTERS. The old code has no reference implementation of this arithmetic: its
python transform in rf_pipelines/retirement_home follows different rules, and there is no C++
test of it. So badchannel_keep() is a transcription of _bind_transform() checked against
nothing else. The routine unit test compares it with a second statement of the same rule,
which catches a slip in either but not a misreading shared by both. This test reads the old
code by running it.

It also pins the quirks that badchannel_keep() reproduces deliberately (see its docstring):
a range ending at the bottom of the band masks the bottom channel, while one starting at the
top masks nothing; a range covering the whole band is refused; a range narrower than 1e-3 of
a channel can mask nothing.

TOLERANCE: none. The output is a set of channel indices, and both codes compute it from the
same double-precision expressions. The one way they could legitimately differ is roundoff at
a floor()/ceil() boundary -- the old code is built with -ffast-math and may evaluate
factor - x*scale as a fused multiply-add -- which needs a range end within ~1e-13 channels of
an integer +/- 1e-3. The random draws skip anything within 1e-9 (near_fudge_boundary()).

NOT COVERED HERE: the kernel, GpuBadChannelMask, which does no arithmetic. The unit test
compares it bitwise with the numpy reference.

Needs a built oldpipe (misc/chimefrb/build_oldpipe.sh). Run me directly, or through
misc/chimefrb/run_spot_tests.py.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import harness

from pirate_frb.chimefrb import badchannel_keep
from pirate_frb.chimefrb.test_badchannel_mask import (PRODUCTION_MASK_RANGES,
                                                      PRODUCTION_MASKED_RUNS_1024,
                                                      near_fudge_boundary, random_range_case)

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(os.path.dirname(HERE), "configs",
                      "21-03-07-low-latency-uniform-badchannel-mask-noplot.json")

# Any fixed value works; it is written down so the comparison is reproducible.
SEED = 137

# The mask does not depend on time; a few columns are enough to see that each row is constant.
NT_CHUNK = 16

# Random draws: mostly one range each, so that one range's mistake cannot hide under another's.
NDRAW_SINGLE = 300
NDRAW_MULTI = 40

WHY_EXACT = ("channel sets, from the same double-precision expressions in both codes;"
             " exact agreement is the only right answer")


def run_old(ranges, nfreq, flo, fhi, catch=False):
    """The old code's keep array (uint8, length nfreq), or None if it refused the input.

    Also checks what only the old side can get wrong: every row of the output must be all
    zeros or all ones.
    """

    x = np.array(ranges, dtype=np.float64).reshape(-1, 2)
    out = harness.run_driver(HERE, x, params={
        "nfreq": nfreq, "freq_lo_MHz": float(flo), "freq_hi_MHz": float(fhi),
        "nt_chunk": NT_CHUNK, "catch": int(catch)})

    if out.size == 0:
        return None

    assert out.shape == (nfreq, NT_CHUNK)
    assert np.all((out == 0) | (out == 1)), "driver output is not 0/1"
    assert np.all(out == out[:, :1]), "the old code's mask is not constant along time"
    return (out[:, 0] != 0).astype(np.uint8)


def new_or_none(ranges, nfreq, flo, fhi):
    try:
        return badchannel_keep(ranges, nfreq, flo, fhi)
    except ValueError:
        return None


def config_ranges():
    """The mask_ranges of the one badchannel_mask node in the production config."""

    found = []

    def walk(o):
        if isinstance(o, dict):
            if o.get("class_name") == "badchannel_mask":
                found.append(o)
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    with open(CONFIG) as f:
        walk(json.load(f))

    assert len(found) == 1, "expected one badchannel_mask in %s" % CONFIG
    assert found[0]["mask_path"] == ""
    return [tuple(r) for r in found[0]["mask_ranges"]]


def check_production(t):
    t.check_allclose("production ranges", np.array(PRODUCTION_MASK_RANGES), np.array(config_ranges()),
                     rtol=0.0, why="test_badchannel_mask.py's copy against the config file itself")

    for nfreq in (1024, 16384):
        old = run_old(PRODUCTION_MASK_RANGES, nfreq, 400.0, 800.0)
        new = badchannel_keep(PRODUCTION_MASK_RANGES, nfreq, 400.0, 800.0)
        t.note("production mask at %d channels: the old code masks %d" % (nfreq, int((old == 0).sum())))
        t.check_allclose("production %d" % nfreq, new, old, rtol=0.0, why=WHY_EXACT)

    pinned = np.ones(1024, dtype=np.uint8)
    for (a, b) in PRODUCTION_MASKED_RUNS_1024:
        pinned[a:b] = 0
    t.check_allclose("production 1024 pin", pinned, run_old(PRODUCTION_MASK_RANGES, 1024, 400.0, 800.0),
                     rtol=0.0, why="the runs test_badchannel_mask.py pins, against the old code")


def check_quirks(t):
    """The documented corners, as named cases: the old code's answer, stated outright."""

    scale = 1024 / 400.0
    cases = [
        ("bottom-edge touch", [(390.0, 400.0)], [1023]),
        ("top-edge touch", [(800.0, 810.0)], []),
        ("whole band as inside", [(400.0, 800.0)], list(range(1024))),
        ("channel-aligned", [(800.0 - 7/scale, 800.0)], list(range(7))),
        # Channel coordinates [5.0002, 5.0008]: narrower than the fudge, at a channel edge.
        ("sub-fudge at an edge", [(800.0 - 5.0008/scale, 800.0 - 5.0002/scale)], []),
        # Channel coordinates [5.3, 5.3005]: just as narrow, but clear of both edges.
        ("sub-fudge mid-channel", [(800.0 - 5.3005/scale, 800.0 - 5.3/scale)], [5]),
    ]

    for (label, ranges, want) in cases:
        old = run_old(ranges, 1024, 400.0, 800.0)
        new = badchannel_keep(ranges, 1024, 400.0, 800.0)
        expected = np.ones(1024, dtype=np.uint8)
        expected[want] = 0
        t.check_allclose(label + " (old)", old, expected, rtol=0.0, why="the documented behaviour")
        t.check_allclose(label + " (new)", new, old, rtol=0.0, why=WHY_EXACT)


def check_refusals(t):
    """Inputs both codes must refuse. (lo >= hi fails in the old constructor; the others in bind.)"""

    cases = [
        ("whole band", [(390.0, 810.0)]),
        ("below the band", [(300.0, 350.0)]),
        ("above the band", [(850.0, 900.0)]),
        ("lo == hi", [(500.0, 500.0)]),
        ("lo > hi", [(501.0, 500.0)]),
        ("one bad among good", [(410.0, 420.0), (850.0, 900.0)]),
    ]

    refused_old = [run_old(r, 1024, 400.0, 800.0, catch=True) is None for (_, r) in cases]
    refused_new = [new_or_none(r, 1024, 400.0, 800.0) is None for (_, r) in cases]
    t.note("refusal cases: " + ", ".join(label for (label, _) in cases))
    t.check_allclose("old code refuses", np.array(refused_old), np.ones(len(cases)), rtol=0.0,
                     why="rf_pipelines::badchannel_mask throws on each of these")
    t.check_allclose("pirate refuses", np.array(refused_new), np.ones(len(cases)), rtol=0.0,
                     why="badchannel_keep() raises ValueError on each of these")


def check_random(t, rng, ndraw, nmax, label):
    old_all, new_all = [], []
    nskip, first_bad = 0, None

    for _ in range(ndraw):
        (ranges, kinds, nfreq, flo, fhi) = random_range_case(rng, nmax=nmax)
        if near_fudge_boundary(ranges, nfreq, flo, fhi):
            nskip += 1
            continue

        old = run_old(ranges, nfreq, flo, fhi)
        new = badchannel_keep(ranges, nfreq, flo, fhi)
        if (first_bad is None) and not np.array_equal(old, new):
            first_bad = (ranges, kinds, nfreq, flo, fhi)
        old_all.append(old)
        new_all.append(new)

    t.note("%s: %d draws (%d skipped near a fudge boundary), %d channels in all"
           % (label, ndraw, nskip, sum(len(x) for x in old_all)))
    ok = t.check_allclose(label, np.concatenate(new_all), np.concatenate(old_all), rtol=0.0,
                          why=WHY_EXACT)
    if not ok:
        t.note("first disagreeing draw: ranges=%r kinds=%r nfreq=%d band=(%r, %r)" % first_bad)


def main():
    t = harness.Test("rfi_badchannel_mask")

    manifest = harness.build_manifest()
    if manifest:
        for line in manifest.splitlines():
            if line.startswith("date:") or line.startswith("arch:"):
                t.note(line.strip())

    rng = np.random.default_rng(SEED)

    check_production(t)
    check_quirks(t)
    check_refusals(t)
    check_random(t, rng, NDRAW_SINGLE, 1, "random, one range")
    check_random(t, rng, NDRAW_MULTI, 10, "random, up to ten ranges")

    return t.done()


if __name__ == "__main__":
    sys.exit(main())
