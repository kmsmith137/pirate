#!/usr/bin/env python3
"""Spot test: the std_dev_clipper, pirate vs rf_kernels.

Compares pirate_frb.chimefrb.ReferenceStdDevClipper against rf_kernels::std_dev_clipper, the
most numerous transform in the old CHIME FRB search's RFI chain (60 of its 120 nodes).
Stage 2 of the transform, std_dev_clipper::_clip_1d(), is also checked on its own, on
constructed inputs, by misc/chimefrb/rfi_std_dev_clip_1d/: the more important of the two
checks, since that function has no reference implementation anywhere in the old code.

The checks here follow the intensity clipper's: the primary one feeds the OLD kernel's own
stage-1 variances into our stage 2 and apply, so only stage 2 and the apply are under test;
the secondary one runs our whole reference, conditioned on the old kernel's stage-1
decisions. Conditioning rather than bracketing, because a stage-1 decision changes the
population stage 2 sees, so bracketing it does not bracket stage 2.

Needs a built oldpipe (misc/chimefrb/build_oldpipe.sh). Run me directly, or through
misc/chimefrb/run_spot_tests.py.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import harness

from pirate_frb.chimefrb import AXIS_FREQ, AXIS_TIME, ReferenceStdDevClipper, std_dev_apply
from pirate_frb.chimefrb.test_std_dev_clipper import (stage2_bracket, end_to_end_bracket,
                                                      ill_conditioned, keep_bracket)

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 137

# Small on purpose, and legal for both codes: rf_kernels needs nfreq % 8 == 0 and
# nt_chunk % 8 == 0, pirate needs F and T divisible by 32.
NFREQ = 128
NT = 512
SIGMA = 3.0          # the production value
AXIS_NAMES = {AXIS_FREQ: "FREQ", AXIS_TIME: "TIME"}


def make_input(rng):
    """A (2, F, T) float32 array: arr[0] = intensity, arr[1] = weights.

    Designed around stage 2, which sees VARIANCES: the injected RFI is rows with the wrong
    noise level -- a few channels and a few time samples with 5x the noise -- rather than
    bright samples, which a clean variance estimate barely notices. Kept to a few percent,
    since outliers inflate the very spread they are measured against. Plus the three ways to
    make a variance invalid (constant, fully masked, nearly constant at a large mean), in
    both directions, so each axis sees some.
    """

    offset, scale = 100.0, 10.0
    intensity = rng.normal(offset, scale, size=(NFREQ, NT))
    weights = (rng.uniform(size=(NFREQ, NT)) < 0.8) * rng.uniform(0.5, 1.5, size=(NFREQ, NT))

    for f in rng.choice(NFREQ, size=4, replace=False):
        intensity[f, :] = rng.normal(offset, 5.0 * scale, size=NT)
    for s in rng.choice(NT, size=8, replace=False):
        intensity[:, s] = rng.normal(offset, 5.0 * scale, size=NFREQ)

    intensity[20:22, :] = offset                                            # constant
    weights[22:24, :] = 0.0                                                 # masked
    intensity[24:26, :] = offset * (1.0 + 1.0e-6 * rng.normal(size=(2, NT)))  # on the cutoff
    intensity[:, 100:102] = offset
    weights[:, 102:104] = 0.0

    return np.stack([intensity, weights]).astype(np.float32)


def make_one_valid(rng, axis):
    """An input in which exactly ONE row has any weight, so the whole-beam branch runs."""

    x = make_input(rng)
    x[1] = 0.0
    if axis == AXIS_TIME:
        x[1, 40, :] = 1.0
    else:
        x[1, :, 300] = 1.0
    return x


def run_old(x, axis, two_pass):
    """One run of the old std_dev_clipper on x: (clipped weights, clipped variances, wrms).
    The last is (2, nrows), mean and rms, from the clipper's stage 1."""

    return harness.run_driver(HERE, x, dtype=np.float32, noutputs=3, params={
        "axis": axis, "sigma": SIGMA, "Df": 1, "Dt": 1, "two_pass": int(two_pass)})


def check_transform(t, label, x, axis, two_pass):
    (w_old, vclip_old, wrms) = run_old(x, axis, two_pass)
    vclip_old = vclip_old.astype(np.float64)
    v_old = wrms[1].astype(np.float64) ** 2         # the driver reports rms

    W = x[1].astype(np.float64)[None]               # (1, F, T): the reference has a beam axis
    I = x[0].astype(np.float64)[None]
    v1 = v_old[None]                                # (1, nrows)

    nkill = int(np.sum((v_old > 0) & (vclip_old == 0)))
    t.note("    %s: old stage 1 valid in %d of %d rows; old stage 2 clips %d of them"
           % (label, int(np.sum(v_old > 0)), v_old.size, nkill))

    if ill_conditioned(v1)[0]:
        t.note("        ill-conditioned (all valid variances equal); skipped")
        return

    # Primary: the OLD stage-1 variances, through our stage 2 and apply. rms^2 is within an
    # ulp or two of the variance the clipper used, and exactly zero where that is.
    d = stage2_bracket(v1, SIGMA)
    (klo, khi) = keep_bracket(v1, SIGMA, d)
    t.check_sandwich("%s stage 2 rows" % label, (vclip_old[None] != 0), klo, khi,
                     why="rows surviving stage 2, given the old stage-1 variances;"
                         " sigma bracket %.2g" % d)
    wlo = std_dev_apply(W, klo.astype(np.float64), axis, 1, 1)
    whi = std_dev_apply(W, khi.astype(np.float64), axis, 1, 1)
    t.check_sandwich("%s weights" % label, w_old[None], wlo, whi,
                     why="the old clipper's weights, against our stage 2 + apply on its"
                         " stage-1 variances")

    # Secondary: our whole reference, conditioned on the old stage-1 decisions.
    (_, v_c) = ReferenceStdDevClipper(axis, SIGMA, 1, 1, two_pass, eps_multiplier=1.5).variances(I, W)
    (m_p, v_p) = ReferenceStdDevClipper(axis, SIGMA, 1, 1, two_pass, eps_multiplier=0.5).variances(I, W)
    t.check_sandwich("%s stage 1 validity" % label, (v1 > 0), (v_c > 0), (v_p > 0),
                     why="a variance is valid or not; eps_multiplier bracket 1.5 / 0.5")

    v_cond = np.where(v1 > 0, v_p, 0.0)
    L = NT if (axis == AXIS_TIME) else NFREQ
    dB = end_to_end_bracket(v_cond, m_p, L, two_pass, SIGMA)
    (klo, khi) = keep_bracket(v_cond, SIGMA, dB)
    wlo = std_dev_apply(W, klo.astype(np.float64), axis, 1, 1)
    whi = std_dev_apply(W, khi.astype(np.float64), axis, 1, 1)
    t.check_sandwich("%s end-to-end" % label, w_old[None], wlo, whi,
                     why="our reference's variances, conditioned on the old stage-1 valid"
                         " set; sigma bracket %.2g" % dB)


def main():
    t = harness.Test("rfi_std_dev_clipper")

    manifest = harness.build_manifest()
    if manifest:
        for line in manifest.splitlines():
            if line.startswith("date:") or line.startswith("arch:"):
                t.note(line.strip())

    rng = np.random.default_rng(SEED)

    x = make_input(rng)
    t.note("(2, %d, %d) intensity/weights from seed %d" % (NFREQ, NT, SEED))

    for axis in (AXIS_TIME, AXIS_FREQ):
        for two_pass in (True, False):
            tag = "%s tp=%d" % (AXIS_NAMES[axis], int(two_pass))
            check_transform(t, tag, x, axis, two_pass)
            check_transform(t, tag + " 1-valid", make_one_valid(rng, axis), axis, two_pass)

    return t.done()


if __name__ == "__main__":
    sys.exit(main())
