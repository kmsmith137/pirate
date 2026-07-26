"""
The adversarial mask zoo, shared by the tests.

Each entry is (name, mask) with mask of shape (S, T), dtype bool.  The
geometries here are the ones that broke earlier candidate algorithms (see the
appendix of notes/tree_dedispersion.tex): long gaps, one-sided windows, narrow
off-center clusters, and fully masked blocks.
"""

import numpy as np


def mask_zoo(S, T, W, rng, include_empty=True):
    out = []

    def add(name, m):
        out.append((name, np.broadcast_to(np.asarray(m, dtype=bool), (S, T)).copy()))

    add('all valid', np.ones(T))

    for frac in (0.99, 0.50, 0.10, 0.01):
        add(f'random {frac:.2f}', rng.random((S, T)) < frac)

    # One long gap, wider than a full window.
    m = np.ones(T, dtype=bool)
    lo = T//3
    m[lo:lo + min(3*W, T - lo - 1)] = False
    add('long gap', m)

    # Periodic gaps commensurate with the block length and with a chunk.
    for period, lbl in ((2*W, 'period B'), (4*W, 'period 2B')):
        m = np.ones(T, dtype=bool)
        m[::period] = False
        m[1::period] = False
        add(f'periodic gaps, {lbl}', m)

    # One-sided: valid only on the right of the midpoint, so windows straddling
    # it see data on one side only.
    m = np.ones(T, dtype=bool)
    m[:T//2] = False
    add('one-sided', m)

    # Narrow cluster far off-center within its window.
    m = np.zeros(T, dtype=bool)
    ctr = T//2 + int(0.8*W)
    m[max(0, ctr-W//25) : ctr + W//25 + 1] = True
    add('narrow off-center cluster', m)

    # Two clumps: G_ii > 0 for every i, but curvature barely determined.
    m = np.zeros(T, dtype=bool)
    for c in (T//2 - (3*W)//4, T//2 + (3*W)//4):
        m[max(0, c-W//20) : c + W//20 + 1] = True
    add('bimodal clumps', m)

    # A fully masked block (the NaN trap for the empty-set rule).
    m = np.ones(T, dtype=bool)
    m[2*W : 4*W] = False
    add('one fully masked block', m)

    # Very small valid counts inside individual windows.
    for nv in (1, 2, 3):
        m = np.zeros(T, dtype=bool)
        m[::max(1, (2*W)//nv)] = True
        add(f'sparse (about nv={nv} per window)', m)

    if include_empty:
        add('all masked', np.zeros(T))

    return out
