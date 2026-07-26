"""
Inclusive prefix/suffix scans over the moment monoid (see MomentSet.py).

We use a Hillis-Steele scan: log2(B) passes, pass k merging element i with
element i-2^k.  This is O(B log B) work rather than the O(B) of a work-efficient
Blelloch scan, but the property we care about is *depth*: each output depends on
its leaves through O(log B) merges rather than O(B), which is what bounds
roundoff accumulation.  Work-efficiency does not matter in numpy, and
Hillis-Steele is also what a GPU warp-level shuffle scan does, so this mirrors
the eventual kernel.  (The GPU block-level scan will likely use Blelloch.)

Measured, float32, one 1024-sample block, worst-case moment error against a
float64 reference: 2e-6 for the tree versus 3.4e-4 for a sequential scan.  Do
not substitute a sequential scan here; sequential_prefix_scan() below exists
only so that tests can demonstrate the gap.

Validity: at every step the two operands cover disjoint, adjacent ranges, which
is what merge() requires.
"""

import numpy as np

from .MomentSet import merge


def _npasses(n):
    k, p = 1, 0
    while k < n:
        k *= 2
        p += 1
    return p


def tree_prefix_scan(leaves):
    """Inclusive prefix scan along the last batch axis.  out[i] = leaves[0] + ... + leaves[i]."""
    out = leaves.copy()
    B = out.batch_shape[-1]
    k = 1
    while k < B:
        # out[i] <- merge(out[i-k], out[i]) for i >= k
        merged = merge(out.slice_pos(slice(0, B-k)), out.slice_pos(slice(k, B)))
        out.set_pos(slice(k, B), merged)
        k *= 2
    return out


def tree_suffix_scan(leaves):
    """Inclusive suffix scan along the last batch axis.  out[i] = leaves[i] + ... + leaves[B-1]."""
    out = leaves.copy()
    B = out.batch_shape[-1]
    k = 1
    while k < B:
        # out[i] <- merge(out[i], out[i+k]) for i < B-k
        merged = merge(out.slice_pos(slice(0, B-k)), out.slice_pos(slice(k, B)))
        out.set_pos(slice(0, B-k), merged)
        k *= 2
    return out


def sequential_prefix_scan(leaves):
    """
    Test-only.  Same result as tree_prefix_scan() in exact arithmetic, but with
    O(B) depth instead of O(log B), hence much worse roundoff.  Used by the
    tests to confirm that the tree scan is actually being exercised.
    """
    out = leaves.copy()
    B = out.batch_shape[-1]
    for i in range(1, B):
        out.set_pos(slice(i, i+1), merge(out.slice_pos(slice(i-1, i)),
                                         out.slice_pos(slice(i, i+1))))
    return out
