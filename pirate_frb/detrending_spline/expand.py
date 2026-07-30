"""
Mask expansion: whole zones only.

The regularized detrender needs far less of this than an unregularized one.  With
eta > 0 there are no dead coefficients to deflate and no contamination to chase
through the connected components of G's zero pattern, so the only expansion rule
is: if a zone's r_min falls below eps, mask that zone entirely.  Zones are exactly
decoupled (a zone boundary is an interior knot of multiplicity n_phi+1, which
makes both G and D_1 exactly block diagonal), so flagging one zone can never
affect another.

In practice the rule almost never fires -- measured over adversarially constructed
masks at n_phi=2, F up to 30000 and K up to 10, the smallest r_min seen is about
2x eps, and the configurations that approach it need a single knot interval tens
of thousands of channels wide.  It is a guardrail, not a routine code path, and
tests have to construct masks adversarially to exercise it at all (see masks.py).
"""

import numpy as np


def zone_channel_ranges(kv):
    """
    [(lo, hi)] channel range of each zone, half-open, covering [0, nfreq).

    A channel belongs to the zone of any basis function nonzero there; those all
    share a zone because no basis function straddles a zone boundary.
    """
    zone_of_channel = kv.zone_id[kv.j0]
    edges = np.flatnonzero(np.diff(zone_of_channel)) + 1
    los = np.concatenate(([0], edges))
    his = np.concatenate((edges, [kv.nfreq]))
    return [(int(lo), int(hi)) for lo, hi in zip(los, his)]


def expand_mask(mask, bad, kv):
    """
    mask: (M, nfreq, ntime) bool.  bad: (M, ntime, nzone) bool.

    Returns a new mask with every channel of every flagged zone cleared.  The
    input is not modified.
    """
    mask = np.asarray(mask) != 0
    out = mask.copy()
    for z, (lo, hi) in enumerate(zone_channel_ranges(kv)):
        out[:, lo:hi, :] &= ~bad[:, None, :, z]
    return out
