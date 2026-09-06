"""
The knot vector behind the spline detrender's frequency basis.

Everything about the frequency axis that does not depend on the data is derived
here, once, from a caller-supplied integer array.  The array is a plain
non-decreasing list of channel indices, with multiplicity expressed by
repetition; there is no separate "zone" concept anywhere in the code, because a
zone boundary IS an interior knot of multiplicity n_phi+1 (it makes the Gram
matrix exactly block diagonal there, and the fits on the two sides decouple).

Convention: channel f occupies [f, f+1) and its data sits at f + 1/2, so a knot
value k is the boundary *below* channel k and no data point ever coincides with
a knot.  Consequences, which are why this convention was chosen over putting
the data at integers:

  - knot values are channel indices, so a zone spanning channels [a,b) is
    clamped by the integers a and b with no offset;
  - the band is [0, nfreq], and the interval test is the ordinary half-open
    integer one, knots[i] <= f < knots[i+1];
  - knots[i+1] - knots[i] is literally a channel count.

Validation is strict and happens in the constructor, since the array comes from
the caller.  The end-multiplicity rule (exactly n_phi+1 copies of 0 and of
nfreq) is the one that is not merely stylistic: it is what puts the constant
function in the span, hence what would make a future offset subtraction exactly
inert, and reducing it does not degrade that gracefully -- it destroys it (the
best fit to the constant 1 is off by 0.99 at end multiplicity n_phi).  It is
also what makes the basis complete on the whole band rather than on the interior
knot span only.

For the regularized detrender the same rule carries a second, sharper
consequence.  Clamped ends make the basis a partition of unity ON EACH ZONE,
sum_{j in zone} phi_j(f) = 1 exactly, so the all-ones coefficient vector IS the
constant function.  The regulator's null space is exactly that vector
(regulator.py), which is why a constant baseline is removed EXACTLY rather than
shrunk, and why one unmasked channel suffices to make the zone's matrix positive
definite.  test_basis() and test_flat_baseline_exact() both depend on it.
"""

import numpy as np


class KnotVector:
    """
    A validated knot vector plus the per-channel and per-basis-function lookups
    derived from it.

    Constructor arguments: 'knots' (integer array, non-decreasing), 'n_phi' (the
    spline degree) and 'nfreq'.  Raises ValueError on anything invalid.

    Attributes:

      knots        int64 array of length M
      n_phi        spline degree
      nfreq        number of channels
      N_phi        number of basis functions, = M - n_phi - 1
      nspan        number of knot intervals, = M - 1  (some may be empty)
      j0[f]        int array of length nfreq: the knot-span index of channel f.
                   The basis functions nonzero at f are exactly
                   j in [j0[f]-n_phi, j0[f]], which is checked against the
                   evaluated basis by test_knots().
      supp_lo[j], supp_hi[j]   channel range of supp(phi_j), half-open
      zone_id[j]   index of the block of G that phi_j belongs to
      nzone        number of blocks
    """

    def __init__(self, knots, n_phi, nfreq):
        knots = np.asarray(knots)
        n_phi = int(n_phi)
        nfreq = int(nfreq)

        if n_phi < 0:
            raise ValueError(f'KnotVector: n_phi={n_phi} must be >= 0')
        if nfreq < 1:
            raise ValueError(f'KnotVector: nfreq={nfreq} must be >= 1')
        if knots.ndim != 1:
            raise ValueError(f'KnotVector: knots must be 1-d, got shape {knots.shape}')
        # Integer representation is load-bearing, not a convenience: every check
        # below (and the multiplicity count in particular) is an exact '==' test.
        if not np.issubdtype(knots.dtype, np.integer):
            raise ValueError(f'KnotVector: knots must be an integer array, got '
                             f'dtype {knots.dtype} (multiplicity is counted exactly)')
        knots = knots.astype(np.int64)

        if np.any(np.diff(knots) < 0):
            raise ValueError('KnotVector: knots must be non-decreasing')
        if knots[0] != 0 or knots[-1] != nfreq:
            raise ValueError(f'KnotVector: knots must run from 0 to nfreq={nfreq}, '
                             f'got [{knots[0]}, {knots[-1]}]')

        for val, what in ((0, 'first'), (nfreq, 'last')):
            mult = int(np.count_nonzero(knots == val))
            if mult != n_phi + 1:
                raise ValueError(f'KnotVector: the {what} knot ({val}) has multiplicity '
                                 f'{mult}, expected exactly n_phi+1 = {n_phi+1}.  Clamped '
                                 f'ends are what put the constant function in the span '
                                 f'and make the basis complete on the whole band.')

        interior = knots[(knots > 0) & (knots < nfreq)]
        if interior.size:
            vals, mults = np.unique(interior, return_counts=True)
            bad = vals[mults > n_phi + 1]
            if bad.size:
                raise ValueError(f'KnotVector: interior knot(s) {bad.tolist()} have '
                                 f'multiplicity above n_phi+1 = {n_phi+1}')

        N_phi = len(knots) - n_phi - 1
        if N_phi < 1:
            raise ValueError(f'KnotVector: N_phi = len(knots)-n_phi-1 = {N_phi} must be >= 1')

        self.knots = knots
        self.n_phi = n_phi
        self.nfreq = nfreq
        self.N_phi = N_phi
        self.nspan = len(knots) - 1

        # The span index of each channel.  'right' is what makes this land on the
        # last knot of a repeated group, hence always on a NON-EMPTY span, which is
        # what the Cox-de Boor recursion in basis.py needs (its denominators are
        # differences of knots straddling knots[j0] < knots[j0+1]).
        self.j0 = np.searchsorted(knots, np.arange(nfreq), side='right') - 1
        assert np.all(self.j0 >= n_phi) and np.all(self.j0 <= N_phi - 1)

        self.supp_lo = knots[:N_phi].copy()
        self.supp_hi = knots[n_phi+1:].copy()

        # A zone boundary is an interior knot of full multiplicity.  No basis
        # function straddles one: phi_j has support [k_j, k_{j+n_phi+1}), and if the
        # boundary occupies knot indices i..i+n_phi then j <= i-1 gives
        # supp_hi <= k_{i+n_phi} = v and j >= i gives supp_lo >= v.  So the zone of
        # phi_j is decided by supp_lo alone.
        if interior.size:
            vals, mults = np.unique(interior, return_counts=True)
            bounds = vals[mults == n_phi + 1]
        else:
            bounds = np.zeros(0, dtype=np.int64)
        self.zone_id = np.searchsorted(bounds, self.supp_lo, side='right')
        self.nzone = len(bounds) + 1

    def __repr__(self):
        return (f'KnotVector(nfreq={self.nfreq}, n_phi={self.n_phi}, '
                f'N_phi={self.N_phi}, nzone={self.nzone}, '
                f'knots={self.knots.tolist()})')

    def support_mask(self, j):
        """Boolean array of length nfreq, true on supp(phi_j).  Interval arithmetic
        only -- test_expand() checks it against the evaluated basis."""
        m = np.zeros(self.nfreq, dtype=bool)
        m[self.supp_lo[j]:self.supp_hi[j]] = True
        return m
