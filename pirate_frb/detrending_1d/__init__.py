"""
Pure-numpy 1-d detrender: a masked, adaptively centered moving local polynomial
fit, evaluated by a van Herk block decomposition over a moment monoid.

The algorithm is specified in notes/tree_dedispersion.tex, section "Detrending".
The appendix of the same document records the alternatives that were considered
and discarded, and the measurements behind the choices made here.

A GPU implementation will follow; this package is the reference that will
validate it.  Everything is parameterized by dtype (float32 or float64) so that
the same instance can be run twice and the results compared.

Deliberate divergences from the eventual GPU version (not bugs):

  - S[...,1] is stored (always zero) rather than omitted, so that the Hankel
    indexing G[j,l] = S[j+l] is trivial.
  - Hillis-Steele scan rather than work-efficient Blelloch: same O(log B) depth,
    more work.  Work does not matter in numpy, and Hillis-Steele is what a GPU
    warp-level shuffle scan does anyway.
  - Full Pref/Suff arrays are materialized; the GPU version will use a two-level
    register-resident decomposition.
  - No offset-tracking state machine beyond a per-chunk constant.
"""

from .MomentSet import MomentSet, merge, pascal_shift
from .scan import tree_prefix_scan, tree_suffix_scan
from .LocalPolyFit import solve, gram, cholesky
from .Detrender import Detrender
from .reference import detrend_reference
from .masks import random_mask, MASK_TYPES
