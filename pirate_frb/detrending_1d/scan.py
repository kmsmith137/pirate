"""
Inclusive prefix/suffix scans over the moment monoid (see MomentSet.py).

We use a Hillis-Steele scan: log2(B) passes, pass k merging element i with
element i-2^k.  This is O(B log B) work rather than the O(B) of a work-efficient
Blelloch scan, but the property we care about is *depth*: each output depends on
its leaves through O(log B) merges rather than O(B), which is what bounds
roundoff accumulation.  Work-efficiency does not matter in numpy, and
Hillis-Steele is also what a GPU warp-level shuffle scan does, so this mirrors
src_lib/Detrender1d.cu, whose block-level scan over 32 sub-block aggregates is
Hillis-Steele as well (over 32 lanes the work difference is irrelevant and the
depth is identical, so there was no reason to reach for Blelloch).

Do not substitute a sequential scan here: it costs about 10x in moment accuracy,
and that degradation is small enough to slip past every tolerance the test suite
asserts.  test_vanherk() therefore checks the structure directly, by counting
merge() calls -- this scan makes log2(B) of them, a sequential one would make
B-1.

Validity: at every step the two operands cover disjoint, adjacent ranges, which
is what merge() requires.
"""

from .MomentSet import merge


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
