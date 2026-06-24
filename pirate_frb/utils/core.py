"""Core pirate_frb utility functions (small, general-purpose helpers).

Currently just bit-reversal; more "core" utils can be added here later.
"""

import functools
import numpy as np


@functools.cache
def bit_reverse_permutation(rank):
    """
    Returns the length-2^rank int permutation perm[d] = bit_reverse(d, rank).

    Indexing a ReferenceTree output's (bit-reversed) delay axis with this
    permutation restores natural delay order.

    Cached, so repeated calls with the same 'rank' share one array. The result
    is marked read-only so the shared array can't be corrupted by a caller.
    """
    n = 1 << rank
    perm = np.zeros(n, dtype=np.intp)
    for d in range(n):
        x, b = d, 0
        for _ in range(rank):
            b = (b << 1) | (x & 1)
            x >>= 1
        perm[d] = b
    perm.flags.writeable = False
    return perm
