import numpy as np

from .PfVariance import PfVariance
from .VarianceMap import VarianceMapBlock, VarianceMapBase
from ..utils import atomic_print


####################################   class VarianceMapExact   ####################################


class VarianceMapExact(VarianceMapBase):
    """Low-rank (SVD) representation of the exact variance map of a PfAvarExact.

    Constructed from an already-built PfAvarExact (see PfVariance.py). Whereas the PfAvarExact
    computes the output variances for one specific input array 'freq_variances', a
    VarianceMapExact represents the whole linear map (input variances) -> (output variances), in
    a compressed form which can be re-evaluated cheaply for any input (see eval() below). This
    matters because in operation the input variances drift (gain drift, RFI excision), and we
    want to recompute peak-finding weights whenever they are updated.

    The map is stored as one truncated SVD per (tree, multiplet) pair -- see VarianceMapBlock,
    and notes/variance_map.tex, appendix "PfAvarExact and VarianceMapExact".

    Note that a VarianceMapExact keeps a reference to the PfAvarExact, which is usually the
    larger of the two objects (check() needs it, and the memory is already committed anyway).

    Caveat: PfAvarExact is an exact-but-slow reference implementation which is only practical at
    moderate ranks, so this class inherits that limitation. It is not usable at CHIME/CHORD
    scale; VarianceMapApproximation is.

    Members
    -------
      tree_M:       length-ntrees array, number of multiplets.
      blocks:       (ntrees, M) ragged array (list of lists) of VarianceMapBlocks.

    plus the members of VarianceMapBase (avar, ntrees, nfreq, tree_r, tree_R, tree_P, epsilon).
    """

    def __init__(self, avar, epsilon=None, progress=False):
        """Build the low-rank representation of an (already constructed) PfAvarExact.

        If epsilon is None (the default), each block chooses its own truncation threshold from
        its matrix sizes; an explicit float applies to all blocks. If progress is set, print one
        line per tree.
        """

        self._init_common(avar, epsilon)
        self.tree_M = np.array([fs.M for fs in avar.tree_fs])

        for itree in range(self.ntrees):
            r, R, P = int(self.tree_r[itree]), int(self.tree_R[itree]), int(self.tree_P[itree])
            fs = avar.tree_fs[itree]
            blocks = [ ]

            for m in range(int(self.tree_M[itree])):
                ifreq_lo = int(avar.per_tm_ifreq_lo[itree][m])
                nf = int(avar.per_tm_nf[itree][m])
                pvs = [avar.per_tfm[itree][ifreq_lo + i][m] for i in range(nf)]
                label = f"tree {itree} m={m} (n={int(fs.m_to_n[m])})"
                blocks.append(VarianceMapBlock(pvs, ifreq_lo, self.nfreq, r - R, P,
                                               epsilon=epsilon, label=label))

            self.blocks[itree] = blocks

            if progress:
                k0 = sum(len(b.sval) for b in blocks)
                atomic_print(f"  VarianceMapExact tree {itree}/{self.ntrees}: "
                             f"{len(blocks)} blocks, sum(K0)={k0}")


    def eval_tree(self, itree, freq_variances):
        """Return a shape (M, 2^{r-R}, P) array, comparable to avar.tree_variance[itree]."""

        itree = int(itree)
        r, R, P = int(self.tree_r[itree]), int(self.tree_R[itree]), int(self.tree_P[itree])
        blocks = self.blocks[itree]

        out = np.zeros((len(blocks), 1 << (r - R), P))
        for m, block in enumerate(blocks):
            out[m, :, :] = block.eval(freq_variances)
        return out


    def _reference(self, itree, freq_variances):
        """Recompute avar.tree_variance[itree] from avar.per_tfm, for arbitrary input variances.

        This deliberately duplicates the frequency sum which PfAvarExact does at construction,
        rather than reusing its result, so that check() can be run with input variances which
        the PfAvarExact was not built with.
        """

        avar = self.avar
        r, R, P = int(self.tree_r[itree]), int(self.tree_R[itree]), int(self.tree_P[itree])
        M = int(self.tree_M[itree])
        all_dbits = (1 << (r - R)) - 1

        out = np.zeros((M, 1 << (r - R), P))

        for m in range(M):
            ifreq_lo = int(avar.per_tm_ifreq_lo[itree][m])
            nf = int(avar.per_tm_nf[itree][m])
            pv = PfVariance(r - R, P)
            for ifreq in range(ifreq_lo, ifreq_lo + nf):
                pv.add(avar.per_tfm[itree][ifreq][m], scale=float(freq_variances[ifreq]))
            out[m, :, :] = pv.unpack(all_dbits)

        return out


    def _tree_desc(self, itree):
        return (f"r={int(self.tree_r[itree])} R={int(self.tree_R[itree])} "
                f"P={int(self.tree_P[itree])} M={int(self.tree_M[itree])}")
