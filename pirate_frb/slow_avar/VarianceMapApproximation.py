import numpy as np

from .PfVariance import PfVariance
from .VarianceMap import VarianceMapBlock, VarianceMapBase
from ..utils import atomic_print


################################   class VarianceMapApproximation   ################################


class VarianceMapApproximation(VarianceMapBase):
    """Low-rank (SVD) representation of the approximate variance map of a PfAvarApproximation.

    Constructed from an already-built slow_avar.PfAvarApproximation (see PfVariance.py). Whereas
    the PfAvarApproximation computes the output variances for one specific input array
    'freq_variances', a VarianceMapApproximation represents the whole linear map (input
    variances) -> (output variances), in a compressed form which can be re-evaluated cheaply for
    any input (see eval() below). This matters because in operation the input variances drift
    (gain drift, RFI excision), and we want to recompute peak-finding weights whenever they are
    updated.

    This is the counterpart of VarianceMapExact for the approximate variance map: the linear
    algebra is identical, but the block index is the level-0 band 0 <= j < 2^R rather than the
    multiplet, the DM axis has 2^(r-L) entries rather than 2^(r-R), and eval_tree() applies the
    per-subband average which PfAvarApproximation applies to per_tf. See
    notes/tree_dedispersion.tex, appendix "PfAvarApproximation, VarianceMapApproximation".

    Unlike VarianceMapExact, this class is usable at CHIME/CHORD scale. Note however that it
    requires the python PfAvarApproximation: the C++ port drops per_tff, keeping only per_tf.

    Note that a VarianceMapApproximation keeps a reference to the PfAvarApproximation, which is
    usually the larger of the two objects (check() needs it, and the memory is already committed
    anyway).

    Members
    -------
      tree_L:       length-ntrees array, log2(wt_dm_downsampling) (= avar.tree_L).
      tree_N:       length-ntrees array, number of frequency subbands.
      blocks:       (ntrees, 2^R) ragged array (list of lists) of VarianceMapBlocks.

    plus the members of VarianceMapBase (avar, ntrees, nfreq, tree_r, tree_R, tree_P, epsilon).
    """

    def __init__(self, avar, epsilon=None, progress=False):
        """Build the low-rank representation of an (already constructed) PfAvarApproximation.

        If epsilon is None (the default), each block chooses its own truncation threshold from
        its matrix sizes; an explicit float applies to all blocks. If progress is set, print one
        line per tree.
        """

        self._init_common(avar, epsilon)
        self.tree_L = np.array(avar.tree_L)
        self.tree_N = np.array([fs.N for fs in avar.tree_fs])

        for itree in range(self.ntrees):
            r, R, L = int(self.tree_r[itree]), int(self.tree_R[itree]), int(self.tree_L[itree])
            P = int(self.tree_P[itree])
            blocks = [ ]

            for j in range(1 << R):
                ifreq_lo = int(avar.per_tf_ifreq_lo[itree][j])
                nf = int(avar.per_tf_nf[itree][j])
                pvs = [avar.per_tff[itree][ifreq_lo + i][j] for i in range(nf)]
                blocks.append(VarianceMapBlock(pvs, ifreq_lo, self.nfreq, r - L, P,
                                               epsilon=epsilon, label=f"tree {itree} j={j}"))

            self.blocks[itree] = blocks

            if progress:
                k0 = sum(len(b.sval) for b in blocks)
                atomic_print(f"  VarianceMapApproximation tree {itree}/{self.ntrees}: "
                             f"{len(blocks)} blocks, sum(K0)={k0}")


    def eval_band(self, itree, j, freq_variances):
        """Return the shape (2^{r-L}, P) variance array of one level-0 band.

        This is the primitive: eval_tree() below is a per-subband average of these.
        """
        return self.blocks[int(itree)][int(j)].eval(freq_variances)


    def eval_tree(self, itree, freq_variances):
        """Return a shape (N, 2^{r-L}, P) array, comparable to avar.tree_variance[itree].

        Entry n is the mean of eval_band() over subband n's coarse-freq range, matching what
        PfAvarApproximation does with per_tf.
        """

        itree = int(itree)
        r, L, P = int(self.tree_r[itree]), int(self.tree_L[itree]), int(self.tree_P[itree])
        fs = self.avar.tree_fs[itree]

        # A level-0 band belongs to several subbands (and there can be more subbands than bands),
        # so evaluate each band once and average, rather than evaluating per subband.
        bands = [block.eval(freq_variances) for block in self.blocks[itree]]

        out = np.zeros((fs.N, 1 << (r - L), P))
        for n in range(fs.N):
            flo, fhi = int(fs.n_to_flo[n]), int(fs.n_to_fhi[n])
            for j in range(flo, fhi):
                out[n, :, :] += bands[j]
            out[n, :, :] /= (fhi - flo)

        return out


    def _reference(self, itree, freq_variances):
        """Recompute avar.tree_variance[itree] from avar.per_tff, for arbitrary input variances.

        This deliberately duplicates the frequency sum and per-subband average which
        PfAvarApproximation does at construction, rather than reusing its result, so that check()
        can be run with input variances which the PfAvarApproximation was not built with.
        """

        avar = self.avar
        r, R, L = int(self.tree_r[itree]), int(self.tree_R[itree]), int(self.tree_L[itree])
        P, fs = int(self.tree_P[itree]), avar.tree_fs[itree]
        all_dbits = (1 << (r - L)) - 1

        bands = [ ]
        for j in range(1 << R):
            ifreq_lo = int(avar.per_tf_ifreq_lo[itree][j])
            nf = int(avar.per_tf_nf[itree][j])
            pv = PfVariance(r - L, P)
            for ifreq in range(ifreq_lo, ifreq_lo + nf):
                pv.add(avar.per_tff[itree][ifreq][j], scale=float(freq_variances[ifreq]))
            bands.append(pv.unpack(all_dbits))

        out = np.zeros((fs.N, 1 << (r - L), P))
        for n in range(fs.N):
            flo, fhi = int(fs.n_to_flo[n]), int(fs.n_to_fhi[n])
            for j in range(flo, fhi):
                out[n, :, :] += bands[j]
            out[n, :, :] /= (fhi - flo)

        return out


    def _tree_desc(self, itree):
        return (f"r={int(self.tree_r[itree])} R={int(self.tree_R[itree])} "
                f"L={int(self.tree_L[itree])} P={int(self.tree_P[itree])} "
                f"N={int(self.tree_N[itree])}")
