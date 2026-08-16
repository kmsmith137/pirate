import numpy as np

from .SparseTile import SparseTile
from ..utils import atomic_print


####################################   class VarianceMapBlock   ####################################


class VarianceMapBlock:
    """Truncated SVD of one block of a variance map.

    Blocks are created by VarianceMapExact and VarianceMapApproximation (one per (tree, multiplet)
    and per (tree, level-0 band) respectively); you'd only construct one directly when debugging
    a single block.

    The block represents the linear map (input frequency variances) -> (one block's variance
    array), as

       var[d,p] = sum_k sval[k] * (sv_freq[k,:] . vbar) * sv_out[k,d,p]

    where vbar = lam * freq_variances[ifreq_lo : ifreq_lo+nf].  Both sv_freq and sv_out have
    orthonormal rows, so this is a genuine truncated SVD; 'lam' is a preconditioner which
    equalizes the scale of the map's frequency columns before truncating (without it, input
    channels which barely overlap the block would be truncated away).

    For the math, see notes/variance_map.tex, appendix "PfAvarExact and VarianceMapExact".

    Members
    -------
      label:        short human-readable description, used in show() and assertion messages.
      rho, P:       number of DM bits and of peak-finding profiles.
      nfreq:        total number of input frequency channels (not just this block's).
      ifreq_lo, nf: input channels contributing to this block are
                    ifreq_lo <= ifreq < ifreq_lo + nf. Note nf can be zero (see __init__).
      lam:          shape (nf,) preconditioner, >= 0. Zero for a channel whose contribution
                    vanishes identically; such channels are dropped (see nzero).
      sval:         shape (K0,) singular values, sorted descending.
      sv_freq:      shape (K0, nf) frequency-side singular vectors (orthonormal rows).
      sv_out:       shape (K0, 2^rho, P) output-side singular vectors (orthonormal rows,
                    after flattening the last two axes).
      epsilon:      relative singular-value threshold used for truncation.
      ktot:         rank before canonicalization (sum of the per-dbits truncated ranks).
                    K0 <= ktot; the gap measures what canonicalization bought.
      ndelta:       number of distinct dbits groups.
      nzero:        number of channels in [ifreq_lo, ifreq_lo+nf) with lam == 0.
    """

    def __init__(self, pvs, ifreq_lo, nfreq, rho, P, epsilon=None, label=''):
        """Build a block from the per-channel variances of a contiguous input-channel range.

        The 'pvs' arg is a list of PfVariances (rank rho, P profiles), one per input channel in
        [ifreq_lo, ifreq_lo+nf) where nf = len(pvs); None entries are not allowed.

        An empty 'pvs' is allowed, and gives an all-zero block (K0 == 0). This would be a block
        which no input channel contributes to. We've never seen one -- every block spans a
        contiguous range of tree-freq channels, which always overlaps at least one input channel
        -- but VarianceMapApproximation's caller can't rule it out structurally (see the
        per_tf_nf docstring in PfVariance.py), so it's handled rather than asserted.

        If epsilon is None (the default), the truncation threshold is chosen from the block's
        matrix sizes (see _default_epsilon); an explicit float overrides it.
        """

        self.label = str(label)
        self.rho, self.P = int(rho), int(P)
        self.nfreq = int(nfreq)
        self.ifreq_lo, self.nf = int(ifreq_lo), len(pvs)
        assert self.rho >= 0 and self.P >= 1, (self.rho, self.P)
        assert 0 <= self.ifreq_lo and self.ifreq_lo + self.nf <= self.nfreq, \
            (self.label, self.ifreq_lo, self.nf, self.nfreq)

        D = 1 << self.rho                  # number of DMs on this block's DM axis
        all_dbits = D - 1                  # bitmask selecting all rho DM bits

        self.epsilon = self._default_epsilon(D * P, self.nf) if (epsilon is None) else float(epsilon)
        assert self.epsilon > 0.0, (self.label, self.epsilon)

        # Preconditioner lam[i] = sum over channel (ifreq_lo+i)'s terms of mean(term array).
        # Term arrays hold variances, hence are >= 0 entrywise, hence lam >= 0 -- and lam[i] == 0
        # happens only if the channel's contribution vanishes identically. Such channels are
        # dropped below: their column of the variance map is zero, and dividing by lam would be
        # 0/0. (Their rows of sv_freq end up zero, so eval() can still take a plain contiguous
        # slice of freq_variances.)
        self.lam = np.zeros(self.nf)
        for i, pv in enumerate(pvs):
            assert pv is not None, (self.label, i)
            assert (pv.rank, pv.P) == (self.rho, self.P), (self.label, i, pv.rank, pv.P)
            for arr in pv.terms.values():
                self.lam[i] += float(arr.mean())

        assert np.all(self.lam >= 0.0), (self.label, float(self.lam.min()))
        self.nzero = int(np.sum(self.lam <= 0.0))

        # Group the (channel, term) pairs by dbits, dividing each term array by its channel's
        # lam. A PfVariance's terms are a dict keyed by dbits, so a channel appears at most once
        # per group -- but a channel with two terms appears in two groups, which is why the
        # groups' channel sets can overlap.
        grp_ifreq = { }    # dbits -> list of channel indices (relative to ifreq_lo)
        grp_arr = { }      # dbits -> list of (2^popcount(dbits), P) arrays

        for i, pv in enumerate(pvs):
            if self.lam[i] <= 0.0:
                continue
            for dbits, arr in pv.terms.items():
                grp_ifreq.setdefault(dbits, []).append(i)
                grp_arr.setdefault(dbits, []).append(arr / self.lam[i])

        self.ndelta = len(grp_arr)

        # Per-group truncated SVDs, stacked along the rank axis into
        #
        #   wbar (ktot, D*P), sbar (ktot,), qbar (ktot, nf)   with
        #   Abar[(d,p), i] = sum_j wbar[j,(d,p)] sbar[j] qbar[j,i],
        #
        # where Abar is the (never materialized) preconditioned variance map. Each group's left
        # singular vectors are expanded from the packed DM axis 2^popcount(dbits) to the full
        # 2^rho (an index gather), and its right singular vectors are scattered into the group's
        # channel positions. Neither wbar nor qbar has orthonormal rows -- fixed below.
        w_rows, s_rows, q_rows = [ ], [ ], [ ]

        for dbits in sorted(grp_arr.keys()):
            ifreq_idx = np.array(grp_ifreq[dbits])            # (n_delta,)
            arrs = np.array(grp_arr[dbits])                   # (n_delta, 2^popcount(dbits), P)
            npack = 1 << int(dbits).bit_count()

            u, s, vh = np.linalg.svd(arrs.reshape(len(ifreq_idx), -1).T, full_matrices=False)
            K = self._nkeep(s, self.epsilon)
            if K == 0:
                continue

            # Expand the packed DM axis: row d of the output reads packed row sel(d, dbits).
            expand = SparseTile._remap_d(np.arange(D), all_dbits, dbits)   # (D,)
            wexp = u[:, :K].T.reshape(K, npack, P)[:, expand, :]           # (K, D, P)
            w_rows.append(wexp.reshape(K, D * P))

            q = np.zeros((K, self.nf))
            q[:, ifreq_idx] = vh[:K, :]
            q_rows.append(q)
            s_rows.append(s[:K])

        self.ktot = int(sum(len(s) for s in s_rows))

        if self.ktot == 0:
            # Reached when the block has no contributing channels at all, or (not seen in
            # practice) when every channel's contribution vanishes.
            self.sval = np.zeros(0)
            self.sv_freq = np.zeros((0, self.nf))
            self.sv_out = np.zeros((0, D, P))
            return

        wbar = np.concatenate(w_rows, axis=0)      # (ktot, D*P)
        sbar = np.concatenate(s_rows)              # (ktot,)
        qbar = np.concatenate(q_rows, axis=0)      # (ktot, nf)

        # Canonicalize: reduce (wbar, sbar, qbar) to a genuine truncated SVD of Abar, without
        # materializing anything of Abar's size. Writing the SVDs of the two stacked factors as
        # wbar = uw diag(sw) vw and qbar = uq diag(sq) vq, with vw, vq semiorthogonal, we get
        # Abar = vw^T M vq with M = diag(sw) uw^T diag(sbar) uq diag(sq) a small (kW,kQ) matrix.
        # This is exact even if wbar or qbar is rank-deficient: a near-null direction shows up as
        # a roundoff-sized entry of sw or sq, which multiplies the corresponding row/column of M
        # and thereby kills the (arbitrary) direction in vw or vq.
        uw, sw, vw = np.linalg.svd(wbar, full_matrices=False)     # vw: (kW, D*P)
        uq, sq, vq = np.linalg.svd(qbar, full_matrices=False)     # vq: (kQ, nf)
        mat = ((sw[:, None] * uw.T) @ (sbar[:, None] * uq)) * sq[None, :]     # (kW, kQ)

        # vw and vq have orthonormal rows, so M's singular values are exactly Abar's. Absorb M's
        # SVD into them, truncating to K0 = the numerical rank of Abar.
        ux, sx, vx = np.linalg.svd(mat, full_matrices=False)
        K0 = self._nkeep(sx, self.epsilon)

        self.sval = sx[:K0]                                # (K0,)
        self.sv_freq = vx[:K0, :] @ vq                     # (K0, nf)
        self.sv_out = (ux[:, :K0].T @ vw).reshape(K0, D, P)


    def eval(self, freq_variances):
        """Return this block's shape (2^rho, P) variance array, for the given input variances.

        The 'freq_variances' arg is a full length-nfreq array (not just this block's channels).
        """

        v = np.asarray(freq_variances, dtype=np.float64)
        assert v.shape == (self.nfreq,), (self.label, v.shape, self.nfreq)

        vbar = self.lam * v[self.ifreq_lo : self.ifreq_lo + self.nf]
        coeffs = (self.sv_freq @ vbar) * self.sval                  # (K0,)
        return np.tensordot(coeffs, self.sv_out, axes=(0, 0))       # (2^rho, P)


    def nbytes(self):
        """Number of bytes stored (the low-rank factors plus the preconditioner)."""
        return self.lam.nbytes + self.sval.nbytes + self.sv_freq.nbytes + self.sv_out.nbytes


    def dense_nbytes(self):
        """Number of bytes a dense (2^rho * P, nf) representation of this block would need."""
        return 8 * (1 << self.rho) * self.P * self.nf


    def check(self, epsabs=1.0e-10):
        """Check that this block's factors are semiorthogonal, and its singular values sorted."""

        k0 = len(self.sval)
        assert self.sv_freq.shape == (k0, self.nf), (self.label, self.sv_freq.shape, k0, self.nf)
        assert self.sv_out.shape == (k0, 1 << self.rho, self.P), (self.label, self.sv_out.shape)
        assert np.all(self.sval > 0.0), self.label
        assert np.all(np.diff(self.sval) <= 0.0), self.label

        # Note: an explicit column count, not -1, since numpy can't infer -1 when k0 == 0.
        for mat in [self.sv_freq, self.sv_out.reshape(k0, (1 << self.rho) * self.P)]:
            err = float(np.max(np.abs(mat @ mat.T - np.eye(k0)))) if k0 else 0.0
            assert err < epsabs, (self.label, mat.shape, err)


    @staticmethod
    def _default_epsilon(nrow, ncol):
        """Relative singular-value threshold for a block whose largest matrix is (nrow, ncol).

        Truncating at eps*S_max perturbs the reconstruction at the eps level, so eps is an
        accuracy knob -- but it is only meaningful above the float64 noise floor on singular
        values, which numpy's matrix_rank estimates as max(shape) * eps_f64 * S_max. At the
        variance maps' sizes that floor can exceed the 1e-11 which is safe for smaller matrices
        (e.g. 1.5e-11 at 2^(r-R) P = 65536), so we take whichever is larger.
        """
        return max(1.0e-11, 16.0 * max(int(nrow), int(ncol)) * float(np.finfo(np.float64).eps))


    @staticmethod
    def _nkeep(s, epsilon):
        """Number of leading singular values above (epsilon * s[0]); 0 if s is empty or zero."""
        if (s.size == 0) or not (s[0] > 0.0):
            return 0
        return int(np.sum(s > epsilon * s[0]))


####################################   class VarianceMapBase   #####################################


class VarianceMapBase:
    """Machinery shared by VarianceMapExact and VarianceMapApproximation (see those classes).

    Not constructed directly. A subclass calls _init_common() and then fills self.blocks with
    VarianceMapBlocks, and implements two hooks:

      eval_tree(itree, freq_variances):  the low-rank evaluation, shaped like avar.tree_variance
      _reference(itree, freq_variances): the same quantity, recomputed directly from the PfAvar's
                                         per-channel PfVariances (used by check())
      _tree_desc(itree):                 a short string like "r=12 R=3 P=13", for show()

    Members
    -------
      avar:         the PfAvarExact or PfAvarApproximation.
      ntrees:       number of DedispersionTrees (= avar.ntrees).
      nfreq:        number of input frequency channels (= avar.nfreq).
      tree_r:       length-ntrees array, tree rank (= avar.tree_r).
      tree_R:       length-ntrees array, pf_rank (= avar.tree_R).
      tree_P:       length-ntrees array, nprofiles (= avar.tree_P).
      epsilon:      the constructor's 'epsilon' arg (None means per-block, see VarianceMapBlock).
      blocks:       ragged array (list of lists) of VarianceMapBlocks, indexed [itree][...].
    """

    def _init_common(self, avar, epsilon):
        self.avar = avar
        self.ntrees, self.nfreq = int(avar.ntrees), int(avar.nfreq)
        self.tree_r = np.array(avar.tree_r)
        self.tree_R = np.array(avar.tree_R)
        self.tree_P = np.array(avar.tree_P)
        self.epsilon = epsilon
        self.blocks = [ None ] * self.ntrees


    def eval(self, freq_variances):
        """Return a length-ntrees list of arrays, comparable to avar.tree_variance."""
        return [self.eval_tree(itree, freq_variances) for itree in range(self.ntrees)]


    def check(self, freq_variances=None, rtol=1.0e-8, progress=False):
        """Check the low-rank representation against a directly computed reference.

        If 'freq_variances' is None (the default), we compare eval() to avar.tree_variance, i.e.
        we re-evaluate the map at the same input variances which the PfAvar was built with.
        Otherwise, 'freq_variances' is a length-nfreq array of positive variances, and the
        reference is recomputed from the PfAvar's per-channel PfVariances. The latter is the more
        meaningful check: it tests that we captured the linear map, not just its value at one
        input.

        Also checks that each block's factors are semiorthogonal and its singular values sorted.

        Returns the largest relative error over all trees; raises if it exceeds 'rtol'.
        """

        if freq_variances is None:
            v, use_reference = self.avar.freq_variances, False
        else:
            v = np.asarray(freq_variances, dtype=np.float64)
            assert v.shape == (self.nfreq,), (v.shape, self.nfreq)
            assert np.all(v > 0.0), float(v.min())
            use_reference = True

        cls, max_eps = type(self).__name__, 0.0

        for itree in range(self.ntrees):
            for block in self.blocks[itree]:
                block.check()

            want = self._reference(itree, v) if use_reference else self.avar.tree_variance[itree]
            got = self.eval_tree(itree, v)
            assert got.shape == want.shape, (itree, got.shape, want.shape)
            assert np.all(want > 0.0), (itree, float(want.min()))

            eps = float(np.max(np.abs(got / want - 1.0)))
            max_eps = max(max_eps, eps)

            if progress:
                atomic_print(f"  {cls}.check tree {itree}/{self.ntrees}: "
                             f"max relative error {eps:.3g}")

            if not (eps <= rtol):
                raise RuntimeError(f"{cls}.check: tree {itree} has max relative error "
                                   f"{eps:.6g}, which exceeds rtol={rtol:.6g}")

        return max_eps


    def show(self, per_block=True):
        """Print block sizes, ranks and compression ratios (per block, per tree, and in total)."""

        tot_nbytes, tot_dense, tot_k0, tot_ktot, tot_blocks = 0, 0, 0, 0, 0

        for itree in range(self.ntrees):
            blocks = self.blocks[itree]
            nbytes = sum(b.nbytes() for b in blocks)
            dense = sum(b.dense_nbytes() for b in blocks)
            k0 = sum(len(b.sval) for b in blocks)
            ktot = sum(b.ktot for b in blocks)

            atomic_print(f"tree {itree} [{self._tree_desc(itree)}]: {len(blocks)} blocks, "
                         f"sum(Ktot)={ktot}, sum(K0)={k0}, "
                         f"{nbytes/2**20:.1f} MiB vs {dense/2**20:.1f} MiB dense "
                         f"({dense/max(nbytes,1):.1f}x)")

            if per_block:
                for b in blocks:
                    atomic_print(f"    {b.label}: ifreq_lo={b.ifreq_lo}, nf={b.nf}, "
                                 f"DP={(1 << b.rho) * b.P}, ndelta={b.ndelta}, Ktot={b.ktot}, "
                                 f"K0={len(b.sval)}, nzero={b.nzero}, epsilon={b.epsilon:.3g}")

            tot_nbytes, tot_dense = tot_nbytes + nbytes, tot_dense + dense
            tot_k0, tot_ktot, tot_blocks = tot_k0 + k0, tot_ktot + ktot, tot_blocks + len(blocks)

        atomic_print(f"total: {tot_blocks} blocks, sum(Ktot)={tot_ktot}, sum(K0)={tot_k0}, "
                     f"{tot_nbytes/2**20:.1f} MiB vs {tot_dense/2**20:.1f} MiB dense "
                     f"({tot_dense/max(tot_nbytes,1):.1f}x)")
