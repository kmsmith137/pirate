import numpy as np

from .SparseTile import SparseTile
from .PfVariance import PfVariance
from ..utils import atomic_print


##################################   class VarianceMapExactBlock   #################################


class VarianceMapExactBlock:
    """Truncated SVD of one (tree, multiplet) block of the exact variance map.

    Blocks are created by VarianceMapExact (see below); you'd only construct one directly
    when debugging a single (tree, multiplet) pair.

    The block represents the linear map (input frequency variances) -> (one multiplet's
    variance array), as

       var[d,p] = sum_k sval[k] * (sv_freq[k,:] . vbar) * sv_out[k,d,p]

    where vbar = lam * freq_variances[ifreq_lo : ifreq_lo+nf].  Both sv_freq and sv_out have
    orthonormal rows, so this is a genuine truncated SVD; 'lam' is a preconditioner which
    equalizes the scale of the map's frequency columns before truncating (without it, input
    channels which barely overlap the multiplet's subband would be truncated away).

    For the math, see notes/tree_dedispersion.tex, section "SVD decomposition of the exact
    variance map".

    Members
    -------
      itree, m, n:  tree index, multiplet index, and the multiplet's frequency subband.
      rho, P:       number of coarse-DM bits (= r-R) and of peak-finding profiles.
      nfreq:        total number of input frequency channels (not just this block's).
      ifreq_lo, nf: input channels contributing to this multiplet are
                    ifreq_lo <= ifreq < ifreq_lo + nf.
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

    def __init__(self, avar, itree, m, epsilon=None):
        """Build the block for multiplet 'm' of tree 'itree' of a PfAvarExact 'avar'.

        If epsilon is None (the default), the truncation threshold is chosen per-block
        (see _default_epsilon); an explicit float overrides it.
        """

        itree, m = int(itree), int(m)
        assert 0 <= itree < avar.ntrees, (itree, avar.ntrees)

        r, R, P = int(avar.tree_r[itree]), int(avar.tree_R[itree]), int(avar.tree_P[itree])
        fs = avar.tree_fs[itree]
        assert 0 <= m < fs.M, (m, fs.M)

        self.itree, self.m = itree, m
        self.n = int(fs.m_to_n[m])
        self.rho, self.P = r - R, P
        self.nfreq = int(avar.nfreq)
        self.ifreq_lo = int(avar.per_tm_ifreq_lo[itree][m])
        self.nf = int(avar.per_tm_nf[itree][m])
        assert self.nf >= 1, (itree, m)

        D = 1 << self.rho                  # number of coarse DMs
        all_dbits = D - 1                  # bitmask selecting all rho DM bits

        self.epsilon = self._default_epsilon(D * P, self.nf) if (epsilon is None) else float(epsilon)
        assert self.epsilon > 0.0, self.epsilon

        pvs = [avar.per_tfm[itree][self.ifreq_lo + i][m] for i in range(self.nf)]

        # Preconditioner lam[i] = sum over channel (ifreq_lo+i)'s terms of mean(term array).
        # Term arrays hold variances, hence are >= 0 entrywise, hence lam >= 0 -- and lam[i] == 0
        # happens only if the channel's contribution vanishes identically. Such channels are
        # dropped below: their column of the variance map is zero, and dividing by lam would be
        # 0/0. (Their rows of sv_freq end up zero, so eval() can still take a plain contiguous
        # slice of freq_variances.)
        self.lam = np.zeros(self.nf)
        for i, pv in enumerate(pvs):
            assert pv is not None, (itree, m, i)   # guaranteed by PfAvarExact's contiguity assert
            assert (pv.rank, pv.P) == (self.rho, P), (itree, m, i, pv.rank, pv.P, self.rho, P)
            for arr in pv.terms.values():
                self.lam[i] += float(arr.mean())

        assert np.all(self.lam >= 0.0), (itree, m, float(self.lam.min()))
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
            # Only reachable if every channel's contribution vanishes, which PfAvarExact's
            # "tree_variance > 0" assert already rules out. Handled anyway so that the members
            # always have consistent shapes.
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
        """Return this multiplet's shape (2^rho, P) variance array, for the given input variances.

        The 'freq_variances' arg is a full length-nfreq array (not just this block's channels).
        """

        v = np.asarray(freq_variances, dtype=np.float64)
        assert v.shape == (self.nfreq,), (v.shape, self.nfreq)

        vbar = self.lam * v[self.ifreq_lo : self.ifreq_lo + self.nf]
        coeffs = (self.sv_freq @ vbar) * self.sval                  # (K0,)
        return np.tensordot(coeffs, self.sv_out, axes=(0, 0))       # (2^rho, P)


    def nbytes(self):
        """Number of bytes stored (the low-rank factors plus the preconditioner)."""
        return self.lam.nbytes + self.sval.nbytes + self.sv_freq.nbytes + self.sv_out.nbytes


    def dense_nbytes(self):
        """Number of bytes a dense (2^rho * P, nf) representation of this block would need."""
        return 8 * (1 << self.rho) * self.P * self.nf


    @staticmethod
    def _default_epsilon(nrow, ncol):
        """Relative singular-value threshold for a block whose largest matrix is (nrow, ncol).

        Truncating at eps*S_max perturbs the reconstruction at the eps level, so eps is an
        accuracy knob -- but it is only meaningful above the float64 noise floor on singular
        values, which numpy's matrix_rank estimates as max(shape) * eps_f64 * S_max. At the
        exact variance map's sizes that floor can exceed the 1e-11 which is safe for smaller
        matrices (e.g. 1.5e-11 at 2^(r-R) P = 65536), so we take whichever is larger.
        """
        return max(1.0e-11, 16.0 * max(int(nrow), int(ncol)) * float(np.finfo(np.float64).eps))


    @staticmethod
    def _nkeep(s, epsilon):
        """Number of leading singular values above (epsilon * s[0]); 0 if s is empty or zero."""
        if (s.size == 0) or not (s[0] > 0.0):
            return 0
        return int(np.sum(s > epsilon * s[0]))


####################################   class VarianceMapExact   ####################################


class VarianceMapExact:
    """Low-rank (SVD) representation of the exact variance map of a PfAvarExact.

    Constructed from an already-built PfAvarExact (see PfVariance.py). Whereas the PfAvarExact
    computes the output variances for one specific input array 'freq_variances', a
    VarianceMapExact represents the whole linear map (input variances) -> (output variances), in
    a compressed form which can be re-evaluated cheaply for any input (see eval() below). This
    matters because in operation the input variances drift (gain drift, RFI excision), and we
    want to recompute peak-finding weights whenever they are updated.

    The map is stored as one truncated SVD per (tree, multiplet) pair -- see
    VarianceMapExactBlock, and notes/tree_dedispersion.tex, section "SVD decomposition of the
    exact variance map".

    Note that a VarianceMapExact keeps a reference to the PfAvarExact, which is usually the
    larger of the two objects (check() needs it, and the memory is already committed anyway).

    Caveat: PfAvarExact is an exact-but-slow reference implementation which is only practical at
    moderate ranks, so this class inherits that limitation. It is not usable at CHIME/CHORD scale.

    Members
    -------
      avar:         the PfAvarExact.
      ntrees:       number of DedispersionTrees (= avar.ntrees).
      nfreq:        number of input frequency channels (= avar.nfreq).
      tree_r:       length-ntrees array, tree rank (= avar.tree_r).
      tree_R:       length-ntrees array, pf_rank (= avar.tree_R).
      tree_P:       length-ntrees array, nprofiles (= avar.tree_P).
      tree_M:       length-ntrees array, number of multiplets.
      epsilon:      the constructor's 'epsilon' arg (None means per-block, see the blocks).
      blocks:       (ntrees, M) ragged array (list of lists) of VarianceMapExactBlocks.
    """

    def __init__(self, avar, epsilon=None, progress=False):
        """Build the low-rank representation of an (already constructed) PfAvarExact.

        If epsilon is None (the default), each block chooses its own truncation threshold from
        its matrix sizes; an explicit float applies to all blocks. If progress is set, print one
        line per tree.
        """

        self.avar = avar
        self.ntrees, self.nfreq = int(avar.ntrees), int(avar.nfreq)
        self.tree_r = np.array(avar.tree_r)
        self.tree_R = np.array(avar.tree_R)
        self.tree_P = np.array(avar.tree_P)
        self.tree_M = np.array([fs.M for fs in avar.tree_fs])
        self.epsilon = epsilon
        self.blocks = [ None ] * self.ntrees

        for itree in range(self.ntrees):
            self.blocks[itree] = [VarianceMapExactBlock(avar, itree, m, epsilon=epsilon)
                                  for m in range(int(self.tree_M[itree]))]
            if progress:
                k0 = sum(len(b.sval) for b in self.blocks[itree])
                atomic_print(f"  VarianceMapExact tree {itree}/{self.ntrees}: "
                             f"{len(self.blocks[itree])} blocks, sum(K0)={k0}")


    def eval_tree(self, itree, freq_variances):
        """Return a shape (M, 2^{r-R}, P) array, comparable to avar.tree_variance[itree]."""

        itree = int(itree)
        r, R, P = int(self.tree_r[itree]), int(self.tree_R[itree]), int(self.tree_P[itree])
        blocks = self.blocks[itree]

        out = np.zeros((len(blocks), 1 << (r - R), P))
        for m, block in enumerate(blocks):
            out[m, :, :] = block.eval(freq_variances)
        return out


    def eval(self, freq_variances):
        """Return a length-ntrees list of arrays, comparable to avar.tree_variance."""
        return [self.eval_tree(itree, freq_variances) for itree in range(self.ntrees)]


    def check(self, freq_variances=None, rtol=1.0e-8, progress=False):
        """Check the low-rank representation against a directly computed reference.

        If 'freq_variances' is None (the default), we compare eval() to avar.tree_variance, i.e.
        we re-evaluate the map at the same input variances which the PfAvarExact was built with.
        Otherwise, 'freq_variances' is a length-nfreq array of positive variances, and the
        reference is recomputed from avar.per_tfm. The latter is the more meaningful check: it
        tests that we captured the linear map, not just its value at one input.

        Also checks that each block's factors are semiorthogonal and its singular values sorted.

        Returns the largest relative error over all trees; raises if it exceeds 'rtol'.
        """

        if freq_variances is None:
            v, reference = self.avar.freq_variances, None
        else:
            v = np.asarray(freq_variances, dtype=np.float64)
            assert v.shape == (self.nfreq,), (v.shape, self.nfreq)
            assert np.all(v > 0.0), float(v.min())
            reference = self._reference

        max_eps = 0.0

        for itree in range(self.ntrees):
            for block in self.blocks[itree]:
                self._check_block(block)

            want = self.avar.tree_variance[itree] if (reference is None) else reference(itree, v)
            got = self.eval_tree(itree, v)
            assert got.shape == want.shape, (itree, got.shape, want.shape)
            assert np.all(want > 0.0), (itree, float(want.min()))

            eps = float(np.max(np.abs(got / want - 1.0)))
            max_eps = max(max_eps, eps)

            if progress:
                atomic_print(f"  VarianceMapExact.check tree {itree}/{self.ntrees}: "
                             f"max relative error {eps:.3g}")

            if not (eps <= rtol):
                raise RuntimeError(f"VarianceMapExact.check: tree {itree} has max relative "
                                   f"error {eps:.6g}, which exceeds rtol={rtol:.6g}")

        return max_eps


    def show(self, per_block=True):
        """Print block sizes, ranks and compression ratios (per block, per tree, and in total)."""

        tot_nbytes, tot_dense, tot_k0, tot_ktot, tot_blocks = 0, 0, 0, 0, 0

        for itree in range(self.ntrees):
            r, R, P = int(self.tree_r[itree]), int(self.tree_R[itree]), int(self.tree_P[itree])
            blocks = self.blocks[itree]
            nbytes = sum(b.nbytes() for b in blocks)
            dense = sum(b.dense_nbytes() for b in blocks)
            k0 = sum(len(b.sval) for b in blocks)
            ktot = sum(b.ktot for b in blocks)

            atomic_print(f"tree {itree} [r={r} R={R} P={P} M={len(blocks)}]: "
                         f"2^(r-R)*P={(1 << (r-R)) * P}, sum(Ktot)={ktot}, sum(K0)={k0}, "
                         f"{nbytes/2**20:.1f} MiB vs {dense/2**20:.1f} MiB dense "
                         f"({dense/max(nbytes,1):.1f}x)")

            if per_block:
                for b in blocks:
                    atomic_print(f"    m={b.m} (n={b.n}): ifreq_lo={b.ifreq_lo}, nf={b.nf}, "
                                 f"ndelta={b.ndelta}, Ktot={b.ktot}, K0={len(b.sval)}, "
                                 f"nzero={b.nzero}, epsilon={b.epsilon:.3g}")

            tot_nbytes, tot_dense = tot_nbytes + nbytes, tot_dense + dense
            tot_k0, tot_ktot, tot_blocks = tot_k0 + k0, tot_ktot + ktot, tot_blocks + len(blocks)

        atomic_print(f"total: {tot_blocks} blocks, sum(Ktot)={tot_ktot}, sum(K0)={tot_k0}, "
                     f"{tot_nbytes/2**20:.1f} MiB vs {tot_dense/2**20:.1f} MiB dense "
                     f"({tot_dense/max(tot_nbytes,1):.1f}x)")


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


    @staticmethod
    def _check_block(block, epsabs=1.0e-10):
        """Check that a block's factors are semiorthogonal, and its singular values sorted."""

        k0 = len(block.sval)
        assert block.sv_freq.shape == (k0, block.nf), (block.sv_freq.shape, k0, block.nf)
        assert block.sv_out.shape == (k0, 1 << block.rho, block.P), block.sv_out.shape
        assert np.all(block.sval > 0.0), (block.itree, block.m)
        assert np.all(np.diff(block.sval) <= 0.0), (block.itree, block.m)

        for mat in [block.sv_freq, block.sv_out.reshape(k0, -1)]:
            err = float(np.max(np.abs(mat @ mat.T - np.eye(k0)))) if k0 else 0.0
            assert err < epsabs, (block.itree, block.m, mat.shape, err)
