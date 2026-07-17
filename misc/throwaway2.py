"""Throwaway script: unpickle the dict written by throwaway1.py (pickled members of a
slow_avar PfAvarApproximation) and compute derived quantities from it.

Computes so far:

  ref_variance: for each tree, a shape (2^R, 2^{r-L}, P) array obtained by unpacking
    per_tf. Analogous to PfAvarApproximation.tree_variance (see PfVariance.py), but
    keeping the per-frequency axis 0 <= j < 2^R, i.e. omitting the average of per_tf
    over each subband's coarse-freq range (fs.n_to_flo[n], fs.n_to_fhi[n]).

Frequency-indexing notation:

  nfreq    toplevel number of input frequency channels. Variables indexing the range
           [0:nfreq] begin with 'ifreq' (e.g. ifreq_lo).
  nf_tj    number of input channels relevant for a given (tree, j) pair -- the
           contiguous range ifreq_lo <= ifreq < ifreq_lo + nf_tj. Variables indexing
           [0:nf_tj] begin with 'if_tj'.
  nf_tjd   number of input channels relevant for a given (tree, j, dbits) triple.
           Variables indexing [0:nf_tjd] begin with 'if_tjd'.
"""

import os
import pickle
import numpy as np

from pirate_frb.slow_avar.SparseTile import SparseTile


script_dir = os.path.dirname(os.path.abspath(__file__))
pkl_filename = os.path.join(script_dir, 'throwaway1.pkl')

with open(pkl_filename, 'rb') as f:
    d = pickle.load(f)

ntrees = d['ntrees']
nfreq = d['nfreq']
freq_variances = d['freq_variances']
tree_r = d['tree_r']
tree_R = d['tree_R']
tree_L = d['tree_L']
tree_P = d['tree_P']
per_tff = d['per_tff']   # (ntrees, nfreq, 2^R) ragged array of (None or single-channel PfVariance)
per_tf = d['per_tf']     # (ntrees, 2^R) ragged array of frequency-summed PfVariances

# ref_variance: (ntrees,) list of shape (2^R, 2^{r-L}, P) arrays, from unpacking per_tf.
ref_variance = [None] * ntrees

for itree in range(ntrees):
    r, R, L, P = int(tree_r[itree]), int(tree_R[itree]), int(tree_L[itree]), int(tree_P[itree])
    assert len(per_tf[itree]) == (1 << R), (itree, len(per_tf[itree]), R)

    all_dbits = (1 << (r-L)) - 1
    ref_variance[itree] = np.zeros((1 << R, 1 << (r-L), P))

    for j in range(1 << R):
        ref_variance[itree][j, :, :] = per_tf[itree][j].unpack(all_dbits)

    assert np.all(ref_variance[itree] > 0.0), (itree, float(ref_variance[itree].min()))
    print(f'tree {itree} [r={r} R={R} L={L} P={P}]: ref_variance shape {ref_variance[itree].shape}')

# Check the structure of per_tff. For each tree and coarse-freq index 0 <= j < 2^R,
# view per_tff[itree][:][j] as a function of the input channel ifreq: the non-None
# entries occupy a contiguous range ifreq_lo <= ifreq < ifreq_lo + nf_tj, and each one
# is a PfVariance of rank (r-L) whose P is the tree's P, with only 1 or 2 terms.

for itree in range(ntrees):
    r, R, L, P = int(tree_r[itree]), int(tree_R[itree]), int(tree_L[itree]), int(tree_P[itree])
    nf_tj_list = []

    for j in range(1 << R):
        col = [per_tff[itree][ifreq][j] for ifreq in range(nfreq)]
        nonnone = [ifreq for (ifreq, pv) in enumerate(col) if pv is not None]
        assert len(nonnone) > 0, (itree, j)

        # Contiguity: since 'nonnone' is sorted and duplicate-free, the non-None ifreq's
        # are exactly {ifreq_lo, ..., ifreq_lo + nf_tj - 1} iff the endpoints are
        # consistent with the count.
        ifreq_lo, nf_tj = nonnone[0], len(nonnone)
        assert nonnone[-1] == ifreq_lo + nf_tj - 1, (itree, j, ifreq_lo, nf_tj, nonnone[-1])
        nf_tj_list.append(nf_tj)

        # Double-loop over all (frequency, dbits, variance_array) triples.
        # Group (frequency, variance_array) by dbits.

        dbits_freqs = { }   # dbits -> (list of ifreq values)
        dbits_varrs = { }   # dbits -> (list of variance_arrays)

        for ifreq in range(ifreq_lo, ifreq_lo + nf_tj):
            pv = col[ifreq]
            assert pv.rank == r - L, (itree, j, ifreq, pv.rank, r - L)
            assert pv.P == P, (itree, j, ifreq, pv.P, P)

            for dbits, varr in pv.terms.items():
                dbits_freqs.setdefault(dbits,[]).append(ifreq)
                dbits_varrs.setdefault(dbits,[]).append(varr)

        # Convert lists to arrays.
        # dbits_freqs: dbits -> (shape (nf_tjd,) array of ifreq values)
        # dbits_varrs: dbits -> (shape (nf_tjd, 2^popcount(dbits), P) array)

        dbits_freqs = { dbits: np.array(ifreqs) for dbits,ifreqs in dbits_freqs.items() }
        dbits_varrs = { dbits: np.array(varrs) for dbits,varrs in dbits_varrs.items() }

        # Check that we can recover ref_variance[j,:,:] from the (dbits_freqs, dbits_varrs)
        # representation. (Loop over dbits,ifreqs,varrs. Use ifreqs to "slice" freq_variances
        # to a length-nf_tjd array. Contract with varrs to get a shape (2^popcount, P) array.
        # Apply SparseTile._remap_d() to get a (2^{r-L}, P) array. Sum this over loop
        # iterations, and assert agreement with ref_variance[j,:,:].

        all_dbits = (1 << (r-L)) - 1
        rows = np.arange(1 << (r-L))
        recon = np.zeros((1 << (r-L), P))

        for dbits, ifreqs in dbits_freqs.items():
            varrs = dbits_varrs[dbits]                                         # (nf_tjd, 2^popcount(dbits), P)
            contracted = np.einsum('f,fdp->dp', freq_variances[ifreqs], varrs) # (2^popcount(dbits), P)
            recon += contracted[SparseTile._remap_d(rows, all_dbits, dbits)]   # (2^{r-L}, P)

        maxdiff = float(np.max(np.abs(recon / ref_variance[itree][j] - 1.0)))
        assert np.allclose(recon, ref_variance[itree][j], rtol=1e-10, atol=0.0), (itree, j, maxdiff)

        # SVD-compress each dbits group: reshape varrs (nf_tjd, 2^popc(dbits), P) -> matrix A
        # of shape (nf_tjd, D) with D = 2^popc(dbits)*P, and decompose A = U^T diag(S) V,
        # truncated to the K singular values above roundoff.
        #
        # Threshold: we cut at S > 1e-11 * S[0]. Since S[0] = ||A||_2, this is scaled to the
        # matrix entries; the float64 noise floor on singular values is ~max(nf_tjd,D)*eps*S[0]
        # ~ 6e-13 * S[0] here (numpy matrix_rank's default), so 1e-11 sits safely above
        # roundoff while far below any genuine component.

        dbits_umat = { }    # dbits -> shape (K, nf_tjd) array (rows = left singular vectors)
        dbits_sdiag = { }   # dbits -> shape (K,) array (singular values)
        dbits_vmat = { }    # dbits -> shape (K, D) array (rows = right singular vectors)

        for dbits, varrs in dbits_varrs.items():
            nf_tjd = varrs.shape[0]
            A = varrs.reshape(nf_tjd, -1)                # (nf_tjd, D), D = 2^popc(dbits) * P
            u, s, vh = np.linalg.svd(A, full_matrices=False)
            K = int(np.sum(s > 1e-11 * s[0]))
            dbits_umat[dbits] = u[:, :K].T               # (K, nf_tjd)
            dbits_sdiag[dbits] = s[:K]                   # (K,)
            dbits_vmat[dbits] = vh[:K, :]                # (K, D)

        # Check that we can recover ref_variance[j,:,:] from the SVD representation:
        # slice freq_variances, multiply by U^T, then S, then V, and remap indices as before.
        # (Truncating at 1e-11 * S[0] perturbs the reconstruction, so the tolerance is looser
        # than the exact check above.)

        recon2 = np.zeros((1 << (r-L), P))

        for dbits, ifreqs in dbits_freqs.items():
            coeffs = dbits_umat[dbits] @ freq_variances[ifreqs]       # (K,)
            coeffs = coeffs * dbits_sdiag[dbits]                      # (K,)
            contracted = (coeffs @ dbits_vmat[dbits]).reshape(-1, P)  # (2^popc(dbits), P)
            recon2 += contracted[SparseTile._remap_d(rows, all_dbits, dbits)]

        maxdiff2 = float(np.max(np.abs(recon2 / ref_variance[itree][j] - 1.0)))
        assert np.allclose(recon2, ref_variance[itree][j], rtol=1e-8, atol=0.0), (itree, j, maxdiff2)

        # Let Abar be the shape (nf_tj, 2^{r-L} * P) matrix representing the linear map
        # freq_variances[ifreq_lo : ifreq_lo + nf_tj] -> ref_variance[j,:,:].ravel().
        # (Never materialized.) Stack the per-dbits (U, S, V) into
        #
        #    Ubar (Ktot, nf_tj),  Sbar (Ktot,),  Vbar (Ktot, 2^{r-L} * P)
        #
        # such that Abar = Ubar^T diag(Sbar) Vbar, with no sum over dbits:
        #
        #  - Ubar: each group's U scatters into columns (ifreqs - ifreq_lo) of a zero-padded
        #    (K, nf_tj) row block. Different groups' channel sets can overlap (the two-term
        #    PfVariances), so the row blocks are not mutually orthogonal, and Ubar does
        #    not satisfy the semiorthogonality property Ubar Ubar^T = id_Ktot.
        #
        #  - Vbar: each group's V gets its packed delay axis 2^popc(dbits) expanded up
        #    to 2^{r-L} by the same _remap_d() gather used above. (This preserves row
        #    orthonormality within a block, but not across blocks.)

        Ktot = sum(len(s) for s in dbits_sdiag.values())
        Ubar = np.zeros((Ktot, nf_tj))
        Sbar = np.zeros(Ktot)
        Vbar = np.zeros((Ktot, (1 << (r-L)) * P))

        k0 = 0
        for dbits, sdiag in dbits_sdiag.items():
            K = len(sdiag)
            if_tj_arr = dbits_freqs[dbits] - ifreq_lo                         # group's channels, as indices in [0:nf_tj]
            Ubar[k0:k0+K, if_tj_arr] = dbits_umat[dbits]
            Sbar[k0:k0+K] = sdiag
            vexp = dbits_vmat[dbits].reshape(K, -1, P)                        # (K, 2^popc(dbits), P)
            vexp = vexp[:, SparseTile._remap_d(rows, all_dbits, dbits), :]    # (K, 2^{r-L}, P)
            Vbar[k0:k0+K, :] = vexp.reshape(K, -1)
            k0 += K

        assert k0 == Ktot

        # Check that freq_variances[ifreq_lo : ifreq_lo + nf_tj] . Ubar^T . diag(Sbar) . Vbar
        # = ref_variance[j,:,:].

        coeffs = Ubar @ freq_variances[ifreq_lo : ifreq_lo + nf_tj]           # (Ktot,)
        recon3 = ((coeffs * Sbar) @ Vbar).reshape(1 << (r-L), P)

        maxdiff3 = float(np.max(np.abs(recon3 / ref_variance[itree][j] - 1.0)))
        assert np.allclose(recon3, ref_variance[itree][j], rtol=1e-8, atol=0.0), (itree, j, maxdiff3)

        # Experiment: is the number of significant singular values of Abar less than Ktot?
        # Computed without materializing Abar-sized matrices: SVD-decompose
        #
        #   Ubar = U1 diag(U2) U3     U1 (Ktot,Ktot), U2 (Ktot,), U3 (Ktot, nf_tj)
        #   Vbar = V1 diag(V2) V3     V1 (Ktot,Ktot), V2 (Ktot,), V3 (Ktot, 2^{r-L} P)
        #
        # so Abar = U3^T M V3 with M = diag(U2) U1^T diag(Sbar) V1 diag(V2), a (Ktot,Ktot)
        # matrix. U3 U3^T = V3 V3^T = id_Ktot (Ktot <= nf_tj and Ktot <= 2^{r-L} P here), so
        # the singular values of M are exactly the singular values of Abar; K0 below is the
        # numerical rank of Abar, thresholded the same way as the per-dbits SVDs.

        U1, U2, U3 = np.linalg.svd(Ubar, full_matrices=False)
        V1, V2, V3 = np.linalg.svd(Vbar, full_matrices=False)
        assert U3.shape == (Ktot, nf_tj), (U3.shape, Ktot, nf_tj)
        assert V3.shape == (Ktot, (1 << (r-L)) * P), (V3.shape, Ktot, (1 << (r-L)) * P)

        M = (U2[:, None] * U1.T) @ (Sbar[:, None] * V1) * V2[None, :]   # (Ktot, Ktot)

        # Sanity check: x . Abar = ((U3 x) M) V3 must still reproduce ref_variance[j,:,:].
        recon4 = (((U3 @ freq_variances[ifreq_lo : ifreq_lo + nf_tj]) @ M) @ V3).reshape(1 << (r-L), P)
        maxdiff4 = float(np.max(np.abs(recon4 / ref_variance[itree][j] - 1.0)))
        assert np.allclose(recon4, ref_variance[itree][j], rtol=1e-8, atol=0.0), (itree, j, maxdiff4)

        uM, sM, vMh = np.linalg.svd(M)      # M = UM^T diag(SM) VM, with UM = uM.T, VM = vMh
        K0 = int(np.sum(sM > 1e-11 * sM[0]))

        # Canonicalize: absorb M's SVD factors into U3, V3, truncating from Ktot to K0:
        #
        #   S0 = SM[:K0]           (K0,)
        #   U0 = UM[:K0,:] . U3    (K0, nf_tj)
        #   V0 = VM[:K0,:] . V3    (K0, 2^{r-L} P)
        #
        # Then Abar = U3^T M V3 = (UM U3)^T diag(SM) (VM V3) ~= U0^T diag(S0) V0, and the
        # canonical form regains semiorthogonality: U0 U0^T = V0 V0^T = id_K0.

        S0 = sM[:K0]
        U0 = uM[:, :K0].T @ U3              # (K0, nf_tj)
        V0 = vMh[:K0, :] @ V3               # (K0, 2^{r-L} P)

        # Check that freq_variances[ifreq_lo : ifreq_lo + nf_tj] . U0^T . diag(S0) . V0
        # = ref_variance[j,:,:].

        coeffs = (U0 @ freq_variances[ifreq_lo : ifreq_lo + nf_tj]) * S0      # (K0,)
        recon5 = (coeffs @ V0).reshape(1 << (r-L), P)

        maxdiff5 = float(np.max(np.abs(recon5 / ref_variance[itree][j] - 1.0)))
        assert np.allclose(recon5, ref_variance[itree][j], rtol=1e-8, atol=0.0), (itree, j, maxdiff5)

        print(f'tree {itree}, {j=}: {ifreq_lo=}, {nf_tj=}, {2**(r-L)*P=}, {Ktot=}, {K0=}')
