import numpy as np


class SparseTreeArray:
    """
    Recall from notes/tree_dedispersion.tex that as tree dedispersion progresses,
    it operates on arrays of shape (2^(r-k), 2^k, ntime), indexed by (f,d,t).

    A SparseTreeArray represents a subset of such an array, in a specific
    representation which is designed to stay small as tree dedispersion is
    iterated. It can be "unpacked" to a dense 3-d array (which will usually
    be mostly zeroes).

    Members
    -------

      self.r, self.k:
        See above. Note that k=0 and k=r are both allowed.

      self.f0, self.nf:
        Array elements whose f-indices (where 0 <= f < 2^(r-k)) are outside
        the range self.f0 <= f < (self.f0 + self.nf) are zero.

      self.nt:
        Array elements whose t-indices are outside the range 0 <= t < nt are zero.

      self.dbits:
        A reverse-sorted list of integers [b0,b1,...] such that 0 <= bi < k.
        Suppose that we represent d by its base-2 digits [d_0, d_1, ..., d_{k-1}].
        Then, the array contents only depend on d via a subset d_{b0}, d_{b1}, ...
        of the digits.

      self.data:
        Shape (nf,2,...,2,nt) array, where the number of 2s is len(dbits).
        These are the (potentially) nonzero array elements.

        Convention: data middle axis (1+j) corresponds to dbits[j]. Equivalently,
        when the len(dbits) middle axes are flattened to length 2^len(dbits) in
        C-order, dbits[0] is the most significant bit. (unpack() relies on this,
        and any future iterate() must build 'data' to match.)

      self.initial_f0, self.initial_nf:
        These are the values of (f0,nf) when a k=0 SparseTreeArray is created via
        SparseTreeArray.make_tree_gridding_output(). Later, when SparseTreeArray.iterate()
        is called, a new SparseTreeArray is returned with new values of (f0,nf), but
        unmodified values of (initial_f0, initial_nf).
    """


    def __init__(self, r, k, f0, nf, nt, dbits, data, initial_f0, initial_nf):
        self.r, self.k = r, k
        self.f0, self.nf = f0, nf
        self.nt = nt
        self.dbits = list(dbits)
        self.data = data
        self.initial_f0, self.initial_nf = initial_f0, initial_nf
        self._check_invariants()


    def _check_invariants(self):
        assert 0 <= self.k <= self.r
        assert 0 <= self.f0 and self.nf >= 1 and self.f0 + self.nf <= 2**(self.r - self.k)
        assert self.nt >= 1
        assert all(0 <= b < self.k for b in self.dbits)
        assert self.dbits == sorted(self.dbits, reverse=True)
        assert len(set(self.dbits)) == len(self.dbits)
        expected = (self.nf,) + (2,) * len(self.dbits) + (self.nt,)
        assert self.data.shape == expected, (self.data.shape, expected)
        assert self.data.dtype == np.float64


    @staticmethod
    def make_tree_gridding_output(channel_map, ifreq):
        """
        Suppose the TreeGriddingKernel is called on a "one-hot" shape (nfreq,ntime)
        array whose (ifreq,0) entry is equal to 1. The output is a shape (2^rank, 1, ntime)
        array which is mostly zeros. This method returns an equivalent SparseTreeArray.
        """
        cm = np.ascontiguousarray(channel_map, dtype=np.float64)
        nchan = len(cm) - 1
        r = nchan.bit_length() - 1
        assert r >= 0 and nchan == (1 << r), "channel_map length must be 2^rank + 1"
        assert np.all(np.diff(cm) < 0.0), "channel_map must be strictly decreasing"
        ifreq = int(ifreq)
        assert ifreq >= 0

        # Tree channel n occupies [cm[n+1], cm[n]); it overlaps the one-hot freq bin
        # [ifreq, ifreq+1) iff cm[n] > ifreq and cm[n+1] < ifreq+1. Since cm is strictly
        # decreasing, the overlapping channels are a contiguous range [f0, f1); find its
        # edges with searchsorted on the ascending array (-cm).
        neg = -cm
        f1 = int(np.searchsorted(neg, -float(ifreq),     side='left'))        # first n: cm[n]   <= ifreq
        f0 = int(np.searchsorted(neg, -float(ifreq + 1), side='right')) - 1   # first n: cm[n+1] <  ifreq+1
        f0 = max(f0, 0)
        f1 = min(f1, nchan)
        assert f0 < f1, "ifreq does not overlap any tree channel"

        n = np.arange(f0, f1)
        w = np.minimum(cm[n], ifreq + 1.0) - np.maximum(cm[n + 1], float(ifreq))
        w = np.maximum(w, 0.0)                  # overlap weights (all > 0 across [f0, f1))
        data = w.reshape(-1, 1)                 # (nf, nt=1), float64

        nf = f1 - f0
        return SparseTreeArray(r=r, k=0, f0=f0, nf=nf, nt=1, dbits=[],
                               data=data, initial_f0=f0, initial_nf=nf)


    @staticmethod
    def make_dedispersion_output(channel_map, ifreq):
        """
        Suppose TreeGriddingKernel -> (Tree dedispersion) is called on a "one-hot"
        shape (nfreq,ntime) array whose (ifreq,0) entry is equal to 1. The output
        is a shape (1, 2^rank, ntime) array which is mostly zeros. This method returns
        an equivalent SparseTreeArray. We assume a non-bit-reversed delay index.
        """
        sarr = SparseTreeArray.make_tree_gridding_output(channel_map, ifreq)
        while sarr.k < sarr.r:
            sarr = sarr.iterate()
        return sarr


    def iterate(self):
        """
        The DD(k) operation defined in notes/tree_dedispersion.tex has input shape
        (2^(r-k), 2^k, ntime) and output shape (2^(r-k-1), 2^(k+1), ntime). This
        method returns a SpareTreeArray representing the output, given a SparseTreeArray
        representing the input. We assume non-bit-reversed delay indices, and pad 'nt'
        as needed.
        """
        pass


    def unpack(self, ntime):
        """
        Returns a dense array of shape (2^(r-k), 2^k, ntime), which will usually
        be mostly zeroes.

        The caller-specified 'ntime' must be >= nt (so that all nonzero elements fit).
        If not, we raise an exception.
        """
        if ntime < self.nt:
            raise RuntimeError(f"unpack: ntime={ntime} too small for nt={self.nt}")

        nf_full = 2**(self.r - self.k)
        nd_full = 2**self.k
        m = len(self.dbits)

        data_flat = self.data.reshape(self.nf, 2**m, self.nt)   # collapse selected-bit axes
        flat_idx = self._delay_to_flat_index(np.arange(nd_full))
        gathered = data_flat[:, flat_idx, :]                    # (nf, nd_full, nt)

        out = np.zeros((nf_full, nd_full, ntime), dtype=self.data.dtype)
        out[self.f0:self.f0 + self.nf, :, :self.nt] = gathered
        return out


    def _delay_to_flat_index(self, d):
        # Map each delay d to a flat index into the 2^m collapsed selected-bit axes.
        # dbits[0] is the most significant bit (matches data's C-order middle block).
        m = len(self.dbits)
        idx = np.zeros_like(d)
        for j, b in enumerate(self.dbits):
            idx |= ((d >> b) & 1) << (m - 1 - j)
        return idx


    @staticmethod
    def random_channel_map():
        """
        Generate a random (channel_map, ifreq) pair for the tree-gridding unit test.

        channel_map is a random strictly-decreasing length 2^rank+1 array with endpoints
        pinned to the band edges (channel_map[0]=nfreq, channel_map[-1]=0) and RANDOM
        interior edges (not linearly spaced -- more adversarial than a real
        DedispersionConfig channel_map). ifreq is uniform in [0, nfreq), so edge bins
        occur and are pinned by the fixed endpoints. (nfreq, rank) are recoverable from
        channel_map, so only (channel_map, ifreq) is returned.

        The "width" of freq channel ifreq -- how many tree channels the unit bin
        [ifreq, ifreq+1) spreads across -- is controlled to be log-spaced (sub-channel up
        to ~ntree): each interior edge lands in [ifreq, ifreq+1) with prob p, else
        uniformly in the rest of [0, nfreq]. The in-bin count is ~Binomial(ntree-1, p), so
        the width ~ (ntree-1)*p; drawing p log-uniformly makes the width log-spaced.
        """
        rank = int(np.random.randint(1, 8))        # 2^rank in [2, 128]
        ntree = 1 << rank
        nfreq = int(np.random.randint(2, 129))     # [2, 128]; >=2 so the "outside" region is nonempty
        ifreq = int(np.random.randint(0, nfreq))   # [0, nfreq-1], includes edge bins

        p = float(np.exp(np.random.uniform(np.log(0.01), np.log(1.0))))
        in_bin = np.random.uniform(0.0, 1.0, size=ntree - 1) < p

        edges = np.empty(ntree - 1, dtype=np.float64)
        edges[in_bin] = ifreq + np.random.uniform(0.0, 1.0, size=int(in_bin.sum()))
        # Out-of-bin: uniform on [0, nfreq] \ [ifreq, ifreq+1) via a shift past the bin
        # (handles edge bins, where one side of the complement is empty).
        u = np.random.uniform(0.0, nfreq - 1.0, size=int((~in_bin).sum()))
        edges[~in_bin] = np.where(u < ifreq, u, u + 1.0)

        cm = np.empty(ntree + 1, dtype=np.float64)
        cm[0] = nfreq
        cm[1:ntree] = np.sort(edges)[::-1]
        cm[ntree] = 0.0
        assert np.all(np.diff(cm) < 0.0), "degenerate random channel_map"   # ~never fires
        return cm, ifreq


    @staticmethod
    def test_one_tree_gridding(channel_map, ifreq):
        """
        Compare make_tree_gridding_output(...).unpack() against ReferenceTreeGriddingKernel
        applied to a one-hot input, for a specific (channel_map, ifreq). nfreq is inferred
        as round(channel_map[0]) (the top edge of a full channel_map).
        """
        import ksgpu
        from ..kernels import ReferenceTreeGriddingKernel

        cm = np.ascontiguousarray(channel_map, dtype=np.float64)
        ntree = len(cm) - 1
        nfreq = int(round(float(cm[0])))
        ntime = 32                                  # kernel needs ntime % (1024/nbits) == 0 (nbits=32);
                                                    # >1 also lightly exercises unpack's zero-padding
        ifreq = int(ifreq)
        assert 0 <= ifreq < nfreq

        one_hot = np.zeros((1, nfreq, ntime), dtype=np.float32)
        one_hot[0, ifreq, 0] = 1.0
        kernel = ReferenceTreeGriddingKernel(nfreq=nfreq, nchan=ntree, ntime=ntime,
                                             beams_per_batch=1, channel_map=cm)
        ref = kernel.apply(one_hot)                 # (1, ntree, ntime) float32

        sarr = SparseTreeArray.make_tree_gridding_output(cm, ifreq)
        got = sarr.unpack(ntime)                    # (ntree, 1, ntime) float64
        assert sarr.k == 0 and got.shape == (ntree, 1, ntime)

        # ref[0] and got[:,0,:] are both (ntree, ntime); raises with a verbose diff on mismatch.
        ksgpu.assert_arrays_equal(ref[0], got[:, 0, :], "ref", "got",
                                  ["tree", "time"], epsabs=0.0)


    @staticmethod
    def test_random_tree_gridding():
        cm, ifreq = SparseTreeArray.random_channel_map()
        SparseTreeArray.test_one_tree_gridding(cm, ifreq)
