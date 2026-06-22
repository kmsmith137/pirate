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
    """


    def __init__(self, r, k, f0, nf, nt, dbits, data):
        self.r, self.k = r, k
        self.f0, self.nf = f0, nf
        self.nt = nt
        self.dbits = list(dbits)
        self.data = data
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
        return SparseTreeArray(r=r, k=0, f0=f0, nf=nf, nt=1, dbits=[], data=data)


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
        method returns a SparseTreeArray representing the output, given a SparseTreeArray
        representing the input. We assume non-bit-reversed delay indices, and pad 'nt'
        as needed.

        The DD(k) recursion (with d = d'>>1, e = d'&1, lower-half shift d+e = ceil(d'/2)):

            out[F, d', t] = (1/sqrt2) ( in[2F, d, t-(d+e)] + in[2F+1, d, t] )

        merges input channels 2F (lower freq) and 2F+1 (upper freq) into output channel F.
        This works DIRECTLY on the compact (nf, 2^|dbits|, nt) representation: it never
        forms the dense 2^(r-k) f-axis or 2^k d-axis. We only build the 2^|dbits_out|
        delay slices that are actually stored (looping over them), gathering input rows
        with out-of-range channels zero-filled.
        """
        assert self.k < self.r, "iterate(): already at k == r"
        k = self.k
        f0_in, nf_in, nt_in, dbits_in = self.f0, self.nf, self.nt, self.dbits

        # Output coarse channels [F0, F0+nf_out): channel F merges input rows 2F and 2F+1.
        F0 = f0_in // 2
        nf_out = (f0_in + nf_in - 1) // 2 - F0 + 1
        base = 2 * F0 - f0_in                       # 0 if f0_in even, -1 if odd
        i_out = np.arange(nf_out)
        lower_idx = base + 2 * i_out                # input data-row for channel 2F  (may be out of range)
        upper_idx = lower_idx + 1                   # input data-row for channel 2F+1

        # dbits_out (decided from channel indices, not values): the lower-half time shift
        # ceil(d'/2) depends on the full delay, so if any lower channel is present the
        # output depends on every delay bit; otherwise (a single upper-only channel) the
        # input's bit dependence just shifts up by one (bit b of d -> bit b+1 of d').
        has_lower = bool(np.any((lower_idx >= 0) & (lower_idx < nf_in)))
        if has_lower:
            dbits_out = list(range(k, -1, -1))      # all k+1 bits
            max_shift = 1 << k                      # max ceil(d'/2) over d' in [0, 2^(k+1))
        else:
            dbits_out = [b + 1 for b in dbits_in]   # reverse-sorted; bits in [1, k]
            max_shift = 0
        m_out = len(dbits_out)

        rsqrt2 = 1.0 / np.sqrt(2.0)
        nt_alloc = nt_in + max_shift
        data_in = self.data.reshape(nf_in, 1 << len(dbits_in), nt_in)
        data_out = np.zeros((nf_out, 1 << m_out, nt_alloc), dtype=np.float64)

        for s_out in range(1 << m_out):
            dp = SparseTreeArray._representative_delay(s_out, dbits_out)   # representative d'
            d, e = dp >> 1, dp & 1
            shift = d + e
            slab = data_in[:, SparseTreeArray._selected_bits_index(d, dbits_in), :]   # B[:, d, :]
            data_out[:, s_out, :nt_in] += rsqrt2 * SparseTreeArray._gather_rows(slab, upper_idx)
            if has_lower:
                # Skip when no lower channels: 'shift' can be nonzero (from d) while
                # data_out is only nt_in wide, so writing the (all-zero) lower term would
                # raise a shape error.
                lower = SparseTreeArray._gather_rows(slab, lower_idx)
                data_out[:, s_out, shift:shift + nt_in] += rsqrt2 * lower

        # Trim trailing all-zero time samples (sign-agnostic: only drops genuinely-zero columns).
        nz_t = np.nonzero(np.any(data_out != 0.0, axis=(0, 1)))[0]
        nt_out = int(nz_t[-1]) + 1 if nz_t.size else 1
        data_out = np.ascontiguousarray(
            data_out[:, :, :nt_out].reshape((nf_out,) + (2,) * m_out + (nt_out,)))

        return SparseTreeArray(r=self.r, k=k + 1, f0=F0, nf=nf_out, nt=nt_out,
                               dbits=dbits_out, data=data_out)


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
        flat_idx = self._selected_bits_index(np.arange(nd_full), self.dbits)
        gathered = data_flat[:, flat_idx, :]                    # (nf, nd_full, nt)

        out = np.zeros((nf_full, nd_full, ntime), dtype=self.data.dtype)
        out[self.f0:self.f0 + self.nf, :, :self.nt] = gathered
        return out


    @staticmethod
    def _selected_bits_index(d, dbits):
        # Flat index into the 2^len(dbits) collapsed selected-bit axes, for delay value(s)
        # d (a python int or a numpy array). dbits[0] is the most significant bit (matches
        # data's C-order middle block). 'd * 0' seeds idx with d's type/shape (so an empty
        # dbits still yields an array when d is an array).
        idx = d * 0
        m = len(dbits)
        for j, b in enumerate(dbits):
            idx = idx | (((d >> b) & 1) << (m - 1 - j))
        return idx


    @staticmethod
    def _representative_delay(s, dbits):
        # Inverse of _selected_bits_index: the delay whose dbits-bits encode flat index s,
        # with all non-dbits bits set to zero.
        d = 0
        m = len(dbits)
        for j, b in enumerate(dbits):
            d |= ((s >> (m - 1 - j)) & 1) << b
        return d


    @staticmethod
    def _gather_rows(slab, idx):
        # Gather rows slab[idx] (slab has shape (nin, T)); out-of-range idx -> zero rows.
        out = np.zeros((len(idx), slab.shape[1]), dtype=slab.dtype)
        valid = (idx >= 0) & (idx < slab.shape[0])
        out[valid] = slab[idx[valid]]
        return out


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
    def _reference_gridding(channel_map, ifreq, ntime):
        # Apply ReferenceTreeGriddingKernel to a one-hot (ifreq, t=0) input; returns a
        # (1, ntree, ntime) float32 array. nfreq is inferred as round(channel_map[0]) (the
        # top edge of a full channel_map). Shared by the gridding + dedispersion tests.
        from ..kernels import ReferenceTreeGriddingKernel
        cm = np.ascontiguousarray(channel_map, dtype=np.float64)
        ntree = len(cm) - 1
        nfreq = int(round(float(cm[0])))
        assert 0 <= int(ifreq) < nfreq
        one_hot = np.zeros((1, nfreq, ntime), dtype=np.float32)
        one_hot[0, int(ifreq), 0] = 1.0
        kernel = ReferenceTreeGriddingKernel(nfreq=nfreq, nchan=ntree, ntime=ntime,
                                             beams_per_batch=1, channel_map=cm)
        return kernel.apply(one_hot)


    @staticmethod
    def _bit_reverse_permutation(rank):
        # perm[d] = bit_reverse(d, rank). ReferenceTree's output uses a bit-reversed delay
        # index; indexing its delay axis with this permutation restores the natural
        # (notes / SparseTreeArray) ordering.
        n = 1 << rank
        perm = np.zeros(n, dtype=np.intp)
        for d in range(n):
            x, b = d, 0
            for _ in range(rank):
                b = (b << 1) | (x & 1)
                x >>= 1
            perm[d] = b
        return perm


    @staticmethod
    def test_one_tree_gridding(channel_map, ifreq):
        """
        Compare make_tree_gridding_output(...).unpack() against ReferenceTreeGriddingKernel
        applied to a one-hot input, for a specific (channel_map, ifreq).
        """
        import ksgpu
        cm = np.ascontiguousarray(channel_map, dtype=np.float64)
        ntree = len(cm) - 1
        ntime = 32                                  # gridding kernel needs ntime % (1024/nbits) == 0 (nbits=32)

        ref = SparseTreeArray._reference_gridding(cm, ifreq, ntime)   # (1, ntree, ntime) float32
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


    @staticmethod
    def test_one_dedispersion(channel_map, ifreq):
        """
        Compare make_dedispersion_output(...).unpack() against
        ReferenceTreeGriddingKernel -> ReferenceTree (no subbands) on a one-hot input.
        """
        import ksgpu
        from ..kernels import ReferenceTree

        cm = np.ascontiguousarray(channel_map, dtype=np.float64)
        ntree = len(cm) - 1
        r = ntree.bit_length() - 1
        # ntime: a multiple of 32 (gridding-kernel constraint) and > 2^r, so the max
        # dispersion delay (2^r - 1) fits in a single ReferenceTree chunk -- a single chunk
        # with zeroed pstate then equals the non-incremental dedispersion.
        ntime = ((ntree + 31) // 32 + 1) * 32

        grid = SparseTreeArray._reference_gridding(cm, ifreq, ntime)   # (1, ntree, ntime) f32
        buf = np.ascontiguousarray(grid.reshape(1, 1, ntree, ntime))   # (1,1,ntree,ntime) f32
        tree = ReferenceTree(num_beams=1, amb_rank=0, dd_rank=r, ntime=ntime,
                             nspec=1, subband_counts=[1])
        tree.dedisperse(buf, None)                 # mutates buf IN PLACE; delay axis is bit-reversed
        perm = SparseTreeArray._bit_reverse_permutation(r)
        ref_natural = buf[0, 0][perm, :]           # (ntree, ntime), natural delay order

        sarr = SparseTreeArray.make_dedispersion_output(cm, ifreq)
        got = sarr.unpack(ntime)                    # (1, ntree, ntime) float64
        assert sarr.k == r and got.shape == (1, ntree, ntime)

        # All-non-negative computation -> epsabs=0 (zeros align structurally).
        ksgpu.assert_arrays_equal(ref_natural, got[0], "reftree", "got",
                                  ["delay", "time"], epsabs=0.0)


    @staticmethod
    def test_random_dedispersion():
        cm, ifreq = SparseTreeArray.random_channel_map()
        SparseTreeArray.test_one_dedispersion(cm, ifreq)
