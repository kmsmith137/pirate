import numpy as np


class SparseTreeTile:
    """
    A contiguous f-index range of a tree-dedispersion array of shape (2^(r-k), 2^k, ntime),
    indexed by (f,d,t), stored compactly. See notes/tree_dedispersion.tex.

    Members
    -------
      r, k:    rank and iteration index (0 <= k <= r).
      f0, nf:  the tile covers f-indices [f0, f0+nf); elements outside are zero.
      nt:      (pre-time-shift) time indices outside [0, nt) are zero.
      dbits:   reverse-sorted list of bits [b0>b1>...], 0 <= bi < k. The (pre-shift) data
               depends on the delay d only through the digits {d_{b0}, d_{b1}, ...}.
      data:    shape (nf, 2^len(dbits), nt), the pre-time-shift array. The middle axis packs
               the selected delay bits in C-order with dbits[0] as the most significant bit
               (the flat index is _selected_bits_index(d, dbits)).
      tshifts: length-k array, applied UNIFORMLY to every f-index of the tile before
               unpacking: unpacked[f,d,t] = data[f-f0, sel(d), t - t0 - T(d)] for all f,
               where T(d) = sum_i tshifts[i]*bit_i(d). (Because of the shift, 'data'
               depends on d only through 'dbits', but the unpacked array may depend on
               more digits.)
      t0:      delay- and f-independent constant forward time shift (>= 0); equivalently
               the data's pre-shift time origin, or a "constant tshift". Supported in all
               tile ops (unpack/slice/iterate_*); see notes/tree_dedispersion.tex.

    The constructor takes an optional trim=False: when True, all-zero leading/trailing time
    slices of 'data' are dropped (trailing shrinks nt; leading folds into t0), giving the
    minimal (nt, t0) for the same unpacked array.
    """

    def __init__(self, r, k, f0, nf, nt, dbits, data, tshifts, t0=0, trim=False):
        self.r, self.k = r, k
        self.f0, self.nf = f0, nf
        self.dbits = list(dbits)
        t0 = int(t0)
        if trim:
            # Drop all-zero leading/trailing time slices (over the f and delay-bit axes).
            # Trailing slices shrink nt; leading slices fold into t0 (the constant forward
            # shift), leaving unpack() unchanged but (nt, t0) minimal.
            data = np.asarray(data)
            nzt = np.nonzero(np.any(data != 0.0, axis=(0, 1)))[0]   # (f, delay-bit) axes
            lo, hi = (int(nzt[0]), int(nzt[-1]) + 1) if nzt.size else (0, 1)
            data = np.ascontiguousarray(data[..., lo:hi])
            t0 += lo
            nt = hi - lo
        self.nt = nt
        self.data = data
        self.tshifts = np.asarray(tshifts, dtype=np.int64)
        self.t0 = t0
        self._check_invariants()

    def _check_invariants(self):
        assert 0 <= self.k <= self.r
        assert 0 <= self.f0 and self.nf >= 1 and self.f0 + self.nf <= 2**(self.r - self.k)
        assert self.nt >= 1
        assert all(0 <= b < self.k for b in self.dbits)
        assert self.dbits == sorted(self.dbits, reverse=True)
        assert len(set(self.dbits)) == len(self.dbits)
        expected = (self.nf, 2 ** len(self.dbits), self.nt)
        assert self.data.shape == expected, (self.data.shape, expected)
        assert self.data.dtype == np.float64
        assert self.tshifts.shape == (self.k,)
        assert np.all(self.tshifts >= 0)
        assert self.t0 >= 0

    def slice(self, c0, c1):
        """
        Return the sub-tile for f-index range [c0, c1) (must lie within [f0, f0+nf)). The
        uniform tshifts make this a pure restriction of the data rows; (nt, dbits, tshifts)
        are inherited unchanged -- valid, but possibly non-minimal for the sub-range.
        (Passing trim=True to the constructor would re-minimize the time range.)
        """
        assert self.f0 <= c0 < c1 <= self.f0 + self.nf
        data = np.ascontiguousarray(self.data[c0 - self.f0 : c1 - self.f0])
        return SparseTreeTile(self.r, self.k, c0, c1 - c0, self.nt, self.dbits, data,
                              self.tshifts, t0=self.t0)

    def unpack(self, ntime):
        """
        Returns a dense (nf, 2^k, ntime) array for this tile's f-rows, applying the uniform
        per-delay time shift t0 + T(d) to every row. 'ntime' must be >= nt + t0 + max_d T(d).
        """
        nd_full = 2**self.k
        tshift = self._delay_tshift(np.arange(nd_full), self.tshifts)   # (nd_full,)
        nt_needed = self.nt + self.t0 + int(tshift.max())
        if ntime < nt_needed:
            raise RuntimeError(f"unpack: ntime={ntime} too small (need >= {nt_needed})")

        flat_idx = self._selected_bits_index(np.arange(nd_full), self.dbits)
        gathered = self.data[:, flat_idx, :]                           # (nf, nd_full, nt)

        out = np.zeros((self.nf, nd_full, ntime), dtype=self.data.dtype)
        for d in range(nd_full):
            sh = self.t0 + int(tshift[d])
            out[:, d, sh:sh + self.nt] = gathered[:, d, :]
        return out

    # ----------------------------- bit-index helpers -----------------------------

    @staticmethod
    def _selected_bits_index(d, dbits):
        # Flat index into the 2^len(dbits) collapsed selected-bit axes, for delay value(s)
        # d (python int or numpy array). dbits[0] is the most significant bit. 'd*0' seeds
        # idx with d's type/shape (so empty dbits yields an array when d is an array).
        idx = d * 0
        m = len(dbits)
        for j, b in enumerate(dbits):
            idx = idx | (((d >> b) & 1) << (m - 1 - j))
        return idx

    @staticmethod
    def _representative_delay(s, dbits):
        # Inverse of _selected_bits_index: the delay whose dbits-bits encode flat index s,
        # with all non-dbits bits zero.
        d = 0
        m = len(dbits)
        for j, b in enumerate(dbits):
            d |= ((s >> (m - 1 - j)) & 1) << b
        return d

    @staticmethod
    def _gather_rows(slab, idx):
        # Gather rows slab[idx] (slab shape (nin, T)); out-of-range idx -> zero rows.
        out = np.zeros((len(idx), slab.shape[1]), dtype=slab.dtype)
        valid = (idx >= 0) & (idx < slab.shape[0])
        out[valid] = slab[idx[valid]]
        return out

    @staticmethod
    def _delay_tshift(d, tshifts):
        # Total forward time shift T(d) = sum_i tshifts[i]*bit_i(d), for delay value(s) d
        # (python int or numpy array).
        T = d * 0
        for i, ti in enumerate(tshifts):
            T = T + (((d >> i) & 1) * ti)
        return T

    @staticmethod
    def _nonzero_bits(vec):
        # Set of bit-indices i where vec[i] != 0.
        return set(int(i) for i in np.nonzero(vec)[0])

    @staticmethod
    def _dd_tlo(k):
        # The DD(k) lower-half time shift ceil(d'/2) as a tshift vector (length k+1):
        # tlo[0]=1, tlo[j]=2^(j-1) for j>=1.
        return np.array([1] + [1 << (j - 1) for j in range(1, k + 1)], dtype=np.int64)

    # ----------------------------- tile-level DD(k) ops -----------------------------

    @staticmethod
    def iterate_aligned(tile):
        """
        DD(k) for an even-aligned tile (tile.f0 and tile.nf both even, so every output
        channel has both halves). The common input time shift is carried into the output
        tshifts ([0]+tin); the DD lower-half shift is baked, so the output uses all (k+1)
        delay bits. Returns the level-(k+1) output tile.
        """
        k = tile.k
        f0, nf, nt_in, dbits_in, tin = tile.f0, tile.nf, tile.nt, tile.dbits, tile.tshifts
        assert f0 % 2 == 0 and nf % 2 == 0 and nf >= 2, "iterate_aligned requires even f0, nf"
        assert k < tile.r

        F0 = f0 // 2
        nf_out = nf // 2
        dbits_out = list(range(k, -1, -1))             # all k+1 bits
        m_out = k + 1
        rsqrt2 = 1.0 / np.sqrt(2.0)
        nt_alloc = nt_in + (1 << k)
        data_in = tile.data                            # (nf, 2^|dbits_in|, nt_in)
        data_out = np.zeros((nf_out, 1 << m_out, nt_alloc), dtype=np.float64)
        for dp in range(1 << m_out):                   # representative is the identity (all bits)
            d = dp >> 1
            slab = data_in[:, SparseTreeTile._selected_bits_index(d, dbits_in), :]   # (nf, nt_in)
            gu = slab[1::2]                            # upper halves (2F+1), (nf_out, nt_in)
            gl = slab[0::2]                            # lower halves (2F)
            sh = (dp >> 1) + (dp & 1)                  # ceil(dp/2)
            data_out[:, dp, :nt_in] += rsqrt2 * gu
            data_out[:, dp, sh:sh + nt_in] += rsqrt2 * gl

        tshifts_out = np.concatenate(([0], tin)).astype(np.int64)
        # t0 is a uniform shift: it factors out of the DD sum, so it passes through.
        # trim=True drops leading/trailing all-zero time slices (leading folds into t0).
        return SparseTreeTile(tile.r, k + 1, F0, nf_out, nt_alloc, dbits_out, data_out,
                              tshifts_out, t0=tile.t0, trim=True)

    @staticmethod
    def iterate_singletons(lower, upper, require_aligned=True):
        """
        DD(k) merge of two adjacent singleton tiles into the output singleton. 'lower' is
        the lower-tree-freq half (gets the DD shift); 'upper' is the upper half. Each is a
        tile with nf==1 and its own (dbits, nt, tshifts, t0). Either may be None (but not
        both). Chooses tshifts/t0 (the elementwise/scalar min of the two halves' total
        shifts) to minimize the output (dbits, nt). Returns the level-(k+1) output tile.

        With require_aligned (default), 'lower' must be even-aligned (channels 2f, 2f+1) so
        the output is tree coarse-freq channel f -- the standard DD step. Case 2 of the
        subband extraction passes require_aligned=False to merge the odd-aligned pair
        (2f+1, 2f+2); the merge math is identical and the output tile is consumed directly
        (its f0 is then cosmetic).
        """
        assert lower is not None or upper is not None
        ref = lower if lower is not None else upper
        r, k = ref.r, ref.k
        assert k < r
        if lower is not None and upper is not None:
            assert (lower.r, lower.k) == (upper.r, upper.k)
            assert lower.f0 + 1 == upper.f0                 # adjacency
            if require_aligned:
                assert lower.f0 % 2 == 0                    # tree-channel semantics
        f_out = ref.f0 // 2
        assert (lower is None or lower.nf == 1) and (upper is None or upper.nf == 1)

        tlo = SparseTreeTile._dd_tlo(k)                # length k+1
        # Each present half's total time shift relative to its stored (pre-shift) data:
        #   lower gets the DD shift plus its own (lifted) input shift; upper gets only its.
        s_L = tlo + np.concatenate(([0], lower.tshifts)).astype(np.int64) if lower is not None else None
        s_U = np.concatenate(([0], upper.tshifts)).astype(np.int64) if upper is not None else None

        present = [s for s in (s_L, s_U) if s is not None]
        tmin = present[0].copy()
        for p in present[1:]:
            tmin = np.minimum(tmin, p)
        res_L = (s_L - tmin) if lower is not None else None
        res_U = (s_U - tmin) if upper is not None else None

        # Constant (t0) shift: absorb the common min into the output t0; each half's
        # residual constant (>= 0) folds into its data placement, exactly like res_L/res_U.
        t0_present = [t.t0 for t in (lower, upper) if t is not None]
        t0_out = min(t0_present)
        c_L = (lower.t0 - t0_out) if lower is not None else 0
        c_U = (upper.t0 - t0_out) if upper is not None else 0

        dbits_set = set()
        nt_alloc = 1
        if lower is not None:
            dbits_set |= set(b + 1 for b in lower.dbits) | SparseTreeTile._nonzero_bits(res_L)
            nt_alloc = max(nt_alloc, lower.nt + c_L + int(res_L.sum()))
        if upper is not None:
            dbits_set |= set(b + 1 for b in upper.dbits) | SparseTreeTile._nonzero_bits(res_U)
            nt_alloc = max(nt_alloc, upper.nt + c_U + int(res_U.sum()))
        dbits_out = sorted(dbits_set, reverse=True)
        m_out = len(dbits_out)

        rsqrt2 = 1.0 / np.sqrt(2.0)
        data_out = np.zeros((1, 1 << m_out, nt_alloc), dtype=np.float64)
        lo_flat = lower.data if lower is not None else None      # (1, 2^|dbits|, nt)
        up_flat = upper.data if upper is not None else None
        for s_out in range(1 << m_out):
            dp = SparseTreeTile._representative_delay(s_out, dbits_out)
            d = dp >> 1
            if lower is not None:
                rL = c_L + int(SparseTreeTile._delay_tshift(dp, res_L))
                col = lo_flat[0, SparseTreeTile._selected_bits_index(d, lower.dbits), :]
                data_out[0, s_out, rL:rL + lower.nt] += rsqrt2 * col
            if upper is not None:
                rU = c_U + int(SparseTreeTile._delay_tshift(dp, res_U))
                col = up_flat[0, SparseTreeTile._selected_bits_index(d, upper.dbits), :]
                data_out[0, s_out, rU:rU + upper.nt] += rsqrt2 * col

        # trim=True drops leading/trailing all-zero time slices (leading folds into t0_out).
        return SparseTreeTile(r, k + 1, f_out, 1, nt_alloc, dbits_out, data_out, tmin,
                              t0=t0_out, trim=True)

    @staticmethod
    def split_to_multiplets(tile, nlow, coarse_lag_coeff):
        """
        Split a singleton tile (nf==1) at level k into 2^nlow fully-iterated rank-(k-nlow)
        tiles, one per value 'e' of the low 'nlow' delay bits (the subband "fine index").

        The high rho = k-nlow delay bits become the new (coarse) delay axis; an extraction
        lag (coarse_lag_coeff * d_hi) is folded into the new tshifts; and the fine bits'
        own (constant) tshift contribution, T_lo(e) = sum_{i<nlow} tshifts[i] bit_i(e),
        becomes the per-multiplet t0. Returns a length-2^nlow list (index = fine index e),
        each a SparseTreeTile with r == k == rho, nf == 1. See notes/tree_dedispersion.tex.
        """
        assert tile.nf == 1
        assert 0 <= nlow <= tile.k
        C = int(coarse_lag_coeff)
        assert C >= 0
        k, rho = tile.k, tile.k - nlow
        high = [b for b in tile.dbits if b >= nlow]    # reverse-sorted (leading data axes)
        low = [b for b in tile.dbits if b < nlow]      # reverse-sorted (trailing data axes)
        nhigh, nlow_b = len(high), len(low)
        new_dbits = [b - nlow for b in high]           # reverse-sorted, in [0, rho)
        new_tshifts = np.array([tile.tshifts[nlow + j] + C * (1 << j) for j in range(rho)],
                               dtype=np.int64)
        # data (1, 2^|dbits|, nt) -> (1, 2^nhigh, 2^nlow_b, nt): high bits are the MSBs.
        data_resh = tile.data.reshape(1, 1 << nhigh, 1 << nlow_b, tile.nt)

        out = []
        for e in range(1 << nlow):
            low_idx = SparseTreeTile._selected_bits_index(e, low)
            data_e = np.ascontiguousarray(data_resh[:, :, low_idx, :])  # (1, 2^nhigh, nt)
            t0 = tile.t0 + int(SparseTreeTile._delay_tshift(e, tile.tshifts[:nlow]))
            out.append(SparseTreeTile(rho, rho, 0, 1, tile.nt, new_dbits, data_e,
                                      new_tshifts, t0=t0))
        return out


class SparseTreeArray:
    """
    A tree-dedispersion array of shape (2^(r-k), 2^k, ntime) over a contiguous f-index
    range [f0, f0+nf), represented as a list of SparseTreeTiles. The split lets the first
    and last f-index carry a different (smaller) sparsity pattern than the bulk:

      nf == 1:  1 tile  over [f0, f0+1)
      nf == 2:  2 tiles over [f0, f0+1), [f0+1, f0+2)
      nf  > 2:  3 tiles over [f0, f0+1), [f0+1, f0+nf-1), [f0+nf-1, f0+nf)

    All tiles share (r, k) but may differ in (nt, dbits, data, tshifts).
    """

    def __init__(self, r, k, f0, nf, tiles):
        self.r, self.k = r, k
        self.f0, self.nf = f0, nf
        self.tiles = list(tiles)
        self._check_invariants()

    def _check_invariants(self):
        assert 0 <= self.k <= self.r
        assert 0 <= self.f0 and self.nf >= 1 and self.f0 + self.nf <= 2**(self.r - self.k)
        bounds = self._tile_bounds(self.f0, self.nf)
        assert len(self.tiles) == len(bounds)
        for tile, (c0, c1) in zip(self.tiles, bounds):
            assert (tile.r, tile.k) == (self.r, self.k)
            assert (tile.f0, tile.nf) == (c0, c1 - c0)

    @staticmethod
    def _tile_bounds(f0, nf):
        # Canonical (c0, c1) tile boundaries for a range [f0, f0+nf).
        if nf == 1:
            return [(f0, f0 + 1)]
        if nf == 2:
            return [(f0, f0 + 1), (f0 + 1, f0 + 2)]
        return [(f0, f0 + 1), (f0 + 1, f0 + nf - 1), (f0 + nf - 1, f0 + nf)]

    @staticmethod
    def _from_tile(tile):
        # Build a canonical SparseTreeArray by splitting a single tile into 1/2/3 sub-tiles.
        bounds = SparseTreeArray._tile_bounds(tile.f0, tile.nf)
        tiles = [tile.slice(c0, c1) for (c0, c1) in bounds]
        return SparseTreeArray(tile.r, tile.k, tile.f0, tile.nf, tiles)

    def get_singleton(self, f, allow_none=False):
        """Return the singleton SparseTreeTile for f-index f. If f is out of [f0, f0+nf):
        return None when allow_none, else raise."""
        if not (self.f0 <= f < self.f0 + self.nf):
            if allow_none:
                return None
            raise IndexError(f"get_singleton: f={f} out of range [{self.f0}, {self.f0 + self.nf})")
        for tile in self.tiles:
            if tile.f0 <= f < tile.f0 + tile.nf:
                return tile.slice(f, f + 1)
        raise AssertionError("unreachable")

    @staticmethod
    def make_tree_gridding_output(channel_map, ifreq):
        """
        Suppose the TreeGriddingKernel is called on a "one-hot" shape (nfreq,ntime) array
        whose (ifreq,0) entry is 1. The output is a shape (2^rank, 1, ntime) array which is
        mostly zeros. This method returns an equivalent SparseTreeArray.
        """
        cm = np.ascontiguousarray(channel_map, dtype=np.float64)
        nchan = len(cm) - 1
        r = nchan.bit_length() - 1
        assert r >= 0 and nchan == (1 << r), "channel_map length must be 2^rank + 1"
        assert np.all(np.diff(cm) < 0.0), "channel_map must be strictly decreasing"
        ifreq = int(ifreq)
        assert ifreq >= 0

        neg = -cm
        f1 = int(np.searchsorted(neg, -float(ifreq),     side='left'))
        f0 = int(np.searchsorted(neg, -float(ifreq + 1), side='right')) - 1
        f0 = max(f0, 0)
        f1 = min(f1, nchan)
        assert f0 < f1, "ifreq does not overlap any tree channel"

        n = np.arange(f0, f1)
        w = np.minimum(cm[n], ifreq + 1.0) - np.maximum(cm[n + 1], float(ifreq))
        w = np.maximum(w, 0.0)
        data = w.reshape(-1, 1, 1)                     # (nf, 2^0=1, nt=1)
        tile = SparseTreeTile(r=r, k=0, f0=f0, nf=f1 - f0, nt=1, dbits=[], data=data,
                              tshifts=np.zeros(0, dtype=np.int64))
        return SparseTreeArray._from_tile(tile)

    @staticmethod
    def make_dedispersion_output(channel_map, ifreq):
        """
        Suppose TreeGriddingKernel -> (Tree dedispersion) is called on a "one-hot" shape
        (nfreq,ntime) array whose (ifreq,0) entry is 1. The output is a shape (1, 2^rank,
        ntime) array which is mostly zeros. This method returns an equivalent
        SparseTreeArray. We assume a non-bit-reversed delay index.
        """
        sarr = SparseTreeArray.make_tree_gridding_output(channel_map, ifreq)
        nf_start = sarr.nf
        while sarr.k < sarr.r:
            sarr = sarr.iterate()
        # Invariant: nf_start == nt_end, where nf_start is the gridding footprint width (the
        # number of adjacent tree channels the one-hot spreads across, at t=0) and nt_end is
        # the final tile's PRE-shift time extent. (At k==r, nf==1, so there is one tile.)
        #
        # Hand-waving argument: the tshifts absorb the per-delay bulk dispersion sweep, so
        # the stored nt measures only the RELATIVE time-smearing across the footprint. The
        # sweep is approximately linear in frequency, so a width-nf_start frequency footprint
        # maps onto a width-nf_start time smear at the maximum delay; hence the residual
        # (pre-shift) time footprint equals the spatial footprint. Equivalently, this is a
        # check that nt stays minimal (no padding) and the tshift/dbits machinery works as
        # intended.
        nt_end = sarr.tiles[0].nt
        assert nf_start == nt_end, (nf_start, nt_end)
        return sarr

    def iterate(self):
        """
        One DD(k) step. The first/last output channels (F0, Fmax) are computed with
        iterate_singletons (which absorbs shifts into tshifts to minimize dbits/nt); the
        bulk output channels [F0+1, Fmax) are computed with iterate_aligned on the
        even-aligned input sub-block [2F0+2, 2Fmax) (which lies inside the input middle
        tile). Returns a canonical SparseTreeArray at level k+1.
        """
        assert self.k < self.r, "iterate(): already at k == r"
        f0, nf = self.f0, self.nf
        F0 = f0 // 2
        last = f0 + nf - 1
        Fmax = last // 2
        nf_out = Fmax - F0 + 1

        tiles = [SparseTreeTile.iterate_singletons(
            self.get_singleton(2 * F0, allow_none=True),
            self.get_singleton(2 * F0 + 1, allow_none=True))]
        if nf_out >= 3:
            mid_in = self.tiles[1].slice(2 * F0 + 2, 2 * Fmax)
            tiles.append(SparseTreeTile.iterate_aligned(mid_in))
        if nf_out >= 2:
            tiles.append(SparseTreeTile.iterate_singletons(
                self.get_singleton(2 * Fmax, allow_none=True),
                self.get_singleton(2 * Fmax + 1, allow_none=True)))
        return SparseTreeArray(self.r, self.k + 1, F0, nf_out, tiles)

    def unpack(self, ntime):
        """Returns a dense (2^(r-k), 2^k, ntime) array, assembled from the tiles."""
        out = np.zeros((2**(self.r - self.k), 2**self.k, ntime), dtype=np.float64)
        for tile in self.tiles:
            out[tile.f0:tile.f0 + tile.nf] = tile.unpack(ntime)
        return out

    # ------------------------------- test utilities -------------------------------

    @staticmethod
    def random_channel_map():
        """
        Generate a random (channel_map, ifreq) pair for the tree-gridding/dedispersion
        tests. channel_map is a random strictly-decreasing length 2^rank+1 array with
        endpoints pinned to the band edges (channel_map[0]=nfreq, channel_map[-1]=0) and
        RANDOM interior edges; ifreq is uniform in [0, nfreq). The "width" of freq channel
        ifreq is log-spaced: each interior edge lands in [ifreq, ifreq+1) with prob p (else
        uniformly elsewhere), with p log-uniform, so the in-bin count ~ Binomial(ntree-1, p).
        """
        rank = int(np.random.randint(1, 8))        # 2^rank in [2, 128]
        ntree = 1 << rank
        nfreq = int(np.random.randint(2, 129))     # [2, 128]; >=2 so the "outside" region is nonempty
        ifreq = int(np.random.randint(0, nfreq))   # [0, nfreq-1], includes edge bins

        p = float(np.exp(np.random.uniform(np.log(0.01), np.log(1.0))))
        in_bin = np.random.uniform(0.0, 1.0, size=ntree - 1) < p

        edges = np.empty(ntree - 1, dtype=np.float64)
        edges[in_bin] = ifreq + np.random.uniform(0.0, 1.0, size=int(in_bin.sum()))
        u = np.random.uniform(0.0, nfreq - 1.0, size=int((~in_bin).sum()))
        edges[~in_bin] = np.where(u < ifreq, u, u + 1.0)

        cm = np.empty(ntree + 1, dtype=np.float64)
        cm[0] = nfreq
        cm[1:ntree] = np.sort(edges)[::-1]
        cm[ntree] = 0.0
        assert np.all(np.diff(cm) < 0.0), "degenerate random channel_map"
        return cm, ifreq

    @staticmethod
    def _reference_gridding(channel_map, ifreq, ntime):
        # ReferenceTreeGriddingKernel on a one-hot (ifreq, t=0) input; (1, ntree, ntime) f32.
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
        # perm[d] = bit_reverse(d, rank); un-bit-reverses a ReferenceTree delay axis.
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
    def _dense_dd(dense_in, k):
        # Reference dense DD(k): (nf, 2^k, ntime) -> (nf//2, 2^(k+1), ntime). nf even.
        nf, nd, ntime = dense_in.shape
        assert nd == 2**k and nf % 2 == 0
        rsqrt2 = 1.0 / np.sqrt(2.0)
        out = np.zeros((nf // 2, 2 * nd, ntime), dtype=np.float64)
        for dp in range(2 * nd):
            d = dp >> 1
            sh = (dp >> 1) + (dp & 1)
            out[:, dp, :] += rsqrt2 * dense_in[1::2, d, :]          # upper (2F+1), unshifted
            if sh < ntime:
                out[:, dp, sh:] += rsqrt2 * dense_in[0::2, d, :ntime - sh]   # lower (2F), shift sh
        return out

    @staticmethod
    def _random_tile(r, k, f0, nf):
        # A random valid SparseTreeTile with the given dims (non-negative data so the
        # structural tests can use epsabs=0). A random t0 is included so that the
        # iterate_* tests exercise nonzero t0 (guards against silent t0==0 assumptions).
        if k > 0:
            nbits = int(np.random.randint(0, k + 1))
            dbits = sorted((int(b) for b in np.random.choice(k, size=nbits, replace=False)), reverse=True)
            tshifts = np.random.randint(0, 4, size=k).astype(np.int64)
        else:
            dbits = []
            tshifts = np.zeros(0, dtype=np.int64)
        nt = int(np.random.randint(1, 5))
        t0 = int(np.random.randint(0, 4))
        shape = (nf, 2 ** len(dbits), nt)
        data = np.random.uniform(0.0, 1.0, size=shape).astype(np.float64)
        return SparseTreeTile(r, k, f0, nf, nt, dbits, data, tshifts, t0=t0)

    @staticmethod
    def test_one_tree_gridding(channel_map, ifreq):
        """Compare make_tree_gridding_output(...).unpack() against ReferenceTreeGriddingKernel."""
        import ksgpu
        cm = np.ascontiguousarray(channel_map, dtype=np.float64)
        ntree = len(cm) - 1
        ntime = 32                                  # gridding kernel needs ntime % (1024/nbits) == 0 (nbits=32)
        ref = SparseTreeArray._reference_gridding(cm, ifreq, ntime)   # (1, ntree, ntime) f32
        sarr = SparseTreeArray.make_tree_gridding_output(cm, ifreq)
        got = sarr.unpack(ntime)                    # (ntree, 1, ntime) f64
        assert sarr.k == 0 and got.shape == (ntree, 1, ntime)
        ksgpu.assert_arrays_equal(ref[0], got[:, 0, :], "ref", "got", ["tree", "time"], epsabs=0.0)

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
        # ntime: multiple of 32 (gridding kernel) and > 2^r, so the max dispersion delay
        # (2^r - 1) fits in one ReferenceTree chunk (-> non-incremental result).
        ntime = ((ntree + 31) // 32 + 1) * 32

        grid = SparseTreeArray._reference_gridding(cm, ifreq, ntime)   # (1, ntree, ntime) f32
        buf = np.ascontiguousarray(grid.reshape(1, 1, ntree, ntime))   # (1,1,ntree,ntime) f32
        tree = ReferenceTree(num_beams=1, amb_rank=0, dd_rank=r, ntime=ntime,
                             nspec=1, subband_counts=[1])
        tree.dedisperse(buf, None)                 # mutates buf IN PLACE; delay axis is bit-reversed
        perm = SparseTreeArray._bit_reverse_permutation(r)
        ref_natural = buf[0, 0][perm, :]           # (ntree, ntime), natural delay order

        sarr = SparseTreeArray.make_dedispersion_output(cm, ifreq)
        got = sarr.unpack(ntime)                    # (1, ntree, ntime) f64
        assert sarr.k == r and got.shape == (1, ntree, ntime)
        ksgpu.assert_arrays_equal(ref_natural, got[0], "reftree", "got", ["delay", "time"], epsabs=0.0)

    @staticmethod
    def test_random_dedispersion():
        cm, ifreq = SparseTreeArray.random_channel_map()
        SparseTreeArray.test_one_dedispersion(cm, ifreq)

    @staticmethod
    def test_random_iterate_aligned():
        """iterate_aligned(tile).unpack() must equal the dense DD(k) of tile.unpack()."""
        import ksgpu
        r = int(np.random.randint(2, 7))
        k = int(np.random.randint(0, r))            # 0 <= k < r
        nfull = 1 << (r - k)
        nf = 2 * int(np.random.randint(1, nfull // 2 + 1))
        f0 = 2 * int(np.random.randint(0, (nfull - nf) // 2 + 1))
        tile = SparseTreeArray._random_tile(r, k, f0, nf)
        ntime = tile.nt + tile.t0 + int(tile.tshifts.sum()) + (1 << k) + 8
        ref = SparseTreeArray._dense_dd(tile.unpack(ntime), k)        # (nf/2, 2^(k+1), ntime)
        got = SparseTreeTile.iterate_aligned(tile).unpack(ntime)
        ksgpu.assert_arrays_equal(ref, got, "ref", "got", ["f", "delay", "time"], epsabs=0.0)

    @staticmethod
    def test_random_iterate_singletons():
        """iterate_singletons(lower, upper).unpack() must equal the dense DD(k) merge."""
        import ksgpu
        r = int(np.random.randint(2, 7))
        k = int(np.random.randint(0, r))
        nfull = 1 << (r - k)
        f = int(np.random.randint(0, nfull // 2))   # output channel; 2f, 2f+1 in range
        mode = int(np.random.randint(0, 3))         # 0 both, 1 lower-only, 2 upper-only
        lower = SparseTreeArray._random_tile(r, k, 2 * f, 1) if mode != 2 else None
        upper = SparseTreeArray._random_tile(r, k, 2 * f + 1, 1) if mode != 1 else None
        merged = SparseTreeTile.iterate_singletons(lower, upper)

        need = 1
        if lower is not None:
            need = max(need, lower.nt + lower.t0 + int(lower.tshifts.sum()))
        if upper is not None:
            need = max(need, upper.nt + upper.t0 + int(upper.tshifts.sum()))
        ntime = need + (1 << k) + 8
        row_lo = lower.unpack(ntime)[0] if lower is not None else np.zeros((1 << k, ntime))
        row_up = upper.unpack(ntime)[0] if upper is not None else np.zeros((1 << k, ntime))
        dense_in = np.stack([row_lo, row_up])       # (2, 2^k, ntime)
        ref = SparseTreeArray._dense_dd(dense_in, k)        # (1, 2^(k+1), ntime)
        got = merged.unpack(ntime)
        ksgpu.assert_arrays_equal(ref, got, "ref", "got", ["f", "delay", "time"], epsabs=0.0)

    @staticmethod
    def test_single_channel_dbits():
        """
        A single tree channel iterated to k==r keeps dbits==[] at every level (the headline
        compactness property), and unpacks to the same impulse response as ReferenceTree.
        """
        import ksgpu
        from ..kernels import ReferenceTree
        r = int(np.random.randint(1, 8))
        ntree = 1 << r
        j = int(np.random.randint(0, ntree))
        tile = SparseTreeTile(r=r, k=0, f0=j, nf=1, nt=1, dbits=[],
                              data=np.ones((1, 1, 1)), tshifts=np.zeros(0, dtype=np.int64))
        sarr = SparseTreeArray._from_tile(tile)
        while sarr.k < sarr.r:
            sarr = sarr.iterate()
            assert sarr.nf == 1
            for t in sarr.tiles:
                assert t.dbits == [], (sarr.k, t.dbits)

        ntime = ((ntree + 31) // 32 + 1) * 32
        buf = np.zeros((1, 1, ntree, ntime), dtype=np.float32)
        buf[0, 0, j, 0] = 1.0
        ReferenceTree(num_beams=1, amb_rank=0, dd_rank=r, ntime=ntime,
                      nspec=1, subband_counts=[1]).dedisperse(buf, None)
        perm = SparseTreeArray._bit_reverse_permutation(r)
        ref_natural = buf[0, 0][perm, :]
        got = sarr.unpack(ntime)
        ksgpu.assert_arrays_equal(ref_natural, got[0], "reftree", "got", ["delay", "time"], epsabs=0.0)

    @staticmethod
    def test_random_split_to_multiplets():
        """split_to_multiplets(tile, nlow, C) vs. brute-force 'fix low bits, lag high bits'."""
        import ksgpu
        r = int(np.random.randint(2, 8))
        k = int(np.random.randint(1, r + 1))            # 1 <= k <= r
        f0 = int(np.random.randint(0, 1 << (r - k)))    # singleton coarse-freq index
        tile = SparseTreeArray._random_tile(r, k, f0, 1)
        nlow = int(np.random.randint(0, k + 1))         # 0 <= nlow <= k
        rho = k - nlow
        C = int(np.random.randint(0, 5))
        ntime = tile.nt + tile.t0 + int(tile.tshifts.sum()) + C * (1 << rho) + 8
        full = tile.unpack(ntime)[0]                    # (2^k, ntime)
        mts = SparseTreeTile.split_to_multiplets(tile, nlow, C)
        assert len(mts) == (1 << nlow)
        for e in range(1 << nlow):
            got = mts[e].unpack(ntime)[0]               # (2^rho, ntime)
            tgt = np.zeros((1 << rho, ntime), dtype=np.float64)
            for d_hi in range(1 << rho):
                D = d_hi * (1 << nlow) + e               # full delay (low bits = e)
                T = C * d_hi
                if T < ntime:
                    tgt[d_hi, T:] = full[D, :ntime - T]
            ksgpu.assert_arrays_equal(got, tgt, "got", "tgt", ["d_hi", "time"], epsabs=0.0)

    @staticmethod
    def test_one_subbanded_dedispersion(channel_map, ifreq, subband_counts):
        """
        Compare SparseSubbandedArray.make_dedispersion_output(...).unpack() against
        ReferenceTreeGriddingKernel -> ReferenceTree(subband_counts) on a one-hot input.
        """
        import ksgpu
        from ..kernels import ReferenceTree
        from ..pirate_pybind11 import FrequencySubbands

        cm = np.ascontiguousarray(channel_map, dtype=np.float64)
        ntree = len(cm) - 1
        r = ntree.bit_length() - 1
        sc = [int(c) for c in subband_counts]
        R = len(sc) - 1
        assert R <= r
        rho = r - R
        fs = FrequencySubbands(sc)
        M = fs.M
        # ntime: multiple of 32 (gridding) and comfortably > 2^(r+1) so the largest
        # (delay + lag) fits one non-incremental ReferenceTree chunk with no wraparound.
        ntime = (((3 << r) + 128) // 32 + 1) * 32

        grid = SparseTreeArray._reference_gridding(cm, ifreq, ntime)    # (1, ntree, ntime) f32
        buf = np.ascontiguousarray(grid.reshape(1, 1, ntree, ntime))   # (1,1,ntree,ntime) f32
        out_rt = np.zeros((1, 1 << rho, M, ntime), dtype=np.float32)
        ReferenceTree(num_beams=1, amb_rank=0, dd_rank=r, ntime=ntime,
                      nspec=1, subband_counts=sc).dedisperse(buf, out_rt)  # natural (d_hi, m)

        ssa = SparseSubbandedArray.make_dedispersion_output(cm, ifreq, fs)
        got = ssa.unpack(ntime)                                         # (2^rho, M, ntime) f64
        assert got.shape == (1 << rho, M, ntime)
        ksgpu.assert_arrays_equal(out_rt[0], got, "reftree", "got", ["d_hi", "m", "time"], epsabs=0.0)

    @staticmethod
    def test_random_subbanded_dedispersion():
        from ..pirate_pybind11 import FrequencySubbands
        cm, ifreq = SparseTreeArray.random_channel_map()
        r = (len(cm) - 1).bit_length() - 1
        R = int(np.random.randint(0, min(r, 4) + 1))    # pf_rank <= min(r, 4)
        sc = [int(c) for c in FrequencySubbands.make_random_subband_counts(R)]
        SparseTreeArray.test_one_subbanded_dedispersion(cm, ifreq, sc)

    @staticmethod
    def test_subbanded_reduces_to_fullband():
        """subband_counts=[1] must reproduce the full-band make_dedispersion_output."""
        import ksgpu
        from ..pirate_pybind11 import FrequencySubbands
        cm, ifreq = SparseTreeArray.random_channel_map()
        ntree = len(cm) - 1
        ntime = ((ntree + 31) // 32 + 1) * 32
        full = SparseTreeArray.make_dedispersion_output(cm, ifreq).unpack(ntime)   # (1, 2^r, ntime)
        ssa = SparseSubbandedArray.make_dedispersion_output(cm, ifreq, FrequencySubbands([1]))
        assert len(ssa.per_m) == 1
        got = ssa.per_m[0].unpack(ntime)                                          # (1, 2^r, ntime)
        ksgpu.assert_arrays_equal(full, got, "full", "subband[1]", ["f", "delay", "time"], epsabs=0.0)


class SparseSubbandedArray:
    """
    Sparse representation of a SUBBANDED tree-dedisperser's output for a one-hot input.
    The dense output has shape (2^(r-R), M, ntime) (notes Section "Subbanded
    dedispersion"), held as a length-M list 'per_m' of SparseTreeTiles, each a rank-(r-R),
    fully-iterated (k == r-R), nf==1 tile carrying that multiplet's 2^(r-R) coarse delays.

    Members
    -------
      r, R:            tree rank and pf_rank (rho = r - R is the per-m tile rank).
      subband_counts:  the length-(R+1) C_l array.
      per_m:           length-M list of SparseTreeTile (r == k == rho, nf == 1).
    """

    def __init__(self, r, R, subband_counts, per_m):
        self.r, self.R = r, R
        self.subband_counts = [int(c) for c in subband_counts]
        self.per_m = list(per_m)
        self._check_invariants()

    def _check_invariants(self):
        assert 0 <= self.R <= self.r
        rho = self.r - self.R
        M = sum((1 << l) * c for l, c in enumerate(self.subband_counts))
        assert len(self.per_m) == M, (len(self.per_m), M)
        for t in self.per_m:
            assert isinstance(t, SparseTreeTile)
            assert (t.r, t.k, t.nf) == (rho, rho, 1), (t.r, t.k, t.nf, rho)

    def unpack(self, ntime):
        """Returns the dense (2^(r-R), M, ntime) output array."""
        rho = self.r - self.R
        out = np.zeros((1 << rho, len(self.per_m), ntime), dtype=np.float64)
        for m, tile in enumerate(self.per_m):
            out[:, m, :] = tile.unpack(ntime)[0]
        return out

    @staticmethod
    def _zero_tile(rho):
        # A zero rank-rho, k==rho, nf==1 tile (for subbands outside the footprint).
        return SparseTreeTile(rho, rho, 0, 1, 1, [], np.zeros((1, 1, 1), dtype=np.float64),
                              np.zeros(rho, dtype=np.int64))

    @staticmethod
    def _emit(per_m, mbase, l, rho, tile, C):
        # Fill the 2^l multiplets of one subband into per_m.
        if tile is None:                                        # subband outside footprint
            for e in range(1 << l):
                per_m[mbase + e] = SparseSubbandedArray._zero_tile(rho)
        else:
            tiles = SparseTreeTile.split_to_multiplets(tile, l, C)
            for e in range(1 << l):
                per_m[mbase + e] = tiles[e]

    @staticmethod
    def make_dedispersion_output(channel_map, ifreq, fs):
        """
        Suppose TreeGriddingKernel -> (subbanded tree dedispersion) is applied to a one-hot
        shape (nfreq, ntime) input whose (ifreq, 0) entry is 1. The output is a mostly-zero
        shape (2^(r-R), M, ntime) array; this returns an equivalent SparseSubbandedArray.
        We assume non-bit-reversed coarse-delay (d_hi) and multiplet (m) indices.

        'fs' is a FrequencySubbands object defining the subband scheme.

        Implementation (notes Section "Subbanded dedispersion"): iterate a single
        under-the-hood SparseTreeArray (the full-band gridding footprint) and extract per-
        multiplet outputs "on the fly". Case 1 (aligned, l=0 or even s) reads the
        level-(r-R+l) singleton f directly; Case 2 (half-aligned, l>0 odd s) merges the
        level-(r-R+l-1) pair (2f+1, 2f+2) via iterate_singletons(require_aligned=False).
        Each extraction is then split into its 2^l multiplets (split_to_multiplets).
        """
        sc = [int(c) for c in fs.subband_counts]
        sarr = SparseTreeArray.make_tree_gridding_output(channel_map, ifreq)
        r = sarr.r
        R = fs.pf_rank
        rho = r - R
        assert 0 <= R <= r, (R, r)
        per_m = [None] * fs.M

        # Per subband: (l, f_blk, case1, mbase), derived from its coarse i-range [ilo, ihi).
        subs = []
        for f in range(fs.F):
            ilo, ihi = int(fs.f_to_ilo[f]), int(fs.f_to_ihi[f])
            l = (ihi - ilo).bit_length() - 1               # band width is 2^l
            f_blk = ilo >> l                               # coarse block index
            case1 = (ilo & ((1 << l) - 1)) == 0            # aligned (always true for l==0)
            subs.append((l, f_blk, case1, int(fs.f_to_mbase[f])))

        for k in range(0, r + 1):
            for (l, f_blk, case1, mbase) in subs:
                if case1 and (r - R + l) == k:                  # Case 1: read level k
                    tile = sarr.get_singleton(f_blk, allow_none=True)
                    C = (1 << l) * ((1 << (R - l)) - 1 - f_blk)
                    SparseSubbandedArray._emit(per_m, mbase, l, rho, tile, C)
                elif (not case1) and (r - R + l - 1) == k:      # Case 2: read level r-R+l-1
                    lo = sarr.get_singleton(2 * f_blk + 1, allow_none=True)
                    up = sarr.get_singleton(2 * f_blk + 2, allow_none=True)
                    C = (1 << (l - 1)) * ((1 << (R - l + 1)) - 2 * f_blk - 3)
                    if lo is None and up is None:
                        tile = None
                    else:
                        tile = SparseTreeTile.iterate_singletons(lo, up, require_aligned=False)
                    SparseSubbandedArray._emit(per_m, mbase, l, rho, tile, C)
            if k < r:
                sarr = sarr.iterate()

        return SparseSubbandedArray(r, R, sc, per_m)
