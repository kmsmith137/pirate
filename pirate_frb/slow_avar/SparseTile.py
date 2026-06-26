import numpy as np

from ..utils import integer_log2


#######################################   class SparseTile   #######################################


class SparseTile:
    """
    A contiguous f-index range of a tree-dedispersion array of shape (2^(r-k), 2^k, ntime),
    indexed by (f,d,t), stored compactly. See notes/tree_dedispersion.tex.

    Members
    -------
      r, k:    rank and iteration index (0 <= k <= r).
      f0, nf:  the tile covers f-indices [f0, f0+nf); elements outside are zero.
      nt:      (pre-time-shift) time indices outside [0, nt) are zero.
      dbits:   integer bitmask of selected delay bits: bit b set (for 0 <= b < k) means the
               (pre-shift) data depends on digit d_b. So data depends on the delay d only
               through the digits {d_b : bit b of dbits is set}.
      data:    shape (nf, 2^popcount(dbits), nt), the pre-time-shift array. The middle axis
               packs the selected delay bits in C-order, with the HIGHEST selected bit as the
               most significant (the flat index is _remap_d(d, (1<<k)-1, dbits)).
      tshifts: length-k array, applied UNIFORMLY to every f-index of the tile before
               unpacking: unpacked[f,d,t] = data[f-f0, sel(d), t - t0 - T(d)] for all f,
               where T(d) = sum_i tshifts[i]*bit_i(d). (Because of the shift, 'data'
               depends on d only through 'dbits', but the unpacked array may depend on
               more digits.)
      t0:      delay- and f-independent constant forward time shift (>= 0); equivalently
               the data's pre-shift time origin, or a "constant tshift". Supported in all
               tile ops (unpack/slice/iterate_*); see notes/tree_dedispersion.tex.
    """

    def __init__(self, r, k, f0, nf, nt, dbits, data, tshifts, t0=0):
        self.r, self.k = r, k
        self.f0, self.nf = f0, nf
        self.dbits = int(dbits)
        self.nt = nt
        self.data = data
        self.tshifts = np.asarray(tshifts, dtype=np.int64)
        self.t0 = int(t0)
        self._check_invariants()

    def _check_invariants(self):
        assert 0 <= self.k <= self.r
        assert 0 <= self.f0 and self.nf >= 1 and self.f0 + self.nf <= 2**(self.r - self.k)
        assert self.nt >= 1
        assert 0 <= self.dbits < (1 << self.k)
        assert self.data.shape == (self.nf, 1 << self.dbits.bit_count(), self.nt)
        assert self.data.dtype == np.float64
        assert self.tshifts.shape == (self.k,)
        assert np.all(self.tshifts >= 0)
        assert self.t0 >= 0

    def slice(self, c0, c1):
        """
        Return the sub-tile for f-index range [c0, c1) (must lie within [f0, f0+nf)). The
        uniform tshifts make this a pure restriction of the data rows; (nt, dbits, tshifts)
        are inherited unchanged -- valid, but possibly non-minimal for the sub-range.
        """
        assert self.f0 <= c0 < c1 <= self.f0 + self.nf
        data = np.ascontiguousarray(self.data[c0 - self.f0 : c1 - self.f0])
        return SparseTile(self.r, self.k, c0, c1 - c0, self.nt, self.dbits, data,
                          self.tshifts, t0=self.t0)

    def unpack(self, ntime):
        """
        Returns a dense (nf, 2^k, ntime) array for this tile's f-rows, applying the uniform
        per-delay time shift t0 + T(d) to every row. 'ntime' must be >= nt + t0 + max_d T(d).
        """
        nd_full = 2**self.k
        tshift = self._eval_tshifts(np.arange(nd_full), nd_full - 1, self.tshifts)   # (nd_full,)
        nt_needed = self.nt + self.t0 + int(tshift.max())
        if ntime < nt_needed:
            raise RuntimeError(f"unpack: ntime={ntime} too small (need >= {nt_needed})")

        flat_idx = self._remap_d(np.arange(nd_full), (1 << self.k) - 1, self.dbits)
        gathered = self.data[:, flat_idx, :]                           # (nf, nd_full, nt)

        out = np.zeros((self.nf, nd_full, ntime), dtype=self.data.dtype)
        for d in range(nd_full):
            sh = self.t0 + int(tshift[d])
            out[:, d, sh:sh + self.nt] = gathered[:, d, :]
        return out

    # ----------------------------- bit-index helpers -----------------------------

    @staticmethod
    def _remap_d(d, dbits_in, dbits_out):
        # Reinterpret 'd' (a flat index in the 2^popcount(dbits_in) packing) as a flat index in the
        # 2^popcount(dbits_out) packing, where dbits_out is a subset of dbits_in. Both packings are
        # highest-bit-first: a set bit's packed position is popcount of the selected bits below it.
        # Vectorized in d (python int or numpy array); dbits_in, dbits_out are scalar python ints.
        # Written to port to C++ line-by-line (popcount / bit_floor intrinsics).
        assert (~dbits_in & dbits_out) == 0                     # dbits_out subset of dbits_in
        assert np.all((np.asarray(d) >= 0) & (np.asarray(d) < (1 << dbits_in.bit_count())))
        dout = d * 0                                            # seed type/shape from d
        tmp = dbits_out
        while tmp:
            bout = 1 << (tmp.bit_length() - 1)                 # highest set bit of tmp (C++ bit_floor)
            tmp &= ~bout
            shift_in = (dbits_in & (bout - 1)).bit_count()     # this bit's packed position in d
            shift_out = (dbits_out & (bout - 1)).bit_count()   # this bit's packed position in dout
            dout = dout | (((d >> shift_in) & 1) << shift_out)
        return dout

    @staticmethod
    def _eval_tshifts(d, dbits, tshifts):
        """
        The 'd' arg satisfies 0 <= d < 2^popcount(dbits), and 'tshifts' is a length-k array.
        Returns the associated forward time shift T, obtained summing tshifts for each bit that has been set.
        Vectorized: 'd' can be an int or a numpy array.
        """
        # 'd' is packed over dbits (highest-bit-first), so bit b of the full delay is d's packed
        # bit at position popcount(dbits below b); non-selected bits are zero (contribute nothing).
        T = d * 0
        tmp = dbits
        while tmp:
            b = tmp.bit_length() - 1                       # highest set bit position (C++ bit_floor)
            tmp &= ~(1 << b)
            shift = (dbits & ((1 << b) - 1)).bit_count()   # bit b's packed position in d
            T = T + (((d >> shift) & 1) * tshifts[b])
        return T

    @staticmethod
    def _dd_tshifts(k):
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
        dbits_out = (1 << (k + 1)) - 1                 # all k+1 bits
        m_out = k + 1
        rsqrt2 = 1.0 / np.sqrt(2.0)
        nt_alloc = nt_in + (1 << k)
        data_in = tile.data                            # (nf, 2^popcount(dbits_in), nt_in)
        data_out = np.zeros((nf_out, 1 << m_out, nt_alloc), dtype=np.float64)
        for dp in range(1 << m_out):                   # representative is the identity (all bits)
            d = dp >> 1
            slab = data_in[:, SparseTile._remap_d(d, (1 << k) - 1, dbits_in), :]   # (nf, nt_in)
            gu = slab[1::2]                            # upper halves (2F+1), (nf_out, nt_in)
            gl = slab[0::2]                            # lower halves (2F)
            sh = (dp >> 1) + (dp & 1)                  # ceil(dp/2)
            data_out[:, dp, :nt_in] += rsqrt2 * gu
            data_out[:, dp, sh:sh + nt_in] += rsqrt2 * gl

        tshifts_out = np.concatenate(([0], tin)).astype(np.int64)
        # t0 is a uniform shift: it factors out of the DD sum, so it passes through.
        return SparseTile(tile.r, k + 1, F0, nf_out, nt_alloc, dbits_out, data_out,
                          tshifts_out, t0=tile.t0)

    @staticmethod
    def iterate_singletons(lower, upper, require_aligned=True):
        """
        DD(k) merge of two adjacent singleton tiles into the output singleton. 'lower' is
        the lower-tree-freq half (gets the DD shift); 'upper' is the upper half. Each is a
        tile with nf==1 and its own (dbits, nt, tshifts, t0). Either may be None (but not
        both); the single-half cases delegate to _iterate_lower / _iterate_upper. With both
        present, chooses tshifts/t0 (the elementwise/scalar min of the two halves' total
        shifts) to minimize the output (dbits, nt). Returns the level-(k+1) output tile.

        With require_aligned (default), 'lower' must be even-aligned (channels 2f, 2f+1) so
        the output is tree coarse-freq channel f -- the standard DD step. Case 2 of the
        subband extraction passes require_aligned=False to merge the odd-aligned pair
        (2f+1, 2f+2); the merge math is identical and the output tile is consumed directly
        (its f0 is then cosmetic).
        """
        assert lower is not None or upper is not None
        if upper is None:
            return SparseTile._iterate_lower(lower)
        if lower is None:
            return SparseTile._iterate_upper(upper)

        # Both halves present: the standard aligned DD(k) merge.
        assert (lower.r, lower.k) == (upper.r, upper.k)
        assert lower.nf == 1 and upper.nf == 1
        r, k = lower.r, lower.k
        assert k < r
        assert lower.f0 + 1 == upper.f0                 # adjacency
        if require_aligned:
            assert lower.f0 % 2 == 0                    # tree-channel semantics
        f_out = lower.f0 // 2

        tlo = SparseTile._dd_tshifts(k)                     # length k+1
        # Each half's total time shift relative to its stored (pre-shift) data: lower gets the
        # DD shift plus its own (lifted) input shift; upper gets only its own.
        s_L = tlo + np.concatenate(([0], lower.tshifts)).astype(np.int64)
        s_U = np.concatenate(([0], upper.tshifts)).astype(np.int64)
        tmin = np.minimum(s_L, s_U)
        res_L, res_U = s_L - tmin, s_U - tmin

        # Constant (t0) shift: absorb the common min into the output t0; each half's residual
        # constant (>= 0) folds into its data placement, exactly like res_L/res_U.
        t0_out = min(lower.t0, upper.t0)
        c_L, c_U = lower.t0 - t0_out, upper.t0 - t0_out

        # 'dbits + 1' (lifting every selected bit one level) is a left shift on the mask.
        dbits_out = (lower.dbits | upper.dbits) << 1
        for i in np.nonzero(res_L + res_U)[0]:
            dbits_out |= (1 << int(i))
        
        nt_alloc = max(lower.nt + c_L + int(res_L.sum()), upper.nt + c_U + int(res_U.sum()))
        m_out = dbits_out.bit_count()

        rsqrt2 = 1.0 / np.sqrt(2.0)
        data_out = np.zeros((1, 1 << m_out, nt_alloc), dtype=np.float64)
        ldb, udb = lower.dbits << 1, upper.dbits << 1      # each half's selected bits, lifted (subset of dbits_out)
        for s_out in range(1 << m_out):
            rL = c_L + int(SparseTile._eval_tshifts(s_out, dbits_out, res_L))
            col = lower.data[0, SparseTile._remap_d(s_out, dbits_out, ldb), :]
            data_out[0, s_out, rL:rL + lower.nt] += rsqrt2 * col
            rU = c_U + int(SparseTile._eval_tshifts(s_out, dbits_out, res_U))
            col = upper.data[0, SparseTile._remap_d(s_out, dbits_out, udb), :]
            data_out[0, s_out, rU:rU + upper.nt] += rsqrt2 * col

        return SparseTile(r, k + 1, f_out, 1, nt_alloc, dbits_out, data_out, tmin,
                          t0=t0_out)

    @staticmethod
    def _iterate_lower(lower):
        """
        iterate_singletons() with upper=None: DD(k) on just the lower half (channel 2f).
        The lower half takes the DD shift, but with no upper half to align against, that shift
        folds entirely into tshifts (the residual is zero), so the output is (1/sqrt2)*lower.data
        with every selected bit lifted one level (dbits << 1). No per-delay loop is needed.
        """
        k = lower.k
        assert lower.nf == 1 and k < lower.r
        rsqrt2 = 1.0 / np.sqrt(2.0)
        tshifts_out = SparseTile._dd_tshifts(k) + np.concatenate(([0], lower.tshifts)).astype(np.int64)
        data_out = np.ascontiguousarray(rsqrt2 * lower.data)
        return SparseTile(lower.r, k + 1, lower.f0 // 2, 1, lower.nt, lower.dbits << 1, data_out,
                          tshifts_out, t0=lower.t0)

    @staticmethod
    def _iterate_upper(upper):
        """
        iterate_singletons() with lower=None: DD(k) on just the upper half (channel 2f+1).
        The upper half gets no DD shift, so the new bit 0 is a pure spectator (tshift 0) and the
        output is (1/sqrt2)*upper.data with every selected bit lifted one level (dbits << 1).
        No per-delay loop is needed.
        """
        k = upper.k
        assert upper.nf == 1 and k < upper.r
        rsqrt2 = 1.0 / np.sqrt(2.0)
        tshifts_out = np.concatenate(([0], upper.tshifts)).astype(np.int64)
        data_out = np.ascontiguousarray(rsqrt2 * upper.data)
        return SparseTile(upper.r, k + 1, upper.f0 // 2, 1, upper.nt, upper.dbits << 1, data_out,
                          tshifts_out, t0=upper.t0)

    def specialize_dbits(self, value, nbits, *, low):
        """
        Specialize this singleton tile's (nf==1) level-k DM (delay) index by fixing 'nbits' of
        its delay bits to the value 'value' (0 <= value < 2^nbits), keeping the other (k-nbits)
        bits. Collapses to a standalone fully-iterated rank-(k-nbits) SparseTile (r == k == rho,
        f0 == 0, nf == 1).

        If low=True, the LOW 'nbits' bits are fixed; the high rho = k-nbits bits become the new
        (coarse) delay axis, inheriting their tshifts. Looping 'value' over 0..2^nbits-1 then
        yields the subband's 2^nbits multiplets.
        If low=False, the HIGH 'nbits' bits are fixed, keeping the low (k-nbits) bits. Used to
        discard part of the DM range: e.g. specialize_dbits(1, 1, low=False) keeps the upper half
        of the delay axis (top bit == 1), dropping rank by 1.

        Only the SELECTED fixed bits affect the data; non-selected fixed bits affect 'value' only
        via t0. The fixed bits' (constant) tshift contribution folds into the tile's t0; the kept
        bits' tshifts carry over. See notes/tree_dedispersion.tex.
        """
        assert self.nf == 1
        assert 0 <= nbits <= self.k
        assert 0 <= value < (1 << nbits)
        
        # The data middle axis packs all dbits MSB-first as (1, 2^nhigh, 2^nlow, nt) (high group
        # leading); we slice out the fixed group's index and keep the other group's axis.
        b = nbits if low else (self.k - nbits)         # split: low group = bits [0,b), high group = [b,k)
        low_mask = self.dbits & ((1 << b) - 1)         # selected bits in [0, b)
        high_mask = self.dbits >> b                    # selected bits in [b, k), shifted to [0, k-b)
        nlow, nhigh = low_mask.bit_count(), high_mask.bit_count()
        data_resh = self.data.reshape(1, 1 << nhigh, 1 << nlow, self.nt)
        
        if low:
            idx = SparseTile._remap_d(value, (1 << nbits) - 1, low_mask)
            data_sel = data_resh[:, :, idx, :]         # keep high (leading) axis -> (1, 2^nhigh, nt)
            new_dbits, new_tshifts = high_mask, self.tshifts[nbits:]
            removed_dbits = (1 << nbits) - 1
        else:
            idx = SparseTile._remap_d(value, (1 << nbits) - 1, high_mask)
            data_sel = data_resh[:, idx, :, :]         # keep low (trailing) axis -> (1, 2^nlow, nt)
            new_dbits, new_tshifts = low_mask, self.tshifts[:b]
            removed_dbits = (1 << self.k) - (1 << (self.k - nbits))
            
        rho = self.k - nbits
        new_t0 = self.t0 + int(SparseTile._eval_tshifts(value, removed_dbits, self.tshifts))
        
        return SparseTile(rho, rho, 0, 1, self.nt, new_dbits,
                          np.ascontiguousarray(data_sel),
                          np.array(new_tshifts, dtype=np.int64), t0=new_t0)

    # ------------------------------- test utilities -------------------------------

    @staticmethod
    def make_random(r, k, f0, nf):
        # A random valid SparseTile with the given dims (non-negative data so the
        # structural tests can use epsabs=0). A random t0 is included so that the
        # iterate_* tests exercise nonzero t0 (guards against silent t0==0 assumptions).
        dbits = int(np.random.randint(0, 1 << k))       # random subset of bits [0, k)
        tshifts = np.random.randint(0, 4, size=k).astype(np.int64)
        nt = int(np.random.randint(1, 5))
        t0 = int(np.random.randint(0, 4))
        shape = (nf, 1 << dbits.bit_count(), nt)
        data = np.random.uniform(0.0, 1.0, size=shape).astype(np.float64)
        return SparseTile(r, k, f0, nf, nt, dbits, data, tshifts, t0=t0)

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
    def test_random_remap_d():
        """_remap_d vs a brute-force 'spread over dbits_in, re-extract dbits_out' reference."""
        n = int(np.random.randint(0, 9))                 # 0..8 total bits
        dbits_in = int(np.random.randint(0, 1 << n))     # any mask over bits [0, n)
        dbits_out = 0                                    # a random subset of dbits_in
        for b in range(n):
            if (dbits_in >> b) & 1 and np.random.rand() < 0.5:
                dbits_out |= (1 << b)
        p_in = dbits_in.bit_count()
        d = np.arange(1 << p_in, dtype=np.int64)         # all packed inputs, vectorized
        got = SparseTile._remap_d(d, dbits_in, dbits_out)
        # Brute-force reference (independent of _remap_d): spread each packed index into a full
        # delay over dbits_in (MSB-first), then re-extract the dbits_out bits (also MSB-first).
        ref = np.zeros_like(d)
        for s in range(1 << p_in):
            D, sh = 0, p_in - 1
            for b in reversed(range(n)):
                if (dbits_in >> b) & 1:
                    D |= ((s >> sh) & 1) << b; sh -= 1
            out, sh = 0, dbits_out.bit_count() - 1
            for b in reversed(range(n)):
                if (dbits_out >> b) & 1:
                    out |= ((D >> b) & 1) << sh; sh -= 1
            ref[s] = out
        assert np.array_equal(got, ref), (n, dbits_in, dbits_out, list(got), list(ref))
        # Scalar (python int) path: _remap_d(0b101, 0b111, 0b101) keeps bits {2,0} -> 0b11.
        assert SparseTile._remap_d(0b101, 0b111, 0b101) == 0b11

    @staticmethod
    def test_random_iterate_aligned():
        """iterate_aligned(tile).unpack() must equal the dense DD(k) of tile.unpack()."""
        import ksgpu
        r = int(np.random.randint(2, 7))
        k = int(np.random.randint(0, r))            # 0 <= k < r
        nfull = 1 << (r - k)
        nf = 2 * int(np.random.randint(1, nfull // 2 + 1))
        f0 = 2 * int(np.random.randint(0, (nfull - nf) // 2 + 1))
        tile = SparseTile.make_random(r, k, f0, nf)
        ntime = tile.nt + tile.t0 + int(tile.tshifts.sum()) + (1 << k) + 8
        ref = SparseTile._dense_dd(tile.unpack(ntime), k)        # (nf/2, 2^(k+1), ntime)
        got = SparseTile.iterate_aligned(tile).unpack(ntime)
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
        lower = SparseTile.make_random(r, k, 2 * f, 1) if mode != 2 else None
        upper = SparseTile.make_random(r, k, 2 * f + 1, 1) if mode != 1 else None
        merged = SparseTile.iterate_singletons(lower, upper)

        need = 1
        if lower is not None:
            need = max(need, lower.nt + lower.t0 + int(lower.tshifts.sum()))
        if upper is not None:
            need = max(need, upper.nt + upper.t0 + int(upper.tshifts.sum()))
        ntime = need + (1 << k) + 8
        row_lo = lower.unpack(ntime)[0] if lower is not None else np.zeros((1 << k, ntime))
        row_up = upper.unpack(ntime)[0] if upper is not None else np.zeros((1 << k, ntime))
        dense_in = np.stack([row_lo, row_up])       # (2, 2^k, ntime)
        ref = SparseTile._dense_dd(dense_in, k)        # (1, 2^(k+1), ntime)
        got = merged.unpack(ntime)
        ksgpu.assert_arrays_equal(ref, got, "ref", "got", ["f", "delay", "time"], epsabs=0.0)

    @staticmethod
    def test_random_specialize_dbits():
        """specialize_dbits(value, nbits, low) vs. brute-force 'fix nbits of the delay to value',
        for both low=True (fix the low nbits bits) and low=False (fix the high nbits bits)."""
        import ksgpu
        r = int(np.random.randint(2, 8))
        k = int(np.random.randint(1, r + 1))                # 1 <= k <= r
        f0 = int(np.random.randint(0, 1 << (r - k)))        # singleton coarse-freq index
        tile = SparseTile.make_random(r, k, f0, 1)
        nbits = int(np.random.randint(0, k + 1))            # 0 <= nbits <= k bits fixed
        rho = k - nbits                                     # number of kept (delay-axis) bits
        ntime = tile.nt + tile.t0 + int(tile.tshifts.sum()) + 8
        full = tile.unpack(ntime)[0]                        # (2^k, ntime)
        for low in (False, True):
            for value in range(1 << nbits):                # the fixed bits are set to 'value'
                got = tile.specialize_dbits(value, nbits, low=low).unpack(ntime)[0]   # (2^rho, ntime)
                tgt = np.zeros((1 << rho, ntime), dtype=np.float64)
                for j in range(1 << rho):                  # j indexes the kept bits
                    # low=True fixes the low nbits bits; low=False fixes the high nbits bits.
                    D = (j << nbits) | value if low else (value << rho) | j
                    tgt[j] = full[D]
                ksgpu.assert_arrays_equal(got, tgt, "got", "tgt", ["kept_d", "time"], epsabs=0.0)


####################################   class SparseTileTriple   ####################################


class SparseTileTriple:
    """
    A tree-dedispersion array of shape (2^(r-k), 2^k, ntime) over a contiguous f-index
    range [f0, f0+nf), represented as a list of SparseTiles. The split lets the first
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
        # Build a canonical SparseTileTriple by splitting a single tile into 1/2/3 sub-tiles.
        bounds = SparseTileTriple._tile_bounds(tile.f0, tile.nf)
        tiles = [tile.slice(c0, c1) for (c0, c1) in bounds]
        return SparseTileTriple(tile.r, tile.k, tile.f0, tile.nf, tiles)

    def get_singleton(self, f, allow_none=False):
        """Return the singleton SparseTile for f-index f. If f is out of [f0, f0+nf):
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
        mostly zeros. This method returns an equivalent SparseTileTriple.
        """
        cm = np.ascontiguousarray(channel_map, dtype=np.float64)
        nchan = len(cm) - 1
        r = integer_log2(nchan)        # channel_map length must be 2^rank + 1
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
        tile = SparseTile(r=r, k=0, f0=f0, nf=f1 - f0, nt=1, dbits=0, data=data,
                          tshifts=np.zeros(0, dtype=np.int64))
        return SparseTileTriple._from_tile(tile)

    def iterate(self):
        """
        One DD(k) step. The first/last output channels (F0, Fmax) are computed with
        iterate_singletons (which absorbs shifts into tshifts to minimize dbits/nt); the
        bulk output channels [F0+1, Fmax) are computed with iterate_aligned on the
        even-aligned input sub-block [2F0+2, 2Fmax) (which lies inside the input middle
        tile). Returns a canonical SparseTileTriple at level k+1.
        """
        assert self.k < self.r, "iterate(): already at k == r"
        f0, nf = self.f0, self.nf
        F0 = f0 // 2
        last = f0 + nf - 1
        Fmax = last // 2
        nf_out = Fmax - F0 + 1

        tiles = [SparseTile.iterate_singletons(
            self.get_singleton(2 * F0, allow_none=True),
            self.get_singleton(2 * F0 + 1, allow_none=True))]
        if nf_out >= 3:
            mid_in = self.tiles[1].slice(2 * F0 + 2, 2 * Fmax)
            tiles.append(SparseTile.iterate_aligned(mid_in))
        if nf_out >= 2:
            tiles.append(SparseTile.iterate_singletons(
                self.get_singleton(2 * Fmax, allow_none=True),
                self.get_singleton(2 * Fmax + 1, allow_none=True)))
        return SparseTileTriple(self.r, self.k + 1, F0, nf_out, tiles)

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
    def test_one_tree_gridding(channel_map, ifreq):
        """Compare make_tree_gridding_output(...).unpack() against ReferenceTreeGriddingKernel."""
        import ksgpu
        cm = np.ascontiguousarray(channel_map, dtype=np.float64)
        ntree = len(cm) - 1
        ntime = 32                                  # gridding kernel needs ntime % (1024/nbits) == 0 (nbits=32)
        ref = SparseTileTriple._reference_gridding(cm, ifreq, ntime)   # (1, ntree, ntime) f32
        sarr = SparseTileTriple.make_tree_gridding_output(cm, ifreq)
        got = sarr.unpack(ntime)                    # (ntree, 1, ntime) f64
        assert sarr.k == 0 and got.shape == (ntree, 1, ntime)
        ksgpu.assert_arrays_equal(ref[0], got[:, 0, :], "ref", "got", ["tree", "time"], epsabs=0.0)

    @staticmethod
    def test_random_tree_gridding():
        cm, ifreq = SparseTileTriple.random_channel_map()
        SparseTileTriple.test_one_tree_gridding(cm, ifreq)


#####################################   class SparseTilePerM   #####################################


class SparseTilePerM:
    """
    Sparse representation of a SUBBANDED tree-dedisperser's output for a one-hot input.
    The dense output has shape (2^(r-R), M, ntime) (notes Section "Subbanded
    dedispersion"), held as a length-M list 'per_m'. Each entry is either a rank-(r-R),
    fully-iterated (k == r-R), nf==1 SparseTile carrying that multiplet's 2^(r-R) coarse
    delays, or None for a multiplet whose subband does not overlap the input frequency channel.

    Members
    -------
      r, R:            tree rank and pf_rank (per_m tiles will have rank rho = r-R)
      subband_counts:  the length-(R+1) C_l array.
      per_m:           length-M list whose entries are a SparseTile (with r == k == rho)
                       or None (multiplet outside the gridding footprint).
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
            if t is None:
                continue
            assert isinstance(t, SparseTile)
            assert (t.r, t.k, t.nf) == (rho, rho, 1), (t.r, t.k, t.nf, rho)

    def unpack(self, ntime):
        """Returns the dense (2^(r-R), M, ntime) output array.
        Multiplets with no overlap (per_m[m] is None) are left as zeros."""
        rho = self.r - self.R
        out = np.zeros((1 << rho, len(self.per_m), ntime), dtype=np.float64)
        for m, tile in enumerate(self.per_m):
            if tile is not None:
                out[:, m, :] = tile.unpack(ntime)[0]
        return out

    @staticmethod
    def _emit(per_m, mbase, l, tile, C, upper_half_only=False):
        # Fill the 2^l multiplets of one subband into per_m. A None tile (subband outside
        # the gridding footprint) leaves those multiplets as None. The extraction lag C*d_hi
        # (the subband's per-coarse-delay time shift) is folded into each multiplet's tshifts
        # here: adding C*2^j to coarse bit j gives a total forward shift of C*d_hi.
        # If upper_half_only, also drop the top coarse-DM bit (keep the upper DM-half), AFTER the
        # lag so that the dropped bit's lag+DD shift fold into t0.
        if tile is None:
            for e in range(1 << l):
                per_m[mbase + e] = None
        else:
            lag = C * (1 << np.arange(tile.k - l, dtype=np.int64))   # coarse bit j -> C*2^j
            for e in range(1 << l):
                t = tile.specialize_dbits(e, l, low=True)
                t = SparseTile(t.r, t.k, t.f0, t.nf, t.nt, t.dbits, t.data, t.tshifts + lag, t0=t.t0)
                if upper_half_only:
                    t = t.specialize_dbits(1, 1, low=False)   # keep upper DM-half (top bit == 1)
                per_m[mbase + e] = t

    @staticmethod
    def make_dedispersion_output(channel_map, ifreq, fs, upper_half_only=False):
        """
        Suppose TreeGriddingKernel -> (subbanded tree dedispersion) is applied to a one-hot
        shape (nfreq, ntime) input whose (ifreq, 0) entry is 1. The output is a mostly-zero
        shape (2^(r-R), M, ntime) array; this returns an equivalent SparseTilePerM.
        We assume non-bit-reversed coarse-delay (d_hi) and multiplet (m) indices.

        'fs' is a FrequencySubbands object defining the subband scheme.

        If upper_half_only, only the upper half of the coarse-DM range is kept (the top d_hi bit
        is fixed to 1 and dropped), so the result has rank (r-R-1) instead of (r-R). This is what
        a downsampled tree (ds_level > 0) keeps.

        Implementation (notes Section "Subbanded dedispersion"): iterate a single
        under-the-hood SparseTileTriple (the full-band gridding footprint) and extract per-
        multiplet outputs "on the fly". Case 1 (aligned, l=0 or even s) reads the
        level-(r-R+l) singleton f directly; Case 2 (half-aligned, l>0 odd s) merges the
        level-(r-R+l-1) pair (2f+1, 2f+2) via iterate_singletons(require_aligned=False).
        Each extraction is then split into its 2^l multiplets (specialize_dbits, low=True).
        """
        sc = [int(c) for c in fs.subband_counts]
        sarr = SparseTileTriple.make_tree_gridding_output(channel_map, ifreq)
        r = sarr.r
        R = fs.pf_rank
        assert 0 <= R <= r, (R, r)
        per_m = [None] * fs.M

        # Per subband: (l, f_blk, case1, mbase), derived from its coarse f-range [flo, fhi).
        subs = []
        for n in range(fs.N):
            flo, fhi = int(fs.n_to_flo[n]), int(fs.n_to_fhi[n])
            l = integer_log2(fhi - flo)                    # band width is 2^l
            f_blk = flo >> l                               # coarse block index
            case1 = (flo & ((1 << l) - 1)) == 0            # aligned (always true for l==0)
            subs.append((l, f_blk, case1, int(fs.n_to_mbase[n])))

        for k in range(0, r + 1):
            for (l, f_blk, case1, mbase) in subs:
                if case1 and (r - R + l) == k:                  # Case 1: read level k
                    tile = sarr.get_singleton(f_blk, allow_none=True)
                    C = (1 << l) * ((1 << (R - l)) - 1 - f_blk)
                    SparseTilePerM._emit(per_m, mbase, l, tile, C, upper_half_only)
                elif (not case1) and (r - R + l - 1) == k:      # Case 2: read level r-R+l-1
                    lo = sarr.get_singleton(2 * f_blk + 1, allow_none=True)
                    up = sarr.get_singleton(2 * f_blk + 2, allow_none=True)
                    C = (1 << (l - 1)) * ((1 << (R - l + 1)) - 2 * f_blk - 3)
                    if lo is None and up is None:
                        tile = None
                    else:
                        tile = SparseTile.iterate_singletons(lo, up, require_aligned=False)
                    SparseTilePerM._emit(per_m, mbase, l, tile, C, upper_half_only)
            if k < r:
                sarr = sarr.iterate()

        return SparseTilePerM(r - (1 if upper_half_only else 0), R, sc, per_m)

    # ------------------------------- test utilities -------------------------------

    @staticmethod
    def test_one_subbanded_dedispersion(channel_map, ifreq, subband_counts, upper_half_only=False):
        """
        Compare SparseTilePerM.make_dedispersion_output(...).unpack() against
        ReferenceTreeGriddingKernel -> ReferenceTree(subband_counts) on a one-hot input.
        With upper_half_only, compares against the upper coarse-DM half of the reference.
        """
        import ksgpu
        from ..kernels import ReferenceTree
        from ..pirate_pybind11 import FrequencySubbands

        cm = np.ascontiguousarray(channel_map, dtype=np.float64)
        ntree = len(cm) - 1
        r = integer_log2(ntree)
        sc = [int(c) for c in subband_counts]
        R = len(sc) - 1
        assert R <= r
        rho = r - R
        upper_half_only = upper_half_only and (rho >= 1)   # need a coarse-DM bit to drop
        fs = FrequencySubbands(sc)
        M = fs.M
        # ntime: multiple of 32 (gridding) and comfortably > 2^(r+1) so the largest
        # (delay + lag) fits one non-incremental ReferenceTree chunk with no wraparound.
        ntime = (((3 << r) + 128) // 32 + 1) * 32

        grid = SparseTileTriple._reference_gridding(cm, ifreq, ntime)    # (1, ntree, ntime) f32
        buf = np.ascontiguousarray(grid.reshape(1, 1, ntree, ntime))   # (1,1,ntree,ntime) f32
        out_rt = np.zeros((1, 1 << rho, M, ntime), dtype=np.float32)
        ReferenceTree(num_beams=1, amb_rank=0, dd_rank=r, ntime=ntime,
                      nspec=1, subband_counts=sc).dedisperse(buf, out_rt)  # natural (d_hi, m)

        ssa = SparseTilePerM.make_dedispersion_output(cm, ifreq, fs, upper_half_only=upper_half_only)
        got = ssa.unpack(ntime)                                         # (2^rho_out, M, ntime) f64
        rho_out = rho - (1 if upper_half_only else 0)
        ref = out_rt[0][1 << rho_out:] if upper_half_only else out_rt[0]   # upper coarse-DM half
        assert got.shape == (1 << rho_out, M, ntime)
        ksgpu.assert_arrays_equal(ref, got, "reftree", "got", ["d_hi", "m", "time"], epsabs=0.0)

        # "Headline compactness" structural check: when ifreq's gridding footprint is a
        # single tree channel, iterating to k==r keeps the tile maximally compact -- nf==1
        # and dbits==0 at every level. (This single-channel case occurs often under
        # random_channel_map; its dedispersion correctness is already covered above, since
        # a single-channel footprint grids to weight 1.0.)
        s = SparseTileTriple.make_tree_gridding_output(cm, ifreq)
        if s.nf == 1:
            while s.k < s.r:
                s = s.iterate()
                assert s.nf == 1, (s.k, s.nf)
                for t in s.tiles:
                    assert t.dbits == 0, (s.k, t.dbits)

    @staticmethod
    def test_random_subbanded_dedispersion():
        from ..pirate_pybind11 import FrequencySubbands
        cm, ifreq = SparseTileTriple.random_channel_map()
        r = integer_log2(len(cm) - 1)
        R = int(np.random.randint(0, min(r, 4) + 1))    # pf_rank <= min(r, 4)
        sc = [int(c) for c in FrequencySubbands.make_random_subband_counts(R)]
        upper = bool(np.random.randint(0, 2))
        SparseTilePerM.test_one_subbanded_dedispersion(cm, ifreq, sc, upper_half_only=upper)
