import time
import numpy as np
import scipy.linalg


def is_integer(n):
    return isinstance(n,int) or isinstance(n,np.int64) or isinstance(n,np.int32)


def is_power_of_two(n):
    return is_integer(n) and (n > 0) and (n == (1 << int(np.log2(n)+0.5)))


def integer_log2(n):
    k = int(np.log2(n+0.5))
    assert n == 2**k
    return k


def postpad_array(arr, new_nt):
    """Pads last index of array (nt -> new_nt), and returns a new array."""
    
    assert arr.ndim >= 1
    assert new_nt >= arr.shape[-1]
    
    new_shape = arr.shape[:-1] + (new_nt,)
    new_arr = np.zeros(new_shape, dtype=arr.dtype)
    new_arr[..., :arr.shape[-1]] = arr[:]
    return new_arr


def lag_array(arr, axis, ix, lag):
    """Lags an array 'slice' in place. The last axis is time."""

    assert 0 <= axis < (arr.ndim-1)
    assert 0 <= ix < arr.shape[axis]
    assert lag >= 0
    
    if lag > 0:
        s = (slice(None),)*axis + (ix,)
        t = arr[s]      # reference
        u = np.copy(t)  # copy
        t[..., :lag] = 0
        t[..., lag:] = u[..., :(-lag)]


def check_bits(bits):
    """Checks that 'bits' is a sorted list of powers of two."""
    
    assert all(is_power_of_two(x) for x in bits)

    if len(bits) >= 2:
        assert all(bits[i] < bits[i+1] for i in range(len(bits)-1))
        
        
def expand_bit_array(arr, old_bits, new_bits, has_spectator_axis):
    """Returns a new array. The spectator axis (if present) must be last."""

    check_bits(old_bits)
    check_bits(new_bits)

    nold = len(old_bits)
    nnew = len(new_bits)
    sflag = 1 if has_spectator_axis else 0
        
    assert arr.ndim == (nold + sflag)
    assert all(i==2 for i in arr.shape[:nold])

    if nnew == 0:
        assert nold == 0
        return np.copy(arr)
        
    shape1 = np.ones(nnew + sflag, dtype=int)
    shape2 = np.full(nnew + sflag, 2, dtype=int)
    
    if has_spectator_axis:
        shape1[-1] = shape2[-1] = arr.shape[-1]   # spectator axis

    for b in old_bits:
        shape1[new_bits.index(b)] = 2

    arr = np.reshape(arr, shape1)
    ret = np.empty(shape2, dtype=arr.dtype)
    ret[:] = arr[:]
    return ret


def compute_cf(arr):
    """Last axis of 'arr' is time. Returns an array with the same shape as 'arr'."""

    nt = arr.shape[-1]
    ret = np.zeros(arr.shape)

    # FIXME brute-force algorithm, could be optimized.
    for i in range(nt):
        for j in range(i+1):
            ret[..., i-j] += arr[..., j] * arr[..., i]

    return ret


def truncated_svd(m, eps=1.0e-12, return_nsvd=True):
    """Returns (u, s, v, nsvd)."""

    u, s, v = scipy.linalg.svd(m, full_matrices=False)
    
    t = eps * s[0]
    n = np.sum(s >= t)
    assert np.all(s[:n] >= t)
    assert np.all(s[n:] <= t)

    u = np.copy(u[:,:n])
    s = np.copy(s[:n])
    v = np.copy(v[:n,:])
    
    return (u,s,v,n) if return_nsvd else (u,s,v)


def bit_reverse(i, nbits):
    assert 0 <= i < 2**nbits
    
    ret = 0
    for b in range(nbits):
        if (i & (1 << b)):
            ret = ret | (1 << (nbits-1-b))
            
    return ret


def tree_delay(tfreq, dm_brev, rank):
    """Vectorized."""

    tfreq = np.asarray(tfreq)
    dm_brev = np.asarray(dm_brev)
    ff = (~tfreq) & (2**rank-1)   # flip bits
    
    shape = np.broadcast_shapes(tfreq.shape, dm_brev.shape)
    ret = np.zeros(shape, dtype=int)

    for b in range(rank):
        ff2 = ff >> 1
        ret += (dm_brev & 1) * ((ff & 1) + ff2)
        dm_brev >>= 1
        ff = ff2

    return ret


####################################################################################################


class ReferenceDedisperser:
    def __init__(self, rank, nfreq, fmin, fmax):
        # FIXME more error-checking
        assert 1 <= rank <= 16
        
        self.rank = rank
        self.nfreq = nfreq
        self.fmin = fmin
        self.fmax = fmax
        
        freq_edges = np.linspace(fmin, fmax, nfreq+1)
        t_edges = freq_edges**(-2)
        t_edges -= t_edges[-1]
        t_edges *= (2**rank / t_edges[0])
        t_edges[0] = 2**rank - 1.0e-10
        t_edges[-1] = 1.0e-10

        self.flo = freq_edges[:-1]   # increasing
        self.fhi = freq_edges[1:]    # increasing
        self.tlo = t_edges[1:]       # decreasing
        self.thi = t_edges[:-1]      # decreasing
        

    def dedisperse(self, arr):
        """Shape (nfreq, nt) -> (2**rank, nt+2**rank-1). The output DM axis is bit-reversed."""

        assert arr.ndim == 2
        assert arr.shape[0] == self.nfreq

        rank = self.rank
        nt_in = arr.shape[1]
        nt_out = nt_in + 2**rank - 1

        ret = np.zeros((2**rank, nt_out))
        brev = np.array([0], dtype=int)
        
        for f in range(self.nfreq):
            tlo, thi = self.tlo[f], self.thi[f]
            it0 = max(int(tlo), 0)
            it1 = min(int(thi)+1, 2**rank)

            for it in range(it0,it1):
                t0 = max(tlo, it)
                t1 = min(thi, it+1)
                w = (t1-t0) / (thi-tlo)
                ret[it,:nt_in] += w * arr[f,:]

        for r in range(rank):
            ret = np.reshape(ret, (2**(rank-r-1), 2, 2**r, nt_out))
            assert brev.shape == (2**r,)
            
            x = np.copy(ret[:,0,:,:])
            ret[:,0,:,:] = ret[:,1,:,:]

            for d in range(2**r):
                l = brev[d]  # lag
                ll = (-l) if (l > 0) else nt_out
                ret[:,0,d,l:] += x[:,d,:ll]
                ret[:,1,d,(l+1):] += x[:,d,:(-l-1)]
            
            new_brev = np.zeros(2**(r+1), dtype=int)
            new_brev[:(2**r)] = 2*brev
            new_brev[(2**r):] = 2*brev+1
            brev = new_brev

        return np.reshape(ret, (2**rank, nt_out))            
        

####################################################################################################


class FreqMatrix:
    def __init__(self, tmin, tmax, itmin=None, itmax=None, wt=1.0):
        """
        The FreqMatrix encodes how one frequency channel (corresponding to
        input tree index range [tmin,tmax]) gets "smeared" across multiple time
        samples in the dedispersion output, as a function of trial DM index.
        
        Members:

           - self.{t,it}{min,max}: same meaning as constructor args
           - self.bits: sorted list (contains powers of 2)
           - self.lags: integer-valued array of length len(bits)
           - self.data: shape (2, ..., 2, nt), where nt = (itmax - itmin + 1).

        Each bit can be interpeted as either an "input tree index bit", or a 
        "bit-reversed output DM index bit".
        """
        
        if itmin is None:
            itmin = int(tmin)
        if itmax is None:
            itmax = int(tmax)

        assert tmin < tmax
        assert itmin <= itmax

        self.tmin = tmin
        self.tmax = tmax
        self.itmin = itmin
        self.itmax = itmax
        self.nt = itmax - itmin + 1

        if itmin == itmax:
            self.bits = [ ]
            self.lags = np.zeros((0,), dtype=int)  # 1-d array of length 0
            self.data = np.array([wt])
            return
        
        # new_bit = highest bit that gets flipped, over range [itmin:itmax+1].
        bit = 1 << int(np.log2((itmin ^ itmax) + 0.5))
        assert (itmax & bit) and not (itmin & bit)

        itmid = itmax & ~(bit-1)
        assert itmin < itmid <= itmax

        m1 = FreqMatrix(tmin, itmid, itmin, itmid-1, wt * (itmid-tmin) / (tmax-tmin))
        m2 = FreqMatrix(itmid, tmax, itmid, itmax, wt * (tmax-itmid) / (tmax-tmin))
        assert all((b < bit) for b in (m1.bits + m2.bits))
        
        bits = sorted(set(m1.bits + m2.bits + [bit]))        
        lags1 = np.zeros(len(bits), dtype=int)
        lags2 = np.zeros(len(bits), dtype=int)
        lags2[-1] = 1

        for b,l in zip(m1.bits, m1.lags):
            lags1[bits.index(b)] = l
        for b,l in zip(m2.bits, m2.lags):
            lags2[bits.index(b)] = l

        self.bits = bits
        self.lags = lags1 + lags2
        self.data = None

        data1 = postpad_array(m1.data, self.nt)
        data2 = postpad_array(m2.data, self.nt)
        
        data1 = expand_bit_array(data1, m1.bits, bits, has_spectator_axis=True)
        data2 = expand_bit_array(data2, m2.bits, bits, has_spectator_axis=True)

        # Apply lags2 to data1
        for i,l in enumerate(lags2):
            lag_array(data1, i, 1, l)

        self.data = data1 + data2


    def expand_lags(self, rank, bit_reverse_dm):
        """
        Returns a length (2**rank) array.
        Note: this function is no longer used, and could be removed.
        """

        # Slow implementation!
        all_bits = [ 2**r for r in range(rank) ]
        ret = np.zeros((2,)*rank, dtype=int)
        
        for b,l in zip(self.bits, self.lags):
            a = np.array([0,l])
            ret += expand_bit_array(a, [b], all_bits, has_spectator_axis=False)

        if bit_reverse_dm:
            ret = np.transpose(ret, range(rank-1,-1,-1))

        return np.reshape(ret, (2**rank,))


    def expand_matrix(self, rank, bit_reverse_dm):
        """Convenient for unit testing. Returns a (2**rank, nt) array."""

        all_bits = [ 2**r for r in range(rank) ]
        ret = expand_bit_array(self.data, self.bits, all_bits, has_spectator_axis=True)

        if bit_reverse_dm:
            ret = np.transpose(ret, list(range(rank-1,-1,-1)) + [ rank ])

        return np.reshape(ret, (2**rank, self.nt))


####################################################################################################
        
    
def test_one_freq_matrix(rank, nfreq, ifreq):
    src = np.zeros((nfreq,1))
    src[ifreq,0] = 1.0

    d = ReferenceDedisperser(rank, nfreq, fmin=400, fmax=800)    
    dst1 = d.dedisperse(src)

    fm = FreqMatrix(d.tlo[ifreq], d.thi[ifreq])
    dst2 = fm.expand_matrix(rank, bit_reverse_dm=True)
    dst2 = postpad_array(dst2, dst1.shape[1])
    
    lags = tree_delay(fm.itmax, np.arange(2**rank), rank)

    for d in range(2**rank):
        lag_array(dst2, 0, d, lags[d])

    eps = float(np.max(np.abs(dst1-dst2)))  # np.float -> float
    print(f'test_freq_matrix({rank=}, {nfreq=}, {ifreq=}): {eps=}')
    assert eps < 1.0e-12

    
def test_freq_matrix():
    for _ in range(100):
        rank = np.random.randint(1, 10)
        nfreq = np.random.randint(1, 2**rank)
        ifreq = np.random.randint(nfreq)
        test_one_freq_matrix(rank, nfreq, ifreq)
        

####################################################################################################


class TreeMatrix:
    def __init__(self, rank, nfreq, fmin, fmax, E, svd1=True, svd2=True):
        """
        For each "group" 0 <= g < (rank+1), we have the following:
        
          - nt: integer
          - bits: sorted list of bits (each bit is a power of two)
          - freqs: 1-d integer-valued array containing frequency indices
          - vmat: array of shape (2, ..., 2, nprof, nfreq) with one "2" for each bit
          - svd1_u: array of shape (2**nbits * nprof, nsvd1)
          - svd1_s: array of shape (nsvd1,)
          - svd1_v: array of shape (nsvd1, nfreq)
          - nsvd1: integer
        """

        self.rank = rank
        self.nfreq = nfreq
        self.fmin = fmin
        self.fmax = fmax
        self.E = E
        
        self.has_svd1 = svd1
        self.has_svd2 = svd2
        self.ref_dd = ReferenceDedisperser(rank, nfreq, fmin, fmax)
        self.freqs = [ [] for _ in range(rank+1) ]
        self.bits = [ set() for _ in range(rank+1) ]
        self.vmat = [ None for _ in range(rank+1) ]
        self.nt = np.zeros(rank+1, dtype=int)
        
        # shape (P,T)
        prof_cf = self.get_profiles(E, normalize=True)
        prof_cf = compute_cf(prof_cf)
        prof_cf[:,1:] *= 2.0
        
        self.P = prof_cf.shape[0]
        nt_p = prof_cf.shape[1]

        # FIXME convenient but uses more memory than necessary.
        fms = [ FreqMatrix(self.ref_dd.tlo[ifreq], self.ref_dd.thi[ifreq]) for ifreq in range(nfreq) ]

        # Initialize freqs, bits, nt
        for f,fm in enumerate(fms):
            g = (integer_log2(fm.bits[-1]) + 1) if fm.bits else 0
            self.freqs[g].append(f)
            self.bits[g] = set.union(self.bits[g], fm.bits)
            self.nt[g] = max(self.nt[g], fm.nt)

        # Initialize vmat
        for g in range(rank+1):
            self.freqs[g] = np.array(self.freqs[g])   # list of integers -> array
            self.bits[g] = sorted(self.bits[g])       # set -> sorted list
            
            F, B, nt = len(self.freqs[g]), len(self.bits[g]), self.nt[g]
            ashape = (2,)*B + (F,nt)
            a = np.zeros(ashape)
            
            for i,f in enumerate(self.freqs[g]):
                a[...,i,:fms[f].nt] = expand_bit_array(fms[f].data, fms[f].bits, self.bits[g], has_spectator_axis=True)

            a = compute_cf(a)   # shape (2, ..., 2, F, nt)

            T = min(nt, nt_p)
            a = np.dot(a[...,:T], np.transpose(prof_cf[:,:T]))   # shape (2, ..., 2, F, nprof)

            perm = tuple(range(B)) + (B+1, B)
            self.vmat[g] = np.copy(np.transpose(a, perm))    # shape (2, ..., 2, nprof, F)

        if not svd1:
            return

        self.nsvd1 = np.zeros(rank+1, dtype=int)
        self.svd1_u = [ None for _ in range(rank+1) ]
        self.svd1_s = [ None for _ in range(rank+1) ]
        self.svd1_v = [ None for _ in range(rank+1) ]

        for g in range(rank+1):
            if len(self.freqs[g]) > 0:
                B, F, P = len(self.bits[g]), len(self.freqs[g]), self.P
                m = np.reshape(self.vmat[g], (2**B * P, F))
                self.svd1_u[g], self.svd1_s[g], self.svd1_v[g], self.nsvd1[g] = truncated_svd(m)

        if not svd2:
            return

        itmp = 0
        ntmp = np.sum(self.nsvd1)
        utmp = np.zeros((2**rank * P, ntmp))
        vtmp = np.zeros((ntmp, nfreq))
        all_bits = [ 2**r for r in range(rank) ]
        
        # Initialize utmp, vtmp (awkward)
        for g in range(rank+1):
            if self.nsvd1[g] > 0:
                B, N, P = len(self.bits[g]), self.nsvd1[g], self.P
                a = self.svd1_u[g] * np.reshape(self.svd1_s[g], (1,N))
                a = np.reshape(a, (2**B, P, N))
                a = np.reshape(a, (2,)*B + (P*N,))
                a = expand_bit_array(a, self.bits[g], all_bits, has_spectator_axis=True)
                a = np.reshape(a, (2**rank, P, N))
                a = np.reshape(a, (2**rank * P, N))
                utmp[:, itmp:(itmp+N)] = a
                vtmp[itmp:(itmp+N), self.freqs[g]] = self.svd1_v[g]
                itmp += N

        # At this point in the code, the variance matrix is V = (utmp * vtmp)
        
        utmp, stmp2, vtmp2 = scipy.linalg.svd(utmp, full_matrices=False)
        mtmp = np.reshape(stmp2,(ntmp,1)) * vtmp2

        # At this point in the code, variance matrix is (utmp * mtmp * vtmp)
        
        utmp2, stmp2, vtmp2, nsvd2 = truncated_svd(mtmp)

        # At this point in the code, variance matrix is (utmp * utmp2 * diag(stmp2) * vtmp2 * vtmp)
        
        self.svd2_u = np.dot(utmp, utmp2)
        self.svd2_v = np.dot(vtmp2, vtmp)
        self.svd2_s = stmp2
        self.nsvd2 = nsvd2
        

    def show(self):
        print(f'TreeMatrix(rank={self.rank}, nfreq={self.nfreq}, fmin={self.fmin}, fmax={self.fmax}, E={self.E}, P={self.P})')
        
        for g in range(self.rank+1):
            s = f', nsvd1={self.nsvd1[g]}' if self.has_svd1 else ''
            print(f'    Group {g}: bits={self.bits[g]}, nfreqs={len(self.freqs[g])}, nt={self.nt[g]}{s}')

        if self.has_svd1:
            print(f'    Sum(nsvd1) = {np.sum(self.nsvd1)}')

        if self.has_svd2:
            print(f'    nsvd2 = {self.nsvd2}')
    
    
    def make_huge_vmat(self):
        """Returns an array of shape (2^rank * P, nfreq)."""
        
        rank, nfreq, P = self.rank, self.nfreq, self.P
        all_bits = [ 2**r for r in range(rank) ]
        ret = np.zeros((2**rank * P, nfreq))
        
        for g in range(rank+1):
            if len(self.freqs[g]) > 0:
                B, F = len(self.bits[g]), len(self.freqs[g])
                s = (2,)*B + (P*F,)
                a = np.reshape(self.vmat[g], s)
                a = expand_bit_array(a, self.bits[g], all_bits, has_spectator_axis=True)
                a = np.reshape(a, (2**rank, P*F))
                a = np.reshape(a, (2**rank * P, F))
                ret[:,self.freqs[g]] += a[:,:]

        return ret


    @classmethod
    def get_profiles(cls, E, normalize):
        """
        Returns shape (P,T) array, where:
           P = number of profiles = 3*log2(E) + 1
           T = number of time samples = 2*E
        """

        assert is_power_of_two(E)
        
        P = 3*integer_log2(E) + 1
        T = 2*E
        
        if normalize:
            ret = cls.get_profiles(E, normalize=False)
            t = np.sum(ret**2, axis=1)
            ret *= np.reshape(t**(-0.5), (P,1))
            return ret

        ret = np.zeros((P,T))
        ret[0,0] = 1.0

        if E > 1:
            ret[1,:2] = [1.0, 1.0]
            ret[2,:3] = [0.5, 1.0, 0.5]
            ret[3,:4] = [0.5, 1.0, 1.0, 0.5]

        if E > 2:
            ret2 = cls.get_profiles(E//2, normalize=False)
            ret[4:,0::2] = ret2[1:,:]
            ret[4:,1::2] = ret2[1:,:]

        return ret
    

####################################################################################################
    

if True:
    test_freq_matrix()
    
    t0 = time.time()
    t = TreeMatrix(16, 2**15, 400., 800., E=16)
    # t = TreeMatrix(11, 2**10, 400., 800., E=4)
    print(f'TreeMatrix constructor: {time.time()-t0} sec')

    t.show()
    fdfasd
    
    # End-to-end test of svd2 (only for small instances)
    m = t.make_huge_vmat()
    m2 = np.reshape(t.svd2_s, (t.nsvd2,1)) * t.svd2_v
    print(f'XXX0 {t.svd2_v.shape=}')
    print(f'XXX1 {m2.shape=}')
    m2 = np.dot(t.svd2_u, m2)
    print(f'XXX2 {m2.shape=}')
    eps = np.max(np.abs(m-m2))
    print(f'{eps=}')
