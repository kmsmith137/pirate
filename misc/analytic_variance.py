import time
import numpy as np


def is_integer(n):
    return isinstance(n,int) or isinstance(n,np.int64) or isinstance(n,np.int32)

def is_power_of_two(n):
    return is_integer(n) and (n > 0) and (n == (1 << int(np.log2(n)+0.5)))


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


####################################################################################################


class ReferenceDedisperser:
    def __init__(self, rank, nfreq, fmin, fmax):
        # FIXME error-check arguments
        
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


class TreeMatrix:
    def __init__(self, tmin, tmax, itmin=None, itmax=None, wt=1.0, lags_only=False):
        """
        The TreeMatrix encodes how one frequency channel (corresponding to
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
            self.data = np.array([wt]) if (not lags_only) else None
            return
        
        # new_bit = highest bit that gets flipped, over range [itmin:itmax+1].
        bit = 1 << int(np.log2((itmin ^ itmax) + 0.5))
        assert (itmax & bit) and not (itmin & bit)

        itmid = itmax & ~(bit-1)
        assert itmin < itmid <= itmax

        m1 = TreeMatrix(tmin, itmid, itmin, itmid-1, wt * (itmid-tmin) / (tmax-tmin), lags_only)
        m2 = TreeMatrix(itmid, tmax, itmid, itmax, wt * (tmax-itmid) / (tmax-tmin), lags_only)
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
        
        if lags_only:
            return

        data1 = postpad_array(m1.data, self.nt)
        data2 = postpad_array(m2.data, self.nt)
        
        data1 = self._expand_helper(data1, m1.bits, bits, has_time_axis=True)
        data2 = self._expand_helper(data2, m2.bits, bits, has_time_axis=True)

        # Apply lags2 to data1
        for i,l in enumerate(lags2):
            lag_array(data1, i, 1, l)

        self.data = data1 + data2


    def expand_lags(self, rank, bit_reverse_dm):
        """Convenient for unit testing. Returns a length (2**rank) array. The DM is not bit-reversed."""

        # Slow implementation!
        all_bits = [ 2**r for r in range(rank) ]
        ret = np.zeros((2,)*rank, dtype=int)
        
        for b,l in zip(self.bits, self.lags):
            a = np.array([0,l])
            ret += self._expand_helper(a, [b], all_bits, has_time_axis=False)

        if bit_reverse_dm:
            ret = np.transpose(ret, range(rank-1,-1,-1))

        return np.reshape(ret, (2**rank,))


    def expand_matrix(self, rank, bit_reverse_dm):
        """Convenient for unit testing. Returns a (2**rank, nt) array."""

        all_bits = [ 2**r for r in range(rank) ]
        ret = self._expand_helper(self.data, self.bits, all_bits, has_time_axis=True)

        if bit_reverse_dm:
            ret = np.transpose(ret, list(range(rank-1,-1,-1)) + [ rank ])

        return np.reshape(ret, (2**rank, self.nt))
    

    def _check_bits(self, bits):
        assert all(is_power_of_two(x) for x in bits)

        if len(bits) >= 2:
            assert all(bits[i] < bits[i+1] for i in range(len(bits)-1))
        
        
    def _expand_helper(self, arr, old_bits, new_bits, has_time_axis):
        """Returns a new array."""

        self._check_bits(old_bits)
        self._check_bits(new_bits)

        nold = len(old_bits)
        nnew = len(new_bits)
        tflag = 1 if has_time_axis else 0
        
        assert arr.ndim == (nold + tflag)
        assert all(i==2 for i in arr.shape[:nold])

        shape1 = np.ones(nnew + tflag, dtype=int)
        shape2 = np.full(nnew + tflag, 2, dtype=int)

        if has_time_axis:
            shape1[-1] = shape2[-1] = arr.shape[-1]   # ntime

        for b in old_bits:
            shape1[new_bits.index(b)] = 2

        arr = np.reshape(arr, shape1)
        ret = np.empty(shape2, dtype=arr.dtype)
        ret[:] = arr[:]
        return ret


####################################################################################################


def test_one_tree_matrix(rank, nfreq, ifreq):
    src = np.zeros((nfreq,1))
    src[ifreq,0] = 1.0

    d = ReferenceDedisperser(rank, nfreq, fmin=400, fmax=800)    
    dst1 = d.dedisperse(src)

    tm = TreeMatrix(d.tlo[ifreq], d.thi[ifreq])
    dst2 = tm.expand_matrix(rank, bit_reverse_dm=True)
    dst2 = postpad_array(dst2, dst1.shape[1])
    
    tm2 = TreeMatrix(d.thi[ifreq], 2**rank, itmin=tm.itmax, itmax=2**rank-1, lags_only=True)    
    lags = tm2.expand_lags(rank, bit_reverse_dm=True)

    for d in range(2**rank):
        lag_array(dst2, 0, d, lags[d])

    eps = np.max(np.abs(dst1-dst2))
    print(f'test_tree_matrix({rank=}, {nfreq=}, {ifreq=}): {eps=}')
    assert eps < 1.0e-12

    
def test_tree_matrix():
    for _ in range(100):
        rank = np.random.randint(1, 10)
        nfreq = np.random.randint(1, 2**rank)
        ifreq = np.random.randint(nfreq)
        test_one_tree_matrix(rank, nfreq, ifreq)
        

####################################################################################################
    

if True:
    test_tree_matrix()
