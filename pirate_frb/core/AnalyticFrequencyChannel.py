import numpy as np


def is_integer(n):
    return isinstance(n,int) or isinstance(n,np.int64) or isinstance(n,np.int32)


def is_power_of_two(n):
    return is_integer(n) and (n > 0) and (n == (1 << int(np.log2(n)+0.5)))


def integer_log2(n):
    k = int(np.log2(n+0.5))
    assert n == 2**k
    return k


def postpad_array(arr, new_nt):
    """Pads last index of array (nt -> new_nt), and returns a new array. The last axis is time."""
    
    assert arr.ndim >= 1
    assert new_nt >= arr.shape[-1]
    
    new_shape = arr.shape[:-1] + (new_nt,)
    new_arr = np.zeros(new_shape, dtype=arr.dtype)
    new_arr[..., :arr.shape[-1]] = arr[:]
    return new_arr


def lag_array(arr, axis, ix, lag):
    """
    Lags an array 'slice' in place. The last axis is time.
    Does not keep persistent state -- zeros/discards samples instead.
    """

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
    """
    Checks that 'bits' is a sorted list of powers of two.
    Used in a context where there is an array with one length-2 axis per bit.
    """
    
    assert all(is_power_of_two(x) for x in bits)

    if len(bits) >= 2:
        assert all(bits[i] < bits[i+1] for i in range(len(bits)-1))
        
        
def expand_bit_array(arr, old_bits, new_bits, has_spectator_axis):
    """
    Returns a new array. The spectator axis (if present) must be last.
    The old_bits must be a subset of the new_bits.
    """

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


####################################################################################################


class AnalyticFrequencyChannel:
    def __init__(self, tmin, tmax, itmin=None, itmax=None, wt=1.0):
        """
        The FreqMatrix encodes how one frequency channel (corresponding to
        input tree index range [tmin,tmax]) gets "smeared" across multiple time
        samples in the dedispersion output, as a function of trial DM index.
        Note that the TreeMatrix below will combine many FreqMatrices.
        
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
