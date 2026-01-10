import functools
import numpy as np

from ..pirate_pybind11 import (
    DedispersionPlan,
    DedispersionConfig,
    ReferenceTreeGriddingKernel
)


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
    def __init__(self, weights, it0):
        """
        The AnalyticFrequencyChannel represents the dedispersion transform restricted
        to a single input frequency channel. The frequency channel spans tree index 
        range [it0:it1], with specified weights (an array of length it1-it0). Usually,
        the sum of the weights will be 1.
        
        Members:

           - self.weights: see above
           - self.{it0,it1}: see above
           - self.nt: integer, equal to it1 - it0
           - self.bits: sorted list (contains powers of 2)
           - self.lags: integer-valued array of length len(bits)
           - self.data: shape (2, ..., 2, it1-it0)

        Each bit can be interpeted as either an "input tree index bit", or a 
        "bit-reversed output DM index bit".
        """

        assert weights.ndim == 1
        assert len(weights) >= 1
        it1 = it0 + len(weights)

        self.weights = weights
        self.it0 = it0
        self.it1 = it1
        self.nt = it1 - it0
        
        if it1 == it0 + 1:
            self.bits = [ ]
            self.lags = np.zeros((0,), dtype=int)  # 1-d array of length 0
            self.data = weights
            return

        # bit = highest bit that gets flipped, over range [it0:it1].
        itmax = it1-1
        bit = 1 << int(np.log2((it0 ^ itmax) + 0.5))
        assert (itmax & bit) and not (it0 & bit)

        itmid = itmax & ~(bit-1)
        assert it0 < itmid < it1

        m1 = AnalyticFrequencyChannel(weights[:(itmid-it0)], it0)
        m2 = AnalyticFrequencyChannel(weights[(itmid-it0):], itmid)
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


    def __repr__(self):
        return f'AnalyticFrequencyChannel({self.weights}, it0={self.it0})'


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


class AnalyticDedisperser:
    def __init__(self, plan):
        assert isinstance(plan, DedispersionPlan)

        channel_map = plan.config.make_channel_map()
        rank = plan.config.tree_rank
        nfreq = plan.nfreq

        assert len(channel_map) == (2**rank + 1)
        assert np.all(channel_map[:-1] > channel_map[1:])  # monotone decreasing
        assert np.abs(channel_map[0] - nfreq) < 1.0e-10
        assert np.abs(channel_map[-1]) < 1.0e-10

        self.plan = plan
        self.ana_freqs = [ ]

        it1 = 2**rank

        for ifreq in range(nfreq):
            # For each 0 <= ifreq < nfreq, find it0 < it1 such that:
            #   - channel_map[it0] > (ifreq+1) > channel_map[it0+1] 
            #   - channel_map[it1-1] > ifreq > channel_map[it1]

            assert 1 <= it1 <= 2**rank
            assert (channel_map[it1-1] + 1.0e-10) > ifreq > (channel_map[it1] - 1.0e-10)

            it0 = it1-1
            while (it0 > 0) and (channel_map[it0] < ifreq+1):
                it0 -= 1

            w = np.copy(channel_map[it0:(it1+1)])
            w[0] = ifreq+1
            w[-1] = ifreq
            weights = w[:-1] - w[1:]

            a = AnalyticFrequencyChannel(weights, it0)
            self.ana_freqs.append(a)

            assert (channel_map[it0] + 1.0e-10) > (ifreq+1) > (channel_map[it0+1] - 1.0e-10)
            it1 = it0 + 1


    @functools.cached_property
    def tree_gridding_kernel(self):
        return ReferenceTreeGriddingKernel(
            nfreq = self.plan.nfreq,
            nchan = 2**self.plan.config.tree_rank,
            ntime = self.plan.nt_in,
            beams_per_batch = 1,
            channel_map = self.plan.config.make_channel_map()
        )


    def test_gridding(self):
        nfreq = self.plan.nfreq
        ntime = self.plan.nt_in
        ntree = 2**self.plan.config.tree_rank

        # src = np.random.uniform(-1.0, 1.0, size=(1,nfreq,ntime))
        src = np.random.uniform(-1.0, 1.0, size=(1,nfreq,ntime))
        src = np.array(src, dtype=np.float32)
        dst_ref = self.tree_gridding_kernel.apply(src)
        assert dst_ref.shape == (1, ntree, ntime)

        dst_ana = np.zeros((1,ntree,ntime), dtype=np.float32)

        for ifreq in range(nfreq):
            a = self.ana_freqs[ifreq]
            for it in range(a.it0, a.it1):
                w = a.weights[it-a.it0]
                dst_ana[:,it,:] += w * src[:,ifreq,:]

        maxdiff = float(np.max(np.abs(dst_ref - dst_ana)))
        print(f'AnalyticDedisperser.test_gridding(): {maxdiff=}')
        assert maxdiff < 1.0e-5


    @classmethod
    def test_random(cls):
        config = DedispersionConfig.make_random()
        print('AnalyticDedisperser.test_random(): start')
        print(config.to_yaml_string())

        plan = DedispersionPlan(config)
        ana_disp = cls(plan)
        ana_disp.test_gridding()
