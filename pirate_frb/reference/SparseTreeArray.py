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

      self.initial_f0, self.initial_nf:
        These are the values of (f0,nf) when a k=0 SparseTreeArray is created via
        SparseTreeArray.make_tree_gridding_output(). Later, when SparseTreeArray.iterate()
        is called, a new SparseTreeArray is returned with new values of (f0,nf), but
        unmodified values of (initial_f0, initial_nf).
    """

        
    @staticmethod
    def make_tree_gridding_output(channel_map, ifreq):
        """
        Suppose the TreeGriddingKernel is called on a "one-hot" shape (nfreq,ntime)
        array whose (ifreq,0) entry is equal to 1. The output is a shape (2^rank, 1, ntime)
        array which is mostly zeros. This method returns an equivalent SparseTreeArray.
        """
        pass


    @staticmethod
    def make_dedispersion_output(channel_map, ifreq):
        """
        Suppose TreeGriddingKernel -> (Tree dedispersion) is called on a "one-hot"
        shape (nfreq,ntime) array whose (ifreq,0) entry is equal to 1. The output
        is a shape (1, 2^rank, ntime) array which is mostly zeros. This method returns
        an equivalent SparseTreeArray. We assume a non-bit-reversed delay index.
        """
        sarr = SparseTreeArray(channel_map, ifreq)
        while sarr.k < sarr.r:
            sarr = sarr.iterate()
        return sarr
    
        
    def iterate(self, sarr):
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
        
        The caller-specified 'ntime' must be large enough to contain all nonzero elements,
        accounting for tshifts. If not, we raise an exception.
        """
        pass
    
