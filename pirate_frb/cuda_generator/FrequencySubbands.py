import re
import numpy as np

from . import utils

# Note 1: the FrequencySubbands class is in the 'cuda_generator' submodule, since it
# might be useful in 'makefile_helper.py', which imports the cuda_generator submodule,
# but not the toplevel pirate_frb module.
#
# Note 2: there is a similar C++ class (pirate::FrequencySubbands), so changes made here
# should also be reflected there.

class FrequencySubbands:
    def __init__(self, *, subband_counts=None, pf_rank=None, fmin=None, fmax=None, threshold=None):
        """
        Constructor syntax 1: FrequencySubbands(subband_counts=[5,3,2,1], pf_rank=None, fmin=None, fmax=None)
        Constructor syntax 2: FrequencySubbands(pf_rank=4, fmin=300., fmax=1500., threshold=0.2)

        These examples are accessible from the command line:
          python -m pirate_frb show_subbands 5 3 2 1
          python -m pirate_frb show_subbands --rank=4 --fmin=300 --fmax=1500 --threshold=0.2
        
        Key members:
        
          - self.F = number of distinct frequency subbands
          - self.M = number of "multiplets", i.e. (frequency_subband, fine_grained_dm) pairs
          - self.f_to_irange: mapping (frequency_subband) -> (index pair 0 <= ilo < ihi <= 2**rank)
          - self.m_to_fd: mapping (multiplet) -> (frequency_subband, fine_grained_dm)
          - self.i_to_f:  mapping (0 <= index <= 2**rank) -> (frequency), only defined if fmin/fmax are specified
          - self.subband_counts: length-(rank+1) list, containing number of frequency subbands at each level.
            This is used as an "identifier" for frequency subbands in low-level code.
        """

        self.subband_counts = subband_counts
        self.threshold = threshold
        self.pf_rank = pf_rank
        self.fmin = fmin
        self.fmax = fmax

        have_sc = (subband_counts is not None)
        have_rk = (pf_rank is not None)
        have_fmin = (fmin is not None)
        have_fmax = (fmax is not None)
        have_th = (threshold is not None)

        constructor_syntax1 = have_sc and (not have_th) and (have_fmin == have_fmax)
        constructor_syntax2 = (not have_sc) and all([have_rk, have_fmin, have_fmax, have_th])

        if not (constructor_syntax1 or constructor_syntax2):
            raise RuntimeError(
                "FrequencySubbands: invalid combination of constructor args: "
                + f"({subband_counts=}, {pf_rank=}, {fmin=}, {fmax=}, {threshold=}")

        if constructor_syntax1:            
            if (pf_rank is not None):
                assert pf_rank == len(subband_counts) - 1
            
            assert len(subband_counts) > 0
            assert subband_counts[-1] == 1   # must search full band
            self.pf_rank = pf_rank = len(subband_counts) - 1

        # Currently, pf_rank=4 is max value supported by the peak-finding kernel,
        # so a larger value would indicate a bug (such as using the total tree rank
        # instead of the peak-finding rank).
        if (pf_rank > 4):
            raise RuntimeError('FrequencySubbands: max allowed pf_rank is 4. This may change in the future.')

        if have_fmin and have_fmax:
            # Initialize self.i_to_f: mapping (0 <= index <= 2**rank) -> (frequency)            
            assert 0 < fmin < fmax
            self.i_to_f = np.linspace(fmax**(-2), fmin**(-2), 2**pf_rank + 1)**(-0.5)

        if constructor_syntax2:
            assert pf_rank >= 0
            self.subband_counts = [0] * (pf_rank+1)
            
            for level in range(pf_rank + 1):
                for b in range(self.max_bands_at_level(level)):
                    ilo, ihi = self.get_band_index_range(level, b)
                    freq_lo, freq_hi = self.i_to_f[ihi], self.i_to_f[ilo]  # note (lo,hi) swap
                    if (level == pf_rank) or ((freq_hi/freq_lo) > (1+threshold)):
                        self.subband_counts[level] += 1
            
        self.F = 0               # number of frequency_subbands
        self.M = 0               # number of "multiplets", i.e. (frequency_subband, fine_grained_dm) pairs
        self.m_to_fd = [ ]       # mapping (multiplet) -> (frequency_subband, fine_grained_dm)
        self.f_to_irange = [ ]   # mapping (frequency_subband) -> (index pair 0 <= ilo < ihi <= 2**rank)
        self.f_to_mrange = [ ]   # mapping (frequency_subband) -> (multiplet pair 0 <= mlo < mhi <= M)

        for level in range(pf_rank+1):
            assert self.subband_counts[level] >= 0
            
            for b in range(self.subband_counts[level]):
                for d in range(2**level):
                    self.m_to_fd.append((self.F,d))
                
                ilo, ihi = self.get_band_index_range(level, b)
                self.f_to_irange.append((ilo,ihi))
                self.f_to_mrange.append((self.M, self.M + 2**level))

                self.M += 2**level
                self.F += 1
        
        # For kernel/file names in code generator.
        self.fstr = '_'.join(f'f{int(n)}' for n in self.subband_counts)

        
    @classmethod
    def from_fstr(cls, fstr):
        # For parsing filenames in code generator.
        if not re.fullmatch(r'f\d+(?:_f\d+)*', fstr):
            raise RuntimeError(f"FrequencySubbands.from_fstr(): couldn't parse fstr='{fstr}'")
        subband_counts = [int(x) for x in re.findall(r'\d+', fstr) ]
        return cls(subband_counts = subband_counts)


    @classmethod
    def make_random_subband_counts(cls, pf_rank=None):
        randi = lambda *args: int(np.random.randint(*args))

        if pf_rank is None:
            pf_rank = randi(1,5)

        assert 1 <= pf_rank <= 4
        return tuple([ randi(2**pf_rank) ] + [ randi(2**(pf_rank-l+1)-1) for l in range(1,pf_rank) ] + [ 1 ])
    
    
    def max_bands_at_level(self, level):
        # Level 0 is special (non-overlapping bands).
        assert 0 <= level <= self.pf_rank
        return (2**(self.pf_rank+1-level) - 1) if (level > 0) else (2**self.pf_rank)

    
    def get_band_index_range(self, level, b):
        """Returns (ilo, ihi), where 0 <= ilo < ihi <= 2**pf_rank."""
        
        assert 0 <= level <= self.pf_rank
        assert 0 <= b < self.max_bands_at_level(level)

        s = 2**max(level-1,0)         # spacing between bands
        return (b*s, b*s + 2**level)  # (ilo, ihi)


    def show(self):
        print(f'FrequencySubbands(pf_rank={self.pf_rank}, subband_counts={self.subband_counts},'
              + f' fmin={self.fmin}, fmax={self.fmax}, threshold={self.threshold})')

        for f,((ilo,ihi),(mlo,mhi)) in enumerate(zip(self.f_to_irange,self.f_to_mrange)):
            level = utils.integer_log2(ihi-ilo)
            line = f'  {f=}: {level=}  (mlo,mhi)=({mlo},{mhi})  (ilo,ihi)=({ilo},{ihi})'
            if hasattr(self, 'i_to_f'):
                flo, fhi = self.i_to_f[ihi], self.i_to_f[ilo]   # note (lo,hi) swap
                line += f'  (flo,fhi)=({flo:.01f},{fhi:.01f})'
            print(line)

        print(f'F={self.F}  # number of distinct frequency subbands')
        print(f'M={self.M}  # number of "multiplets", i.e. (frequency_subband, fine_grained_dm) pairs')
