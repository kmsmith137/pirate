
# Note: the FrequencySubbands class is in the 'cuda_generator' submodule, since it
# might be useful in 'makefile_helper.py', which imports the cuda_generator submodule,
# but not the toplevel pirate_frb module.

class FrequencySubbands:
    def __init__(self, *, subband_counts=None, pf_rank=None, fmin=None, fmax=None, threshold=None):
        """
    
        
                 
class SubbandHelper:
    def __init__(self, pf_rank):
        # Currently, pf_rank=4 is max value supported by the peak-finding kernel,
        # so a larger value would indicate a bug (such as using the total tree rank
        # instead of the peak-finding rank).
        assert 0 <= pf_rank <= 4
        
        self.pf_rank = pf_rank
        self.num_levels = pf_rank + 1
        self.band_width_at_level = [ 2**l for l in range(pf_rank+1) ]
        self.band_spacing_at_level = [ 2**max(l-1,0) for l in range(pf_rank+1) ]

    
    def max_bands_at_level(self, level):
        assert 0 <= level <= self.pf_rank
        bw = self.band_width_at_level[level]
        bs = self.band_spacing_at_level[level]
        return ((2**self.pf_rank - bw) // bs) + 1

    
    def get_band_limits(self, level, ix):
        """Returns (ilo, ihi), where 0 <= ilo < ihi <= 2**pf_rank."""
        
        assert 0 <= level <= self.pf_rank
        assert 0 <= ix < self.max_bands_at_level(level)

        ilo = ix * self.band_spacing_at_level[level]
        ihi = ilo + self.band_width_at_level[level]
        assert 0 <= ilo < ihi <= 2**self.pf_rank
        return ilo, ihi


class FrequencySubbands:
    def __init__(self, pf_rank, fmin, fmax, threshold):
        """Example: FrequencySubbands(4, 300, 1500, 0.1) for CHORD."""
        
        self.fmin = fmin
        self.fmax = fmax
        self.pf_rank = pf_rank
        self.threshold = threshold
        self.h = SubbandHelper(pf_rank)

        # subband_counts: length-(rank+1) list, containing number of frequency subbands
        # at each level. This is used as an "identifier" for frequency subbands in other
        # parts of the code generator (e.g. peak-finding kernel).
        
        self.subband_counts = [0]*pf_rank + [1]
        
        for level in range(pf_rank):  # not range(pf_rank+1)
            for ix in range(self.h.max_bands_at_level(level)):
                freq_lo, freq_hi = self.get_subband(level, ix)
                if (freq_hi / freq_lo) > (1+threshold):
                    self.subband_counts[level] += 1


    def get_subband(self, level, ix):
        """Returns (freq_lo, freq_hi)."""

        ilo, ihi = self.h.get_band_limits(level, ix)
        dmin, dmax = self.fmax**(-2), self.fmin**(-2)
        dlo = dmin + (ilo / 2**self.pf_rank) * (dmax-dmin)
        dhi = dmin + (ihi / 2**self.pf_rank) * (dmax-dmin)
        return dhi**(-0.5), dlo**(-0.5)   # freq_lo, freq_hi


    def show(self):
        print(f'pf_rank={self.pf_rank}, fmin={self.fmin}, fmax={self.fmax}, threshold={self.threshold}')
        F = 0  # number of distinct frequency subbands
        M = 0  # number of "multiplets" in search, i.e. (frequency subband, fine dm) pairs
        
        for level,n in enumerate(self.subband_counts):
            F += n
            M += n * 2**level
            
            print(f'  {level=}:', end='')
            for ix in range(n):
                freq_lo, freq_hi = self.get_subband(level,ix)
                print(f' [{freq_lo:.01f},{freq_hi:.01f}]', end='')
            print()

        print(f'  {F=}  (number of distinct frequency subbands)')
        print(f'  {M=}  (number of "multiplets" in search, i.e. (frequency subband, fine dm) pairs)')
        

####################################################################################################


if __name__ == '__main__':
    FrequencySubbands(4, 300, 1500, 0.1).show()
    FrequencySubbands(4, 400, 800, 0.1).show()
