import re
import numpy as np

from . import utils


class FrequencySubbands:
    def __init__(self, subband_counts):    
        """
        This is probably not the class you want! You probably want the C++ class 
        pirate_frb.FrequencySubbands (or equivalently, pirate_frb.pybind11.FrequencySubbands).
        
        This class (pirate_frb.cuda_generator.FrequencySubbands) is a python reimplementation
        of a subset of the functionality of the C++ class. It is only used by the code generator
        (and makefile_helper.py), not the main pirate_frb package. This klduge is necessary becuase
        the code generator runs during build time, when the C++ code has not been compiled yet.
        """
        
        assert len(subband_counts) > 0
        assert subband_counts[-1] == 1   # must search full band

        self.subband_counts = subband_counts
        self.pf_rank = pf_rank = len(subband_counts) - 1

        # Currently, pf_rank=4 is max value supported by the peak-finding kernel,
        # so a larger value would indicate a bug (such as using the total tree rank
        # instead of the peak-finding rank).
        if (pf_rank > 4):
            raise RuntimeError('FrequencySubbands: max allowed pf_rank is 4. This may change in the future.')

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
                
                # Compute (ilo, ihi) for this band: 0 <= ilo < ihi <= 2**pf_rank
                s = 2**max(level-1, 0)  # spacing between bands
                ilo = b * s
                ihi = b * s + 2**level
                
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


    @staticmethod
    def validate_subband_counts(subband_counts):
        """Validates subband_counts, raising an exception if invalid."""
        
        if len(subband_counts) == 0:
            raise RuntimeError("FrequencySubbands: subband_counts must be non-empty")
        
        pf_rank = len(subband_counts) - 1
        
        if subband_counts[pf_rank] != 1:
            raise RuntimeError("FrequencySubbands: last element of subband_counts must be 1 (must search full band)")
        
        # Currently, pf_rank=4 is max value supported by the peak-finding kernel,
        # so a larger value would indicate a bug (such as using the total tree rank
        # instead of the peak-finding rank).
        if pf_rank > 4:
            raise RuntimeError("FrequencySubbands: max allowed pf_rank is 4. This may change in the future.")
        
        for level in range(pf_rank + 1):
            # Level 0 is special (non-overlapping bands).
            max_bands = (2**(pf_rank+1-level) - 1) if (level > 0) else 2**pf_rank
            if not (0 <= subband_counts[level] <= max_bands):
                raise RuntimeError(f"FrequencySubbands: subband_counts[{level}]={subband_counts[level]} out of range [0,{max_bands}]")


    @classmethod
    def restrict_subband_counts(cls, subband_counts, et_delta_rank, new_pf_rank):
        """
        "Restricts" top-level subband counts to a specific tree.
        The tree may have an early trigger (et_delta_rank > 0) or a different pf_rank.
        """
        
        if et_delta_rank < 0:
            raise RuntimeError("FrequencySubbands.restrict_subband_counts: et_delta_rank must be >= 0")
        if new_pf_rank < 0:
            raise RuntimeError("FrequencySubbands.restrict_subband_counts: new_pf_rank must be >= 0")
        
        cls.validate_subband_counts(subband_counts)
        
        # Step 1: apply early trigger (et_delta_rank).

        src_rank = len(subband_counts) - 1
        early_rank = max(src_rank - et_delta_rank, 0)
        
        early_counts = [0] * (early_rank + 1)
        early_counts[early_rank] = 1
        
        for pf_level in range(early_rank):
            max_count = (2**(early_rank+1-pf_level) - 1) if (pf_level > 0) else 2**early_rank
            early_counts[pf_level] = min(subband_counts[pf_level], max_count)
        
        # Step 2: apply new_pf_rank.

        dst_counts = [0] * (new_pf_rank + 1)
        dst_counts[new_pf_rank] = 1
        
        for pf_level in range(new_pf_rank):
            src_level = pf_level - new_pf_rank + early_rank
            
            # Some awkward logic here, to account for pf_level==0 being "special".
            if (src_level < 0) or (early_counts[src_level] == 0):
                dst_counts[pf_level] = 0
            elif (src_level == 0) and (pf_level > 0):
                dst_counts[pf_level] = 2 * early_counts[src_level] - 1
            elif (src_level > 0) and (pf_level == 0):
                dst_counts[pf_level] = early_counts[src_level] // 2 + 1
            else:
                dst_counts[pf_level] = early_counts[src_level]
        
        cls.validate_subband_counts(dst_counts)
        return dst_counts
    
    
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


    def check_m(self, m, expected_ilo, expected_ihi, expected_d):
        f,d = self.m_to_fd[m]
        assert self.f_to_irange[f] == (expected_ilo, expected_ihi)
        assert d == expected_d