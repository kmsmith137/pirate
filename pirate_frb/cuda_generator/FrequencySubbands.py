import re
import numpy as np

from . import utils


class FrequencySubbands:
    def __init__(self, subband_counts):    
        """Python reimplementation of the C++ FrequencySubbands, for kernel generation.

        This is probably not the class you want! You probably want the C++ class 
        pirate_frb.FrequencySubbands (or equivalently, pirate_frb.pybind11.FrequencySubbands).
        
        This class (pirate_frb.cuda_generator.FrequencySubbands) is a python reimplementation
        of a subset of the functionality of the C++ class. It is only used by the code generator
        (and makefile_helper.py), not the main pirate_frb package. This klduge is necessary becuase
        the code generator runs during build time, when the C++ code has not been compiled yet.
        """
        
        # Full validation (same checks as the C++ constructor, which also calls
        # validate_subband_counts() -- keep the two constructors consistent).
        self.validate_subband_counts(subband_counts)

        self.subband_counts = subband_counts
        self.pf_rank = pf_rank = len(subband_counts) - 1

        self.N = 0               # number of frequency_subbands
        self.M = 0               # number of "multiplets", i.e. (frequency_subband, fine_grained_dm) pairs
        self.m_to_nd = [ ]       # mapping (multiplet) -> (frequency_subband, fine_grained_dm)
        self.n_to_frange = [ ]   # mapping (frequency_subband) -> (coarse-freq index pair 0 <= flo < fhi <= 2**rank)
        self.n_to_mrange = [ ]   # mapping (frequency_subband) -> (multiplet pair 0 <= mlo < mhi <= M)

        for level in range(pf_rank+1):
            for b in range(self.subband_counts[level]):
                for d in range(2**level):
                    self.m_to_nd.append((self.N,d))
                
                # Compute (flo, fhi) for this band: 0 <= flo < fhi <= 2**pf_rank
                s = 2**max(level-1, 0)  # spacing between bands
                flo = b * s
                fhi = b * s + 2**level
                
                self.n_to_frange.append((flo,fhi))
                self.n_to_mrange.append((self.M, self.M + 2**level))

                self.M += 2**level
                self.N += 1
        
        # For kernel/file names in code generator. Note that leading zeros are meaningful and
        # are never stripped: 'n0_n0_n3_n2_n1' and 'n3_n2_n1' are different kernels. For a
        # dedispersion kernel they are different band sets; for a peak-finding kernel, whose
        # counts denote a multiplet run-length structure rather than a band set, k leading
        # zeros mean 2^k times as many multiplets per band (see the top of PeakFinder.py).
        self.fstr = '_'.join(f'n{int(c)}' for c in self.subband_counts)

        
    @classmethod
    def from_fstr(cls, fstr):
        # For parsing filenames in code generator.
        if not re.fullmatch(r'n\d+(?:_n\d+)*', fstr):
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
    def can_early_trigger(cls, subband_counts, early_trigger_level):
        """True if restrict_subband_counts() is well-defined for this pair.

        Two conditions: the truncation must be in range (early_trigger_level <= pf_rank),
        and the early-trigger tree's own full band must already be one of the config's
        bands -- otherwise the early trigger would ADD a subband that the config never
        asked to search.

        Keep in sync with the C++ FrequencySubbands::can_early_trigger().
        """

        if early_trigger_level < 0:
            raise RuntimeError("FrequencySubbands.can_early_trigger: early_trigger_level must be >= 0")

        cls.validate_subband_counts(subband_counts)
        pf_rank = len(subband_counts) - 1

        if early_trigger_level > pf_rank:
            return False

        return subband_counts[pf_rank - early_trigger_level] >= 1


    @classmethod
    def restrict_subband_counts(cls, subband_counts, early_trigger_level):
        """"Restrict" a config's toplevel subband counts to one tree of that config.

        Truncates by 'early_trigger_level' levels (a no-op if it is zero), then clamps
        each surviving level to the number of bands that fit in the smaller tree. The
        result is always a SUBSET of the input band set. Note it can have
        pf_rank < (dd_rank+1)//2, which is the case the "extra DM" kernels handle.

        Keep in sync with the C++ FrequencySubbands::restrict_subband_counts(): this
        function decides which kernels makefile_helper.py compiles, while the C++ one
        decides which kernels a DedispersionPlan's trees ask for, and a divergence surfaces
        much later as "Kernel not found in registry".
        """

        if early_trigger_level < 0:
            raise RuntimeError("FrequencySubbands.restrict_subband_counts: early_trigger_level must be >= 0")

        cls.validate_subband_counts(subband_counts)
        ret = list(subband_counts)

        if early_trigger_level > 0:
            if not cls.can_early_trigger(subband_counts, early_trigger_level):
                raise RuntimeError(f"FrequencySubbands.restrict_subband_counts: early_trigger_level="
                                   f"{early_trigger_level} is not usable with subband_counts={subband_counts}")

            del ret[len(ret) - early_trigger_level : ]
            new_rank = len(ret) - 1

            # Drop the bands that stick out past the early-trigger tree's narrowed range.
            # Bands are enumerated from the low tree-freq end, which is the end an early
            # trigger keeps, so the survivors are a prefix.
            for level in range(new_rank + 1):
                max_bands = (2**(new_rank+1-level) - 1) if (level > 0) else 2**new_rank
                ret[level] = min(ret[level], max_bands)

        cls.validate_subband_counts(ret)
        return ret
    
    
    def max_bands_at_level(self, level):
        # Level 0 is special (non-overlapping bands).
        assert 0 <= level <= self.pf_rank
        return (2**(self.pf_rank+1-level) - 1) if (level > 0) else (2**self.pf_rank)

    
    def get_band_index_range(self, level, b):
        """Returns (flo, fhi), where 0 <= flo < fhi <= 2**pf_rank."""
        
        assert 0 <= level <= self.pf_rank
        assert 0 <= b < self.max_bands_at_level(level)

        s = 2**max(level-1,0)         # spacing between bands
        return (b*s, b*s + 2**level)  # (flo, fhi)


    def check_m(self, m, expected_flo, expected_fhi, expected_d):
        n,d = self.m_to_nd[m]
        assert self.n_to_frange[n] == (expected_flo, expected_fhi)
        assert d == expected_d