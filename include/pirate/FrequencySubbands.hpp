#ifndef _PIRATE_FREQUENCY_SUBBANDS_HPP
#define _PIRATE_FREQUENCY_SUBBANDS_HPP

#include <vector>
#include <iostream>


namespace pirate {
#if 0
}  // editor auto-indent
#endif


// FrequencySubbands: defines frequency sub-bands for the FRB search. 
// This can improve SNR for bursts that don't span the full frequency range.
//
// Note: there is a similar python class (pirate_frb.cuda_generator.FrequencySubbands), so changes
// made here should also be reflected there.

struct FrequencySubbands
{
    FrequencySubbands(const std::vector<long> &subband_counts);

    // subband_counts: length-(pf_rank+1) vector, containing number of frequency subbands 
    // at each level. This vector fully parameterizes the frequency subbands as follows.
    //
    //   - Divide the full frequency range into subranges 0 <= i < 2^pf_rank, equally
    //     spaced in delay (not frequency!), ordered so that i=0 is high-frequency.
    //
    //   - Frequency subbands are indexed by a pair (pf_level,s) where 0 <= pf_level <= pf_rank 
    //     and 0 <= s < subband_counts[f].
    //
    //   - pf_level=0 is special: s=0,1,... corresponds directly to i=0,1,... (non-overlapping)
    //
    //   - For pf_level > 0, each s=0,1,.. corresponds to an overlapping band 
    //      (s * 2^(pf_level-1)) <= i < ((s+2) * 2^(pf_level-1))
    //
    //   - We require subband_counts[pf_rank]=1, and the only "subband" at pf_level==pf_rank
    //     is the full frequency range 0 <= i < 2^(pf_rank).
    //
    //   - The details of this scheme, including the "specialness" of pf_level==0, are
    //     dictated by convenience in the GPU kernel.
    //
    //   - To disable subbands, and only search the full frequency band, set subband_counts = {1}.
    //
    //   - The commands 'python -m pirate_frb show_subbands' and 'python -m pirate_frb show_config --subbands'
    //     may be useful for constructing new subband_counts, or displaying subband_counts verbosely. 

    std::vector<long> subband_counts;   

    long pf_rank = -1;  // = subband_counts.size() - 1
    long F = 0;  // number of distinct frequency subbands
    long M = 0;  // number of "multiplets", i.e. (frequency_subband, fine_grained_dm) pairs

    std::vector<long> m_to_f;     // mapping (multiplet) -> (frequency_subband, fine_grained_dm)
    std::vector<long> m_to_d;     // mapping (multiplet) -> (frequency_subband, fine_grained_dm)
    std::vector<long> f_to_ilo;   // mapping (frequency_subband) -> (index pair 0 <= ilo < ihi <= 2^pf_rank)
    std::vector<long> f_to_ihi;   // mapping (frequency_subband) -> (index pair 0 <= ilo < ihi <= 2^pf_rank)
    std::vector<long> f_to_mbase; // mapping (frequency_subband) -> m-index range (mbase : mbase + 2^level)

    // These members are used in the peak-finding kernel, whose 'out_argmax' array consists
    // of "tokens" of the form (t) | (p << 8) | (m << 16).

    // For debugging/testing.
    static void validate_subband_counts(const std::vector<long> &subband_counts);
    static std::vector<long> make_random_subband_counts();
    static FrequencySubbands make_random();

    void show_token(uint token, std::ostream &os = std::cout) const;
    void show(std::ostream &os = std::cout) const;

    inline long m_to_ilo(int m) const { long f = m_to_f.at(m); return f_to_ilo.at(f); }
    inline long m_to_ihi(int m) const { long f = m_to_f.at(m); return f_to_ihi.at(f); }
};


}  // namespace pirate

#endif // _PIRATE_FREQUENCY_SUBBANDS_HPP

