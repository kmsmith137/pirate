#ifndef _PIRATE_FREQUENCY_SUBBANDS_HPP
#define _PIRATE_FREQUENCY_SUBBANDS_HPP

#include <vector>
#include <iostream>


namespace pirate {
#if 0
}  // editor auto-indent
#endif


// FrequencySubbands: describes the frequency subband structure used in peak-finding.
//
// Note: there is a similar python class (pirate_frb.cuda_generator.FrequencySubbands), so changes
// made here should also be reflected there.

struct FrequencySubbands
{
    FrequencySubbands(const std::vector<long> &subband_counts);

    // Length-(pf_rank+1) vector, containing number of frequency subbands at each level.
    // This vector is used as an "identifier" for frequency subbands in low-level code.
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

