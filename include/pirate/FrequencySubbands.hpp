#ifndef _PIRATE_FREQUENCY_SUBBANDS_HPP
#define _PIRATE_FREQUENCY_SUBBANDS_HPP

#include <vector>
#include <string>
#include <iostream>
#include <sstream>


namespace pirate {
#if 0
}  // editor auto-indent
#endif


// FrequencySubbands: defines frequency sub-bands for the FRB search. 
// This can improve SNR for bursts that don't span the full frequency range.
//
// Note: there is a similar python class (pirate_frb.cuda_generator.FrequencySubbands), 
// so changes made here should also be reflected there.

struct FrequencySubbands
{
    FrequencySubbands();  // default constructor, equivalent to subband_counts={1}
    FrequencySubbands(const std::vector<long> &subband_counts);
    FrequencySubbands(const std::vector<long> &subband_counts, double fmin, double fmax);

    // subband_counts: length-(pf_rank+1) vector, containing number of frequency subbands 
    // at each level. This vector fully parameterizes the frequency subbands as follows.
    //
    //   - Divide the full frequency range into coarse-freq subranges 0 <= f < 2^pf_rank, equally
    //     spaced in delay (not frequency!), ordered so that f=0 is high-frequency.
    //
    //   - Frequency subbands are indexed by a pair (pf_level,s) where 0 <= pf_level <= pf_rank
    //     and 0 <= s < subband_counts[pf_level].
    //
    //   - pf_level=0 is special: s=0,1,... corresponds directly to f=0,1,... (non-overlapping)
    //
    //   - For pf_level > 0, each s=0,1,.. corresponds to an overlapping band
    //      (s * 2^(pf_level-1)) <= f < ((s+2) * 2^(pf_level-1))
    //
    //   - We require subband_counts[pf_rank]=1, i.e. the only "subband" at pf_level==pf_rank
    //     is the full frequency range 0 <= i < 2^(pf_rank).
    //
    //   - The details of this scheme, including the "specialness" of pf_level==0, are
    //     dictated by convenience in the GPU kernel.
    //
    //   - To disable subbands, and only search the full frequency band, set subband_counts = {1}.
    //
    //   - The command 'python -m pirate_frb dev make_subbands FMIN FMAX THRESHOLD [-r PF_RANK]'
    //     may be useful for constructing new subband_counts: it calls from_threshold()
    //     and prints the result via show().
    //
    // In the larger peak-finding kernel, each frequency subband is associated with 2^pf_level
    // "fine-grained" DMs. We define a "multiplet" to be a (frequency_subband, fine_grained_dm)
    // pair. The total number of multiplets M is obtained by summing (2^pf_level) over all
    // subbands. See comments in PeakFinding.hpp for more context.

    std::vector<long> subband_counts;   

    long pf_rank = -1;  // = subband_counts.size() - 1
    long N = 0;  // number of distinct frequency subbands
    long M = 0;  // number of "multiplets", i.e. (frequency_subband, fine_grained_dm) pairs

    std::vector<long> m_to_n;     // mapping (multiplet) -> frequency_subband
    std::vector<long> m_to_d;     // mapping (multiplet) -> fine_grained_dm
    std::vector<long> n_to_flo;   // mapping (frequency_subband) -> (coarse-freq index pair 0 <= flo < fhi <= 2^pf_rank)
    std::vector<long> n_to_fhi;   // mapping (frequency_subband) -> (coarse-freq index pair 0 <= flo < fhi <= 2^pf_rank)
    std::vector<long> n_to_level; // mapping (frequency_subband) -> (pf_level, i.e. the index into subband_counts)
    std::vector<long> n_to_mbase; // mapping (frequency_subband) -> m-index range (mbase : mbase + 2^level)

    // f_to_freq: mapping (coarse-freq index 0 <= f <= 2^pf_rank) -> (physical frequency)
    // Only defined if fmin/fmax are specified in the constructor; otherwise empty.
    std::vector<double> f_to_freq;
    double fmin = 0.0;
    double fmax = 0.0;

    inline long m_to_flo(int m) const { long n = m_to_n.at(m); return n_to_flo.at(n); }
    inline long m_to_fhi(int m) const { long n = m_to_n.at(m); return n_to_fhi.at(n); }

    void show(std::ostream &os = std::cout) const;
    void show_compact(std::stringstream &ss) const;  // requires fmin/fmax specified at construction
    std::string to_string() const;

    // Static member function.
    // Creates FrequencySubbands from frequency range and threshold.

    static FrequencySubbands from_threshold(double fmin, double fmax, double threshold, long pf_rank = 4);

    // Static member function.
    // "Restricts" a config's toplevel subband counts to one tree of that config.
    //
    // Truncates 'subband_counts' by 'early_trigger_level' levels (a no-op if it is zero),
    // then clamps each surviving level to the number of bands that fit in the smaller tree.
    // The result is always a SUBSET of the input band set: a tree never searches a band the
    // config did not ask for. (Both trees grid the band identically, because truncating one
    // subband level per early-trigger level is exactly what keeps the coarse-freq channel
    // width fixed -- see DedispersionTree.cpp's _toplevel_channels_per_coarse_channel().)
    //
    // Note the returned counts can have pf_rank < (dd_rank+1)/2, in which case the tree's
    // cdd2 kernel folds the difference into its argmax tokens (see DedispersionTree::xdm_rank).
    //
    // Requires can_early_trigger(subband_counts, early_trigger_level); throws otherwise. Any
    // config which passed DedispersionConfig::validate() satisfies this for every tree it
    // generates, so a throw here means the counts came from somewhere else.

    static std::vector<long> restrict_subband_counts(const std::vector<long> &subband_counts, long early_trigger_level);

    // Static member function.
    // True if restrict_subband_counts(subband_counts, early_trigger_level) is well-defined:
    // the truncation must be in range (early_trigger_level <= pf_rank), AND the
    // early-trigger tree's own full band must already be one of the config's bands. The
    // second condition is not a technicality: that band is a strict sub-range of the
    // toplevel band, so manufacturing it would ADD a subband that the config never asked to
    // search. Always true for early_trigger_level == 0.

    static bool can_early_trigger(const std::vector<long> &subband_counts, long early_trigger_level);

    // For debugging/testing.
    static void validate_subband_counts(const std::vector<long> &subband_counts);
    static std::vector<long> make_random_subband_counts(long pf_rank);
    static std::vector<long> make_random_subband_counts();
    static FrequencySubbands make_random();
};


}  // namespace pirate

#endif // _PIRATE_FREQUENCY_SUBBANDS_HPP

