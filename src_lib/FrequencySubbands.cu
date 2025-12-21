#include "../include/pirate/FrequencySubbands.hpp"
#include "../include/pirate/inlines.hpp"  // pow2()
#include "../include/pirate/utils.hpp"    // integer_log2()

#include <stdexcept>
#include <ksgpu/xassert.hpp>
#include <ksgpu/rand_utils.hpp>    // rand_int()
#include <ksgpu/string_utils.hpp>  // tuple_str()

using namespace std;
using namespace ksgpu;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// Note: there is a similar python class (pirate_frb.cuda_generator.FrequencySubbands), so changes
// made here should also be reflected there.


FrequencySubbands::FrequencySubbands(const vector<long> &subband_counts_) :
    subband_counts(subband_counts_)
{
    validate_subband_counts(subband_counts);

    this->pf_rank = subband_counts.size() - 1;
    this->F = 0;
    this->M = 0;
    
    for (long level = 0; level <= pf_rank; level++) {
        for (long b = 0; b < subband_counts.at(level); b++) {
            long s = pow2(max(level-1,0L));   // spacing between bands
            long f = this->F;                 // current value

            this->f_to_ilo.push_back(b*s);
            this->f_to_ihi.push_back(b*s + pow2(level));
            this->f_to_mbase.push_back(M);
                
            for (long d = 0; d < pow2(level); d++) {
                this->m_to_f.push_back(f);
                this->m_to_d.push_back(d);
            }

            this->M += pow2(level);
            this->F += 1;
        }
    }

    xassert_eq(m_to_f.size(), uint(M));
    xassert_eq(f_to_ilo.size(), uint(F));
}


// Static member function
void FrequencySubbands::validate_subband_counts(const std::vector<long> &subband_counts)
{
    long pf_rank = subband_counts.size() - 1;

    xassert(subband_counts.size() > 0);
    xassert_eq(subband_counts.at(pf_rank), 1);  // must search full band
    
    // Currently, pf_rank=4 is max value supported by the peak-finding kernel,
    // so a larger value would indicate a bug (such as using the total tree rank
    // instead of the peak-finding rank).
    if (pf_rank > 4)
        throw std::runtime_error("FrequencySubbands: max allowed pf_rank is 4. This may change in the future.");
    
    for (long level = 0; level <= pf_rank; level++) {
        // Level 0 is special (non-overlapping bands).
        long max_bands = (level > 0) ? (pow2(pf_rank+1-level)-1) : pow2(pf_rank);
        xassert_ge(subband_counts.at(level), 0);
        xassert_le(subband_counts.at(level), max_bands);
    }        
}


// Static member function
vector<long> FrequencySubbands::make_random_subband_counts()
{
    long pf_rank = rand_int(0,5);
    vector<long> subband_counts(pf_rank+1);

    for (long level = 0; level < pf_rank; level++) {
        // Level 0 is special (non-overlapping bands).
        long max_bands = (level > 0) ? (pow2(pf_rank+1-level)-1) : pow2(pf_rank);
        subband_counts[level] = rand_int(0,max_bands+1);
    }

    subband_counts[pf_rank] = 1;
    return subband_counts;
}


// Static member function
FrequencySubbands FrequencySubbands::make_random()
{
    vector<long> subband_counts = make_random_subband_counts();
    return FrequencySubbands(subband_counts);
}


void FrequencySubbands::show_token(uint token, ostream &os) const
{
    // (t) | (p << 8) | (m << 16)
    uint t = (token) & 0xffu;
    uint p = (token >> 8) & 0xffu;
    uint m = (token >> 16);

    os << " -> (t=" << t << ", p=" << p << ", m=" << m << ")";

    if (m < M) {
        os << " -> BAD M-VALUE";
        return;
    }

    long f = m_to_f.at(m);
    long d = m_to_d.at(m);
    long f0 = f_to_ilo.at(f);
    long f1 = f_to_ihi.at(f);
    os << " -> (f0=" << f0 << ", f1=" << f1 << ", d=" << d << ")";
}


void FrequencySubbands::show(ostream &os) const
{
    os << "FrequencySubbands(pf_rank=" << pf_rank
       << ", subband_counts=" << ksgpu::tuple_str(subband_counts) << ")\n";

    for (long f = 0; f < F; f++) {
        long ilo = f_to_ilo.at(f);
        long ihi = f_to_ihi.at(f);
        long mlo = f_to_mbase.at(f);
        long mhi = mlo + (ihi - ilo);
        long level = integer_log2(ihi - ilo);
        os << "  f=" << f << ": level=" << level
           << "  (mlo,mhi)=(" << mlo << "," << mhi << ")"
           << "  (ilo,ihi)=(" << ilo << "," << ihi << ")\n";
    }

    os << "F=" << F << "  # number of distinct frequency subbands\n";
    os << "M=" << M << "  # number of \"multiplets\", i.e. (frequency_subband, fine_grained_dm) pairs\n";
}


string FrequencySubbands::to_string() const
{
    stringstream ss;
    this->show(ss);
    return ss.str();
}


}  // namespace pirate

