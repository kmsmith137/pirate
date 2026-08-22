#include "../include/pirate/FrequencySubbands.hpp"
#include "../include/pirate/constants.hpp"  // constants::max_peak_finding_rank
#include "../include/pirate/inlines.hpp"  // pow2()
#include "../include/pirate/utils.hpp"    // integer_log2()

#include <cmath>
#include <iomanip>    // std::fixed, std::setprecision
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


// Note: there is a similar python class (pirate_frb.cuda_generator.FrequencySubbands)
// so changes made here should also be reflected there.


// Default constructor, equivalent to subband_counts={1}
FrequencySubbands::FrequencySubbands() :
    FrequencySubbands(vector<long>({1}))
{ }


FrequencySubbands::FrequencySubbands(const vector<long> &subband_counts_) :
    subband_counts(subband_counts_)
{
    validate_subband_counts(subband_counts);

    this->pf_rank = subband_counts.size() - 1;
    this->N = 0;
    this->M = 0;

    for (long level = 0; level <= pf_rank; level++) {
        for (long b = 0; b < subband_counts.at(level); b++) {
            long s = pow2(max(level-1,0L));   // spacing between bands
            long n = this->N;                 // current subband index

            this->n_to_flo.push_back(b*s);
            this->n_to_fhi.push_back(b*s + pow2(level));
            this->n_to_level.push_back(level);
            this->n_to_mbase.push_back(M);

            for (long d = 0; d < pow2(level); d++) {
                this->m_to_n.push_back(n);
                this->m_to_d.push_back(d);
            }

            this->M += pow2(level);
            this->N += 1;
        }
    }

    xassert_eq(m_to_n.size(), uint(M));
    xassert_eq(n_to_flo.size(), uint(N));
    xassert_eq(n_to_level.size(), uint(N));
}


FrequencySubbands::FrequencySubbands(const vector<long> &subband_counts_, double fmin_, double fmax_) :
    FrequencySubbands(subband_counts_)
{
    this->fmin = fmin_;
    this->fmax = fmax_;
    
    // Initialize f_to_freq: mapping (coarse-freq index 0 <= f <= 2^pf_rank) -> (physical frequency)
    // Following Python logic: np.linspace(fmax**(-2), fmin**(-2), 2**pf_rank + 1)**(-0.5)

    xassert(fmin > 0);
    xassert(fmax > fmin);

    long nf = pow2(pf_rank) + 1;
    double start = pow(fmax, -2.0);
    double end = pow(fmin, -2.0);

    this->f_to_freq.resize(nf);
    for (long f = 0; f < nf; f++) {
        double t = double(f) / double(nf-1);
        double val = start + t * (end - start);
        this->f_to_freq[f] = pow(val, -0.5);
    }
}


// Static member function
FrequencySubbands FrequencySubbands::from_threshold(double fmin, double fmax, double threshold, long pf_rank)
{
    // Input validation
    xassert(fmin > 0);
    xassert(fmax > fmin);
    xassert(pf_rank >= 0);
    
    // Currently, pf_rank=4 is max value supported by the peak-finding kernel
    if (pf_rank > constants::max_peak_finding_rank)
        throw std::runtime_error("FrequencySubbands::from_threshold: max allowed pf_rank is "
                                 + std::to_string(constants::max_peak_finding_rank)
                                 + ". This may change in the future.");
    
    // Initialize f_to_freq: mapping (coarse-freq index 0 <= f <= 2^pf_rank) -> (physical frequency)
    // Following Python logic: np.linspace(fmax**(-2), fmin**(-2), 2**pf_rank + 1)**(-0.5)

    long nf = pow2(pf_rank) + 1;
    double start = pow(fmax, -2.0);
    double end = pow(fmin, -2.0);

    vector<double> f_to_freq(nf);
    for (long f = 0; f < nf; f++) {
        double t = double(f) / double(nf-1);
        double val = start + t * (end - start);
        f_to_freq[f] = pow(val, -0.5);
    }

    // Build subband_counts by iterating through levels and bands
    vector<long> subband_counts(pf_rank+1, 0);

    for (long level = 0; level <= pf_rank; level++) {
        // Level 0 is special (non-overlapping bands).
        long max_bands = (level > 0) ? (pow2(pf_rank+1-level) - 1) : pow2(pf_rank);

        for (long b = 0; b < max_bands; b++) {
            // Compute (flo, fhi) for this band: 0 <= flo < fhi <= 2^pf_rank
            long s = pow2(max(level-1, 0L));  // spacing between bands
            long flo = b * s;
            long fhi = b * s + pow2(level);

            // Note: (lo,hi) swap when mapping coarse-freq indices to physical frequencies
            double freq_lo = f_to_freq[fhi];
            double freq_hi = f_to_freq[flo];
            
            // Include band if at top level, or if fractional bandwidth exceeds threshold
            if ((level == pf_rank) || ((freq_hi / freq_lo) > (1.0 + threshold))) {
                subband_counts[level]++;
            }
        }
    }
    
    // Call existing constructor with computed subband_counts, fmin, and fmax
    return FrequencySubbands(subband_counts, fmin, fmax);
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
    if (pf_rank > constants::max_peak_finding_rank)
        throw std::runtime_error("FrequencySubbands: max allowed pf_rank is "
                                 + std::to_string(constants::max_peak_finding_rank)
                                 + ". This may change in the future.");
    
    for (long level = 0; level <= pf_rank; level++) {
        // Level 0 is special (non-overlapping bands).
        long max_bands = (level > 0) ? (pow2(pf_rank+1-level)-1) : pow2(pf_rank);
        xassert_ge(subband_counts.at(level), 0);
        xassert_le(subband_counts.at(level), max_bands);
    }        
}

// Static member function.
// Keep in sync with the python twin, pirate_frb.cuda_generator.FrequencySubbands.
// See the doc-comment in FrequencySubbands.hpp.
bool FrequencySubbands::can_early_trigger(const vector<long> &subband_counts, long early_trigger_level)
{
    validate_subband_counts(subband_counts);
    xassert(early_trigger_level >= 0);

    long pf_rank = subband_counts.size() - 1;

    if (early_trigger_level > pf_rank)
        return false;

    return (subband_counts.at(pf_rank - early_trigger_level) >= 1);
}

// Static member function.
// Keep in sync with the python twin, pirate_frb.cuda_generator.FrequencySubbands
// (makefile_helper.py uses it to decide which kernels to compile, and a divergence would
// silently compile a kernel set that no DedispersionTree asks for).
// See the doc-comment in FrequencySubbands.hpp.
vector<long> FrequencySubbands::restrict_subband_counts(const vector<long> &subband_counts, long early_trigger_level)
{
    validate_subband_counts(subband_counts);
    xassert(early_trigger_level >= 0);

    vector<long> ret = subband_counts;

    if (early_trigger_level > 0) {
        // Defensive: unreachable from a config which passed DedispersionConfig::validate(),
        // which checks can_early_trigger() for every early-trigger level the config can
        // produce. Asserting here rather than manufacturing the missing band is what makes
        // "a tree's bands are a subset of the config's bands" true.
        xassert(can_early_trigger(subband_counts, early_trigger_level));

        ret.resize(ret.size() - early_trigger_level);
        long new_rank = ret.size() - 1;

        // Drop the bands that stick out past the early-trigger tree's narrowed range. Bands
        // are enumerated from the low tree-freq end, which is the end an early trigger
        // keeps, so the survivors are a prefix and clamping the count is the right
        // operation. Note the loop covers level new_rank too, where max_bands is 1 and the
        // can_early_trigger() check above guarantees the count is already >= 1.
        for (long level = 0; level <= new_rank; level++) {
            long max_bands = (level > 0) ? (pow2(new_rank+1-level)-1) : pow2(new_rank);
            ret.at(level) = min(ret.at(level), max_bands);
        }
    }

    validate_subband_counts(ret);
    return ret;
}

// Static member function
vector<long> FrequencySubbands::make_random_subband_counts(long pf_rank)
{
    xassert(pf_rank >= 0);
    vector<long> subband_counts(pf_rank+1);

    for (long level = 0; level < pf_rank; level++) {
        // Level 0 is special (non-overlapping bands).
        long max_count = (level > 0) ? (pow2(pf_rank+1-level)-1) : pow2(pf_rank);
        subband_counts[level] = rand_int(0,max_count+1);
    }

    subband_counts[pf_rank] = 1;
    return subband_counts;
}

// Static member function
vector<long> FrequencySubbands::make_random_subband_counts()
{
    long pf_rank = rand_int(0,5);
    return make_random_subband_counts(pf_rank);
}

// Static member function
FrequencySubbands FrequencySubbands::make_random()
{
    vector<long> subband_counts = make_random_subband_counts();
    return FrequencySubbands(subband_counts);
}


void FrequencySubbands::show(ostream &os) const
{
    // Save stream state (format flags and precision)
    ios::fmtflags oldFlags = os.flags();
    streamsize oldPrecision = os.precision();

    os << "FrequencySubbands(subband_counts=" << ksgpu::tuple_str(subband_counts);

    if (fmin > 0.0)
        os << fixed << setprecision(1) << ", fmin=" << fmin << ", fmax=" << fmax;

    os << ")\n";

    for (long n = 0; n < N; n++) {
        long flo = n_to_flo.at(n);
        long fhi = n_to_fhi.at(n);
        long mlo = n_to_mbase.at(n);
        long mhi = mlo + (fhi - flo);
        long level = integer_log2(fhi - flo);
        os << "  n=" << n << ": level=" << level
           << "  (mlo,mhi)=(" << mlo << "," << mhi << ")"
           << "  (flo,fhi)=(" << flo << "," << fhi << ")";

        // Add (freq_lo,freq_hi) if f_to_freq is available
        if (f_to_freq.size() > 0) {
            double freq_lo = f_to_freq.at(fhi);  // note fhi here
            double freq_hi = f_to_freq.at(flo);  // note flo here
            os << "  (freq_lo,freq_hi)=(" << fixed << setprecision(1)
               << freq_lo << "," << freq_hi << ")";
        }

        os << "\n";
    }

    os << "N=" << N << "  # number of distinct frequency subbands\n";
    os << "M=" << M << "  # number of \"multiplets\", i.e. (frequency_subband, fine_grained_dm) pairs\n";

    // Restore stream state
    os.flags(oldFlags);
    os.precision(oldPrecision);
}


string FrequencySubbands::to_string() const
{
    stringstream ss;
    this->show(ss);
    return ss.str();
}


void FrequencySubbands::show_compact(stringstream &ss) const
{
    if (f_to_freq.size() == 0)
        throw runtime_error("FrequencySubbands::show_compact(): fmin/fmax must be specified at construction");

    // Save stream state (format flags and precision)
    ios::fmtflags oldFlags = ss.flags();
    streamsize oldPrecision = ss.precision();

    long curr_level = -1;
    int bands_on_line = 0;

    for (long n = 0; n < N; n++) {
        long flo = n_to_flo.at(n);
        long fhi = n_to_fhi.at(n);
        long level = integer_log2(fhi-flo);
        double freq_lo = f_to_freq.at(fhi);  // note fhi here
        double freq_hi = f_to_freq.at(flo);  // note flo here

        if (level != curr_level) {
            ss << "\n  pf_level=" << level << ": ";
            bands_on_line = 0;
        }
        else if (bands_on_line >= 5) {
            ss << ",\n              ";  // align with first band after "pf_level=N: "
            bands_on_line = 0;
        }
        else {
            ss << ", ";
        }

        ss << "[" << fixed << setprecision(1) << freq_lo << "," << freq_hi << "]";
        curr_level = level;
        bands_on_line++;
    }

    // Restore stream state
    ss.flags(oldFlags);
    ss.precision(oldPrecision);
}


}  // namespace pirate

