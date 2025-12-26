#include "../include/pirate/DedispersionConfig.hpp"

#include <cstring>                 // strlen()
#include <algorithm>               // std::sort()
#include <iomanip>                 // std::fixed, std::setprecision

#include <ksgpu/Dtype.hpp>
#include <ksgpu/xassert.hpp>
#include <ksgpu/cuda_utils.hpp>    // CUDA_CALL()
#include <ksgpu/rand_utils.hpp>    // ksgpu::rand_*()
#include <ksgpu/string_utils.hpp>  // ksgpu::tuple_str()

#include "../include/pirate/constants.hpp"
#include "../include/pirate/utils.hpp"           // integer_log2()
#include "../include/pirate/inlines.hpp"         // xdiv(), pow2(), print_kv(), is_power_of_two()
#include "../include/pirate/file_utils.hpp"      // File
#include "../include/pirate/YamlFile.hpp"
#include "../include/pirate/FrequencySubbands.hpp"
#include "../include/pirate/CoalescedDdKernel2.hpp"

#include <yaml-cpp/emitter.h>

using namespace std;
using namespace ksgpu;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


bool operator==(const DedispersionConfig::EarlyTrigger &x, const DedispersionConfig::EarlyTrigger &y)
{
    return (x.ds_level == y.ds_level) && (x.delta_rank == y.delta_rank);
}

bool operator>(const DedispersionConfig::EarlyTrigger &x, const DedispersionConfig::EarlyTrigger &y)
{
    if (x.ds_level > y.ds_level)
        return true;
    if (x.ds_level < y.ds_level)
        return false;
    
    // Note reversed inequalities here, so that std::sort() is first by increasing ds_level, 
    // and second by decreasing delta_rank.
    if (x.delta_rank < y.delta_rank)
        return true;
    if (x.delta_rank > y.delta_rank)
        return false;
    
    return false;  // equal
}

bool operator<(const DedispersionConfig::EarlyTrigger &x, const DedispersionConfig::EarlyTrigger &y)
{
    return y > x;
}


// -------------------------------------------------------------------------------------------------


// Also validates DedispersionConfig::dtype.
int DedispersionConfig::get_nelts_per_segment() const
{
    if (dtype == ksgpu::Dtype::native<float>())
        return xdiv(constants::bytes_per_gpu_cache_line, 4);
    else if (dtype == ksgpu::Dtype::native<__half>())
        return xdiv(constants::bytes_per_gpu_cache_line, 2);

    throw runtime_error("DedispersionConfig: invalid dtype: " + dtype.str());
}


long DedispersionConfig::get_total_nfreq() const
{
    long ret = 0;
    for (long n: zone_nfreq)
        ret += n;
    return ret;
}


float DedispersionConfig::frequency_to_index(float f) const
{
    // Allow small roundoff error at band edges.
    float fmin = zone_freq_edges.front();
    float fmax = zone_freq_edges.back();
    float eps = 1.0e-5f * (fmax - fmin);
    
    if ((f < fmin - eps) || (f > fmax + eps)) {
        stringstream ss;
        ss << "DedispersionConfig::frequency_to_index(): frequency " << f
           << " is out of range [" << fmin << ", " << fmax << "]";
        throw runtime_error(ss.str());
    }
    
    // Clamp to band edges (in case of small roundoff error).
    f = std::max(f, fmin);
    f = std::min(f, fmax);
    
    // Linear search through zones.
    float channel_offset = 0;
    for (size_t i = 0; i < zone_nfreq.size(); i++) {
        float f0 = zone_freq_edges[i];
        float f1 = zone_freq_edges[i+1];
        
        if (f <= f1) {
            // Frequency is in zone i.
            float frac = (f - f0) / (f1 - f0);
            channel_offset += frac * zone_nfreq[i];
            break;
        }
        
        channel_offset += zone_nfreq[i];
    }
    
    float tot_nfreq = this->get_total_nfreq();

    // Clamp channel_offset to [0, tot_nfreq]. (Mostly redundant with
    // previous clamping logic, but roundoff error may spill slightly
    // outside the range.)
    
    channel_offset = std::max(channel_offset, 0.0f);
    channel_offset = std::min(channel_offset, tot_nfreq);
    return channel_offset;
}


float DedispersionConfig::index_to_frequency(float index) const
{
    float tot_nfreq = this->get_total_nfreq();
    float eps = 1.0e-5f * tot_nfreq;
    
    if ((index < -eps) || (index > tot_nfreq + eps)) {
        stringstream ss;
        ss << "DedispersionConfig::index_to_frequency(): index " << index
           << " is out of range [0, " << tot_nfreq << "]";
        throw runtime_error(ss.str());
    }
    
    // Clamp to valid range (in case of small roundoff error).
    index = std::max(index, 0.0f);
    index = std::min(index, tot_nfreq);
    
    // Linear search through zones.
    float channel_offset = 0;
    for (size_t i = 0; i < zone_nfreq.size(); i++) {
        float next_offset = channel_offset + zone_nfreq[i];
        
        if (index <= next_offset) {
            // Index is in zone i.
            float frac = (index - channel_offset) / zone_nfreq[i];
            float f0 = zone_freq_edges[i];
            float f1 = zone_freq_edges[i+1];
            return f0 + frac * (f1 - f0);
        }
        
        channel_offset = next_offset;
    }
    
    // Should only reach here if index == tot_nfreq (after clamping).
    return zone_freq_edges.back();
}


float DedispersionConfig::delay_to_frequency(float delay) const
{
    // Delay is defined so that d=0 corresponds to f=fhi, and d=ntree corresponds to f=flo.
    // Formula: f = 1 / sqrt(d/scale + 1/fhi^2), where scale = ntree / (1/flo^2 - 1/fhi^2).
    
    float flo = zone_freq_edges.front();
    float fhi = zone_freq_edges.back();
    float ntree = pow2(tree_rank);
    float eps = 1.0e-5f * ntree;
    
    if ((delay < -eps) || (delay > ntree + eps)) {
        stringstream ss;
        ss << "DedispersionConfig::delay_to_frequency(): delay " << delay
           << " is out of range [0, " << ntree << "]";
        throw runtime_error(ss.str());
    }
    
    // Clamp to valid range (in case of small roundoff error).
    delay = std::max(delay, 0.0f);
    delay = std::min(delay, ntree);
    
    float scale = ntree / (1.0f/(flo*flo) - 1.0f/(fhi*fhi));
    float inv_fhi_sq = 1.0f / (fhi * fhi);
    float f = 1.0f / sqrtf(delay/scale + inv_fhi_sq);
    
    return f;
}


float DedispersionConfig::frequency_to_delay(float f) const
{
    // Delay is defined so that d=0 corresponds to f=fhi, and d=ntree corresponds to f=flo.
    // Formula: d = scale * (1/f^2 - 1/fhi^2), where scale = ntree / (1/flo^2 - 1/fhi^2).
    
    float flo = zone_freq_edges.front();
    float fhi = zone_freq_edges.back();
    float ntree = pow2(tree_rank);
    float eps = 1.0e-5f * (fhi - flo);
    
    if ((f < flo - eps) || (f > fhi + eps)) {
        stringstream ss;
        ss << "DedispersionConfig::frequency_to_delay(): frequency " << f
           << " is out of range [" << flo << ", " << fhi << "]";
        throw runtime_error(ss.str());
    }
    
    // Clamp to valid range (in case of small roundoff error).
    f = std::max(f, flo);
    f = std::min(f, fhi);
    
    // Delays before rescaling.
    float d = 1.0f / (f*f);
    float dlo = 1.0f / (flo*flo);
    float dhi = 1.0f / (fhi*fhi);

    // Return rescaled delay.
    // FIXME ntree or (ntree-1) here?
    return ntree * (d-dhi) / (dlo-dhi);
}

double DedispersionConfig::dm_per_unit_delay() const
{
    // Returns the DM (in pc cm^{-3}) whose dispersion delay across the full band
    // equals one time sample.
    
    xassert(zone_freq_edges.size() >= 2);

    double f_lo = zone_freq_edges.front();  // MHz
    double f_hi = zone_freq_edges.back();   // MHz

    xassert(f_lo > 0.0);
    xassert(f_lo < f_hi);
    
    double inv_f_lo_sq = 1.0 / (f_lo * f_lo);
    double inv_f_hi_sq = 1.0 / (f_hi * f_hi);

    // Delay = K_DM * DM * (f_lo^{-2} - f_hi^{-2})
    // where K_DM = 4.148808 ms MHz^2 (with DM in pc cm^{-3})
    double K_DM = 4.148808e6;
    return time_sample_ms / (K_DM * (inv_f_lo_sq - inv_f_hi_sq));
}


Array<double> DedispersionConfig::make_channel_map() const
{
    long nchan = pow2(tree_rank);
    Array<double> channel_map({nchan+1}, af_rhost);
    
    for (long n = 0; n <= nchan; n++) {
        double f = this->delay_to_frequency(n);
        channel_map.data[n] = this->frequency_to_index(f);
    }
    
    return channel_map;
}


void DedispersionConfig::test() const
{
    this->validate();
    
    float flo = zone_freq_edges.front();
    float fhi = zone_freq_edges.back();
    float tot_nfreq = this->get_total_nfreq();
    float ntree = pow2(tree_rank);
    
    // Test frequency_to_index / index_to_frequency at all zone boundaries.
    // E.g. if f=zone_freq_edges[i], then index = sum_{j<i} zone_nfreq[j].
    float expected_index = 0;
    for (size_t i = 0; i < zone_freq_edges.size(); i++) {
        float f = zone_freq_edges[i];
        float index = frequency_to_index(f);
        xassert(fabsf(index - expected_index) < 1.0e-4f * tot_nfreq);
        
        float f2 = index_to_frequency(expected_index);
        xassert(fabsf(f - f2) < 1.0e-4f * fhi);
        
        if (i < zone_nfreq.size())
            expected_index += zone_nfreq[i];
    }
    
    // Test delay_to_frequency / frequency_to_delay at endpoints.
    xassert(fabsf(delay_to_frequency(0.0f) - fhi) < 1.0e-4f * fhi);
    xassert(fabsf(delay_to_frequency(ntree) - flo) < 1.0e-4f * fhi);
    xassert(fabsf(frequency_to_delay(fhi)) < 1.0e-4f * ntree);
    xassert(fabsf(frequency_to_delay(flo) - ntree) < 1.0e-4f * ntree);
    
    // Test that frequency_to_index and index_to_frequency are inverses.
    for (int i = 0; i < 10; i++) {
        float index = rand_uniform(0, tot_nfreq);
        float f = index_to_frequency(index);
        float index2 = frequency_to_index(f);
        xassert(fabsf(index - index2) < 1.0e-4f * tot_nfreq);
        
        f = rand_uniform(flo, fhi);
        index = frequency_to_index(f);
        float f2 = index_to_frequency(index);
        xassert(fabsf(f - f2) < 1.0e-4f * fhi);
    }
    
    // Test that delay_to_frequency and frequency_to_delay are inverses.
    for (int i = 0; i < 10; i++) {
        float delay = rand_uniform(0, ntree);
        float f = delay_to_frequency(delay);
        float delay2 = frequency_to_delay(f);
        xassert(fabsf(delay - delay2) < 1.0e-4f * ntree);
        
        f = rand_uniform(flo, fhi);
        delay = frequency_to_delay(f);
        float f2 = delay_to_frequency(delay);
        xassert(fabsf(f - f2) < 1.0e-4f * fhi);
    }
}


void DedispersionConfig::add_early_trigger(long ds_level, long delta_rank)
{
    EarlyTrigger e;
    e.ds_level = ds_level;
    e.delta_rank = delta_rank;
    this->early_triggers.push_back(e);
    
    // Incredibly lazy -- add and re-sort
    std::sort(early_triggers.begin(), early_triggers.end());
}


void DedispersionConfig::add_early_triggers(long ds_level, std::initializer_list<long> delta_ranks)
{
    for (long delta_rank: delta_ranks) {
        EarlyTrigger e;
        e.ds_level = ds_level;
        e.delta_rank = delta_rank;
        this->early_triggers.push_back(e);
    }
    
    // Incredibly lazy -- add and re-sort
    std::sort(early_triggers.begin(), early_triggers.end());
}

                        
void DedispersionConfig::validate() const
{
    // Check that all members have been initialized.
    xassert(tree_rank > 0);
    xassert(num_downsampling_levels > 0);
    xassert(time_samples_per_chunk > 0);
    xassert(time_sample_ms > 0);
    xassert(beams_per_gpu > 0);
    xassert(beams_per_batch > 0);
    xassert(num_active_batches > 0);

    xassert_le(tree_rank, constants::max_tree_rank);
    xassert_le(num_downsampling_levels, constants::max_downsampling_level);

    // Validate zone_nfreq and zone_freq_edges.
    xassert(zone_nfreq.size() > 0);
    xassert(zone_freq_edges.size() == zone_nfreq.size() + 1);
    
    for (size_t i = 0; i < zone_nfreq.size(); i++)
        xassert(zone_nfreq[i] > 0);
    
    for (size_t i = 0; i+1 < zone_freq_edges.size(); i++) {
        xassert(zone_freq_edges[i] > 0.0f);
        xassert(zone_freq_edges[i] < zone_freq_edges[i+1]);
    }

    // Note: calling get_nelts_per_segment() checks 'dtype' for validity.
    int nelts_per_segment = this->get_nelts_per_segment();
    int min_nt = nelts_per_segment * pow2(num_downsampling_levels-1);
    
    if (time_samples_per_chunk % min_nt) {
        stringstream ss;
        ss << "DedispersionConfig: time_samples_per_chunk=" << time_samples_per_chunk
           << " must be a multiple of " << min_nt
           << " (this value depends on dtype and num_downsampling levels)";
        throw runtime_error(ss.str());
    }

    // GPU configuration.
    xassert_divisible(beams_per_gpu, beams_per_batch);  // assumed for convenience
    xassert_le(num_active_batches * beams_per_batch, beams_per_gpu);
    xassert_ge(gpu_clag_maxfrac, 0.0);
    xassert_le(gpu_clag_maxfrac, 1.0);

    for (const EarlyTrigger &et: early_triggers) {
        long ds_rank = et.ds_level ? (tree_rank-1) : (tree_rank);
        long ds_stage1_rank = ds_rank / 2;
        
        xassert((et.ds_level >= 0) && (et.ds_level < num_downsampling_levels));
        xassert((et.delta_rank > 0) && (et.delta_rank < ds_stage1_rank));
    }

    // Check validity of early triggers.
    if (!is_sorted(early_triggers))
        throw runtime_error("DedispersionConfig: early triggers must be sorted first by"
                            " increasing ds_level, then second by decreasing delta_rank");
        
    // Validate frequency_subband_counts.
    // FIXME add check that pf_rank is not too large for tree_index=0.
    // (Not sure yet what the exact constraint will be, after dust settles on all code.)
    FrequencySubbands::validate_subband_counts(frequency_subband_counts);

    // Validate peak_finding_params.
    xassert(long(peak_finding_params.size()) == num_downsampling_levels);
    
    for (long ds_level = 0; ds_level < num_downsampling_levels; ds_level++) {
        const PeakFindingConfig &pfp = peak_finding_params.at(ds_level);

        xassert(pfp.max_width > 0);
        xassert(pfp.wt_dm_downsampling > 0);
        xassert(pfp.wt_time_downsampling > 0);

        xassert(is_power_of_two(pfp.max_width));
        xassert(is_power_of_two(pfp.wt_dm_downsampling));
        xassert(is_power_of_two(pfp.wt_time_downsampling));

        long ds_rank = ds_level ? (tree_rank-1) : (tree_rank);
        long min_rank = ds_rank;
        for (const EarlyTrigger &et: early_triggers)
            if (et.ds_level == ds_level)
                min_rank = std::min(min_rank, ds_rank - et.delta_rank);
        
        if (pfp.wt_dm_downsampling > pow2(min_rank)) {
            stringstream ss;
            ss << "DedispersionConfig: wt_dm_downsampling[" << ds_level << "]=" << pfp.wt_dm_downsampling
               << " must be <= " << pow2(min_rank) << ". This upper bound is set by the max rank of"
               << " all trees at ds_level=" << ds_level << ", including early triggers.";
            throw runtime_error(ss.str());
        }

        // dm_downsampling and time_downsampling are optional (can be zero).
        // If specified, they must be powers of two and <= wt_* counterparts.

        if (pfp.dm_downsampling > 0) {
            xassert(is_power_of_two(pfp.dm_downsampling));
            xassert(pfp.wt_dm_downsampling >= pfp.dm_downsampling);
        }

        if (pfp.time_downsampling > 0) {
            xassert(is_power_of_two(pfp.time_downsampling));
            xassert(pfp.wt_time_downsampling >= pfp.time_downsampling);
        }
    }
}


void DedispersionConfig::to_yaml(YAML::Emitter &emitter, bool verbose) const
{
    this->validate();

    emitter << YAML::BeginMap;

    // ---- Frequency channels ----

    if (verbose) {
        stringstream ss;
        ss << "Frequency channels. The observed frequency band is divided into \"zones\".\n";
        ss << "Within each zone, all frequency channels have the same width, but the\n";
        ss << "channel width may differ between zones.\n";
        ss << "  zone_nfreq: number of frequency channels in each zone.\n";
        ss << "  zone_freq_edges: frequency band edges in MHz.\n";
        ss << "For example:\n";
        ss << "  zone_nfreq: [N]      zone_freq_edges: [400,800]      one zone, channel width (400/N) MHz\n";
        ss << "  zone_nfreq: [2*N,N]  zone_freq_edges: [400,600,800]  width (100/N), (200/N) MHz in lower/upper band";
        ss << "In this config, we have:\n";
        ss << "  Total frequency channels: " << get_total_nfreq() << "\n";
        ss << "  Channel widths (MHz): [ ";
        for (size_t i = 0; i < zone_nfreq.size(); i++) {
            double width = (zone_freq_edges[i+1] - zone_freq_edges[i]) / zone_nfreq[i];
            ss << (i ? ", " : "") << width;
        }
        ss << " ]";
        emitter << YAML::Comment(ss.str()) << YAML::Newline << YAML::Newline;
    }

    emitter << YAML::Key << "zone_nfreq"
            << YAML::Value << YAML::Flow << YAML::BeginSeq;
    for (long n: zone_nfreq)
        emitter << n;
    emitter << YAML::EndSeq;


    emitter << YAML::Key << "zone_freq_edges"
            << YAML::Value << YAML::Flow << YAML::BeginSeq;
    for (double f: zone_freq_edges)
        emitter << f;
    emitter << YAML::EndSeq;

    if (verbose)
        emitter << YAML::Newline << YAML::Newline << YAML::Comment("Time sample length in milliseconds.");

    emitter << YAML::Key << "time_sample_ms" << YAML::Value << time_sample_ms;

    // ---- Core dedispersion parameters ----

    if (verbose) {
        stringstream ss;
        ss << "Core dedispersion parameters.\n";
        ss << "The number of \"tree\" channels is ntree = 2^tree_rank.\n";
        ss << "The first tree (ds=0) searches delay range [0, 2^tree_rank] time samples.\n";
        ss << "Downsampled trees (ds > 0) downsample in time by 2^ds, to search beyond the diagonal DM.\n";
        ss << "In this config, the following DM ranges are searched at each downsampling level:";
        
        for (long ds = 0; ds < num_downsampling_levels; ds++) {
            long delay_lo = (ds == 0) ? 0 : pow2(tree_rank + ds - 1);
            long delay_hi = pow2(tree_rank + ds);
            double dm_lo = delay_lo * this->dm_per_unit_delay();
            double dm_hi = delay_hi * this->dm_per_unit_delay();
            double dt = time_sample_ms * pow2(ds);
            
            ss << fixed << setprecision(1)
               << "\n   ds=" << ds << ": " << dt << " ms samples, "
               << "max delay " << (1.0e-3 * delay_hi * time_sample_ms) << " seconds, "
               << "DM range [" << dm_lo << ", " << dm_hi << "] pc/cm^3";
        }
        
        emitter << YAML::Newline << YAML::Newline << YAML::Comment(ss.str()) << YAML::Newline << YAML::Newline;
    }

    emitter << YAML::Key << "tree_rank" << YAML::Value << tree_rank
            << YAML::Key << "num_downsampling_levels" << YAML::Value << num_downsampling_levels;

    if (verbose)
        emitter << YAML::Newline;

    emitter << YAML::Key << "time_samples_per_chunk" << YAML::Value << time_samples_per_chunk;

    emitter << YAML::Key << "dtype" << YAML::Value << dtype.str();
    if (verbose)
        emitter << YAML::Comment("can be either float32 or float16");

    // ---- Early triggers ----

    if (verbose) {
        emitter << YAML::Newline << YAML::Newline << YAML::Comment(
            "Early triggers: search a subset [fmid,fmax] of the full frequency range [flo,fhi]\n"
            "at reduced latency. Each downsampling level has an independent set of early triggers.\n"
            "Early triggers are optional (this can be an empty list).\n"
            "Syntax: list of {ds_level, delta_rank} pairs.\n"
            "Here, delta_rank > 0 controls 'early-ness' of the trigger."
        ) << YAML::Newline << YAML::Newline;
    }

    emitter << YAML::Key << "early_triggers"
            << YAML::Value 
            << YAML::BeginSeq;

    for (const auto &early_trigger: this->early_triggers) {
        long ds = early_trigger.ds_level;
        double dm_lo = this->dm_per_unit_delay() * ((ds == 0) ? 0 : pow2(tree_rank + ds - 1));
        double dm_hi = this->dm_per_unit_delay() * pow2(tree_rank + ds);
        double max_delay = 1.0e-3 * time_sample_ms * pow2(tree_rank + ds - early_trigger.delta_rank);
        double f = this->delay_to_frequency(pow2(tree_rank - early_trigger.delta_rank));
       
        stringstream ss;
        ss << fixed << setprecision(1)
           << "triggers at " << f << " MHz, "
           << "max delay " << max_delay << " seconds, "
           << "DM range [" << dm_lo << ", " << dm_hi << "] pc/cm^3";

        emitter
            << YAML::Flow
            << YAML::BeginMap
            << YAML::Key << "ds_level" << YAML::Value << early_trigger.ds_level
            << YAML::Key << "delta_rank" << YAML::Value << early_trigger.delta_rank
            << YAML::EndMap
            << YAML::Comment(ss.str());
    }
    
    emitter << YAML::EndSeq;

    // ---- Frequency subbands ----

    if (verbose) {
        double flo = zone_freq_edges.front();
        double fhi = zone_freq_edges.back();

        FrequencySubbands fs(frequency_subband_counts, flo, fhi);
        stringstream ss;

        ss << "Frequency subbands: can improve SNR for bursts that don't span the full frequency range.\n"
           << "This is a length-(pf_rank+1) vector containing the number of frequency subbands at each level.\n"
           << "To disable subbands and only search the full frequency band, set to [1].\n"
           << "For a tool for creating frequency_subband_counts, see 'python -m pirate_frb show_subbands --help'.\n"
           << "Note: these are the 'top-level' frequency subbands; fewer subbands may be searched in individual trees.\n"
           << "In this config, there are " << fs.F << " top-level frequency subband(s):";

        fs.show_compact(ss);

        emitter << YAML::Newline << YAML::Newline << YAML::Comment(ss.str()) << YAML::Newline << YAML::Newline;
    }

    emitter << YAML::Key << "frequency_subband_counts"
            << YAML::Value << YAML::Flow << YAML::BeginSeq;
    for (long n: frequency_subband_counts)
        emitter << n;
    emitter << YAML::EndSeq;

    // ---- Peak finding params ----

    if (verbose) {
        emitter << YAML::Newline << YAML::Newline << YAML::Comment(
            "Peak finding params: one entry per downsampling level.\n"
            "All values must be powers of two.\n"
            "  max_width: max width of peak-finding kernel, in \"tree\" time samples (required)\n"
            "  dm_downsampling: downsampling factor of coarse-grained array, relative to tree (optional, default=2^ceil(tree_rank/4))\n"
            "  time_downsampling: downsampling factor of coarse-grained array (optional, default=dm_downsampling)\n"
            "  wt_dm_downsampling: downsampling factor of weights array (required, must be >= dm_downsampling)\n"
            "  wt_time_downsampling: downsampling factor of weights array (required, must be >= time_downsampling)"
        ) << YAML::Newline << YAML::Newline;
    }

    emitter << YAML::Key << "peak_finding_params"
            << YAML::Value
            << YAML::BeginSeq;
    
    for (const auto &pfp: this->peak_finding_params) {
        emitter
            << YAML::Flow
            << YAML::BeginMap
            << YAML::Key << "max_width" << YAML::Value << pfp.max_width
            << YAML::Key << "dm_downsampling" << YAML::Value << pfp.dm_downsampling
            << YAML::Key << "time_downsampling" << YAML::Value << pfp.time_downsampling
            << YAML::Key << "wt_dm_downsampling" << YAML::Value << pfp.wt_dm_downsampling
            << YAML::Key << "wt_time_downsampling" << YAML::Value << pfp.wt_time_downsampling
            << YAML::EndMap;
    }

    emitter << YAML::EndSeq;

    // ---- GPU configuration ----

    if (verbose)
        emitter << YAML::Newline << YAML::Newline << YAML::Comment("GPU configuration.") << YAML::Newline;

    emitter << YAML::Key << "beams_per_gpu" << YAML::Value << beams_per_gpu;

    if (verbose)
        emitter << YAML::Newline;

    emitter << YAML::Key << "beams_per_batch" << YAML::Value << beams_per_batch;

    if (verbose)
        emitter << YAML::Newline;

    emitter << YAML::Key << "num_active_batches" << YAML::Value << num_active_batches
            << YAML::EndMap;
}


string DedispersionConfig::to_yaml_string(bool verbose) const
{
    YAML::Emitter emitter;
    this->to_yaml(emitter, verbose);
    return emitter.c_str();
}


void DedispersionConfig::to_yaml(const std::string &filename, bool verbose) const
{
    YAML::Emitter emitter;
    this->to_yaml(emitter, verbose);
    const char *s = emitter.c_str();

    File f(filename, O_WRONLY | O_CREAT | O_TRUNC);
    f.write(s, strlen(s));
}


// -------------------------------------------------------------------------------------------------


// static member function
DedispersionConfig DedispersionConfig::from_yaml(const string &filename)
{
    YamlFile f(filename);
    return DedispersionConfig::from_yaml(f);
}


// static member function
DedispersionConfig DedispersionConfig::from_yaml(const YamlFile &f)
{
    DedispersionConfig ret;

    ret.zone_nfreq = f.get_vector<long> ("zone_nfreq");
    ret.zone_freq_edges = f.get_vector<double> ("zone_freq_edges");
    ret.time_sample_ms = f.get_scalar<float> ("time_sample_ms");
    ret.tree_rank = f.get_scalar<long> ("tree_rank");
    ret.num_downsampling_levels = f.get_scalar<long> ("num_downsampling_levels");
    ret.time_samples_per_chunk = f.get_scalar<long> ("time_samples_per_chunk");
    ret.dtype = Dtype::from_str(f.get_scalar<string> ("dtype"));
    ret.beams_per_gpu = f.get_scalar<long> ("beams_per_gpu");
    ret.beams_per_batch = f.get_scalar<long> ("beams_per_batch");
    ret.num_active_batches = f.get_scalar<long> ("num_active_batches");

    YamlFile ets = f["early_triggers"];

    for (long i = 0; i < ets.size(); i++) {
        YamlFile et = ets[i];
        long ds_level = et.get_scalar<long> ("ds_level");
        long delta_rank = et.get_scalar<long> ("delta_rank");
        ret.add_early_trigger(ds_level, delta_rank);
        et.check_for_invalid_keys();
    }

    ret.frequency_subband_counts = f.get_vector<long> ("frequency_subband_counts");

    YamlFile pfps = f["peak_finding_params"];

    for (long i = 0; i < pfps.size(); i++) {
        YamlFile p = pfps[i];
        PeakFindingConfig pfp;
        pfp.max_width = p.get_scalar<long> ("max_width");
        pfp.dm_downsampling = p.get_scalar<long> ("dm_downsampling", 0L);
        pfp.time_downsampling = p.get_scalar<long> ("time_downsampling", 0L);
        pfp.wt_dm_downsampling = p.get_scalar<long> ("wt_dm_downsampling");
        pfp.wt_time_downsampling = p.get_scalar<long> ("wt_time_downsampling");
        ret.peak_finding_params.push_back(pfp);
        p.check_for_invalid_keys();
    }
    
    f.check_for_invalid_keys();
    
    ret.validate();
    return ret;
}

// Helper for DedispersionConfig::emit_cpp()
ostream &operator<<(ostream &os, const DedispersionConfig::EarlyTrigger &et)
{
    os << "{" << et.ds_level << "," << et.delta_rank << "}";
    return os;
}

// Helper for DedispersionConfig::emit_cpp()
ostream &operator<<(ostream &os, const DedispersionConfig::PeakFindingConfig &pfc)
{
    os << "{"
       << pfc.max_width << ","
       << pfc.dm_downsampling << ","
       << pfc.time_downsampling << ","
       << pfc.wt_dm_downsampling << ","
       << pfc.wt_time_downsampling 
       << "}";

    return os;
}

// Emit C++ code to initialize this DedispersionConfig.
// (Sometimes convenient in unit tests.)
void DedispersionConfig::emit_cpp(ostream &os, const char *name, int indent) const
{
    stringstream ss;
    for (int i = 0; i < indent; i++)
        ss << " ";
    ss << name << ".";
    string s = ss.str();

    os << s << "zone_nfreq = " << ksgpu::brace_str(zone_nfreq) << ";\n"
       << s << "zone_freq_edges = " << ksgpu::brace_str(zone_freq_edges) << ";\n"
       << s << "time_sample_ms = " << time_sample_ms << ";\n"
       << s << "tree_rank = " << tree_rank << ";\n"
       << s << "num_downsampling_levels = " << num_downsampling_levels << ";\n"
       << s << "time_samples_per_chunk = " << time_samples_per_chunk << ";\n"
       << s << "dtype = Dtype::from_str(" << dtype.str() << ");\n"
       << s << "frequency_subband_counts = " << ksgpu::brace_str(frequency_subband_counts) << ";\n"
       << s << "peak_finding_params = " << ksgpu::brace_str(peak_finding_params) << ";\n"
       << s << "early_triggers = " << ksgpu::brace_str(early_triggers) << ";\n"
       << s << "beams_per_gpu = " << beams_per_gpu << ";\n"
       << s << "beams_per_batch = " << beams_per_batch << ";\n"
       << s << "num_active_batches = " << num_active_batches << ";\n";

    if (gpu_clag_maxfrac < 1.0)
        os << s << "gpu_clag_maxfrac = " << gpu_clag_maxfrac << ";\n";
}


// Helper function for DedispersionConfig::make_random().
static CoalescedDdKernel2::RegistryKey _make_random_cdd2_key(Dtype dtype, long dd_rank)
{
    CoalescedDdKernel2::RegistryKey ret;

    ret.dtype = dtype;
    ret.dd_rank = dd_rank;
    ret.Wmax = pow2(rand_int(0,6));

    long i = rand_int(0,6);
    long j = rand_int(0,6-i);
    ret.Tinner = pow2(j);
    ret.Dout = pow2(i) * xdiv(32,dtype.nbits);

    long pf_rank = (dd_rank+1) / 2;
    ret.subband_counts = FrequencySubbands::make_random_subband_counts(pf_rank);

    return ret;
}

// static member function
DedispersionConfig DedispersionConfig::make_random(const RandomArgs &args)
{
    xassert(args.max_rank >= 2);
    xassert(args.max_early_triggers >= 0);
    long max_stage2_rank = (args.max_rank + 1) / 2;

    using Key2 = CoalescedDdKernel2::RegistryKey;  // (dtype, dd_rank, Tinner, Dout, Wmax)
    vector<Key2> all_keys = CoalescedDdKernel2::registry().get_all_keys();
    vector<Key2> my_keys;

    // Choose my_keys[0] ("primary" key).

    if (args.gpu_valid) {
        vector<Key2> valid_keys;
        for (const Key2 &k: all_keys)
            if (k.dd_rank <= max_stage2_rank)
                valid_keys.push_back(k);
        
        if (valid_keys.size() == 0) {
            stringstream ss;
            ss << "DedispersionConfig::make_random(): no precompiled cdd2 kernel is available "
               << "(max_rank=" << args.max_rank << ", max_stage2_rank=" << max_stage2_rank << ")";
            throw runtime_error(ss.str());
        }

        long ix = rand_int(0, valid_keys.size());
        my_keys.push_back(valid_keys.at(ix));
    }
    else {
        Dtype dtype = rand_bool() ? Dtype::from_str("float32") : Dtype::from_str("float16");
        long dd_rank = rand_int(1, max_stage2_rank + 1);

        Key2 primary_key = _make_random_cdd2_key(dtype, dd_rank);
        my_keys.push_back(primary_key);
    }

    // Primary key -> (dtype, subband_counts).
    DedispersionConfig ret;
    ret.dtype = my_keys.at(0).dtype;
    ret.frequency_subband_counts = my_keys.at(0).subband_counts;

    // Tree rank.
    long min_tree_rank = max(2 * my_keys.at(0).dd_rank - 1, 2L);
    long max_tree_rank = (2 * my_keys.at(0).dd_rank);
    ret.tree_rank = rand_int(min_tree_rank, max_tree_rank+1);

    // Frequency zones.
    long nzones = rand_int(1,6);
    long zone_nfreq_min = max(pow2(ret.tree_rank)/4, 1L);
    long zone_nfreq_max = pow2(ret.tree_rank);
    ret.zone_nfreq = rand_int_vec(nzones, zone_nfreq_min, zone_nfreq_max);
    ret.zone_freq_edges = rand_uniform_vec(nzones+1, 200.0, 2000.0);
    std::sort(ret.zone_freq_edges.begin(), ret.zone_freq_edges.end());

    // Sample time.
    ret.time_sample_ms = rand_uniform(1.0, 10.0);

    // Choose ret.num_downsampling_levels and my_keys[1:].

    long ds_stage2_dd_rank = ret.tree_rank / 2;
    long ds_pf_rank = (ds_stage2_dd_rank + 1) / 2;
    vector<long> ds_subband_counts = FrequencySubbands::rerank_subband_counts(ret.frequency_subband_counts, ds_pf_rank);

    // May be overridden shortly.
    ret.num_downsampling_levels = 1;

    if (args.gpu_valid && (ds_stage2_dd_rank >= 1)) {
        // All keys with correct (dtype, dd_rank, subband_counts).
        vector<Key2> valid_keys;
        for (const Key2 &k: all_keys)
            if ((k.dtype == ret.dtype) && (k.dd_rank == ds_stage2_dd_rank) && (k.subband_counts == ds_subband_counts))
                valid_keys.push_back(k);

        if (valid_keys.size() > 0) {
            ret.num_downsampling_levels = rand_int(1,5);
            for (int ds_level = 1; ds_level < ret.num_downsampling_levels; ds_level++) {
                // For each downsampling level, we choose independent (Dout, Tinner, Wmax).
                // This is artificial, but does a good job of exercising kernels.
                long ix = rand_int(0, valid_keys.size());
                my_keys.push_back(valid_keys.at(ix));
            }
        }
    }
    else if (!args.gpu_valid && (ds_stage2_dd_rank >= 1)) {
        ret.num_downsampling_levels = rand_int(1,5);

        for (int ds_level = 1; ds_level < ret.num_downsampling_levels; ds_level++) {
            // For each downsampling level, we choose independent (Dout, Tinner, Wmax).   
            Key2 ds_key = _make_random_cdd2_key(ret.dtype, ds_stage2_dd_rank);
            ds_key.subband_counts = ds_subband_counts;  // clobber
            my_keys.push_back(ds_key);
        }
    }

    // Time_samples_per_chunk, beam configuration.

    long nt_divisor = ret.get_nelts_per_segment() * pow2(ret.num_downsampling_levels-1);
    long n = xdiv(8192, nt_divisor);
    auto v = ksgpu::random_integers_with_bounded_product(3, n);

    ret.time_samples_per_chunk = v[0] * nt_divisor;
    ret.beams_per_batch = v[1];
    ret.beams_per_gpu = v[1] * v[2];
    ret.num_active_batches = rand_int(1,v[2]+1);

    // GPU configuration.
    ret.gpu_clag_maxfrac = rand_uniform(0, 1.5);
    ret.gpu_clag_maxfrac = min(ret.gpu_clag_maxfrac, 1.0);

    // For later convenience: set nt_divisor to the largest power of 2
    // which divides time_samples_per_chunk.
    while ((xdiv(ret.time_samples_per_chunk,nt_divisor) % 2) == 0)
        nt_divisor *= 2;

    if (args.verbose) {
        for (int ds_level = 0; ds_level < ret.num_downsampling_levels; ds_level++)
            cout << "DedispersionConfig::make_random(): key[" << ds_level << "]"
                 << " = " << my_keys.at(ds_level) << endl;

        cout << "DedispersionConfig::make_random(): "
             << "time_samples_per_chunk=" << ret.time_samples_per_chunk << ", "
             << "nt_divisor=" << nt_divisor << endl;
    }

    // Loop over stage1 trees. Assign peak-finding params and early trigger candidates.

    vector<EarlyTrigger> et_candidates;

    for (long ds_level = 0; ds_level < ret.num_downsampling_levels; ds_level++) {
        const Key2 &k = my_keys.at(ds_level);

        long tot_rank = ds_level ? (ret.tree_rank-1) : ret.tree_rank;
        long stage1_dd_rank = tot_rank / 2;
        long stage2_dd_rank = tot_rank - stage1_dd_rank;

        // Min/max log2(PeakFindingConfig::wt_time_downsampling).
        long nt_ds = xdiv(nt_divisor, pow2(ds_level)); 
        long min_wtds = xdiv(1024, k.Tinner * ret.dtype.nbits);
        long min_lg2_wtds = integer_log2(min_wtds);
        long max_lg2_wtds = (k.Tinner == 1) ? integer_log2(nt_ds) : min_lg2_wtds;

        // Min/max log2(PeakFindingConfig::wt_dm_downsampling).
        // FIXME: assuming default DM downsampling (PeakFindingConfig::dm_downsampling == 0) for now.
        long min_lg2_wdds = (stage2_dd_rank + 1) / 2;  // same as pf_rank
        long max_lg2_wdds = tot_rank;

        xassert_eq(k.dd_rank, stage2_dd_rank);
        xassert_le(min_lg2_wdds, max_lg2_wdds);
        xassert_le(min_lg2_wtds, max_lg2_wtds);

        // FIXME using default dm/time downsampling factors for now.

        PeakFindingConfig pfc;
        pfc.max_width = k.Wmax;
        pfc.dm_downsampling = 0;    // see above
        pfc.time_downsampling = k.Dout;
        pfc.wt_dm_downsampling = pow2(rand_int(min_lg2_wdds, max_lg2_wdds+1));
        pfc.wt_time_downsampling = pow2(rand_int(min_lg2_wtds, max_lg2_wtds+1));
        ret.peak_finding_params.push_back(pfc);

        // FIXME min_et_rank should be (stage1_dd_rank). I'm currently using (stage1_dd_rank + 1)
        // as a kludge, since my GpuDedispersionKernel doesn't support dd_rank=0.

        int min_et_rank = stage1_dd_rank + 1;
        int max_et_rank = tot_rank - 1;

        // The early trigger tree size can't be less than the wt_dm downsampling factor.
        min_et_rank = max(min_et_rank, integer_log2(pfc.wt_dm_downsampling));

        for (long et_rank = min_et_rank; et_rank <= max_et_rank; et_rank++) {
            if (args.gpu_valid) {
                Key2 ds_key = k;
                ds_key.dd_rank = et_rank - stage1_dd_rank;

                // Mimics the logic used in the DedispersionPlan constructor, 
                // to modify the subband_counts for the stage2 tree.
                long delta_rank = tot_rank - et_rank;
                long pf_rank = (ds_key.dd_rank + 1) / 2;
                ds_key.subband_counts = FrequencySubbands::early_subband_counts(ret.frequency_subband_counts, delta_rank);
                ds_key.subband_counts = FrequencySubbands::rerank_subband_counts(ds_key.subband_counts, pf_rank);

                // If there is no kernel in the registry for this (dd_rank, subband_counts),
                // then this et_rank is not an early trigger candidate.
                if (!CoalescedDdKernel2::registry().has_key(ds_key))
                    continue;
            }

            EarlyTrigger et;
            et.ds_level = ds_level;
            et.delta_rank = tot_rank - et_rank;
            et_candidates.push_back(et);
        }
    }

    // Choose early triggers (from et_candidates).

    ksgpu::randomly_permute(et_candidates);

    int max_et = min(int(et_candidates.size()), args.max_early_triggers);
    int num_et = rand_int(0, max_et+1);

    for (int i = 0; i < num_et; i++) {
        const EarlyTrigger &et = et_candidates.at(i);
        ret.add_early_trigger(et.ds_level, et.delta_rank);
    }

    ret.validate();
    return ret;
}


// static member function
DedispersionConfig DedispersionConfig::make_mini_chord(Dtype dtype)
{
    // Parameters modelled on configs/dedispersion/chord_sb0.yml:
    //   zone_freq_edges: [300, 350, 450, 600, 800, 1500]
    //   zone_nfreq: [8192, 8192, 6144, 2048, 3584]   # total = 28160
    //   tree_rank: 16
    //   time_samples_per_chunk: 2048
    //   beams_per_batch: 2
    //   time_sample_ms: 1.0

    DedispersionConfig ret;
    ret.zone_freq_edges = { 300, 350, 450, 600, 800, 1500 };
    ret.zone_nfreq = { 8192, 8192, 6144, 2048, 3584 };
    ret.tree_rank = 16;
    ret.time_sample_ms = 1.0;
    ret.num_downsampling_levels = 4;
    ret.time_samples_per_chunk = 2048;
    ret.dtype = dtype;
    ret.beams_per_gpu = 4;
    ret.beams_per_batch = 2;
    ret.num_active_batches = 2;
    ret.frequency_subband_counts = { 0, 0, 0, 0, 1 };
    ret.peak_finding_params = {
        { 16, 0, 0, 64, 64 },
        { 16, 0, 0, 64, 64 },
        { 16, 0, 0, 64, 64 },
        { 16, 0, 0, 64, 64 }
    };

    ret.validate();
    return ret;
}


}  // namespace pirate
