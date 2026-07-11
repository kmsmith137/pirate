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


double DedispersionConfig::frequency_to_index(double freq) const
{
    // Allow small roundoff error at band edges.
    double fmin = zone_freq_edges.front();
    double fmax = zone_freq_edges.back();
    double eps = 1.0e-5 * (fmax - fmin);
    
    if ((freq < fmin - eps) || (freq > fmax + eps)) {
        stringstream ss;
        ss << "DedispersionConfig::frequency_to_index(): frequency " << freq
           << " is out of range [" << fmin << ", " << fmax << "]";
        throw runtime_error(ss.str());
    }
    
    // Clamp to band edges (in case of small roundoff error).
    freq = std::max(freq, fmin);
    freq = std::min(freq, fmax);
    
    // Linear search through zones.
    double channel_offset = 0;
    for (size_t i = 0; i < zone_nfreq.size(); i++) {
        double freq0 = zone_freq_edges[i];
        double freq1 = zone_freq_edges[i+1];
        
        if (freq <= freq1) {
            // Frequency is in zone i.
            double frac = (freq - freq0) / (freq1 - freq0);
            channel_offset += frac * zone_nfreq[i];
            break;
        }
        
        channel_offset += zone_nfreq[i];
    }
    
    double tot_nfreq = this->get_total_nfreq();

    // Clamp channel_offset to [0, tot_nfreq]. (Mostly redundant with
    // previous clamping logic, but roundoff error may spill slightly
    // outside the range.)
    
    channel_offset = std::max(channel_offset, 0.0);
    channel_offset = std::min(channel_offset, tot_nfreq);
    return channel_offset;
}


double DedispersionConfig::index_to_frequency(double index) const
{
    double tot_nfreq = this->get_total_nfreq();
    double eps = 1.0e-5 * tot_nfreq;
    
    if ((index < -eps) || (index > tot_nfreq + eps)) {
        stringstream ss;
        ss << "DedispersionConfig::index_to_frequency(): index " << index
           << " is out of range [0, " << tot_nfreq << "]";
        throw runtime_error(ss.str());
    }
    
    // Clamp to valid range (in case of small roundoff error).
    index = std::max(index, 0.0);
    index = std::min(index, tot_nfreq);
    
    // Linear search through zones.
    double channel_offset = 0;
    for (size_t i = 0; i < zone_nfreq.size(); i++) {
        double next_offset = channel_offset + zone_nfreq[i];
        
        if (index <= next_offset) {
            // Index is in zone i.
            double frac = (index - channel_offset) / zone_nfreq[i];
            double freq0 = zone_freq_edges[i];
            double freq1 = zone_freq_edges[i+1];
            return freq0 + frac * (freq1 - freq0);
        }
        
        channel_offset = next_offset;
    }
    
    // Should only reach here if index == tot_nfreq (after clamping).
    return zone_freq_edges.back();
}


double DedispersionConfig::delay_to_frequency(double delay) const
{
    // Delay is defined so that d=0 corresponds to freq=freq_hi, and d=ntree corresponds to freq=freq_lo.
    // Formula: freq = 1 / sqrt(d/scale + 1/freq_hi^2), where scale = ntree / (1/freq_lo^2 - 1/freq_hi^2).
    
    double freq_lo = zone_freq_edges.front();
    double freq_hi = zone_freq_edges.back();
    double ntree = pow2(toplevel_tree_rank);
    double eps = 1.0e-5 * ntree;
    
    if ((delay < -eps) || (delay > ntree + eps)) {
        stringstream ss;
        ss << "DedispersionConfig::delay_to_frequency(): delay " << delay
           << " is out of range [0, " << ntree << "]";
        throw runtime_error(ss.str());
    }
    
    // Clamp to valid range (in case of small roundoff error).
    delay = std::max(delay, 0.0);
    delay = std::min(delay, ntree);
    
    double scale = ntree / (1.0/(freq_lo*freq_lo) - 1.0/(freq_hi*freq_hi));
    double inv_fhi_sq = 1.0 / (freq_hi * freq_hi);
    double freq = 1.0 / sqrt(delay/scale + inv_fhi_sq);
    
    return freq;
}


double DedispersionConfig::frequency_to_delay(double freq) const
{
    // Delay is defined so that d=0 corresponds to freq=freq_hi, and d=ntree corresponds to freq=freq_lo.
    // Formula: d = scale * (1/freq^2 - 1/freq_hi^2), where scale = ntree / (1/freq_lo^2 - 1/freq_hi^2).
    
    double freq_lo = zone_freq_edges.front();
    double freq_hi = zone_freq_edges.back();
    double ntree = pow2(toplevel_tree_rank);
    double eps = 1.0e-5 * (freq_hi - freq_lo);
    
    if ((freq < freq_lo - eps) || (freq > freq_hi + eps)) {
        stringstream ss;
        ss << "DedispersionConfig::frequency_to_delay(): frequency " << freq
           << " is out of range [" << freq_lo << ", " << freq_hi << "]";
        throw runtime_error(ss.str());
    }
    
    // Clamp to valid range (in case of small roundoff error).
    freq = std::max(freq, freq_lo);
    freq = std::min(freq, freq_hi);
    
    // Delays before rescaling.
    double d = 1.0/ (freq*freq);
    double dlo = 1.0 / (freq_lo*freq_lo);
    double dhi = 1.0 / (freq_hi*freq_hi);

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

    // Delay = k_dm * DM * (f_lo^{-2} - f_hi^{-2}), with DM in pc cm^{-3} and
    // frequencies in MHz (see pirate::constants::k_dm).
    return time_sample_ms / (constants::k_dm * (inv_f_lo_sq - inv_f_hi_sq));
}


double DedispersionConfig::max_dm_of_all_trees() const
{
    // See DedispersionPlan.cpp: primary tree 'p' searches up to dm_max = dm0 * 2^p,
    // with dm0 = dm_per_unit_delay() * 2^toplevel_tree_rank. dm_max is monotonic in p and does
    // not depend on early_trigger_level, so the largest DM across all trees
    // is at p = num_primary_trees() - 1.
    xassert(num_primary_trees() >= 1);
    double dm0 = dm_per_unit_delay() * double(pow2(toplevel_tree_rank));
    return dm0 * double(pow2(num_primary_trees() - 1));
}


long DedispersionConfig::max_width_of_base_tree() const
{
    // primary_trees has one entry per primary tree; index 0 is the base (p=0) tree.
    // Its max_width is in "tree" time samples, which at p=0 equals native (frame)
    // time samples. (validate() guarantees this is nonempty.)
    xassert(!primary_trees.empty());
    return primary_trees.at(0).max_width;
}


Array<double> DedispersionConfig::make_channel_map() const
{
    long nchan = pow2(toplevel_tree_rank);
    Array<double> channel_map({nchan+1}, af_rhost);
    
    for (long n = 0; n <= nchan; n++) {
        double freq = this->delay_to_frequency(n);
        channel_map.data[n] = this->frequency_to_index(freq);
    }

    return channel_map;
}


Array<double> DedispersionConfig::make_random_freq_variances(bool noisy) const
{
    long nzones = zone_nfreq.size();
    long nfreq = this->get_total_nfreq();
    Array<double> ret({nfreq}, af_rhost);

    vector<double> zone_var(nzones);   // per-zone variances (also used by the 'noisy' print)
    long ifreq = 0;
    for (long z = 0; z < nzones; z++) {
        zone_var[z] = rand_uniform(0.0, 1.0);
        for (long i = 0; i < zone_nfreq[z]; i++)
            ret.data[ifreq++] = zone_var[z];
    }
    xassert_eq(ifreq, nfreq);

    if (noisy)
        cout << "make_random_freq_variances: per-zone variances = " << tuple_str(zone_var) << endl;

    return ret;
}


void DedispersionConfig::test() const
{
    this->validate();
    
    double freq_lo = zone_freq_edges.front();
    double freq_hi = zone_freq_edges.back();
    double tot_nfreq = this->get_total_nfreq();
    double ntree = pow2(toplevel_tree_rank);
    
    // Test frequency_to_index / index_to_frequency at all zone boundaries.
    // E.g. if freq=zone_freq_edges[i], then index = sum_{j<i} zone_nfreq[j].
    double expected_index = 0;
    for (size_t i = 0; i < zone_freq_edges.size(); i++) {
        double freq = zone_freq_edges[i];
        double index = frequency_to_index(freq);
        xassert(fabs(index - expected_index) < 1.0e-4 * tot_nfreq);
        
        double freq2 = index_to_frequency(expected_index);
        xassert(fabs(freq - freq2) < 1.0e-4 * freq_hi);
        
        if (i < zone_nfreq.size())
            expected_index += zone_nfreq[i];
    }
    
    // Test delay_to_frequency / frequency_to_delay at endpoints.
    xassert(fabs(delay_to_frequency(0.0) - freq_hi) < 1.0e-4 * freq_hi);
    xassert(fabs(delay_to_frequency(ntree) - freq_lo) < 1.0e-4 * freq_hi);
    xassert(fabs(frequency_to_delay(freq_hi)) < 1.0e-4 * ntree);
    xassert(fabs(frequency_to_delay(freq_lo) - ntree) < 1.0e-4 * ntree);
    
    // Test that frequency_to_index and index_to_frequency are inverses.
    for (int i = 0; i < 10; i++) {
        double index = rand_uniform(0, tot_nfreq);
        double freq = index_to_frequency(index);
        double index2 = frequency_to_index(freq);
        xassert(fabs(index - index2) < 1.0e-4 * tot_nfreq);
        
        freq = rand_uniform(freq_lo, freq_hi);
        index = frequency_to_index(freq);
        double freq2 = index_to_frequency(index);
        xassert(fabs(freq - freq2) < 1.0e-4 * freq_hi);
    }
    
    // Test that delay_to_frequency and frequency_to_delay are inverses.
    for (int i = 0; i < 10; i++) {
        double delay = rand_uniform(0, ntree);
        double freq = delay_to_frequency(delay);
        double delay2 = frequency_to_delay(freq);
        xassert(fabs(delay - delay2) < 1.0e-4 * ntree);
        
        freq = rand_uniform(freq_lo, freq_hi);
        delay = frequency_to_delay(freq);
        double freq2 = delay_to_frequency(delay);
        xassert(fabs(freq - freq2) < 1.0e-4 * freq_hi);
    }
}


void DedispersionConfig::validate() const
{
    // Check that all members have been initialized.
    xassert(toplevel_tree_rank > 0);
    xassert(primary_trees.size() > 0);
    xassert(time_samples_per_chunk > 0);
    xassert(time_sample_ms > 0);
    xassert(beams_per_gpu > 0);
    xassert(beams_per_batch > 0);
    xassert(num_active_batches > 0);

    xassert_le(toplevel_tree_rank, constants::max_tree_rank);
    xassert_le(num_primary_trees(), constants::max_primary_trees);

    // Validate zone_nfreq and zone_freq_edges.
    xassert(zone_nfreq.size() > 0);
    xassert(zone_freq_edges.size() == zone_nfreq.size() + 1);
    
    for (size_t i = 0; i < zone_nfreq.size(); i++)
        xassert(zone_nfreq[i] > 0);
    
    for (size_t i = 0; i+1 < zone_freq_edges.size(); i++) {
        xassert(zone_freq_edges[i] > 0.0);
        xassert(zone_freq_edges[i] < zone_freq_edges[i+1]);
    }

    // Note: calling get_nelts_per_segment() checks 'dtype' for validity.
    int nelts_per_segment = this->get_nelts_per_segment();
    int min_nt = nelts_per_segment * pow2(num_primary_trees()-1);

    if (time_samples_per_chunk % min_nt) {
        stringstream ss;
        ss << "DedispersionConfig: time_samples_per_chunk=" << time_samples_per_chunk
           << " must be a multiple of " << min_nt
           << " (this value depends on dtype and the number of primary trees)";
        throw runtime_error(ss.str());
    }

    // GPU configuration.
    xassert_divisible(beams_per_gpu, beams_per_batch);  // assumed for convenience
    xassert_le(num_active_batches * beams_per_batch, beams_per_gpu);
    xassert_ge(max_gpu_clag, 0);
    xassert_ge(future_write_max_samples, 0);

    // Validate frequency_subband_counts.
    // FIXME add check that pf_rank is not too large for tree_index=0.
    // (Not sure yet what the exact constraint will be, after dust settles on all code.)
    FrequencySubbands::validate_subband_counts(frequency_subband_counts);

    // Validate primary_trees.
    for (long ipri = 0; ipri < num_primary_trees(); ipri++) {
        const PrimaryTree &pt = primary_trees.at(ipri);

        long primary_tree_rank = ipri ? (toplevel_tree_rank-1) : (toplevel_tree_rank);
        long stage1_dd_rank = primary_tree_rank / 2;

        // Primary tree ipri expands into early-trigger trees with early_trigger_level =
        // 1..num_early_triggers, of rank (primary_tree_rank - et_level). Every early-trigger
        // tree must be no smaller than the stage1 tree, i.e. num_early_triggers <=
        // primary_tree_rank - stage1_dd_rank. Note primary_tree_rank - stage1_dd_rank =
        // ceil(primary_tree_rank/2), which for odd primary_tree_rank is one MORE than
        // stage1_dd_rank -- so do not write 'num_early_triggers < stage1_dd_rank' here
        // (that rejects the largest legal value, which make_random() can emit for odd
        // primary_tree_rank).
        xassert(pt.num_early_triggers >= 0);
        xassert_le(pt.num_early_triggers, primary_tree_rank - stage1_dd_rank);

        xassert(pt.max_width > 0);
        xassert(pt.wt_dm_downsampling > 0);
        xassert(pt.wt_time_downsampling > 0);

        xassert(is_power_of_two(pt.max_width));
        xassert(is_power_of_two(pt.wt_dm_downsampling));
        xassert(is_power_of_two(pt.wt_time_downsampling));

        xassert_le(pt.max_width, constants::max_pf_width);

        // Smallest rank of any tree in this primary tree's family (the earliest
        // trigger has early_trigger_level = num_early_triggers).
        long min_rank = primary_tree_rank - pt.num_early_triggers;

        if (pt.wt_dm_downsampling > pow2(min_rank)) {
            stringstream ss;
            ss << "DedispersionConfig: wt_dm_downsampling[" << ipri << "]=" << pt.wt_dm_downsampling
               << " must be <= " << pow2(min_rank) << ". This upper bound is set by the smallest"
               << " tree at primary_tree_index=" << ipri << " (i.e. the earliest trigger).";
            throw runtime_error(ss.str());
        }

        // dm_downsampling and time_downsampling are optional (can be zero).
        // If specified, they must be powers of two and <= wt_* counterparts.

        if (pt.dm_downsampling > 0) {
            xassert(is_power_of_two(pt.dm_downsampling));
            xassert(pt.wt_dm_downsampling >= pt.dm_downsampling);
        }

        if (pt.time_downsampling > 0) {
            xassert(is_power_of_two(pt.time_downsampling));
            xassert(pt.wt_time_downsampling >= pt.time_downsampling);
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
        ss << "  zone_nfreq: [2*N,N]  zone_freq_edges: [400,600,800]  width (100/N), (200/N) MHz in lower/upper band\n";
        ss << "\n";
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
    for (double freq: zone_freq_edges)
        emitter << freq;
    emitter << YAML::EndSeq;

    if (verbose)
        emitter << YAML::Newline << YAML::Newline << YAML::Comment("Time sample length in milliseconds.");

    emitter << YAML::Key << "time_sample_ms" << YAML::Value << time_sample_ms;

    // ---- Core dedispersion parameters ----

    if (verbose) {
        stringstream ss;
        ss << "Core dedispersion parameters.\n";
        ss << "The number of \"tree\" channels is ntree = 2^toplevel_tree_rank.\n";
        ss << "The first primary tree (p=0) searches delay range [0, 2^toplevel_tree_rank] time samples.\n";
        ss << "Downsampled primary trees (p > 0) downsample in time by 2^p, to search beyond the diagonal DM.\n";
        ss << "In this config, the following DM ranges are searched by each primary tree:";

        for (long p = 0; p < num_primary_trees(); p++) {
            long delay_lo = (p == 0) ? 0 : pow2(toplevel_tree_rank + p - 1);
            long delay_hi = pow2(toplevel_tree_rank + p);
            double dm_lo = delay_lo * this->dm_per_unit_delay();
            double dm_hi = delay_hi * this->dm_per_unit_delay();
            double dt = time_sample_ms * pow2(p);

            ss << fixed << setprecision(1)
               << "\n   p=" << p << ": " << dt << " ms samples, "
               << "max delay " << (1.0e-3 * delay_hi * time_sample_ms) << " seconds, "
               << "DM range [" << dm_lo << ", " << dm_hi << "] pc/cm^3";
        }

        emitter << YAML::Newline << YAML::Newline << YAML::Comment(ss.str()) << YAML::Newline << YAML::Newline;
    }

    emitter << YAML::Key << "toplevel_tree_rank" << YAML::Value << toplevel_tree_rank;

    if (verbose)
        emitter << YAML::Newline;

    emitter << YAML::Key << "time_samples_per_chunk" << YAML::Value << time_samples_per_chunk;

    emitter << YAML::Key << "dtype" << YAML::Value << dtype.str();
    if (verbose)
        emitter << YAML::Comment("can be either float32 or float16");

    // ---- Frequency subbands ----

    if (verbose) {
        double freq_lo = zone_freq_edges.front();
        double freq_hi = zone_freq_edges.back();

        FrequencySubbands fs(frequency_subband_counts, freq_lo, freq_hi);
        stringstream ss;

        ss << "Frequency subbands: can improve SNR for bursts that don't span the full frequency range.\n"
           << "This is a length-(pf_rank+1) vector containing the number of frequency subbands at each level.\n"
           << "To disable subbands and only search the full frequency band, set to [1].\n"
           << "For a tool for creating frequency_subband_counts, see 'python -m pirate_frb make_subbands --help'.\n"
           << "Note: these are the 'top-level' frequency subbands; fewer subbands may be searched in individual trees.\n"
           << "In this config, there are " << fs.N << " top-level frequency subband(s):";

        fs.show_compact(ss);

        emitter << YAML::Newline << YAML::Newline << YAML::Comment(ss.str()) << YAML::Newline << YAML::Newline;
    }

    emitter << YAML::Key << "frequency_subband_counts"
            << YAML::Value << YAML::Flow << YAML::BeginSeq;
    for (long n: frequency_subband_counts)
        emitter << n;
    emitter << YAML::EndSeq;

    // ---- Primary trees ----

    if (verbose) {
        emitter << YAML::Newline << YAML::Newline << YAML::Comment(
            "Primary trees: one entry per DM range searched, ordered from low to high DM\n"
            "(see the DM ranges in the comment above). Each primary tree is expanded into\n"
            "(num_early_triggers+1) dedispersion trees: the main full-band tree, plus one\n"
            "early-trigger tree for each early_trigger_level = 1..num_early_triggers. Early triggers\n"
            "search a high-frequency subset of the band at reduced latency.\n"
            "All values must be powers of two.\n"
            "  num_early_triggers: number of early triggers (required, can be zero)\n"
            "  max_width: max width of peak-finding kernel, in \"tree\" time samples (required)\n"
            "  dm_downsampling: downsampling factor of coarse-grained array, relative to tree (optional, default=2^ceil(toplevel_tree_rank/4))\n"
            "  time_downsampling: downsampling factor of coarse-grained array (optional, default=dm_downsampling)\n"
            "  wt_dm_downsampling: downsampling factor of weights array (required, must be >= dm_downsampling)\n"
            "  wt_time_downsampling: downsampling factor of weights array (required, must be >= time_downsampling)"
        ) << YAML::Newline << YAML::Newline;
    }

    emitter << YAML::Key << "primary_trees"
            << YAML::Value
            << YAML::BeginSeq;

    for (long p = 0; p < num_primary_trees(); p++) {
        const PrimaryTree &pt = primary_trees.at(p);

        emitter
            << YAML::Flow
            << YAML::BeginMap
            << YAML::Key << "num_early_triggers" << YAML::Value << pt.num_early_triggers
            << YAML::Key << "max_width" << YAML::Value << pt.max_width
            << YAML::Key << "dm_downsampling" << YAML::Value << pt.dm_downsampling
            << YAML::Key << "time_downsampling" << YAML::Value << pt.time_downsampling
            << YAML::Key << "wt_dm_downsampling" << YAML::Value << pt.wt_dm_downsampling
            << YAML::Key << "wt_time_downsampling" << YAML::Value << pt.wt_time_downsampling
            << YAML::EndMap;

        // In verbose mode, show the early-trigger frequencies (highest early_trigger_level =
        // earliest trigger first, matching the order of trees in the DedispersionPlan).
        if (verbose && (pt.num_early_triggers > 0)) {
            stringstream ss;
            ss << fixed << setprecision(1) << "p=" << p << ": early triggers at ";
            for (long et_level = pt.num_early_triggers; et_level > 0; et_level--) {
                double freq = this->delay_to_frequency(pow2(toplevel_tree_rank - et_level));
                ss << freq << ((et_level > 1) ? ", " : " MHz");
            }
            emitter << YAML::Comment(ss.str());
        }
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

    emitter << YAML::Key << "num_active_batches" << YAML::Value << num_active_batches;

    if (verbose)
        emitter << YAML::Newline;

    emitter << YAML::Key << "future_write_max_samples" << YAML::Value << future_write_max_samples;

    if (verbose)
        emitter << YAML::Comment("max samples a WriteFiles request may extend into the future (0 = no future writes)");

    if (max_gpu_clag < 10000) {
        if (verbose)
            emitter << YAML::Newline;
        emitter << YAML::Key << "max_gpu_clag" << YAML::Value << max_gpu_clag;
        if (verbose)
            emitter << YAML::Comment("for testing: limit on-gpu ring buffers (default 10000 = disabled)");
    }

    emitter << YAML::EndMap;
}


string DedispersionConfig::to_yaml_string(bool verbose) const
{
    YAML::Emitter emitter;
    this->to_yaml(emitter, verbose);
    return emitter.c_str();
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
    // Detect old config syntax and give targeted errors.
    if (f.has_key("tree_rank")) {
        stringstream ss;
        ss << f.name << ": key 'tree_rank' is from an old config syntax -- it has been"
           << " renamed to 'toplevel_tree_rank' (same meaning).";
        throw runtime_error(ss.str());
    }

    // (These keys were replaced by 'primary_trees' -- one entry per DM range, with
    // early triggers folded in as 'num_early_triggers'.)
    for (const char *k: { "peak_finding_params", "early_triggers", "num_downsampling_levels" }) {
        if (f.has_key(k)) {
            stringstream ss;
            ss << f.name << ": key '" << k << "' is from an old config syntax. The keys"
               << " 'peak_finding_params', 'early_triggers', and 'num_downsampling_levels'"
               << " have been replaced by 'primary_trees' -- see configs/dedispersion/*.yml"
               << " for examples of the new syntax.";
            throw runtime_error(ss.str());
        }
    }

    DedispersionConfig ret;

    ret.zone_nfreq = f.get_vector<long> ("zone_nfreq");
    ret.zone_freq_edges = f.get_vector<double> ("zone_freq_edges");
    ret.time_sample_ms = f.get_scalar<double> ("time_sample_ms");
    ret.toplevel_tree_rank = f.get_scalar<long> ("toplevel_tree_rank");
    ret.time_samples_per_chunk = f.get_scalar<long> ("time_samples_per_chunk");
    ret.dtype = Dtype::from_str(f.get_scalar<string> ("dtype"));
    ret.beams_per_gpu = f.get_scalar<long> ("beams_per_gpu");
    ret.beams_per_batch = f.get_scalar<long> ("beams_per_batch");
    ret.num_active_batches = f.get_scalar<long> ("num_active_batches");
    ret.future_write_max_samples = f.get_scalar<long> ("future_write_max_samples");  // required (no default)
    ret.max_gpu_clag = f.get_scalar<long> ("max_gpu_clag", 10000);

    ret.frequency_subband_counts = f.get_vector<long> ("frequency_subband_counts");

    YamlFile pts = f["primary_trees"];

    for (long i = 0; i < pts.size(); i++) {
        YamlFile p = pts[i];
        PrimaryTree pt;
        pt.num_early_triggers = p.get_scalar<long> ("num_early_triggers");
        pt.max_width = p.get_scalar<long> ("max_width");
        pt.dm_downsampling = p.get_scalar<long> ("dm_downsampling", 0L);
        pt.time_downsampling = p.get_scalar<long> ("time_downsampling", 0L);
        pt.wt_dm_downsampling = p.get_scalar<long> ("wt_dm_downsampling");
        pt.wt_time_downsampling = p.get_scalar<long> ("wt_time_downsampling");
        ret.primary_trees.push_back(pt);
        p.check_for_invalid_keys();
    }

    f.check_for_invalid_keys();

    ret.validate();
    return ret;
}

// Helper for DedispersionConfig::emit_cpp()
ostream &operator<<(ostream &os, const DedispersionConfig::PrimaryTree &pt)
{
    os << "{"
       << pt.num_early_triggers << ","
       << pt.max_width << ","
       << pt.dm_downsampling << ","
       << pt.time_downsampling << ","
       << pt.wt_dm_downsampling << ","
       << pt.wt_time_downsampling
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
       << s << "toplevel_tree_rank = " << toplevel_tree_rank << ";\n"
       << s << "time_samples_per_chunk = " << time_samples_per_chunk << ";\n"
       << s << "dtype = Dtype::from_str(" << dtype.str() << ");\n"
       << s << "frequency_subband_counts = " << ksgpu::brace_str(frequency_subband_counts) << ";\n"
       << s << "primary_trees = " << ksgpu::brace_str(primary_trees) << ";\n"
       << s << "beams_per_gpu = " << beams_per_gpu << ";\n"
       << s << "beams_per_batch = " << beams_per_batch << ";\n"
       << s << "num_active_batches = " << num_active_batches << ";\n";

    if (future_write_max_samples > 0)
        os << s << "future_write_max_samples = " << future_write_max_samples << ";\n";

    if (max_gpu_clag < 10000)
        os << s << "max_gpu_clag = " << max_gpu_clag << ";\n";
}


// Helper function for DedispersionConfig::make_random().
static CoalescedDdKernel2::RegistryKey _make_random_cdd2_key(Dtype dtype, long dd_rank)
{
    CoalescedDdKernel2::RegistryKey ret;

    ret.dtype = dtype;
    ret.dd_rank = dd_rank;
    ret.Wmax = pow2(rand_int(0, integer_log2(constants::max_pf_width) + 1));

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
    xassert(args.max_toplevel_rank >= 2);
    xassert(args.max_early_triggers >= 0);
    long max_stage2_rank = (args.max_toplevel_rank + 1) / 2;

    using Key2 = CoalescedDdKernel2::RegistryKey;  // (dtype, dd_rank, Tinner, Dout, Wmax)
    vector<Key2> all_keys = CoalescedDdKernel2::registry().get_all_keys();
    vector<Key2> my_keys;

    // Choose my_keys[0] (the key for the base primary tree, ipri=0).

    if (args.gpu_valid) {
        vector<Key2> valid_keys;
        for (const Key2 &k: all_keys)
            if (k.dd_rank <= max_stage2_rank)
                valid_keys.push_back(k);

        if (valid_keys.size() == 0) {
            stringstream ss;
            ss << "DedispersionConfig::make_random(): no precompiled cdd2 kernel is available "
               << "(max_toplevel_rank=" << args.max_toplevel_rank << ", max_stage2_rank=" << max_stage2_rank
               << ")";
            throw runtime_error(ss.str());
        }

        long ix = rand_int(0, valid_keys.size());
        my_keys.push_back(valid_keys.at(ix));
    }
    else {
        Dtype dtype = rand_bool() ? Dtype::from_str("float32") : Dtype::from_str("float16");
        long dd_rank = rand_int(1, max_stage2_rank + 1);

        Key2 base_key = _make_random_cdd2_key(dtype, dd_rank);
        my_keys.push_back(base_key);
    }

    // Base key -> (dtype, subband_counts).
    DedispersionConfig ret;
    ret.dtype = my_keys.at(0).dtype;
    ret.frequency_subband_counts = my_keys.at(0).subband_counts;

    // Toplevel tree rank. (Locals named *_min/*_max to avoid confusion with
    // args.max_toplevel_rank, the caller-specified bound.)
    long toplevel_rank_min = max(2 * my_keys.at(0).dd_rank - 1, 2L);
    long toplevel_rank_max = (2 * my_keys.at(0).dd_rank);
    ret.toplevel_tree_rank = rand_int(toplevel_rank_min, toplevel_rank_max+1);

    // Frequency zones.
    long nzones = rand_int(1,6);
    long zone_nfreq_min = max(pow2(ret.toplevel_tree_rank)/4, 1L);
    long zone_nfreq_max = pow2(ret.toplevel_tree_rank);
    ret.zone_nfreq = rand_int_vec(nzones, zone_nfreq_min, zone_nfreq_max);
    ret.zone_freq_edges = rand_uniform_vec(nzones+1, 200.0, 2000.0);
    std::sort(ret.zone_freq_edges.begin(), ret.zone_freq_edges.end());

    // Sample time.
    ret.time_sample_ms = rand_uniform(1.0, 10.0);

    // Choose the number of primary trees (npri) and my_keys[1:].

    long ds_stage2_dd_rank = ret.toplevel_tree_rank / 2;
    long ds_pf_rank = (ds_stage2_dd_rank + 1) / 2;
    vector<long> ds_subband_counts = FrequencySubbands::restrict_subband_counts(ret.frequency_subband_counts, 0, ds_pf_rank);

    // May be overridden shortly.
    long npri = 1;

    if (args.gpu_valid && (ds_stage2_dd_rank >= 1)) {
        // All keys with correct (dtype, dd_rank, subband_counts).
        vector<Key2> valid_keys;
        for (const Key2 &k: all_keys)
            if ((k.dtype == ret.dtype) && (k.dd_rank == ds_stage2_dd_rank) && (k.subband_counts == ds_subband_counts))
                valid_keys.push_back(k);

        if (valid_keys.size() > 0) {
            npri = rand_int(1,5);
            for (int ipri = 1; ipri < npri; ipri++) {
                // For each downsampled primary tree, we choose independent (Dout, Tinner, Wmax).
                // This is artificial, but does a good job of exercising kernels.
                long ix = rand_int(0, valid_keys.size());
                my_keys.push_back(valid_keys.at(ix));
            }
        }
    }
    else if (!args.gpu_valid && (ds_stage2_dd_rank >= 1)) {
        npri = rand_int(1,5);

        for (int ipri = 1; ipri < npri; ipri++) {
            // For each downsampled primary tree, we choose independent (Dout, Tinner, Wmax).
            Key2 ds_key = _make_random_cdd2_key(ret.dtype, ds_stage2_dd_rank);
            ds_key.subband_counts = ds_subband_counts;  // clobber
            my_keys.push_back(ds_key);
        }
    }

    // Time_samples_per_chunk, beam configuration.

    long nt_divisor = ret.get_nelts_per_segment() * pow2(npri-1);
    long n = xdiv(8192, nt_divisor);
    auto v = ksgpu::random_integers_with_bounded_product(3, n);

    ret.time_samples_per_chunk = v[0] * nt_divisor;
    ret.beams_per_batch = v[1];
    ret.beams_per_gpu = v[1] * v[2];
    ret.num_active_batches = rand_int(1,v[2]+1);

    // GPU configuration.
    long max_delay = pow2(ret.toplevel_tree_rank + npri - 1);
    long max_clag = (max_delay / ret.time_samples_per_chunk) + 1;
    ret.max_gpu_clag = rand_int(0, max_clag+1);

    // future_write_max_samples: zero 25% of the time (future writes disabled),
    // else uniform in [0, 4*time_samples_per_chunk] (i.e. up to ~4 chunks).
    ret.future_write_max_samples = (rand_uniform(0.0, 1.0) < 0.25) ? 0
        : rand_int(0, 4 * ret.time_samples_per_chunk + 1);

    // For later convenience: set nt_divisor to the largest power of 2
    // which divides time_samples_per_chunk.
    while ((xdiv(ret.time_samples_per_chunk,nt_divisor) % 2) == 0)
        nt_divisor *= 2;

    if (args.verbose) {
        for (int ipri = 0; ipri < npri; ipri++)
            cout << "DedispersionConfig::make_random(): key[" << ipri << "]"
                 << " = " << my_keys.at(ipri) << endl;

        cout << "DedispersionConfig::make_random(): "
             << "time_samples_per_chunk=" << ret.time_samples_per_chunk << ", "
             << "nt_divisor=" << nt_divisor << endl;
    }

    // Loop over primary trees. Assign peak-finding params, and compute the max
    // supported num_early_triggers for each primary tree.

    vector<long> max_et(npri, 0);

    for (long ipri = 0; ipri < npri; ipri++) {
        const Key2 &k = my_keys.at(ipri);

        long primary_tree_rank = ipri ? (ret.toplevel_tree_rank-1) : ret.toplevel_tree_rank;
        long stage1_dd_rank = primary_tree_rank / 2;
        long stage2_dd_rank = primary_tree_rank - stage1_dd_rank;

        // Min/max log2(PrimaryTree::wt_time_downsampling).
        long nt_ds = xdiv(nt_divisor, pow2(ipri));
        long min_wtds = xdiv(1024, k.Tinner * ret.dtype.nbits);
        long min_lg2_wtds = integer_log2(min_wtds);
        long max_lg2_wtds = (k.Tinner == 1) ? integer_log2(nt_ds) : min_lg2_wtds;

        // Min/max log2(PrimaryTree::wt_dm_downsampling).
        // FIXME: assuming default DM downsampling (PrimaryTree::dm_downsampling == 0) for now.
        long min_lg2_wdds = (stage2_dd_rank + 1) / 2;  // same as pf_rank
        long max_lg2_wdds = primary_tree_rank;

        xassert_eq(k.dd_rank, stage2_dd_rank);
        xassert_le(min_lg2_wdds, max_lg2_wdds);
        xassert_le(min_lg2_wtds, max_lg2_wtds);

        // FIXME using default dm/time downsampling factors for now.

        PrimaryTree pt;
        pt.num_early_triggers = 0;   // assigned below
        pt.max_width = k.Wmax;
        pt.dm_downsampling = 0;    // see above
        pt.time_downsampling = k.Dout;
        pt.wt_dm_downsampling = pow2(rand_int(min_lg2_wdds, max_lg2_wdds+1));
        pt.wt_time_downsampling = pow2(rand_int(min_lg2_wtds, max_lg2_wtds+1));
        ret.primary_trees.push_back(pt);

        // FIXME min_et_rank should be (stage1_dd_rank). I'm currently using (stage1_dd_rank + 1)
        // as a kludge, since my GpuDedispersionKernel doesn't support dd_rank=0.

        long min_et_rank = stage1_dd_rank + 1;

        // The early trigger tree size can't be less than the wt_dm downsampling factor.
        min_et_rank = max(min_et_rank, (long)integer_log2(pt.wt_dm_downsampling));

        // Early triggers are consecutive (et_level = 1..num_early_triggers), so walk
        // et_level upward (tree rank downward from primary_tree_rank-1) and stop at the
        // first unsupported value: a gap in GPU-kernel coverage caps num_early_triggers.

        for (long et_level = 1; ; et_level++) {
            if (primary_tree_rank - et_level < min_et_rank)
                break;

            if (args.gpu_valid) {
                Key2 ds_key = k;
                ds_key.dd_rank = stage2_dd_rank - et_level;

                // Mimics the logic used in the DedispersionPlan constructor,
                // to modify the subband_counts for the stage2 tree.
                long pf_rank = (ds_key.dd_rank + 1) / 2;
                ds_key.subband_counts = FrequencySubbands::restrict_subband_counts(ret.frequency_subband_counts, et_level, pf_rank);

                // If there is no kernel in the registry for this (dd_rank, subband_counts),
                // then this et_level (and all larger ones) is not supported.
                if (!CoalescedDdKernel2::registry().has_key(ds_key))
                    break;
            }

            max_et.at(ipri) = et_level;
        }
    }

    // Assign num_early_triggers to each primary tree, treating args.max_early_triggers
    // as a bound on the TOTAL early-trigger count. Process the primary trees in random
    // order, so the budget does not systematically starve large ipri.

    vector<long> ipri_order(npri);
    for (long i = 0; i < npri; i++)
        ipri_order[i] = i;
    ksgpu::randomly_permute(ipri_order);

    long et_budget = args.max_early_triggers;

    for (long ipri: ipri_order) {
        long cap = min(max_et.at(ipri), et_budget);
        long num_et = rand_int(0, cap+1);
        ret.primary_trees.at(ipri).num_early_triggers = num_et;
        et_budget -= num_et;
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
    //   toplevel_tree_rank: 16
    //   time_samples_per_chunk: 2048
    //   beams_per_batch: 2
    //   time_sample_ms: 1.0

    DedispersionConfig ret;
    ret.zone_freq_edges = { 300, 350, 450, 600, 800, 1500 };
    ret.zone_nfreq = { 8192, 8192, 6144, 2048, 3584 };
    ret.toplevel_tree_rank = 16;
    ret.time_sample_ms = 1.0;
    ret.time_samples_per_chunk = 2048;
    ret.dtype = dtype;
    ret.beams_per_gpu = 4;
    ret.beams_per_batch = 2;
    ret.num_active_batches = 2;
    ret.frequency_subband_counts = { 0, 0, 0, 0, 1 };

    // Four primary trees, no early triggers.
    // (num_early_triggers, max_width, dm_downsampling, time_downsampling, wt_dm_downsampling, wt_time_downsampling)
    ret.primary_trees = {
        { 0, 16, 0, 0, 64, 64 },
        { 0, 16, 0, 0, 64, 64 },
        { 0, 16, 0, 0, 64, 64 },
        { 0, 16, 0, 0, 64, 64 }
    };

    ret.validate();
    return ret;
}


}  // namespace pirate
