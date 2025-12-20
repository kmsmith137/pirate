#include "../include/pirate/DedispersionConfig.hpp"

#include <cstring>                 // strlen()
#include <algorithm>               // std::sort()

#include <ksgpu/Dtype.hpp>
#include <ksgpu/xassert.hpp>
#include <ksgpu/cuda_utils.hpp>    // CUDA_CALL()
#include <ksgpu/rand_utils.hpp>    // ksgpu::rand_*()
#include <ksgpu/string_utils.hpp>  // ksgpu::tuple_str()

#include "../include/pirate/constants.hpp"
#include "../include/pirate/utils.hpp"           // check_rank(), is_empty_string()
#include "../include/pirate/inlines.hpp"         // xdiv(), pow2(), print_kv(), is_power_of_two()
#include "../include/pirate/file_utils.hpp"      // File
#include "../include/pirate/YamlFile.hpp"
#include "../include/pirate/FrequencySubbands.hpp"  // FrequencySubbands::validate_subband_counts()

#include <yaml-cpp/emitter.h>

using namespace std;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


bool operator==(const DedispersionConfig::EarlyTrigger &x, const DedispersionConfig::EarlyTrigger &y)
{
    return (x.ds_level == y.ds_level) && (x.tree_rank == y.tree_rank);
}

bool operator>(const DedispersionConfig::EarlyTrigger &x, const DedispersionConfig::EarlyTrigger &y)
{
    if (x.ds_level > y.ds_level)
        return true;
    if (x.ds_level < y.ds_level)
        return false;
    
    if (x.tree_rank > y.tree_rank)
        return true;
    if (x.tree_rank < y.tree_rank)
        return false;
    
    return false;  // equal
}

bool operator<(const DedispersionConfig::EarlyTrigger &x, const DedispersionConfig::EarlyTrigger &y)
{
    return y > x;
}

ostream &operator<<(ostream &os, const DedispersionConfig::EarlyTrigger &et)
{
    os << "(ds=" << et.ds_level << ",rk=" << et.tree_rank << ")";
    return os;
};


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


float DedispersionConfig::get_frequency_index(float f) const
{
    // Allow small roundoff error at band edges.
    float fmin = zone_freq_edges.front();
    float fmax = zone_freq_edges.back();
    float eps = 1.0e-6f * (fmax - fmin);
    
    if ((f < fmin - eps) || (f > fmax + eps)) {
        stringstream ss;
        ss << "DedispersionConfig::get_frequency_index(): frequency " << f
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


void DedispersionConfig::add_early_trigger(long ds_level, long tree_rank)
{
    EarlyTrigger e;
    e.ds_level = ds_level;
    e.tree_rank = tree_rank;
    this->early_triggers.push_back(e);
    
    // Incredibly lazy -- add and re-sort
    std::sort(early_triggers.begin(), early_triggers.end());
}


void DedispersionConfig::add_early_triggers(long ds_level, std::initializer_list<long> tree_ranks)
{
    for (long tree_rank: tree_ranks) {
        EarlyTrigger e;
        e.ds_level = ds_level;
        e.tree_rank = tree_rank;
        this->early_triggers.push_back(e);
    }
    
    // Incredibly lazy -- add and re-sort
    std::sort(early_triggers.begin(), early_triggers.end());
}

                        
void DedispersionConfig::validate() const
{
    // Check that all members have been initialized.
    xassert(tree_rank >= 0);
    xassert(num_downsampling_levels > 0);
    xassert(time_samples_per_chunk > 0);
    xassert(is_sorted(early_triggers));
    xassert(beams_per_gpu > 0);
    xassert(beams_per_batch > 0);
    xassert(num_active_batches > 0);

    // Validate zone_nfreq and zone_freq_edges.
    xassert(zone_nfreq.size() > 0);
    xassert(zone_freq_edges.size() == zone_nfreq.size() + 1);
    
    for (size_t i = 0; i < zone_nfreq.size(); i++)
        xassert(zone_nfreq[i] > 0);
    
    for (size_t i = 0; i+1 < zone_freq_edges.size(); i++) {
        xassert(zone_freq_edges[i] > 0.0f);
        xassert(zone_freq_edges[i] < zone_freq_edges[i+1]);
    }

    int min_rank = (num_downsampling_levels > 1) ? 1 : 0;
    check_rank(tree_rank, "DedispersionConfig", min_rank);

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

    // Check validity of early triggers.
    for (const EarlyTrigger &et: early_triggers) {
        long ds_rank = et.ds_level ? (tree_rank-1) : (tree_rank);
        long ds_rank0 = ds_rank / 2;
        
        xassert((et.ds_level >= 0) && (et.ds_level < num_downsampling_levels));
        xassert((et.tree_rank >= ds_rank0) && (et.tree_rank < ds_rank));
    }

    // Validate frequency_subband_counts.
    FrequencySubbands::validate_subband_counts(frequency_subband_counts);

    // Validate peak_finding_params.
    xassert(long(peak_finding_params.size()) == num_downsampling_levels);
    
    for (const PeakFindingParams &pfp: peak_finding_params) {
        xassert(pfp.max_width > 0);
        xassert(is_power_of_two(pfp.max_width));
        xassert(pfp.wt_dm_downsampling > 0);
        xassert(is_power_of_two(pfp.wt_dm_downsampling));
        xassert(pfp.wt_time_downsampling > 0);
        xassert(is_power_of_two(pfp.wt_time_downsampling));
        
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


void DedispersionConfig::print(ostream &os, int indent) const
{
    print_kv("zone_nfreq", ksgpu::tuple_str(zone_nfreq), os, indent);
    print_kv("zone_freq_edges", ksgpu::tuple_str(zone_freq_edges), os, indent);
    print_kv("tree_rank", tree_rank, os, indent);
    print_kv("num_downsampling_levels", num_downsampling_levels, os, indent);
    print_kv("time_samples_per_chunk", time_samples_per_chunk, os, indent);
    print_kv("dtype", dtype.str(), os, indent);
    print_kv("early_triggers", ksgpu::tuple_str(early_triggers, " "), os, indent);
    
    print_kv("beams_per_gpu", beams_per_gpu, os, indent);
    print_kv("beams_per_batch", beams_per_batch, os, indent);
    print_kv("num_active_batches", num_active_batches, os, indent);

    if (gpu_clag_maxfrac < 1.0)
        print_kv("gpu_clag_maxfrac", gpu_clag_maxfrac, os, indent);
}


void DedispersionConfig::to_yaml(YAML::Emitter &emitter, bool verbose) const
{
    this->validate();

    emitter << YAML::BeginMap;

    // ---- Frequency channels ----

    if (verbose) {
        emitter << YAML::Newline << YAML::Comment(
            "Frequency channels. The observed frequency band is divided into \"zones\".\n"
            "Within each zone, all frequency channels have the same width, but the\n"
            "channel width may differ between zones. For example:\n"
            "  zone_nfreq: [N]      zone_freq_edges: [400,800]      one zone, channel width (400/N)\n"
            "  zone_nfreq: [2*N,N]  zone_freq_edges: [400,600,800]  width (100/N), (200/N) in lower/upper band"
        ) << YAML::Newline;
    }

    emitter << YAML::Key << "zone_nfreq"
            << YAML::Value << YAML::Flow << YAML::BeginSeq;
    for (long n: zone_nfreq)
        emitter << n;
    emitter << YAML::EndSeq;

    if (verbose)
        emitter << YAML::Newline;

    emitter << YAML::Key << "zone_freq_edges"
            << YAML::Value << YAML::Flow << YAML::BeginSeq;
    for (double f: zone_freq_edges)
        emitter << f;
    emitter << YAML::EndSeq;

    // ---- Core dedispersion parameters ----

    if (verbose) {
        emitter << YAML::Newline << YAML::Newline << YAML::Comment(
            "Core dedispersion parameters.\n"
            "The number of \"tree\" channels is ntree = 2^tree_rank.\n"
            "The first tree searches to dispersion delay given by 2^tree_rank time samples.\n"
            "Downsampled trees (0 < ds < num_downsampling_levels) downsample in time by 2^ds,\n"
            "then search delay range 2^(tree_rank+ds-1) <= delay <= 2^(tree_rank+ds)."
        ) << YAML::Newline;
    }

    emitter << YAML::Key << "tree_rank" << YAML::Value << tree_rank;

    if (verbose)
        emitter << YAML::Newline;

    emitter << YAML::Key << "num_downsampling_levels" << YAML::Value << num_downsampling_levels;

    if (verbose)
        emitter << YAML::Newline;

    emitter << YAML::Key << "time_samples_per_chunk" << YAML::Value << time_samples_per_chunk;

    if (verbose)
        emitter << YAML::Newline << YAML::Comment("dtype: can be either float32 or float16.");

    emitter << YAML::Key << "dtype" << YAML::Value << dtype.str();

    // ---- Early triggers ----

    if (verbose) {
        emitter << YAML::Newline << YAML::Newline << YAML::Comment(
            "Early triggers: search a subset [fmid,fmax] of the full frequency range [flo,fhi]\n"
            "at reduced latency. Each downsampling level has an independent set of early triggers.\n"
            "Early triggers are optional (this can be an empty list).\n"
            "Syntax: list of {ds_level, tree_rank} pairs, sorted by ds_level then tree_rank."
        ) << YAML::Newline;
    }

    emitter << YAML::Key << "early_triggers"
            << YAML::Value 
            << YAML::BeginSeq;

    for (const auto &early_trigger: this->early_triggers) {
        emitter
            << YAML::Flow
            << YAML::BeginMap
            << YAML::Key << "ds_level" << YAML::Value << early_trigger.ds_level
            << YAML::Key << "tree_rank" << YAML::Value << early_trigger.tree_rank
            << YAML::EndMap;
    }
    
    emitter << YAML::EndSeq;

    // ---- Frequency subbands ----

    if (verbose) {
        emitter << YAML::Newline << YAML::Newline << YAML::Comment(
            "Frequency subbands: can improve SNR for bursts that don't span the full frequency range.\n"
            "This is a length-(pf_rank+1) vector containing the number of frequency subbands at each level.\n"
            "Must satisfy subband_counts[pf_rank] = 1.\n"
            "To disable subbands and only search the full frequency band, set to [1].\n"
            "For more info, see 'python -m pirate_frb show_subbands --help'."
        ) << YAML::Newline;
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
        ) << YAML::Newline;
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
    ret.tree_rank = f.get_scalar<long> ("tree_rank");
    ret.num_downsampling_levels = f.get_scalar<long> ("num_downsampling_levels");
    ret.time_samples_per_chunk = f.get_scalar<long> ("time_samples_per_chunk");
    ret.dtype = ksgpu::Dtype::from_str(f.get_scalar<string> ("dtype"));
    ret.beams_per_gpu = f.get_scalar<long> ("beams_per_gpu");
    ret.beams_per_batch = f.get_scalar<long> ("beams_per_batch");
    ret.num_active_batches = f.get_scalar<long> ("num_active_batches");

    YamlFile ets = f["early_triggers"];

    for (long i = 0; i < ets.size(); i++) {
        YamlFile et = ets[i];
        long ds_level = et.get_scalar<long> ("ds_level");
        long tree_rank = et.get_scalar<long> ("tree_rank");
        ret.add_early_trigger(ds_level, tree_rank);
        et.check_for_invalid_keys();
    }

    ret.frequency_subband_counts = f.get_vector<long> ("frequency_subband_counts");

    YamlFile pfps = f["peak_finding_params"];

    for (long i = 0; i < pfps.size(); i++) {
        YamlFile p = pfps[i];
        PeakFindingParams pfp;
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


// static member function
DedispersionConfig DedispersionConfig::make_random(bool allow_early_triggers)
{
    DedispersionConfig ret;
    ret.num_downsampling_levels = ksgpu::rand_int(1, 5);
    ret.dtype = (ksgpu::rand_uniform() < 0.5) ? ksgpu::Dtype::native<float>() : ksgpu::Dtype::native<__half>();
    
    // Randomly choose a tree rank, but bias toward a high number.
    // FIXME min_rank should be (ret.num_downsampling_levels > 1) ? 1 : 0
    // I'm using a larger value as a kludge, since my GpuDedispersionKernel doesn't support dd_rank=0.
    int max_rank = 10;
    int min_rank = (ret.num_downsampling_levels > 1) ? 3 : 2;
    double x = ksgpu::rand_uniform(min_rank*min_rank, (max_rank+1)*(max_rank+1));
    ret.tree_rank = int(sqrt(x));

    // Frequency band: single zone [400,800] with zone_nfreq = pow2(tree_rank).
    // (Placeholder for more complex logic later.)
    ret.zone_nfreq = { pow2(ret.tree_rank) };
    ret.zone_freq_edges = { 400.0, 800.0 };

    // Randomly choose nt_chunk, but bias toward a low number.
    // Note: call ret.get_nelts_per_segment() after setting ret.dtype
    long min_nt_chunk = ret.get_nelts_per_segment() * pow2(ret.num_downsampling_levels-1);
    long max_nt_chunk = max(2 * pow2(ret.tree_rank), 2 * min_nt_chunk);
    long rmax = xdiv(max_nt_chunk, min_nt_chunk);  // r = nt_chunk / min_nt_chunk
    double rlog = ksgpu::rand_uniform(0, log(rmax+1));
    ret.time_samples_per_chunk = min_nt_chunk * int(exp(rlog));

    if (allow_early_triggers) {
        for (int ds_level = 0; ds_level < ret.num_downsampling_levels; ds_level++) {
            // FIXME min_et_rank should be (rank/2). I'm currently using (rank/2+1)
            // as a kludge, since my GpuDedispersionKernel doesn't support dd_rank=0.
            int rank = ds_level ? (ret.tree_rank-1) : ret.tree_rank;
            int min_et_rank = (rank/2) + 1;
            int max_et_rank = rank-1;
        
            // Use at most 4 early triggers per downsampling level (arbitrary cutoff)
            int num_candidates = max_et_rank - min_et_rank + 1;
            int max_triggers = std::min(num_candidates, 4);

            if (max_triggers <= 0)
                continue;

            // Randomly choose a trigger count, but bias toward a low number.
            double y = ksgpu::rand_uniform(-1.0, log(max_triggers+0.5));
            int num_triggers = int(exp(y));

            vector<int> et_ranks(num_candidates);
            for (int i = 0; i < num_candidates; i++)
                et_ranks[i] = min_et_rank + i;

            ksgpu::randomly_permute(et_ranks);
            et_ranks.resize(num_triggers);
            std::sort(et_ranks.begin(), et_ranks.end());

            for (int et_rank: et_ranks)
                ret.add_early_trigger(ds_level, et_rank);
        }
    }

    int nbatches = ksgpu::rand_int(1,6);
    ret.beams_per_batch = ksgpu::rand_int(1,4);
    ret.beams_per_gpu = ret.beams_per_batch * nbatches;
    ret.num_active_batches = ksgpu::rand_int(1,nbatches+1);

    ret.gpu_clag_maxfrac = ksgpu::rand_uniform(0, 1.1);
    ret.gpu_clag_maxfrac = min(ret.gpu_clag_maxfrac, 1.0);

    // Placeholder values for frequency_subband_counts and peak_finding_params.
    ret.frequency_subband_counts = { 1 };
    
    ret.peak_finding_params.resize(ret.num_downsampling_levels);
    for (int i = 0; i < ret.num_downsampling_levels; i++) {
        ret.peak_finding_params[i].max_width = 16;
        ret.peak_finding_params[i].wt_dm_downsampling = 64;
        ret.peak_finding_params[i].wt_time_downsampling = 64;
    }
        
    ret.validate();
    return ret;
}


}  // namespace pirate
