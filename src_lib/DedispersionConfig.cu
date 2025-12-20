#include "../include/pirate/DedispersionConfig.hpp"

#include <cstring>                 // strlen()
#include <algorithm>               // std::sort()
#include <numeric>                 // std::accumulate()

#include <ksgpu/Dtype.hpp>
#include <ksgpu/xassert.hpp>
#include <ksgpu/cuda_utils.hpp>    // CUDA_CALL()
#include <ksgpu/rand_utils.hpp>    // ksgpu::rand_*()
#include <ksgpu/string_utils.hpp>  // ksgpu::tuple_str()

#include "../include/pirate/constants.hpp"
#include "../include/pirate/utils.hpp"       // check_rank(), is_empty_string()
#include "../include/pirate/inlines.hpp"     // xdiv(), pow2(), print_kv()
#include "../include/pirate/file_utils.hpp"  // File
#include "../include/pirate/YamlFile.hpp"

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


float DedispersionConfig::get_frequency_index(float f) const
{
    // Allow small roundoff error at band edges.
    float fmin = freq_edges.front();
    float fmax = freq_edges.back();
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
    for (size_t i = 0; i < nfreq.size(); i++) {
        float f0 = freq_edges[i];
        float f1 = freq_edges[i+1];
        
        if (f <= f1) {
            // Frequency is in zone i.
            float frac = (f - f0) / (f1 - f0);
            channel_offset += frac * nfreq[i];
            break;
        }
        
        channel_offset += nfreq[i];
    }
    
    float tot_nfreq = std::accumulate(nfreq.begin(), nfreq.end(), 0.0f);

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

    // Validate nfreq and freq_edges.
    xassert(nfreq.size() > 0);
    xassert(freq_edges.size() == nfreq.size() + 1);
    
    for (size_t i = 0; i < nfreq.size(); i++)
        xassert(nfreq[i] > 0);
    
    for (size_t i = 0; i+1 < freq_edges.size(); i++) {
        xassert(freq_edges[i] > 0.0f);
        xassert(freq_edges[i] < freq_edges[i+1]);
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
}


void DedispersionConfig::print(ostream &os, int indent) const
{
    print_kv("nfreq", ksgpu::tuple_str(nfreq), os, indent);
    print_kv("freq_edges", ksgpu::tuple_str(freq_edges), os, indent);
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


void DedispersionConfig::to_yaml(YAML::Emitter &emitter) const
{
    this->validate();
    
    emitter
        << YAML::BeginMap
        << YAML::Key << "nfreq"
        << YAML::Value << YAML::Flow << YAML::BeginSeq;
    for (long n: nfreq)
        emitter << n;
    emitter
        << YAML::EndSeq
        << YAML::Key << "freq_edges"
        << YAML::Value << YAML::Flow << YAML::BeginSeq;
    for (double f: freq_edges)
        emitter << f;
    emitter
        << YAML::EndSeq
        << YAML::Key << "tree_rank" << YAML::Value << tree_rank
        << YAML::Key << "num_downsampling_levels" << YAML::Value << num_downsampling_levels
        << YAML::Key << "time_samples_per_chunk" << YAML::Value << time_samples_per_chunk
        << YAML::Key << "dtype" << YAML::Value << dtype.str()
        << YAML::Key << "early_triggers"
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
    
    emitter
        << YAML::EndSeq
        << YAML::Key << "beams_per_gpu" << YAML::Value << beams_per_gpu
        << YAML::Key << "beams_per_batch" << YAML::Value << beams_per_batch
        << YAML::Key << "num_active_batches" << YAML::Value << num_active_batches
        << YAML::EndMap;
}


string DedispersionConfig::to_yaml_string() const
{
    YAML::Emitter emitter;
    this->to_yaml(emitter);
    return emitter.c_str();
}


void DedispersionConfig::to_yaml(const std::string &filename) const
{
    YAML::Emitter emitter;
    this->to_yaml(emitter);
    const char *s = emitter.c_str();

    File f(filename, O_WRONLY | O_CREAT | O_TRUNC);
    f.write(s, strlen(s));
}


// -------------------------------------------------------------------------------------------------


// static member function
DedispersionConfig DedispersionConfig::from_yaml(const string &filename, int verbosity)
{
    YamlFile f(filename, verbosity);
    return DedispersionConfig::from_yaml(f);
}


// static member function
DedispersionConfig DedispersionConfig::from_yaml(const YamlFile &f)
{
    DedispersionConfig ret;

    ret.nfreq = f.get_vector<long> ("nfreq");
    ret.freq_edges = f.get_vector<double> ("freq_edges");
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

    // Frequency band: single zone [400,800] with nfreq = pow2(tree_rank).
    // (Placeholder for more complex logic later.)
    ret.nfreq = { pow2(ret.tree_rank) };
    ret.freq_edges = { 400.0, 800.0 };

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
        
    ret.validate();
    return ret;
}


}  // namespace pirate
