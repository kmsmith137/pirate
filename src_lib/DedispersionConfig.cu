#include "../include/pirate/DedispersionConfig.hpp"

#include <cstring>                   // strlen()
#include <algorithm>                 // std::sort()
#include <gputils/cuda_utils.hpp>    // CUDA_CALL()
#include <gputils/rand_utils.hpp>    // gputils::rand_*()
#include <gputils/string_utils.hpp>  // gputils::tuple_str()

#include "../include/pirate/constants.hpp"
#include "../include/pirate/internals/File.hpp"
#include "../include/pirate/internals/utils.hpp"    // check_rank(), is_empty_string()
#include "../include/pirate/internals/inlines.hpp"  // xdiv(), pow2(), print_kv()
#include "../include/pirate/internals/YamlFile.hpp"

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
    if (is_empty_string(dtype))
	throw runtime_error("DedispersionConfig::dtype is unintialized");

    if (dtype == "float32")
	return xdiv(constants::bytes_per_gpu_cache_line, 4);
    else if (dtype == "float16")
	return xdiv(constants::bytes_per_gpu_cache_line, 2);

    stringstream ss;
    ss << "DedispersionConfig: dtype='" << dtype << "' is invalid."
       << " Valid values are 'float32' and 'float16'.";
    
    throw runtime_error(ss.str());
}

void DedispersionConfig::add_early_trigger(ssize_t ds_level, ssize_t tree_rank)
{
    EarlyTrigger e;
    e.ds_level = ds_level;
    e.tree_rank = tree_rank;
    this->early_triggers.push_back(e);
    
    // Incredibly lazy -- add and re-sort
    std::sort(early_triggers.begin(), early_triggers.end());
}


void DedispersionConfig::add_early_triggers(ssize_t ds_level, std::initializer_list<ssize_t> tree_ranks)
{
    for (ssize_t tree_rank: tree_ranks) {
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
    assert(tree_rank >= 0);
    assert(num_downsampling_levels > 0);
    assert(time_samples_per_chunk > 0);
    assert(is_sorted(early_triggers));
    assert(beams_per_gpu > 0);
    assert(beams_per_batch > 0);
    assert(num_active_batches > 0);
    assert(gmem_nbytes_per_gpu > 0);

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
    assert((beams_per_gpu % beams_per_batch) == 0);
    assert((num_active_batches * beams_per_batch) <= beams_per_gpu);

    // Assumed for convenience, to simplify logic in a few places -- might revisit later.
    assert((beams_per_gpu % beams_per_batch) == 0);
    
    // Check validity of early triggers.

    int dslevel_curr = 0;
    int ntrigger_curr = 0;  // running trigger count at current downsampling level
    
    for (const EarlyTrigger &et: early_triggers) {
	ssize_t ds_rank = et.ds_level ? (tree_rank-1) : (tree_rank);
	ssize_t ds_rank0 = ds_rank / 2;
	
	assert((et.ds_level >= 0) && (et.ds_level < num_downsampling_levels));
	assert((et.tree_rank >= ds_rank0) && (et.tree_rank < ds_rank));

	if (et.ds_level != dslevel_curr) {
	    dslevel_curr = et.ds_level;
	    ntrigger_curr = 0;
	}

	ntrigger_curr++;
    }
}


void DedispersionConfig::print(ostream &os, int indent) const
{
    print_kv("tree_rank", tree_rank, os, indent);
    print_kv("num_downsampling_levels", num_downsampling_levels, os, indent);
    print_kv("time_samples_per_chunk", time_samples_per_chunk, os, indent);
    print_kv("dtype", dtype, os, indent);
    print_kv("early_triggers", gputils::tuple_str(early_triggers, " "), os, indent);
    
    print_kv("beams_per_gpu", beams_per_gpu, os, indent);
    print_kv("beams_per_batch", beams_per_batch, os, indent);
    print_kv("num_active_batches", num_active_batches, os, indent);
    print_kv_nbytes("gmem_nbytes_per_gpu", gmem_nbytes_per_gpu, os, indent);
}


void DedispersionConfig::to_yaml(YAML::Emitter &emitter) const
{
    this->validate();
    
    emitter
	<< YAML::BeginMap
	<< YAML::Key << "tree_rank" << YAML::Value << tree_rank
	<< YAML::Key << "num_downsampling_levels" << YAML::Value << num_downsampling_levels
	<< YAML::Key << "time_samples_per_chunk" << YAML::Value << time_samples_per_chunk
	<< YAML::Key << "dtype" << YAML::Value << dtype
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
	<< YAML::Key << "gmem_nbytes_per_gpu" << YAML::Value << gmem_nbytes_per_gpu
	<< YAML::Comment(gputils::nbytes_to_str(gmem_nbytes_per_gpu))
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

    ret.tree_rank = f.get_scalar<long> ("tree_rank");
    ret.num_downsampling_levels = f.get_scalar<long> ("num_downsampling_levels");
    ret.time_samples_per_chunk = f.get_scalar<long> ("time_samples_per_chunk");
    ret.dtype = f.get_scalar<string> ("dtype");
    ret.beams_per_gpu = f.get_scalar<long> ("beams_per_gpu");
    ret.beams_per_batch = f.get_scalar<long> ("beams_per_batch");
    ret.num_active_batches = f.get_scalar<long> ("num_active_batches");
    ret.gmem_nbytes_per_gpu = f.get_scalar<long> ("gmem_nbytes_per_gpu");

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
DedispersionConfig DedispersionConfig::make_random(bool reference)
{
    DedispersionConfig ret;
    ret.num_downsampling_levels = gputils::rand_int(1, 5);
    ret.dtype = (gputils::rand_uniform() < 0.5) ? "float32" : "float16";
    
    // Randomly choose a tree rank, but bias toward a high number.
    int max_rank = 10;
    int min_rank = (ret.num_downsampling_levels > 1) ? 1 : 0;
    double x = gputils::rand_uniform(min_rank*min_rank, (max_rank+1)*(max_rank+1));
    ret.tree_rank = int(sqrt(x));

    // Note: call ret.get_nelts_per_segment() after setting ret.dtype
    int max_nt_chunk = 2048;
    int min_nt_chunk = ret.get_nelts_per_segment() * pow2(ret.num_downsampling_levels-1);
    int nchunks = gputils::rand_int(1, xdiv(max_nt_chunk,min_nt_chunk)+1);
    ret.time_samples_per_chunk = min_nt_chunk * nchunks;

    // Early triggers
    
    for (int ds_level = 0; ds_level < ret.num_downsampling_levels; ds_level++) {
	int rank = ds_level ? (ret.tree_rank-1) : ret.tree_rank;;
	int min_et_rank = rank/2;
	int max_et_rank = rank-1;
	
	// Use at most 4 early triggers per downsampling level (arbitrary cutoff)
	int num_candidates = max_et_rank - min_et_rank + 1;
	int max_triggers = std::min(num_candidates, 4);

	if (max_triggers <= 0)
	    continue;

	// Randomly choose a trigger count, but bias toward a low number.
	double y = gputils::rand_uniform(-1.0, log(max_triggers+0.5));
	int num_triggers = int(exp(y));

	vector<int> et_ranks(num_candidates);
	for (int i = 0; i < num_candidates; i++)
	    et_ranks[i] = min_et_rank + i;

	gputils::randomly_permute(et_ranks);
	et_ranks.resize(num_triggers);
	std::sort(et_ranks.begin(), et_ranks.end());

	for (int et_rank: et_ranks)
	    ret.add_early_trigger(ds_level, et_rank);
    }
	
    // FIXME support these members
    ret.beams_per_gpu = 1;
    ret.beams_per_batch = 1;
    ret.num_active_batches = 1;
    ret.gmem_nbytes_per_gpu = 10L * 1000L * 1000L * 1000L;

    ret.validate();
    return ret;
}


}  // namespace pirate
