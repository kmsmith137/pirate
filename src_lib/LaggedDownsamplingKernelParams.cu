#include "../include/pirate/internals/LaggedDownsamplingKernel.hpp"
#include "../include/pirate/DedispersionPlan.hpp"

#include "../include/pirate/constants.hpp"
#include "../include/pirate/internals/inlines.hpp"  // pow2()

#include <ksgpu/xassert.hpp>

using namespace std;
using namespace ksgpu;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


LaggedDownsamplingKernelParams::LaggedDownsamplingKernelParams(const shared_ptr<DedispersionPlan> &plan)
{
    xassert(plan);
    
    int nds = plan->config.num_downsampling_levels;
    xassert(nds > 0);
    
    this->dtype = plan->config.dtype;
    this->small_input_rank = (nds > 1) ? (plan->stage0_trees.at(1).rank0 + 1) : 0;
    this->large_input_rank = plan->config.tree_rank;
    this->num_downsampling_levels = nds - 1;   // note (-1) here!
    this->total_beams = plan->config.beams_per_gpu;
    this->beams_per_batch = plan->config.beams_per_batch;
    this->ntime = plan->config.time_samples_per_chunk;
}


bool LaggedDownsamplingKernelParams::operator==(const LaggedDownsamplingKernelParams &x) const
{
    return (dtype == x.dtype)
	&& (small_input_rank == x.small_input_rank)
	&& (large_input_rank == x.large_input_rank)
	&& (num_downsampling_levels == x.num_downsampling_levels)
	&& (total_beams == x.total_beams)
	&& (beams_per_batch == x.beams_per_batch)
	&& (ntime == x.ntime);
}


void LaggedDownsamplingKernelParams::print(std::ostream &os, int indent) const
{
    print_kv("dtype", this->dtype, os, indent);
    print_kv("small_input_rank", this->small_input_rank, os, indent);
    print_kv("large_input_rank", this->large_input_rank, os, indent);
    print_kv("num_downsampling_levels", this->num_downsampling_levels, os, indent);
    print_kv("total_beams", this->total_beams, os, indent);
    print_kv("beams_per_batch", this->beams_per_batch, os, indent);
    print_kv("ntime", this->ntime, os, indent);
}


void LaggedDownsamplingKernelParams::validate() const
{
    xassert(!dtype.is_empty());
    xassert(small_input_rank >= 0);
    xassert(small_input_rank <= 8);
    xassert(large_input_rank >= small_input_rank);
    xassert(large_input_rank <= constants::max_tree_rank);
    xassert(num_downsampling_levels >= 0);
    xassert(num_downsampling_levels <= constants::max_downsampling_level);
    xassert(total_beams > 0);
    xassert(beams_per_batch > 0);
    xassert(ntime > 0);

    xassert_divisible(total_beams, beams_per_batch);
    xassert_divisible(ntime, pow2(num_downsampling_levels));
    
    if ((dtype != Dtype::native<float>()) && (dtype != Dtype::native<__half>()))
	throw runtime_error("LaggedDownsamplingKernelParams: unsupported dtype: " + dtype.str());    
}


}  // namespace pirate
