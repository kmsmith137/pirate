#include "../include/pirate/LaggedDownsamplingKernel.hpp"
#include "../include/pirate/DedispersionPlan.hpp"
#include "../include/pirate/constants.hpp"
#include "../include/pirate/inlines.hpp"  // pow2()

#include <ksgpu/xassert.hpp>

using namespace std;
using namespace ksgpu;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


void LaggedDownsamplingKernelParams::print(std::ostream &os, int indent) const
{
    print_kv("dtype", this->dtype, os, indent);
    print_kv("input_total_rank", this->input_total_rank, os, indent);
    print_kv("output_dd_rank", this->output_dd_rank, os, indent);
    print_kv("num_downsampling_levels", this->num_downsampling_levels, os, indent);
    print_kv("total_beams", this->total_beams, os, indent);
    print_kv("beams_per_batch", this->beams_per_batch, os, indent);
    print_kv("ntime", this->ntime, os, indent);
}


void LaggedDownsamplingKernelParams::validate() const
{
    xassert(!dtype.is_empty());
    xassert(output_dd_rank >= 0);
    xassert(output_dd_rank <= 7);
    xassert(input_total_rank >= output_dd_rank+1);
    xassert(input_total_rank <= constants::max_tree_rank);
    xassert(num_downsampling_levels > 0);
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
