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

void LaggedDownsamplingKernelParams::emit_cpp(ostream &os, const char *name, int indent)
{
    stringstream ss;
    for (int i = 0; i < indent; i++)
        ss << " ";
    ss << name << ".";
    string s = ss.str();

    os << s << "dtype = Dtype::from_str(" << dtype.str() << ");\n"
       << s << "input_total_rank = " << input_total_rank << ";\n"
       << s << "output_dd_rank = " << output_dd_rank << ";\n"
       << s << "num_downsampling_levels = " << num_downsampling_levels << ";\n"
       << s << "total_beams = " << total_beams << ";\n"
       << s << "beams_per_batch = " << beams_per_batch << ";\n"
       << s << "ntime = " << ntime << ";\n";
}


}  // namespace pirate
