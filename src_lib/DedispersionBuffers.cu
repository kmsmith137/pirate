#include "../include/pirate/internals/DedispersionBuffers.hpp"
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


DedispersionInbufParams::DedispersionInbufParams(const shared_ptr<DedispersionPlan> &plan)
{
    xassert(plan);
    
    int nds = plan->config.num_downsampling_levels;
    xassert(nds > 0);
    
    this->dtype = plan->config.dtype;
    this->small_input_rank = (nds > 1) ? (plan->stage0_trees.at(1).rank0 + 1) : 0;
    this->large_input_rank = plan->config.tree_rank;
    this->num_downsampling_levels = nds;
    this->total_beams = plan->config.beams_per_gpu;
    this->beams_per_batch = plan->config.beams_per_batch;
    this->ntime = plan->config.time_samples_per_chunk;
}


bool DedispersionInbufParams::operator==(const DedispersionInbufParams &x) const
{
    return (dtype == x.dtype)
	&& (small_input_rank == x.small_input_rank)
	&& (large_input_rank == x.large_input_rank)
	&& (num_downsampling_levels == x.num_downsampling_levels)
	&& (total_beams == x.total_beams)
	&& (beams_per_batch == x.beams_per_batch)
	&& (ntime == x.ntime);
}


void DedispersionInbufParams::print(std::ostream &os, int indent) const
{
    print_kv("dtype", this->dtype, os, indent);
    print_kv("small_input_rank", this->small_input_rank, os, indent);
    print_kv("large_input_rank", this->large_input_rank, os, indent);
    print_kv("num_downsampling_levels", this->num_downsampling_levels, os, indent);
    print_kv("total_beams", this->total_beams, os, indent);
    print_kv("beams_per_batch", this->beams_per_batch, os, indent);
    print_kv("ntime", this->ntime, os, indent);
}


void DedispersionInbufParams::validate() const
{
    xassert(!dtype.is_empty());
    xassert(small_input_rank >= 0);
    xassert(small_input_rank <= 8);
    xassert(large_input_rank >= small_input_rank);
    xassert(large_input_rank <= constants::max_tree_rank);
    xassert(num_downsampling_levels > 0);
    xassert(num_downsampling_levels <= constants::max_downsampling_level);
    xassert(total_beams > 0);
    xassert(beams_per_batch > 0);
    xassert(ntime > 0);

    xassert_divisible(total_beams, beams_per_batch);
    xassert_divisible(ntime, pow2(num_downsampling_levels));
    
    if ((dtype != Dtype::native<float>()) && (dtype != Dtype::native<__half>()))
	throw runtime_error("DedispersionInbufParams: unsupported dtype: " + dtype.str());    
}


// -------------------------------------------------------------------------------------------------


DedispersionInbuf::DedispersionInbuf(const DedispersionInbufParams &params_)
    : params(params_)
{
    params.validate();
}


DedispersionInbuf::DedispersionInbuf(const shared_ptr<DedispersionPlan> &plan)
    : DedispersionInbuf(DedispersionInbufParams(plan))
{ }


void DedispersionInbuf::allocate(int aflags)
{
    if (is_allocated())
	throw runtime_error("double call to DedispersionInbuf::allocate()");

    long r = params.large_input_rank;
    long nb = params.beams_per_batch;
    long nds = params.num_downsampling_levels;
    
    long bstride = pow2(r) * params.ntime;
    for (long ids = 1; ids < nds; ids++)
	bstride += pow2(r-1) * xdiv(params.ntime, pow2(ids));

    this->ref = Array<void> (params.dtype, {nb,bstride}, aflags);
    this->bufs.resize(nds);

    long j = 0;
    for (long ids = 0; ids < nds; ids++) {
	long nr = ids ? pow2(r-1) : pow2(r);
	long nt_ds = xdiv(params.ntime, pow2(ids));
	Array<void> a = ref.slice(1, j, j + nr*nt_ds);
	this->bufs[ids] = a.reshape({ nb, nr, nt_ds });
	j += nr*nt_ds;
    }

    xassert(bstride == j);
}


bool DedispersionInbuf::is_allocated() const
{
    return (bufs.size() > 0);
}


bool DedispersionInbuf::on_host() const
{
    xassert(is_allocated());
    return ref.on_host();
}


bool DedispersionInbuf::on_gpu() const
{
    xassert(is_allocated());
    return ref.on_gpu();
}


}  // namespace pirate
