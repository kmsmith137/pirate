#include "../include/pirate/internals/LaggedDownsamplingKernel.hpp"

#include "../include/pirate/constants.hpp"
#include "../include/pirate/internals/inlines.hpp"  // pow2()

#include <ksgpu/xassert.hpp>

using namespace std;
using namespace ksgpu;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// Helper for constructor.
static long get_min_beam_stride(const LaggedDownsamplingKernelParams &params)
{
    long ret = 0;
    
    for (long ids = 0; ids < params.num_downsampling_levels; ids++) {
	long nr = pow2(params.large_input_rank - 1);
	long nt_ds = xdiv(params.ntime, pow2(ids+1));
	ret += nr * nt_ds;
    }

    return ret;
}


LaggedDownsamplingKernelOutbuf::LaggedDownsamplingKernelOutbuf(const LaggedDownsamplingKernelParams &params_)
    : params(params_), min_beam_stride(get_min_beam_stride(params_))
{
    params.validate();
}


LaggedDownsamplingKernelOutbuf::LaggedDownsamplingKernelOutbuf(const shared_ptr<DedispersionPlan> &plan)
    : LaggedDownsamplingKernelOutbuf(LaggedDownsamplingKernelParams(plan))
{ }


void LaggedDownsamplingKernelOutbuf::allocate(long beam_stride, int aflags)
{
    xassert_ge(beam_stride, min_beam_stride);

    if (params.num_downsampling_levels == 0)
	return;
    
    if (is_allocated())
	throw runtime_error("double call to LaggedDownsamplingKernelOutbuf::allocate()");

    long nb = params.beams_per_batch;
    long nr = pow2(params.large_input_rank - 1);
    long nt_cumul = 0;
	
    this->big_arr = Array<void> (params.dtype, {nb,min_beam_stride}, {beam_stride,1}, aflags);
    this->small_arrs.resize(params.num_downsampling_levels);
	
    for (int i = 0; i < params.num_downsampling_levels; i++) {
	long nt_ds = xdiv(params.ntime, pow2(i+1));
	Array<void> a = big_arr.slice(1, nr * nt_cumul, nr * (nt_cumul + nt_ds));
	small_arrs[i] = a.reshape({ nb, nr, nt_ds });
	nt_cumul += nt_ds;
    }

    xassert(nr * nt_cumul == min_beam_stride);
    xassert(is_allocated());  // paranoid
}


void LaggedDownsamplingKernelOutbuf::allocate(int aflags)
{
    this->allocate(min_beam_stride, aflags);
}


bool LaggedDownsamplingKernelOutbuf::is_allocated() const
{
    return (params.num_downsampling_levels == 0) || (big_arr.size > 0);
}


bool LaggedDownsamplingKernelOutbuf::on_host() const
{
    return (params.num_downsampling_levels == 0) || (big_arr.on_host());
}


bool LaggedDownsamplingKernelOutbuf::on_gpu() const
{
    return (params.num_downsampling_levels == 0) || (big_arr.on_gpu());
}


}  // namespace pirate
