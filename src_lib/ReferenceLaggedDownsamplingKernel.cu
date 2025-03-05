#include "../include/pirate/internals/LaggedDownsamplingKernel.hpp"

#include "../include/pirate/constants.hpp"
#include "../include/pirate/internals/inlines.hpp"  // pow2()
#include "../include/pirate/internals/utils.hpp"    // reference_downsample_{freq,time}()

#include <ksgpu/xassert.hpp>

using namespace std;
using namespace ksgpu;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// FIXME there is a lot of reshaping/flattening/unflattening happening in this code,
// and the code would be clearer if this were removed. I think this needs some minor
// changes elsewhere though (e.g. currently reference_downsample_time() assumes a
// 2-d array).


ReferenceLaggedDownsamplingKernel2::ReferenceLaggedDownsamplingKernel2(const LaggedDownsamplingKernelParams &params_) :
    params(params_),
    nbatches(xdiv(params_.total_beams, params_.beams_per_batch))
{
    params.validate();
    
    if (params.dtype != Dtype::native<float>())
	throw runtime_error("ReferenceLaggedDownsamplingKernel2 is only implemented for dtype==float32");
    
    int nb = params.beams_per_batch;
    int r = params.large_input_rank;
    int s = params.small_input_rank;
    int nds = params.num_downsampling_levels;
    int nt2 = xdiv(params.ntime, 2);
    
    if (nds == 0)
	return;
    
    xassert(params.small_input_rank > 0);
    Array<int> small_lags({nb * pow2(r)}, af_uhost | af_zero);
    Array<int> large_lags({nb * pow2(r-1)}, af_uhost | af_zero);

    for (int i = 0; i < nb * pow2(r); i++)
	small_lags.data[i] = (i & 1) ? 0 : 1;
    
    for (int i = 0; i < nb * pow2(r-s); i++)
	for (int j = 0; j < pow2(s-1); j++)
	    large_lags.data[i*pow2(s-1)+j] = pow2(s-1)-j-1;

    for (int i = 0; i < nbatches; i++) {
	this->lagbufs_small.push_back({ small_lags, nt2 });
	this->lagbufs_large.push_back({ large_lags, nt2 });
    }
    
    if (nds == 1)
	return;
    
    LaggedDownsamplingKernelParams next_params = params;
    next_params.num_downsampling_levels = nds-1;
    next_params.ntime = nt2;
    
    this->next = make_shared<ReferenceLaggedDownsamplingKernel2> (next_params);
}

void ReferenceLaggedDownsamplingKernel2::apply(const Array<void> &in_, LaggedDownsamplingKernelOutbuf &out, long ibatch)
{
    xassert(in_.on_host());
    xassert_shape_eq(in_, ({ params.beams_per_batch, pow2(params.large_input_rank), params.ntime }));
    Array<float> in = in_.template cast<float> ("ReferenceLaggedDownsamplingKernel2::apply(): 'in' array");

    xassert(out.params == this->params);
    xassert(out.is_allocated());
    xassert(out.on_host());

    xassert((ibatch >= 0) && (ibatch < nbatches));
    xassert(long(out.small_arrs.size()) == params.num_downsampling_levels);  // should never fail
    
    if (params.num_downsampling_levels > 0)
	this->_apply(in, &out.small_arrs[0], ibatch);
}

	    
void ReferenceLaggedDownsamplingKernel2::_apply(const Array<float> &in, Array<void> *outp, long ibatch)
{
    // Reminder: the input/output arrays have the following shapes:
    //
    //   in.shape = (nbeams, 2^large_input_rank, ntime)
    //   out[i].shape = (nbeams, 2^(large_input_rank-1), ntime/2^(i+1))
    //   out.size() = Params::num_downsampling_levels

    long r = params.large_input_rank;
    long nb = params.beams_per_batch;
    long nds = params.num_downsampling_levels;
    long ntime = params.ntime;

    xassert(nds > 0);
    xassert_divisible(ntime, 2);
    xassert_shape_eq(in, ({ nb, pow2(r), ntime }));

    Array<float> out = outp[0].template cast<float> ("ReferenceLaggedDownsamplingKernel2::apply(): 'out' array");
    xassert_shape_eq(out, ({ nb, pow2(r-1), ntime/2 }));

    // Reshape input array to 2-d, since reference_downsample_time() assumes a 2-d array.
    Array<float> in_2d = in.clone();   // copy, to avoid reshape failure if strides are non-contiguous
    in_2d = in_2d.reshape({ nb * pow2(r), ntime });
    
    // Reshaped time-downsampled input array: (nb * 2^r, ntime/2)
    Array<float> in_ds({ nb * pow2(r), ntime/2 }, af_uhost | af_zero);
    reference_downsample_time(in_2d, in_ds, false);  // normalize=false, i.e. sum with no factor 0.5
    
    // Apply "small" lags (one-sample lags in even channels), before frequency downsampling.
    Array<float> in_ds2 = in_ds.clone();   // copy since we'll need 'in_ds' later.
    lagbufs_small.at(ibatch).apply_lags(in_ds2);

    // Downsample in frequency, and apply "large" lags.
    Array<float> out_tmp({ nb * pow2(r-1), ntime/2 }, af_uhost | af_zero);
    reference_downsample_freq(in_ds2, out_tmp, false);   // normalize=false, i.e. sum with no factor 0.5
    lagbufs_large.at(ibatch).apply_lags(out_tmp);
    
    // Reshape output_tmp array from 2-d to target shape, and copy to caller-specified array.
    out_tmp = out_tmp.reshape(outp[0].ndim, outp[0].shape);
    out.fill(out_tmp);

    if (nds == 1)
	return;
    
    // Recurse to next downsampling level.
    in_ds = in_ds.reshape({ nb, pow2(r), ntime/2 });
    next->_apply(in_ds, outp+1, ibatch);
}


}  // namespace pirate
