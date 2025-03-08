#include "../include/pirate/LaggedDownsamplingKernel.hpp"
#include "../include/pirate/DedispersionBuffer.hpp"
#include "../include/pirate/ReferenceLagbuf.hpp"
#include "../include/pirate/constants.hpp"
#include "../include/pirate/inlines.hpp"  // pow2()
#include "../include/pirate/utils.hpp"    // reference_downsample_{freq,time}()

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
//
// FIXME in hindsight, I think a non-recursive implementation of apply() would be clearer.


ReferenceLaggedDownsamplingKernel::ReferenceLaggedDownsamplingKernel(const LaggedDownsamplingKernelParams &params_) :
    params(params_)
{
    // The reference kernel uses float32, regardless of the dtype specified in 'params'.
    params.dtype = Dtype::native<float>();
    params.validate();
    
    this->nbatches = xdiv(params.total_beams, params.beams_per_batch);
    
    int nb = params.beams_per_batch;
    int r = params.large_input_rank;
    int s = params.small_input_rank;
    int nds = params.num_downsampling_levels;
    int nt2 = xdiv(params.ntime, 2);
    
    if (nds <= 1)
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
	this->lagbufs_small.push_back(make_shared<ReferenceLagbuf> (small_lags, nt2));
	this->lagbufs_large.push_back(make_shared<ReferenceLagbuf> (large_lags, nt2));
    }
    
    if (nds == 2)
	return;
    
    LaggedDownsamplingKernelParams next_params = params;
    next_params.num_downsampling_levels = nds-1;
    next_params.ntime = nt2;
    
    this->next = make_shared<ReferenceLaggedDownsamplingKernel> (next_params);
}


void ReferenceLaggedDownsamplingKernel::apply(DedispersionBuffer &buf, long ibatch)
{
    buf.params.validate();
    xassert_eq(buf.params.nbuf, params.num_downsampling_levels);
    xassert_eq(buf.params.beams_per_batch, params.beams_per_batch);	    
    xassert(buf.is_allocated);
    xassert(buf.on_host());

    xassert((ibatch >= 0) && (ibatch < nbatches));
    
    for (long ids = 0; ids < params.num_downsampling_levels; ids++) {
	long nb = params.beams_per_batch;
	long rk = params.large_input_rank - (ids ? 1 : 0);
	long nt = xdiv(params.ntime, pow2(ids));
	xassert_shape_eq(buf.bufs.at(ids), ({ nb, pow2(rk), nt }));
    }

    if (params.num_downsampling_levels <= 1)
	return;

    // The reference kernel uses float32, regardless of the dtype specified in 'params'.
    Array<float> in = buf.bufs.at(0).template cast<float> ("ReferenceLaggedDownsamplingKernel::apply(): 'in' array");
    this->_apply(in, &buf.bufs[1], ibatch);
}

	    
void ReferenceLaggedDownsamplingKernel::_apply(const Array<float> &in, Array<void> *outp, long ibatch)
{
    long r = params.large_input_rank;
    long nb = params.beams_per_batch;
    long nds = params.num_downsampling_levels;
    long ntime = params.ntime;

    xassert(nds >= 2);
    xassert_divisible(ntime, 2);
    xassert_shape_eq(in, ({ nb, pow2(r), ntime }));

    // The reference kernel uses float32, regardless of the dtype specified in 'params'.
    Array<float> out = outp[0].template cast<float> ("ReferenceLaggedDownsamplingKernel::apply(): 'out' array");
    xassert_shape_eq(out, ({ nb, pow2(r-1), ntime/2 }));

    // Reshape input array to 2-d, since reference_downsample_time() assumes a 2-d array.
    Array<float> in_2d = in.clone();   // copy, to avoid reshape failure if strides are non-contiguous
    in_2d = in_2d.reshape({ nb * pow2(r), ntime });
    
    // Reshaped time-downsampled input array: (nb * 2^r, ntime/2)
    Array<float> in_ds({ nb * pow2(r), ntime/2 }, af_uhost | af_zero);
    reference_downsample_time(in_2d, in_ds, false);  // normalize=false, i.e. sum with no factor 0.5
    
    // Apply "small" lags (one-sample lags in even channels), before frequency downsampling.
    Array<float> in_ds2 = in_ds.clone();   // copy since we'll need 'in_ds' later.
    lagbufs_small.at(ibatch)->apply_lags(in_ds2);

    // Downsample in frequency, and apply "large" lags.
    Array<float> out_tmp({ nb * pow2(r-1), ntime/2 }, af_uhost | af_zero);
    reference_downsample_freq(in_ds2, out_tmp, false);   // normalize=false, i.e. sum with no factor 0.5
    lagbufs_large.at(ibatch)->apply_lags(out_tmp);
    
    // Reshape output_tmp array from 2-d to target shape, and copy to caller-specified array.
    out_tmp = out_tmp.reshape(outp[0].ndim, outp[0].shape);
    out.fill(out_tmp);

    if (nds == 2)
	return;
    
    // Recurse to next downsampling level.
    in_ds = in_ds.reshape({ nb, pow2(r), ntime/2 });
    next->_apply(in_ds, outp+1, ibatch);
}


}  // namespace pirate
