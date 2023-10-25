#include "../include/pirate/internals/ReferenceLaggedDownsampler.hpp"
#include "../include/pirate/internals/ReferenceLagbuf.hpp"

#include "../include/pirate/constants.hpp"
#include "../include/pirate/internals/inlines.hpp"  // pow2(), bit_reverse_slow()
#include "../include/pirate/internals/utils.hpp"    // reference_downsample_freq(), reference_downsample_time()

#include <cassert>

using namespace std;
using namespace gputils;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


ReferenceLaggedDownsampler::ReferenceLaggedDownsampler(const Params &params_)
    : params(params_)
{
    assert(params.small_input_rank >= 1);
    assert(params.small_input_rank <= 8);
    assert(params.large_input_rank >= params.small_input_rank);
    assert(params.large_input_rank <= constants::max_tree_rank);
    assert(params.num_downsampling_levels > 0);
    assert(params.num_downsampling_levels <= constants::max_downsampling_level);
    assert(params.nbeams > 0);
    assert(params.ntime > 0);
    assert((params.ntime % pow2(params.num_downsampling_levels)) == 0);

    int nb = params.nbeams;
    int r = params.large_input_rank;
    int s = params.small_input_rank;

    vector<int> small_lags(nb * pow2(r), 0);
    vector<int> large_lags(nb * pow2(r-1), 0);
    
    for (int i = 0; i < nb * pow2(r); i++)
	small_lags[i] = (i & 1) ? 0 : 1;
    
    for (int i = 0; i < nb * pow2(r-s); i++)
	for (int j = 0; j < pow2(s-1); j++)
	    large_lags[i*pow2(s-1)+j] = pow2(s-1)-j-1;
    
    this->lagbuf_small = make_shared<ReferenceLagbuf> (small_lags, params.ntime/2);
    this->lagbuf_large = make_shared<ReferenceLagbuf> (large_lags, params.ntime/2);

    if (params.num_downsampling_levels == 1)
	return;
    
    Params next_params;
    next_params.small_input_rank = params.small_input_rank;
    next_params.large_input_rank = params.large_input_rank;
    next_params.num_downsampling_levels = params.num_downsampling_levels - 1;
    next_params.ntime = xdiv(params.ntime, 2);
    next_params.nbeams = params.nbeams;
    
    this->next = make_shared<ReferenceLaggedDownsampler> (next_params);
}


void ReferenceLaggedDownsampler::apply(const Array<float> &in, vector<Array<float>> &out)
{
    assert(out.size() == params.num_downsampling_levels);
    this->apply(in, &out[0]);
}


// Helper for ReferenceLaggedDownsampler::apply()
static void _check_shape(const char *name, const Array<float> &arr, ssize_t nbeams, ssize_t nfreq, ssize_t ntime)
{
    if (arr.shape_equals({ nbeams, nfreq, ntime }))
	return;

    if ((nbeams == 1) && arr.shape_equals({nfreq,ntime}))
	return;

    stringstream ss;
    ss << "ReferenceLaggedDownsampler::apply(): expected '" << name << "' array to have"
       << " shape=(" << nbeams << "," << nfreq << "," << ntime << ")";

    if (nbeams == 1)
	ss << " or shape=(" << nfreq << "," << ntime << ")";

    ss << ", got shape=" << arr.shape_str();
    throw runtime_error(ss.str());
}


void ReferenceLaggedDownsampler::apply(const Array<float> &in, Array<float> *outp)
{
    int r = params.large_input_rank;
    int nbeams = params.nbeams;
    long ntime = params.ntime;

    _check_shape("in", in, nbeams, pow2(r), ntime);
    _check_shape("out", outp[0], nbeams, pow2(r-1), xdiv(ntime,2));

    // Reshape input array to 2-d, since reference_downsample_time() assumes a 2-d array.
    Array<float> in_2d = in.clone();   // copy, to avoid reshape failure if strides are non-contiguous
    in_2d = in_2d.reshape_ref({ nbeams * pow2(r), ntime });
    
    // Reshaped time-downsampled input array: (nbeams * 2^r, ntime/2)
    Array<float> in_ds({ nbeams * pow2(r), ntime/2 }, af_uhost | af_zero);
    reference_downsample_time(in_2d, in_ds, false);  // normalize=false, i.e. sum with no factor 0.5
    
    // Apply "small" lags (one-sample lags in even channels), before frequency downsampling.
    Array<float> in_ds2 = in_ds.clone();   // copy since we'll need 'in_ds' later.
    lagbuf_small->apply_lags(in_ds2);

    // Downsample in frequency, and apply "large" lags.
    Array<float> out_tmp({ nbeams * pow2(r-1), ntime/2 }, af_uhost | af_zero);
    reference_downsample_freq(in_ds2, out_tmp, false);   // normalize=false, i.e. sum with no factor 0.5
    lagbuf_large->apply_lags(out_tmp);
    
    // Reshape output_tmp array from 2-d to 3-d, and copy to caller-specified array.
    out_tmp = out_tmp.reshape_ref({ nbeams, pow2(r-1), ntime/2 });
    outp[0].fill(out_tmp);
    
    if (params.num_downsampling_levels == 1)
	return;
    
    // Recurse to next downsampling level.
    in_ds = in_ds.reshape_ref({ nbeams, pow2(r), ntime/2 });
    next->apply(in_ds, outp+1);
}


}  // namespace pirate
