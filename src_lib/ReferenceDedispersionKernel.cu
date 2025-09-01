#include "../include/pirate/DedispersionKernel.hpp"
#include "../include/pirate/ReferenceLagbuf.hpp"
#include "../include/pirate/ReferenceTree.hpp"
#include "../include/pirate/inlines.hpp"     // pow2()
#include "../include/pirate/utils.hpp"       // bit_reverse_slow()

using namespace std;
using namespace ksgpu;

namespace pirate {
#if 0
}  // editor auto-indent
#endif

// GPU dedispersion kernels assumes (nelts_per_segment == nelts_per_cache_line), but
// reference kernels allow (nelts_per_segment) to be a multiple of (nelts_per_cache_line),
// where:
//
//   nelts_per_cache_line = (8 * constants::bytes_per_gpu_cache_line) / dtype.nbits.
//
// This is in order to enable a unit test where we check agreement between a float16
// GPU kernel, and a float32 reference kernel derived from the same DedispersionPlan.
// In this case, we want the reference kernel to have dtype float32, but use a value
// of 'nelts_per_segment' which matched to the float16 GPU kernel.
//
// Enabling this feature is straightforward: we just compute residual lags and ring
// buffer offsets using the value of (params.nelts_per_segment), rather than assuming
// nelts_per_segment == 32.


ReferenceDedispersionKernel::ReferenceDedispersionKernel(const Params &params_) :
    params(params_)
{
    params.dtype = Dtype::native<float>();
    params.validate();

    this->nbatches = xdiv(params.total_beams, params.beams_per_batch);
    
    long B = params.beams_per_batch;
    long A = pow2(params.amb_rank);
    long F = pow2(params.dd_rank);
    long T = params.ntime;
    long S = params.nspec;
    
    this->trees.resize(nbatches);			  
    for (long n = 0; n < nbatches; n++)
	trees[n] = ReferenceTree::make({B,A,F,T*S}, S);   // (shape, nspec)

    if (params.apply_input_residual_lags)
	this->_init_rlags();
}


void ReferenceDedispersionKernel::apply(Array<void> &in_, Array<void> &out_, long ibatch, long it_chunk)
{
    long B = params.beams_per_batch;
    long A = pow2(params.amb_rank);
    long N = pow2(params.dd_rank);
    long T = params.ntime;
    long S = params.nspec;
    
    // Error checking, shape checking.
    DedispersionKernelIobuf inbuf(params, in_, params.input_is_ringbuf, false);     // on_gpu=false
    DedispersionKernelIobuf outbuf(params, out_, params.output_is_ringbuf, false);  // on_gpu=false
    
    xassert((ibatch >= 0) && (ibatch < nbatches));
    xassert(it_chunk >= 0);

    // The reference kernel uses float32, regardless of what dtype is specified.
    Array<float> in = in_.template cast<float> ("ReferenceDedispersionKernel::apply(): 'in' array");
    Array<float> out = out_.template cast<float> ("ReferenceDedispersionKernel::apply(): 'out' array");

    // Reshape "simple" bufs to 4-d.
    in = params.input_is_ringbuf ? in : in.reshape({B,A,N,T*S});
    out = params.output_is_ringbuf ? out : out.reshape({B,A,N,T*S});

    long rb_pos = it_chunk * params.total_beams + (ibatch * params.beams_per_batch);
    Array<float> dd = params.output_is_ringbuf ? in : out;

    if (params.input_is_ringbuf)
	_copy_from_ringbuf(in, dd, rb_pos);
    else if (dd.data != in.data)
	dd.fill(in);
	
    if (params.apply_input_residual_lags)
	rlag_bufs.at(ibatch)->apply_lags(dd);

    trees.at(ibatch)->dedisperse(dd);

    if (params.output_is_ringbuf)
	_copy_to_ringbuf(dd, out, rb_pos);
    else
	xassert(out.data == dd.data);   // FIXME in-place assumed
}


// Helper function called by constructor.
void ReferenceDedispersionKernel::_init_rlags()
{
    long B = params.beams_per_batch;
    long A = pow2(params.amb_rank);
    long F = pow2(params.dd_rank);
    long N = params.nt_per_segment;
    long T = params.ntime;
    long S = params.nspec;
    
    Array<int> rlags({B,A,F}, af_uhost);
    
    for (long b = 0; b < B; b++) {
	for (long a = 0; a < A; a++) {
	    // Ambient index 'a' represents a bit-reversed coarse DM.
	    // Index 'f' represents a fine frequency.
	    for (long f = 0; f < F; f++) {
		long lag = rb_lag(f, a, params.amb_rank, params.dd_rank, params.input_is_downsampled_tree);
		rlags.data[b*A*F + a*F + f] = (lag % N) * S;  // residual lag (including factor nspec)
	    }
	}
    }
    
    this->rlag_bufs.resize(nbatches);
    for (long n = 0; n < nbatches; n++)
	rlag_bufs[n] = make_shared<ReferenceLagbuf> (rlags, T*S);
}

// Helper function called by apply().
void ReferenceDedispersionKernel::_copy_to_ringbuf(const Array<float> &in, Array<float> &out, long rb_pos)
{
    long B = params.beams_per_batch;
    long A = pow2(params.amb_rank);
    long N = pow2(params.dd_rank);
    long T = params.ntime;
    long S = params.nspec;
    long R = params.ringbuf_nseg;
    long RS = params.nt_per_segment * params.nspec;
    long ns = xdiv(T, params.nt_per_segment);

    xassert_shape_eq(in, ({B,A,N,T*S}));  // dedispersion buffer (4-d assumed)
    xassert_shape_eq(out, ({R*RS}));      // ringbuf
    xassert_shape_eq(params.ringbuf_locations, ({ns*A*N,4}));

    xassert(in.get_ncontig() >= 1);
    xassert(out.is_fully_contiguous());

    long dd_bstride = in.strides[0];
    long dd_astride = in.strides[1];
    long dd_nstride = in.strides[2];

    const uint *rb_loc = params.ringbuf_locations.data;
    const float *dd = in.data;
    float *ringbuf = out.data;
    
    // Loop over segments in tree.
    for (long s = 0; s < ns; s++) {
	for (long a = 0; a < A; a++) {
	    for (long n = 0; n < N; n++) {
		long iseg = s*A*N + a*N + n;               // index in rb_loc array (same for all beams)
		const float *dd0 = dd + a*dd_astride + n*dd_nstride + s*RS;  // address in dedispersion buf (at beam 0)

		uint rb_offset = rb_loc[4*iseg];    // in segments, not bytes
		uint rb_phase = rb_loc[4*iseg+1];   // index of (time chunk, beam) pair, relative to current pair
		uint rb_len = rb_loc[4*iseg+2];     // number of (time chunk, beam) pairs in ringbuf (same as Ringbuf::rb_len)
		uint rb_nseg = rb_loc[4*iseg+3];    // number of segments per (time chunk, beam) (same as Ringbuf::nseg_per_beam)
		
		for (long b = 0; b < B; b++) {
		    uint i = (rb_pos + rb_phase + b) % rb_len;  // note "+b" here
		    long s = rb_offset + (i * rb_nseg);         // segment offset, relative to (float *ringbuf)
		    memcpy(ringbuf + s*RS, dd0 + b*dd_bstride, RS * sizeof(float));
		}
	    }
	}
    }
}


// Helper function called by apply().
void ReferenceDedispersionKernel::_copy_from_ringbuf(const Array<float> &in, Array<float> &out, long rb_pos)
{
    long B = params.beams_per_batch;
    long A = pow2(params.amb_rank);
    long N = pow2(params.dd_rank);
    long T = params.ntime;
    long S = params.nspec;
    long R = params.ringbuf_nseg;
    long RS = params.nt_per_segment * params.nspec;
    long ns = xdiv(T, params.nt_per_segment);

    xassert_shape_eq(in, ({R*RS}));        // ringbuf
    xassert_shape_eq(out, ({B,A,N,T*S}));  // dedispersion buffer (4-d assumed)
    xassert_shape_eq(params.ringbuf_locations, ({ns*A*N,4}));
    
    xassert(in.is_fully_contiguous());
    xassert(out.get_ncontig() >= 1);

    const uint *rb_loc = params.ringbuf_locations.data;
    const float *ringbuf = in.data;
    float *dd = out.data;
    
    long dd_bstride = out.strides[0];
    long dd_astride = out.strides[1];
    long dd_nstride = out.strides[2];
    xassert(out.strides[3] == 1);

    // Loop over segments in tree.
    for (long s = 0; s < ns; s++) {
	for (long a = 0; a < A; a++) {
	    for (long n = 0; n < N; n++) {
		long iseg = s*A*N + a*N + n;         // index in rb_loc array (same for all beams)
		float *dd0 = dd + n*dd_nstride + a*dd_astride + s*RS; // address in dedispersion buf (at beam 0)
		
		uint rb_offset = rb_loc[4*iseg];    // in segments, not bytes
		uint rb_phase = rb_loc[4*iseg+1];   // index of (time chunk, beam) pair, relative to current pair
		uint rb_len = rb_loc[4*iseg+2];     // number of (time chunk, beam) pairs in ringbuf (same as Ringbuf::rb_len)
		uint rb_nseg = rb_loc[4*iseg+3];    // number of segments per (time chunk, beam) (same as Ringbuf::nseg_per_beam)
		
		for (long b = 0; b < B; b++) {
		    uint i = (rb_pos + rb_phase + b) % rb_len;  // note "+b" here
		    long s = rb_offset + (i * rb_nseg);         // segment offset, relative to (float *ringbuf)
		    memcpy(dd0 + b*dd_bstride, ringbuf + s*RS, RS * sizeof(float));
		}
	    }
	}
    }
}


} // namespace pirate
