#include "../include/pirate/internals/ReferenceDedispersionKernel.hpp"
#include "../include/pirate/internals/ReferenceLagbuf.hpp"
#include "../include/pirate/internals/ReferenceTree.hpp"
#include "../include/pirate/internals/inlines.hpp"     // pow2()
#include "../include/pirate/internals/utils.hpp"       // bit_reverse_slow()

using namespace std;
using namespace ksgpu;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


ReferenceDedispersionKernel::ReferenceDedispersionKernel(const Params &params_) :
    params(params_)
{
    params.validate(false);    // on_gpu=false

    long B = params.beams_per_batch;
    long A = params.nambient;
    long F = pow2(params.rank);
    long T = params.ntime;
    long Ar = integer_log2(A);
    long N = xdiv(params.total_beams, B);

    this->trees.resize(N);
    for (long n = 0; n < N; n++)
	trees[n] = ReferenceTree::make({B,A,F,T});

    if (!params.apply_input_residual_lags)
	return;

    // Remaining code initializes this->rlag_bufs (only if params.apply_input_residual_lags == false).

    if (params.nelts_per_segment <= 0) {
	throw runtime_error("ReferenceDedispersionKernel: if params.apply_input_residual_lags==true,"
			    " then params.nelts_per_segment must be initialized and > 0" );
    }
    
    Array<int> rlags({B,A,F}, af_uhost);
    
    for (long b = 0; b < B; b++) {
	for (long a = 0; a < A; a++) {
	    // Ambient index 'a' represents a bit-reversed coarse DM.
	    // Index 'f' represents a fine frequency.
	    for (long f = 0; f < F; f++) {
		long lag = rb_lag(f, a, Ar, params.rank, params.input_is_downsampled_tree);
		rlags.data[b*A*F + a*F + f] = lag % params.nelts_per_segment;  // residual lag
	    }
	}
    }
    
    this->rlag_bufs.resize(N);
    for (long n = 0; n < N; n++)
	rlag_bufs[n] = make_shared<ReferenceLagbuf> (rlags, T);
}


void ReferenceDedispersionKernel::apply(Array<float> &in, Array<float> &out, long itime, long ibeam)
{
    long B = params.beams_per_batch;
    long A = params.nambient;
    long N = pow2(params.rank);
    long T = params.ntime;
    long R = params.ringbuf_nseg;
    long S = params.nelts_per_segment;

    std::initializer_list<ssize_t> dd_shape = {B,A,N,T};
    std::initializer_list<ssize_t> rb_shape = {R*S};

    assert(!params.input_is_ringbuf || !params.output_is_ringbuf);
    assert(in.shape_equals(params.input_is_ringbuf ? rb_shape : dd_shape));
    assert(out.shape_equals(params.output_is_ringbuf ? rb_shape : dd_shape));

    // Compare (itime, ibeam) with expected values.
    assert(itime == expected_itime);
    assert(ibeam == expected_ibeam);

    // Update expected (itime, ibeam).
    expected_ibeam += B;
    assert(expected_ibeam <= params.total_beams);
    
    if (expected_ibeam == params.total_beams) {
	expected_ibeam = 0;
	expected_itime++;
    }
    
    int batch = xdiv(ibeam, B);
    long rb_pos = itime * params.total_beams + ibeam;

    Array<float> dd = params.output_is_ringbuf ? in : out;

    if (params.input_is_ringbuf)
	_copy_from_ringbuf(in, dd, rb_pos);
    else if (dd.data != in.data)
	dd.fill(in);
	
    if (params.apply_input_residual_lags)
	rlag_bufs.at(batch)->apply_lags(dd);

    trees.at(batch)->dedisperse(dd);

    if (params.output_is_ringbuf)
	_copy_to_ringbuf(dd, out, rb_pos);
    else
	assert(out.data == dd.data);
}


void ReferenceDedispersionKernel::_copy_to_ringbuf(const Array<float> &in, Array<float> &out, long rb_pos)
{
    long B = params.beams_per_batch;
    long A = params.nambient;
    long N = pow2(params.rank);
    long T = params.ntime;
    long R = params.ringbuf_nseg;
    long S = params.nelts_per_segment;
    long ns = xdiv(T,S);

    assert(in.shape_equals({B,A,N,T}));  // dedispersion buffer
    assert(out.shape_equals({R*S}));     // ringbuf
    assert(out.is_fully_contiguous());
    assert(params.ringbuf_locations.shape_equals({ns*A*N,4}));

    long dd_bstride = in.strides[0];
    long dd_astride = in.strides[1];
    long dd_nstride = in.strides[2];
    assert(in.strides[3] == 1);

    const uint *rb_loc = params.ringbuf_locations.data;
    const float *dd = in.data;
    float *ringbuf = out.data;
    
    // Loop over segments in tree.
    for (long s = 0; s < ns; s++) {
	for (long a = 0; a < A; a++) {
	    for (long n = 0; n < N; n++) {
		long iseg = s*A*N + a*N + n;               // index in rb_loc array (same for all beams)
		const float *dd0 = dd + a*dd_astride + n*dd_nstride + s*S;  // address in dedispersion buf (at beam 0)

		uint rb_offset = rb_loc[4*iseg];    // in segments, not bytes
		uint rb_phase = rb_loc[4*iseg+1];   // index of (time chunk, beam) pair, relative to current pair
		uint rb_len = rb_loc[4*iseg+2];     // number of (time chunk, beam) pairs in ringbuf (same as Ringbuf::rb_len)
		uint rb_nseg = rb_loc[4*iseg+3];    // number of segments per (time chunk, beam) (same as Ringbuf::nseg_per_beam)
		
		for (long b = 0; b < B; b++) {
		    uint i = (rb_pos + rb_phase + b) % rb_len;  // note "+b" here
		    long s = rb_offset + (i * rb_nseg);         // segment offset, relative to (float *ringbuf)
		    memcpy(ringbuf + s*S, dd0 + b*dd_bstride, S * sizeof(float));
		}
	    }
	}
    }
}


void ReferenceDedispersionKernel::_copy_from_ringbuf(const Array<float> &in, Array<float> &out, long rb_pos)
{
    long B = params.beams_per_batch;
    long A = params.nambient;
    long N = pow2(params.rank);
    long T = params.ntime;
    long R = params.ringbuf_nseg;
    long S = params.nelts_per_segment;
    long ns = xdiv(T,S);

    assert(in.shape_equals({R*S}));  // ringbuf
    assert(in.is_fully_contiguous());
    assert(out.shape_equals({B,A,N,T}));  // dedispersion buffer
    assert(params.ringbuf_locations.shape_equals({ns*A*N,4}));

    const uint *rb_loc = params.ringbuf_locations.data;
    const float *ringbuf = in.data;
    float *dd = out.data;
    
    long dd_bstride = out.strides[0];
    long dd_astride = out.strides[1];
    long dd_nstride = out.strides[2];
    assert(out.strides[3] == 1);

    // Loop over segments in tree.
    for (long s = 0; s < ns; s++) {
	for (long a = 0; a < A; a++) {
	    for (long n = 0; n < N; n++) {
		long iseg = s*A*N + a*N + n;         // index in rb_loc array (same for all beams)
		float *dd0 = dd + n*dd_nstride + a*dd_astride + s*S; // address in dedispersion buf (at beam 0)
		
		uint rb_offset = rb_loc[4*iseg];    // in segments, not bytes
		uint rb_phase = rb_loc[4*iseg+1];   // index of (time chunk, beam) pair, relative to current pair
		uint rb_len = rb_loc[4*iseg+2];     // number of (time chunk, beam) pairs in ringbuf (same as Ringbuf::rb_len)
		uint rb_nseg = rb_loc[4*iseg+3];    // number of segments per (time chunk, beam) (same as Ringbuf::nseg_per_beam)
		
		for (long b = 0; b < B; b++) {
		    uint i = (rb_pos + rb_phase + b) % rb_len;  // note "+b" here
		    long s = rb_offset + (i * rb_nseg);         // segment offset, relative to (float *ringbuf)
		    memcpy(dd0 + b*dd_bstride, ringbuf + s*S, S * sizeof(float));
		}
	    }
	}
    }
}


} // namespace pirate
