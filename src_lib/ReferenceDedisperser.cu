#include "../include/pirate/internals/ReferenceDedisperser.hpp"
#include "../include/pirate/internals/CacheLineRingbuf.hpp"
#include "../include/pirate/internals/LaggedCacheLine.hpp"

// pow2(), xdiv(), bit_reverse_slow(), print_kv()
#include "../include/pirate/internals/inlines.hpp"

// check_rank(), reference_downsample_time(), reference_extract_odd_channels()
#include "../include/pirate/internals/utils.hpp"

#include <cassert>

using namespace std;
using namespace gputils;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// -------------------------------------------------------------------------------------------------
//
// ReferenceDedipserser


// Helper for ReferenceDedisperser constructor.
// Prevents constructor from segfaulting, if invoked with empty shared_ptr.
static DedispersionPlan *deref(const shared_ptr<DedispersionPlan> &p)
{
    if (!p)
	throw runtime_error("ReferenceDedisperser constructor called with empty shared_ptr");
    return p.get();
}


ReferenceDedisperser::ReferenceDedisperser(const shared_ptr<DedispersionPlan> &plan_, int sophistication_) :
    sophistication(sophistication_),
    config(deref(plan_)->config),
    plan(plan_)
{
    assert((sophistication >= 0) && (sophistication <= 3));
    
    this->input_rank = config.tree_rank;
    this->input_nt = config.time_samples_per_chunk;
    this->output_ntrees = plan->stage1_trees.size();
    this->nds = plan->stage0_trees.size();
    
    _allocate_output_arrays();

    if (sophistication == 0) {
	_allocate_downsampled_inputs();
	_init_simple_trees();
    }
    
    if (sophistication >= 1) {
	_allocate_intermediate_arrays();
	_init_lagged_downsampler();
	_init_first_trees();
	_init_second_trees();
    }
    
    if (sophistication == 1)
	_init_big_lagbufs();
    
    if (sophistication == 2)
	_init_max_ringbuf();
    
    if (sophistication >= 2)
	_init_residual_lagbufs();

    if (sophistication == 3)
	_init_proper_ringbufs();
}


void ReferenceDedisperser::dedisperse(const Array<float> &in)
{
    if (sophistication == 0) {
	_compute_downsampled_inputs(in);
	_apply_simple_trees();
    }
    else if (sophistication == 1) {
	_apply_lagged_downsampler(in);
	_apply_first_trees();
	_apply_big_lagbufs();
	_apply_second_trees();	
    }
    else if (sophistication == 2) {
	_apply_lagged_downsampler(in);
	_apply_first_trees();	
	_apply_max_ringbuf();
	_apply_residual_lagbufs();
	_apply_second_trees();
    }
    else if (sophistication == 3) {
	_apply_lagged_downsampler(in);
	_apply_first_trees();	
	_proper_ringbuf_to_staging();
	_proper_s0_to_staging();
	_proper_staging_to_staging();
	_proper_staging_to_ringbuf();
	_proper_staging_to_s1();
	_proper_s0_to_s1();
	_proper_s1_to_s1();
	_apply_residual_lagbufs();
	_apply_second_trees();
    }

    this->pos++;
}


void ReferenceDedisperser::_allocate_downsampled_inputs()
{
    this->downsampled_inputs.resize(nds);

    for (int ids = 0; ids < nds; ids++) {
	int nchan = pow2(input_rank);
	int nt_ds = xdiv(input_nt, pow2(ids));	
	this->downsampled_inputs[ids] = Array<float> ({nchan,nt_ds}, af_uhost | af_zero);
    }    
}


void ReferenceDedisperser::_compute_downsampled_inputs(const Array<float> &in)
{
    assert(downsampled_inputs.size() == nds);  // check that _allocate_downsampled_inputs() was called
    assert(in.shape_equals({pow2(input_rank), input_nt}));
    assert(in.strides[1] == 1);

    downsampled_inputs[0].fill(in);  // FIXME is this copy necessary?
    for (int ids = 1; ids < nds; ids++)
	reference_downsample_time(downsampled_inputs[ids-1], downsampled_inputs[ids], false);  // normalize=false, i.e. no factor 0.5
}


void ReferenceDedisperser::_init_lagged_downsampler()
{
    if (nds <= 1)
	return;
    
    const DedispersionPlan::Stage0Tree &st0 = this->plan->stage0_trees[1];
	
    ReferenceLaggedDownsampler::Params ld_params;
    ld_params.small_input_rank = st0.rank0 + 1;
    ld_params.large_input_rank = st0.rank0 + st0.rank1 + 1;
    ld_params.num_downsampling_levels = nds - 1;
    ld_params.nbeams = 1;
    ld_params.ntime = config.time_samples_per_chunk;

    this->lagged_downsampler = make_shared<ReferenceLaggedDownsampler> (ld_params);
}


void ReferenceDedisperser::_apply_lagged_downsampler(const gputils::Array<float> &in)
{
    assert(intermediate_arrays.size() == nds);  // check that _allocate_intermediate_arrays() was called
    assert(intermediate_arrays[0].shape_equals({pow2(input_rank), input_nt}));
    assert(in.shape_equals({pow2(input_rank), input_nt}));
    assert(in.strides[1] == 1);

    intermediate_arrays[0].fill(in);
    
    if (nds > 1)
	this->lagged_downsampler->apply(in, &intermediate_arrays[1]);
}

    
void ReferenceDedisperser::_allocate_intermediate_arrays()
{
    ssize_t nflat = plan->stage0_iobuf_segments_per_beam * plan->nelts_per_segment;
    ssize_t pos = 0;
    
    this->intermediate_flattened = Array<float> ({nflat}, af_uhost | af_zero);
    this->intermediate_arrays.resize(nds);
    
    for (int ids = 0; ids < nds; ids++) {
	const DedispersionPlan::Stage0Tree &st0 = plan->stage0_trees[ids];
	assert(pos == st0.iobuf_base_segment * plan->nelts_per_segment);
	
	ssize_t rank = st0.rank0 + st0.rank1;
	ssize_t nt_ds = st0.nt_ds;
	ssize_t nelts = pow2(rank) * nt_ds;

	Array<float> s = intermediate_flattened.slice(0, pos, pos+nelts);
	this->intermediate_arrays[ids] = s.reshape_ref({pow2(rank), nt_ds});
	pos += nelts;
    }

    assert(pos == nflat);    
}


void ReferenceDedisperser::_allocate_output_arrays()
{
    ssize_t nflat = plan->stage1_iobuf_segments_per_beam * plan->nelts_per_segment;
    ssize_t pos = 0;
    
    this->output_flattened = Array<float> ({nflat}, af_uhost | af_zero);
    this->output_arrays.resize(output_ntrees);
    
    for (int itree = 0; itree < output_ntrees; itree++) {
	const DedispersionPlan::Stage1Tree &st1 = plan->stage1_trees[itree];
	assert(pos == st1.iobuf_base_segment * plan->nelts_per_segment);
	
	ssize_t rank = st1.rank0 + st1.rank1_trigger;
	ssize_t nt_ds = st1.nt_ds;
	ssize_t nelts = pow2(rank) * nt_ds;

	Array<float> s = output_flattened.slice(0, pos, pos+nelts);
	this->output_arrays[itree] = s.reshape_ref({pow2(rank), nt_ds});
	pos += nelts;
    }

    assert(pos == nflat);
}


void ReferenceDedisperser::_init_simple_trees()
{
    for (const DedispersionPlan::Stage1Tree &st1: plan->stage1_trees)
	this->simple_trees.push_back(SimpleTree(st1));
}


void ReferenceDedisperser::_apply_simple_trees()
{
    assert(downsampled_inputs.size() == nds);       // check that _allocate_downsampled_inputs() was called
    assert(output_arrays.size() == output_ntrees);  // check that _allocate_output_arrays() was called
    assert(simple_trees.size() == output_ntrees);   // check that _init_simple_trees() was called
    
    for (int itree = 0; itree < output_ntrees; itree++) {
	int ids = plan->stage1_trees[itree].ds_level;
	simple_trees[itree].dedisperse(downsampled_inputs.at(ids), output_arrays.at(itree));
    }
}


void ReferenceDedisperser::_init_first_trees()
{
    for (const DedispersionPlan::Stage0Tree &st0: plan->stage0_trees)
	this->first_trees.push_back(FirstTree(st0));
}


void ReferenceDedisperser::_apply_first_trees()
{
    assert(intermediate_arrays.size() == nds);         // check that _allocate_intermediate_arrays() was called
    assert(first_trees.size() == nds);                 // check that _init_first_trees() was called

    for (int ids = 0; ids < nds; ids++)
	first_trees.at(ids).dedisperse(intermediate_arrays.at(ids));
}


void ReferenceDedisperser::_init_big_lagbufs()
{
    for (const DedispersionPlan::Stage1Tree &st1: plan->stage1_trees) {
	int rank0 = st1.rank0;
	int rank1 = st1.rank1_trigger;
	bool is_downsampled = (st1.ds_level > 0);
	    
	vector<int> lags(pow2(rank0+rank1));
	
	for (int i = 0; i < pow2(rank1); i++) {
	    for (int j = 0; j < pow2(rank0); j++) {
		int row = i*pow2(rank0) + j;
		int slag = st1.segment_lags.at({row});
		int rlag = st1.residual_lags.at({row});
		int lag = rb_lag(i, j, rank0, rank1, is_downsampled);
		
		assert(lag == slag * plan->nelts_per_segment + rlag);		
		lags[row] = lag;
	    }
	}

	auto lagbuf = make_shared<ReferenceLagbuf> (lags, st1.nt_ds);
	this->big_lagbufs.push_back(lagbuf);
    }
}


void ReferenceDedisperser::_apply_big_lagbufs()
{
    assert(intermediate_arrays.size() == nds);       // check that _allocate_intermediate_arrays() was called
    assert(output_arrays.size() == output_ntrees);   // check that _allocate_output_arrays() was called
    assert(big_lagbufs.size() == output_ntrees);     // check that _init_big_lagbufs() was called

    for (int itree = 0; itree < output_ntrees; itree++) {
	const DedispersionPlan::Stage1Tree &st1 = plan->stage1_trees.at(itree);
	const Array<float> &in = intermediate_arrays.at(st1.ds_level);
	Array<float> &out = output_arrays.at(itree);

	int nchan = pow2(st1.rank0 + st1.rank1_trigger);
	Array<float> s = in.slice(0, 0, nchan);
	out.fill(s);

	big_lagbufs.at(itree)->apply_lags(out);
    }
}


void ReferenceDedisperser::_init_max_ringbuf()
{
    assert(config.bloat_dedispersion_plan);

    for (const LaggedCacheLine &cl: plan->lagged_cache_lines) {
	for (int idst = 0; idst < cl.ndst; idst++)
	    this->max_clag = std::max(max_clag, cl.dst_clag[idst]);
    }

    ssize_t nint = intermediate_flattened.size;
    
    assert(max_clag >= 0);
    assert(nint > 0);
    
    this->max_ringbuf = Array<float>({max_clag+1,nint}, af_uhost | af_zero);
}


void ReferenceDedisperser::_apply_max_ringbuf()
{
    assert(intermediate_arrays.size() == nds);       // check that _allocate_intermediate_arrays() was called
    assert(output_arrays.size() == output_ntrees);   // check that _allocate_output_arrays() was called
    assert(max_clag >= 0);                           // check that _init_max_ringbuf() was called

    ssize_t nseg = plan->nelts_per_segment;
    ssize_t nint = intermediate_flattened.size;
    ssize_t nout = output_flattened.size;
    ssize_t rpos = pos % (max_clag+1);

    assert(nint > 0);
    assert(nout > 0);

    memcpy(max_ringbuf.data + rpos * nint,
	   intermediate_flattened.data,
	   nint * sizeof(float));
    
    for (const LaggedCacheLine &cl: plan->lagged_cache_lines) {
	int ksrc = cl.src_segment * nseg;
	assert((ksrc >= 0) && (ksrc+nseg <= nint));
	
	for (int idst = 0; idst < cl.ndst; idst++) {
	    int clag = cl.dst_clag[idst];
	    int jsrc = (rpos - clag + max_clag+1) % (max_clag+1);
	    assert((clag >= 0) && (clag <= max_clag));
	    
	    int kdst = cl.dst_segment[idst] * nseg;
	    assert((kdst >= 0) && (kdst+nseg <= nout));

	    memcpy(output_flattened.data + kdst,
		   max_ringbuf.data + jsrc*nint + ksrc,
		   nseg * sizeof(float));
	}
    }
}


void ReferenceDedisperser::_init_residual_lagbufs()
{
    for (const DedispersionPlan::Stage1Tree &st1: plan->stage1_trees) {
	int rank0 = st1.rank0;
	int rank1 = st1.rank1_trigger;
	vector<int> lags(pow2(rank0+rank1));
	
	for (int i = 0; i < pow2(rank1); i++) {
	    for (int j = 0; j < pow2(rank0); j++) {
		int row = i*pow2(rank0) + j;
		lags[row] = st1.residual_lags.at({row});
	    }
	}

	auto lagbuf = make_shared<ReferenceLagbuf> (lags, st1.nt_ds);
	this->residual_lagbufs.push_back(lagbuf);
    }
}


void ReferenceDedisperser::_apply_residual_lagbufs()
{
    assert(output_arrays.size() == output_ntrees);     // check that _allocate_output_arrays() was called
    assert(residual_lagbufs.size() == output_ntrees);  // check that _init_residual_lagbufs() was called

    for (int itree = 0; itree < output_ntrees; itree++) {
	Array<float> &out = output_arrays.at(itree);
	residual_lagbufs.at(itree)->apply_lags(out);
    }
}


void ReferenceDedisperser::_init_second_trees()
{
    for (const DedispersionPlan::Stage1Tree &st1: plan->stage1_trees)
	this->second_trees.push_back(SecondTree(st1));
}


void ReferenceDedisperser::_apply_second_trees()
{
    assert(output_arrays.size() == output_ntrees);   // check that _allocate_output_arrays() was called
    assert(second_trees.size() == output_ntrees);    // check that _init_second_trees() was called

    for (int itree = 0; itree < output_ntrees; itree++)
	second_trees.at(itree).dedisperse(output_arrays.at(itree));
}


void ReferenceDedisperser::_init_proper_ringbufs()
{
    const vector<CacheLineRingbuf::Buffer> &buffers = plan->cache_line_ringbuf->buffers;

    for (unsigned int rb_lag = 0; rb_lag < buffers.size(); rb_lag++) {
	ssize_t npri = buffers[rb_lag].primary_entries.size();
	ssize_t nsec = buffers[rb_lag].secondary_entries.size();
	ssize_t nelts = (npri + nsec) * plan->nelts_per_segment;

	Array<float> rb({rb_lag, nelts}, af_uhost | af_zero);
	Array<float> sin({nelts}, af_uhost | af_zero);
	Array<float> sout({nelts}, af_uhost | af_zero);
	
	this->proper_ringbufs.push_back(rb);
	this->staging_inbufs.push_back(sin);
	this->staging_outbufs.push_back(sout);
    }
}


// (oldest ringbuf entries) -> (staging_inbuf)
void ReferenceDedisperser::_proper_ringbuf_to_staging()
{
    // Check that _init_proper_ringbufs() was called.
    ssize_t nbuf = plan->cache_line_ringbuf->buffers.size();
    assert(proper_ringbufs.size() == nbuf);
    assert(staging_inbufs.size() == nbuf);

    for (unsigned int rb_lag = 1; rb_lag < nbuf; rb_lag++) {
	ssize_t rpos = pos % rb_lag;
	Array<float> s = proper_ringbufs.at(rb_lag).slice(0,rpos);
	staging_inbufs.at(rb_lag).fill(s);
    }
}


// (stage0 iobufs) -> (staging outbuf), based on CacheLineRingbuf::PrimaryEntry::src_segment
void ReferenceDedisperser::_proper_s0_to_staging()
{
    ssize_t nseg = plan->nelts_per_segment;
    ssize_t nsrc = plan->stage0_iobuf_segments_per_beam * nseg;
    ssize_t nbuf = plan->cache_line_ringbuf->buffers.size();

    // Check that _init_intermediate_arrays() and _init_proper_ringbufs() were called.
    assert(intermediate_flattened.shape_equals({nsrc}));
    assert(staging_outbufs.size() == nbuf);

    for (unsigned int rb_lag = 1; rb_lag < nbuf; rb_lag++) {
	const CacheLineRingbuf::Buffer &buf = plan->cache_line_ringbuf->buffers.at(rb_lag);
	ssize_t npri = buf.primary_entries.size();
	
	Array<float> dst = staging_outbufs.at(rb_lag);
	assert(npri * nseg <= dst.size);

	for (ssize_t ipri = 0; ipri < npri; ipri++) {
	    ssize_t isrc = buf.primary_entries.at(ipri).src_segment * nseg;
	    assert((isrc >= 0) && (isrc + nseg <= nsrc));

	    memcpy(dst.data + ipri*nseg,
		   intermediate_flattened.data + isrc,
		   nseg * sizeof(float));
	}
    }
}


// (staging inbuf) -> (staging outbuf), based on CacheLineRingbuf::SecondaryEntry::src_*
void ReferenceDedisperser::_proper_staging_to_staging()
{
    const vector<CacheLineRingbuf::Buffer> &buffers = plan->cache_line_ringbuf->buffers;
    ssize_t nseg = plan->nelts_per_segment;
    ssize_t nbuf = buffers.size();

    // Check that _init_proper_ringbufs() was called.
    assert(staging_inbufs.size() == nbuf);
    assert(staging_outbufs.size() == nbuf);

    for (unsigned int rb_lag = 1; rb_lag < nbuf; rb_lag++) {
	const CacheLineRingbuf::Buffer &buf = plan->cache_line_ringbuf->buffers.at(rb_lag);
	ssize_t npri = buf.primary_entries.size();
	ssize_t nsec = buf.secondary_entries.size();

	Array<float> &dst_arr = staging_outbufs.at(rb_lag);
	assert(dst_arr.shape_equals({ (npri+nsec) * nseg }));
	float *dst = dst_arr.data + (npri * nseg);

	for (ssize_t isec = 0; isec < nsec; isec++) {
	    // Reminder: SecondaryEntry has members 'src_rb_lag', 'src_rb_index', 'src_is_primary'.
	    const CacheLineRingbuf::SecondaryEntry &e = buf.secondary_entries.at(isec);
	    const CacheLineRingbuf::Buffer &src_buf = plan->cache_line_ringbuf->buffers.at(e.src_rb_lag);
	    const Array<float> &src_arr = staging_inbufs.at(e.src_rb_lag);
	    
	    ssize_t src_base = e.src_is_primary ? 0 : src_buf.primary_entries.size();
	    ssize_t isrc = (src_base + e.src_rb_index) * nseg;
	    assert((isrc >= 0) && (isrc + nseg <= src_arr.size));

	    memcpy(dst + isec*nseg,       // not (dst_arr.data + isec*nseg)
		   src_arr.data + isrc,
		   nseg * sizeof(float));
	}
    }
    
}


// (staging outbuf) -> (newest ringbuf entries)
void ReferenceDedisperser::_proper_staging_to_ringbuf()
{
    // Check that _init_proper_ringbufs() was called.
    ssize_t nbuf = plan->cache_line_ringbuf->buffers.size();
    assert(proper_ringbufs.size() == nbuf);
    assert(staging_outbufs.size() == nbuf);

    for (unsigned int rb_lag = 1; rb_lag < nbuf; rb_lag++) {
	ssize_t rpos = pos % rb_lag;
	Array<float> s = proper_ringbufs.at(rb_lag).slice(0,rpos);
	s.fill(staging_outbufs.at(rb_lag));
    }
}


// (staging inbuf) -> (stage1 iobufs), based on CacheLineRingbuf::*Entry::dst_segment
void ReferenceDedisperser::_proper_staging_to_s1()
{
    ssize_t nseg = plan->nelts_per_segment;
    ssize_t nbuf = plan->cache_line_ringbuf->buffers.size();
    ssize_t ndst = plan->stage1_iobuf_segments_per_beam * nseg;
    
    assert(staging_inbufs.size() == nbuf);          // check that _init_proper_ringbufs() was called
    assert(output_flattened.shape_equals({ndst}));  // check that _allocate_output_arrays() was called
    
    for (unsigned int rb_lag = 1; rb_lag < nbuf; rb_lag++) {
	const CacheLineRingbuf::Buffer &buf = plan->cache_line_ringbuf->buffers.at(rb_lag);
	Array<float> src = staging_inbufs.at(rb_lag);
	
	ssize_t npri = buf.primary_entries.size();
	ssize_t nsec = buf.secondary_entries.size();
	assert(src.shape_equals({ (npri+nsec) * nseg }));
	
	for (ssize_t ipri = 0; ipri < npri; ipri++) {
	    ssize_t isrc = (ipri) * nseg;
	    ssize_t idst = buf.primary_entries.at(ipri).dst_segment * nseg;
	    assert((idst >= 0) && (idst+nseg <= ndst));
	    memcpy(output_flattened.data + idst, src.data + isrc, nseg * sizeof(float));
	}
	
	for (ssize_t isec = 0; isec < nsec; isec++) {
	    ssize_t isrc = (npri+isec) * nseg;
	    ssize_t idst = buf.secondary_entries.at(isec).dst_segment * nseg;
	    assert((idst >= 0) && (idst+nseg <= ndst));
	    memcpy(output_flattened.data + idst, src.data + isrc, nseg * sizeof(float));
	}
    }
}


// (stage0 iobufs) -> (stage1 iobufs), based on CacheLineRingbuf::stage0_stage1_copies
void ReferenceDedisperser::_proper_s0_to_s1()
{
    ssize_t nseg = plan->nelts_per_segment;
    ssize_t nsrc = plan->stage0_iobuf_segments_per_beam * nseg;
    ssize_t ndst = plan->stage1_iobuf_segments_per_beam * nseg;
    
    assert(intermediate_flattened.shape_equals({nsrc}));  // check that _allocate_intermediate_arrays() was called
    assert(output_flattened.shape_equals({ndst}));        // check that _allocate_output_arrays() was called

    for (const CacheLineRingbuf::PrimaryEntry &e: plan->cache_line_ringbuf->stage0_stage1_copies) {
	ssize_t isrc = e.src_segment * nseg;
	ssize_t idst = e.dst_segment * nseg;

	assert((isrc >= 0) && (isrc+nseg <= nsrc));
	assert((idst >= 0) && (idst+nseg <= ndst));

	memcpy(output_flattened.data + idst, intermediate_flattened.data + isrc, nseg * sizeof(float));
    }
}


// (stage1 iobufs) -> (stage1 iobufs), based on CacheLineRingbuf::stage0_stage1_copies
void ReferenceDedisperser::_proper_s1_to_s1()
{
    ssize_t nseg = plan->nelts_per_segment;
    ssize_t nelts = plan->stage1_iobuf_segments_per_beam * nseg;
    assert(output_flattened.shape_equals({nelts}));   // check that _allocate_output_arrays() was called

    for (const CacheLineRingbuf::PrimaryEntry &e: plan->cache_line_ringbuf->stage1_stage1_copies) {
	ssize_t isrc = e.src_segment * nseg;
	ssize_t idst = e.dst_segment * nseg;

	assert((isrc >= 0) && (isrc+nseg <= nelts));
	assert((idst >= 0) && (idst+nseg <= nelts));
	assert(isrc != idst);

	memcpy(output_flattened.data + idst, output_flattened.data + isrc, nseg * sizeof(float));
    }
}


// Helper for ReferenceDedisperser::print().
ssize_t ReferenceDedisperser::_print_array(const string &name, const Array<float> &arr,
					   ostream &os, int indent, bool active_beams_only) const
{
    int nbeams_active = config.beams_per_batch * config.num_active_batches;
    int nbeams = active_beams_only ? nbeams_active : config.beams_per_gpu;    
    ssize_t nbytes = nbeams * arr.size * plan->uncompressed_dtype_size;
    
    os << Indent(indent) << name << ": shape=" << gputils::tuple_str(arr.ndim, arr.shape)
       << ", nbytes=" << gputils::nbytes_to_str(nbytes) << " (all beams)" << endl;

    return nbytes;
}


// Helper for ReferenceDedisperser::print().
ssize_t ReferenceDedisperser::_print_ringbuf(const string &name, const vector<Array<float>> &arr_vec,
					     ostream &os, int indent, bool active_beams_only) const
{
    if (arr_vec.size() == 0)
	return 0;

    int active_beams = config.beams_per_batch * config.num_active_batches;
    int nbeams = active_beams_only ? active_beams : config.beams_per_gpu;
    ssize_t nbytes_tot = 0;
    
    os << Indent(indent) << name << endl;

    for (unsigned int i = 0; i < arr_vec.size(); i++) {
	ssize_t nelts = arr_vec[i].size;
	ssize_t nsegments = xdiv(nelts, plan->nelts_per_segment);
	ssize_t nbytes = nsegments * plan->bytes_per_compressed_segment * nbeams;
	nbytes_tot += nbytes;

	if (!active_beams_only) {
	    os << Indent(indent+4) << i << ": nbytes="
	       << gputils::nbytes_to_str(nbytes) << " (all beams)\n";
	}
    }

    os << Indent(indent+4) << "nbytes_tot=" << gputils::nbytes_to_str(nbytes_tot) << endl;
    return nbytes_tot;
}


void ReferenceDedisperser::print(ostream &os, int indent) const
{
    cout << Indent(indent) << "DedispersionConfig" << endl;
    this->config.print(os, indent+4);

    print_kv("sophistication", sophistication, os, indent);
    print_kv("input_rank", input_rank, os, indent);
    print_kv("input_nt", input_nt, os, indent);
    print_kv("output_ntrees", output_ntrees, os, indent);
    print_kv("nds", nds, os, indent);
    print_kv("pos", pos, os, indent);

    if (first_trees.size() > 0) {
	// FIXME should include lagged_downsampler rstate
	ssize_t nelts_per_beam = 0;
	for (const FirstTree &t: first_trees)
	    nelts_per_beam += t.rstate.size;
	
	ssize_t nbytes = nelts_per_beam * config.beams_per_gpu * plan->uncompressed_dtype_size;
	print_kv_nbytes("stage0 rstate", nbytes, os, indent);
    }

    if (second_trees.size() > 0) {
	ssize_t nelts_per_beam = 0;
	for (const SecondTree &t: second_trees)
	    nelts_per_beam += t.rstate.size;
	
	ssize_t nbytes = nelts_per_beam * config.beams_per_gpu * plan->uncompressed_dtype_size;
	print_kv_nbytes("stage1 rstate [non-residual]", nbytes, os, indent);
    }

    if (residual_lagbufs.size() > 0) {
	ssize_t nelts_per_beam = 0;
	for (const auto &lagbuf: residual_lagbufs)
	    nelts_per_beam += lagbuf->nrstate;
	
	ssize_t nbytes = nelts_per_beam * config.beams_per_gpu * plan->uncompressed_dtype_size;
	print_kv_nbytes("stage1 rstate [residual]", nbytes, os, indent);
    }	
    
    this->_print_array("intermediate_flattened [stage0 iobuf]",
		       intermediate_flattened, os, indent, true);
    
    this->_print_array("output_flattened [stage1 iobuf]",
		       output_flattened, os, indent, true);

    this->_print_ringbuf("proper_ringbufs", proper_ringbufs, os, indent, false);
    this->_print_ringbuf("staging_inbufs", staging_inbufs, os, indent, true);
    this->_print_ringbuf("staging_outbufs", staging_outbufs, os, indent, true);
}


// -------------------------------------------------------------------------------------------------
//
// ReferenceDedisperser::SimpleTree


ReferenceDedisperser::SimpleTree::SimpleTree(const DedispersionPlan::Stage1Tree &st1) :
    is_downsampled(st1.ds_level > 0),
    output_rank(st1.rank0 + st1.rank1_trigger),
    nt_ds(st1.nt_ds)
{
    int rtree_rank = is_downsampled ? (output_rank+1) : output_rank;

    this->rtree = make_shared<ReferenceTree> (rtree_rank, nt_ds);
    this->rstate = Array<float> ({rtree->nrstate}, af_uhost | af_zero);
    this->scratch = Array<float> ({rtree->nscratch}, af_uhost | af_zero);
    this->iobuf = Array<float> ({pow2(rtree_rank), nt_ds}, af_uhost | af_zero);
}


void ReferenceDedisperser::SimpleTree::dedisperse(const gputils::Array<float> &in, gputils::Array<float> &out)
{
    assert(in.ndim == 2);
    assert(in.shape[0] >= pow2(rtree->rank));
    assert(in.shape[1] == nt_ds);
    assert(in.strides[1] == 1);
    assert(out.shape_equals({pow2(output_rank), nt_ds}));

    Array<float> s = in.slice(0, 0, pow2(rtree->rank));
    iobuf.fill(s);

    rtree->dedisperse(iobuf, rstate.data, scratch.data);

    if (is_downsampled)
	reference_extract_odd_channels(iobuf, out);
    else
	out.fill(iobuf);
}


// -------------------------------------------------------------------------------------------------
//
// ReferenceDedisperser::FirstTree


ReferenceDedisperser::FirstTree::FirstTree(const DedispersionPlan::Stage0Tree &st0) :
    is_downsampled(st0.ds_level > 0),
    output_rank0(st0.rank0),
    output_rank1(st0.rank1),
    nt_ds(st0.nt_ds)
{
    this->rtree = make_shared<ReferenceTree> (output_rank0, nt_ds);
    this->rstate = Array<float> ({pow2(output_rank1) * rtree->nrstate}, af_uhost | af_zero);
    this->scratch = Array<float> ({rtree->nscratch}, af_uhost | af_zero);
}


// Output goes to FirstTree::iobuf. Modifies input if is_downsampled==true!
void ReferenceDedisperser::FirstTree::dedisperse(gputils::Array<float> &arr)
{
    int output_rank = output_rank0 + output_rank1;
    assert(arr.shape_equals({ pow2(output_rank), nt_ds }));

    int ndm0 = pow2(output_rank0);
    float *rp = rstate.data;
    
    for (int i = 0; i < pow2(output_rank1); i++) {
	Array<float> s = arr.slice(0, i*ndm0, (i+1)*ndm0);
	rtree->dedisperse(s, rp, scratch.data);
	rp += rtree->nrstate;
    }
}


// -------------------------------------------------------------------------------------------------
//
// ReferenceDedisperser::SecondTree


ReferenceDedisperser::SecondTree::SecondTree(const DedispersionPlan::Stage1Tree &st1) :
    rank0(st1.rank0),
    rank1(st1.rank1_trigger),
    nt_ds(st1.nt_ds)
{
    this->rtree = make_shared<ReferenceTree> (rank1, nt_ds);
    this->rstate = Array<float> ({ pow2(rank0) * rtree->nrstate }, af_uhost | af_zero);
    this->scratch = Array<float> ({ rtree->nscratch }, af_uhost | af_zero);
}


void ReferenceDedisperser::SecondTree::dedisperse(Array<float> &arr)
{
    assert(arr.shape_equals({ pow2(rank0+rank1), nt_ds }));
    assert(arr.strides[1] == 1);

    float *rp = rstate.data;
    int s = arr.strides[0];
    int nj = pow2(rank0);
    
    for (int j = 0; j < nj; j++) {
	rtree->dedisperse(arr.data + j*s, nj*s, rp, scratch.data);
	rp += rtree->nrstate;
    }
}


}  // namespace pirate
