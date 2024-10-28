#include "../include/pirate/internals/ReferenceDedisperser.hpp"

#include "../include/pirate/constants.hpp"
#include "../include/pirate/internals/inlines.hpp"
#include "../include/pirate/internals/utils.hpp"


using namespace std;
using namespace gputils;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// -------------------------------------------------------------------------------------------------


// Returns tree as shape (1, 2^rank, nt) array.
static Array<float> stage0_tree_view(const shared_ptr<DedispersionPlan> &plan, const Array<float> &stage0_iobuf, int ids)
{
    long S = plan->nelts_per_segment;
    assert((ids >= 0) && (ids < plan->stage0_trees.size()));
    assert(stage0_iobuf.shape_equals({plan->stage0_iobuf_segments_per_beam * S}));
    
    const DedispersionPlan::Stage0Tree &st0 = plan->stage0_trees.at(ids);
    long start = st0.iobuf_base_segment * S;
    long nelts = st0.segments_per_beam * S;
    long rank = st0.rank0 + st0.rank1;
    long nt_ds = st0.nt_ds;
    
    Array<float> view = stage0_iobuf.slice(0, start, start+nelts);
    return view.reshape_ref({1, pow2(rank), nt_ds});
}


// Returns tree as shape (1, 2^rank, nt) array.
static Array<float> stage1_tree_view(const shared_ptr<DedispersionPlan> &plan, const Array<float> &stage1_iobuf, int itree)
{
    long S = plan->nelts_per_segment;
    assert((itree >= 0) && (itree < plan->stage1_trees.size()));
    assert(stage1_iobuf.shape_equals({plan->stage1_iobuf_segments_per_beam * S}));
    
    const DedispersionPlan::Stage1Tree &st1 = plan->stage1_trees.at(itree);
    long start = st1.iobuf_base_segment * S;
    long nelts = st1.segments_per_beam * S;
    long rank = st1.rank0 + st1.rank1_trigger;
    long nt_ds = st1.nt_ds;
    
    Array<float> view = stage1_iobuf.slice(0, start, start+nelts);
    return view.reshape_ref({1, pow2(rank), nt_ds});
}


// -------------------------------------------------------------------------------------------------


static shared_ptr<ReferenceLaggedDownsampler>
make_reference_lagged_downsampler(const shared_ptr<DedispersionPlan> &plan)
{
    // If (nds == 1), use a placeholder value of s.
    long nds = plan->config.num_downsampling_levels;
    long s = (nds > 1) ? (plan->stage0_trees.at(1).rank0 + 1) : (plan->stage0_trees.at(0).rank0);
    long r = plan->config.tree_rank;
	
    ReferenceLaggedDownsampler::Params ld_params;
    ld_params.small_input_rank = s;
    ld_params.large_input_rank = r;
    ld_params.num_downsampling_levels = nds - 1;   // note (-1) here!
    ld_params.nbeams = 1;
    ld_params.ntime = plan->config.time_samples_per_chunk;
    
    return make_shared<ReferenceLaggedDownsampler> (ld_params);
}


static void apply_lagged_downsampler(
    const shared_ptr<DedispersionPlan> &plan,
    const shared_ptr<ReferenceLaggedDownsampler> &lds,
    const Array<float> &in, Array<float> &stage0_iobuf)
{
    long rank = plan->config.tree_rank;
    long nt = plan->config.time_samples_per_chunk;
    int nds = plan->config.num_downsampling_levels;
    
    Array<float> in_reshaped = in.reshape_ref({1, pow2(rank), nt});
    Array<float> tree0 = stage0_tree_view(plan, stage0_iobuf, 0);
    tree0.fill(in_reshaped);
    
    vector<Array<float>> downsampled_trees(nds-1);
    
    for (long ids = 1; ids < nds; ids++)
	downsampled_trees.at(ids-1) = stage0_tree_view(plan, stage0_iobuf, ids);

    lds->apply(tree0, downsampled_trees);
}


// -------------------------------------------------------------------------------------------------


static vector<shared_ptr<ReferenceDedispersionKernel>>
make_stage0_dedispersion_kernels(const shared_ptr<DedispersionPlan> &plan)
{
    int nds = plan->stage0_trees.size();
    vector<shared_ptr<ReferenceDedispersionKernel>> kernels(nds);
    
    for (int ids = 0; ids < nds; ids++) {
	const DedispersionPlan::Stage0Tree &st0 = plan->stage0_trees.at(ids);
	
	ReferenceDedispersionKernel::Params params;
	params.rank = st0.rank0;
	params.ntime = st0.nt_ds;
	params.nambient = pow2(st0.rank1);
	params.nbeams = 1;
	
        kernels.at(ids) = make_shared<ReferenceDedispersionKernel> (params);
    }

    return kernels;
}


static void apply_stage0_dedispersion_kernels(
    const shared_ptr<DedispersionPlan> &plan,
    const vector<shared_ptr<ReferenceDedispersionKernel>> &stage0_kernels,
    Array<float> &stage0_iobuf)
{
    int nds = plan->stage0_trees.size();
    assert(stage0_kernels.size() == nds);
    
    for (int ids = 0; ids < nds; ids++) {
	const DedispersionPlan::Stage0Tree &st0 = plan->stage0_trees.at(ids);
	Array<float> buf = stage0_tree_view(plan, stage0_iobuf, ids);             // shape (1, 2^rank, nt_ds)
	buf = buf.reshape_ref({1, pow2(st0.rank1), pow2(st0.rank0), st0.nt_ds});  // shape (1, 2^rank1, 2^rank0, nt_ds)
	stage0_kernels.at(ids)->apply(buf);
    }
}


// -------------------------------------------------------------------------------------------------


static vector<shared_ptr<ReferenceDedispersionKernel>>
make_stage1_dedispersion_kernels(const shared_ptr<DedispersionPlan> &plan)
{
    int output_ntrees = plan->stage1_trees.size();
    vector<shared_ptr<ReferenceDedispersionKernel>> kernels(output_ntrees);
    
    for (int itree = 0; itree < output_ntrees; itree++) {
	const DedispersionPlan::Stage1Tree &st1 = plan->stage1_trees.at(itree);

	ReferenceDedispersionKernel::Params params;
	params.rank = st1.rank1_trigger;
	params.ntime = st1.nt_ds;
	params.nambient = pow2(st1.rank0);
	params.nbeams = 1;
	params.apply_input_residual_lags = true;
	params.is_downsampled_tree = (st1.ds_level > 0);
	params.nelts_per_segment = plan->nelts_per_segment;
	
	kernels.at(itree) = make_shared<ReferenceDedispersionKernel> (params);
    }

    return kernels;
}


static void apply_stage1_dedispersion_kernels(
    const shared_ptr<DedispersionPlan> &plan,
    const vector<shared_ptr<ReferenceDedispersionKernel>> &stage1_kernels,
    Array<float> &stage1_iobuf,
    vector<Array<float>> &output_arrays)
{
    int output_ntrees = plan->stage1_trees.size();
    assert(stage1_kernels.size() == output_ntrees);
    assert(output_arrays.size() == output_ntrees);
    
    for (int itree = 0; itree < output_ntrees; itree++) {
	const DedispersionPlan::Stage1Tree &st1 = plan->stage1_trees.at(itree);
	long rank0 = st1.rank0;
	long rank1 = st1.rank1_trigger;

	Array<float> buf1 = stage1_tree_view(plan, stage1_iobuf, itree);          // shape (1, 2^rank, nt_ds)
	
	// Stage-1 dedispersion.
	buf1 = buf1.reshape_ref({1, pow2(rank1), pow2(rank0), st1.nt_ds});       // shape (1, 2^rank1, 2^rank0, nt_ds)
	Array<float> buf2 = buf1.transpose({0,2,1,3});                           // shape (1, 2^rank0, 2^rank1, nt_ds)
	stage1_kernels.at(itree)->apply(buf2);

	// Copy stage1_iobuf -> output_arrays.
	buf1 = buf1.reshape_ref({pow2(rank0+rank1), st1.nt_ds});                 // shape (2^rank, nt_ds)
	output_arrays.at(itree).fill(buf1);
    }
}


// -------------------------------------------------------------------------------------------------
//
// ReferenceDedisperserBase


// Helper for ReferenceDedisperserBase constructor.
// Prevents constructor from segfaulting, if invoked with empty shared_ptr.
static DedispersionPlan *deref(const shared_ptr<DedispersionPlan> &p)
{
    if (!p)
	throw runtime_error("ReferenceDedisperser constructor called with empty shared_ptr");
    return p.get();
}


ReferenceDedisperserBase::ReferenceDedisperserBase(const shared_ptr<DedispersionPlan> &plan_, int sophistication_) :
    plan(plan_),
    config(deref(plan_)->config),
    sophistication(sophistication_)
{
    // FIXME relax these constraints
    assert(config.dtype == "float32");
    assert(config.beams_per_gpu == 1);
    assert(config.beams_per_batch == 1);
    assert(config.num_active_batches == 1);

    // FIXME do I need these asserts?
    assert(plan->nelts_per_segment == xdiv(constants::bytes_per_gpu_cache_line, 4));
    assert(plan->nbytes_per_segment == constants::bytes_per_gpu_cache_line);
    
    this->input_rank = config.tree_rank;
    this->input_nt = config.time_samples_per_chunk;
    this->output_ntrees = plan->stage1_trees.size();
    this->nds = plan->stage0_trees.size();
    this->nelts_per_segment = plan->nelts_per_segment;
    this->output_arrays.resize(output_ntrees);

    // Allocate output_arrays.
    for (int itree = 0; itree < output_ntrees; itree++) {
	const DedispersionPlan::Stage1Tree &st1 = plan->stage1_trees.at(itree);
	ssize_t rank = st1.rank0 + st1.rank1_trigger;
	this->output_arrays.at(itree) = Array<float>({pow2(rank), st1.nt_ds}, af_uhost | af_zero);
    }
}


void ReferenceDedisperserBase::dedisperse(const gputils::Array<float> &in)
{
    assert(in.shape_equals({pow2(input_rank), input_nt}));
    assert(in.is_fully_contiguous());   // probably not necessary
    assert(in.on_host());
    
    this->_dedisperse(in);
    this->pos++;
}


// Static member function
std::shared_ptr<ReferenceDedisperserBase> ReferenceDedisperserBase::make(const std::shared_ptr<DedispersionPlan> &plan_, int sophistication)
{
    if (sophistication == 0)
	return make_shared<ReferenceDedisperser0> (plan_);
    else if (sophistication == 1)
	return make_shared<ReferenceDedisperser1> (plan_);
    else if (sophistication == 2)
	return make_shared<ReferenceDedisperser2> (plan_);
    throw runtime_error("ReferenceDedisperserBase::make(): invalid value of 'sophistication' parameter");
}


// -------------------------------------------------------------------------------------------------
//
// ReferenceDedisperser0


ReferenceDedisperser0::ReferenceDedisperser0(const shared_ptr<DedispersionPlan> &plan_) :
    ReferenceDedisperserBase(plan_, 0)
{
    this->downsampled_inputs.resize(nds);
    this->dedispersion_buffers.resize(output_ntrees);
    this->trees.resize(output_ntrees);

    for (int ids = 0; ids < nds; ids++) {
	long nfreq = pow2(input_rank);
	long nt_ds = xdiv(input_nt, pow2(ids));
	downsampled_inputs.at(ids) = Array<float> ({nfreq,nt_ds}, af_uhost | af_zero);
    }

    for (int itree = 0; itree < output_ntrees; itree++) {
	const DedispersionPlan::Stage1Tree &st1 = plan->stage1_trees.at(itree);
	long ids = st1.ds_level;
	long rank = st1.rank0 + st1.rank1_trigger + (ids ? 1 : 0);

	dedispersion_buffers.at(itree) = Array<float> ({pow2(rank), st1.nt_ds}, af_uhost | af_zero);
	trees.at(itree) = ReferenceTree::make({ pow2(rank), st1.nt_ds });
    }
}


// virtual override
void ReferenceDedisperser0::_dedisperse(const gputils::Array<float> &in)
{
    downsampled_inputs.at(0).fill(in);
    for (int ids = 1; ids < nds; ids++)
	reference_downsample_time(downsampled_inputs.at(ids-1), downsampled_inputs.at(ids), false);   // normalize=false, i.e. no factor 0.5

    for (int itree = 0; itree < output_ntrees; itree++) {
	const DedispersionPlan::Stage1Tree &st1 = plan->stage1_trees.at(itree);
	long ids = st1.ds_level;
	long rank = st1.rank0 + st1.rank1_trigger + (ids ? 1 : 0);

	Array<float> dst = dedispersion_buffers.at(itree);
	Array<float> src = downsampled_inputs.at(ids).slice(0, 0, pow2(rank));

	dst.fill(src);
	trees.at(itree)->dedisperse(dst);

	if (ids == 0)
	    output_arrays.at(itree).fill(dedispersion_buffers.at(itree));
	else
	    reference_extract_odd_channels(dedispersion_buffers.at(itree), output_arrays.at(itree));
    }
}


// -------------------------------------------------------------------------------------------------
//
// ReferenceDedisperser1


ReferenceDedisperser1::ReferenceDedisperser1(const shared_ptr<DedispersionPlan> &plan_) :
    ReferenceDedisperserBase(plan_, 1)
{
    long S = nelts_per_segment;
    long nseg0 = plan->stage0_iobuf_segments_per_beam;
    long nseg1 = plan->stage1_iobuf_segments_per_beam;

    this->stage0_iobuf = Array<float> ({nseg0*S}, af_uhost | af_zero);
    this->stage1_iobuf = Array<float> ({nseg1*S}, af_uhost | af_zero);
    this->lagged_downsampler = make_reference_lagged_downsampler(plan);
    this->stage0_kernels = make_stage0_dedispersion_kernels(plan);
    this->stage1_kernels = make_stage1_dedispersion_kernels(plan);
    
    this->stage1_lagbufs.resize(output_ntrees);

    for (int itree = 0; itree < output_ntrees; itree++) {
	const DedispersionPlan::Stage1Tree &st1 = plan->stage1_trees.at(itree);
	int rank0 = st1.rank0;
	int rank1 = st1.rank1_trigger;
	bool is_downsampled = (st1.ds_level > 0);

	Array<int> lags({1, pow2(rank0+rank1)}, af_uhost);

	for (ssize_t i1 = 0; i1 < pow2(rank1); i1++) {
	    for (ssize_t i0 = 0; i0 < pow2(rank0); i0++) {
		int row = i1 * pow2(rank0) + i0;
		int lag = rb_lag(i1, i0, rank0, rank1, is_downsampled);
		int segment_lag = lag / S;   // round down
		lags.data[row] = segment_lag * S;
	    }
	}
	
	stage1_lagbufs.at(itree) = make_shared<ReferenceLagbuf> (lags, st1.nt_ds);
    }
}


// virtual override
void ReferenceDedisperser1::_dedisperse(const gputils::Array<float> &in)
{
    // Copy input data to stage0_iobuf, and apply lagged downsampling kernel.
    apply_lagged_downsampler(plan, lagged_downsampler, in, stage0_iobuf);

    // Stage-0 dedispersion.
    apply_stage0_dedispersion_kernels(plan, stage0_kernels, stage0_iobuf);

    // Copy stage0_iobuf -> stage1_iobuf, and apply lags.
    for (int itree = 0; itree < output_ntrees; itree++) {
	const DedispersionPlan::Stage1Tree &st1 = plan->stage1_trees.at(itree);
	long rank0 = st1.rank0;
	long rank1 = st1.rank1_trigger;

	Array<float> buf0 = stage0_tree_view(plan, stage0_iobuf, st1.ds_level);   // shape (1, 2^rank_ambient, nt_ds)
	buf0 = buf0.slice(1, 0, pow2(rank0+rank1));                               // shape (1, 2^rank, nt_ds)

	// Copy stage0_iobuf -> stage1_iobuf
	Array<float> buf1 = stage1_tree_view(plan, stage1_iobuf, itree);          // shape (1, 2^rank, nt_ds)
	buf1.fill(buf0);
	
	// Apply lags.
	stage1_lagbufs.at(itree)->apply_lags(buf1);
    }

    // Stage-1 dedispersion, and copy stage1_iobuf -> this->output_arrays.
    apply_stage1_dedispersion_kernels(plan, stage1_kernels, stage1_iobuf, output_arrays);;
}


// -------------------------------------------------------------------------------------------------


static float *get_rb(const Array<float> &gpu_ringbuf, const uint *rb_locs, uint pos, uint nelts_per_segment)
{
    uint rb_offset = rb_locs[0];  // in segments, not bytes
    uint rb_phase = rb_locs[1];   // index of (time chunk, beam) pair, relative to current pair
    uint rb_len = rb_locs[2];     // number of (time chunk, beam) pairs in ringbuf (same as Ringbuf::rb_len)
    uint rb_nseg = rb_locs[3];    // number of segments per (time chunk, beam) (same as Ringbuf::nseg_per_beam)

    // FIXME assumes nbeams=1
    uint i = (pos + rb_phase) % rb_len;
    long s = rb_offset + (i * rb_nseg);
    
    return gpu_ringbuf.data + (s * nelts_per_segment);
}


ReferenceDedisperser2::ReferenceDedisperser2(const shared_ptr<DedispersionPlan> &plan_) :
    ReferenceDedisperserBase(plan_, 2)
{
    long S = nelts_per_segment;
    long nseg0 = plan->stage0_iobuf_segments_per_beam;
    long nseg1 = plan->stage1_iobuf_segments_per_beam;
    
    this->stage0_iobuf = Array<float> ({nseg0*S}, af_uhost | af_zero);
    this->stage1_iobuf = Array<float> ({nseg1*S}, af_uhost | af_zero);
    this->gpu_ringbuf = Array<float> ({plan->gmem_ringbuf_nseg * S}, af_uhost | af_zero);
    
    this->lagged_downsampler = make_reference_lagged_downsampler(plan);
    this->stage0_kernels = make_stage0_dedispersion_kernels(plan);
    this->stage1_kernels = make_stage1_dedispersion_kernels(plan);
}


void ReferenceDedisperser2::_dedisperse(const gputils::Array<float> &in)
{
    long S = nelts_per_segment;
    const uint *rb_locs0 = plan->stage0_rb_locs.data;
    const uint *rb_locs1 = plan->stage1_rb_locs.data;
	
    // Copy input data to stage0_iobuf, and apply lagged downsampling kernel.
    apply_lagged_downsampler(plan, lagged_downsampler, in, stage0_iobuf);

    // Stage-0 dedispersion.
    apply_stage0_dedispersion_kernels(plan, stage0_kernels, stage0_iobuf);

    // Copy stage0_iobuf -> gpu_ringbuf.

    for (int ids = 0; ids < nds; ids++) {
	const DedispersionPlan::Stage0Tree &st0 = plan->stage0_trees.at(ids);
	long nchan0 = pow2(st0.rank0);
	long nchan1 = pow2(st0.rank1);
	long nt_ds = st0.nt_ds;
	long ns = xdiv(nt_ds, S);

	Array<float> buf = stage0_tree_view(plan, stage0_iobuf, ids);   // shape (1, 2^rank, nt_ds)
	
	for (long s = 0; s < ns; s++) {
	    for (long i1 = 0; i1 < nchan1; i1++) {
		for (long i0 = 0; i0 < nchan0; i0++) {
		    long iseg0 = st0.iobuf_base_segment + s*nchan1*nchan0 + i1*nchan0 + i0;
		    float *dst = get_rb(this->gpu_ringbuf, rb_locs0 + 4*iseg0, this->pos, S);
		    float *src = buf.data + (i1*nchan0+i0)*nt_ds + s*S;
		    memcpy(dst, src, S * sizeof(float));
		}
	    }
	}
    }

    // Copy gpu_ringbuf -> stage1_iobuf.

    for (int itree = 0; itree < output_ntrees; itree++) {
	const DedispersionPlan::Stage1Tree &st1 = plan->stage1_trees.at(itree);
	long nchan0 = pow2(st1.rank0);
	long nchan1 = pow2(st1.rank1_trigger);
	long nt_ds = st1.nt_ds;
	long ns = xdiv(st1.nt_ds, S);

	Array<float> buf = stage1_tree_view(plan, stage1_iobuf, itree);   // shape (1, 2^rank, nt_ds)
	
	for (long s = 0; s < ns; s++) {
	    for (long i0 = 0; i0 < nchan0; i0++) {
		for (long i1 = 0; i1 < nchan1; i1++) {
		    long iseg1 = st1.iobuf_base_segment + s*nchan1*nchan0 + i0*nchan1 + i1;
		    float *src = get_rb(this->gpu_ringbuf, rb_locs1 + 4*iseg1, this->pos, S);
		    float *dst = buf.data + (i1*nchan0+i0)*nt_ds + s*S;
		    memcpy(dst, src, S * sizeof(float));
		}
	    }
	}
    }
    
    // Stage-1 dedispersion, and copy stage1_iobuf -> this->output_arrays.
    apply_stage1_dedispersion_kernels(plan, stage1_kernels, stage1_iobuf, output_arrays);
}


}  // namespace pirate
