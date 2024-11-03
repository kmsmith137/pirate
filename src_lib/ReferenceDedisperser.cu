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
//
// Stage0Buffers


// Helper class used in ReferenceDedisperser1, ReferenceDedisperser2.
struct Stage0Buffers
{
    shared_ptr<DedispersionPlan> plan;
    
    long nds = 0;                  // same as plan->stage0_trees.size()
    long nseg = 0;                 // same as plan->stage0_total_segments_per_beam
    long nelts_per_segment = 0;    // same as plan->nelts_per_segment
    long beams_per_batch = 0;      // same as plan->config.beams_per_batch
    long total_beams = 0;          // same as plan->config.beams_per_gpu
    long nbatches = 0;             // same as (total_beams / beams_per_batch)
    
    Array<float> flat_buf;         // shape (beams_per_batch, nseg * nelts_per_segment)
    vector<Array<float>> dd_bufs;  // length nds, inner shape is (beams_per_batch, pow2(st0.rank), nt_ds)
    vector<Array<float>> ds_bufs;  // length (nds-1), same as dd_bufs[1:]

    vector<shared_ptr<ReferenceLaggedDownsampler>> lds_kernels;   // length (nbatches)
    vector<shared_ptr<ReferenceDedispersionKernel>> dd_kernels;   // length (nds)

    Stage0Buffers(const shared_ptr<DedispersionPlan> &plan_) : plan(plan_)
    {
	this->nds = plan->config.num_downsampling_levels;
	this->nseg = plan->stage0_total_segments_per_beam;
	this->nelts_per_segment = plan->nelts_per_segment;
	this->beams_per_batch = plan->config.beams_per_batch;
	this->total_beams = plan->config.beams_per_gpu;
	this->nbatches = xdiv(total_beams, beams_per_batch);

	this->dd_bufs.resize(nds);
	this->ds_bufs.resize(nds-1);
	this->lds_kernels.resize(nbatches);
	this->dd_kernels.resize(nds);
	
	// Allocate buffers.

	this->flat_buf = Array<float> ({beams_per_batch, nseg * nelts_per_segment}, af_uhost | af_zero);
	long pos = 0;
	
	for (long ids = 0; ids < nds; ids++) {
	    const DedispersionPlan::Stage0Tree &st0 = plan->stage0_trees.at(ids);
	    assert(pos == st0.base_segment * nelts_per_segment);
	    
	    long nelts = st0.segments_per_beam * nelts_per_segment;
	    long rank = st0.rank0 + st0.rank1;
	    long nt_ds = st0.nt_ds;

	    assert(nelts == pow2(rank) * nt_ds);
	    Array<float> view = flat_buf.slice(1, pos, pos+nelts);
	    view = view.reshape_ref({ beams_per_batch, pow2(rank), nt_ds });

	    dd_bufs.at(ids) = view;
	    
	    if (ids > 0)
		ds_bufs.at(ids-1) = view;
	    
	    pos += nelts;
	}
	
	// LaggedDownsampler kernels.

	if (nds > 1) {
	    ReferenceLaggedDownsampler::Params ld_params;
	    ld_params.small_input_rank = plan->stage0_trees.at(1).rank0 + 1;
	    ld_params.large_input_rank = plan->config.tree_rank;
	    ld_params.num_downsampling_levels = nds - 1;   // note (-1) here!
	    ld_params.nbeams = beams_per_batch;
	    ld_params.ntime = plan->config.time_samples_per_chunk;

	    for (long b = 0; b < nbatches; b++)
		this->lds_kernels.at(b) = make_shared<ReferenceLaggedDownsampler> (ld_params);
	}

	// Dedispersion kernels.

	for (long ids = 0; ids < nds; ids++) {
	    const DedispersionPlan::Stage0Tree &st0 = plan->stage0_trees.at(ids);
	    
	    ReferenceDedispersionKernel::Params params;
	    params.dtype = plan->config.dtype;
	    params.rank = st0.rank0;
	    params.nambient = pow2(st0.rank1);
	    params.total_beams = plan->config.beams_per_gpu;
	    params.beams_per_kernel_launch = plan->config.beams_per_batch;
	    params.ntime = st0.nt_ds;
	    params.input_is_ringbuf = false;
	    params.output_is_ringbuf = false;
	    params.apply_input_residual_lags = false;
	    params.input_is_downsampled_tree = (ids > 0);
	    params.nelts_per_segment = plan->nelts_per_segment;
	    
	    dd_kernels.at(ids) = make_shared<ReferenceDedispersionKernel> (params);
	}
    }

    void apply_lagged_downsampler(long ibeam)
    {
	if (nds > 1) {
	    long b = xdiv(ibeam, beams_per_batch);
	    lds_kernels.at(b)->apply(dd_bufs.at(0), ds_bufs);
	}
    }
    
    void apply_dedispersion_kernels(long itime, long ibeam)
    {
	for (long ids = 0; ids < nds; ids++) {
	    const DedispersionPlan::Stage0Tree &st0 = plan->stage0_trees.at(ids);
	    Array<float> buf = dd_bufs.at(ids);
	    buf = buf.reshape_ref({beams_per_batch, pow2(st0.rank1), pow2(st0.rank0), st0.nt_ds});  // shape (1, 2^rank1, 2^rank0, nt_ds)
	    dd_kernels.at(ids)->apply(buf, buf, itime, ibeam);
	}
    }
};


// -------------------------------------------------------------------------------------------------
//
// Stage1Buffers


// Helper class used in ReferenceDedisperser1, ReferenceDedisperser2.
struct Stage1Buffers
{
    shared_ptr<DedispersionPlan> plan;

    long nout = 0;                // same as plan->stage1_trees.size()
    long nseg = 0;                // same as plan->stage1_total_segments_per_beam
    long nelts_per_segment = 0;   // same as plan->nelts_per_segment
    long beams_per_batch = 0;     // same as plan->config.beams_per_batch
    long total_beams = 0;         // same as plan->config.beams_per_gpu
    long nbatches = 0;            // same as (total_beams / beams_per_batch)

    Array<float> flat_buf;         // shape (beams_per_batch, nseg * nelts_per_segment)
    vector<Array<float>> dd_bufs;  // length nout, inner shape is (beams_per_batch, pow2(st1.rank), nt_ds)
    vector<shared_ptr<ReferenceDedispersionKernel>> dd_kernels;   // length (nout)

    
    Stage1Buffers(const shared_ptr<DedispersionPlan> &plan_) : plan(plan_)
    {
	this->nout = plan->stage1_trees.size();
	this->nseg = plan->stage1_total_segments_per_beam;
	this->nelts_per_segment = plan->nelts_per_segment;
	this->beams_per_batch = plan->config.beams_per_batch;
	this->total_beams = plan->config.beams_per_gpu;
	this->nbatches = xdiv(total_beams, beams_per_batch);

	this->dd_bufs.resize(nout);
	this->dd_kernels.resize(nout);
	
	// Allocate buffers.

	this->flat_buf = Array<float> ({beams_per_batch, nseg * nelts_per_segment}, af_uhost | af_zero);
	long pos = 0;

	for (long iout = 0; iout < nout; iout++) {
	    const DedispersionPlan::Stage1Tree &st1 = plan->stage1_trees.at(iout);
	    assert(pos == st1.base_segment * nelts_per_segment);
	    
	    long nelts = st1.segments_per_beam * nelts_per_segment;
	    long rank = st1.rank0 + st1.rank1_trigger;
	    long nt_ds = st1.nt_ds;

	    assert(nelts == pow2(rank) * nt_ds);
	    Array<float> view = flat_buf.slice(1, pos, pos+nelts);
	    view = view.reshape_ref({ beams_per_batch, pow2(rank), nt_ds });

	    dd_bufs.at(iout) = view;
	    pos += nelts;
	}

	// Dedispersion kernels.

	for (long iout = 0; iout < nout; iout++) {
	    const DedispersionPlan::Stage1Tree &st1 = plan->stage1_trees.at(iout);
	    
	    ReferenceDedispersionKernel::Params params;		
	    params.dtype = plan->config.dtype;
	    params.rank = st1.rank1_trigger;
	    params.nambient = pow2(st1.rank0);
	    params.total_beams = total_beams;
	    params.beams_per_kernel_launch = beams_per_batch;
	    params.ntime = st1.nt_ds;
	    params.input_is_ringbuf = false;
	    params.output_is_ringbuf = false;
	    params.apply_input_residual_lags = true;
	    params.input_is_downsampled_tree = (st1.ds_level > 0);
	    params.nelts_per_segment = plan->nelts_per_segment;
	    
	    dd_kernels.at(iout) = make_shared<ReferenceDedispersionKernel> (params);
	}
    }

    void apply_dedispersion_kernels(long itime, long ibeam)
    {
	for (int iout = 0; iout < nout; iout++) {
	    const DedispersionPlan::Stage1Tree &st1 = plan->stage1_trees.at(iout);
	    long rank0 = st1.rank0;
	    long rank1 = st1.rank1_trigger;

	    Array<float> buf = dd_bufs.at(iout);  // shape (beams_per_batch, 2^(rank0+rank1), nt_ds)
	    buf = buf.reshape_ref({beams_per_batch, pow2(rank1), pow2(rank0), st1.nt_ds});
	    buf = buf.transpose({0,2,1,3});       // shape (beams_per_batch, 2^rank0, 2^rank1, nt_ds)
	    dd_kernels.at(iout)->apply(buf, buf, itime, ibeam);
	}
    }
};


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
    this->config_rank = config.tree_rank;
    this->config_ntime = config.time_samples_per_chunk;
    this->total_beams = config.beams_per_gpu;
    this->beams_per_batch = config.beams_per_batch;
    this->nbatches = xdiv(total_beams, beams_per_batch);

    this->nds = plan->stage0_trees.size();
    this->nout = plan->stage1_trees.size();
    this->nelts_per_segment = plan->nelts_per_segment;

    // Note: 'input_array' and 'output_arrays' are members of ReferenceDedisperserBase,
    // but are initialized by the subclass constructor.
}


void ReferenceDedisperserBase::check_iobuf_shapes()
{
    assert(input_array.shape_equals({ beams_per_batch, pow2(config_rank), config_ntime }));
    assert(output_arrays.size() == nout);

    for (long iout = 0; iout < nout; iout++) {
	const DedispersionPlan::Stage1Tree &st1 = plan->stage1_trees.at(iout);
	int rank = st1.rank0 + st1.rank1_trigger;
	assert(output_arrays.at(iout).shape_equals({ beams_per_batch, pow2(rank), st1.nt_ds }));
    }
}

void ReferenceDedisperserBase::dedisperse(long itime, long ibeam)
{
    assert(itime == expected_itime);
    assert(ibeam == expected_ibeam);
    
    this->_dedisperse(itime, ibeam);

    expected_ibeam += beams_per_batch;

    if (expected_ibeam >= total_beams) {
	expected_itime++;
	expected_ibeam = 0;
    }
}


// Note: ReferenceDedisperserBase::make() is defined at the end of the file.


// -------------------------------------------------------------------------------------------------
//
// Sophistication == 0:
//
//   - Uses one-stage dedispersion instead of two stages.
//   - In downsampled trees, compute twice as many DMs as necessary, then drop the bottom half.
//   - Each early trigger is computed in an independent tree, by disregarding some input channels.


struct ReferenceDedisperser0 : public ReferenceDedisperserBase
{
    ReferenceDedisperser0(const shared_ptr<DedispersionPlan> &plan);

    virtual void _dedisperse(long itime, long ibeam) override;

    // Step 1: downsample input array (straightforward downsample, not "lagged" downsample!)
    // Outer length is nds, inner shape is (beams_per_batch, 2^config_rank, input_nt / pow2(ids)).
    
    vector<Array<float>> downsampled_inputs;

    // Step 2: copy from 'downsampled_inputs' to 'dedispersion_buffers'.
    // In downsampled trees, we compute twice as many DMs as necessary, then drop the bottom half.
    // Each early trigger is computed in an independent tree, by disregarding some input channels.
    // Outer vector length is nout, inner shape is (beams_per_batch, 2^weird_rank, input_nt / pow2(ids)).
    //   where weird_rank = rank0 + rank1_trigger + (is_downsampled ? 1 : 0)
    
    vector<Array<float>> dedispersion_buffers;

    // Step 3: apply tree dedispersion (one-stage, not two-stage).
    // Vector length is (nbatches * nout).
    // Inner shape is (beams_per_batch, 2^weird_rank, input_nt / pow2(ids)).
    
    vector<shared_ptr<ReferenceTree>> trees;

    // Step 4: copy from 'dedispersion_buffers' to 'output_arrays'.
    // In downsampled trees, we compute twice as many DMs as necessary, then copy the bottom half.
    // Reminder: 'output_arrays' is a member of ReferenceDedisperserBase.
};


ReferenceDedisperser0::ReferenceDedisperser0(const shared_ptr<DedispersionPlan> &plan_) :
    ReferenceDedisperserBase(plan_, 0)
{    
    this->downsampled_inputs.resize(nds);
    this->dedispersion_buffers.resize(nout);
    this->trees.resize(nbatches * nout);    
    this->output_arrays.resize(nout);

    for (int ids = 0; ids < nds; ids++) {
	long nt_ds = xdiv(config_ntime, pow2(ids));
	downsampled_inputs.at(ids) = Array<float> ({beams_per_batch, pow2(config_rank), nt_ds}, af_uhost | af_zero);
    }
    
    for (int iout = 0; iout < nout; iout++) {
	const DedispersionPlan::Stage1Tree &st1 = plan->stage1_trees.at(iout);
	long ids = st1.ds_level;
	long out_rank = st1.rank0 + st1.rank1_trigger;
	long weird_rank = out_rank + (ids ? 1 : 0);
	
	this->dedispersion_buffers.at(iout) = Array<float> ({beams_per_batch, pow2(weird_rank), st1.nt_ds}, af_uhost | af_zero);
	this->output_arrays.at(iout) = Array<float>({beams_per_batch, pow2(out_rank), st1.nt_ds}, af_uhost | af_zero);

	for (int batch = 0; batch < nbatches; batch++)
	    this->trees.at(batch*nout + iout) = ReferenceTree::make({ beams_per_batch, pow2(weird_rank), st1.nt_ds });
    }
    
    // Reminder: 'input_array' and 'output_arrays' are members of ReferenceDedisperserBase,
    // but are initialized by the subclass constructor.

    this->input_array = downsampled_inputs.at(0);   // alias
    this->check_iobuf_shapes();
}


// virtual override
void ReferenceDedisperser0::_dedisperse(long itime, long ibeam)
{
    for (int ids = 1; ids < nds; ids++) {
	
	// Step 1: downsample input array (straightforward downsample, not "lagged" downsample).
	// Outer length is nds, inner shape is (beams_per_batch, 2^config_rank, input_nt / pow2(ids)).
	// Reminder: 'input_array' is an alias for downsampled_inputs[0].

	Array<float> src = downsampled_inputs.at(ids-1);
	Array<float> dst = downsampled_inputs.at(ids);

	// FIXME reference_downsample_time() should operate on N-dimensional array.
	for (long b = 0; b < beams_per_batch; b++) {
	    Array<float> src2 = src.slice(0,b);
	    Array<float> dst2 = dst.slice(0,b);
	    reference_downsample_time(src2, dst2, false);  // normalize=false, i.e. no factor 0.5
	}
    }

    for (int iout = 0; iout < nout; iout++) {
	const DedispersionPlan::Stage1Tree &st1 = plan->stage1_trees.at(iout);
	long ids = st1.ds_level;
	long weird_rank = st1.rank0 + st1.rank1_trigger + (ids ? 1 : 0);
	
	Array<float> in = downsampled_inputs.at(ids).slice(1, 0, pow2(weird_rank));
	Array<float> dd = dedispersion_buffers.at(iout);
	Array<float> out = output_arrays.at(iout);
	
	// Step 2: copy from 'downsampled_inputs' to 'dedispersion_buffers'.
	
	dd.fill(in);

	// Step 3: apply tree dedispersion (one-stage, not two-stage).
	// Vector length is (nbatches * nout).
	
	long batch = xdiv(ibeam, beams_per_batch);
	auto tree = trees.at(batch*nout + iout);
	tree->dedisperse(dd);
	
	// Step 4: copy from 'dedispersion_buffers' to 'output_arrays'.
	// In downsampled trees, we compute twice as many DMs as necessary, then copy the bottom half.
	
	if (ids == 0)
	    out.fill(dd);
	else {
	    // FIXME refence_extract_odd_channels() should operate on N-dimensional array.
	    // reference_extract_odd_channels(dd, out);
	    for (long b = 0; b < beams_per_batch; b++) {
		Array<float> src2 = dd.slice(0,b);
		Array<float> dst2 = out.slice(0,b);
		reference_extract_odd_channels(src2, dst2);
	    }
	}
    }
}


// -------------------------------------------------------------------------------------------------
//
// Sophistication == 1:
//
//   - Uses same two-stage tree/lag structure as plan.
//   - Lags are split into segments + residuals, but not further split into chunks.
//   - Lags are applied with a per-tree ReferenceLagbuf, rather than using ring/staging buffers.


struct ReferenceDedisperser1 : public ReferenceDedisperserBase
{
    ReferenceDedisperser1(const shared_ptr<DedispersionPlan> &plan_);

    // Step 1: run LaggedDownsampler.
    // Step 2: run stage0 dedispersion kernels 
    Stage0Buffers stage0_buffers;

    // Step 3: copy stage0 -> stage1
    Stage1Buffers stage1_buffers;
    
    // Step 4: apply lags
    // Step 5: run stage1 dedispersion kernels.
    vector<shared_ptr<ReferenceLagbuf>> stage1_lagbufs;  // length (nbatches * nout)
    
    virtual void _dedisperse(long itime, long ibeam) override;
};


ReferenceDedisperser1::ReferenceDedisperser1(const shared_ptr<DedispersionPlan> &plan_) :
    ReferenceDedisperserBase(plan_, 1),
    stage0_buffers(plan_),
    stage1_buffers(plan_)
{
    long S = nelts_per_segment;
    
    // Reminder: 'input_array' and 'output_arrays' are members of ReferenceDedisperserBase,
    // but are initialized by the subclass constructor.

    this->input_array = stage0_buffers.dd_bufs.at(0);  // alias
    this->output_arrays = stage1_buffers.dd_bufs;      // alias
    this->check_iobuf_shapes();

    this->stage1_lagbufs.resize(nbatches * nout);

    for (long iout = 0; iout < nout; iout++) {
	const DedispersionPlan::Stage1Tree &st1 = plan->stage1_trees.at(iout);
	int rank0 = st1.rank0;
	int rank1 = st1.rank1_trigger;
	int nchan = pow2(rank0+rank1);
	bool is_downsampled = (st1.ds_level > 0);

	Array<int> lags({beams_per_batch, nchan}, af_uhost);

	for (long i1 = 0; i1 < pow2(rank1); i1++) {
	    for (long i0 = 0; i0 < pow2(rank0); i0++) {
		int row = i1 * pow2(rank0) + i0;
		int lag = rb_lag(i1, i0, rank0, rank1, is_downsampled);
		int segment_lag = lag / S;   // round down

		for (long b = 0; b < beams_per_batch; b++)
		    lags.data[b*nchan + row] = segment_lag * S;
	    }
	}

	for (long b = 0; b < nbatches; b++)
	    stage1_lagbufs.at(b*nout + iout) = make_shared<ReferenceLagbuf> (lags, st1.nt_ds);
    }
}


// virtual override
void ReferenceDedisperser1::_dedisperse(long itime, long ibeam)
{
    // Step 1: run LaggedDownsampler.
    // Step 2: run stage0 dedispersion kernels.
    this->stage0_buffers.apply_lagged_downsampler(ibeam);    
    this->stage0_buffers.apply_dedispersion_kernels(itime, ibeam);

    for (int iout = 0; iout < nout; iout++) {
	const DedispersionPlan::Stage1Tree &st1 = plan->stage1_trees.at(iout);
	long rank0 = st1.rank0;
	long rank1 = st1.rank1_trigger;

	// Step 3: copy stage0 -> stage1
	
	Array<float> src = stage0_buffers.dd_bufs.at(st1.ds_level);  // shape (beams_per_batch, 2^rank_ambient, nt_ds)
	src = src.slice(1, 0, pow2(rank0+rank1));                    // shape (beams_per_batch, 2^rank, nt_ds)

	Array<float> dst = stage1_buffers.dd_bufs.at(iout);
	dst.fill(src);

	// Step 4: apply lags
	
	long b = xdiv(ibeam, beams_per_batch);
	auto lagbuf = stage1_lagbufs.at(b*nout + iout);
	lagbuf->apply_lags(dst);
    }

    // Step 5: run stage1 dedispersion kernels
    this->stage1_buffers.apply_dedispersion_kernels(itime, ibeam);
}


// -------------------------------------------------------------------------------------------------
//
// ReferenceDedisperser2: as close to the GPU implementation as possible.


struct ReferenceDedisperser2 : public ReferenceDedisperserBase
{
    ReferenceDedisperser2(const std::shared_ptr<DedispersionPlan> &plan);

    virtual void _dedisperse(long itime, long ibeam) override;
    
    // Step 1: run LaggedDownsampler.
    // Step 2: run stage0 dedispersion kernels.
    Stage0Buffers stage0_buffers;
    
    // Step 3: copy stage0 -> ringbuf.
    Array<float> gpu_ringbuf;

    // Step 4: copy ringbuf -> stage1.
    // Step 5: run stage1 dedispersion kernels.
    Stage1Buffers stage1_buffers;
};


ReferenceDedisperser2::ReferenceDedisperser2(const shared_ptr<DedispersionPlan> &plan_) :
    ReferenceDedisperserBase(plan_, 2),
    stage0_buffers(plan_),
    stage1_buffers(plan_)
{
    long S = nelts_per_segment;
    this->gpu_ringbuf = Array<float> ({plan->gmem_ringbuf_nseg * S}, af_uhost | af_zero);
    
    // Reminder: 'input_array' and 'output_arrays' are members of ReferenceDedisperserBase,
    // but are initialized by the subclass constructor.

    this->input_array = stage0_buffers.dd_bufs.at(0);   // alias
    this->output_arrays = stage1_buffers.dd_bufs;       // alias
    this->check_iobuf_shapes();
}


// Helper for ReferenceDedisperser2::_dedisperse()
// Returns segment offset in ring buffer.
static long rb_segment(const uint *rb_locs, long rb_pos, uint nelts_per_segment)
{
    uint rb_offset = rb_locs[0];  // in segments, not bytes
    uint rb_phase = rb_locs[1];   // index of (time chunk, beam) pair, relative to current pair
    uint rb_len = rb_locs[2];     // number of (time chunk, beam) pairs in ringbuf (same as Ringbuf::rb_len)
    uint rb_nseg = rb_locs[3];    // number of segments per (time chunk, beam) (same as Ringbuf::nseg_per_beam)

    uint i = (rb_pos + rb_phase) % rb_len;
    long s = rb_offset + (i * rb_nseg);

    return s;
}


void ReferenceDedisperser2::_dedisperse(long itime, long ibeam)
{
    long S = nelts_per_segment;
    long rb_pos = itime*total_beams + ibeam;
    float *rbuf = gpu_ringbuf.data;
    
    // Step 1: run LaggedDownsampler.
    // Step 2: run stage0 dedispersion kernels.
    this->stage0_buffers.apply_lagged_downsampler(ibeam);    
    this->stage0_buffers.apply_dedispersion_kernels(itime, ibeam);

    // Step 3: copy stage0 -> ringbuf.

    for (int ids = 0; ids < nds; ids++) {
	const DedispersionPlan::Stage0Tree &st0 = plan->stage0_trees.at(ids);

	long nchan0 = pow2(st0.rank0);
	long nchan1 = pow2(st0.rank1);
	long nt_ds = st0.nt_ds;
	long ns = xdiv(nt_ds, S);

	// (rb_locs0, src0) = base pointers for tree
	const uint *rb_locs0 = plan->stage0_rb_locs.data + (4 * st0.base_segment);
	const float *src0 = stage0_buffers.dd_bufs.at(ids).data;
	long src_bstride = stage0_buffers.dd_bufs.at(ids).strides[0];

	// Loop over segments in tree.
	for (long s = 0; s < ns; s++) {
	    for (long i1 = 0; i1 < nchan1; i1++) {
		for (long i0 = 0; i0 < nchan0; i0++) {
		    // (rb_locs1, src1) = base pointers for segment.
		    long iseg0 = s*nchan1*nchan0 + i1*nchan0 + i0;
		    const uint *rb_locs1 = rb_locs0 + 4*iseg0;
		    const float *src1 = src0 + (i1*nchan0+i0)*nt_ds + s*S;
		    
		    for (long b = 0; b < beams_per_batch; b++) {
			long s = rb_segment(rb_locs1, rb_pos+b, S);
			memcpy(rbuf + s*S, src1 + b*src_bstride, S * sizeof(float));
		    }
		}
	    }
	}
    }

    // Step 4: copy ringbuf -> stage1.

    for (int iout = 0; iout < nout; iout++) {
	const DedispersionPlan::Stage1Tree &st1 = plan->stage1_trees.at(iout);
	
	long nchan0 = pow2(st1.rank0);
	long nchan1 = pow2(st1.rank1_trigger);
	long nt_ds = st1.nt_ds;
	long ns = xdiv(st1.nt_ds, S);

	// (rb_locs0, dst0) = base pointers for tree
	const uint *rb_locs0 = plan->stage1_rb_locs.data + (4 * st1.base_segment);
	float *dst0 = stage1_buffers.dd_bufs.at(iout).data;
	long dst_bstride = stage1_buffers.dd_bufs.at(iout).strides[0];

	// Loop over segments in tree.
	for (long s = 0; s < ns; s++) {
	    for (long i0 = 0; i0 < nchan0; i0++) {
		for (long i1 = 0; i1 < nchan1; i1++) {
		    // (rb_locs1, dst1) = base pointers for segment.
		    long iseg1 = s*nchan1*nchan0 + i0*nchan1 + i1;
		    const uint *rb_locs1 = rb_locs0 + 4*iseg1;
		    float *dst1 = dst0 + (i1*nchan0+i0)*nt_ds + s*S;

		    for (long b = 0; b < beams_per_batch; b++) {
			long s = rb_segment(rb_locs1, rb_pos+b, S);
			memcpy(dst1 + b*dst_bstride, rbuf + s*S, S * sizeof(float));
		    }
		}
	    }
	}
    }

    // Step 5: run stage1 dedispersion kernels
    this->stage1_buffers.apply_dedispersion_kernels(itime, ibeam);    
}


// -------------------------------------------------------------------------------------------------


// Static member function
shared_ptr<ReferenceDedisperserBase> ReferenceDedisperserBase::make(const shared_ptr<DedispersionPlan> &plan_, int sophistication)
{
    if (sophistication == 0)
	return make_shared<ReferenceDedisperser0> (plan_);
    else if (sophistication == 1)
	return make_shared<ReferenceDedisperser1> (plan_);
    else if (sophistication == 2)
	return make_shared<ReferenceDedisperser2> (plan_);
    throw runtime_error("ReferenceDedisperserBase::make(): invalid value of 'sophistication' parameter");
}


}  // namespace pirate
