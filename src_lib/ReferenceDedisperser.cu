#include "../include/pirate/Dedisperser.hpp"

#include "../include/pirate/DedispersionPlan.hpp"
#include "../include/pirate/DedispersionConfig.hpp"
#include "../include/pirate/ReferenceTree.hpp"
#include "../include/pirate/ReferenceLagbuf.hpp"
#include "../include/pirate/constants.hpp"
#include "../include/pirate/inlines.hpp"
#include "../include/pirate/utils.hpp"


using namespace std;
using namespace ksgpu;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// Morally equivalent to calling the DedisperionBuffer constructor, with two differences:
//  - Uses dtype float32, regardless of what dtype is specified in 'params'.
//  - Returns an allocated DedispersionBuffer (constructor does not allocate by default).

static DedispersionBuffer _make_dd_buffer(const DedispersionBufferParams &params_)
{
    DedispersionBufferParams params = params_;
    params.dtype = Dtype::native<float> ();
    
    DedispersionBuffer buf(params);
    buf.allocate(af_uhost);
    return buf;
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
    this->input_rank = config.tree_rank;
    this->input_ntime = config.time_samples_per_chunk;
    this->total_beams = config.beams_per_gpu;
    this->beams_per_batch = config.beams_per_batch;
    this->gpu_ringbuf_nelts = plan->gmem_ringbuf_nseg * plan->nelts_per_segment;
    this->nbatches = xdiv(total_beams, beams_per_batch);

    const DedispersionBufferParams &out_params = plan->stage2_dd_buf_params;
    this->output_ntrees = out_params.nbuf;
    this->output_rank = out_params.buf_rank;
    this->output_ntime = out_params.buf_ntime;
    this->output_ds_level = plan->stage2_ds_level;

    // Some paranoid asserts follow.
    
    xassert(long(output_rank.size()) == output_ntrees);
    xassert(long(output_ntime.size()) == output_ntrees);
    xassert(long(output_ds_level.size()) == output_ntrees);

    for (long i = 0; i < output_ntrees; i++) {
	xassert((output_ds_level[i] >= 0) && (output_ds_level[i] < config.num_downsampling_levels));
	xassert(output_ntime[i] == xdiv(input_ntime, pow2(output_ds_level[i])));
    }
    
    // Note: 'input_array' and 'output_arrays' are members of ReferenceDedisperserBase,
    // but are initialized by the subclass constructor.
}


void ReferenceDedisperserBase::_init_iobufs(Array<float> &in, vector<Array<float>> &out)
{
    this->input_array = in;
    this->output_arrays = out;
    
    xassert_shape_eq(input_array, ({ beams_per_batch, pow2(input_rank), input_ntime }));
    xassert_eq(long(output_arrays.size()), output_ntrees);

    for (long i = 0; i < output_ntrees; i++)
	xassert_shape_eq(output_arrays.at(i), ({ beams_per_batch, pow2(output_rank.at(i)), output_ntime.at(i) }));
}


void ReferenceDedisperserBase::_init_iobufs(Array<void> &in_, vector<Array<void>> &out_)
{
    Array<float> in = in_.template cast<float> ("ReferenceDedisperser::_init_iobufs(): 'in' array");

    vector<Array<float>> out;
    for (ulong i = 0; i < out_.size(); i++)
	out.push_back(out_.at(i).template cast<float> ("ReferenceDedisperser::_init_iobufs(): 'out' array"));

    this->_init_iobufs(in, out);
}


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

    virtual void dedisperse(long itime, long ibeam) override;

    // Step 1: downsample input array (straightforward downsample, not "lagged" downsample!)
    // Outer length is nds, inner shape is (beams_per_batch, 2^input_rank, input_nt / pow2(ids)).
    
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
    long nds = config.num_downsampling_levels;
    
    this->downsampled_inputs.resize(nds);
    this->dedispersion_buffers.resize(output_ntrees);
    this->trees.resize(nbatches * output_ntrees);    
    this->output_arrays.resize(output_ntrees);

    for (long ids = 0; ids < nds; ids++) {
	long nt_ds = xdiv(input_ntime, pow2(ids));
	downsampled_inputs.at(ids) = Array<float> ({beams_per_batch, pow2(input_rank), nt_ds}, af_uhost | af_zero);
    }
    
    for (long iout = 0; iout < output_ntrees; iout++) {
	long out_rank = output_rank.at(iout);
	long out_ntime = output_ntime.at(iout);
	bool is_downsampled = (output_ds_level[iout] > 0);
	long dd_rank = out_rank + (is_downsampled ? 1 : 0);
	
	this->dedispersion_buffers.at(iout) = Array<float> ({ beams_per_batch, pow2(dd_rank), out_ntime }, af_uhost | af_zero);
	this->output_arrays.at(iout) = Array<float>({ beams_per_batch, pow2(out_rank), out_ntime }, af_uhost | af_zero);

	for (int batch = 0; batch < nbatches; batch++)
	    this->trees.at(batch*output_ntrees + iout) = ReferenceTree::make({ beams_per_batch, pow2(dd_rank), out_ntime });
    }

    // Reminder: subclass constructor is responsible for calling _init_iobufs(), to initialize
    // 'input_arrays' and 'output_arrays' in the case class.
    this->_init_iobufs(downsampled_inputs.at(0), output_arrays);
}


// virtual override
void ReferenceDedisperser0::dedisperse(long ibatch, long it_chunk)
{
    long nds = config.num_downsampling_levels;
    
    for (int ids = 1; ids < nds; ids++) {
	
	// Step 1: downsample input array (straightforward downsample, not "lagged" downsample).
	// Outer length is nds, inner shape is (beams_per_batch, 2^input_rank, input_nt / pow2(ids)).
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

    for (int iout = 0; iout < output_ntrees; iout++) {
	long out_rank = output_rank.at(iout);
	long out_ntime = output_ntime.at(iout);
	long ids = output_ds_level.at(iout);
	bool is_downsampled = (ids > 0);
	long dd_rank = out_rank + (is_downsampled ? 1 : 0);
	
	Array<float> in = downsampled_inputs.at(ids).slice(1, 0, pow2(dd_rank));
	Array<float> dd = dedispersion_buffers.at(iout);
	Array<float> out = output_arrays.at(iout);
	
	// Step 2: copy from 'downsampled_inputs' to 'dedispersion_buffers'.
	
	dd.fill(in);

	// Step 3: apply tree dedispersion (one-stage, not two-stage).
	// Vector length is (nbatches * output_ntrees).
	
	auto tree = trees.at(ibatch*output_ntrees + iout);
	tree->dedisperse(dd);
	
	// Step 4: copy from 'dedispersion_buffers' to 'output_arrays'.
	// In downsampled trees, we compute twice as many DMs as necessary, then copy the bottom half.
	
	if (!is_downsampled)
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
    // Step 2: run stage1 dedispersion kernels 
    // Step 3: copy stage1 -> stage2
    // Step 4: apply lags
    // Step 5: run stage2 dedispersion kernels.
    
    DedispersionBuffer stage1_dd_buf;
    DedispersionBuffer stage2_dd_buf;
    vector<shared_ptr<ReferenceLagbuf>> stage2_lagbufs;  // length (nbatches * output_ntrees)

    shared_ptr<ReferenceLaggedDownsamplingKernel> lds_kernel;
    vector<shared_ptr<ReferenceDedispersionKernel>> stage1_dd_kernels;
    vector<shared_ptr<ReferenceDedispersionKernel>> stage2_dd_kernels;
    
    virtual void dedisperse(long ibatch, long it_chunk) override;
};


ReferenceDedisperser1::ReferenceDedisperser1(const shared_ptr<DedispersionPlan> &plan_) :
    ReferenceDedisperserBase(plan_, 1)
{
    this->stage1_dd_buf = _make_dd_buffer(plan->stage1_dd_buf_params);
    this->stage2_dd_buf = _make_dd_buffer(plan->stage2_dd_buf_params);
    this->lds_kernel = make_shared<ReferenceLaggedDownsamplingKernel> (plan->lds_params);

    for (const DedispersionKernelParams &kparams_: plan->stage1_dd_kernel_params) {
	DedispersionKernelParams kparams = kparams_;
	kparams.output_is_ringbuf = false;  // in ReferenceDedisperer1, ringbufs are disabled.
	this->stage1_dd_kernels.push_back(make_shared<ReferenceDedispersionKernel> (kparams));
    }

    for (const DedispersionKernelParams &kparams_: plan->stage2_dd_kernel_params) {
	DedispersionKernelParams kparams = kparams_;
	kparams.input_is_ringbuf = false;   // in ReferenceDeidsperser1, ringbufs are disabled.
	this->stage2_dd_kernels.push_back(make_shared<ReferenceDedispersionKernel> (kparams));
    }
    
    // Initalize stage2_lagbufs.
    // (Note that these lagbufs are used in ReferenceDedisperser1, but not ReferenceDedisperser2.)
    
    long S = plan->nelts_per_segment;
    this->stage2_lagbufs.resize(nbatches * output_ntrees);

    for (long iout = 0; iout < output_ntrees; iout++) {
	long rank = output_rank.at(iout);
	long ntime = output_ntime.at(iout);
	bool is_downsampled = (output_ds_level.at(iout) > 0);
	
	// Hmm, getting rank0/rank1 split from current data structures is awkward.
	const DedispersionPlan::Stage2Tree &st2 = plan->stage2_trees.at(iout);
	long rank0 = st2.rank0;
	long rank1 = st2.rank1_trigger;
	xassert(rank == rank0+rank1);

	Array<int> lags({ beams_per_batch, pow2(rank) }, af_uhost);

	for (long i1 = 0; i1 < pow2(rank1); i1++) {
	    for (long i0 = 0; i0 < pow2(rank0); i0++) {
		long row = i1 * pow2(rank0) + i0;
		long lag = rb_lag(i1, i0, rank0, rank1, is_downsampled);
		long segment_lag = lag / S;   // round down

		for (long b = 0; b < beams_per_batch; b++)
		    lags.data[b*pow2(rank) + row] = segment_lag * S;
	    }
	}

	for (long b = 0; b < nbatches; b++)
	    stage2_lagbufs.at(b*output_ntrees + iout) = make_shared<ReferenceLagbuf> (lags, ntime);
    }
    
    // Reminder: subclass constructor is responsible for calling _init_iobufs(), to initialize
    // 'input_arrays' and 'output_arrays' in the case class.
    this->_init_iobufs(stage1_dd_buf.bufs.at(0), stage2_dd_buf.bufs);
}


// virtual override
void ReferenceDedisperser1::dedisperse(long ibatch, long it_chunk)
{
    // Step 1: run LaggedDownsampler.    
    lds_kernel->apply(stage1_dd_buf, ibatch);

    // Step 2: run stage1 dedispersion kernels.
    for (uint i = 0; i < stage1_dd_kernels.size(); i++) {
	shared_ptr<ReferenceDedispersionKernel> kernel = stage1_dd_kernels.at(i);
	const DedispersionKernelParams &kp = kernel->params;
	Array<void> dd_buf = stage1_dd_buf.bufs.at(i);

	// See comments in DedispersionKernel.hpp for an explanation of this reshape operation.
	dd_buf = dd_buf.reshape({ kp.beams_per_batch, pow2(kp.amb_rank), pow2(kp.dd_rank), kp.ntime });	
	kernel->apply(dd_buf, dd_buf, ibatch, it_chunk);
    }

    // Step 3: copy stage1 -> stage2
    // Step 4: apply lags
    for (int iout = 0; iout < output_ntrees; iout++) {
	long rank = output_rank.at(iout);
	long ds_level = output_ds_level.at(iout);
	
	Array<void> src = stage1_dd_buf.bufs.at(ds_level);  // shape (beams_per_batch, 2^rank_ambient, ntime)
	src = src.slice(1, 0, pow2(rank));                  // shape (beams_per_batch, 2^rank, ntime)

	Array<void> dst_ = stage2_dd_buf.bufs.at(iout);
	Array<float> dst = dst_.template cast<float> ();
	dst.fill(src);

	auto lagbuf = stage2_lagbufs.at(ibatch*output_ntrees + iout);
	lagbuf->apply_lags(dst);
    }

    // Step 5: run stage2 dedispersion kernels (in-place).
    for (uint i = 0; i < stage2_dd_kernels.size(); i++) {
	shared_ptr<ReferenceDedispersionKernel> kernel = stage2_dd_kernels.at(i);
	const DedispersionKernelParams &kp = kernel->params;
	Array<void> dd_buf = stage2_dd_buf.bufs.at(i);

	// See comments in DedispersionKernel.hpp for an explanation of this reshape/transpose operation.
	dd_buf = dd_buf.reshape({ kp.beams_per_batch, pow2(kp.dd_rank), pow2(kp.amb_rank), kp.ntime });
	dd_buf = dd_buf.transpose({0,2,1,3});
	kernel->apply(dd_buf, dd_buf, ibatch, it_chunk);
    }
}


// -------------------------------------------------------------------------------------------------
//
// ReferenceDedisperser2: as close to the GPU implementation as possible.


struct ReferenceDedisperser2 : public ReferenceDedisperserBase
{
    ReferenceDedisperser2(const std::shared_ptr<DedispersionPlan> &plan);
    
    // Step 1: run LaggedDownsampler.
    // Step 2: run stage1 dedispersion kernels (output to ringbuf)
    // Step 3: run stage2 dedispersion kernels (input from ringbuf)

    DedispersionBuffer stage1_dd_buf;
    DedispersionBuffer stage2_dd_buf;
    Array<float> gpu_ringbuf;

    shared_ptr<ReferenceLaggedDownsamplingKernel> lds_kernel;
    vector<shared_ptr<ReferenceDedispersionKernel>> stage1_dd_kernels;
    vector<shared_ptr<ReferenceDedispersionKernel>> stage2_dd_kernels;
    
    virtual void dedisperse(long ibatch, long it_chunk) override;
};


ReferenceDedisperser2::ReferenceDedisperser2(const shared_ptr<DedispersionPlan> &plan_) :
    ReferenceDedisperserBase(plan_, 2)
{
    this->stage1_dd_buf = _make_dd_buffer(plan->stage1_dd_buf_params);
    this->stage2_dd_buf = _make_dd_buffer(plan->stage2_dd_buf_params);
    this->gpu_ringbuf = Array<float>({ gpu_ringbuf_nelts }, af_uhost | af_zero);
    this->lds_kernel = make_shared<ReferenceLaggedDownsamplingKernel> (plan->lds_params);
					
    for (const DedispersionKernelParams &kparams: plan->stage1_dd_kernel_params)
	this->stage1_dd_kernels.push_back(make_shared<ReferenceDedispersionKernel> (kparams));

    for (const DedispersionKernelParams &kparams: plan->stage2_dd_kernel_params)
	this->stage2_dd_kernels.push_back(make_shared<ReferenceDedispersionKernel> (kparams));
    
    // Reminder: subclass constructor is responsible for calling _init_iobufs(), to initialize
    // 'input_arrays' and 'output_arrays' in the case class.
    this->_init_iobufs(stage1_dd_buf.bufs.at(0), stage2_dd_buf.bufs);
}


void ReferenceDedisperser2::dedisperse(long ibatch, long it_chunk)
{
    // Step 1: run LaggedDownsampler.
    lds_kernel->apply(stage1_dd_buf, ibatch);

    // Step 2: run stage1 dedispersion kernels (output to ringbuf)
    for (uint i = 0; i < stage1_dd_kernels.size(); i++) {
	shared_ptr<ReferenceDedispersionKernel> kernel = stage1_dd_kernels.at(i);
	const DedispersionKernelParams &kp = kernel->params;
	Array<void> dd_buf = stage1_dd_buf.bufs.at(i);

	// See comments in DedispersionKernel.hpp for an explanation of this reshape operation.
	dd_buf = dd_buf.reshape({ kp.beams_per_batch, pow2(kp.amb_rank), pow2(kp.dd_rank), kp.ntime });	
	kernel->apply(dd_buf, this->gpu_ringbuf, ibatch, it_chunk);
    }

    // Step 3: run stage2 dedispersion kernels (input from ringbuf)
    for (uint i = 0; i < stage2_dd_kernels.size(); i++) {
	shared_ptr<ReferenceDedispersionKernel> kernel = stage2_dd_kernels.at(i);
	const DedispersionKernelParams &kp = kernel->params;
	Array<void> dd_buf = stage2_dd_buf.bufs.at(i);

	// See comments in DedispersionKernel.hpp for an explanation of this reshape/transpose operation.
	dd_buf = dd_buf.reshape({ kp.beams_per_batch, pow2(kp.dd_rank), pow2(kp.amb_rank), kp.ntime });
	dd_buf = dd_buf.transpose({0,2,1,3});
	kernel->apply(this->gpu_ringbuf, dd_buf, ibatch, it_chunk);
    }
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
