#include "../include/pirate/Dedisperser.hpp"

#include "../include/pirate/DedispersionPlan.hpp"
#include "../include/pirate/DedispersionConfig.hpp"
#include "../include/pirate/ReferenceTree.hpp"
#include "../include/pirate/ReferenceLagbuf.hpp"
#include "../include/pirate/constants.hpp"
#include "../include/pirate/inlines.hpp"
#include "../include/pirate/utils.hpp"  // dedisperse_non_incremental(), lag_non_incremental()

#include <ksgpu/rand_utils.hpp>    // rand_int()
#include <ksgpu/string_utils.hpp>  // tuple_str()
#include <ksgpu/test_utils.hpp>    // make_random_strides()


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
            this->trees.at(batch*output_ntrees + iout) = ReferenceTree::make({beams_per_batch, pow2(dd_rank), out_ntime}, 1);  // nspec=1
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

    long gpu_ringbuf_nelts = 0;    // = (plan->gmem_ringbuf_nseg * plan->nelts_per_segment)
    long host_ringbuf_nelts = 0;   // = (plan->hmem_ringbuf_nseg * plan->nelts_per_segment)
    
    Array<float> gpu_ringbuf;
    Array<float> host_ringbuf;

    shared_ptr<ReferenceLaggedDownsamplingKernel> lds_kernel;
    vector<shared_ptr<ReferenceDedispersionKernel>> stage1_dd_kernels;
    vector<shared_ptr<ReferenceDedispersionKernel>> stage2_dd_kernels;
    shared_ptr<CpuRingbufCopyKernel> g2g_copy_kernel;
    shared_ptr<CpuRingbufCopyKernel> h2h_copy_kernel;
    
    virtual void dedisperse(long ibatch, long it_chunk) override;
};


ReferenceDedisperser2::ReferenceDedisperser2(const shared_ptr<DedispersionPlan> &plan_) :
    ReferenceDedisperserBase(plan_, 2)
{
    this->stage1_dd_buf = _make_dd_buffer(plan->stage1_dd_buf_params);
    this->stage2_dd_buf = _make_dd_buffer(plan->stage2_dd_buf_params);

    this->gpu_ringbuf_nelts = plan->gmem_ringbuf_nseg * plan->nelts_per_segment;
    this->host_ringbuf_nelts = plan->hmem_ringbuf_nseg * plan->nelts_per_segment;
        
    this->gpu_ringbuf = Array<float>({ gpu_ringbuf_nelts }, af_uhost | af_zero);
    this->host_ringbuf = Array<float>({ host_ringbuf_nelts }, af_uhost | af_zero);
    
    this->lds_kernel = make_shared<ReferenceLaggedDownsamplingKernel> (plan->lds_params);
    this->g2g_copy_kernel = make_shared<CpuRingbufCopyKernel> (plan->g2g_copy_kernel_params);
    this->h2h_copy_kernel = make_shared<CpuRingbufCopyKernel> (plan->h2h_copy_kernel_params);
    
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
    const int BT = this->config.beams_per_gpu;            // total beams
    const int BB = this->config.beams_per_batch;          // beams per batch
    const int BA = this->config.num_active_batches * BB;  // active beams
    const int S = plan->nelts_per_segment;

    long iframe = (it_chunk * BT) + (ibatch * BB);

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

    // Step 3: extra copying steps needed for early triggers.

    DedispersionPlan::Ringbuf &rb_eth = plan->et_host_ringbuf;
    DedispersionPlan::Ringbuf &rb_etg = plan->et_gpu_ringbuf;
    
    xassert(rb_eth.nseg_per_beam == rb_etg.nseg_per_beam);
    xassert(rb_eth.rb_len == BA);
    xassert(rb_etg.rb_len == BA);

    long et_off = (iframe % rb_eth.rb_len) * rb_eth.nseg_per_beam;
    float *et_src = this->host_ringbuf.data + (rb_eth.base_segment * S) + et_off;
    float *et_dst = this->gpu_ringbuf.data + (rb_etg.base_segment * S) + et_off;
    long et_nbytes = BB * rb_eth.nseg_per_beam * S * sizeof(float);
    
    this->g2g_copy_kernel->apply(this->gpu_ringbuf, ibatch, it_chunk);     // gpu -> xfer
    this->h2h_copy_kernel->apply(this->host_ringbuf, ibatch, it_chunk);    // host -> et_host
    memcpy(et_dst, et_src, et_nbytes);  // et_host -> et_gpu (must come after h2h_copy_kernel)
    
    //
    // Step 4: copy host <-> xfer
    //
           
    xassert(plan->host_ringbufs.size() == uint(plan->max_clag+1));
    xassert(plan->xfer_ringbufs.size() == uint(plan->max_clag+1));
    xassert_divisible(BT, BB);   // assert that length-BB copies don't "wrap"
    
    for (int clag = 0; clag <= plan->max_clag; clag++) {
        DedispersionPlan::Ringbuf &rb_host = plan->host_ringbufs.at(clag);
        DedispersionPlan::Ringbuf &rb_xfer = plan->xfer_ringbufs.at(clag);

        xassert(rb_host.nseg_per_beam == rb_xfer.nseg_per_beam);
        xassert(rb_host.rb_len == clag*BT + BA);
        xassert(rb_xfer.rb_len == 2*BA);

        if (rb_host.nseg_per_beam == 0)
            continue;
        
        float *hp = this->host_ringbuf.data + (rb_host.base_segment * S);
        float *xp = this->gpu_ringbuf.data + (rb_xfer.base_segment * S);
            
        long hsrc = (iframe + BA) % rb_host.rb_len;  // host src phase
        long hdst = (iframe) % rb_host.rb_len;       // host dst phase
        long xsrc = (iframe) % rb_xfer.rb_len;       // xfer src phase
        long xdst = (iframe + BA) % rb_xfer.rb_len;  // xfer dst phase
        
        long m = rb_host.nseg_per_beam * S;   // nelts per beam (=frame)
        long n = BB * m * sizeof(float);      // nbytes to copy

        // Ordering of memcopies is arbitrary. (On the GPU they happen in parallel.)
        memcpy(xp + xdst*m, hp + hsrc*m, n);
        memcpy(hp + hdst*m, xp + xsrc*m, n);
    }
    
    // Step 5: run stage2 dedispersion kernels (input from ringbuf).    
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


// -------------------------------------------------------------------------------------------------
//
// ReferenceDedisperserBase::test_dedispersion_basics()
//
// Tests dedisperse_non_incremental(), ReferenceLagbuf, ReferenceTree, and tree recursion.
// Test helpers are in anonymous namespace to avoid cluttering header.


namespace {

// -------------------------------------------------------------------------------------------------
// Test dedisperse_non_incremental()


// This utility function currently isn't used anywhere except test_non_incremental_dedispersion().
static long dedispersion_delay(int rank, long freq, long dm_brev)
{
    long delay = 0;
    long delay0 = 0;

    for (int r = 0; r < rank; r++) {
        long d = (dm_brev & 1) ? (delay0+1) : delay0;
        delay += ((freq & 1) ? 0 : d);
        delay0 += d;
        dm_brev >>= 1;
        freq >>= 1;
    }

    return delay;
}


static void test_non_incremental_dedispersion(int rank, long ntime, long nspec, long dm_brev, long t0, long s0)
{
    cout << "test_non_incremental_dedispersion(rank=" << rank << ", ntime=" << ntime
         << ", nspec=" << nspec << ", dm_brev=" << dm_brev << ", t0=" << t0
         << ", s0=" << s0 << ")" << endl;
    
    check_rank(rank, "test_non_incremental_dedispersion");
    xassert((dm_brev >= 0) && (dm_brev < pow2(rank)));
    xassert((t0 >= 0) && (t0 < ntime));
    xassert((s0 >= 0) && (s0 < nspec));

    // dedisperse_non_incremental() expects a 2-d array.
    long nchan = pow2(rank);
    Array<float> arr({nchan, ntime*nspec}, af_uhost | af_random);

    float x = 0.0;
    for (int ifreq = 0; ifreq < nchan; ifreq++) {
        long t = t0 - dedispersion_delay(rank, ifreq, dm_brev);
        if (t >= 0)
            x += arr.at({ifreq, t*nspec + s0 });
    }

    dedisperse_non_incremental(arr, nspec);
    float y = arr.at({dm_brev, t0*nspec + s0});

    float eps = fabs(x-y) / sqrt(nchan);
    xassert(eps < 1.0e-5);
}


static void test_non_incremental_dedispersion()
{
    int rank = rand_int(0, 9);
    long ntime = rand_int(1, 100);
    long nspec = rand_int(1, 10);
    long dm_brev = rand_int(0, pow2(rank));
    long t0 = rand_int(0, ntime);
    long s0 = rand_int(0, nspec);
    
    test_non_incremental_dedispersion(rank, ntime, nspec, dm_brev, t0, s0);
}


// -------------------------------------------------------------------------------------------------
// Test 'class ReferenceLagbuf'


static void lag_non_incremental(Array<float> &arr, const Array<int> &lags)
{
    xassert(arr.ndim > 1);
    xassert(lags.shape_equals(arr.ndim-1, arr.shape));
    xassert(arr.is_fully_contiguous());

    long nchan = lags.size;
    long nt = arr.shape[arr.ndim-1];
    
    Array<float> arr_2d = arr.reshape({nchan, nt});
    Array<int> lags_1d = lags.clone();
    lags_1d = lags_1d.reshape({nchan});

    for (long i = 0; i < nchan; i++) {
        float *row = arr_2d.data + i*nt;
        long lag = lags_1d.data[i];

        lag = min(lag, nt);
        memmove(row+lag, row, (nt-lag) * sizeof(float));
        memset(row, 0, lag * sizeof(float));
    }
}


static void test_reference_lagbuf(const Array<int> &lags, const vector<long> data_strides, int nt_chunk, int nchunks)
{
    cout << "test_reference_lagbuf:"
         << " lags.shape=" << lags.shape_str()
         << ", lags.strides=" << lags.stride_str()
         << ", data_strides=" << tuple_str(data_strides)
         << ", nt_chunk=" << nt_chunk
         << ", nchunks=" << nchunks << endl;

    xassert(long(data_strides.size()) == lags.ndim+1);
    xassert(data_strides[lags.ndim] == 1);
    
    int d = lags.ndim;
    int nt_tot = nt_chunk * nchunks;

    // Creating axis names feels silly, but assert_arrays_equal() requires them.
    vector<string> axis_names(d+1);
    for (int i = 0; i < d; i++)
        axis_names[i] = "ix" + to_string(i);
    axis_names[d] = "t";
    
    vector<long> shape_lg(d+1);
    vector<long> shape_sm(d+1);
    
    for (int i = 0; i < d; i++)
        shape_lg[i] = shape_sm[i] = lags.shape[i];

    shape_lg[d] = nt_tot;
    shape_sm[d] = nt_chunk;

    Array<float> arr_lg(shape_lg, af_uhost | af_random);
    Array<float> arr_lg_ref = arr_lg.clone();
    lag_non_incremental(arr_lg_ref, lags);
    
    Array<float> arr_sm(shape_sm, data_strides, af_uhost | af_zero);  // note strides
    Array<float> arr_sm_ref(shape_sm, af_uhost | af_zero);

    ReferenceLagbuf rbuf(lags, nt_chunk);
    
    for (int c = 0; c < nchunks; c++) {
        // Extract chunk (arr_lg) -> (arr_sm)
        Array<float> s = arr_lg.slice(d, c*nt_chunk, (c+1)*nt_chunk);
        arr_sm.fill(s);

        // Apply lagbuf
        rbuf.apply_lags(arr_sm);

        // Extract chunk (arr_lg_ref) -> (arr_sm_ref)
        s = arr_lg_ref.slice(d, c*nt_chunk, (c+1)*nt_chunk);
        arr_sm_ref.fill(s);

        // Compare arr_sm, arr_sm_ref.
        ksgpu::assert_arrays_equal(arr_sm, arr_sm_ref, "incremental", "non-incremental", axis_names);
    }
}


static void test_reference_lagbuf()
{
    // Number of dimensions in 'lags' array
    int nd = rand_int(1, 4);

    // lags.shape + (nt_chunk, nchunks)
    vector<long> v = random_integers_with_bounded_product(nd+2, 10000);
    int nt_chunk = v[nd];
    int nchunks = v[nd+1];
    
    vector<long> lag_shape(nd);
    vector<long> data_shape(nd+1);
    memcpy(&data_shape[0], &v[0], (nd+1) * sizeof(long));
    memcpy(&lag_shape[0], &v[0], nd * sizeof(long));

    vector<long> lag_strides = make_random_strides(lag_shape);
    vector<long> data_strides = make_random_strides(data_shape, 1);   // time axis guaranteed continuous

    Array<int> lags(lag_shape, lag_strides, af_uhost | af_zero);
    double maxlog = log(1.5 * nt_chunk * nchunks);
    
    for (auto ix = lags.ix_start(); lags.ix_valid(ix); lags.ix_next(ix)) {
        double t = rand_uniform(-1.0, maxlog);
        lags.at(ix) = int(exp(t));
    }

    test_reference_lagbuf(lags, data_strides, nt_chunk, nchunks);
}


// -------------------------------------------------------------------------------------------------
// Test 'class ReferenceTree'


static void test_reference_tree(const vector<long> &shape, const vector<long> &strides, long nchunks, long nspec)
{
    cout << "test_reference_tree: shape=" << tuple_str(shape)
         << ", strides=" << tuple_str(strides)
         << ", nchunks=" << nchunks
         << ", nspec=" << nspec
         << endl;

    int ndim = shape.size();
    xassert(ndim >= 2);
    
    long nfreq = shape.at(ndim-2);
    long ninner_chunk = shape.at(ndim-1);
    long ninner_tot = ninner_chunk * nchunks;
    
    // Input data (multiple chunks)
    vector<long> big_shape = shape;
    big_shape[ndim-1] *= nchunks;
    Array<float> arr0(big_shape, af_uhost | af_random);

    // Step 1. reshape to (nouter, nfreq, ninner_tot), with precisely one spectator axis.

    long nouter = 1;
    for (int d = 0; d < ndim-2; d++)
        nouter *= shape[d];
    
    Array<float> arr1 = arr0.clone();  // note deep copy here
    arr1 = arr1.reshape({nouter, nfreq, ninner_tot});

    // Step 2. loop over outer spectator axis, and call dedisperse_non_incremental().
    
    for (long i = 0; i < nouter; i++) {
        Array<float> view_2d = arr1.slice(0, i);  // shape (nfreq, ninner_tot)
        dedisperse_non_incremental(view_2d, nspec);
    }

    // Step 3. reshape back to original shape.
    // (This concludes the non-incremental dedispersion.)
    
    arr1 = arr1.reshape(big_shape);

    // Now apply incremental dedispersion in chunks, and compare.
    
    ReferenceTree rtree(shape, nspec);
    Array<float> chunk(shape, strides, af_uhost | af_zero);

    // Apply incremental dedispersion to arr0 (in place)
    for (long c = 0; c < nchunks; c++) {
        Array<float> slice = arr0.slice(ndim-1, c*ninner_chunk, (c+1)*ninner_chunk);
        chunk.fill(slice);
        rtree.dedisperse(chunk);
        slice.fill(chunk);
    }

    // Need axis names for assert_arrays_equal().
    vector<string> axis_names(ndim);
    for (int d = 0; d < ndim-2; d++)
        axis_names[d] = "spec" + to_string(d);
    axis_names[ndim-2] = "dm_brev";
    axis_names[ndim-1] = "inner";
    
    ksgpu::assert_arrays_equal(arr1, arr0, "non-incremental", "incremental", axis_names);
}


static void test_reference_tree()
{
    int rank = rand_int(1, 9);
    int ndim = rand_int(2, 6);

    // v = (spectators) + (ntime,nspec,nchunks)
    vector<long> v = ksgpu::random_integers_with_bounded_product(ndim+1, 100000 / pow2(rank));
    long ntime = v[ndim-2];
    long nspec = v[ndim-1];
    long nchunks = v[ndim];

    // shape = (spectators) + (2^rank, nspec*nchunks)
    vector<long> shape(ndim);
    for (long d = 0; d < ndim-2; d++)
        shape[d] = v[d];
    shape[ndim-2] = pow2(rank);
    shape[ndim-1] = nspec * nchunks;

    vector<long> strides = ksgpu::make_random_strides(shape, 1);  // ncontig=1
    test_reference_tree(shape, strides, nchunks, nspec);
}


// -------------------------------------------------------------------------------------------------
// Test tree recursion


static void test_tree_recursion(int rank0, int rank1, long nt_chunk, long nchunks)
{
    cout << "test_tree_recursion: rank0=" << rank0 << ", rank1=" << rank1
         << ", nt_chunk=" << nt_chunk << ", nchunks=" << nchunks << endl;

    int rank_tot = rank0 + rank1;
    
    check_rank(rank0, "test_tree_recursion [rank0]");
    check_rank(rank1, "test_tree_recursion [rank1]");
    check_rank(rank_tot, "test_tree_recursion [rank_tot]");
               
    long nfreq_tot = pow2(rank_tot);
    long nfreq0 = pow2(rank0);
    long nfreq1 = pow2(rank1);

    Array<int> lags({nfreq1,nfreq0}, af_uhost | af_zero);
    for (long i = 0; i < nfreq1; i++)
        for (long j = 0; j < nfreq0; j++)
            lags.at({i,j}) = rb_lag(i, j, rank0, rank1, false);  // uflag=false

    ReferenceTree big_tree({nfreq_tot, nt_chunk}, 1);    // nspec=1
    ReferenceTree tree0({nfreq1, nfreq0, nt_chunk}, 1);  // nspec=1
    ReferenceTree tree1({nfreq0, nfreq1, nt_chunk}, 1);  // nspec=1
    ReferenceLagbuf lagbuf(lags, nt_chunk);

    for (long c = 0; c < nchunks; c++) {
        Array<float> chunk0({nfreq_tot, nt_chunk}, af_uhost | af_random);

        // "Two-step" dedispersion.
        Array<float> chunk1 = chunk0.clone();
        chunk1 = chunk1.reshape({nfreq1, nfreq0, nt_chunk});
        tree0.dedisperse(chunk1);
        lagbuf.apply_lags(chunk1);
        chunk1 = chunk1.transpose({1,0,2});
        tree1.dedisperse(chunk1);
        chunk1 = chunk1.transpose({1,0,2});
        chunk1 = chunk1.reshape({nfreq_tot, nt_chunk});

        // "One-step" dedispersion.
        big_tree.dedisperse(chunk0);

        // Third step: compare chunk0 / chunk1
        // (arr0, arr1, name0, name1, axis_names)
        ksgpu::assert_arrays_equal(chunk0, chunk1, "1-step", "2-step", {"dm_brev","t"});
    }
}


static void test_tree_recursion()
{
    int rank = rand_int(0, 9);
    int rank0 = rand_int(0, rank+1);
    int rank1 = rank - rank0;
    long nt_chunk = rand_int(1, pow2(std::max(rank0,rank1)+1));
    long maxchunks = std::max(3L, 10000 / (pow2(rank) * nt_chunk));
    long nchunks = rand_int(1, maxchunks+1);
    
    test_tree_recursion(rank0, rank1, nt_chunk, nchunks);
}

}  // anonymous namespace


void ReferenceDedisperserBase::test_dedispersion_basics()
{
    test_non_incremental_dedispersion();
    test_reference_lagbuf();
    test_reference_tree();
    test_tree_recursion();
}


}  // namespace pirate
