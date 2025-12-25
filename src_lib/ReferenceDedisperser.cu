#include "../include/pirate/Dedisperser.hpp"

#include "../include/pirate/DedispersionPlan.hpp"
#include "../include/pirate/DedispersionConfig.hpp"
#include "../include/pirate/ReferenceTree.hpp"
#include "../include/pirate/ReferenceLagbuf.hpp"
#include "../include/pirate/MegaRingbuf.hpp"
#include "../include/pirate/TreeGriddingKernel.hpp"
#include "../include/pirate/PeakFindingKernel.hpp"
#include "../include/pirate/constants.hpp"
#include "../include/pirate/inlines.hpp"
#include "../include/pirate/utils.hpp"

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


ReferenceDedisperserBase::ReferenceDedisperserBase(
    const shared_ptr<DedispersionPlan> &plan_, 
    const vector<long> &Dcore_,
    int sophistication_
) :
    plan(plan_),
    config(deref(plan_)->config),
    Dcore(Dcore_),
    sophistication(sophistication_)
{
    xassert_eq(long(Dcore.size()), plan->stage2_ntrees);

    this->nfreq = config.get_total_nfreq();
    this->input_rank = config.tree_rank;
    this->input_ntime = config.time_samples_per_chunk;
    this->total_beams = config.beams_per_gpu;
    this->beams_per_batch = config.beams_per_batch;
    this->nbatches = xdiv(total_beams, beams_per_batch);

    const DedispersionBufferParams &out_params = plan->stage2_dd_buf_params;
    this->output_ntrees = out_params.nbuf;
    this->output_rank = out_params.buf_rank;
    this->output_ntime = out_params.buf_ntime;

    // Some paranoid asserts.
    xassert_eq(long(output_rank.size()), output_ntrees);
    xassert_eq(long(output_ntime.size()), output_ntrees);
    xassert_eq(plan->stage2_ntrees, output_ntrees);

    // Tree gridding kernel.
    this->tree_gridding_kernel = make_shared<ReferenceTreeGriddingKernel> (plan->tree_gridding_kernel_params);

    // Peak-finding kernels.
    for (long i = 0; i < output_ntrees; i++) {
        const PeakFindingKernelParams &params = plan->stage2_pf_params.at(i);
        auto pf = make_shared<ReferencePeakFindingKernel> (params, Dcore.at(i));
        this->pf_kernels.push_back(pf);
    }

    // Allocate frequency-space input array.
    this->input_array = Array<float>({beams_per_batch, nfreq, input_ntime}, af_uhost | af_zero);
    
    // Allocate weights arrays.
    this->wt_arrays.resize(output_ntrees);
    for (long i = 0; i < output_ntrees; i++) {
        long ndm_wt = plan->stage2_pf_params.at(i).ndm_wt;
        long nt_wt = plan->stage2_pf_params.at(i).nt_wt;
        long P = plan->stage2_trees.at(i).nprofiles;
        long F = plan->stage2_trees.at(i).frequency_subbands.F;        
        this->wt_arrays[i] = Array<float>({beams_per_batch, ndm_wt, nt_wt, P, F}, af_uhost | af_zero);
    }

    // Alllocate out_max, out_argmax arrays.

    this->out_max.resize(output_ntrees);
    this->out_argmax.resize(output_ntrees);

    for (long i = 0; i < output_ntrees; i++) {
        long ndm_out = plan->stage2_pf_params.at(i).ndm_out;
        long nt_out = plan->stage2_pf_params.at(i).nt_out;
        this->out_max[i] = Array<float>({beams_per_batch, ndm_out, nt_out}, af_uhost | af_zero);
        this->out_argmax[i] = Array<uint>({beams_per_batch, ndm_out, nt_out}, af_uhost | af_zero);
    }
    
    // Note: 'output_arrays' is a member of ReferenceDedisperserBase,
    // but is initialized by the subclass constructor.
}


void ReferenceDedisperserBase::_init_output_arrays(vector<Array<float>> &out)
{
    this->output_arrays = out;
    
    xassert_eq(long(output_arrays.size()), output_ntrees);

    for (long i = 0; i < output_ntrees; i++)
        xassert_shape_eq(output_arrays.at(i), ({ beams_per_batch, pow2(output_rank.at(i)), output_ntime.at(i) }));
}


void ReferenceDedisperserBase::_init_output_arrays(vector<Array<void>> &out_)
{
    vector<Array<float>> out;
    for (ulong i = 0; i < out_.size(); i++)
        out.push_back(out_.at(i).template cast<float> ("ReferenceDedisperser::_init_output_arrays(): 'out' array"));

    this->_init_output_arrays(out);
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
    ReferenceDedisperser0(const shared_ptr<DedispersionPlan> &plan, const vector<long> &Dcore);

    virtual void dedisperse(long itime, long ibeam) override;

    // Step 0: Run tree gridding kernel (input_array -> downsampled_inputs.at(0)).
    // Step 1: downsample input array (straightforward downsample, not "lagged" downsample!)
    // Outer length is nds, inner shape is (beams_per_batch, 2^input_rank, input_nt / pow2(ids)).
    
    vector<Array<float>> downsampled_inputs;   // length num_downsampling_levels

    // Step 2: copy from 'downsampled_inputs' to 'dedispersion_buffers'.
    // In downsampled trees, we compute twice as many DMs as necessary, then drop the bottom half.
    // Each early trigger is computed in an independent tree, by disregarding some input channels.
    // Outer vector length is nout, inner shape is (beams_per_batch, 2^weird_rank, input_nt / pow2(ids)).
    //   where weird_rank = stage1_dd_rank + early_stage2_dd_rank + (is_downsampled ? 1 : 0)
    
    vector<Array<float>> dedispersion_buffers;  // length plan->stage2_ntrees

    // Step 3: apply tree dedispersion (one-stage, not two-stage).
    
    vector<shared_ptr<ReferenceTree>> trees;    // length (nbatches * plan->stage2_ntrees)
    vector<Array<float>> subband_buffers;       // length (plan->stage2_ntrees)

    // Step 4: copy from 'dedispersion_buffers' to 'output_arrays'.
    // In downsampled trees, we compute twice as many DMs as necessary, then copy the bottom half.
    // Reminder: 'output_arrays' is a member of ReferenceDedisperserBase.

    // Step 5: run peak-finding kernel.
    // In downsampled trees, we just run on the upper half of 'subband_buffers'.
};


ReferenceDedisperser0::ReferenceDedisperser0(const shared_ptr<DedispersionPlan> &plan_, const vector<long> &Dcore_) :
    ReferenceDedisperserBase(plan_, Dcore_, 0)
{
    long nds = config.num_downsampling_levels;
    
    this->downsampled_inputs.resize(nds);
    this->dedispersion_buffers.resize(output_ntrees);
    this->trees.resize(nbatches * output_ntrees);
    this->subband_buffers.resize(output_ntrees);
    this->output_arrays.resize(output_ntrees);

    for (long ids = 0; ids < nds; ids++) {
        long nt_ds = xdiv(input_ntime, pow2(ids));
        downsampled_inputs.at(ids) = Array<float> ({beams_per_batch, pow2(input_rank), nt_ds}, af_uhost | af_zero);
    }
    
    for (long iout = 0; iout < output_ntrees; iout++) {
        const DedispersionPlan::Stage2Tree &st2 = plan->stage2_trees.at(iout);
        const PeakFindingKernelParams &pf_params = plan->stage2_pf_params.at(iout);

        long out_rank = output_rank.at(iout);
        long out_ntime = output_ntime.at(iout);
        bool is_downsampled = (st2.ds_level > 0);
        long dd_rank = out_rank + (is_downsampled ? 1 : 0);
        long ndm_out = pf_params.ndm_out * (is_downsampled ? 2 : 1);
        long M = st2.frequency_subbands.M;

        this->dedispersion_buffers.at(iout) = Array<float> ({ beams_per_batch, pow2(dd_rank), out_ntime }, af_uhost | af_zero);
        this->output_arrays.at(iout) = Array<float>({ beams_per_batch, pow2(out_rank), out_ntime }, af_uhost | af_zero);
        this->subband_buffers.at(iout) = Array<float> ({beams_per_batch, ndm_out, M, pf_params.nt_in}, af_uhost | af_zero);

        for (int batch = 0; batch < nbatches; batch++) {
            ReferenceTree::Params tree_params;
            tree_params.num_beams = beams_per_batch;
            tree_params.amb_rank = 0;
            tree_params.dd_rank = dd_rank;
            tree_params.ntime = out_ntime;
            tree_params.nspec = 1;
            tree_params.subband_counts = pf_params.subband_counts;

            this->trees.at(batch*output_ntrees + iout) = make_shared<ReferenceTree> (tree_params);
        }
    }

    // Reminder: subclass constructor is responsible for calling _init_output_arrays(), to initialize
    // 'output_arrays' in the base class. 
    this->_init_output_arrays(output_arrays);
}


// virtual override
void ReferenceDedisperser0::dedisperse(long ichunk, long ibatch)
{
    long nds = config.num_downsampling_levels;
    
    // Step 0: Run tree gridding kernel (input_array -> downsampled_inputs.at(0)).
    tree_gridding_kernel->apply(downsampled_inputs.at(0), input_array);
    
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
        const DedispersionPlan::Stage2Tree &st2 = plan->stage2_trees.at(iout);

        long out_rank = output_rank.at(iout);
        long out_ntime = output_ntime.at(iout);
        long ids = st2.ds_level;
        bool is_downsampled = (ids > 0);
        long dd_rank = out_rank + (is_downsampled ? 1 : 0);
        
        Array<float> in = downsampled_inputs.at(ids).slice(1, 0, pow2(dd_rank));
        Array<float> dd = dedispersion_buffers.at(iout);
        Array<float> out = output_arrays.at(iout);
        
        // Step 2: copy from 'downsampled_inputs' to 'dedispersion_buffers'.
        
        dd.fill(in);

        // Step 3: apply tree dedispersion (one-stage, not two-stage).
        // Vector length is (nbatches * output_ntrees).
        // Note dd->dd2 reshape: (B,D,T) -> (B,1,D,T).

        Array<float> dd2 = dd.reshape({beams_per_batch, 1, pow2(dd_rank), out_ntime});
        auto tree = trees.at(ibatch*output_ntrees + iout);
        tree->dedisperse(dd2, subband_buffers.at(iout));
        
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

        // Step 5: run peak-finding kernel.
        // In downsampled trees, we just run on the upper half of 'subband_buffers'.

        Array<float> sb = subband_buffers.at(iout);

        if (is_downsampled)
            sb = sb.slice(1, sb.shape[1]/2, sb.shape[1]);

        auto pf_kernel = pf_kernels.at(iout);
        pf_kernel->apply(out_max.at(iout), out_argmax.at(iout), sb, wt_arrays.at(iout), ibatch);
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
    ReferenceDedisperser1(const shared_ptr<DedispersionPlan> &plan_, const vector<long> &Dcore_);

    // Step 0: Run tree gridding kernel (input_array -> stage1_dd_buf.bufs.at(0)).
    // Step 1: run LaggedDownsampler.
    // Step 2: run stage1 dedispersion kernels 
    // Step 3: copy stage1 -> stage2
    // Step 4: apply lags
    // Step 5: run stage2 dedispersion kernels.
    // Step 6: run peak-finding kernels.
    
    DedispersionBuffer stage1_dd_buf;
    DedispersionBuffer stage2_dd_buf;
    vector<shared_ptr<ReferenceLagbuf>> stage2_lagbufs;  // length (nbatches * output_ntrees)

    // Dedispersion output in subbands ('sb_out' arg to ReferenceDedispersionKernel::apply())
    // Shape (beams_per_batch, ndm_out, M, nt_in)
    vector<Array<float>> stage2_subband_bufs;
    
    shared_ptr<ReferenceLaggedDownsamplingKernel> lds_kernel;
    vector<shared_ptr<ReferenceDedispersionKernel>> stage1_dd_kernels;
    vector<shared_ptr<ReferenceDedispersionKernel>> stage2_dd_kernels;
    
    virtual void dedisperse(long ichunk, long ibatch) override;
};


ReferenceDedisperser1::ReferenceDedisperser1(const shared_ptr<DedispersionPlan> &plan_, const vector<long> &Dcore_) :
    ReferenceDedisperserBase(plan_, Dcore_, 1)
{
    this->stage1_dd_buf = _make_dd_buffer(plan->stage1_dd_buf_params);
    this->stage2_dd_buf = _make_dd_buffer(plan->stage2_dd_buf_params);
    this->lds_kernel = make_shared<ReferenceLaggedDownsamplingKernel> (plan->lds_params);

    for (const DedispersionKernelParams &kparams_: plan->stage1_dd_kernel_params) {
        DedispersionKernelParams kparams = kparams_;
        kparams.output_is_ringbuf = false;  // in ReferenceDedisperer1, ringbufs are disabled.
        kparams.producer_id = -1;

        vector<long> subband_counts = { 1 };  // no subband_counts needed in stage1 kernels
        this->stage1_dd_kernels.push_back(make_shared<ReferenceDedispersionKernel> (kparams, subband_counts));
    }

    for (long iout = 0; iout < plan->stage2_ntrees; iout++) {
        DedispersionKernelParams dd_params = plan->stage2_dd_kernel_params.at(iout);  // make copy
        dd_params.input_is_ringbuf = false;   // in ReferenceDeidsperser1, ringbufs are disabled.
        dd_params.consumer_id = -1;

        vector<long> subband_counts = plan->stage2_pf_params.at(iout).subband_counts;
        this->stage2_dd_kernels.push_back(make_shared<ReferenceDedispersionKernel> (dd_params, subband_counts));
    }
    
    // Initalize stage2_lagbufs.
    // (Note that these lagbufs are used in ReferenceDedisperser1, but not ReferenceDedisperser2.)
    
    long S = plan->nelts_per_segment;
    this->stage2_lagbufs.resize(nbatches * output_ntrees);

    for (long iout = 0; iout < output_ntrees; iout++) {
        const DedispersionPlan::Stage2Tree &st2 = plan->stage2_trees.at(iout);

        long rank = output_rank.at(iout);
        long ntime = output_ntime.at(iout);
        long ds_level = st2.ds_level;
        bool is_downsampled = (ds_level > 0);
        long stage2_amb_rank = st2.amb_rank;
        long stage2_dd_rank = st2.early_dd_rank;

        xassert_eq(rank, stage2_amb_rank + stage2_dd_rank);
        xassert_eq(ds_level, st2.ds_level);
        xassert_eq(ntime, st2.nt_ds);

        Array<int> lags({ beams_per_batch, pow2(rank) }, af_uhost);

        for (long freq_coarse = 0; freq_coarse < pow2(stage2_dd_rank); freq_coarse++) {
            for (long dm_brev = 0; dm_brev < pow2(stage2_amb_rank); dm_brev++) {
                long row = freq_coarse * pow2(stage2_amb_rank) + dm_brev;
                long lag = rb_lag(freq_coarse, dm_brev, stage2_amb_rank, stage2_dd_rank, is_downsampled);
                long segment_lag = lag / S;   // round down

                for (long b = 0; b < beams_per_batch; b++)
                    lags.data[b*pow2(rank) + row] = segment_lag * S;
            }
        }

        for (long b = 0; b < nbatches; b++)
            stage2_lagbufs.at(b*output_ntrees + iout) = make_shared<ReferenceLagbuf> (lags, ntime);
    }
    
    // Initialize stage2_subband_bufs.
    // Shape (beams_per_batch, ndm_out, M, nt_in)

    this->stage2_subband_bufs.resize(output_ntrees);

    for (long iout = 0; iout < output_ntrees; iout++) {
        long ndm_out = plan->stage2_pf_params.at(iout).ndm_out;
        long nt_in = plan->stage2_pf_params.at(iout).nt_in;
        long M = plan->stage2_trees.at(iout).frequency_subbands.M;
        stage2_subband_bufs.at(iout) = Array<float> ({beams_per_batch, ndm_out, M, nt_in}, af_uhost | af_zero);
    }

    // Reminder: subclass constructor is responsible for calling _init_output_arrays(), to initialize
    // 'output_arrays' in the base class. 
    this->_init_output_arrays(stage2_dd_buf.bufs);
}


// virtual override
void ReferenceDedisperser1::dedisperse(long ichunk, long ibatch)
{
    // Step 0: Run tree gridding kernel (input_array -> stage1_dd_buf.bufs.at(0)).
    Array<float> dd = stage1_dd_buf.bufs.at(0).template cast<float> ();
    tree_gridding_kernel->apply(dd, input_array);
    
    // Step 1: run LaggedDownsampler.    
    lds_kernel->apply(stage1_dd_buf, ibatch);

    // Step 2: run stage1 dedispersion kernels.
    for (uint i = 0; i < stage1_dd_kernels.size(); i++) {
        shared_ptr<ReferenceDedispersionKernel> kernel = stage1_dd_kernels.at(i);
        const DedispersionKernelParams &kp = kernel->params;
        Array<void> dd_buf = stage1_dd_buf.bufs.at(i);

        // See comments in DedispersionKernel.hpp for an explanation of this reshape operation.
        dd_buf = dd_buf.reshape({ kp.beams_per_batch, pow2(kp.amb_rank), pow2(kp.dd_rank), kp.ntime }); 

        Array<float> sb_empty;  // no subbands yet
        kernel->apply(dd_buf, dd_buf, sb_empty, ichunk, ibatch);
    }

    // Step 3: copy stage1 -> stage2
    // Step 4: apply lags
    for (int iout = 0; iout < output_ntrees; iout++) {
        const DedispersionPlan::Stage2Tree &st2 = plan->stage2_trees.at(iout);

        long rank = output_rank.at(iout);
        long ds_level = st2.ds_level;
        
        Array<void> src = stage1_dd_buf.bufs.at(ds_level);  // shape (beams_per_batch, 2^rank_ambient, ntime)
        src = src.slice(1, 0, pow2(rank));                  // shape (beams_per_batch, 2^rank, ntime)

        Array<void> dst_ = stage2_dd_buf.bufs.at(iout);
        Array<float> dst = dst_.template cast<float> ();
        dst.fill(src);

        auto lagbuf = stage2_lagbufs.at(ibatch*output_ntrees + iout);
        lagbuf->apply_lags(dst);
    }

    // Step 5: run stage2 dedispersion kernels (in-place).
    // Step 6: run peak-finding kernels.

    for (uint i = 0; i < stage2_dd_kernels.size(); i++) {
        shared_ptr<ReferenceDedispersionKernel> dd_kernel = stage2_dd_kernels.at(i);
        shared_ptr<ReferencePeakFindingKernel> pf_kernel = pf_kernels.at(i);
        const DedispersionKernelParams &kp = dd_kernel->params;

        // See comments in DedispersionKernel.hpp for an explanation of this reshape/transpose operation.
        Array<void> dd_buf = stage2_dd_buf.bufs.at(i);
        dd_buf = dd_buf.reshape({ kp.beams_per_batch, pow2(kp.dd_rank), pow2(kp.amb_rank), kp.ntime });
        dd_buf = dd_buf.transpose({0,2,1,3});

        Array<float> sb_buf = stage2_subband_bufs.at(i);
        dd_kernel->apply(dd_buf, dd_buf, sb_buf, ichunk, ibatch);
        pf_kernel->apply(out_max.at(i), out_argmax.at(i), sb_buf, wt_arrays.at(i), ibatch);
    }
}


// -------------------------------------------------------------------------------------------------
//
// ReferenceDedisperser2: as close to the GPU implementation as possible.


struct ReferenceDedisperser2 : public ReferenceDedisperserBase
{
    ReferenceDedisperser2(const shared_ptr<DedispersionPlan> &plan, const vector<long> &Dcore);
    
    // Step 0: Run tree gridding kernel (input_array -> stage1_dd_buf.bufs.at(0)).
    // Step 1: run LaggedDownsampler.
    // Step 2: run stage1 dedispersion kernels (output to ringbuf)
    // Step 3: extra copying steps needed for early triggers.
    // Step 4: copy host <-> xfer
    // Step 5: run stage2 dedispersion kernels (input from ringbuf)
    // Step 6: run peak-finding kernels.

    DedispersionBuffer stage1_dd_buf;
    DedispersionBuffer stage2_dd_buf;

    // Dedispersion output in subbands ('sb_out' arg to ReferenceDedispersionKernel::apply())
    // Shape (beams_per_batch, ndm_out, M, nt_in)
    vector<Array<float>> stage2_subband_bufs;

    long gpu_ringbuf_nelts = 0;    // = (mega_ringbuf->gpu_global_nseg * plan->nelts_per_segment)
    long host_ringbuf_nelts = 0;   // = (mega_ringbuf->host_global_nseg * plan->nelts_per_segment)
    
    Array<float> gpu_ringbuf;
    Array<float> host_ringbuf;

    shared_ptr<ReferenceLaggedDownsamplingKernel> lds_kernel;
    vector<shared_ptr<ReferenceDedispersionKernel>> stage1_dd_kernels;
    vector<shared_ptr<ReferenceDedispersionKernel>> stage2_dd_kernels;
    shared_ptr<CpuRingbufCopyKernel> g2g_copy_kernel;
    shared_ptr<CpuRingbufCopyKernel> h2h_copy_kernel;
    
    virtual void dedisperse(long ichunk, long ibatch) override;
};


ReferenceDedisperser2::ReferenceDedisperser2(const shared_ptr<DedispersionPlan> &plan_, const vector<long> &Dcore_) :
    ReferenceDedisperserBase(plan_, Dcore_, 2)
{
    this->stage1_dd_buf = _make_dd_buffer(plan->stage1_dd_buf_params);
    this->stage2_dd_buf = _make_dd_buffer(plan->stage2_dd_buf_params);

    this->gpu_ringbuf_nelts = plan->mega_ringbuf->gpu_global_nseg * plan->nelts_per_segment;
    this->host_ringbuf_nelts = plan->mega_ringbuf->host_global_nseg * plan->nelts_per_segment;
    
    this->gpu_ringbuf = Array<float>({ gpu_ringbuf_nelts }, af_uhost | af_zero);
    this->host_ringbuf = Array<float>({ host_ringbuf_nelts }, af_uhost | af_zero);
    
    this->lds_kernel = make_shared<ReferenceLaggedDownsamplingKernel> (plan->lds_params);
    this->g2g_copy_kernel = make_shared<CpuRingbufCopyKernel> (plan->g2g_copy_kernel_params);
    this->h2h_copy_kernel = make_shared<CpuRingbufCopyKernel> (plan->h2h_copy_kernel_params);
    
    this->stage2_subband_bufs.resize(output_ntrees);

    for (long iout = 0; iout < output_ntrees; iout++) {
        long ndm_out = plan->stage2_pf_params.at(iout).ndm_out;
        long nt_in = plan->stage2_pf_params.at(iout).nt_in;
        long M = plan->stage2_trees.at(iout).frequency_subbands.M;
        stage2_subband_bufs.at(iout) = Array<float> ({beams_per_batch, ndm_out, M, nt_in}, af_uhost | af_zero);
    }

    for (const DedispersionKernelParams &kparams: plan->stage1_dd_kernel_params) {
        vector<long> subband_counts = { 1 };  // no subbands needed in stage1 kernels
        this->stage1_dd_kernels.push_back(make_shared<ReferenceDedispersionKernel> (kparams, subband_counts));
    }
    
    for (long iout = 0; iout < plan->stage2_ntrees; iout++) {
        const DedispersionKernelParams &dd_params = plan->stage2_dd_kernel_params.at(iout);
        vector<long> subband_counts = plan->stage2_pf_params.at(iout).subband_counts;
        this->stage2_dd_kernels.push_back(make_shared<ReferenceDedispersionKernel> (dd_params, subband_counts));
    }
    
    // Reminder: subclass constructor is responsible for calling _init_output_arrays(), to initialize
    // 'output_arrays' in the base class. 
    this->_init_output_arrays(stage2_dd_buf.bufs);
}


void ReferenceDedisperser2::dedisperse(long ichunk, long ibatch)
{
    const long BT = this->config.beams_per_gpu;            // total beams
    const long BB = this->config.beams_per_batch;          // beams per batch
    const long BA = this->config.num_active_batches * BB;  // active beams
    const long S = plan->nelts_per_segment;

    long iframe = (ichunk * BT) + (ibatch * BB);

    // Step 0: Run tree gridding kernel (input_array -> stage1_dd_buf.bufs.at(0)).
    Array<float> dd = stage1_dd_buf.bufs.at(0).template cast<float> ();
    tree_gridding_kernel->apply(dd, input_array);
    
    // Step 1: run LaggedDownsampler.
    lds_kernel->apply(stage1_dd_buf, ibatch);

    // Step 2: run stage1 dedispersion kernels (output to ringbuf)
    for (uint i = 0; i < stage1_dd_kernels.size(); i++) {
        shared_ptr<ReferenceDedispersionKernel> kernel = stage1_dd_kernels.at(i);
        const DedispersionKernelParams &kp = kernel->params;
        Array<void> dd_buf = stage1_dd_buf.bufs.at(i);

        // See comments in DedispersionKernel.hpp for an explanation of this reshape operation.
        dd_buf = dd_buf.reshape({ kp.beams_per_batch, pow2(kp.amb_rank), pow2(kp.dd_rank), kp.ntime }); 

        Array<float> sb_empty;  // no subbands yet
        kernel->apply(dd_buf, this->gpu_ringbuf, sb_empty, ichunk, ibatch);
    }

    // Step 3: extra copying steps needed for early triggers.

    MegaRingbuf::Zone &eth_zone = plan->mega_ringbuf->et_host_zone;
    MegaRingbuf::Zone &etg_zone = plan->mega_ringbuf->et_gpu_zone;
    
    xassert(eth_zone.segments_per_frame == etg_zone.segments_per_frame);
    xassert(eth_zone.num_frames == BA);
    xassert(etg_zone.num_frames == BA);

    long et_off = (iframe % eth_zone.num_frames) * eth_zone.segments_per_frame;
    float *et_src = this->host_ringbuf.data + (eth_zone.global_segment_offset + et_off) * S;
    float *et_dst = this->gpu_ringbuf.data + (etg_zone.global_segment_offset + et_off) * S;
    long et_nbytes = BB * eth_zone.segments_per_frame * S * sizeof(float);
    
    this->g2g_copy_kernel->apply(this->gpu_ringbuf, ichunk, ibatch);     // gpu -> xfer
    this->h2h_copy_kernel->apply(this->host_ringbuf, ichunk, ibatch);    // host -> et_host
    memcpy(et_dst, et_src, et_nbytes);  // et_host -> et_gpu (must come after h2h_copy_kernel)
    
    // Step 4: copy host <-> xfer
    
    long max_clag = plan->mega_ringbuf->max_clag;
    xassert(plan->mega_ringbuf->host_zones.size() == uint(max_clag+1));
    xassert(plan->mega_ringbuf->xfer_zones.size() == uint(max_clag+1));
    xassert_divisible(BT, BB);   // assert that length-BB copies don't "wrap"
    
    for (int clag = 0; clag <= max_clag; clag++) {
        MegaRingbuf::Zone &host_zone = plan->mega_ringbuf->host_zones.at(clag);
        MegaRingbuf::Zone &xfer_zone = plan->mega_ringbuf->xfer_zones.at(clag);
     
        xassert(host_zone.segments_per_frame == xfer_zone.segments_per_frame);
        xassert(host_zone.num_frames == clag*BT + BA);
        xassert(xfer_zone.num_frames == 2*BA);

        if (host_zone.segments_per_frame == 0)
            continue;
        
        float *hp = this->host_ringbuf.data + (host_zone.global_segment_offset * S);
        float *xp = this->gpu_ringbuf.data + (xfer_zone.global_segment_offset * S);
        
        long hsrc = (iframe + BA) % host_zone.num_frames;  // host src phase
        long hdst = (iframe) % host_zone.num_frames;       // host dst phase
        long xsrc = (iframe) % xfer_zone.num_frames;       // xfer src phase
        long xdst = (iframe + BA) % xfer_zone.num_frames;  // xfer dst phase
        
        long m = host_zone.segments_per_frame * S;   // nelts per beam (=frame)
        long n = BB * m * sizeof(float);             // nbytes to copy

        // Ordering of memcopies is arbitrary. (On the GPU they happen in parallel.)
        memcpy(xp + xdst*m, hp + hsrc*m, n);
        memcpy(hp + hdst*m, xp + xsrc*m, n);
    }
    
    // Step 5: run stage2 dedispersion kernels (input from ringbuf). 
    // Step 6: run peak-finding kernels.

    for (uint i = 0; i < stage2_dd_kernels.size(); i++) {
        shared_ptr<ReferenceDedispersionKernel> dd_kernel = stage2_dd_kernels.at(i);
        shared_ptr<ReferencePeakFindingKernel> pf_kernel = pf_kernels.at(i);
        const DedispersionKernelParams &kp = dd_kernel->params;

        // See comments in DedispersionKernel.hpp for an explanation of this reshape/transpose operation.
        Array<void> dd_buf = stage2_dd_buf.bufs.at(i);
        dd_buf = dd_buf.reshape({ kp.beams_per_batch, pow2(kp.dd_rank), pow2(kp.amb_rank), kp.ntime });
        dd_buf = dd_buf.transpose({0,2,1,3});

        Array<float> sb_buf = stage2_subband_bufs.at(i);
        dd_kernel->apply(this->gpu_ringbuf, dd_buf, sb_buf, ichunk, ibatch);
        pf_kernel->apply(out_max.at(i), out_argmax.at(i), sb_buf, wt_arrays.at(i), ibatch);
    }
}


// -------------------------------------------------------------------------------------------------


// Static member function
shared_ptr<ReferenceDedisperserBase> ReferenceDedisperserBase::make(
    const shared_ptr<DedispersionPlan> &plan_, 
    const vector<long> &Dcore_,
    int sophistication_)
{
    if (sophistication_ == 0)
        return make_shared<ReferenceDedisperser0> (plan_, Dcore_);
    else if (sophistication_ == 1)
        return make_shared<ReferenceDedisperser1> (plan_, Dcore_);
    else if (sophistication_ == 2)
        return make_shared<ReferenceDedisperser2> (plan_, Dcore_);
    throw runtime_error("ReferenceDedisperserBase::make(): invalid value of 'sophistication' parameter");
}


}  // namespace pirate
