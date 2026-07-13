#include "../include/pirate/Dedisperser.hpp"

#include "../include/pirate/DedispersionConfig.hpp"
#include "../include/pirate/DedispersionPlan.hpp"
#include "../include/pirate/DedispersionTree.hpp"
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
    BumpAllocator allocator(af_uhost | af_zero, -1);  // dummy allocator
    buf.allocate(allocator);
    return buf;
}


// -------------------------------------------------------------------------------------------------
//
// ReferenceDedisperserBase


ReferenceDedisperserBase::ReferenceDedisperserBase(const Params &params_) :
    params(params_)
{
    const auto &plan = params.plan;   // local alias -- keeps the plan->... body below unchanged
    xassert(plan);
    // Incomplete plans lack the buffer/kernel params used below. (gpu_runnable=false
    // plans are fine here: reference dedispersion works with the default Dcore values.)
    xassert(!plan->params.is_incomplete);

    this->config = plan->config;
    this->dtype = plan->dtype;
    this->nfreq = plan->nfreq;
    this->nt_in = plan->nt_in;
    this->total_beams = plan->beams_per_gpu;
    this->beams_per_batch = plan->beams_per_batch;
    this->num_primary_trees = plan->num_primary_trees;
    this->nbatches = xdiv(total_beams, beams_per_batch);
    this->ntrees = plan->ntrees;
    this->trees = plan->trees;

    // Tree gridding kernel. (Not needed in tree-domain-input mode, where the caller
    // supplies an already-gridded toplevel array and "step 0" is a copy.)
    if (!params.tree_domain_input)
        this->tree_gridding_kernel = make_shared<ReferenceTreeGriddingKernel> (plan->tree_gridding_kernel_params);

    // Peak-finding kernels. (Per-tree Dcore comes from the plan, via pf_params.Dcore.)
    for (long itree = 0; itree < ntrees; itree++) {
        const PeakFindingKernelParams &pf_params = plan->stage2_pf_params.at(itree);
        auto pf_kernel = make_shared<ReferencePeakFindingKernel> (pf_params);
        this->pf_kernels.push_back(pf_kernel);
    }

    // Allocate input array (frequency-space, or toplevel tree-domain if tree_domain_input).
    long input_nchan = params.tree_domain_input ? pow2(config.toplevel_tree_rank) : nfreq;
    this->input_array = Array<float>({beams_per_batch, input_nchan, nt_in}, af_uhost | af_zero);
    
    // Allocate weights arrays.
    this->wt_arrays.resize(ntrees);
    for (long itree = 0; itree < ntrees; itree++) {
        const DedispersionTree &tree = trees.at(itree);
        long N = tree.frequency_subbands.N;
        long ndm_wt = tree.ndm_wt;
        long nt_wt = tree.nt_wt;
        long P = tree.nprofiles;
        this->wt_arrays[itree] = Array<float>({beams_per_batch, ndm_wt, nt_wt, P, N}, af_uhost | af_zero);
    }

    // Alllocate out_max, out_argmax arrays.

    this->out_max.resize(ntrees);
    this->out_argmax.resize(ntrees);

    for (long itree = 0; itree < ntrees; itree++) {
        const DedispersionTree &tree = trees.at(itree);
        long ndm_out = tree.ndm_out;
        long nt_out = tree.nt_out;

        this->out_max[itree] = Array<float>({beams_per_batch, ndm_out, nt_out}, af_uhost | af_zero);
        this->out_argmax[itree] = Array<uint>({beams_per_batch, ndm_out, nt_out}, af_uhost | af_zero);
    }

    // Optionally allocate out_var (per-chunk peak-finding variances). Sized from the pf
    // kernels so the shape matches what ReferencePeakFindingKernel::apply() expects.
    this->out_var.resize(ntrees);   // empty unless enable_variances
    if (params.enable_variances) {
        for (long itree = 0; itree < ntrees; itree++) {
            const ReferencePeakFindingKernel &pf = *pf_kernels.at(itree);
            this->out_var[itree] = Array<double>(
                {beams_per_batch, pf.params.ndm_out, pf.fs.M, pf.nprofiles}, af_uhost | af_zero);
        }
    }
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
    ReferenceDedisperser0(const Params &params);

    virtual void dedisperse(long itime, long ibeam) override;

    // Step 0: Run tree gridding kernel (input_array -> downsampled_inputs.at(0)).
    // Step 1: downsample input array (straightforward downsample, not "lagged" downsample!)
    // Outer length is npri, inner shape is (beams_per_batch, 2^config.toplevel_tree_rank, input_nt / pow2(ipri)).
    
    vector<Array<float>> downsampled_inputs;   // length num_primary_trees

    // Step 2: copy from 'downsampled_inputs' to 'dedispersion_buffers'.
    // In downsampled trees, we compute twice as many DMs as necessary, then drop the bottom half.
    // Each early trigger is computed in an independent tree, by disregarding some input channels.
    // Outer vector length is nout, inner shape is (beams_per_batch, 2^ref_rank, input_nt / pow2(ipri)),
    //   where ref_rank = tree.total_rank() + (is_downsampled ? 1 : 0) is the rank of the
    //   one-stage reference tree.
    
    vector<Array<float>> dedispersion_buffers;  // length ntrees

    // Step 3: apply tree dedispersion (one-stage, not two-stage).

    // Warning: don't confuse 'reference_trees' with 'trees' (inherited from base class!)
    vector<shared_ptr<ReferenceTree>> reference_trees;  // length (nbatches * ntrees)
    vector<Array<float>> subband_buffers;               // length (ntrees)

    // Step 4: run peak-finding kernel.
    // In downsampled trees, we just run on the upper half of 'subband_buffers'.
};


ReferenceDedisperser0::ReferenceDedisperser0(const Params &params) :
    ReferenceDedisperserBase(params)
{   
    this->downsampled_inputs.resize(num_primary_trees);
    this->dedispersion_buffers.resize(ntrees);
    this->reference_trees.resize(nbatches * ntrees);
    this->subband_buffers.resize(ntrees);

    for (long ipri = 0; ipri < num_primary_trees; ipri++) {
        long nt_ds = xdiv(nt_in, pow2(ipri));
        downsampled_inputs.at(ipri) = Array<float> ({beams_per_batch, pow2(config.toplevel_tree_rank), nt_ds}, af_uhost | af_zero);
    }
    
    for (long itree = 0; itree < ntrees; itree++) {
        const DedispersionTree &tree = trees.at(itree);

        long out_ntime = tree.nt_ds;
        bool is_downsampled = (tree.primary_tree_index > 0);
        long ref_rank = tree.total_rank() + (is_downsampled ? 1 : 0);
        long ndm_out = tree.ndm_out * (is_downsampled ? 2 : 1);
        long M = tree.frequency_subbands.M;

        this->dedispersion_buffers.at(itree) = Array<float> ({ beams_per_batch, pow2(ref_rank), out_ntime }, af_uhost | af_zero);
        this->subband_buffers.at(itree) = Array<float> ({beams_per_batch, ndm_out, M, tree.nt_ds}, af_uhost | af_zero);

        for (int ibatch = 0; ibatch < nbatches; ibatch++) {
            ReferenceTree::Params tree_params;
            tree_params.num_beams = beams_per_batch;
            tree_params.amb_rank = 0;
            tree_params.dd_rank = ref_rank;
            tree_params.ntime = out_ntime;
            tree_params.nspec = 1;
            tree_params.subband_counts = tree.frequency_subbands.subband_counts;

            this->reference_trees.at(ibatch*ntrees + itree) = make_shared<ReferenceTree> (tree_params);
        }
    }
}


// virtual override
void ReferenceDedisperser0::dedisperse(long ichunk, long ibatch)
{
    // Step 0: Run tree gridding kernel (input_array -> downsampled_inputs.at(0)).
    // In tree-domain-input mode, input_array is already gridded and step 0 is a copy.
    if (params.tree_domain_input)
        downsampled_inputs.at(0).fill(input_array);
    else
        tree_gridding_kernel->apply(downsampled_inputs.at(0), input_array);

    for (int ipri = 1; ipri < num_primary_trees; ipri++) {
        
        // Step 1: downsample input array (straightforward downsample, not "lagged" downsample).
        // Outer length is num_primary_trees.
        // Inner shape is (beams_per_batch, 2^config.toplevel_tree_rank, input_nt / pow2(ipri)).
        // Reminder: 'input_array' is an alias for downsampled_inputs[0].

        Array<float> src = downsampled_inputs.at(ipri-1);
        Array<float> dst = downsampled_inputs.at(ipri);

        // FIXME reference_downsample_time() should operate on N-dimensional array.
        for (long b = 0; b < beams_per_batch; b++) {
            Array<float> src2 = src.slice(0,b);
            Array<float> dst2 = dst.slice(0,b);
            reference_downsample_time(src2, dst2);  // note factor 1/sqrt(2) here!
        }
    }

    for (int itree = 0; itree < ntrees; itree++) {
        const DedispersionTree &tree = trees.at(itree);

        long out_ntime = tree.nt_ds;
        long ipri = tree.primary_tree_index;
        bool is_downsampled = (ipri > 0);
        long ref_rank = tree.total_rank() + (is_downsampled ? 1 : 0);

        Array<float> in = downsampled_inputs.at(ipri).slice(1, 0, pow2(ref_rank));
        Array<float> dd = dedispersion_buffers.at(itree);
        
        // Step 2: copy from 'downsampled_inputs' to 'dedispersion_buffers'.
        
        dd.fill(in);

        // Step 3: apply tree dedispersion (one-stage, not two-stage).
        // Vector length is (nbatches * ntrees).
        // Note dd->dd2 reshape: (B,D,T) -> (B,1,D,T).

        Array<float> dd2 = dd.reshape({beams_per_batch, 1, pow2(ref_rank), out_ntime});
        shared_ptr<ReferenceTree> t = reference_trees.at(ibatch*ntrees + itree);
        t->dedisperse(dd2, subband_buffers.at(itree));

        // Step 4: run peak-finding kernel.
        // In downsampled trees, we just run on the upper half of 'subband_buffers'.

        Array<float> sb = subband_buffers.at(itree);

        if (is_downsampled)
            sb = sb.slice(1, sb.shape[1]/2, sb.shape[1]);

        auto pf_kernel = pf_kernels.at(itree);
        pf_kernel->apply(out_max.at(itree), out_argmax.at(itree), out_var.at(itree), sb, wt_arrays.at(itree), ibatch);
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
    ReferenceDedisperser1(const Params &params);

    // Step 0: Run tree gridding kernel (input_array -> stage1_dd_buf.bufs.at(0)).
    // Step 1: run LaggedDownsampler.
    // Step 2: run stage1 dedispersion kernels 
    // Step 3: copy stage1 -> stage2
    // Step 4: apply lags
    // Step 5: run stage2 dedispersion kernels.
    // Step 6: run peak-finding kernels.
    
    DedispersionBuffer stage1_dd_buf;
    DedispersionBuffer stage2_dd_buf;
    vector<shared_ptr<ReferenceLagbuf>> stage2_lagbufs;  // length (nbatches * ntrees)

    // Dedispersion output in subbands ('sb_out' arg to ReferenceDedispersionKernel::apply())
    // Shape (beams_per_batch, ndm_out, M, nt_in)
    vector<Array<float>> stage2_subband_bufs;
    
    shared_ptr<ReferenceLaggedDownsamplingKernel> lds_kernel;
    vector<shared_ptr<ReferenceDedispersionKernel>> stage1_dd_kernels;
    vector<shared_ptr<ReferenceDedispersionKernel>> stage2_dd_kernels;
    
    virtual void dedisperse(long ichunk, long ibatch) override;
};


ReferenceDedisperser1::ReferenceDedisperser1(const Params &params) :
    ReferenceDedisperserBase(params)
{
    const auto &plan = params.plan;   // local alias -- keeps the plan->... body below unchanged
    this->stage1_dd_buf = _make_dd_buffer(plan->stage1_dd_buf_params);
    this->stage2_dd_buf = _make_dd_buffer(plan->stage2_dd_buf_params);
    this->lds_kernel = make_shared<ReferenceLaggedDownsamplingKernel> (plan->lds_params);

    for (long ipri = 0; ipri < num_primary_trees; ipri++) {
        // In ReferenceDedisperser1, ringbufs are disabled, so make a copy of the dd_params.
        DedispersionKernelParams dd_params = plan->stage1_dd_kernel_params.at(ipri);
        dd_params.output_is_ringbuf = false;
        dd_params.producer_id = -1;

        vector<long> subband_counts = { 1 };  // no subband_counts needed in stage1 kernels
        this->stage1_dd_kernels.push_back(make_shared<ReferenceDedispersionKernel> (dd_params, subband_counts));
    }

    for (long itree = 0; itree < ntrees; itree++) {
        // In ReferenceDedisperser1, ringbufs are disabled, so make a copy of the dd_params.
        DedispersionKernelParams dd_params = plan->stage2_dd_kernel_params.at(itree);  // make copy
        dd_params.input_is_ringbuf = false;   // in ReferenceDeidsperser1, ringbufs are disabled.
        dd_params.consumer_id = -1;

        vector<long> subband_counts = trees.at(itree).frequency_subbands.subband_counts;
        this->stage2_dd_kernels.push_back(make_shared<ReferenceDedispersionKernel> (dd_params, subband_counts));
    }
    
    // Initalize stage2_lagbufs.
    // (Note that these lagbufs are used in ReferenceDedisperser1, but not ReferenceDedisperser2.)
    
    long S = plan->nelts_per_segment;
    this->stage2_lagbufs.resize(nbatches * ntrees);

    for (long itree = 0; itree < ntrees; itree++) {
        const DedispersionTree &tree = trees.at(itree);

        long rank = tree.total_rank();
        long ntime = tree.nt_ds;
        long ipri = tree.primary_tree_index;
        bool is_downsampled = (ipri > 0);
        long stage2_amb_rank = tree.amb_rank;
        long stage2_dd_rank = tree.dd_rank;

        xassert_eq(rank, stage2_amb_rank + stage2_dd_rank);
        xassert_eq(ipri, tree.primary_tree_index);
        xassert_eq(ntime, tree.nt_ds);

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
            stage2_lagbufs.at(b*ntrees + itree) = make_shared<ReferenceLagbuf> (lags, ntime);
    }
    
    // Initialize stage2_subband_bufs.
    // Shape (beams_per_batch, ndm_out, M, nt_in)

    this->stage2_subband_bufs.resize(ntrees);

    for (long itree = 0; itree < ntrees; itree++) {
        const DedispersionTree &tree = trees.at(itree);
        long ndm_out = tree.ndm_out;
        long nt_in = tree.nt_ds;
        long M = tree.frequency_subbands.M;
        stage2_subband_bufs.at(itree) = Array<float> ({beams_per_batch, ndm_out, M, nt_in}, af_uhost | af_zero);
    }
}


// virtual override
void ReferenceDedisperser1::dedisperse(long ichunk, long ibatch)
{
    // Step 0: Run tree gridding kernel (input_array -> stage1_dd_buf.bufs.at(0)).
    // In tree-domain-input mode, input_array is already gridded and step 0 is a copy.
    Array<float> dd = stage1_dd_buf.bufs.at(0).template cast<float> ();
    if (params.tree_domain_input)
        dd.fill(input_array);
    else
        tree_gridding_kernel->apply(dd, input_array);

    // Step 1: run LaggedDownsampler.    
    lds_kernel->apply(stage1_dd_buf, ibatch);

    // Step 2: run stage1 dedispersion kernels.
    for (long ipri = 0; ipri < num_primary_trees; ipri++) {
        shared_ptr<ReferenceDedispersionKernel> kernel = stage1_dd_kernels.at(ipri);
        const DedispersionKernelParams &kp = kernel->params;
        Array<void> dd_buf = stage1_dd_buf.bufs.at(ipri);

        // See comments in DedispersionKernel.hpp for an explanation of this reshape operation.
        dd_buf = dd_buf.reshape({ kp.beams_per_batch, pow2(kp.amb_rank), pow2(kp.dd_rank), kp.ntime }); 

        Array<float> sb_empty;  // no subbands yet
        kernel->apply(dd_buf, dd_buf, sb_empty, ichunk, ibatch);
    }

    // Step 3: copy stage1 -> stage2
    // Step 4: apply lags
    for (long itree = 0; itree < ntrees; itree++) {
        const DedispersionTree &tree = trees.at(itree);

        long rank = tree.total_rank();
        long ipri = tree.primary_tree_index;

        Array<void> src = stage1_dd_buf.bufs.at(ipri);  // shape (beams_per_batch, 2^rank_ambient, ntime)
        src = src.slice(1, 0, pow2(rank));                  // shape (beams_per_batch, 2^rank, ntime)

        Array<void> dst_ = stage2_dd_buf.bufs.at(itree);
        Array<float> dst = dst_.template cast<float> ();
        dst.fill(src);

        auto lagbuf = stage2_lagbufs.at(ibatch*ntrees + itree);
        lagbuf->apply_lags(dst);
    }

    // Step 5: run stage2 dedispersion kernels (in-place).
    // Step 6: run peak-finding kernels.

    for (long itree = 0; itree < ntrees; itree++) {
        shared_ptr<ReferenceDedispersionKernel> dd_kernel = stage2_dd_kernels.at(itree);
        shared_ptr<ReferencePeakFindingKernel> pf_kernel = pf_kernels.at(itree);
        const DedispersionKernelParams &kp = dd_kernel->params;

        // See comments in DedispersionKernel.hpp for an explanation of this reshape/transpose operation.
        Array<void> dd_buf = stage2_dd_buf.bufs.at(itree);
        dd_buf = dd_buf.reshape({ kp.beams_per_batch, pow2(kp.dd_rank), pow2(kp.amb_rank), kp.ntime });
        dd_buf = dd_buf.transpose({0,2,1,3});

        Array<float> sb_buf = stage2_subband_bufs.at(itree);
        dd_kernel->apply(dd_buf, dd_buf, sb_buf, ichunk, ibatch);
        pf_kernel->apply(out_max.at(itree), out_argmax.at(itree), out_var.at(itree), sb_buf, wt_arrays.at(itree), ibatch);
    }
}


// -------------------------------------------------------------------------------------------------
//
// ReferenceDedisperser2: as close to the GPU implementation as possible.


struct ReferenceDedisperser2 : public ReferenceDedisperserBase
{
    ReferenceDedisperser2(const Params &params);
    
    // Step 0: Run tree gridding kernel (input_array -> stage1_dd_buf.bufs.at(0)).
    // Step 1: run LaggedDownsampler.
    // Step 2: run stage1 dedispersion kernels (output to ringbuf)
    // Step 3: extra copying steps needed for early triggers.
    // Step 4: copy host <-> gpu
    // Step 5: run stage2 dedispersion kernels (input from ringbuf)
    // Step 6: run peak-finding kernels.

    DedispersionBuffer stage1_dd_buf;
    DedispersionBuffer stage2_dd_buf;

    // Dedispersion output in subbands ('sb_out' arg to ReferenceDedispersionKernel::apply())
    // Shape (beams_per_batch, ndm_out, M, nt_in)
    vector<Array<float>> stage2_subband_bufs;
    
    Array<float> gpu_ringbuf;
    Array<float> host_ringbuf;

    shared_ptr<ReferenceLaggedDownsamplingKernel> lds_kernel;
    vector<shared_ptr<ReferenceDedispersionKernel>> stage1_dd_kernels;
    vector<shared_ptr<ReferenceDedispersionKernel>> stage2_dd_kernels;
    shared_ptr<CpuRingbufCopyKernel> g2g_copy_kernel;
    shared_ptr<CpuRingbufCopyKernel> h2h_copy_kernel;
    
    virtual void dedisperse(long ichunk, long ibatch) override;
};


ReferenceDedisperser2::ReferenceDedisperser2(const Params &params) :
    ReferenceDedisperserBase(params)
{
    const auto &plan = params.plan;   // local alias -- keeps the plan->... body below unchanged
    this->stage1_dd_buf = _make_dd_buffer(plan->stage1_dd_buf_params);
    this->stage2_dd_buf = _make_dd_buffer(plan->stage2_dd_buf_params);

    long gpu_ringbuf_nelts = plan->mega_ringbuf->gpu_global_nseg * plan->nelts_per_segment;
    long host_ringbuf_nelts = plan->mega_ringbuf->host_global_nseg * plan->nelts_per_segment;
    
    this->gpu_ringbuf = Array<float>({ gpu_ringbuf_nelts }, af_uhost | af_zero);
    this->host_ringbuf = Array<float>({ host_ringbuf_nelts }, af_uhost | af_zero);
    
    this->lds_kernel = make_shared<ReferenceLaggedDownsamplingKernel> (plan->lds_params);
    this->g2g_copy_kernel = make_shared<CpuRingbufCopyKernel> (plan->g2g_copy_kernel_params);
    this->h2h_copy_kernel = make_shared<CpuRingbufCopyKernel> (plan->h2h_copy_kernel_params);
    
    this->stage2_subband_bufs.resize(ntrees);

    for (long itree = 0; itree < ntrees; itree++) {
        const DedispersionTree &tree = trees.at(itree);
        long nt_in = tree.nt_ds;
        long ndm_out = tree.ndm_out;
        long M = tree.frequency_subbands.M;
        stage2_subband_bufs.at(itree) = Array<float> ({beams_per_batch, ndm_out, M, nt_in}, af_uhost | af_zero);
    }

    for (long ipri = 0; ipri < num_primary_trees; ipri++) {
        const DedispersionKernelParams &dd_params = plan->stage1_dd_kernel_params.at(ipri);
        vector<long> subband_counts = { 1 };  // no subbands needed in stage1 kernels
        this->stage1_dd_kernels.push_back(make_shared<ReferenceDedispersionKernel> (dd_params, subband_counts));
    }
    
    for (long itree = 0; itree < ntrees; itree++) {
        const DedispersionKernelParams &dd_params = plan->stage2_dd_kernel_params.at(itree);
        vector<long> subband_counts = trees.at(itree).frequency_subbands.subband_counts;
        this->stage2_dd_kernels.push_back(make_shared<ReferenceDedispersionKernel> (dd_params, subband_counts));
    }
}


void ReferenceDedisperser2::dedisperse(long ichunk, long ibatch)
{
    const auto &plan = params.plan;   // local alias -- keeps the plan->... body below unchanged
    const long BT = this->config.beams_per_gpu;            // total beams
    const long BB = this->config.beams_per_batch;          // beams per batch
    const long BA = this->config.num_active_batches * BB;  // active beams
    const long S = plan->nelts_per_segment;

    long iframe = (ichunk * BT) + (ibatch * BB);

    // Step 0: Run tree gridding kernel (input_array -> stage1_dd_buf.bufs.at(0)).
    // In tree-domain-input mode, input_array is already gridded and step 0 is a copy.
    Array<float> dd = stage1_dd_buf.bufs.at(0).template cast<float> ();
    if (params.tree_domain_input)
        dd.fill(input_array);
    else
        tree_gridding_kernel->apply(dd, input_array);

    // Step 1: run LaggedDownsampler.
    lds_kernel->apply(stage1_dd_buf, ibatch);

    // Step 2: run stage1 dedispersion kernels (output to ringbuf)
    for (long ipri = 0; ipri < num_primary_trees; ipri++) {
        shared_ptr<ReferenceDedispersionKernel> kernel = stage1_dd_kernels.at(ipri);
        const DedispersionKernelParams &kp = kernel->params;
        Array<void> dd_buf = stage1_dd_buf.bufs.at(ipri);

        // See comments in DedispersionKernel.hpp for an explanation of this reshape operation.
        dd_buf = dd_buf.reshape({ kp.beams_per_batch, pow2(kp.amb_rank), pow2(kp.dd_rank), kp.ntime }); 

        Array<float> sb_empty;  // no subbands yet
        kernel->apply(dd_buf, this->gpu_ringbuf, sb_empty, ichunk, ibatch);
    }

    // Step 3: extra copying steps needed for early triggers.

    this->g2g_copy_kernel->apply(this->gpu_ringbuf, ichunk, ibatch);     // gpu -> g2h
    this->h2h_copy_kernel->apply(this->host_ringbuf, ichunk, ibatch);    // host -> et_host

    MegaRingbuf::Zone &eth_zone = plan->mega_ringbuf->et_host_zone;
    MegaRingbuf::Zone &etg_zone = plan->mega_ringbuf->et_gpu_zone;

    xassert(eth_zone.segments_per_frame == etg_zone.segments_per_frame);
    xassert(eth_zone.num_frames == BA);
    xassert(etg_zone.num_frames == BA);

    long soff = eth_zone.segment_offset_of_frame(iframe);
    long doff = etg_zone.segment_offset_of_frame(iframe);
    float *src = this->host_ringbuf.data + (soff * S);
    float *dst = this->gpu_ringbuf.data + (doff * S);
    long nbytes = beams_per_batch * etg_zone.segments_per_frame * S * sizeof(float); 
    memcpy(dst, src, nbytes);  // et_host -> et_gpu (must come after h2h_copy_kernel)
    
    // Step 4: copy host <-> gpu
    
    long max_clag = plan->mega_ringbuf->max_clag;
    xassert(plan->mega_ringbuf->host_zones.size() == uint(max_clag+1));
    xassert(plan->mega_ringbuf->h2g_zones.size() == uint(max_clag+1));
    xassert(plan->mega_ringbuf->g2h_zones.size() == uint(max_clag+1));
    xassert_divisible(BT, BB);   // assert that length-BB copies don't "wrap"
    
    for (int clag = 0; clag <= max_clag; clag++) {
        MegaRingbuf::Zone &host_zone = plan->mega_ringbuf->host_zones.at(clag);
        MegaRingbuf::Zone &h2g_zone = plan->mega_ringbuf->h2g_zones.at(clag);
        MegaRingbuf::Zone &g2h_zone = plan->mega_ringbuf->g2h_zones.at(clag);

        xassert(host_zone.segments_per_frame == g2h_zone.segments_per_frame);
        xassert(host_zone.segments_per_frame == h2g_zone.segments_per_frame);
        xassert(host_zone.num_frames == clag*BT + BA);
        xassert(h2g_zone.num_frames == BA);
        xassert(g2h_zone.num_frames == BA);

        if (host_zone.segments_per_frame == 0)
            continue;

        // First, GPU->host copy.
        // (Ordering of memcopies is arbitrary. On the GPU they happen in parallel.)

        long soff = g2h_zone.segment_offset_of_frame(iframe);
        long doff = host_zone.segment_offset_of_frame(iframe);
        float *src = this->gpu_ringbuf.data + (soff * S);
        float *dst = this->host_ringbuf.data + (doff * S);
        long nbytes = beams_per_batch * host_zone.segments_per_frame * S * sizeof(float);
        memcpy(dst, src, nbytes);

        // Second, host->GPU copy.

        soff = host_zone.segment_offset_of_frame(iframe - clag*BT);
        doff = h2g_zone.segment_offset_of_frame(iframe);
        src = this->host_ringbuf.data + (soff * S);
        dst = this->gpu_ringbuf.data + (doff * S);
        memcpy(dst, src, nbytes);  // same nbytes as before
    }
    
    // Step 5: run stage2 dedispersion kernels (input from ringbuf). 
    // Step 6: run peak-finding kernels.

    for (long itree = 0; itree < ntrees; itree++) {
        shared_ptr<ReferenceDedispersionKernel> dd_kernel = stage2_dd_kernels.at(itree);
        shared_ptr<ReferencePeakFindingKernel> pf_kernel = pf_kernels.at(itree);
        const DedispersionKernelParams &kp = dd_kernel->params;

        // See comments in DedispersionKernel.hpp for an explanation of this reshape/transpose operation.
        Array<void> dd_buf = stage2_dd_buf.bufs.at(itree);
        dd_buf = dd_buf.reshape({ kp.beams_per_batch, pow2(kp.dd_rank), pow2(kp.amb_rank), kp.ntime });
        dd_buf = dd_buf.transpose({0,2,1,3});

        Array<float> sb_buf = stage2_subband_bufs.at(itree);
        dd_kernel->apply(this->gpu_ringbuf, dd_buf, sb_buf, ichunk, ibatch);
        pf_kernel->apply(out_max.at(itree), out_argmax.at(itree), out_var.at(itree), sb_buf, wt_arrays.at(itree), ibatch);
    }
}


// -------------------------------------------------------------------------------------------------


// Static member function
shared_ptr<ReferenceDedisperserBase> ReferenceDedisperserBase::make(const Params &params)
{
    xassert(params.plan);

    if (params.sophistication == 0)
        return make_shared<ReferenceDedisperser0> (params);
    else if (params.sophistication == 1)
        return make_shared<ReferenceDedisperser1> (params);
    else if (params.sophistication == 2)
        return make_shared<ReferenceDedisperser2> (params);
    throw runtime_error("ReferenceDedisperserBase::make(): sophistication must be 0, 1, or 2");
}


}  // namespace pirate
