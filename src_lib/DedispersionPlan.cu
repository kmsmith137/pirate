#include "../include/pirate/DedispersionPlan.hpp"
#include "../include/pirate/MegaRingbuf.hpp"
#include "../include/pirate/constants.hpp"
#include "../include/pirate/inlines.hpp"  // align_up(), pow2(), print_kv(), Indent
#include "../include/pirate/utils.hpp"    // bit_reverse_slow(), rb_lag()

#include <ksgpu/xassert.hpp>

using namespace std;
using namespace ksgpu;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


DedispersionPlan::DedispersionPlan(const DedispersionConfig &config_) :
    config(config_)
{
    config.validate();

    // 'nelts_per_segment' is always (constants::bytes_per_gpu_cache_line / sizeof(dtype)).
    this->nelts_per_segment = config.get_nelts_per_segment();
    this->nbytes_per_segment = constants::bytes_per_gpu_cache_line;

    // Part 1:
    //   - Initialize stage1_trees, stage2_trees.
    //   - Initialize stage1_ntrees, stage2_ntrees.
    
    for (int ids = 0; ids < config.num_downsampling_levels; ids++) {
        
        // Note that Stage1Tree::dd_rank can be different for downsampled trees vs the
        // non-downsampled tree, but is the same for different downsampled trees.
        // This property is necessary in order for the LaggedDownsampler to work later.
        
        int st1_rank = ids ? (config.tree_rank - 1) : config.tree_rank;
        int st1_rank0 = st1_rank/2;
        vector<int> trigger_ranks;
        
        for (const DedispersionConfig::EarlyTrigger &et: config.early_triggers) {
            if (et.ds_level == ids)
                trigger_ranks.push_back(et.tree_rank);
        }

        trigger_ranks.push_back(st1_rank);
        xassert(is_sorted(trigger_ranks));

        Stage1Tree st1;
        st1.ds_level = ids;
        st1.dd_rank = st1_rank0;
        st1.amb_rank = st1_rank - st1.dd_rank;
        st1.nt_ds = xdiv(config.time_samples_per_chunk, pow2(ids));

        // FIXME should replace hardcoded 7,8 by something more descriptive
        // (GpuDedispersionKernel::max_rank?)
        int max_rank = ids ? 7 : 8;
        xassert((st1.dd_rank >= 0) && (st1.dd_rank <= max_rank));
        xassert(st1.nt_ds > 0);
        
        this->stage1_trees.push_back(st1);

        for (int trigger_rank: trigger_ranks) {
            Stage2Tree st2;
            st2.ds_level = ids;
            st2.amb_rank = st1.dd_rank;
            st2.pri_dd_rank = st1.amb_rank;
            st2.early_dd_rank = trigger_rank - st2.amb_rank;
            st2.nt_ds = st1.nt_ds;

            xassert((st2.early_dd_rank >= 0) && (st2.early_dd_rank <= 8));
            xassert(st2.early_dd_rank <= st2.pri_dd_rank);
                     
            this->stage2_trees.push_back(st2);
        }
    }

    this->stage1_ntrees = stage1_trees.size();
    this->stage2_ntrees = stage2_trees.size();
    xassert(stage1_ntrees == config.num_downsampling_levels);
    
    // Part 2:
    //
    // Set up the MegaRingbuf, a central data structure that buffers data between kernels.

    MegaRingbuf::Params mrb_params;
    mrb_params.total_beams = config.beams_per_gpu;
    mrb_params.active_beams = config.num_active_batches * config.beams_per_batch;
    mrb_params.gpu_clag_maxfrac = config.gpu_clag_maxfrac;

    for (const Stage1Tree &st1: this->stage1_trees) {
        long nquads = pow2(st1.dd_rank + st1.amb_rank) * xdiv(st1.nt_ds, nelts_per_segment);
        mrb_params.producer_nquads.push_back(nquads);
    }
    for (const Stage2Tree &st2: this->stage2_trees) {
        long nquads = pow2(st2.amb_rank + st2.early_dd_rank) * xdiv(st2.nt_ds, nelts_per_segment);
        mrb_params.consumer_nquads.push_back(nquads);
    }

    this->mega_ringbuf = std::make_shared<MegaRingbuf>(mrb_params);

    for (int itree2 = 0; itree2 < stage2_ntrees; itree2++) {
        const Stage2Tree &st2 = this->stage2_trees.at(itree2);

        int itree1 = st2.ds_level;
        const Stage1Tree &st1 = this->stage1_trees.at(itree1);

        // Some truly paranoid asserts.
        xassert(st1.nt_ds == st2.nt_ds);
        xassert(st1.dd_rank == st2.amb_rank);
        xassert(st1.ds_level == st2.ds_level);
        xassert(st1.amb_rank == st2.pri_dd_rank);

        // For the stage1 -> stage2 intermediate array, we use variable names
        //   0 <= freq_c < nfreq     (= pow2(st2.early_dd_rank))
        //   0 <= dm_brev < ndm      (= pow2(st2.amb_rank))
        //
        // From the perspective of the stage1 tree, 'dm_brev' is the active dedispersion
        // index, and 'freq_c' is the ambient spectator index. This is reversed for the
        // stage2 tree.

        int ndm = pow2(st2.amb_rank);
        int nfreq_tr = pow2(st2.early_dd_rank);
        int nfreq_amb = pow2(st2.pri_dd_rank);
        
        int ns = xdiv(st2.nt_ds, this->nelts_per_segment);
        bool is_downsampled = (st2.ds_level > 0);
        
        for (int dm_brev = 0; dm_brev < ndm; dm_brev++) {
            for (int freq = 0; freq < nfreq_tr; freq++) {
                int lag = rb_lag(freq, dm_brev, st2.amb_rank, st2.early_dd_rank, is_downsampled);
                int slag = lag / nelts_per_segment;  // segment lag (round down)
                
                for (int ssrc = 0; ssrc < ns; ssrc++) {
                    int clag = (ssrc + slag) / ns;   // chunk lag (see MegaRingbuf)
                    int sdst = (ssrc + slag) - (clag * ns);
                    xassert((sdst >= 0) && (sdst < ns));

                    // Recall that the MegaRingbuf producers/consumers interact with the
                    // buffer via "quadruples", and are free to choose the quadruple ordering.
                    // The stage1 dedispersion kernel (or "producer") uses the ordering:
                    //      (nt_ds / nelts_per_segment, freq, dm_brev)
                    //
                    // The stage2 dedispersion kernel (or "consumer") uses the ordering:
                    //      (nt_ds / nelts<per_segment, dm_brev, freq)
                    //
                    // (Note that in both cases, the active dedipsersion index is fastest varying.)

                    long producer_id = itree1;
                    long producer_iquad = (ssrc * nfreq_amb * ndm) + (freq * ndm) + dm_brev;

                    long consumer_id = itree2;
                    long consumer_iquad = (sdst * ndm * nfreq_tr) + (dm_brev * nfreq_tr) + freq;

                    mega_ringbuf->add_segment(producer_id, producer_iquad, consumer_id, consumer_iquad, clag);
                }
            }
        }
    }
    
    mega_ringbuf->finalize();

    // Part 3: initialize all "params" members:
    //
    //   TreeGriddingKernelParams tree_gridding_kernel_params;
    //   DedispersionBufferParams stage1_dd_buf_params;
    //   DedispersionBufferParams stage2_dd_buf_params;
    //
    //   std::vector<DedispersionKernelParams> stage1_dd_kernel_params;  // length stage1_ntrees
    //   std::vector<DedispersionKernelParams> stage2_dd_kernel_params;  // length stage2_ntrees
    //   std::vector<long> stage2_ds_level;                              // length stage2_ntrees
    //
    //   LaggedDownsamplingKernelParams lds_params;
    //   RingbufCopyKernelParams g2g_copy_kernel_params;
    //   RingbufCopyKernelParams h2h_copy_kernel_params;
    
    // Initialize tree_gridding_kernel_params.
    tree_gridding_kernel_params.channel_map = config.make_channel_map();
    tree_gridding_kernel_params.dtype = config.dtype;
    tree_gridding_kernel_params.nfreq = config.get_total_nfreq();
    tree_gridding_kernel_params.nchan = pow2(config.tree_rank);
    tree_gridding_kernel_params.ntime = config.time_samples_per_chunk;
    tree_gridding_kernel_params.beams_per_batch = config.beams_per_batch;
    tree_gridding_kernel_params.validate();

    // Initialize remaining 'params' members.
    
    stage1_dd_buf_params.dtype = config.dtype;
    stage1_dd_buf_params.beams_per_batch = config.beams_per_batch;
    stage1_dd_buf_params.nbuf = stage1_ntrees;

    for (uint itree1 = 0; itree1 < stage1_trees.size(); itree1++) {
        const Stage1Tree &st1 = stage1_trees.at(itree1);

        DedispersionKernelParams kparams;
        kparams.dtype = config.dtype;
        kparams.dd_rank = st1.dd_rank;
        kparams.amb_rank = st1.amb_rank;
        kparams.total_beams = config.beams_per_gpu;
        kparams.beams_per_batch = config.beams_per_batch;
        kparams.ntime = st1.nt_ds;
        kparams.nspec = 1;
        kparams.input_is_ringbuf = false;
        kparams.output_is_ringbuf = true;   // note output_is_ringbuf = true
        kparams.apply_input_residual_lags = false;
        kparams.input_is_downsampled_tree = (st1.ds_level > 0);
        kparams.nt_per_segment = this->nelts_per_segment;
        kparams.mega_ringbuf = mega_ringbuf;
        kparams.producer_id = itree1;
        kparams.validate();

        stage1_dd_buf_params.buf_rank.push_back(st1.dd_rank + st1.amb_rank);
        stage1_dd_buf_params.buf_ntime.push_back(st1.nt_ds);
        stage1_dd_kernel_params.push_back(kparams);
    }

    stage2_dd_buf_params.dtype = config.dtype;
    stage2_dd_buf_params.beams_per_batch = config.beams_per_batch;
    stage2_dd_buf_params.nbuf = stage2_ntrees;

    for (uint itree2 = 0; itree2 < stage2_trees.size(); itree2++) {
        const Stage2Tree &st2 = stage2_trees.at(itree2);
        long ds_level = st2.ds_level;

        xassert(st2.nt_ds == xdiv(config.time_samples_per_chunk, pow2(ds_level)));

        DedispersionKernelParams kparams;
        kparams.dtype = config.dtype;
        kparams.dd_rank = st2.early_dd_rank;
        kparams.amb_rank = st2.amb_rank;
        kparams.total_beams = config.beams_per_gpu;
        kparams.beams_per_batch = config.beams_per_batch;
        kparams.ntime = st2.nt_ds;
        kparams.nspec = 1;
        kparams.input_is_ringbuf = true;   // note input_is_ringbuf = true
        kparams.output_is_ringbuf = false;
        kparams.apply_input_residual_lags = true;
        kparams.input_is_downsampled_tree = (ds_level > 0);
        kparams.nt_per_segment = this->nelts_per_segment;
        kparams.mega_ringbuf = mega_ringbuf;
        kparams.consumer_id = itree2;
        kparams.validate();
        
        stage2_dd_buf_params.buf_rank.push_back(st2.amb_rank + st2.early_dd_rank);
        stage2_dd_buf_params.buf_ntime.push_back(st2.nt_ds);
        stage2_dd_kernel_params.push_back(kparams);
        stage2_ds_level.push_back(ds_level);

        // The rest of the loop body initializes PeakFindingKernelParams for this stage2 tree.
        const DedispersionConfig::PeakFindingConfig &pfc = config.peak_finding_params.at(ds_level);
        long tot_rank = st2.amb_rank + st2.early_dd_rank;
        long delta_rank = st2.pri_dd_rank - st2.early_dd_rank;
        long pf_rank = (st2.early_dd_rank + 1) / 2;

        // Modify the subband_counts for the stage2 tree.
        // (Accounts for early triggering, downsampling.)
        vector<long> subband_counts = FrequencySubbands::early_subband_counts(config.frequency_subband_counts, delta_rank);
        subband_counts = FrequencySubbands::rerank_subband_counts(subband_counts, pf_rank);

        // Downsampling factors.
        long ndm_in_per_out = pfc.dm_downsampling ? pfc.dm_downsampling : pow2(pf_rank);
        long nt_in_per_out = pfc.time_downsampling ? pfc.time_downsampling : ndm_in_per_out;
        long ndm_in_per_wt = pfc.wt_dm_downsampling;
        long nt_in_per_wt = pfc.wt_time_downsampling;

        xassert_le(ndm_in_per_wt, pow2(tot_rank));
        xassert_le(nt_in_per_out, nt_in_per_wt);
        xassert_le(nt_in_per_wt, st2.nt_ds);

        PeakFindingKernelParams pf_params;
        pf_params.subband_counts = subband_counts;  // not config.frequency_subband_counts
        pf_params.dtype = config.dtype;
        pf_params.max_kernel_width = pfc.max_width;
        pf_params.beams_per_batch = config.beams_per_batch;
        pf_params.total_beams = config.beams_per_gpu;
        pf_params.ndm_out = xdiv(pow2(tot_rank), ndm_in_per_out);
        pf_params.ndm_wt = xdiv(pow2(tot_rank), ndm_in_per_wt);
        pf_params.nt_in = st2.nt_ds;
        pf_params.nt_out = xdiv(pf_params.nt_in, nt_in_per_out);
        pf_params.nt_wt = xdiv(pf_params.nt_in, nt_in_per_wt);
        pf_params.validate();

        stage2_pf_params.push_back(pf_params);
    }

    // Note that 'output_dd_rank' is guaranteed to be the same for all downsampled trees.
    lds_params.dtype = config.dtype;
    lds_params.input_total_rank = config.tree_rank;
    lds_params.output_dd_rank = (stage1_ntrees > 1) ? stage1_trees.at(1).dd_rank : 0;
    lds_params.num_downsampling_levels = config.num_downsampling_levels;
    lds_params.total_beams = config.beams_per_gpu;
    lds_params.beams_per_batch = config.beams_per_batch;
    lds_params.ntime = config.time_samples_per_chunk;

    g2g_copy_kernel_params.total_beams = config.beams_per_gpu;
    g2g_copy_kernel_params.beams_per_batch = config.beams_per_batch;
    g2g_copy_kernel_params.nelts_per_segment = this->nelts_per_segment;
    g2g_copy_kernel_params.octuples = mega_ringbuf->g2g_octuples;
    
    h2h_copy_kernel_params.total_beams = config.beams_per_gpu;
    h2h_copy_kernel_params.beams_per_batch = config.beams_per_batch;
    h2h_copy_kernel_params.nelts_per_segment = this->nelts_per_segment;
    h2h_copy_kernel_params.octuples = mega_ringbuf->h2h_octuples;
    
    lds_params.validate();
    stage1_dd_buf_params.validate();
    stage2_dd_buf_params.validate();
    g2g_copy_kernel_params.validate();
    h2h_copy_kernel_params.validate();
}


// ------------------------------------------------------------------------------------------------


void DedispersionPlan::print(ostream &os, int indent) const
{
    xassert(long(stage1_trees.size()) == stage1_ntrees);
    xassert(long(stage2_trees.size()) == stage2_ntrees);
    
    os << Indent(indent) << "DedispersionPlan" << endl;
    this->config.print(os, indent+4);
    
    print_kv("nelts_per_segment", nelts_per_segment, os, indent);
    print_kv("nbytes_per_segment", nbytes_per_segment, os, indent);

    os << Indent(indent) << "Stage1Trees" << endl;

    for (long i = 0; i < stage1_ntrees; i++) {
        const Stage1Tree &st1 = stage1_trees.at(i);
        
        os << Indent(indent+4) << i
           << ": ds_level=" << st1.ds_level
           << ", rank0=" << st1.dd_rank
           << ", rank1=" << st1.amb_rank
           << ", nt_ds=" << st1.nt_ds
           << endl;
    }
    
    os << Indent(indent) << "Stage2Trees" << endl;

    for (long i = 0; i < stage2_ntrees; i++) {
        const Stage2Tree &st2 = stage2_trees.at(i);
        
        os << Indent(indent+4) << i
           << ": ds_level=" << st2.ds_level
           << ", rank0=" << st2.amb_rank
           << ", rank1_amb=" << st2.pri_dd_rank
           << ", rank1_tri=" << st2.early_dd_rank
           << ", nt_ds=" << st2.nt_ds
           << endl;
    }
}


}  // namespace pirate
