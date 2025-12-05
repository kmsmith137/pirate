#include "../include/pirate/DedispersionPlan.hpp"
#include "../include/pirate/MegaRingbuf.hpp"
#include "../include/pirate/constants.hpp"
#include "../include/pirate/inlines.hpp"  // align_up(), pow2(), print_kv(), Indent
#include "../include/pirate/utils.hpp"    // bit_reverse_slow(), rb_lag(), rstate_len(), mean_bytes_per_unaligned_chunk()

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
    //   - Initialize max_n1 (max number of Stage2Trees, per Stage1Tree).
    //   - Initialize stage1_ntrees, stage2_ntrees.

    int max_n1 = 0;
    
    for (int ids = 0; ids < config.num_downsampling_levels; ids++) {
        
        // Note that Stage1Tree::rank0 can be different for downsampled trees vs the
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
        st1.rank0 = st1_rank0;
        st1.rank1 = st1_rank - st1.rank0;
        st1.nt_ds = xdiv(config.time_samples_per_chunk, pow2(ids));
        st1.segments_per_beam = pow2(st1_rank) * xdiv(st1.nt_ds, nelts_per_segment);
        st1.base_segment = this->stage1_total_segments_per_beam;

        // FIXME should replace hardcoded 7,8 by something more descriptive
        // (GpuDedispersionKernel::max_rank?)
        int max_rank = ids ? 7 : 8;
        xassert((st1.rank0 >= 0) && (st1.rank0 <= max_rank));
        xassert(st1.nt_ds > 0);
        
        this->stage1_trees.push_back(st1);
        this->stage1_total_segments_per_beam += st1.segments_per_beam;

        for (int trigger_rank: trigger_ranks) {
            Stage2Tree st2;
            st2.ds_level = ids;
            st2.rank0 = st1.rank0;
            st2.rank1_ambient = st1.rank1;
            st2.rank1_trigger = trigger_rank - st2.rank0;
            st2.nt_ds = st1.nt_ds;
            st2.segments_per_beam = pow2(trigger_rank) * xdiv(st2.nt_ds, nelts_per_segment);
            st2.base_segment = this->stage2_total_segments_per_beam;

            xassert((st2.rank1_trigger >= 0) && (st2.rank1_trigger <= 8));
            xassert(st2.rank1_trigger <= st2.rank1_ambient);
                     
            this->stage2_trees.push_back(st2);
            this->stage2_total_segments_per_beam += st2.segments_per_beam;
        }

        max_n1 = max(max_n1, int(trigger_ranks.size()));
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

    for (const Stage1Tree &st1: this->stage1_trees)
        mrb_params.producer_nviews.push_back(st1.segments_per_beam);
    for (const Stage2Tree &st2: this->stage2_trees)
        mrb_params.consumer_nviews.push_back(st2.segments_per_beam);

    this->mega_ringbuf = std::make_shared<MegaRingbuf>(mrb_params);

    for (int itree2 = 0; itree2 < stage2_ntrees; itree2++) {
        const Stage2Tree &st2 = this->stage2_trees.at(itree2);

        int itree1 = st2.ds_level;
        const Stage1Tree &st1 = this->stage1_trees.at(itree1);

        // Some truly paranoid asserts.
        xassert(st1.nt_ds == st2.nt_ds);
        xassert(st1.rank0 == st2.rank0);
        xassert(st1.ds_level == st2.ds_level);
        xassert(st1.rank1 == st2.rank1_ambient);

        // For the stage1 -> stage2 intermediate array, we use variable names
        //   0 <= freq_c < nfreq     (= pow2(st2.rank1_trigger))
        //   0 <= dm_brev < ndm      (= pow2(st2.rank0))
        //
        // From the perspective of the stage1 tree, 'dm_brev' is the active dedispersion
        // index, and 'freq_c' is the ambient spectator index. This is reversed for the
        // stage2 tree.

        int ndm = pow2(st2.rank0);
        int nfreq_tr = pow2(st2.rank1_trigger);
        int nfreq_amb = pow2(st2.rank1_ambient);
        
        int ns = xdiv(st2.nt_ds, this->nelts_per_segment);
        bool is_downsampled = (st2.ds_level > 0);
        
        for (int dm_brev = 0; dm_brev < ndm; dm_brev++) {
            for (int freq = 0; freq < nfreq_tr; freq++) {
                int lag = rb_lag(freq, dm_brev, st2.rank0, st2.rank1_trigger, is_downsampled);
                int slag = lag / nelts_per_segment;  // segment lag (round down)
                
                for (int ssrc = 0; ssrc < ns; ssrc++) {
                    int clag = (ssrc + slag) / ns;   // chunk lag (see MegaRingbuf)
                    int sdst = (ssrc + slag) - (clag * ns);
                    xassert((sdst >= 0) && (sdst < ns));

                    // The stage1 dedispersion kernel (or "producer") uses the following ordering 
                    // (or "view") of the MegaRingbuf quadruples:
                    //      (nt_ds / nelts_per_segment, freq, dm_brev)
                    //
                    // The stage2 dedispersion kernel (or "consumer") uses the following ordering:
                    //      (nt_ds / nelts<per_segment, dm_brev, freq)
                    //
                    // (Note that in both cases, the active dedipsersion index is fastest varying.)

                    long producer_id = itree1;
                    long producer_iview = (ssrc * nfreq_amb * ndm) + (freq * ndm) + dm_brev;

                    long consumer_id = itree2;
                    long consumer_iview = (sdst * ndm * nfreq_tr) + (dm_brev * nfreq_tr) + freq;

                    // mega_ringbuf->add_segment(itree1, iview0, itree2, iview1, clag);
                    mega_ringbuf->add_segment(producer_id, producer_iview, consumer_id, consumer_iview, clag);
                }
            }
        }
    }
    
    mega_ringbuf->finalize(true);

    // Part 3: initialize all "params" members:
    //
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
    
    stage1_dd_buf_params.dtype = config.dtype;
    stage1_dd_buf_params.beams_per_batch = config.beams_per_batch;
    stage1_dd_buf_params.nbuf = stage1_ntrees;

    for (uint itree1 = 0; itree1 < stage1_trees.size(); itree1++) {
        const Stage1Tree &st1 = stage1_trees.at(itree1);

        DedispersionKernelParams kparams;
        kparams.dtype = config.dtype;
        kparams.dd_rank = st1.rank0;
        kparams.amb_rank = st1.rank1;
        kparams.total_beams = config.beams_per_gpu;
        kparams.beams_per_batch = config.beams_per_batch;
        kparams.ntime = st1.nt_ds;
        kparams.nspec = 1;
        kparams.input_is_ringbuf = false;
        kparams.output_is_ringbuf = true;   // note output_is_ringbuf = true
        kparams.apply_input_residual_lags = false;
        kparams.input_is_downsampled_tree = (st1.ds_level > 0);
        kparams.nt_per_segment = this->nelts_per_segment;
        kparams.ringbuf_locations = mega_ringbuf->producer_quadruples.at(itree1);
        kparams.ringbuf_nseg = mega_ringbuf->gpu_giant_nseg;
        kparams.validate();
        
        stage1_dd_buf_params.buf_rank.push_back(st1.rank0 + st1.rank1);
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
        kparams.dd_rank = st2.rank1_trigger;
        kparams.amb_rank = st2.rank0;
        kparams.total_beams = config.beams_per_gpu;
        kparams.beams_per_batch = config.beams_per_batch;
        kparams.ntime = st2.nt_ds;
        kparams.nspec = 1;
        kparams.input_is_ringbuf = true;   // note input_is_ringbuf = true
        kparams.output_is_ringbuf = false;
        kparams.apply_input_residual_lags = true;
        kparams.input_is_downsampled_tree = (ds_level > 0);
        kparams.nt_per_segment = this->nelts_per_segment;
        kparams.ringbuf_locations = mega_ringbuf->consumer_quadruples.at(itree2);
        kparams.ringbuf_nseg = mega_ringbuf->gpu_giant_nseg;
        kparams.validate();

        stage2_dd_buf_params.buf_rank.push_back(st2.rank0 + st2.rank1_trigger);
        stage2_dd_buf_params.buf_ntime.push_back(st2.nt_ds);
        stage2_dd_kernel_params.push_back(kparams);
        stage2_ds_level.push_back(ds_level);
    }

    // Note that 'output_dd_rank' is guaranteed to be the same for all downsampled trees.
    lds_params.dtype = config.dtype;
    lds_params.input_total_rank = config.tree_rank;
    lds_params.output_dd_rank = (stage1_ntrees > 1) ? stage1_trees.at(1).rank0 : 0;
    lds_params.num_downsampling_levels = config.num_downsampling_levels;
    lds_params.total_beams = config.beams_per_gpu;
    lds_params.beams_per_batch = config.beams_per_batch;
    lds_params.ntime = config.time_samples_per_chunk;

    g2g_copy_kernel_params.total_beams = config.beams_per_gpu;
    g2g_copy_kernel_params.beams_per_batch = config.beams_per_batch;
    g2g_copy_kernel_params.nelts_per_segment = this->nelts_per_segment;
    g2g_copy_kernel_params.locations = mega_ringbuf->g2g_octuples;
    
    h2h_copy_kernel_params.total_beams = config.beams_per_gpu;
    h2h_copy_kernel_params.beams_per_batch = config.beams_per_batch;
    h2h_copy_kernel_params.nelts_per_segment = this->nelts_per_segment;
    h2h_copy_kernel_params.locations = mega_ringbuf->h2h_octuples;
    
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
           << ", rank0=" << st1.rank0
           << ", rank1=" << st1.rank1
           << ", nt_ds=" << st1.nt_ds
           << endl;
    }
    
    os << Indent(indent) << "Stage2Trees" << endl;

    for (long i = 0; i < stage2_ntrees; i++) {
        const Stage2Tree &st2 = stage2_trees.at(i);
        
        os << Indent(indent+4) << i
           << ": ds_level=" << st2.ds_level
           << ", rank0=" << st2.rank0
           << ", rank1_amb=" << st2.rank1_ambient
           << ", rank1_tri=" << st2.rank1_trigger
           << ", nt_ds=" << st2.nt_ds
           << endl;
    }
}


}  // namespace pirate
