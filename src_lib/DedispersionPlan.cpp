#include "../include/pirate/DedispersionPlan.hpp"
#include "../include/pirate/CoalescedDdKernel2.hpp"  // get_registry_dcore()
#include "../include/pirate/MegaRingbuf.hpp"
#include "../include/pirate/YamlFile.hpp"  // used in make_incomplete_plan_from_yaml()
#include "../include/pirate/constants.hpp"
#include "../include/pirate/inlines.hpp"  // align_up(), pow2(), print_kv(), Indent
#include "../include/pirate/utils.hpp"    // bit_reverse_slow(), rb_lag()

#include <sstream>
#include <iomanip>
#include <algorithm>   // std::min
#include <ksgpu/xassert.hpp>
#include <yaml-cpp/emitter.h>   // YAML::Emitter, used in to_yaml()

using namespace std;
using namespace ksgpu;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


DedispersionPlan::DedispersionPlan(const DedispersionConfig &config_) :
    DedispersionPlan(config_, Params())
{ }


DedispersionPlan::DedispersionPlan(const DedispersionConfig &config_, const Params &params_) :
    config(config_), params(params_)
{
    config.validate();

    // Incomplete plans take their Dcore values from the producer's yaml, never from
    // the local kernel registry (see Params doc-comment in DedispersionPlan.hpp).
    xassert(!(params.is_incomplete && params.gpu_runnable));

    // Incomplete plans stop here: make_incomplete_plan_from_yaml() fills some (but not
    // all) of the remaining members after the constructor returns.
    if (params.is_incomplete)
        return;

    this->dtype = config.dtype;
    this->nfreq = config.get_total_nfreq();
    this->nt_in = config.time_samples_per_chunk;
    this->num_primary_trees = config.num_primary_trees();
    this->beams_per_gpu = config.beams_per_gpu;
    this->beams_per_batch = config.beams_per_batch;
    this->num_active_batches = config.num_active_batches;

    // Note: DedispersionPlan::nbits is a temporary hack, in order to communicate the bit depth
    // to python. In the future, when ksgpu::Dtype has python bindings, this won't be needed.
    this->nbits = config.dtype.nbits;

    // 'nelts_per_segment' is always (constants::bytes_per_gpu_cache_line / sizeof(dtype)).
    this->nelts_per_segment = config.get_nelts_per_segment();
    this->nbytes_per_segment = constants::bytes_per_gpu_cache_line;

    // Part 1:
    //   - Initialize stage1_dd_rank, stage1_amb_rank.
    //   - Initialize trees, ntrees.
    
    this->stage1_dd_rank.resize(num_primary_trees);
    this->stage1_amb_rank.resize(num_primary_trees);

    for (long ipri = 0; ipri < num_primary_trees; ipri++) {

        // Note that stage1_dd_rank can be different for downsampled trees vs the
        // non-downsampled tree, but is the same for different downsampled trees.
        // This property is necessary in order for the LaggedDownsampler to work later.

        int primary_tree_rank = ipri ? (config.toplevel_tree_rank - 1) : config.toplevel_tree_rank;
        int st1_dd_rank = (primary_tree_rank / 2);
        int st1_amb_rank = (primary_tree_rank - st1_dd_rank);

        this->stage1_dd_rank.at(ipri) = st1_dd_rank;
        this->stage1_amb_rank.at(ipri) = st1_amb_rank;

        // Expand the primary tree into (num_early_triggers+1) stage2 trees, ordered by
        // decreasing et_level (earliest trigger first, then the main et_level=0 tree).
        long num_early_triggers = config.primary_trees.at(ipri).num_early_triggers;

        for (long et_level = num_early_triggers; et_level >= 0; et_level--) {
            DedispersionTree tree;
            tree.primary_tree_index = ipri;
            tree.early_trigger_level = et_level;
            tree.amb_rank = st1_dd_rank;              // note amb <-> dd swap
            tree.dd_rank = st1_amb_rank - et_level;   // note amb <-> dd swap
            tree.nt_ds = xdiv(nt_in, pow2(ipri));

            xassert_ge(tree.dd_rank, 1);

            long tot_rank = tree.total_rank();
            long pf_rank = (tree.dd_rank + 1) / 2;

            // Frequency range searched by tree, accounting for early trigger.
            long dmax = pow2(config.toplevel_tree_rank - et_level);
            double fmin = config.delay_to_frequency(dmax);
            double fmax = config.zone_freq_edges.back();

            // Modify the subband_counts for the stage2 tree.
            // (Accounts for early triggering, downsampling.)
            vector<long> sc = FrequencySubbands::restrict_subband_counts(config.frequency_subband_counts, et_level, pf_rank);
            tree.frequency_subbands = FrequencySubbands(sc, fmin, fmax);

            tree.pf = config.primary_trees.at(ipri);

            if (tree.pf.dm_downsampling == 0)
                tree.pf.dm_downsampling = pow2(pf_rank);

            if (tree.pf.time_downsampling == 0)
                tree.pf.time_downsampling = tree.pf.dm_downsampling;

            // All four downsampling factors are now powers of two: the wt factors and any
            // explicitly-set dm/time factors are checked by config.validate() (called above); the
            // dm/time factors left at 0 (which validate() leaves unchecked) are pow2() by the
            // auto-fill just above. Assert all four here -- where the resolved values are first
            // established and much downstream code assumes the property.
            xassert(is_power_of_two(tree.pf.dm_downsampling));
            xassert(is_power_of_two(tree.pf.time_downsampling));
            xassert(is_power_of_two(tree.pf.wt_dm_downsampling));
            xassert(is_power_of_two(tree.pf.wt_time_downsampling));

            xassert_le(tree.pf.dm_downsampling, tree.pf.wt_dm_downsampling);
            xassert_le(tree.pf.wt_dm_downsampling, pow2(tot_rank));
            xassert_le(tree.pf.time_downsampling, tree.pf.wt_time_downsampling);
            xassert_le(tree.pf.wt_time_downsampling, tree.nt_ds);

            tree.nprofiles = 1 + 3 * integer_log2(tree.pf.max_width);
            tree.ndm_out = xdiv(pow2(tot_rank), tree.pf.dm_downsampling);
            tree.ndm_wt = xdiv(pow2(tot_rank), tree.pf.wt_dm_downsampling);
            tree.nt_out = xdiv(tree.nt_ds, tree.pf.time_downsampling);
            tree.nt_wt = xdiv(tree.nt_ds, tree.pf.wt_time_downsampling);

            // Dcore (the peak-finder's internal time-downsampling factor, which sets the
            // time granularity of out_argmax tokens) is a compile-time property of the
            // autogenerated cdd2 kernel -- a registry value, not derivable from the
            // config. If the plan is gpu_runnable, a missing kernel throws (via
            // get_registry_dcore()). Otherwise a default is assigned: the plan can't be
            // used in a GpuDedisperser, so tokens can only come from a
            // ReferenceDedisperser, whose historical convention is
            // Dcore = time_downsampling (= Dout).
            tree.Dcore = params.gpu_runnable
                ? CoalescedDdKernel2::get_registry_dcore(dtype, tree)
                : tree.pf.time_downsampling;

            double dm0 = config.dm_per_unit_delay() * pow2(config.toplevel_tree_rank);
            tree.dm_min = dm0 * ((ipri > 0) ? pow2(ipri-1) : 0);
            tree.dm_max = dm0 * pow2(ipri);
            tree.trigger_frequency = fmin;

            this->trees.push_back(tree);
        }
    }

    this->ntrees = trees.size();
    
    // Part 2:
    // Set up the MegaRingbuf, a central data structure that buffers data between kernels.
    // See MegaRingbuf.hpp for more info.

    MegaRingbuf::Params mrb_params;
    mrb_params.total_beams = beams_per_gpu;
    mrb_params.active_beams = num_active_batches * beams_per_batch;
    mrb_params.max_gpu_clag = config.max_gpu_clag;

    for (long ipri = 0; ipri < num_primary_trees; ipri++) {
        long primary_tree_rank = stage1_dd_rank.at(ipri) + stage1_amb_rank.at(ipri);
        long nt_ds = xdiv(nt_in, pow2(ipri));
        long nquads = pow2(primary_tree_rank) * xdiv(nt_ds, nelts_per_segment);
        mrb_params.producer_nquads.push_back(nquads);
    }

    for (long itree = 0; itree < ntrees; itree++) {
        const DedispersionTree &tree = trees.at(itree);
        long nquads = pow2(tree.total_rank()) * xdiv(tree.nt_ds, nelts_per_segment);
        mrb_params.consumer_nquads.push_back(nquads);
    }

    this->mega_ringbuf = std::make_shared<MegaRingbuf>(mrb_params);

    for (int itree = 0; itree < ntrees; itree++) {
        const DedispersionTree &tree = this->trees.at(itree);

        // Some truly paranoid asserts.
        xassert(tree.early_trigger_level >= 0);
        xassert(tree.amb_rank == stage1_dd_rank.at(tree.primary_tree_index));
        xassert(tree.dd_rank + tree.early_trigger_level == stage1_amb_rank.at(tree.primary_tree_index));
        xassert(tree.nt_ds == xdiv(nt_in, pow2(tree.primary_tree_index)));

        // For the stage1 -> stage2 intermediate array, we use variable names
        //   0 <= freq_c < nfreq     (= pow2(tree.dd_rank))
        //   0 <= dm_brev < ndm      (= pow2(tree.amb_rank))
        //
        // From the perspective of the stage1 tree, 'dm_brev' is the active dedispersion
        // index, and 'freq_c' is the ambient spectator index. This is reversed for the
        // stage2 tree.

        int ndm = pow2(tree.amb_rank);
        int nfreq_tr = pow2(tree.dd_rank);
        int nfreq_amb = pow2(tree.dd_rank + tree.early_trigger_level);
        
        int ns = xdiv(tree.nt_ds, this->nelts_per_segment);
        bool is_downsampled = (tree.primary_tree_index > 0);
        
        for (int dm_brev = 0; dm_brev < ndm; dm_brev++) {
            for (int freq = 0; freq < nfreq_tr; freq++) {
                int lag = rb_lag(freq, dm_brev, tree.amb_rank, tree.dd_rank, is_downsampled);
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

                    long producer_id = tree.primary_tree_index;
                    long producer_iquad = (ssrc * nfreq_amb * ndm) + (freq * ndm) + dm_brev;

                    long consumer_id = itree;
                    long consumer_iquad = (sdst * ndm * nfreq_tr) + (dm_brev * nfreq_tr) + freq;

                    mega_ringbuf->add_segment(producer_id, producer_iquad, consumer_id, consumer_iquad, clag);
                }
            }
        }
    }
    
    mega_ringbuf->finalize();

    // Part 3: initialize low-level kernel data (*_params members).
    //
    //   TreeGriddingKernelParams tree_gridding_kernel_params;
    //   DedispersionBufferParams stage1_dd_buf_params;
    //   DedispersionBufferParams stage2_dd_buf_params;
    //
    //   std::vector<DedispersionKernelParams> stage1_dd_kernel_params;  // length num_primary_trees
    //   std::vector<DedispersionKernelParams> stage2_dd_kernel_params;  // length ntrees
    //
    //   LaggedDownsamplingKernelParams lds_params;
    //   RingbufCopyKernelParams g2g_copy_kernel_params;
    //   RingbufCopyKernelParams h2h_copy_kernel_params;
    
    // Initialize tree_gridding_kernel_params.
    tree_gridding_kernel_params.channel_map = config.make_channel_map();
    tree_gridding_kernel_params.dtype = dtype;
    tree_gridding_kernel_params.nfreq = nfreq;
    tree_gridding_kernel_params.nchan = pow2(config.toplevel_tree_rank);
    tree_gridding_kernel_params.ntime = nt_in;
    tree_gridding_kernel_params.beams_per_batch = beams_per_batch;
    tree_gridding_kernel_params.validate();

    // Initialize remaining 'params' members.
    
    stage1_dd_buf_params.dtype = dtype;
    stage1_dd_buf_params.beams_per_batch = beams_per_batch;
    stage1_dd_buf_params.nbuf = num_primary_trees;

    for (long ipri = 0; ipri < num_primary_trees; ipri++) {
        long dd_rank = stage1_dd_rank.at(ipri);
        long amb_rank = stage1_amb_rank.at(ipri);
        long nt_ds = xdiv(nt_in, pow2(ipri));

        DedispersionKernelParams kparams;
        kparams.dtype = dtype;
        kparams.dd_rank = dd_rank;
        kparams.amb_rank = amb_rank;
        kparams.total_beams = beams_per_gpu;
        kparams.beams_per_batch = beams_per_batch;
        kparams.ntime = xdiv(nt_in, pow2(ipri));
        kparams.nspec = 1;
        kparams.input_is_ringbuf = false;
        kparams.output_is_ringbuf = true;   // note output_is_ringbuf = true
        kparams.apply_input_residual_lags = false;
        kparams.input_is_downsampled_tree = (ipri > 0);
        kparams.nt_per_segment = this->nelts_per_segment;
        kparams.mega_ringbuf = mega_ringbuf;
        kparams.producer_id = ipri;
        kparams.validate();

        stage1_dd_buf_params.buf_rank.push_back(dd_rank + amb_rank);
        stage1_dd_buf_params.buf_ntime.push_back(nt_ds);
        stage1_dd_kernel_params.push_back(kparams);
    }

    stage2_dd_buf_params.dtype = dtype;
    stage2_dd_buf_params.beams_per_batch = beams_per_batch;
    stage2_dd_buf_params.nbuf = ntrees;

    for (long itree = 0; itree < ntrees; itree++) {
        DedispersionTree &tree = trees.at(itree);
        long ipri = tree.primary_tree_index;

        DedispersionKernelParams kparams;
        kparams.dtype = dtype;
        kparams.dd_rank = tree.dd_rank;
        kparams.amb_rank = tree.amb_rank;
        kparams.total_beams = beams_per_gpu;
        kparams.beams_per_batch = beams_per_batch;
        kparams.ntime = tree.nt_ds;
        kparams.nspec = 1;
        kparams.input_is_ringbuf = true;   // note input_is_ringbuf = true
        kparams.output_is_ringbuf = false;
        kparams.apply_input_residual_lags = true;
        kparams.input_is_downsampled_tree = (ipri > 0);
        kparams.nt_per_segment = this->nelts_per_segment;
        kparams.mega_ringbuf = mega_ringbuf;
        kparams.consumer_id = itree;
        kparams.validate();
        
        stage2_dd_buf_params.buf_rank.push_back(tree.total_rank());
        stage2_dd_buf_params.buf_ntime.push_back(tree.nt_ds);
        stage2_dd_kernel_params.push_back(kparams);

        PeakFindingKernelParams pf_params;
        pf_params.subband_counts = tree.frequency_subbands.subband_counts;  // not config.frequency_subband_counts
        pf_params.dtype = dtype;
        pf_params.max_kernel_width = tree.pf.max_width;
        pf_params.beams_per_batch = beams_per_batch;
        pf_params.total_beams = beams_per_gpu;
        pf_params.ndm_out = tree.ndm_out;
        pf_params.ndm_wt = tree.ndm_wt;
        pf_params.nt_out = tree.nt_out;
        pf_params.nt_wt = tree.nt_wt;
        pf_params.nt_in = tree.nt_ds;
        pf_params.Dcore = tree.Dcore;   // filled in Part 1 (from the cdd2 registry if gpu_runnable, else a default)
        pf_params.validate();

        stage2_pf_params.push_back(pf_params);
    }

    // Note that 'output_dd_rank' is guaranteed to be the same for all downsampled trees.
    lds_params.dtype = dtype;
    lds_params.input_toplevel_rank = config.toplevel_tree_rank;
    lds_params.output_dd_rank = (num_primary_trees > 1) ? stage1_dd_rank.at(1) : 0;
    lds_params.num_primary_trees = num_primary_trees;
    lds_params.total_beams = beams_per_gpu;
    lds_params.beams_per_batch = beams_per_batch;
    lds_params.ntime = nt_in;

    g2g_copy_kernel_params.total_beams = beams_per_gpu;
    g2g_copy_kernel_params.beams_per_batch = beams_per_batch;
    g2g_copy_kernel_params.nelts_per_segment = this->nelts_per_segment;
    g2g_copy_kernel_params.octuples = mega_ringbuf->g2g_octuples;
    
    h2h_copy_kernel_params.total_beams = beams_per_gpu;
    h2h_copy_kernel_params.beams_per_batch = beams_per_batch;
    h2h_copy_kernel_params.nelts_per_segment = this->nelts_per_segment;
    h2h_copy_kernel_params.octuples = mega_ringbuf->h2h_octuples;
    
    lds_params.validate();
    stage1_dd_buf_params.validate();
    stage2_dd_buf_params.validate();
    g2g_copy_kernel_params.validate();
    h2h_copy_kernel_params.validate();
}


// ------------------------------------------------------------------------------------------------
//
// make_incomplete_plan_from_yaml() and helpers. See doc-comment in DedispersionPlan.hpp.


// Static member function.
shared_ptr<DedispersionPlan> DedispersionPlan::make_incomplete_plan_from_yaml(
    const string &config_yaml_str, const string &plan_yaml_str)
{
    // Parse the config first, so that the (const) 'config' member can be initialized
    // in the constructor's init-list. (Note: from_yaml() calls config.validate().)
    DedispersionConfig cfg = DedispersionConfig::from_yaml(
        YamlFile::from_string(config_yaml_str, "dedispersion_config"));

    // With is_incomplete=true, the constructor just sets 'config' and 'params'; the
    // transcription below fills the rest. (gpu_runnable=false: Dcore values are the
    // PRODUCER's, adopted verbatim from the yaml, never from the local kernel registry.)
    Params params;
    params.gpu_runnable = false;
    params.is_incomplete = true;
    shared_ptr<DedispersionPlan> plan = make_shared<DedispersionPlan>(cfg, params);

    // Everything below is a naive transcription of the plan yaml ("dumb" by design: no
    // code shared with the normal constructor path, no consistency asserts against
    // re-derived values, no kernel registry queries -- producer values are adopted
    // verbatim). The "low-level data needed for compute kernels" (mega_ringbuf,
    // kernel/buffer params, nelts_per_segment) is deliberately left uninitialized. The
    // YamlFile accessors throw (naming the key/index) on a missing key or type mismatch;
    // that strictness is what keeps to_yaml() and this function in sync (enforced by the
    // round-trip unit test in test_decode_argmax.py).

    YamlFile py = YamlFile::from_string(plan_yaml_str, "dedispersion_plan");

    plan->dtype = ksgpu::Dtype::from_str(py.get_scalar<string>("dtype"));
    plan->nfreq = py.get_scalar<long>("nfreq");
    plan->nt_in = py.get_scalar<long>("nt_in");
    plan->num_primary_trees = py.get_scalar<long>("num_primary_trees");
    plan->beams_per_gpu = py.get_scalar<long>("beams_per_gpu");
    plan->beams_per_batch = py.get_scalar<long>("beams_per_batch");
    plan->num_active_batches = py.get_scalar<long>("num_active_batches");
    plan->nbits = plan->dtype.nbits;
    plan->stage1_dd_rank = py.get_vector<long>("stage1_dd_rank");
    plan->stage1_amb_rank = py.get_vector<long>("stage1_amb_rank");
    plan->ntrees = py.get_scalar<long>("ntrees");

    YamlFile ytrees = py["trees"];
    xassert(plan->ntrees > 0);
    xassert_eq(plan->ntrees, ytrees.size());

    for (long i = 0; i < plan->ntrees; i++) {
        YamlFile yt = ytrees[i];
        DedispersionTree tree;

        tree.primary_tree_index = yt.get_scalar<int>("primary_tree_index");
        tree.early_trigger_level = yt.get_scalar<int>("early_trigger_level");
        tree.amb_rank = yt.get_scalar<int>("amb_rank");
        tree.dd_rank = yt.get_scalar<int>("dd_rank");
        tree.nt_ds = yt.get_scalar<int>("nt_ds");
        tree.Dcore = yt.get_scalar<long>("Dcore");
        tree.nprofiles = yt.get_scalar<long>("nprofiles");
        tree.ndm_out = yt.get_scalar<long>("ndm_out");
        tree.ndm_wt = yt.get_scalar<long>("ndm_wt");
        tree.nt_out = yt.get_scalar<long>("nt_out");
        tree.nt_wt = yt.get_scalar<long>("nt_wt");

        // Informational members. Note that these round-trip lossily (to_yaml() uses
        // yaml-cpp's default ~6-significant-digit precision for doubles); they are
        // print/display values, not used by decode_argmax*().
        tree.dm_min = yt.get_scalar<double>("dm_min");
        tree.dm_max = yt.get_scalar<double>("dm_max");
        tree.trigger_frequency = yt.get_scalar<double>("trigger_frequency");

        // 'pf' is seeded from the config's PrimaryTree (for num_early_triggers), then
        // the per-tree values are overwritten from the yaml. Note that the config's
        // {dm,time}_downsampling can be 0 (= "choose for me"), but the yaml carries the
        // post-auto-fill values -- the auto-fill rule is deliberately not reimplemented here.
        xassert((tree.primary_tree_index >= 0) && (tree.primary_tree_index < long(cfg.primary_trees.size())));
        tree.pf = cfg.primary_trees.at(tree.primary_tree_index);
        tree.pf.max_width = yt.get_scalar<long>("max_width");
        tree.pf.dm_downsampling = yt.get_scalar<long>("dm_downsampling");
        tree.pf.time_downsampling = yt.get_scalar<long>("time_downsampling");
        tree.pf.wt_dm_downsampling = yt.get_scalar<long>("wt_dm_downsampling");
        tree.pf.wt_time_downsampling = yt.get_scalar<long>("wt_time_downsampling");

        // Same (subband_counts, fmin, fmax) call as the normal constructor path, where
        // fmin == trigger_frequency by construction, and fmax == top edge of the band.
        vector<long> sc = yt.get_vector<long>("frequency_subband_counts");
        tree.frequency_subbands = FrequencySubbands(sc, tree.trigger_frequency, cfg.zone_freq_edges.back());

        // Light local sanity checks (deliberately NOT consistency-vs-rederivation checks).
        xassert(tree.Dcore > 0);
        xassert(tree.nprofiles > 0);
        xassert(tree.ndm_out > 0);
        xassert(tree.nt_out > 0);
        xassert(tree.nt_ds > 0);

        plan->trees.push_back(tree);
    }

    return plan;
}


// ------------------------------------------------------------------------------------------------


// For a detailed specification (and the definitions of the output params), see the
// doc-comment in DedispersionPlan.hpp. Background for the formulas below: the token
// encoding and its time quantization are described in PeakFindingKernel.hpp, the
// subband time-lag conventions in notes/tree_dedispersion.tex (subband search section)
// and ReferenceTree.cpp, and the output-array indexing in the "Dedispersion output
// arrays" section of the tex notes.

void DedispersionPlan::decode_argmax(
    uint argmax_token,
    long itree, long idm_coarse, long itime_coarse,
    long &fmin, long &fmax, long &tlo, long &thi, long &p) const
{
    xassert((itree >= 0) && (itree < ntrees));
    const DedispersionTree &tr = trees.at(itree);
    const FrequencySubbands &fs = tr.frequency_subbands;

    xassert((idm_coarse >= 0) && (idm_coarse < tr.ndm_out));
    xassert((itime_coarse >= 0) && (itime_coarse < tr.nt_out));

    long Dout = xdiv(tr.nt_ds, tr.nt_out);   // = tr.pf.time_downsampling
    long Dcore = tr.Dcore;                   // token time granularity (see PeakFindingKernel.hpp)

    // Parse token = (t) | (p << 8) | (m << 16).
    long m = (argmax_token >> 16) & 0xffff;
    p      = (argmax_token >> 8) & 0xff;
    long t = argmax_token & 0xff;

    xassert_lt(m, fs.M);           // m = multiplet (frequency subband, fine dm)
    xassert_lt(p, tr.nprofiles);   // p = peak-finding profile
    xassert_lt(t, Dout);           // t = fine time within coarse output bin

    // The token's fine time is quantized: t = isamp * dt, where dt = min(Dcore, 2^lpf)
    // and lpf is the peak-finding level (boxcar length 2^lpf) of profile p.
    long lpf = p ? ((p-1)/3) : 0;
    long dt = std::min(Dcore, pow2(lpf));
    xassert_eq(t % dt, 0);

    long n = fs.m_to_n.at(m);                 // frequency subband
    long dfine = fs.m_to_d.at(m);             // fine dm within subband
    long flo = fs.n_to_flo.at(n);             // subband range, in coarse-freq channels
    long fhi = fs.n_to_fhi.at(n);
    long lsb = integer_log2(fhi - flo);       // subband level

    long ipri = tr.primary_tree_index;
    long rr = tr.total_rank() + ((ipri > 0) ? 1 : 0);   // rank of underlying dedispersion
    xassert_eq(rr, config.toplevel_tree_rank - tr.early_trigger_level);

    // Frequency: the tree's channels ARE toplevel tree-freq channels (early triggers
    // restrict the search to a prefix; time-downsampling leaves the freq axis alone).

    long G = pow2(rr - fs.pf_rank);   // toplevel channels per coarse-freq channel
    fmin = flo * G;
    fmax = fhi * G - 1;

    // Times, first in the tree's (time-downsampled) frame. The trailing pf-input sample
    // read by the winning trial is Tpf. The pf input at time T sums channel f at time
    // (T - Delta(f)), where Delta is exact at the subband edges: Delta(fmax) = Tlag
    // (the extrapolate-to-band-top lag) and Delta(fmin) = Tlag + Dsub (Dsub = delay
    // across the subband).

    long dhi = idm_coarse + ((ipri > 0) ? tr.ndm_out : 0);   // coarse delay (downsampled trees search the upper half)
    long Tpf = itime_coarse * Dout + t + dt - 1;
    long thi_ds = Tpf - (pow2(fs.pf_rank) - fhi) * dhi;      // Tpf - Tlag
    long tlo_ds = thi_ds - (dhi * pow2(lsb) + dfine);        // Tpf - Tlag - Dsub

    // Convert to toplevel full-resolution samples. Downsampled sample T covers full-res
    // samples [T*2^ipri, (T+1)*2^ipri - 1]. The reported trailing edge is EXCLUSIVE
    // (one past the last full-res sample summed), i.e. the end boundary of the
    // trailing bin.

    thi = (thi_ds + 1) << ipri;
    tlo = (tlo_ds + 1) << ipri;
}


// See DedispersionPlan.hpp for details of input/output params.
void DedispersionPlan::decode_argmax2(
    long itree, long fmin, long fmax, long tlo, long thi, long p,
    double &freq_lo_MHz, double &freq_hi_MHz, double &dm,
    double &timestamp_samp, double &width_samp) const
{
    xassert_ge(itree, 0);
    xassert_lt(itree, this->ntrees);
    
    long ntree = pow2(config.toplevel_tree_rank);  // note "toplevel"
    long ipri = trees.at(itree).primary_tree_index;
    
    xassert_ge(fmin, 0);
    xassert_lt(fmin, fmax);
    xassert_lt(fmax, ntree);  // strict inequality
    xassert_le(tlo, thi);
    xassert_ge(p, 0);
    xassert_lt(p, trees.at(itree).nprofiles);

    // dispersion delay (in samples) per tree-freq
    double dslope = double(thi-tlo) / double(fmax-fmin);

    // The next block of code computes (based on peak-finding kernel index 0 <= p < P):
    //
    //  pf_width = nominal width of peak-finding kernel, in time samples (not sec or ms)
    //  pf_shift = offset between pf-kernel center-of-mass and "trailing edge" of kernel
    //
    // Currently, we use an informal definition of pf_width, but pf_shift is unambiguous.
    
    long pdiv = p / 3;
    long pmod = p - 3*pdiv;
    double pf_width, pf_shift;

    if (p == 0) {
        // Boxcar of width 2^ipri.
        pf_width = 1.0 * (1 << ipri);
        pf_shift = 0.5 * (1 << ipri);
    }
    else if (pmod == 1) {
        // Boxcar of width 2^{ipri+pdiv+1}
        pf_width = 1.0 * (1 << (ipri+pdiv+1));
        pf_shift = 0.5 * (1 << (ipri+pdiv+1));
    }
    else if (pmod == 2) {
        // kernel = [0.5,1,0.5] upsampled by 2^{ipri+pdiv}.
        pf_width = 2.0 * (1 << (ipri+pdiv));    // let's say pre-upsampled kernel has nomimal width 2
        pf_shift = 1.5 * (1 << (ipri+pdiv));    // pre-upsampled kernel has pshift 1.5 (unambiguous)
    }
    else {
        // kernel = [0.5,1,1,0.5] upsamled by 2^(ipri+pdiv-1)
        pf_width = 3.0 * (1 << (ipri+pdiv-1));   // let's say "base" kernel has nominal width 3
        pf_shift = 2.0 * (1 << (ipri+pdiv-1));   // pre-upsampled kernel has pshift 2.0 (unambiguous)
    }

    // Now we're ready to compute output params.
    // Note that the DM is estimated by converting "dslope" to a full-band delay (ntree tree-freqs)
    // The timestamp is computed as (thi + tdd - pf_shift), where
    //   thi = (trailing-edge timesamp at tree-freq f = fmax + 0.5)
    //   tdd = (dedispersion delay between f=ntree and f=fmax+0.5)
    //   pf_shift = (offset between pulse center and trailing edge)
    
    freq_lo_MHz = config.delay_to_frequency(fmax+1);
    freq_hi_MHz = config.delay_to_frequency(fmin);
    dm = dslope * ntree * config.dm_per_unit_delay();
    timestamp_samp = thi + dslope * (ntree-0.5-fmax) - pf_shift;
    width_samp = pf_width;
}


// See DedispersionPlan.hpp for the meaning of the returned array.
//
// Implementation: element (ichunk, idm, it) of tree 'itree' is unaffected by the
// zero-padding before the start of the acquisition iff
//
//     n*T_ds >= d0 + (idm+1)*D_ds - 1 + 4*Wmax,    n = ichunk*nt_out + it
//
// in "tree" samples (= 2^p input samples; max_width has these units too), where
// T_ds/D_ds are the tree's time/dm downsampling factors, and d0 = d_lo / 2^(e+p) is
// the tree's lowest internal delay (d_lo = 0 for the base tree, 2^(r_top+p-1) for
// downsampled trees). DM bin idm covers internal delays [d0 + idm*D_ds,
// d0 + (idm+1)*D_ds): the dedispersion output at internal delay d and (trigger-freq)
// time tau references input samples [tau - d, tau], subband multiplets reference
// within that range, output time bin n starts at tree sample n*T_ds, and the causal
// peak-finding kernels reach back up to 2*Wmax - 1 more samples (padded to 4*Wmax).
// Solving for the smallest steady-state n (ceil division; exact for integer n) gives
// the per-idm array below.
//
// Note: only uses members transcribed by make_incomplete_plan_from_yaml(), so this
// works on "incomplete" plans (FrbGrouper::_compute_steady_state_it0() relies on this).
Array<long> DedispersionPlan::compute_steady_state_it0(long itree) const
{
    xassert((itree >= 0) && (itree < ntrees));
    const DedispersionTree &tr = trees.at(itree);

    long p = tr.primary_tree_index;
    long e = tr.early_trigger_level;
    long T_ds = tr.pf.time_downsampling;
    long D_ds = tr.pf.dm_downsampling;
    long Wmax = tr.pf.max_width;
    long r_top = config.toplevel_tree_rank;

    long d_lo = (p > 0) ? pow2(r_top + p - 1) : 0;   // lowest full-band delay searched by tree
    long d0 = xdiv(d_lo, pow2(e + p));               // lowest internal delay

    Array<long> ret({tr.ndm_out}, af_uhost);

    for (long idm = 0; idm < tr.ndm_out; idm++) {
        long dmax = d0 + (idm+1) * D_ds - 1;         // max internal delay in DM bin idm
        ret.data[idm] = (dmax + 4*Wmax + T_ds - 1) / T_ds;
    }

    return ret;
}


void DedispersionPlan::to_yaml(YAML::Emitter &emitter, bool verbose, bool zones) const
{
    // to_yaml() walks the mega_ringbuf, which an "incomplete" plan leaves uninitialized
    // (see make_incomplete_plan_from_yaml()).
    xassert(!params.is_incomplete);

    // Top-of-file header comment (verbose only). Note that the 'show_dedisperser' CLI
    // additionally prints a "# Created with: pirate_frb ..." line above this header,
    // recording the exact command line used to generate the file.
    if (verbose) {
        emitter << YAML::Comment(
            "The dedispersion_plan yaml file is used internally by pirate, and is also one of three\n"
            "metadata files sent from pirate to the grouper. (Most fields are only useful internally,\n"
            "and won't be needed in the grouper.)")
                << YAML::Newline;
    }

    emitter << YAML::BeginMap;

    emitter << YAML::Key << "dtype" << YAML::Value << dtype.str();
    if (verbose)
        emitter << YAML::Comment("Data type for dedispersion computations");

    emitter << YAML::Key << "nfreq" << YAML::Value << nfreq;
    if (verbose)
        emitter << YAML::Comment("Total number of frequency channels across all zones");

    emitter << YAML::Key << "nt_in" << YAML::Value << nt_in;
    if (verbose)
        emitter << YAML::Comment("Number of time samples per input chunk");

    emitter << YAML::Key << "toplevel_tree_rank" << YAML::Value << config.toplevel_tree_rank;
    if (verbose)
        emitter << YAML::Comment("Same as config toplevel_tree_rank");

    emitter << YAML::Key << "num_primary_trees" << YAML::Value << num_primary_trees;
    if (verbose)
        emitter << YAML::Comment("Number of primary trees (one per DM range searched)");

    emitter << YAML::Key << "beams_per_gpu" << YAML::Value << beams_per_gpu;
    if (verbose)
        emitter << YAML::Comment("Number of beams processed per GPU");

    emitter << YAML::Key << "beams_per_batch" << YAML::Value << beams_per_batch;
    if (verbose)
        emitter << YAML::Comment("Number of beams per batch");

    emitter << YAML::Key << "num_active_batches" << YAML::Value << num_active_batches;
    if (verbose)
        emitter << YAML::Comment("Number of active batches");

    emitter << YAML::Key << "stage1_dd_rank"
            << YAML::Value << YAML::Flow << YAML::BeginSeq;
    for (long r: stage1_dd_rank)
        emitter << r;
    emitter << YAML::EndSeq;
    if (verbose)
        emitter << YAML::Comment("Active dedispersion rank of each stage1 tree");

    emitter << YAML::Key << "stage1_amb_rank"
            << YAML::Value << YAML::Flow << YAML::BeginSeq;
    for (long r: stage1_amb_rank)
        emitter << r;
    emitter << YAML::EndSeq;
    if (verbose)
        emitter << YAML::Comment("Ambient rank of each stage1 tree (= number of coarse freq channels)");

    emitter << YAML::Key << "ntrees" << YAML::Value << ntrees;
    if (verbose)
        emitter << YAML::Comment("Number of output trees (== length of the 'trees' sequence below)");

    if (verbose) {
        emitter << YAML::Newline << YAML::Newline << YAML::Comment(
            "As explained in notes/tree_dedispersion.tex, the dedisperser consists of multiple \"trees\"\n"
            "corresponding to pairs (primary_tree_index, early_trigger_level). Here, primary_tree_index\n"
            "(denoted p) selects the primary tree: the input is time-downsampled by 2^p before dedispersion,\n"
            "which controls the DM-range of the tree. If early_trigger_level > 0, then the tree has an\n"
            "\"early trigger\" and searches a subset of the frequency range (the level is the \"earliness\").\n"
            "\n"
            "The details of the trees are nontrivial -- see notes/tree_dedispersion.tex for info/plots.");
    }

    emitter << YAML::Newline << YAML::Newline
            << YAML::Key << "trees"
            << YAML::Value
            << YAML::BeginSeq;

    for (long tree_index = 0; tree_index < ntrees; tree_index++) {
        const DedispersionTree &tree = this->trees.at(tree_index);
        long et_level = tree.early_trigger_level;
        double time_sample_ms = config.time_sample_ms;
        double ds_factor = pow2(tree.primary_tree_index);
        double max_delay = 1.0e-3 * time_sample_ms * ds_factor * pow2(config.toplevel_tree_rank - et_level);

        emitter << YAML::Newline;
        emitter << YAML::BeginMap;

        emitter << YAML::Key << "tree_index" << YAML::Value << tree_index;

        emitter << YAML::Key << "ndm_out" << YAML::Value << tree.ndm_out;
        if (verbose)
            emitter << YAML::Comment("Number of output (dedispersed) DM channels");

        emitter << YAML::Key << "nt_out" << YAML::Value << tree.nt_out;
        if (verbose)
            emitter << YAML::Comment("Number of output time samples");

        emitter << YAML::Key << "dm_min" << YAML::Value << tree.dm_min;
        if (verbose)
            emitter << YAML::Comment("Minimum DM (pc/cm^3)");

        emitter << YAML::Key << "dm_max" << YAML::Value << tree.dm_max;
        if (verbose)
            emitter << YAML::Comment("Maximum DM (pc/cm^3)");

        emitter << YAML::Key << "trigger_frequency" << YAML::Value << tree.trigger_frequency;
        if (verbose)
            emitter << YAML::Comment("Early-trigger frequency (MHz)");

        emitter << YAML::Key << "primary_tree_index" << YAML::Value << tree.primary_tree_index;
        if (verbose) {
            stringstream ss;
            ss << (time_sample_ms * ds_factor) << " ms samples"
               << ", DM range [" << tree.dm_min << ", " << tree.dm_max << "]";
            emitter << YAML::Comment(ss.str());
        }

        emitter << YAML::Key << "early_trigger_level" << YAML::Value << et_level;
        if (verbose) {
            stringstream ss;
            ss << (et_level > 0 ? "early" : "non-early")
               << " trigger at " << tree.trigger_frequency << " MHz"
               << ", max delay " << max_delay << " seconds";
            emitter << YAML::Comment(ss.str());
        }

        emitter << YAML::Key << "amb_rank" << YAML::Value << tree.amb_rank;
        if (verbose)
            emitter << YAML::Comment("Ambient rank of this tree (see DedispersionTree.hpp)");

        emitter << YAML::Key << "dd_rank" << YAML::Value << tree.dd_rank;
        if (verbose)
            emitter << YAML::Comment("Active dedispersion rank of this tree (see DedispersionTree.hpp)");

        emitter << YAML::Key << "nt_ds" << YAML::Value << tree.nt_ds;
        if (verbose)
            emitter << YAML::Comment("Downsampled time samples per chunk (= nt_in / 2^primary_tree_index)");

        emitter << YAML::Key << "max_width" << YAML::Value << tree.pf.max_width;
        if (verbose) {
            stringstream ss;
            ss << (tree.pf.max_width * ds_factor * time_sample_ms) << " ms";
            emitter << YAML::Comment(ss.str());
        }

        emitter << YAML::Key << "dm_downsampling" << YAML::Value << tree.pf.dm_downsampling;
        if (verbose && (tree.primary_tree_index > 0)) {
            stringstream ss;
            ss << (tree.pf.dm_downsampling * ds_factor) << " before downsampling";
            emitter << YAML::Comment(ss.str());
        }

        emitter << YAML::Key << "time_downsampling" << YAML::Value << tree.pf.time_downsampling;
        if (verbose && (tree.primary_tree_index > 0)) {
            stringstream ss;
            ss << (tree.pf.time_downsampling * ds_factor) << " before downsampling";
            emitter << YAML::Comment(ss.str());
        }

        emitter << YAML::Key << "Dcore" << YAML::Value << tree.Dcore;
        if (verbose)
            emitter << YAML::Comment("Peak-finder internal time-downsampling (sets out_argmax token granularity)");

        emitter << YAML::Key << "nprofiles" << YAML::Value << tree.nprofiles;
        if (verbose)
            emitter << YAML::Comment("Number of peak-finding profiles (= 1 + 3*log2(max_width))");

        emitter << YAML::Key << "wt_dm_downsampling" << YAML::Value << tree.pf.wt_dm_downsampling;
        if (verbose && (tree.primary_tree_index > 0)) {
            stringstream ss;
            ss << (tree.pf.wt_dm_downsampling * ds_factor) << " before downsampling";
            emitter << YAML::Comment(ss.str());
        }

        emitter << YAML::Key << "wt_time_downsampling" << YAML::Value << tree.pf.wt_time_downsampling;
        if (verbose && (tree.primary_tree_index > 0)) {
            stringstream ss;
            ss << (tree.pf.wt_time_downsampling * ds_factor) << " before downsampling";
            emitter << YAML::Comment(ss.str());
        }

        emitter << YAML::Key << "ndm_wt" << YAML::Value << tree.ndm_wt;
        if (verbose)
            emitter << YAML::Comment("Number of DMs in peak-finding weights array");

        emitter << YAML::Key << "nt_wt" << YAML::Value << tree.nt_wt;
        if (verbose)
            emitter << YAML::Comment("Number of time samples in peak-finding weights array");

        if (verbose) {
            const FrequencySubbands &fs = tree.frequency_subbands;
            
            // Note: the multiline comment starting with "# At tree_index=..." is indented
            // by a Python post-processing hack in pirate_frb/__main__.py (show_dedisperser).
            // If you change the format of this comment, update the Python code accordingly!
            stringstream ss;
            ss << "At tree_index=" << tree_index << ", " << fs.N << " frequency subband(s) are searched:\n";
            fs.show_compact(ss);
            emitter << YAML::Newline << YAML::Newline << YAML::Comment(ss.str()) 
                    << YAML::Newline << YAML::Newline;
        }

        emitter << YAML::Key << "frequency_subband_counts"
                << YAML::Value << YAML::Flow << YAML::BeginSeq;
        for (long n: tree.frequency_subbands.subband_counts)
            emitter << n;
        emitter << YAML::EndSeq;

        emitter << YAML::EndMap;
    }

    emitter << YAML::EndSeq;

    // Output mega_ringbuf section
    double T = 1.0e-3 * config.time_samples_per_chunk * config.time_sample_ms;
    double frames_per_second = beams_per_gpu / T;

    emitter << YAML::Newline << YAML::Newline
            << YAML::Key << "mega_ringbuf"
            << YAML::Value;
    
    mega_ringbuf->to_yaml(emitter, frames_per_second, nfreq, config.time_samples_per_chunk, verbose, zones);

    // Compute dedispersion output bandwidth
    long dd_out_N = 0;
    for (const DedispersionTree &t: trees)
        dd_out_N += t.ndm_out * t.nt_out;
    dd_out_N *= beams_per_gpu;
    double dd_out_gbps = 1.0e-9 * dd_out_N * (4 + dtype.nbits/8) / T;
    
    {
        stringstream ss;
        ss << fixed << setprecision(3) << dd_out_gbps << " GB/s";
        emitter << YAML::Newline << YAML::Newline
                << YAML::Key << "dedispersion_outputs" << YAML::Value << ss.str();
    }

    emitter << YAML::EndMap;
}


string DedispersionPlan::to_yaml_string(bool verbose, bool zones) const
{
    YAML::Emitter emitter;
    this->to_yaml(emitter, verbose, zones);
    return emitter.c_str();
}


}  // namespace pirate
