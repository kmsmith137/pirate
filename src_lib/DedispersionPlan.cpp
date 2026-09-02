#include "../include/pirate/DedispersionPlan.hpp"
#include "../include/pirate/CoalescedDdKernel2.hpp"  // get_registry_dcore()
#include "../include/pirate/MegaRingbuf.hpp"
#include "../include/pirate/YamlFile.hpp"  // used in from_yaml()
#include "../include/pirate/constants.hpp"
#include "../include/pirate/inlines.hpp"  // pow2(), xdiv(), is_power_of_two()
#include "../include/pirate/utils.hpp"    // rb_lag(), integer_log2()

#include <sstream>
#include <iomanip>
#include <algorithm>   // std::min, std::max
#include <unordered_map>
#include <ksgpu/xassert.hpp>
#include <yaml-cpp/emitter.h>   // YAML::Emitter, used in to_yaml()

using namespace std;
using namespace ksgpu;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// -------------------------------------------------------------------------------------------
//
// Constructor. It runs in three parts, and DedispersionPlan::Params selects how many of them
// happen: Part 1 (config-derived scalars, stage1 ranks, trees) always, Part 2 (the
// MegaRingbuf, which allocates page-locked host memory and therefore needs a CUDA device) if
// Params::mega_ringbuf, and Part 3 (the gpu kernel params) if Params::gpu_kernels.


// Static member function. See doc-comment in DedispersionPlan.hpp.
DedispersionPlan::Params DedispersionPlan::Params::minimal()
{
    Params ret;
    ret.mega_ringbuf = false;
    ret.gpu_kernels = false;
    ret.dcore_from_cdd2_registry = false;
    return ret;
}


DedispersionPlan::DedispersionPlan(const DedispersionConfig &config_) :
    DedispersionPlan(config_, Params())
{ }


DedispersionPlan::DedispersionPlan(const DedispersionConfig &config_, const Params &params_) :
    config(config_), params(params_)
{
    if (params.gpu_kernels && !params.mega_ringbuf)
        throw runtime_error("DedispersionPlan: Params::gpu_kernels=true"
                            " requires mega_ringbuf=true");

    config.validate();

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
    //
    // THIS IS THE ONLY IMPLEMENTATION OF THE TREE GEOMETRY. Nothing else derives a
    // DedispersionTree from a DedispersionConfig -- which is why every GPU-less consumer of
    // the geometry (pirate_frb.varmap, FrbGrouper) builds a Params::minimal() plan rather
    // than a tree.
    //
    // Note the asserts below assume config.validate() has been called (just above). In
    // particular the power-of-two checks on the peak-finding downsampling factors are only
    // meaningful for a validated config.

    this->stage1_dd_rank.resize(num_primary_trees);
    this->stage1_amb_rank.resize(num_primary_trees);

    for (long ipri = 0; ipri < num_primary_trees; ipri++) {

        // Note that stage1_dd_rank can be different for downsampled trees vs the
        // non-downsampled tree, but is the same for different downsampled trees.
        // This property is necessary in order for the LaggedDownsampler to work later.

        int primary_tree_rank = ipri ? (config.toplevel_tree_rank - 1) : config.toplevel_tree_rank;
        int st1_dd_rank = (primary_tree_rank / 2);

        this->stage1_dd_rank.at(ipri) = st1_dd_rank;
        this->stage1_amb_rank.at(ipri) = primary_tree_rank - st1_dd_rank;
    }

    // The loop order below IS the tree enumeration: primary tree, then DECREASING
    // early-trigger level. This is the only place it is defined, and it is not free to
    // change -- an 'itree' is a bare integer in the plan yaml and in variance-map files, so
    // a different order would silently reinterpret every archived index. (A producer and
    // consumer that disagree do fail loudly: _verify_tree_yaml() checks each tree's
    // primary_tree_index and early_trigger_level against its position in the yaml.)
    //
    // DedispersionPlan::dedispersion_tree_index() is the inverse map.

    for (long ipri = 0; ipri < num_primary_trees; ipri++) {
        long net = config.primary_trees.at(ipri).num_early_triggers;

        for (long et_level = net; et_level >= 0; et_level--) {
            DedispersionTree tree;

            tree.primary_tree_index = ipri;
            tree.early_trigger_level = et_level;
            tree.amb_rank = stage1_dd_rank.at(ipri);                     // note amb <-> dd swap
            tree.dd_rank = stage1_amb_rank.at(ipri) - et_level;          // note amb <-> dd swap
            tree.nt_ds = xdiv(config.time_samples_per_chunk, pow2(ipri));

            xassert_ge(tree.dd_rank, 1);

            long tot_rank = tree.total_rank();
            long dd_rank1 = tree.dd_rank1();   // the GPU kernel's second-stage rank

            // Frequency range searched by tree, accounting for early trigger.
            long dmax = pow2(config.toplevel_tree_rank - et_level);
            double fmin = config.delay_to_frequency(dmax);
            double fmax = config.zone_freq_edges.back();

            // Restrict the config's subband_counts to this tree (accounts for early
            // triggering). The result can have pf_rank < dd_rank1: an early trigger drops
            // subband levels, and nothing puts them back. The kernel folds the difference
            // K = dd_rank1 - pf_rank into its argmax tokens -- see
            // DedispersionTree::xdm_rank() and CoalescedDdKernel2.hpp.
            vector<long> sc = FrequencySubbands::restrict_subband_counts(
                config.frequency_subband_counts, et_level);
            tree.frequency_subbands = FrequencySubbands(sc, fmin, fmax);

            tree.pf = config.primary_trees.at(ipri);

            // NOT pow2(pf_rank): the coarse-grained DM axis is one DM per warp of the
            // kernel's second dedispersion stage, whatever the subbands turn out to be.
            // Pinning it to pow2(dd_rank1) is what holds ndm_out -- and every output array
            // shape -- fixed as pf_rank drops. It is also what makes
            // DedispersionTree::xdm_rank() recoverable from the tree.
            //
            // Not config fields at all: see the doc-comments in DedispersionTree.hpp. Note
            // time_downsampling is what becomes the cdd2 registry key's Dout, so this line
            // and makefile_helper.autogenerated_cdd2_kernels()'s emit loop state the same
            // invariant.
            tree.dm_downsampling = pow2(dd_rank1);
            tree.time_downsampling = pow2(dd_rank1);

            // All four downsampling factors are powers of two: the wt factors are checked by
            // config.validate(), and the two output factors are pow2() by the assignments
            // just above. Assert all four here -- where the resolved values are first
            // established and much downstream code assumes the property.
            xassert(is_power_of_two(tree.dm_downsampling));
            xassert(is_power_of_two(tree.time_downsampling));
            xassert(is_power_of_two(tree.pf.wt_dm_downsampling));
            xassert(is_power_of_two(tree.pf.wt_time_downsampling));

            // The two wt_* lower bounds are also checked by config.validate() (with a message
            // that names the offending primary tree), so these asserts are a backstop on the
            // per-tree values rather than the user-facing check. The time bound is what makes
            // the xdiv() below exact: time_downsampling <= wt_time_downsampling <= nt_ds, all
            // powers of two.
            xassert_le(tree.dm_downsampling, tree.pf.wt_dm_downsampling);
            xassert_le(tree.pf.wt_dm_downsampling, pow2(tot_rank));
            xassert_le(tree.time_downsampling, tree.pf.wt_time_downsampling);
            xassert_le(tree.pf.wt_time_downsampling, tree.nt_ds);

            tree.nprofiles = 1 + 3 * integer_log2(tree.pf.max_width);
            tree.ndm_out = xdiv(pow2(tot_rank), tree.dm_downsampling);
            tree.ndm_wt = xdiv(pow2(tot_rank), tree.pf.wt_dm_downsampling);
            tree.nt_out = xdiv(tree.nt_ds, tree.time_downsampling);
            tree.nt_wt = xdiv(tree.nt_ds, tree.pf.wt_time_downsampling);

            // Dcore (the peak-finder's internal time-downsampling factor, which sets the time
            // granularity of out_argmax tokens) is a compile-time property of the
            // autogenerated cdd2 kernel -- a registry value, not derivable from the config.
            // If params.dcore_from_cdd2_registry is set, a missing kernel throws (via
            // get_registry_dcore(), whose two xassert_eq additionally check this tree's
            // {dm,time}_downsampling against the registry entry's). Otherwise a placeholder
            // is assigned: such a plan cannot be used in a GpuDedisperser, so tokens can only
            // come from a ReferenceDedisperser, whose historical convention is
            // Dcore = time_downsampling (= Dout).
            //
            // Note this query needs the (nearly complete) tree, which is why it is last.
            tree.Dcore = params.dcore_from_cdd2_registry
                ? CoalescedDdKernel2::get_registry_dcore(config.dtype, tree)
                : tree.time_downsampling;

            double dm0 = config.dm_per_unit_delay() * pow2(config.toplevel_tree_rank);
            tree.dm_min = dm0 * ((ipri > 0) ? pow2(ipri-1) : 0);
            tree.dm_max = dm0 * pow2(ipri);
            tree.trigger_frequency = fmin;

            this->trees.push_back(tree);
        }
    }

    this->ntrees = trees.size();

    if (!params.mega_ringbuf)
        return;

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

    if (!params.gpu_kernels)
        return;

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
        pf_params.xdm_rank = tree.xdm_rank();
        pf_params.ndm_wt = tree.ndm_wt;
        pf_params.nt_out = tree.nt_out;
        pf_params.nt_wt = tree.nt_wt;
        pf_params.nt_in = tree.nt_ds;
        // Filled in Part 1: from the cdd2 registry if params.dcore_from_cdd2_registry,
        // else a placeholder.
        pf_params.Dcore = tree.Dcore;
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


// -------------------------------------------------------------------------------------------
//
// to_yaml() and its per-tree helper. The plan yaml is used internally, and is also one of the
// three metadata files sent to the grouper -- see notes/grouper_interface.md.
//
// The per-tree emitter below and DedispersionPlan::from_yaml()'s verifier are two
// transcriptions of one field list, so a field added to one must be added to the other. The
// perturbation loop in pirate_frb/tests/test_decode_argmax.py is what enforces this.


// One entry of the yaml's 'trees' sequence. 'tree_index' is emitted as a yaml key, and is not
// a member of DedispersionTree (a tree does not know its own position in 'trees'), so the
// caller supplies it. 'config' is used only for the verbose comments.
static void _tree_to_yaml(YAML::Emitter &emitter, const DedispersionTree &tree,
                          const DedispersionConfig &config, long tree_index, bool verbose)
{
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

    emitter << YAML::Key << "dm_downsampling" << YAML::Value << tree.dm_downsampling;
    if (verbose && (tree.primary_tree_index > 0)) {
        stringstream ss;
        ss << (tree.dm_downsampling * ds_factor) << " before downsampling";
        emitter << YAML::Comment(ss.str());
    }

    emitter << YAML::Key << "time_downsampling" << YAML::Value << tree.time_downsampling;
    if (verbose && (tree.primary_tree_index > 0)) {
        stringstream ss;
        ss << (tree.time_downsampling * ds_factor) << " before downsampling";
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


void DedispersionPlan::to_yaml(YAML::Emitter &emitter, bool verbose, bool zones) const
{
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
        emitter << YAML::Comment("Ambient rank of each stage1 tree (log2 of the number of coarse freq channels)");

    emitter << YAML::Key << "ntrees" << YAML::Value << ntrees;
    if (verbose)
        emitter << YAML::Comment("Number of output trees (== length of the 'trees' sequence below)");

    if (verbose) {
        emitter << YAML::Newline << YAML::Newline << YAML::Comment(
            "As explained in notes/dedispersion.tex, the dedisperser consists of multiple \"trees\"\n"
            "corresponding to pairs (primary_tree_index, early_trigger_level). Here, primary_tree_index\n"
            "(denoted p) selects the primary tree: the input is time-downsampled by 2^p before dedispersion,\n"
            "which controls the DM-range of the tree. If early_trigger_level > 0, then the tree has an\n"
            "\"early trigger\" and searches a subset of the frequency range (the level is the \"earliness\").\n"
            "\n"
            "The details of the trees are nontrivial -- see notes/dedispersion.tex for info/plots.");
    }

    emitter << YAML::Newline << YAML::Newline
            << YAML::Key << "trees"
            << YAML::Value
            << YAML::BeginSeq;

    for (long tree_index = 0; tree_index < ntrees; tree_index++)
        _tree_to_yaml(emitter, this->trees.at(tree_index), config, tree_index, verbose);

    emitter << YAML::EndSeq;

    // Output mega_ringbuf section. A plan built with Params::mega_ringbuf=false has none,
    // and its yaml is just this document with the key absent. (from_yaml() never reads the
    // key, so both forms parse identically -- which is what lets a GPU-less process write a
    // plan yaml that a GPU process can read back.)
    double T = 1.0e-3 * config.time_samples_per_chunk * config.time_sample_ms;
    double frames_per_second = beams_per_gpu / T;

    if (params.mega_ringbuf) {
        emitter << YAML::Newline << YAML::Newline
                << YAML::Key << "mega_ringbuf"
                << YAML::Value;

        mega_ringbuf->to_yaml(emitter, frames_per_second, nfreq, config.time_samples_per_chunk, verbose, zones);
    }
    else if (verbose) {
        emitter << YAML::Newline << YAML::Newline
                << YAML::Comment("mega_ringbuf: not emitted (plan built with Params::mega_ringbuf=false)");
    }

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


// -------------------------------------------------------------------------------------------
//
// from_yaml(): rebuild a producer's plan. See the doc-comment in DedispersionPlan.hpp.
//
// The plan is CONSTRUCTED from 'config' (so there is one construction path, the one above),
// and the yaml pass is then pure comparison: every key that to_yaml() emits is read and
// checked against the constructed value. The two sides are independent in the way that
// matters -- one was computed by the producer's build, the other by this one -- so the
// comparison has teeth even though both run "the same" code.


// Boilerplate for the comparisons below: 'ctx' is a caller-supplied prefix naming the tree
// (empty for plan-level fields).
template<typename T>
static void _cmp(const string &ctx, const char *name, const T &from_yaml, const T &rebuilt)
{
    if (from_yaml == rebuilt)
        return;

    stringstream ss;
    ss << "DedispersionPlan::from_yaml(): " << ctx << name << " is " << from_yaml
       << " in the yaml but " << rebuilt << " when rebuilt from the config. The config and"
       " the plan describe different instruments -- most likely the producer and this process"
       " are running pirate_frb builds whose dedispersion-tree geometry differs, or"
       " mismatched config/plan yamls were sent.";
    throw runtime_error(ss.str());
}


static string _vec_str(const vector<long> &v)
{
    stringstream ss;
    ss << "[";
    for (size_t i = 0; i < v.size(); i++)
        ss << (i ? ", " : "") << v[i];
    ss << "]";
    return ss.str();
}


// The display doubles (dm_min, dm_max, trigger_frequency) round-trip lossily: yaml-cpp
// emits ~6 significant digits. Compared loosely and never adopted -- the rebuilt values are
// the exact ones.
static void _cmp_double(const string &ctx, const char *name, double from_yaml, double rebuilt)
{
    double tol = 1.0e-4 * std::max(std::abs(rebuilt), 1.0);

    if (std::abs(from_yaml - rebuilt) <= tol)
        return;

    stringstream ss;
    ss << "DedispersionPlan::from_yaml(): " << ctx << name << " is " << from_yaml
       << " in the yaml but " << rebuilt << " when rebuilt from the config (compared at"
       " relative tolerance 1e-4, since yaml-cpp emits ~6 significant digits). The config"
       " and the plan describe different instruments.";
    throw runtime_error(ss.str());
}


// Verifies one block of the yaml's 'trees' sequence against the corresponding rebuilt tree,
// and returns the producer's Dcore (the one field the caller adopts).
static long _verify_tree_yaml(const YamlFile &yt, long itree, const DedispersionTree &ref)
{
    stringstream ctx_ss;
    ctx_ss << "tree " << itree << " (primary_tree_index=" << ref.primary_tree_index
           << ", early_trigger_level=" << ref.early_trigger_level << "): ";
    string ctx = ctx_ss.str();

    _cmp<long>(ctx, "tree_index", yt.get_scalar<long>("tree_index"), itree);
    _cmp<long>(ctx, "primary_tree_index", yt.get_scalar<long>("primary_tree_index"), ref.primary_tree_index);
    _cmp<long>(ctx, "early_trigger_level", yt.get_scalar<long>("early_trigger_level"), ref.early_trigger_level);
    _cmp<long>(ctx, "amb_rank", yt.get_scalar<long>("amb_rank"), ref.amb_rank);
    _cmp<long>(ctx, "dd_rank", yt.get_scalar<long>("dd_rank"), ref.dd_rank);
    _cmp<long>(ctx, "nt_ds", yt.get_scalar<long>("nt_ds"), ref.nt_ds);
    _cmp<long>(ctx, "max_width", yt.get_scalar<long>("max_width"), ref.pf.max_width);
    _cmp<long>(ctx, "dm_downsampling", yt.get_scalar<long>("dm_downsampling"), ref.dm_downsampling);
    _cmp<long>(ctx, "time_downsampling", yt.get_scalar<long>("time_downsampling"), ref.time_downsampling);
    _cmp<long>(ctx, "nprofiles", yt.get_scalar<long>("nprofiles"), ref.nprofiles);
    _cmp<long>(ctx, "wt_dm_downsampling", yt.get_scalar<long>("wt_dm_downsampling"), ref.pf.wt_dm_downsampling);
    _cmp<long>(ctx, "wt_time_downsampling", yt.get_scalar<long>("wt_time_downsampling"), ref.pf.wt_time_downsampling);
    _cmp<long>(ctx, "ndm_wt", yt.get_scalar<long>("ndm_wt"), ref.ndm_wt);
    _cmp<long>(ctx, "nt_wt", yt.get_scalar<long>("nt_wt"), ref.nt_wt);
    _cmp<long>(ctx, "ndm_out", yt.get_scalar<long>("ndm_out"), ref.ndm_out);
    _cmp<long>(ctx, "nt_out", yt.get_scalar<long>("nt_out"), ref.nt_out);

    // The tree's RESTRICTED subband counts: FrequencySubbands::restrict_subband_counts() is
    // the most intricate derivation in the tree, and drift in it would silently reinterpret
    // every subband. (The FrequencySubbands tables derived from these are deliberately not
    // compared -- both sides rebuild them from the counts with the same code.)
    vector<long> sc = yt.get_vector<long>("frequency_subband_counts");
    _cmp<string>(ctx, "frequency_subband_counts", _vec_str(sc),
                 _vec_str(ref.frequency_subbands.subband_counts));

    _cmp_double(ctx, "dm_min", yt.get_scalar<double>("dm_min"), ref.dm_min);
    _cmp_double(ctx, "dm_max", yt.get_scalar<double>("dm_max"), ref.dm_max);
    _cmp_double(ctx, "trigger_frequency", yt.get_scalar<double>("trigger_frequency"), ref.trigger_frequency);

    // 'Dcore' is the one field NOT compared: it is the producer's cdd2-registry value, and
    // the rebuilt tree carries the placeholder. Check its standalone invariant instead --
    // it is otherwise the one member nothing constrains, and it sets the token time
    // granularity (decode_argmax() uses dt = min(Dcore, 2^lpf) and requires t % dt == 0).
    // See PeakFindingKernelParams::Dcore: a power of two dividing (nt_in / nt_out), where
    // the kernel's nt_in is this tree's nt_ds.
    long Dcore = yt.get_scalar<long>("Dcore");
    long Dout = xdiv(ref.nt_ds, ref.nt_out);

    if ((Dcore <= 0) || !is_power_of_two(Dcore) || (Dout % Dcore != 0)) {
        stringstream ss;
        ss << "DedispersionPlan::from_yaml(): " << ctx << "Dcore=" << Dcore << " is not a"
           " power of two dividing nt_ds/nt_out = " << ref.nt_ds << "/" << ref.nt_out
           << ". Dcore is the producer's cdd2-registry value and is adopted verbatim (it is"
           " not re-derived here, which is what lets decoding work across builds), so this"
           " is the only check on it -- and a wrong Dcore mis-decodes the fine-time field of"
           " every out_argmax token.";
        throw runtime_error(ss.str());
    }

    return Dcore;
}


// Static member function.
shared_ptr<DedispersionPlan> DedispersionPlan::from_yaml(const DedispersionConfig &config,
                                                        const YamlFile &plan_yaml)
{
    shared_ptr<DedispersionPlan> plan = make_shared<DedispersionPlan> (config, Params::minimal());
    const string no_ctx = "";

    _cmp<string>(no_ctx, "dtype", plan_yaml.get_scalar<string>("dtype"), plan->dtype.str());
    _cmp<long>(no_ctx, "nfreq", plan_yaml.get_scalar<long>("nfreq"), plan->nfreq);
    _cmp<long>(no_ctx, "nt_in", plan_yaml.get_scalar<long>("nt_in"), plan->nt_in);
    _cmp<long>(no_ctx, "toplevel_tree_rank", plan_yaml.get_scalar<long>("toplevel_tree_rank"),
               config.toplevel_tree_rank);
    _cmp<long>(no_ctx, "num_primary_trees", plan_yaml.get_scalar<long>("num_primary_trees"),
               plan->num_primary_trees);
    _cmp<long>(no_ctx, "beams_per_gpu", plan_yaml.get_scalar<long>("beams_per_gpu"), plan->beams_per_gpu);
    _cmp<long>(no_ctx, "beams_per_batch", plan_yaml.get_scalar<long>("beams_per_batch"), plan->beams_per_batch);
    _cmp<long>(no_ctx, "num_active_batches", plan_yaml.get_scalar<long>("num_active_batches"),
               plan->num_active_batches);
    _cmp<string>(no_ctx, "stage1_dd_rank", _vec_str(plan_yaml.get_vector<long>("stage1_dd_rank")),
                 _vec_str(plan->stage1_dd_rank));
    _cmp<string>(no_ctx, "stage1_amb_rank", _vec_str(plan_yaml.get_vector<long>("stage1_amb_rank")),
                 _vec_str(plan->stage1_amb_rank));
    _cmp<long>(no_ctx, "ntrees", plan_yaml.get_scalar<long>("ntrees"), plan->ntrees);

    YamlFile ytrees = plan_yaml["trees"];
    _cmp<long>(no_ctx, "len(trees)", ytrees.size(), plan->ntrees);

    // Dcore is the one field ADOPTED rather than checked; everything else in the block is
    // compared against the tree the constructor just built.
    for (long itree = 0; itree < plan->ntrees; itree++) {
        DedispersionTree &tree = plan->trees.at(itree);
        tree.Dcore = _verify_tree_yaml(ytrees[itree], itree, tree);
    }

    return plan;
}


// Static member function.
shared_ptr<DedispersionPlan> DedispersionPlan::from_yaml_string(const DedispersionConfig &config,
                                                                const string &plan_yaml)
{
    YamlFile f = YamlFile::from_string(plan_yaml, "dedispersion_plan");
    return from_yaml(config, f);
}


// -------------------------------------------------------------------------------------------
//
// Naming and interpreting trees: dedispersion_tree_index(), decode_argmax(), decode_argmax2(),
// compute_steady_state_it0(), n_index_mapping(), m_index_mapping().
// See the doc-comments in DedispersionPlan.hpp for the full specifications.


long DedispersionPlan::dedispersion_tree_index(long primary_tree_index,
                                               long early_trigger_level) const
{
    for (long itree = 0; itree < this->ntrees; itree++) {
        const DedispersionTree &t = this->trees.at(itree);
        if ((t.primary_tree_index == primary_tree_index)
            && (t.early_trigger_level == early_trigger_level))
            return itree;
    }

    // Not found. Recover the valid ranges from 'trees' so the message can say WHICH argument
    // is wrong and what its bound is. (Callers usually pass a computed index, so "no such
    // tree" on its own would not be enough to debug from.) This is an error path, so the
    // second pass over 'trees' costs nothing in practice.

    long npri = config.num_primary_trees();

    if ((primary_tree_index < 0) || (primary_tree_index >= npri)) {
        stringstream ss;
        ss << "DedispersionPlan::dedispersion_tree_index: primary_tree_index="
           << primary_tree_index << " is out of range [0, " << npri << ")";
        throw runtime_error(ss.str());
    }

    long net = 0;
    for (const DedispersionTree &t: this->trees)
        if (t.primary_tree_index == primary_tree_index)
            net = std::max(net, long(t.early_trigger_level));

    stringstream ss;
    ss << "DedispersionPlan::dedispersion_tree_index: early_trigger_level="
       << early_trigger_level << " is out of range [0, " << net << "] for primary tree "
       << primary_tree_index;
    throw runtime_error(ss.str());
}


// Helper for the five methods below: range-checks an 'itree' argument and returns the tree.
// ('where' is the calling method's name, for the error message; a bare plan.trees.at(itree)
// would throw std::out_of_range with no context at all.)
static const DedispersionTree &_get_tree(const DedispersionPlan &plan, long itree,
                                         const char *where)
{
    if ((itree < 0) || (itree >= plan.ntrees)) {
        stringstream ss;
        ss << "DedispersionPlan::" << where << ": itree=" << itree
           << " is out of range [0, " << plan.ntrees << ")";
        throw runtime_error(ss.str());
    }

    return plan.trees.at(itree);
}


// Background for the formulas below: the token encoding and its time quantization are
// described in PeakFindingKernel.hpp and (for the extra 'mu' field, which is what these
// tokens carry) CoalescedDdKernel2.hpp, the subband time-lag conventions in
// notes/dedispersion.tex (subband search section) and ReferenceTree.cpp, and the
// output-array indexing in the "Dedispersion output arrays" section of the dedispersion
// tex notes.

void DedispersionPlan::decode_argmax(
    uint argmax_token, long itree, long idm_coarse, long itime_coarse,
    long &fmin, long &fmax, long &tlo, long &thi, long &p) const
{
    const DedispersionTree &tr = _get_tree(*this, itree, "decode_argmax");
    const FrequencySubbands &fs = tr.frequency_subbands;

    xassert((idm_coarse >= 0) && (idm_coarse < tr.ndm_out));
    xassert((itime_coarse >= 0) && (itime_coarse < tr.nt_out));

    long Dout = xdiv(tr.nt_ds, tr.nt_out);   // = tr.time_downsampling
    long Dcore = tr.Dcore;                   // token time granularity (see PeakFindingKernel.hpp)

    // Parse token = (t) | (p << 8) | (mu << 16) | (m << (16+K)).
    //
    // The tree's cdd2 kernel computes 2^K output DMs per warp of its second dedispersion
    // stage, where K = xdm_rank(). Those extra DMs do NOT get their own rows of out_max /
    // out_argmax: the index 'mu' is folded into the token's m-field as
    // m_ext = (m << K) | mu, and the peak-finder max-reduces over it. See
    // CoalescedDdKernel2.hpp. K is zero unless the tree has an early trigger.
    //
    // This is the only place that splits the m-field; every producer (cdd2 kernel,
    // ReferencePeakFindingKernel) writes it whole.

    long K = tr.xdm_rank();
    long m_ext = (argmax_token >> 16) & 0xffff;
    long mu    = m_ext & (pow2(K) - 1);
    long m     = m_ext >> K;
    p          = (argmax_token >> 8) & 0xff;
    long t     = argmax_token & 0xff;

    // Note (m < fs.M) is exactly the bound (m_ext < pow2(K) * fs.M), so 'mu' needs no
    // separate range check.
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

    // Frequency: the tree's channels ARE toplevel tree-freq channels (early triggers
    // restrict the search to a prefix; time-downsampling leaves the freq axis alone).
    // Note n_to_toplevel_fhi() is exclusive, whereas the reported 'fmax' is inclusive.

    fmin = tr.n_to_toplevel_flo(n);
    fmax = tr.n_to_toplevel_fhi(n) - 1;

    // Times, first in the tree's (time-downsampled) frame. The trailing pf-input sample
    // read by the winning trial is Tpf. The pf input at time T sums channel f at time
    // (T - Delta(f)), where Delta is exact at the subband edges: Delta(fmax) = Tlag
    // (the extrapolate-to-band-top lag) and Delta(fmin) = Tlag + Dsub (Dsub = delay
    // across the subband).

    // Coarse delay, at the pow2(fs.pf_rank) granularity that the lag formulas below assume:
    // 'idm_coarse' indexes bins of width pow2(pf_rank + K), and 'mu' selects one of the 2^K
    // sub-bins inside it. (Do not confuse 'mu' with 'dfine': 'mu' is the low K bits of the
    // COARSE DM index, whereas 'dfine' is the multiplet's fine DM within its subband -- two
    // different axes.) Downsampled trees search the upper half of a tree one rank larger.

    long dhi = (idm_coarse << K) | mu;

    if (ipri > 0)
        dhi += (tr.ndm_out << K);
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


void DedispersionPlan::decode_argmax2(
    long itree, long fmin, long fmax, long tlo, long thi, long p,
    double &freq_lo_MHz, double &freq_hi_MHz, double &dm,
    double &timestamp_samp, double &width_samp) const
{
    const DedispersionTree &tr = _get_tree(*this, itree, "decode_argmax2");

    long ntree = pow2(config.toplevel_tree_rank);  // note "toplevel"
    long ipri = tr.primary_tree_index;

    xassert_ge(fmin, 0);
    xassert_lt(fmin, fmax);
    xassert_lt(fmax, ntree);  // strict inequality
    xassert_le(tlo, thi);
    xassert_ge(p, 0);
    xassert_lt(p, tr.nprofiles);

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

Array<long> DedispersionPlan::compute_steady_state_it0(long itree) const
{
    const DedispersionTree &tr = _get_tree(*this, itree, "compute_steady_state_it0");

    long p = tr.primary_tree_index;
    long e = tr.early_trigger_level;
    long T_ds = tr.time_downsampling;
    long D_ds = tr.dm_downsampling;
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


// Helper for the index mappings: names a tree by the index the caller passed and the pair
// a caller thinks in, for error messages.
static string _tree_name(long itree, const DedispersionTree &t)
{
    stringstream ss;
    ss << itree << " (primary_tree_index=" << t.primary_tree_index
       << ", early_trigger_level=" << t.early_trigger_level << ")";
    return ss.str();
}


// Helper for n_index_mapping(): a tree's bands as toplevel (flo,fhi) pairs, packed into a
// long. Both endpoints are bounded by pow2(constants::max_tree_rank), so the shift is safe.
static std::unordered_map<long,long> _band_map(const DedispersionTree &t)
{
    std::unordered_map<long,long> ret;
    for (long n = 0; n < t.frequency_subbands.N; n++)
        ret[(t.n_to_toplevel_flo(n) << 32) | t.n_to_toplevel_fhi(n)] = n;
    return ret;
}


vector<long> DedispersionPlan::n_index_mapping(long iparent, long ichild) const
{
    const DedispersionTree &parent = _get_tree(*this, iparent, "n_index_mapping");
    const DedispersionTree &child = _get_tree(*this, ichild, "n_index_mapping");

    std::unordered_map<long,long> pmap = _band_map(parent);
    long Nc = child.frequency_subbands.N;
    vector<long> ret(Nc);

    for (long n = 0; n < Nc; n++) {
        long flo = child.n_to_toplevel_flo(n);
        long fhi = child.n_to_toplevel_fhi(n);
        auto it = pmap.find((flo << 32) | fhi);

        if (it == pmap.end()) {
            // A two-argument function invites a swapped call, so check whether the reverse
            // containment holds and say so if it does.
            std::unordered_map<long,long> cmap = _band_map(child);
            bool reversed = true;
            for (const auto &kv : pmap)
                reversed = reversed && (cmap.count(kv.first) > 0);

            stringstream ss;
            ss << "DedispersionPlan::n_index_mapping(): child tree " << _tree_name(ichild, child)
               << " searches toplevel band [" << flo << "," << fhi << "), which parent tree "
               << _tree_name(iparent, parent) << " does not";
            if (reversed)
                ss << " (arguments may be reversed: every band of the parent IS a band of the child)";
            throw runtime_error(ss.str());
        }

        ret[n] = it->second;
    }

    return ret;
}


vector<long> DedispersionPlan::m_index_mapping(long iparent, long ichild) const
{
    const DedispersionTree &parent = _get_tree(*this, iparent, "m_index_mapping");
    const DedispersionTree &child = _get_tree(*this, ichild, "m_index_mapping");

    const FrequencySubbands &fsp = parent.frequency_subbands;
    const FrequencySubbands &fsc = child.frequency_subbands;

    vector<long> nmap = n_index_mapping(iparent, ichild);
    vector<long> ret(fsc.M);

    for (long m = 0; m < fsc.M; m++) {
        long nc = fsc.m_to_n.at(m);
        long np = nmap.at(nc);

        // Bands are matched by toplevel range, so equal levels are a CONSEQUENCE (one
        // coarse-freq channel is the same width in every tree of a config, so equal ranges
        // force equal levels), not the matching criterion. This can never fire today; it is
        // the tripwire for a future change that makes the channel width tree-dependent.
        long lc = fsc.n_to_level.at(nc);
        long lp = fsp.n_to_level.at(np);

        if (lc != lp) {
            stringstream ss;
            ss << "DedispersionPlan::m_index_mapping(): toplevel band ["
               << child.n_to_toplevel_flo(nc) << "," << child.n_to_toplevel_fhi(nc)
               << ") has subband level " << lc << " in child tree " << _tree_name(ichild, child)
               << ", but level " << lp << " in parent tree " << _tree_name(iparent, parent);
            throw runtime_error(ss.str());
        }

        ret[m] = fsp.n_to_mbase.at(np) + fsc.m_to_d.at(m);
    }

    return ret;
}


}  // namespace pirate
