#include "../include/pirate/DedispersionPlan.hpp"
#include "../include/pirate/CoalescedDdKernel2.hpp"  // get_registry_dcore()
#include "../include/pirate/MegaRingbuf.hpp"
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


DedispersionPlan::DedispersionPlan(const DedispersionConfig &config_,
                                   bool cdd2_kernel_required_) :
    config(config_), cdd2_kernel_required(cdd2_kernel_required_)
{
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

    // Per-tree geometry lives in the DedispersionTree constructor, which needs no plan and no
    // GPU (see DedispersionTree.hpp). The tree ORDERING is the config's -- see
    // DedispersionConfig::num_dedispersion_trees() and its two companion mapping functions,
    // which this loop and the tree constructor are the two users of.
    this->ntrees = config.num_dedispersion_trees();

    for (long itree = 0; itree < this->ntrees; itree++)
        this->trees.push_back(DedispersionTree(config, itree,
                                               /*Dcore_from_cdd2_registry=*/ cdd2_kernel_required));
    
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
        pf_params.xdm_rank = tree.xdm_rank();
        pf_params.ndm_wt = tree.ndm_wt;
        pf_params.nt_out = tree.nt_out;
        pf_params.nt_wt = tree.nt_wt;
        pf_params.nt_in = tree.nt_ds;
        // Filled in Part 1: from the cdd2 registry if cdd2_kernel_required, else a default.
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
        this->trees.at(tree_index).to_yaml(emitter, config, tree_index, verbose);

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
