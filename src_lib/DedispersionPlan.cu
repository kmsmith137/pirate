#include "../include/pirate/DedispersionPlan.hpp"
#include "../include/pirate/MegaRingbuf.hpp"
#include "../include/pirate/constants.hpp"
#include "../include/pirate/inlines.hpp"  // align_up(), pow2(), print_kv(), Indent
#include "../include/pirate/utils.hpp"    // bit_reverse_slow(), rb_lag()

#include <sstream>
#include <iomanip>
#include <ksgpu/xassert.hpp>
#include <yaml-cpp/emitter.h>

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

    this->dtype = config.dtype;
    this->nfreq = config.get_total_nfreq();
    this->nt_in = config.time_samples_per_chunk;
    this->num_downsampling_levels = config.num_downsampling_levels;
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
    
    this->stage1_dd_rank.resize(num_downsampling_levels);
    this->stage1_amb_rank.resize(num_downsampling_levels);

    for (long ids = 0; ids < num_downsampling_levels; ids++) {
        
        // Note that stage1_dd_rank can be different for downsampled trees vs the
        // non-downsampled tree, but is the same for different downsampled trees.
        // This property is necessary in order for the LaggedDownsampler to work later.
        
        int total_rank = ids ? (config.tree_rank - 1) : config.tree_rank;
        int st1_dd_rank = (total_rank / 2);
        int st1_amb_rank = (total_rank - st1_dd_rank);

        vector<int> delta_ranks;
        for (const DedispersionConfig::EarlyTrigger &et: config.early_triggers)
            if (et.ds_level == ids)
                delta_ranks.push_back(et.delta_rank);

        delta_ranks.push_back(0);
        xassert(is_sorted(delta_ranks,true)); // reversed=true

        this->stage1_dd_rank.at(ids) = st1_dd_rank;
        this->stage1_amb_rank.at(ids) = st1_amb_rank;

        for (int delta_rank: delta_ranks) {
            DedispersionTree tree;
            tree.ds_level = ids;
            tree.amb_rank = st1_dd_rank;       // note amb <-> dd swap
            tree.pri_dd_rank = st1_amb_rank;   // note amb <-> dd swap
            tree.early_dd_rank = st1_amb_rank - delta_rank;
            tree.nt_ds = xdiv(nt_in, pow2(ids));

            xassert_ge(tree.early_dd_rank, 1);
            xassert_le(tree.early_dd_rank, tree.pri_dd_rank);

            long tot_rank = tree.amb_rank + tree.early_dd_rank;
            long pf_rank = (tree.early_dd_rank + 1) / 2;

            // Frequency range searched by tree, accounting for early trigger.
            long dmax = pow2(config.tree_rank - delta_rank);
            double fmin = config.delay_to_frequency(dmax);
            double fmax = config.zone_freq_edges.back();

            // Modify the subband_counts for the stage2 tree.
            // (Accounts for early triggering, downsampling.)
            vector<long> sc = FrequencySubbands::restrict_subband_counts(config.frequency_subband_counts, delta_rank, pf_rank);
            tree.frequency_subbands = FrequencySubbands(sc, fmin, fmax);

            tree.pf = config.peak_finding_params.at(ids);

            if (tree.pf.dm_downsampling == 0)
                tree.pf.dm_downsampling = pow2(pf_rank);

            if (tree.pf.time_downsampling == 0)
                tree.pf.time_downsampling = tree.pf.dm_downsampling;

            xassert_le(tree.pf.dm_downsampling, tree.pf.wt_dm_downsampling);
            xassert_le(tree.pf.wt_dm_downsampling, pow2(tot_rank));
            xassert_le(tree.pf.time_downsampling, tree.pf.wt_time_downsampling);
            xassert_le(tree.pf.wt_time_downsampling, tree.nt_ds);

            tree.nprofiles = 1 + 3 * integer_log2(tree.pf.max_width);
            tree.ndm_out = xdiv(pow2(tot_rank), tree.pf.dm_downsampling);
            tree.ndm_wt = xdiv(pow2(tot_rank), tree.pf.wt_dm_downsampling);
            tree.nt_out = xdiv(tree.nt_ds, tree.pf.time_downsampling);
            tree.nt_wt = xdiv(tree.nt_ds, tree.pf.wt_time_downsampling);

            double dm0 = config.dm_per_unit_delay() * pow2(config.tree_rank);
            tree.dm_min = dm0 * ((ids > 0) ? pow2(ids-1) : 0);
            tree.dm_max = dm0 * pow2(ids);
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

    for (long ids = 0; ids < num_downsampling_levels; ids++) {
        long tot_rank = stage1_dd_rank.at(ids) + stage1_amb_rank.at(ids);
        long nt_ds = xdiv(nt_in, pow2(ids));
        long nquads = pow2(tot_rank) * xdiv(nt_ds, nelts_per_segment);
        mrb_params.producer_nquads.push_back(nquads);
    }

    for (long itree = 0; itree < ntrees; itree++) {
        const DedispersionTree &tree = trees.at(itree);
        long tot_rank = tree.amb_rank + tree.early_dd_rank;  // not primary_dd_rank
        long nquads = pow2(tot_rank) * xdiv(tree.nt_ds, nelts_per_segment);
        mrb_params.consumer_nquads.push_back(nquads);
    }

    this->mega_ringbuf = std::make_shared<MegaRingbuf>(mrb_params);

    for (int itree = 0; itree < ntrees; itree++) {
        const DedispersionTree &tree = this->trees.at(itree);

        // Some truly paranoid asserts.
        xassert(tree.amb_rank == stage1_dd_rank.at(tree.ds_level));
        xassert(tree.pri_dd_rank == stage1_amb_rank.at(tree.ds_level));
        xassert(tree.nt_ds == xdiv(nt_in, pow2(tree.ds_level)));

        // For the stage1 -> stage2 intermediate array, we use variable names
        //   0 <= freq_c < nfreq     (= pow2(tree.early_dd_rank))
        //   0 <= dm_brev < ndm      (= pow2(tree.amb_rank))
        //
        // From the perspective of the stage1 tree, 'dm_brev' is the active dedispersion
        // index, and 'freq_c' is the ambient spectator index. This is reversed for the
        // stage2 tree.

        int ndm = pow2(tree.amb_rank);
        int nfreq_tr = pow2(tree.early_dd_rank);
        int nfreq_amb = pow2(tree.pri_dd_rank);
        
        int ns = xdiv(tree.nt_ds, this->nelts_per_segment);
        bool is_downsampled = (tree.ds_level > 0);
        
        for (int dm_brev = 0; dm_brev < ndm; dm_brev++) {
            for (int freq = 0; freq < nfreq_tr; freq++) {
                int lag = rb_lag(freq, dm_brev, tree.amb_rank, tree.early_dd_rank, is_downsampled);
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

                    long producer_id = tree.ds_level;
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
    //   std::vector<DedispersionKernelParams> stage1_dd_kernel_params;  // length num_downsampling_levels
    //   std::vector<DedispersionKernelParams> stage2_dd_kernel_params;  // length ntrees
    //
    //   LaggedDownsamplingKernelParams lds_params;
    //   RingbufCopyKernelParams g2g_copy_kernel_params;
    //   RingbufCopyKernelParams h2h_copy_kernel_params;
    
    // Initialize tree_gridding_kernel_params.
    tree_gridding_kernel_params.channel_map = config.make_channel_map();
    tree_gridding_kernel_params.dtype = dtype;
    tree_gridding_kernel_params.nfreq = nfreq;
    tree_gridding_kernel_params.nchan = pow2(config.tree_rank);
    tree_gridding_kernel_params.ntime = nt_in;
    tree_gridding_kernel_params.beams_per_batch = beams_per_batch;
    tree_gridding_kernel_params.validate();

    // Initialize remaining 'params' members.
    
    stage1_dd_buf_params.dtype = dtype;
    stage1_dd_buf_params.beams_per_batch = beams_per_batch;
    stage1_dd_buf_params.nbuf = num_downsampling_levels;

    for (long ids = 0; ids < num_downsampling_levels; ids++) {
        long dd_rank = stage1_dd_rank.at(ids);
        long amb_rank = stage1_amb_rank.at(ids);
        long nt_ds = xdiv(nt_in, pow2(ids));

        DedispersionKernelParams kparams;
        kparams.dtype = dtype;
        kparams.dd_rank = dd_rank;
        kparams.amb_rank = amb_rank;
        kparams.total_beams = beams_per_gpu;
        kparams.beams_per_batch = beams_per_batch;
        kparams.ntime = xdiv(nt_in, pow2(ids));
        kparams.nspec = 1;
        kparams.input_is_ringbuf = false;
        kparams.output_is_ringbuf = true;   // note output_is_ringbuf = true
        kparams.apply_input_residual_lags = false;
        kparams.input_is_downsampled_tree = (ids > 0);
        kparams.nt_per_segment = this->nelts_per_segment;
        kparams.mega_ringbuf = mega_ringbuf;
        kparams.producer_id = ids;
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
        long ds_level = tree.ds_level;

        DedispersionKernelParams kparams;
        kparams.dtype = dtype;
        kparams.dd_rank = tree.early_dd_rank;
        kparams.amb_rank = tree.amb_rank;
        kparams.total_beams = beams_per_gpu;
        kparams.beams_per_batch = beams_per_batch;
        kparams.ntime = tree.nt_ds;
        kparams.nspec = 1;
        kparams.input_is_ringbuf = true;   // note input_is_ringbuf = true
        kparams.output_is_ringbuf = false;
        kparams.apply_input_residual_lags = true;
        kparams.input_is_downsampled_tree = (ds_level > 0);
        kparams.nt_per_segment = this->nelts_per_segment;
        kparams.mega_ringbuf = mega_ringbuf;
        kparams.consumer_id = itree;
        kparams.validate();
        
        stage2_dd_buf_params.buf_rank.push_back(tree.amb_rank + tree.early_dd_rank);
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
        pf_params.validate();

        stage2_pf_params.push_back(pf_params);
    }

    // Note that 'output_dd_rank' is guaranteed to be the same for all downsampled trees.
    lds_params.dtype = dtype;
    lds_params.input_total_rank = config.tree_rank;
    lds_params.output_dd_rank = (num_downsampling_levels > 1) ? stage1_dd_rank.at(1) : 0;
    lds_params.num_downsampling_levels = num_downsampling_levels;
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


void DedispersionPlan::to_yaml(YAML::Emitter &emitter, bool verbose) const
{
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

    emitter << YAML::Key << "num_downsampling_levels" << YAML::Value << num_downsampling_levels;
    if (verbose)
        emitter << YAML::Comment("Number of downsampling levels");

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

    emitter << YAML::Newline << YAML::Newline
            << YAML::Key << "trees"
            << YAML::Value
            << YAML::BeginSeq;

    for (long tree_index = 0; tree_index < ntrees; tree_index++) {
        const DedispersionTree &tree = this->trees.at(tree_index);
        long delta_et = tree.pri_dd_rank - tree.early_dd_rank;
        double time_sample_ms = config.time_sample_ms;
        double ds_factor = pow2(tree.ds_level);
        double max_delay = 1.0e-3 * time_sample_ms * ds_factor * pow2(config.tree_rank - delta_et);

        emitter << YAML::Newline;
        emitter << YAML::BeginMap;

        emitter << YAML::Key << "tree_index" << YAML::Value << tree_index;

        emitter << YAML::Key << "ds_level" << YAML::Value << tree.ds_level;
        if (verbose) {
            stringstream ss;
            ss << (time_sample_ms * ds_factor) << " ms samples"
               << ", DM range [" << tree.dm_min << ", " << tree.dm_max << "]";
            emitter << YAML::Comment(ss.str());
        }

        emitter << YAML::Key << "delta_et" << YAML::Value << delta_et;
        if (verbose) {
            stringstream ss;
            ss << (delta_et > 0 ? "early" : "non-early")
               << " trigger at " << tree.trigger_frequency << " MHz"
               << ", max delay " << max_delay << " seconds";
            emitter << YAML::Comment(ss.str());
        }

        emitter << YAML::Key << "max_width" << YAML::Value << tree.pf.max_width;
        if (verbose) {
            stringstream ss;
            ss << (tree.pf.max_width * ds_factor * time_sample_ms) << " ms";
            emitter << YAML::Comment(ss.str());
        }

        emitter << YAML::Key << "dm_downsampling" << YAML::Value << tree.pf.dm_downsampling;
        if (verbose && (tree.ds_level > 0)) {
            stringstream ss;
            ss << (tree.pf.dm_downsampling * ds_factor) << " before downsampling";
            emitter << YAML::Comment(ss.str());
        }

        emitter << YAML::Key << "time_downsampling" << YAML::Value << tree.pf.time_downsampling;
        if (verbose && (tree.ds_level > 0)) {
            stringstream ss;
            ss << (tree.pf.time_downsampling * ds_factor) << " before downsampling";
            emitter << YAML::Comment(ss.str());
        }

        emitter << YAML::Key << "wt_dm_downsampling" << YAML::Value << tree.pf.wt_dm_downsampling;
        if (verbose && (tree.ds_level > 0)) {
            stringstream ss;
            ss << (tree.pf.wt_dm_downsampling * ds_factor) << " before downsampling";
            emitter << YAML::Comment(ss.str());
        }

        emitter << YAML::Key << "wt_time_downsampling" << YAML::Value << tree.pf.wt_time_downsampling;
        if (verbose && (tree.ds_level > 0)) {
            stringstream ss;
            ss << (tree.pf.wt_time_downsampling * ds_factor) << " before downsampling";
            emitter << YAML::Comment(ss.str());
        }

        if (verbose) {
            const FrequencySubbands &fs = tree.frequency_subbands;
            
            // Note: the multiline comment starting with "# At tree_index=..." is indented
            // by a Python post-processing hack in pirate_frb/__main__.py (show_dedisperser).
            // If you change the format of this comment, update the Python code accordingly!
            stringstream ss;
            ss << "At tree_index=" << tree_index << ", " << fs.F << " frequency subband(s) are searched:\n";
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
    
    mega_ringbuf->to_yaml(emitter, frames_per_second, nfreq, config.time_samples_per_chunk, verbose);

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


string DedispersionPlan::to_yaml_string(bool verbose) const
{
    YAML::Emitter emitter;
    this->to_yaml(emitter, verbose);
    return emitter.c_str();
}


}  // namespace pirate
