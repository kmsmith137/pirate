#include "../include/pirate/DedispersionTree.hpp"
#include "../include/pirate/CoalescedDdKernel2.hpp"  // get_registry_dcore()
#include "../include/pirate/YamlFile.hpp"
#include "../include/pirate/inlines.hpp"  // pow2(), xdiv(), is_power_of_two()
#include "../include/pirate/utils.hpp"    // integer_log2()

#include <sstream>
#include <ksgpu/xassert.hpp>
#include <yaml-cpp/emitter.h>

using namespace std;
using namespace ksgpu;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// -------------------------------------------------------------------------------------------
//
// Constructor.


// The DedispersionPlan constructor builds its trees by looping over primary trees, and within
// each primary tree over DECREASING early-trigger level. This inverts that enumeration, so
// that tree 'itree' can be built without building trees 0..itree-1.
//
// Must be kept consistent with DedispersionConfig::num_dedispersion_trees(), which is the
// same sum without the inversion.
static void _locate_tree(const DedispersionConfig &config, long itree,
                         long *ipri_out, long *et_level_out)
{
    xassert_ge(itree, 0);

    long i = itree;
    for (long ipri = 0; ipri < config.num_primary_trees(); ipri++) {
        long n = config.primary_trees.at(ipri).num_early_triggers + 1;
        if (i < n) {
            *ipri_out = ipri;
            *et_level_out = n - 1 - i;   // et_level descends within a primary tree
            return;
        }
        i -= n;
    }

    throw runtime_error("DedispersionTree: itree=" + to_string(itree) + " is out of range;"
                        " this config has " + to_string(config.num_dedispersion_trees())
                        + " dedispersion trees");
}


// Note: assumes config.validate() has been called (DedispersionPlan's constructor calls it,
// and DedispersionConfig::from_yaml() calls it). Several asserts below -- in particular the
// power-of-two checks on the peak-finding downsampling factors -- are only meaningful for a
// validated config.
DedispersionTree::DedispersionTree(const DedispersionConfig &config, long itree,
                                   bool Dcore_from_cdd2_registry)
{
    long ipri, et_level;
    _locate_tree(config, itree, &ipri, &et_level);

    // Note that stage1_dd_rank can be different for downsampled trees vs the
    // non-downsampled tree, but is the same for different downsampled trees.
    // This property is necessary in order for the LaggedDownsampler to work later.

    int primary_tree_rank = ipri ? (config.toplevel_tree_rank - 1) : config.toplevel_tree_rank;
    int st1_dd_rank = (primary_tree_rank / 2);
    int st1_amb_rank = (primary_tree_rank - st1_dd_rank);

    this->primary_tree_index = ipri;
    this->early_trigger_level = et_level;
    this->amb_rank = st1_dd_rank;              // note amb <-> dd swap
    this->dd_rank = st1_amb_rank - et_level;   // note amb <-> dd swap
    this->nt_ds = xdiv(config.time_samples_per_chunk, pow2(ipri));

    xassert_ge(this->dd_rank, 1);

    long tot_rank = this->total_rank();
    long pf_rank = (this->dd_rank + 1) / 2;

    // Frequency range searched by tree, accounting for early trigger.
    long dmax = pow2(config.toplevel_tree_rank - et_level);
    double fmin = config.delay_to_frequency(dmax);
    double fmax = config.zone_freq_edges.back();

    // Modify the subband_counts for the stage2 tree.
    // (Accounts for early triggering, downsampling.)
    vector<long> sc = FrequencySubbands::restrict_subband_counts(config.frequency_subband_counts, et_level, pf_rank);
    this->frequency_subbands = FrequencySubbands(sc, fmin, fmax);

    this->pf = config.primary_trees.at(ipri);

    if (this->pf.dm_downsampling == 0)
        this->pf.dm_downsampling = pow2(pf_rank);

    if (this->pf.time_downsampling == 0)
        this->pf.time_downsampling = this->pf.dm_downsampling;

    // All four downsampling factors are now powers of two: the wt factors and any
    // explicitly-set dm/time factors are checked by config.validate(); the dm/time factors
    // left at 0 (which validate() leaves unchecked) are pow2() by the auto-fill just above.
    // Assert all four here -- where the resolved values are first established and much
    // downstream code assumes the property.
    xassert(is_power_of_two(this->pf.dm_downsampling));
    xassert(is_power_of_two(this->pf.time_downsampling));
    xassert(is_power_of_two(this->pf.wt_dm_downsampling));
    xassert(is_power_of_two(this->pf.wt_time_downsampling));

    xassert_le(this->pf.dm_downsampling, this->pf.wt_dm_downsampling);
    xassert_le(this->pf.wt_dm_downsampling, pow2(tot_rank));
    xassert_le(this->pf.time_downsampling, this->pf.wt_time_downsampling);
    xassert_le(this->pf.wt_time_downsampling, this->nt_ds);

    this->nprofiles = 1 + 3 * integer_log2(this->pf.max_width);
    this->ndm_out = xdiv(pow2(tot_rank), this->pf.dm_downsampling);
    this->ndm_wt = xdiv(pow2(tot_rank), this->pf.wt_dm_downsampling);
    this->nt_out = xdiv(this->nt_ds, this->pf.time_downsampling);
    this->nt_wt = xdiv(this->nt_ds, this->pf.wt_time_downsampling);

    // Dcore (the peak-finder's internal time-downsampling factor, which sets the time
    // granularity of out_argmax tokens) is a compile-time property of the autogenerated cdd2
    // kernel -- a registry value, not derivable from the config. If Dcore_from_cdd2_registry
    // is set, a missing kernel throws (via get_registry_dcore()). Otherwise a placeholder is
    // assigned: such a tree cannot be used in a GpuDedisperser, so tokens can only come from
    // a ReferenceDedisperser, whose historical convention is Dcore = time_downsampling
    // (= Dout).
    //
    // Note this query needs the (nearly complete) tree, which is why it is last.
    this->Dcore = Dcore_from_cdd2_registry
        ? CoalescedDdKernel2::get_registry_dcore(config.dtype, *this)
        : this->pf.time_downsampling;

    double dm0 = config.dm_per_unit_delay() * pow2(config.toplevel_tree_rank);
    this->dm_min = dm0 * ((ipri > 0) ? pow2(ipri-1) : 0);
    this->dm_max = dm0 * pow2(ipri);
    this->trigger_frequency = fmin;
}


// -------------------------------------------------------------------------------------------
//
// Yaml I/O. See doc-comments in DedispersionTree.hpp.
//
// to_yaml() and from_yaml() are two transcriptions of the same field list, so a field added
// to one must be added to the other. The round-trip unit test in
// pirate_frb/tests/test_decode_argmax.py is what enforces this.


void DedispersionTree::to_yaml(YAML::Emitter &emitter, const DedispersionConfig &config,
                               long tree_index, bool verbose) const
{
    const DedispersionTree &tree = *this;
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


// Static member function.
DedispersionTree DedispersionTree::from_yaml(const YamlFile &yt, const DedispersionConfig &cfg)
{
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

    return tree;
}


string DedispersionTree::to_yaml_string(const DedispersionConfig &config, long tree_index,
                                        bool verbose) const
{
    YAML::Emitter emitter;
    this->to_yaml(emitter, config, tree_index, verbose);
    return emitter.c_str();
}


// Static member function.
DedispersionTree DedispersionTree::from_yaml_string(const string &yaml_string,
                                                    const DedispersionConfig &config)
{
    YamlFile f = YamlFile::from_string(yaml_string, "<dedispersion tree string>");
    return from_yaml(f, config);
}


}  // namespace pirate
