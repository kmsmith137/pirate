#ifndef _PIRATE_DEDISPERSION_TREE_HPP
#define _PIRATE_DEDISPERSION_TREE_HPP

#include <string>
#include <ksgpu/Array.hpp>

#include "DedispersionConfig.hpp"
#include "FrequencySubbands.hpp"

// Forward declarations, so that this header does not pull in yaml-cpp.
namespace YAML { class Emitter; }

namespace pirate {
#if 0
}  // editor auto-indent
#endif


struct YamlFile;


// DedispersionTree: a "data" class representing the output of the dedisperser, for one
// choice of (primary tree, early trigger). Apart from construction and yaml I/O, its member
// functions are trivial.
//
// A vector of DedispersionTrees is created in the DedispersionPlan constructor, and gets copied
// into the dedisperser classes (GpuDedisperser, ReferenceDedisperser).
//
// Constructing trees needs NO DedispersionPlan and NO GPU. That matters because a
// DedispersionPlan cannot be constructed without a CUDA device (its MegaRingbuf allocates
// page-locked host memory), whereas the tree geometry is pure arithmetic -- and analysis code
// such as pirate_frb.varmap wants the geometry on machines with no GPU.

struct DedispersionTree
{
    // Default-constructed trees are field-filled by from_yaml(). Explicit because the
    // (config, itree) constructor below would otherwise suppress it.
    DedispersionTree() = default;

    // Constructs tree 'itree' of 'config', where 0 <= itree < config.num_dedispersion_trees().
    // Trees are ordered by primary tree, then by DECREASING early-trigger level (earliest
    // trigger first, then the main early_trigger_level=0 tree).
    //
    // Dcore_from_cdd2_registry: if true, 'Dcore' is taken from the cdd2 kernel registry, and
    // an exception is thrown if the kernel is missing from this build. If false, 'Dcore' is
    // assigned the placeholder value pf.time_downsampling, which is ReferenceDedisperser's
    // historical convention -- appropriate for callers which do not decode out_argmax tokens.
    // There is deliberately no default value: DedispersionPlan and pirate_frb.varmap want
    // opposite values, and the two ways of getting it wrong fail very differently (a caller
    // which wrongly asks for the registry throws immediately; a caller which wrongly accepts
    // the placeholder gets tokens that decode incorrectly, much later). Invoke it as
    //
    //   DedispersionTree(config, itree, /*Dcore_from_cdd2_registry=*/ true);
    //
    DedispersionTree(const DedispersionConfig &config, long itree,
                     bool Dcore_from_cdd2_registry);

    int primary_tree_index = -1;   // Also identifies associated stage1 tree (input downsampled in time by 2^primary_tree_index).
    int early_trigger_level = -1;  // "Earliness" of trigger: 0 for the main tree, 1..num_early_triggers for early triggers.
    int amb_rank = 0;              // Ambient rank of this DedispersionTree (= dd_rank of associated stage1 tree)
    int dd_rank = 0;               // Active rank of this DedispersionTree (= amb_rank of stage1 tree, minus early_trigger_level)
    int nt_ds = 0;                 // Downsampled time samples per chunk (= config.time_samples_per_chunk / pow2(primary_tree_index))

    // Total tree rank. Equal to (config.toplevel_tree_rank - early_trigger_level - (primary_tree_index ? 1 : 0)).
    long total_rank() const { return amb_rank + dd_rank; }

    // Subbands searched in this tree.
    // Can differ from DedispersionConfig::frequency_subbands, due to early triggers and downsampling.
    FrequencySubbands frequency_subbands;

    // Contains members: num_early_triggers, max_width, {dm,time}_downsampling, wt_{dm,time}_downsampling.
    // Note that {dm,time}_downsampling can be 0 in the config, but are filled with nonzero values here.
    DedispersionConfig::PrimaryTree pf;

    // Internal time-downsampling ("core") factor of this tree's peak-finding kernel; sets
    // the time granularity of out_argmax tokens (see PeakFindingKernelParams::Dcore).
    // A property of the compiled cdd2 kernel (registry value), NOT derivable from the
    // config; equals pf.time_downsampling if the kernel is not compiled into this build.
    long Dcore = 0;

    // Number of time profiles used in peak-finder. (Equal to 1 + 3*log2(pf.max_width).)
    long nprofiles = 0;

    // For peak-finding array shapes.
    // 'wt' array shape is (beams_per_batch, ndm_wt, nt_wt, nprofiles, frequency_subbands.N).
    // 'out_max', 'out_argmax' shapes are (beams_per_batch, ndm_out, nt_out).
    long ndm_out = 0;
    long ndm_wt = 0;
    long nt_out = 0;
    long nt_wt = 0;

    // Currently, these informational members are just used in print-statements.
    double dm_min = 0.0;
    double dm_max = 0.0;
    double trigger_frequency = 0.0f;

    // Decoding peak-finder output for this tree. For a detailed specification, and the
    // definitions of the output params, see the doc-comments on DedispersionPlan's
    // same-named methods, which are thin forwarders to these.
    //
    // 'config' supplies what is not per-tree: the band geometry and the dispersion relation
    // (delay_to_frequency(), dm_per_unit_delay()), which belong to the instrument rather
    // than to one tree. decode_argmax() uses it only to cross-check this tree against the
    // config -- a check with real teeth when the two were deserialized from separate yamls,
    // as FrbGrouper does.
    void decode_argmax(const DedispersionConfig &config, uint argmax_token,
                       long idm_coarse, long itime_coarse,
                       long &fmin, long &fmax, long &tlo, long &thi, long &p) const;

    void decode_argmax2(const DedispersionConfig &config,
                        long fmin, long fmax, long tlo, long thi, long p,
                        double &freq_lo_MHz, double &freq_hi_MHz, double &dm,
                        double &timestamp_samp, double &width_samp) const;

    // Returns a length-ndm_out array; see DedispersionPlan::compute_steady_state_it0().
    // Allocated in ordinary (unregistered) host memory, so this needs no CUDA device.
    ksgpu::Array<long> compute_steady_state_it0(const DedispersionConfig &config) const;

    // Yaml I/O for one tree: the per-tree entry of the DedispersionPlan yaml. Both
    // DedispersionPlan::to_yaml() and FrbGrouper's handshake go through these, so the
    // emitted and parsed field lists cannot drift apart.
    //
    // 'config' supplies what is not on the tree: from_yaml() needs it to seed 'pf' and to
    // reconstruct 'frequency_subbands', and to_yaml() uses it only for verbose comments.
    //
    // from_yaml() is a NAIVE transcription: producer values are adopted verbatim, with no
    // consistency checks against re-derived ones. In particular 'Dcore' is the producer's,
    // which is what makes decode_argmax() correct for producer-generated tokens even if this
    // process runs a different pirate_frb build -- the reason FrbGrouper rebuilds trees this
    // way rather than from its own config. Note that
    // dm_{min,max} and trigger_frequency round-trip LOSSILY (yaml-cpp emits doubles at ~6
    // significant digits); they are print/display values, not used by decode_argmax*().
    // 'tree_index' is emitted as a yaml key, and is not a member of this class (a tree does
    // not know its own position in DedispersionPlan::trees), so the caller supplies it.
    void to_yaml(YAML::Emitter &emitter, const DedispersionConfig &config,
                 long tree_index, bool verbose) const;

    static DedispersionTree from_yaml(const YamlFile &yt, const DedispersionConfig &config);

    // String forms, for callers which store one tree's yaml on its own (rather than as an
    // element of a DedispersionPlan yaml). Used by pirate_frb.varmap, which stores a tree
    // yaml per tree in its variance-map files, and by the python bindings.
    std::string to_yaml_string(const DedispersionConfig &config, long tree_index,
                               bool verbose = false) const;

    static DedispersionTree from_yaml_string(const std::string &yaml_string,
                                             const DedispersionConfig &config);
};


}  // namespace pirate

#endif // _PIRATE_DEDISPERSION_TREE_HPP

