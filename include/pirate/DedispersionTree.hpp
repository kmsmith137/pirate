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

    // decode_argmax2() and compute_steady_state_it0() take a DedispersionConfig, because
    // they need what is not per-tree: the band geometry and the dispersion relation
    // (delay_to_frequency(), dm_per_unit_delay(), toplevel_tree_rank), which belong to the
    // instrument and are shared by every tree. decode_argmax() needs none of it.
    //
    // ----------------------------------------------------------------------------------
    //
    // decode_argmax(): converts an out_argmax token (plus its array indices) into the
    // winning trial parameters, i.e. the (subband, peak-finding profile, fine-grained dm,
    // fine-grained arrival time) responsible for the coarse-grained maximum in 'out_max'.
    //
    // Inputs:
    //
    //   argmax_token = uint32 token from this tree's out_argmax array
    //   0 <= idm_coarse < ndm_out     (dm index in out_max/out_argmax)
    //   0 <= itime_coarse < nt_out    (time index in out_max/out_argmax)
    //
    // Outputs are TOPLEVEL-relative: tree-freq channels of the rank-toplevel_tree_rank
    // gridding, and full-resolution time samples with t=0 at the start of the current
    // chunk (i.e. no per-tree time downsampling or early-trigger reindexing -- all
    // per-tree reindexing is done here, not by the caller):
    //
    //   0 <= fmin < fmax < pow2(toplevel_tree_rank)
    //       Tree-freq range (inclusive) spanned by the winning frequency subband.
    //       (Sharper per-tree bound: fmax < pow2(toplevel_tree_rank - early_trigger_level).)
    //
    //   tlo <= thi <= nt_in
    //       Trailing edges (EXCLUSIVE): tlo (resp. thi) is one past the last time sample
    //       of channel fmin (resp. fmax) which is summed into the winning out_max value,
    //       i.e. the exclusive upper endpoint of the summed range. Negative values are
    //       frequent (dedispersion delays usually exceed the chunk length), and refer to
    //       earlier chunks. For downsampled trees (primary_tree_index > 0), tlo/thi lie
    //       on downsampled-bin boundaries, i.e. always satisfy t == 0 (mod pow2(ipri)).
    //
    //   0 <= p < nprofiles
    //       Winning peak-finding profile index.
    //
    // Note: the sum over channel f spans an f-dependent half-open range
    // tmin(f) <= t < tmax(f), with tmax(f) nondecreasing in f; this function reports
    // tmax at the two edge channels (where the tree delays are exact, not tree-rounded).
    // The range has length 1 (tmax == tmin + 1) iff (p == 0 and primary_tree_index == 0).
    //
    // Throws an exception on out-of-range indices or a malformed token.

    void decode_argmax(uint argmax_token, long idm_coarse, long itime_coarse,
                       long &fmin, long &fmax, long &tlo, long &thi, long &p) const;


    // Convert the parameters (fmin, fmax, tlo, thi, p) returned by decode_argmax()
    // to "physical" params:
    //
    //   - freq_{lo,hi}_MHz: low/high radio frequency of "winning" subband
    //   - dm:               dispersion measure in pc/cm^3
    //   - timestamp_samp:   "winning" arrival time, see below
    //   - width_samp:       "winning" peak-finder width, in toplevel time samples.
    //
    // The 'timestamp_samp' is the estimated arrival time of the pulse center at the
    // lowest radio frequency (highest tree-freq), in toplevel full-resolution time
    // samples with t=0 at the START OF THE CURRENT CHUNK -- the same convention as
    // decode_argmax()'s tlo/thi, and NOT relative to fpga_seq=0. (The caller adds the
    // chunk's absolute FPGA start to convert to an absolute timestamp.)
    //
    // 'timestamp_samp' is NOT confined to [0, nt_in): an early-trigger tree extrapolates
    // to the band bottom, so the time can lie past the chunk end (in the future); and
    // the finite peak-finder kernel width (the pf_shift center-of-mass offset subtracted
    // in the implementation) can push an event detected near the chunk start to a
    // slightly negative value, i.e. slightly before the chunk start.

    void decode_argmax2(const DedispersionConfig &config,
                        long fmin, long fmax, long tlo, long thi, long p,
                        double &freq_lo_MHz, double &freq_hi_MHz, double &dm,
                        double &timestamp_samp, double &width_samp) const;


    // Returns 1-d array of shape (ndm_out,) (int64, on the host).
    //
    // A dedispersion output element (ichunk, ibeam, idm, it) of this tree is
    // "steady-state", i.e. unaffected by the zero-padding before the start of the
    // acquisition, iff
    //
    //     ichunk * nt_out + it >= compute_steady_state_it0(config)[idm].
    //
    // Earlier elements are computed from sums whose dedispersion + peak-finding
    // footprint extends past the start of the acquisition, so their out_max values
    // are artificially low (warmup artifacts, not real triggers).
    //
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

