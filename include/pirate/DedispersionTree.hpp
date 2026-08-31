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

    // Total tree rank. (For its relation to the config's toplevel rank, see
    // toplevel_tree_rank() just below.)
    long total_rank() const { return amb_rank + dd_rank; }

    // The toplevel_tree_rank of the DedispersionConfig this tree came from, recovered from
    // the tree's own members. Named to match DedispersionConfig::toplevel_tree_rank.
    //
    // Note '(primary_tree_index > 0)' rather than a bare 'primary_tree_index': the member is
    // -1 in a default-constructed tree, which the bare form would silently read as 1.
    long toplevel_tree_rank() const {
        return total_rank() + ((primary_tree_index > 0) ? 1 : 0) + early_trigger_level;
    }

    // Subbands searched in this tree.
    // Can differ from DedispersionConfig::frequency_subbands, due to early triggers and downsampling.
    FrequencySubbands frequency_subbands;

    // Toplevel-relative tree-freq range spanned by subband 'n', i.e. channels of the
    // rank-toplevel_tree_rank() gridding. HALF-OPEN, matching FrequencySubbands::n_to_f{lo,hi},
    // of which these are just a rescaling:
    //
    //     0 <= flo < fhi <= pow2(toplevel_tree_rank())
    //
    // with the sharper per-tree bound fhi <= pow2(toplevel_tree_rank() - early_trigger_level).
    //
    // NOTE decode_argmax() reports an INCLUSIVE upper channel, i.e. its 'fmax' output is
    // n_to_toplevel_fhi(n) - 1.
    long n_to_toplevel_flo(long n) const;
    long n_to_toplevel_fhi(long n) const;

    // Contains members: num_early_triggers, max_width, time_downsampling, wt_{dm,time}_downsampling.
    // Note that time_downsampling can be 0 in the config, but is filled with a nonzero value here.
    DedispersionConfig::PrimaryTree pf;

    // The GPU kernel's second-stage rank: the constructor pins dm_downsampling to
    // pow2(dd_rank1), and it is what xdm_rank() is measured against.
    long dd_rank1() const { return (dd_rank + 1) / 2; }

    // DM downsampling factor of the coarse-grained array, relative to this tree. NOT a
    // config field: it is fixed by the GPU kernel's warp geometry (one coarse DM per warp
    // of the second dedispersion stage), and dd_rank1 varies WITHIN a primary-tree family,
    // so no single per-primary-tree value could be right for all of its trees.
    //
    // STORED rather than derived, even though it equals pow2(dd_rank1()). That is the
    // opposite convention from xdm_rank() just below, and the choice is deliberate:
    // from_yaml() transcribes members verbatim with no re-derivation, which is the property
    // the round-trip test in pirate_frb/tests/test_decode_argmax.py is built on. Do not
    // "fix" the inconsistency by making this an accessor without moving that test too.
    long dm_downsampling = 0;

    // K = dd_rank1() - frequency_subbands.pf_rank: the number of "extra DM" bits that this
    // tree's cdd2 kernel folds into the argmax token's m-field (see CoalescedDdKernel2.hpp).
    // Zero unless the tree has an early trigger.
    //
    // Derived from dm_downsampling (which the tree constructor pins to pow2(dd_rank1()))
    // rather than stored, so it is not one more field for check_consistency() and the yaml
    // round-trip to keep honest.
    long xdm_rank() const;

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

    // Throws if this tree disagrees with what 'config' implies.
    //
    // Intended for trees that were DESERIALIZED (from_yaml) rather than constructed, where
    // the tree and the config travelled separately and could describe different instruments:
    // FrbGrouper's handshake receives a config yaml and a plan yaml from the producer, and
    // a variance-map file stores a config alongside per-tree records written at a different
    // time. In both cases the tree is what SOME build derived from the config, so a
    // disagreement means the deriving build and this one differ -- which no protocol-version
    // check catches, since it is a change within a version.
    //
    // Compares every member the decode paths read: the ranks, nt_ds, nprofiles, ndm_out /
    // ndm_wt / nt_out / nt_wt, the peak-finding downsampling factors, and the tree's
    // (restricted) subband_counts.
    //
    // Two deliberate exclusions:
    //
    //   - 'Dcore' is NOT compared, because it is the producer's cdd2-registry value,
    //     transcribed verbatim, and is exactly what lets decoding stay correct when the two
    //     processes run different builds. A local rebuild would carry the placeholder
    //     instead. Its standalone invariant IS checked (power of two dividing nt_ds/nt_out).
    //   - dm_min / dm_max / trigger_frequency are display values, and round-trip lossily
    //     through yaml, so an exact comparison would fail and a fuzzy one would prove
    //     nothing.
    //
    // Note the FrequencySubbands tables (m_to_n, the n_to_* ranges, M, N) are not compared
    // either: both sides rebuild them from subband_counts with the same code, so comparing
    // them would test determinism rather than agreement. Comparing subband_counts is the
    // meaningful check, since one side came off the wire and the other from the restriction
    // rule.
    void check_consistency(const DedispersionConfig &config) const;


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
    //   argmax_token = uint32 token from this tree's out_argmax array. Format is
    //       (t) | (p << 8) | (mu << 16) | (m << (16+K)) with K = xdm_rank(); see
    //       CoalescedDdKernel2.hpp. Note the m-field is m_ext = (m << K) | mu, not m,
    //       when K > 0 -- so do not decode it by hand, call this.
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

    // These methods are intended for use in VarianceMap, but may be useful elsewhere.
    //
    // n_index_mapping(): length child.frequency_subbands.N. Entry n_c is the parent subband
    //   searching the same toplevel band. Throws if the child's subbands are not a subset
    //   of the parents's subbands.
    //
    // m_index_mapping(): length child.frequency_subbands.M. Entry m_c is the parent
    //   multiplet with the same band and the same fine-DM index within it. Additionally
    //   throws unless matched bands have the same subband level.

    static std::vector<long> n_index_mapping(const DedispersionTree &parent,
                                             const DedispersionTree &child);

    static std::vector<long> m_index_mapping(const DedispersionTree &parent,
                                             const DedispersionTree &child);

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

