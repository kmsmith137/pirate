#ifndef _PIRATE_DEDISPERSION_PLAN_HPP
#define _PIRATE_DEDISPERSION_PLAN_HPP

#include "DedispersionConfig.hpp"

namespace YAML { class Emitter; }  // #include <yaml-cpp/yaml.h>
#include "DedispersionTree.hpp"           // struct DedispersionTree
#include "DedispersionBuffer.hpp"        // struct DedispersionBufferParams
#include "DedispersionKernel.hpp"        // struct DedispersionKernelParams
#include "LaggedDownsamplingKernel.hpp"  // struct LaggedDownsamplingKernelParams
#include "RingbufCopyKernel.hpp"         // struct RingbufCopyKernelParams
#include "TreeGriddingKernel.hpp"        // struct TreeGriddingKernelParams
#include "PeakFindingKernel.hpp"         // struct PeakFindingKernelParams
#include "MegaRingbuf.hpp"               // struct MegaRingbuf

#include <vector>
#include <memory>  // shared_ptr
#include <ksgpu/Array.hpp>


namespace pirate {
#if 0
}  // editor auto-indent
#endif


struct YamlFile;


// Dedispersion is a two-stage process. In the first stage, there is one tree for each
// primary tree (see DedispersionConfig.hpp). In the second stage, each primary tree is
// expanded into (num_early_triggers+1) trees: the main (full-band) tree, plus one "early"
// tree for each early_trigger_level = 1..num_early_triggers. Thus, there is a many-to-one mapping
// from stage2 trees to stage1 trees.

struct DedispersionPlan
{
    // Selects how much of the plan gets initialized.
    struct Params
    {
        // If false, then DedispersionPlan::mega_ringbuf is an empty pointer.
        bool mega_ringbuf = true;

        // If false, then all gpu kernel params (tree_gridding_kernel_params
        // ... h2h_copy_kernel_params) will be uninitialized. If mega_ringbuf=false
        // and gpu_kernels=true, then an exception will be raised.
        bool gpu_kernels = true;

        // Both false: config-derived scalars, stage1 ranks and 'trees' only. Needs no CUDA
        // device -- this is the plan that GPU-less consumers (pirate_frb.varmap,
        // FrbGrouper) build in order to get at the trees.
        static Params minimal();
    };

    DedispersionPlan(const DedispersionConfig &config, const Params &params);
    explicit DedispersionPlan(const DedispersionConfig &config);  // default Params

    const DedispersionConfig config;
    const Params params;

    // Some key members of DedispersionConfig, copied into DedispersionPlan for convenience.
    ksgpu::Dtype dtype;                  // same as config.dtype
    long nfreq = 0;                      // same as config.get_total_nfreq()
    long nt_in = 0;                      // same as config.time_samples_per_chunk
    long num_primary_trees = 0;          // same as config.num_primary_trees()
    long beams_per_gpu = 0;              // same as config.beams_per_gpu
    long beams_per_batch = 0;            // same as config.beams_per_batch
    long num_active_batches = 0;         // same as config.num_active_batches
    long nbits = 0;                      // same as config.dtype.nbits

    // Stage1 trees. These trees are "internal" to dedispersion, and can probably be ignored "from outside".
    // Total tree rank (dd_rank + amb_rank) is equal to (toplevel_tree_rank - (primary_tree_index ? 1 : 0)).
    // Both vectors have length (num_primary_trees).
    std::vector<long> stage1_dd_rank;    // "Active" dedispersion rank of each stage1 tree.
    std::vector<long> stage1_amb_rank;   // "Ambient" rank of each stage1 tree (= number of coarse freq channels)

    // Stage2 trees. These trees contain the output of the dedispersion, and are useful "from outside".
    // There is a lot of per-tree data, so I defined a helper class 'DedispersionTree'.
    // The number of trees is (config.num_primary_trees() + total number of early triggers).
    //
    // Trees are ordered by primary tree, then by DECREASING early-trigger level (earliest
    // trigger first, then the main early_trigger_level=0 tree).
    long ntrees = 0;
    std::vector<DedispersionTree> trees;  // length ntrees

    // (primary_tree_index, early_trigger_level) -> itree, i.e. the inverse of reading
    // trees[itree].{primary_tree_index, early_trigger_level}. Throws if either argument is
    // out of range. Available on a Params::minimal() plan, i.e. with no GPU.
    long dedispersion_tree_index(long primary_tree_index, long early_trigger_level) const;

    // 'verbose' controls explanatory comments; 'zones' independently controls
    // whether the mega_ringbuf per-clag host/gpu zone breakdown is emitted.
    //
    // An incomplete plan (Params::mega_ringbuf=false) serializes fine: its yaml is the same
    // document with the 'mega_ringbuf' key absent. This is what lets a GPU-less process
    // write a plan yaml (see pirate_frb.varmap's variance-map files).
    void to_yaml(YAML::Emitter &emitter, bool verbose = false, bool zones = false) const;
    std::string to_yaml_string(bool verbose = false, bool zones = false) const;

    // The plan a PRODUCER serialized (DedispersionPlan::to_yaml_string(), as sent in the
    // grouper handshake), rebuilt for a consumer that may be running a different pirate_frb
    // build.
    //
    // Returns a Params::minimal() plan built from 'config', cross-checked field by field
    // against the yaml; a disagreement throws, naming the field and both values. Nothing is
    // adopted from the yaml: a plan is a pure function of its config.
    //
    // Note that decoding the producer's out_argmax tokens needs one thing this plan does
    // NOT carry, because it is a property of the producer's compiled kernels rather than of
    // the plan: their per-tree Dcore. See decode_argmax() below, and FrbGrouper::dcores.
    static std::shared_ptr<DedispersionPlan> from_yaml(const DedispersionConfig &config,
                                                       const YamlFile &plan_yaml);

    static std::shared_ptr<DedispersionPlan> from_yaml_string(const DedispersionConfig &config,
                                                              const std::string &plan_yaml);


    // -------------------------------------------------------------------------------------------------
    //
    // Interpreting the dedisperser's output. All five methods below take an 'itree' index
    // into 'trees' (throwing if it is out of range), and read nothing but 'config' and
    // 'trees' -- so they are available on a Params::minimal() plan, i.e. with no GPU.


    // decode_argmax(): converts an out_argmax token (plus its array indices) into the
    // winning trial parameters, i.e. the (subband, peak-finding profile, fine-grained dm,
    // fine-grained arrival time) responsible for the coarse-grained maximum in 'out_max'.
    //
    // Inputs (see notes/dedispersion.tex for details of the out_argmax token format):
    //
    //   argmax_token = uint32 token from trees[itree]'s out_argmax array.
    //   Dcore = the PRODUCING kernel's internal time-downsampling factor (see below)
    //   0 <= itree < ntrees
    //   0 <= idm_coarse < trees[itree].ndm_out     (dm index in out_max/out_argmax)
    //   0 <= itime_coarse < trees[itree].nt_out    (time index in out_max/out_argmax)
    //
    // 'Dcore' sets the time granularity of the token's fine-time field, and is a property of
    // the peak-finding kernel that WROTE the token rather than of this plan. Take it from:
    //    - GpuDedisperser::Dcores[itree] for GPU tokens,
    //    - ReferenceDedisperserBase::Dcores[itree] for reference tokens
    //    - FrbGrouper::dcores[itree] on the consumer side of a grouper handshake
    //
    // Outputs are TOPLEVEL-relative: tree-freq channels of the rank-toplevel_tree_rank
    // gridding, and full-resolution time samples with t=0 at the start of the current
    // chunk (i.e. no per-tree time downsampling or early-trigger reindexing -- all
    // per-tree reindexing is done here, not by the caller):
    //
    //   0 <= fmin < fmax < pow2(config.toplevel_tree_rank)
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
    //   0 <= p < trees[itree].nprofiles
    //       Winning peak-finding profile index.
    //
    // Note: the sum over channel f spans an f-dependent half-open range
    // tmin(f) <= t < tmax(f), with tmax(f) nondecreasing in f; this function reports
    // tmax at the two edge channels (where the tree delays are exact, not tree-rounded).
    // The range has length 1 (tmax == tmin + 1) iff (p == 0 and primary_tree_index == 0).
    //
    // Throws an exception on out-of-range indices or a malformed token.

    void decode_argmax(uint argmax_token, long itree, long Dcore,
                       long idm_coarse, long itime_coarse,
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

    void decode_argmax2(long itree, long fmin, long fmax, long tlo, long thi, long p,
                        double &freq_lo_MHz, double &freq_hi_MHz, double &dm,
                        double &timestamp_samp, double &width_samp) const;


    // Returns 1-d array of shape (trees[itree].ndm_out,) (int64, on the host).
    //
    // A dedispersion output element (ichunk, ibeam, idm, it) of tree 'itree' is
    // "steady-state", i.e. unaffected by the zero-padding before the start of the
    // acquisition, iff
    //
    //     ichunk * trees[itree].nt_out + it >= compute_steady_state_it0(itree)[idm].
    //
    // Earlier elements are computed from sums whose dedispersion + peak-finding
    // footprint extends past the start of the acquisition, so their out_max values
    // are artificially low (warmup artifacts, not real triggers).
    //
    
    // Allocated in ordinary (unregistered) host memory, so this needs no CUDA device.
    ksgpu::Array<long> compute_steady_state_it0(long itree) const;


    // Relate the frequency subbands of two trees of this plan. Intended for use in
    // pirate_frb.varmap, but may be useful elsewhere.
    //
    // n_index_mapping(): length trees[ichild].frequency_subbands.N. Entry n_c is the parent
    //   subband searching the same toplevel band. Throws if the child's subbands are not a
    //   subset of the parent's subbands.
    //
    // m_index_mapping(): length trees[ichild].frequency_subbands.M. Entry m_c is the parent
    //   multiplet with the same band and the same fine-DM index within it. Additionally
    //   throws unless matched bands have the same subband level.

    std::vector<long> n_index_mapping(long iparent, long ichild) const;
    std::vector<long> m_index_mapping(long iparent, long ichild) const;


    // -------------------------------------------------------------------------------------------------
    //
    // Low-level data needed for compute kernels.


    int nelts_per_segment = 0;   // currently always constants::bytes_per_gpu_cache_line / (sizeof config dtype)
    int nbytes_per_segment = 0;  // currently always constants::bytes_per_gpu_cache_line
    
    // MegaRingbuf: this data structure is the "nerve center" of the real-time FRB search.
    // I have written a short novel explaining how it works, in MegaRingbuf.hpp.
    std::shared_ptr<MegaRingbuf> mega_ringbuf;

    TreeGriddingKernelParams tree_gridding_kernel_params;
    LaggedDownsamplingKernelParams lds_params;

    DedispersionBufferParams stage1_dd_buf_params;  // (number of buffers) = num_primary_trees
    DedispersionBufferParams stage2_dd_buf_params;  // (number of buffers) = ntrees

    std::vector<DedispersionKernelParams> stage1_dd_kernel_params;  // length num_primary_trees
    std::vector<DedispersionKernelParams> stage2_dd_kernel_params;  // length ntrees

    std::vector<PeakFindingKernelParams> stage2_pf_params;          // length ntrees

    // Only needed if early triggers are used.
    RingbufCopyKernelParams g2g_copy_kernel_params;
    RingbufCopyKernelParams h2h_copy_kernel_params;
};


}  // namespace pirate

#endif // _PIRATE_DEDISPERSION_PLAN_HPP
