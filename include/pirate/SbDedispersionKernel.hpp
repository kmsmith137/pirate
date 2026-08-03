#ifndef _PIRATE_SB_DEDISPERSION_KERNEL_HPP
#define _PIRATE_SB_DEDISPERSION_KERNEL_HPP

#include <vector>
#include <ksgpu/Array.hpp>

#include "BumpAllocator.hpp"
#include "DedispersionKernel.hpp"
#include "FrequencySubbands.hpp"
#include "KernelRegistry.hpp"
#include "ResourceTracker.hpp"


namespace pirate {
#if 0
}  // editor auto-indent
#endif


// -------------------------------------------------------------------------------------------------
//
// GpuSbDedispersionKernel: stage-2 tree dedispersion which writes the subband array
// ('sb_out') directly to global memory, with no peak-finding.
//
// This is the GPU counterpart of ReferenceDedispersionKernel::apply(), restricted to its
// 'sb_out' output. Its intended use is variance calculations, which need the subband array
// itself rather than the peak-finder's max/argmax; the reference kernel computes the same
// thing on the CPU, but too slowly to use at CHIME/CHORD scale.
//
// Relation to CoalescedDdKernel2: that kernel exists precisely so that the shape
// (ndm_out, M, ntime) subband array is never materialized in GPU memory -- it fuses
// peak-finding into the second dedispersion stage. This kernel does materialize it, so it
// does strictly less arithmetic but moves a lot more memory: the output is larger than the
// input by a factor (M / 2^pf_rank), which is ~5.7 for the CHORD subband counts. Expect it
// to be output-bandwidth bound.
//
// The output contains the plain (non-subbanded) dedispersion output as a sub-array, so
// there is no separate 'dd_out' argument. FrequencySubbands always includes the full band
// (subband_counts[pf_rank] == 1), and that subband has zero time lag, so its 2^pf_rank
// multiplets -- the last ones, (M - 2^pf_rank) <= m < M -- together with the 'ndm_out'
// coarse DMs are exactly the 2^(amb_rank + dd_rank) DM outputs which a GpuDedispersionKernel
// would have written.
//
// Constraints (same as CoalescedDdKernel2, plus float32-only):
//
//   - dtype is float32 (the kernel is not implemented for float16)
//   - dd_rank >= 3 (the kernel needs the two-stage dedisperser, see below)
//   - nspec == 1
//   - apply_input_residual_lags == true
//   - input_is_ringbuf == true, output_is_ringbuf == false
//   - frequency_subbands.pf_rank == dd_rank - (dd_rank/2)
//
// That last constraint deserves a word, since it is not arbitrary. The kernel gets the
// per-subband time lag "for free": the shared memory ring buffer of the two-stage
// dedisperser applies lag d' * (2^rank1 - 1 - f), where rank1 = dd_rank - dd_rank/2 is the
// second-stage rank, f is a coarse frequency and d' is a coarse DM. That is *identically*
// the lag which ReferenceTree::final_lagbuf applies, namely (2^pf_rank - fhi(m)) * d',
// provided pf_rank == rank1. See cuda_generator/Dedisperser.emit_subband_extraction() and
// the "subbanded dedispersion" section of notes/tree_dedispersion.tex.


struct GpuSbDedispersionKernel
{
    // 'frequency_subbands' is implicitly constructible from a vector<long> of subband
    // counts, so callers may pass e.g. {0,5,7,3,1} directly.
    GpuSbDedispersionKernel(
        const DedispersionKernelParams &dd_params,
        const FrequencySubbands &frequency_subbands
    );

    // Note: allocate() initializes or zeroes all arrays (i.e. no array is left uninitialized).
    void allocate(BumpAllocator &allocator);

    // launch(): asynchronously launch the kernel, and return without synchronizing stream.
    //
    // Reminder: a "chunk" is a range of time indices, and a "batch" is a range of beam indices.

    void launch(
        ksgpu::Array<float> &sb_out,      // shape (beams_per_batch, Dpf, fs.M, ntime)
        const ksgpu::Array<float> &in,    // shape (mega_ringbuf->gpu_global_nseg * nt_per_segment,)
        long ichunk,                      // time-chunk index 0, 1, ...
        long ibatch,                      // 0 <= ibatch < nbatches
        cudaStream_t stream               // NULL stream is allowed, but is not the default
    );

    // Static member functions to query registry.
    static long registry_size() { return registry().size(); }
    static void show_registry() { registry().show(); }

    // Static member function: runs one randomized test iteration.
    // Called by 'python -m pirate_frb test --sbdd'.
    static void test_random();


    // ------------------------  Members  ------------------------

    DedispersionKernelParams dd_params;  // dd_rank, amb_rank, total_beams, beams_per_batch, ntime, mega_ringbuf
    FrequencySubbands fs;                // pf_rank, N, M

    long Dpf = 0;        // = pow2(amb_rank + dd_rank - fs.pf_rank), the 'sb_out' DM axis
    long nbatches = 0;   // = (total_beams / beams_per_batch)
    bool is_allocated = false;

    // All rates are "per call to launch()".
    ResourceTracker resource_tracker;

    // -------------------- Internals start here --------------------

    // The 'persistent_state' and 'gpu_ringbuf_quadruples' arrays are
    // allocated in GpuSbDedispersionKernel::allocate(), not the constructor.

    // Shape (total_beams, pow2(dd_params.amb_rank), registry_value.pstate32_per_small_tree)
    ksgpu::Array<float> persistent_state;

    // FIXME should add run-time check that current cuda device is consistent.
    ksgpu::Array<uint> gpu_ringbuf_quadruples;   // shape (nsegments_per_beam, 4)
    long nsegments_per_beam = 0;

    struct RegistryKey
    {
        // Note: no 'dtype' (the kernel is float32-only), and none of the peak-finding
        // parameters (Wmax, Dcore, Dout, Tinner) which appear in CoalescedDdKernel2's key.
        // If a float16 variant is ever added, a dtype must be added here AND to the
        // generated filename (see cuda_generator/SbDedisperser.py).

        long dd_rank = -1;
        std::vector<long> subband_counts;  // length (pf_rank+1)
    };

    struct RegistryValue
    {
        // cuda_kernel(
        //     void *grb_base_, uint *grb_quads_, long grb_frame0,  // input ring buffer
        //     float *sb_out_,                                      // (beams, ndm_out, M, ntime)
        //     void *pstate_, int ntime,
        //     ulong nt_cumul, bool is_downsampled_tree);
        //
        // The 'grb_base', 'grb_quads', 'grb_frame0' args parameterize the input ring
        // buffer. See MegaRingbuf.hpp for details.
        //
        // 'sb_out' is a fully contiguous shape (beams, ndm_out, M, ntime) array, where
        // ndm_out = blockDim.y * gridDim.x.
        //
        // Launch with {32,W,1} threads/block and {Namb,Nbeams,1} threadblocks.

        void (*cuda_kernel)(void *, uint *, long, float *, void *, int, ulong, bool) = nullptr;

        int shmem_nbytes = -1;
        int warps_per_threadblock = 0;
        int pstate32_per_small_tree = -1;  // see 'persistent_state' array dims above
        int nt_per_segment = 0;            // value of 'nt_per_segment' assumed by the kernel
    };

    using Registry = KernelRegistry<RegistryKey, RegistryValue>;

    // Non-static members for interacting with the kernel registry.
    RegistryKey registry_key;
    RegistryValue registry_value;

    // Static member function to access registry.
    static Registry &registry();
};


// Defined in SbDedispersionKernel.cu
extern bool operator==(const GpuSbDedispersionKernel::RegistryKey &k1, const GpuSbDedispersionKernel::RegistryKey &k2);
extern std::ostream &operator<<(std::ostream &os, const GpuSbDedispersionKernel::RegistryKey &k);
extern std::ostream &operator<<(std::ostream &os, const GpuSbDedispersionKernel::RegistryValue &v);


}  // namespace pirate

#endif // _PIRATE_SB_DEDISPERSION_KERNEL_HPP
