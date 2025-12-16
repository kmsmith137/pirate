#ifndef _PIRATE_COALESCED_DD_KERNEL_2_HPP
#define _PIRATE_COALESCED_DD_KERNEL_2_HPP

#include <vector>
#include <unordered_map>
#include <ksgpu/Dtype.hpp>
#include <ksgpu/Array.hpp>

#include "DedispersionKernel.hpp"
#include "PeakFindingKernel.hpp"
#include "FrequencySubbands.hpp"
#include "KernelRegistry.hpp"
#include "trackers.hpp"  // BandwidthTracker


namespace pirate {
#if 0
}  // editor auto-indent
#endif


struct CoalescedDdKernel2
{
    CoalescedDdKernel2(
        const DedispersionKernelParams &dd_params,   // dedispersion
        const PeakFindingKernelParams &pf_params     // peak-finding
    );

    void allocate();

    // The 'weights' array has logical shape (beams_per_batch, ndm_wt, nt_wt, P, F),
    // but is passed to the gpu kernel in a complicated, non-contiguous layout. To put
    // an array into the proper layout, call GpuPfWeightLayout::to_gpu().

    void launch(
        ksgpu::Array<void> &out_max,      // shape (beams_per_batch, ndm_out, nt_out)
        ksgpu::Array<uint> &out_argmax,   // shape (beams_per_batch, ndm_out, nt_out)
        const ksgpu::Array<void> &in,     // shape (mega_ringbuf->gpu_global_nseg * nt_per_segment * nspec,)
        const ksgpu::Array<void> &wt,     // see comment above
        long ibatch,                      // 0 <= ibatch < nbatches
        long it_chunk,                    // time-chunk index 0, 1, ...
        cudaStream_t stream               // NULL stream is allowed, but is not the default);
    );

    // Static member functions to query registry.
    static long registry_size() { return registry().size(); }
    static void show_registry() { registry().show(); }

    // Static member function: runs one randomized test iteration.
    static void test();

    // Static member functions: run timing for representative kernels.
    static void time_one(const std::vector<long> &subband_counts, const std::string &name);
    static void time();


    // ------------------------  Members  ------------------------

    DedispersionKernelParams dd_params;  // dd_rank, amb_rank, total_beams, beams_per_batch, ntime, mega_ringbuf
    PeakFindingKernelParams pf_params;   // beams_per_batch, total_beams, ndm_out, ndm_wt, nt_out, nt_in, nt_wt
    FrequencySubbands fs;                // pf_rank, F, M

    bool is_allocated = false;

    // Derived parameters chosen by the kernel.
    GpuPfWeightLayout pf_weight_layout;     // layout of peak-finding weights in GPU memory
    std::vector<long> expected_wt_shape;    // from pf_weight_layout.get_shape()
    std::vector<long> expected_wt_strides;  // from pf_weight_layout.get_strides()
    long Dcore = 0;                         // internal downsampling factor

    // Derived parameters, computed in constructor.
    ksgpu::Dtype dtype;        // = params.dtype
    long Dout = 0;             // = (nt_in/nt_out) = time downsampling factor of output array 
    long nbatches = 0;         // = (total_beams / beams_per_batch)
    long nprofiles = 0;        // = (3 * log2(max_kernel_width) + 1)

    // Bandwidth per call to GpuDedispersionKernel::launch().
    // To get bandwidth per time chunk, multiply by 'nbatches'.

    BandwidthTracker bw_per_launch;       // all gpu arrays including pstate
    BandwidthTracker bw_core_per_launch;  // only input/output arrays

    // -------------------- Internals start here --------------------

    // The 'persistent_state' and 'gpu_ringbuf_quadruples' arrays are
    // allocated in CoalescedDdKernel2::allocate(), not the constructor.

    // Shape (total_beams, pow2(params.amb_rank), ninner)
    // where ninner = cuda_kernel.pstate_32_per_small_tree * (32/nbits)
    ksgpu::Array<void> persistent_state;

    // FIXME should add run-time check that current cuda device is consistent.
    ksgpu::Array<uint> gpu_ringbuf_quadruples;   // shape (nsegments_per_beam, 4)
    long nsegments_per_beam = 0;

    struct RegistryKey
    {
        ksgpu::Dtype dtype;   // either float16 or float32
        long dd_rank = -1;
        long Tinner = 0;      // for weights
        long Dout = 0;
        long Wmax = 0;

        std::vector<long> subband_counts;  // length (pf_rank+1)
    };

    struct RegistryValue
    {
        // cuda_kernel(
        //     void *grb_base_, uint *grb_loc_, long grb_pos,     // dedisperser input (ring buffer)
        //     void *out_max_, uint *out_argmax, const void *wt_, // peak-finder output
        //     void *pstate_, int ntime,                          // shared between dedisperser and peak-finder
        //     ulong nt_cumul, bool is_downsampled_tree,          // dedisperser
        //     uint ndm_out_per_wt, uint nt_in_per_wt             // peak-finder
        // );
        //
        // Launch with {32,W,1} threads/block and {Namb,Nbeams,1} threadblocks.

        void (*cuda_kernel)(
            void *, uint *, long, 
            void *, uint *, const void *, 
            void *, int, 
            ulong, bool, 
            uint, uint
        ) = nullptr;

        // Layout of peak-finding weights in GPU memory, expected by the kernel.
        GpuPfWeightLayout pf_weight_layout;

        int shmem_nbytes = -1;
        int warps_per_threadblock = 0;
        int pstate32_per_small_tree = -1;  // see 'persistent_state' array dims above
        int nt_per_segment = 0;            // value of 'nt_per_segment' assumed by dd-kernel
        long Dcore = 0;                    // internal downsampling factor, chosen by pf-kernel
    };

    // Non-static members for interacting with the kernel registry.
    RegistryKey registry_key;
    RegistryValue registry_value;

    using Registry = KernelRegistry<RegistryKey, RegistryValue>;

    // Static member function to access registry.
    static Registry &registry();
};

// Defined in CoalescedDdKernel2.cu
extern bool operator==(const CoalescedDdKernel2::RegistryKey &k1, const CoalescedDdKernel2::RegistryKey &k2);
extern std::ostream &operator<<(std::ostream &os, const CoalescedDdKernel2::RegistryKey &k);
extern std::ostream &operator<<(std::ostream &os, const CoalescedDdKernel2::RegistryValue &v);


}  // namespace pirate

#endif // _PIRATE_COALESCED_DD_KERNEL_2_HPP
