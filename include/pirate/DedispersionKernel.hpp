#ifndef _PIRATE_DEDISPERSION_KERNEL_HPP
#define _PIRATE_DEDISPERSION_KERNEL_HPP

#include <ksgpu/Array.hpp>

#include "KernelRegistry.hpp"
#include "trackers.hpp"  // BandwidthTracker

namespace pirate {
#if 0
}  // editor auto-indent
#endif

struct ReferenceTree;    // defined in ReferenceTree.hpp
struct ReferenceLagbuf;  // defined in ReferenceLagbuf.hpp
struct MegaRingbuf;      // defined in MegaRingbuf.hpp

// Dedispersion kernels have two ranks:
//
//   dd_rank = "dedispersion" rank
//           = log2(number of "active" tree channels)
//
//   amb_rank = "ambient" rank
//            = log2(number of "spectator" tree channels)
//
// The dedispersion kernel operates on a buffer of logical shape
//
//   (beams_per_batch, pow2(amb_rank), pow2(dd_rank), ntime, nspec).
//
// The buffer is not assumed contiguous -- this detail is important, since the
// amb/dd axes may be swapped (relative to contiguous ordering). To explain this in
// more detail, consider two-stage tree dedispersion with total rank (rank1 + rank2):
//
//  - Let's "reshape" the dedispersion buffer to a 4-d array
//
//      (nbeams, pow2(rank2), pow2(rank1), ntime * nspec)
//
//    This 4-d shape will be preserved throughout the two-stage dedispersion, but
//    the meaning of the axes will be different at different stages.
//
//  - In the input array, the meaning of the axes is:
//
//        (beam, coarse freq, fine freq, time+spec).
//
//  - When we apply the first dedispersion kernel, we map these axes to:
//
//        (beam, amb, dd, time+spec)       (*)
//
//  - After the first dedispersion kernel, the meaning of the axes is:
//
//        (beam, coarse freq, bit-reversed coarse dm, time+spec)
//
//  - When we apply the second dedispersion kernel, we map these axes to:
//
//        (beam, dd, amb, time+spec)      (**)
//
//    Note the transpose in (**) relative to (*), which implies that
//    dedispersion kernels will sometimes need non-contiguous strides.
//
//  - In the output array (after the second dedispersion kernel), the
//    meaning of the axes is:
//
//        (beam, bit-reversed fine dm, bit-reversed coarse dm, time+spec)
//
//   Note that this cleanly "reshapes" to a 3-d array:
//
//        (beam, bit-reversed total dm, time+spec)
    

struct DedispersionKernelParams
{
    ksgpu::Dtype dtype;   // either float32 or float16
    long dd_rank = -1;    // satisfies 1 <= dd_rank <= 8
    long amb_rank = -1;   // satisfies 0 <= amb_rank <= 8
    long total_beams = 0;
    long beams_per_batch = 0;
    long ntime = 0;       // includes downsampling factor, if any
    long nspec = 0;       // "inner" spectator index
    
    // Input/output buffer types.
    bool input_is_ringbuf = false;
    bool output_is_ringbuf = false;
    
    // Residual lags (see comment above).
    // The meaning of Params::apply_residual_lags needs some explanation!
    //
    // This is used in the second dedisperser stage, where each tree channel is labelled
    // by two indices:
    //
    //   - a bit-reversed DM 0 <= d < 2^amb_rank
    //   - a coarse frequency 0 <= f < 2^dd_rank
    //
    // Before dedispersing the data, the following residual lag is applied:
    //
    //   int lag = rb_lag(f, d, amb_rank, dd_rank, params.input_is_downsampled_tree);
    //   int residual_lag = lag % nelts_per_segment;
    
    bool apply_input_residual_lags = false;
    bool input_is_downsampled_tree = false;   // only matters if apply_input_residual_lags=true

    // The value of 'nt_per_segment' affects behavior of the kernel if one of the flags is set:
    //
    //   (apply_input_residual_lags || input_is_ringbuf || output_is_ringbuf).
    //
    // For the GPU kernel, the caller must set 'nt_per_segment', and the GpuDedispersionKernel
    // constructor will throw an exception if there is a mismatch. Currently, the GPU kernel uses:
    //
    //   nt_per_segment = 1024 / (nspec * dtype.nbits)
    //
    // The ReferenceDedispersionKernel accepts any value of 'nt_per_segment'. This is convenient
    // for testing, since the ReferenceDedispersionKernel can always use the same value as the
    // corresponding GpuDedispersionKernel.
    
    int nt_per_segment = 0;

    // XXX to be deleted soon
    // Notes on the DedispersionKernelParams::ringbuf_locations array:
    //
    //    - Only used if (input_is_ringbuf || output_is_ringbuf).
    //
    //    - Always on the host (even for a GPU kernel). The copy from host to GPU
    //      happens in GpuDedispersionKernel::allocate().
    //
    //    - Shape is (nsegments_per_tree, 4), where:
    //       nsegments_per_tree = pow2(dd_rank + amb_rank) * xdiv(ntime,nt_per_segment)
    //
    //    - See DedispersionPlan.hpp for more info on array contents.
    ksgpu::Array<uint> ringbuf_locations;
    long ringbuf_nseg = 0;

    // Ringbuf info (only used if (input_is_ringbuf || output_is_ringbuf)
    std::shared_ptr<MegaRingbuf> mega_ringbuf;  // used if (input_is_ringbuf || output_is_ringbuf)
    long producer_id = -1;                      // used if (output_is_ringbuf)
    long consumer_id = -1;                      // used if (input_is_ringbuf)
    
    // Throws an exception if anything is wrong.ples.
    void validate() const;

    // Intended for test/timing programs.
    void print(const char *prefix = "  ") const;
};


// -------------------------------------------------------------------------------------------------
//
// DedispersionKernelIobuf: helper class for ReferenceDedispersionKernel and GpuDedispersionKernel,
// to process and error-check the input/output arrays.
//
// Represents an input/output buffer for a dedispersion kernel, which could be either a
// "simple" buffer, or a ring buffer. Shapes are (where all variables beams_per_batch,
// amb_rank, ... are members of 'struct DedispersionKernelParams'):
//
//   Simple: either (beams_per_batch, pow2(amb_rank), pow2(dd_rank), ntime, nspec)
//                 or (beams_per_batch, pow2(amb_rank), pow2(dd_rank), ntime)  if nspec==1
//
//   Ring: 1-d array of length (ringbuf_nseg * nt_per_segment * nspec).


struct DedispersionKernelIobuf
{
    DedispersionKernelIobuf(const DedispersionKernelParams &params,
                            const ksgpu::Array<void> &arr,
                            bool is_ringbuf_, bool on_gpu_);

    void *buf = nullptr;
    bool is_ringbuf;
    bool on_gpu;

    // Convenient for GPU kernel. Only initialized if (is_ringbuf == false).
    long beam_stride32 = 0;
    int amb_stride32 = 0;
    int act_stride32 = 0;
}; 


// -------------------------------------------------------------------------------------------------
//
// ReferenceDedispersionKernel
//
//  - The reference kernel allocates persistent state in its constructor
//    (i.e. no allocate() method is defined).
//
//  - The reference kernel uses float32, regardless of what dtype is specified.


struct ReferenceDedispersionKernel
{
    using Params = DedispersionKernelParams;
    Params params;  // reminder: contains shared_ptr<MegaRingbuf>

    ReferenceDedispersionKernel(const Params &params);
    
    // The 'in' and 'out' arrays are either "simple" buffers or ringbufs, depending on values
    // of Params::input_is_ringbuf and Params::output_is_ringbuf. Shapes are (where variables
    // beams_per_batch, amb_rank, ... are members of Params):
    //
    //   Simple: either (beams_per_batch, pow2(amb_rank), pow2(dd_rank), ntime, nspec)
    //                 or (beams_per_batch, pow2(amb_rank), pow2(dd_rank), ntime)  if nspec==1
    //
    //   Ring: 1-d array of length (mega_ringbuf->gpu_giant_nseg * nt_per_segment * nspec).
    //   !!! Note that we use the "GPU" part of the mega_ringbuf, not the "host" part!!!

    void apply(ksgpu::Array<void> &in, ksgpu::Array<void> &out, long ibatch, long it_chunk);

    long nbatches = 0;     // same as (params.total_beams / params.beams_per_batch)
    std::vector<std::shared_ptr<ReferenceTree>> trees;        // length (nbatches)
    std::vector<std::shared_ptr<ReferenceLagbuf>> rlag_bufs;  // length (params.apply_input_residual_lags ? nbatches : 0)

    // Used internally
    void _init_rlags();
    void _copy_to_ringbuf(const ksgpu::Array<float> &in, ksgpu::Array<float> &out, long rb_pos);
    void _copy_from_ringbuf(const ksgpu::Array<float> &in, ksgpu::Array<float> &out, long rb_pos);
};


// -------------------------------------------------------------------------------------------------
//
// GpuDedispersionKernel


// The GpuDedispersionKernel uses externally-allocated buffers for its inputs/outputs,
// but internally allocates and manages its persistent state.

class GpuDedispersionKernel
{
    using Params = DedispersionKernelParams;
    
public:
    GpuDedispersionKernel(const Params &params);
    
    Params params;   // reminder: contains shared_ptr<MegaRingbuf>
    bool is_allocated = false;
    
    // Note: allocate() initializes or zeroes all arrays (i.e. no array is left uninitialized)
    void allocate();
    
    // launch(): asynchronously launch dedispersion kernel, and return without synchronizing stream.
    //
    // The 'in' and 'out' arrays are either "simple" buffers or ringbufs, depending on values
    // of Params::input_is_ringbuf and Params::output_is_ringbuf. Shapes are (where variables
    // beams_per_batch, amb_rank, ... are members of Params):
    //
    //   Simple: either (beams_per_batch, pow2(amb_rank), pow2(dd_rank), ntime, nspec)
    //                 or (beams_per_batch, pow2(amb_rank), pow2(dd_rank), ntime)  if nspec==1
    //
    //   Ring: 1-d array of length (mega_ringbuf->gpu_giant_nseg * nt_per_segment * nspec).

    void launch(
        ksgpu::Array<void> &in,
        ksgpu::Array<void> &out,
        long ibatch,
        long it_chunk,
        cudaStream_t stream  // NULL stream is allowed, but is not the default
    );

    long nbatches = 0;   // = (total_beams / beams_per_batch)
    
    // Bandwidth per call to GpuDedispersionKernel::launch().
    // To get bandwidth per time chunk, multiply by 'nbatches'.
    BandwidthTracker bw_per_launch;

    // Static member functions to query registry.
    static long registry_size() { return registry().size(); }
    static void show_registry() { registry().show(); }

    // Static member function: runs one randomized test iteration.
    // Called by 'python -m pirate_frb test --gddk'.
    static void test();

    // Static member function: run timing for representative kernels.
    // Called by 'python -m pirate_frb time --gddk'.
    static void time();
    static void _time(const DedispersionKernelParams &params, long nchunks=24);

    // -------------------- Internals start here --------------------

    // The 'persistent_state' and 'gpu_ringbuf_quadruples' arrays are
    // allocated in GpuDedipsersionKernel::allocate(), not the constructor.
    
    // Shape (total_beams, pow2(params.amb_rank), ninner)
    // where ninner = cuda_kernel.pstate_32_per_small_tree * (32/nbits)
    ksgpu::Array<void> persistent_state;

    // FIXME should add run-time check that current cuda device is consistent.
    ksgpu::Array<uint> gpu_ringbuf_quadruples;

    struct RegistryKey
    {
        ksgpu::Dtype dtype;   // either float16 or float32
        int rank = -1;
        int nspec = 0;
        
        bool input_is_ringbuf = false;
        bool output_is_ringbuf = false;
        bool apply_input_residual_lags = false;
    };

    struct RegistryValue
    {
        // The low-level cuda kernel is called as:
        //
        // void cuda_kernel_no_rb(
        //     void *inbuf, long beam_istride32, int amb_istride32, int act_istride32,
        //     void *outbuf, long beam_ostride32, int amb_ostride32, int act_ostride32,
        //     void *pstate, int ntime, ulong nt_cumul, bool input_is_downsampled_tree);
        //
        // void cuda_kernel_in_rb(
        //     void *rb_base, uint *rb_loc, long rb_pos,
        //     void *outbuf, long beam_ostride32, int amb_ostride32, int act_ostride32,
        //     void *pstate, int ntime, ulong nt_cumul, bool input_is_downsampled_tree);
        //
        // void cuda_kernel_out_rb(
        //     void *inbuf, long beam_istride32, int amb_istride32, int act_istride32,
        //     void *rb_base, uint *rb_loc, long rb_pos,
        //     void *pstate, int ntime, ulong nt_cumul, bool input_is_downsampled_tree);
        //
        // where:
        //
        //   - 'inbuf' is an array of shape (nbeams, namb, 2**rank, ntime)
        //      with strides (beam_istride32, amb_istride32, act_istride32, 1).
        //
        //   - 'outbuf' is an array of shape (nbeams, namb, 2**rank, ntime)
        //      with strides (beam_ostride32, amb_ostride32, act_ostride32, 1).
        //
        //   - Note that strides are in "32-bit" units (e.g. __half2 not __half).
        //
        //   - 'pstate' is an array of shape (nbeams, namb, pstate32_per_small_tree),
        //     with 32-bit dtype (e.g. __half2 not __half).
        //
        //   - TODO explain 'rb_base', 'rb_loc', 'rb_pos' and other args here.
        //
        //   - 'input_is_downsampled_tree' has the same meaning as
        //     DedispersionKernelParams::input_is_downsampled_tree.
        //
        //   -  The kernel is launched with {32, warps_per_threadblock} warps
        //      and {namb, nbeams} blocks.
        
        void (*cuda_kernel_no_rb)(void *, long, int, int, void *, long, int, int, void *, int, ulong, bool) = nullptr;
        void (*cuda_kernel_in_rb)(void *, uint *, long, void *, long, int, int, void *, int, ulong, bool) = nullptr;
        void (*cuda_kernel_out_rb)(void *, long, int, int, void *, uint *, long, void *, int, ulong, bool) = nullptr;
        
        int shmem_nbytes = 0;
        int warps_per_threadblock = 0;
        int pstate32_per_small_tree = 0;
        int nt_per_segment = 0;
    };

    using Registry = KernelRegistry<RegistryKey, RegistryValue>;
    
    // Low-level cuda kernel and associated metadata (non-static).
    RegistryValue registry_value;

    // Static member function to access registry.
    static Registry &registry();
};


// Defined in GpuDedispersionKernel.cu
extern bool operator==(const GpuDedispersionKernel::RegistryKey &k1, const GpuDedispersionKernel::RegistryKey &k2);
extern std::ostream &operator<<(std::ostream &os, const GpuDedispersionKernel::RegistryKey &k);
extern std::ostream &operator<<(std::ostream &os, const GpuDedispersionKernel::RegistryValue &v);


}  // namespace pirate

#endif // _PIRATE_DEDISPERSION_KERNEL_HPP
