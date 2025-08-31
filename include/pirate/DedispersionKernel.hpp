#ifndef _PIRATE_DEDISPERSION_KERNEL_HPP
#define _PIRATE_DEDISPERSION_KERNEL_HPP

#include <ksgpu/Array.hpp>
#include "trackers.hpp"  // BandwidthTracker

namespace pirate {
#if 0
}  // editor auto-indent
#endif

struct ReferenceTree;    // defined in ReferenceTree.hpp
struct ReferenceLagbuf;  // defined in ReferenceLagbuf.hpp


// Dedispersion kernels have two ranks:
//
//   dd_rank = "dedispersion" rank
//           = log2(number of "active" tree channels)
//
//   amb_rank = "ambient" rank
//            = log2(number of "spectator" tree channels)
//
// The dedispersion kernel operates on a buffer of logical shape
//   (beams_per_batch, pow2(amb_rank), pow2(dd_rank), ntime).
//
// The buffer is not assumed contiguous -- this detail is important, since the
// amb/dd axes may be swapped (relative to contiguous ordering). To explain this in
// more detail, consider two-stage tree dedispersion with total rank (rank1 + rank2):
//
//  - Let's "reshape" the dedispersion buffer to a 4-d array
//
//      (nbeams, pow2(rank2), pow2(rank1), ntime)
//
//    This 4-d shape will be preserved throughout the two-stage dedispersion, but
//    the meaning of the axes will be different at different stages.
//
//  - In the input array, the meaning of the axes is:
//
//        (beam, coarse freq, fine freq, time).
//
//  - When we apply the first dedispersion kernel, we map these axes to:
//
//        (beam, amb, dd, time)       (*)
//
//  - After the first dedispersion kernel, the meaning of the axes is:
//
//        (beam, coarse freq, bit-reversed coarse dm, time)
//
//  - When we apply the second dedispersion kernel, we map these axes to:
//
//        (beam, dd, amb, time)      (**)
//
//    Note the transpose in (**) relative to (*), which implies that
//    dedispersion kernels will sometimes need non-contiguous strides.
//
//  - In the output array (after the second dedispersion kernel), the
//    meaning of the axes is:
//
//        (beam, bit-reversed fine dm, bit-reversed coarse dm, time)
//
//   Note that this cleanly "reshapes" to a 3-d array:
//
//        (beam, bit-reversed total dm, time)
    

struct DedispersionKernelParams
{
    ksgpu::Dtype dtype;   // either float32 or float16
    long dd_rank = -1;    // satisfies 1 <= dd_rank <= 8
    long amb_rank = -1;   // satisfies 0 <= amb_rank <= 8
    long total_beams = 0;
    long beams_per_batch = 0;
    long ntime = 0;       // includes downsampling factor, if any
    
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

    // The value of 'nelts_per_segment' matters if:
    //   (apply_input_residual_lags || input_is_ringbuf || output_is_ringbuf).
    //
    // The GPU kernel assumes (nelts_per_segment == nelts_per_cache_line), but the reference
    // kernel allows (nelts_per_segment) to be a multiple of (nelts_per_cache_line), where:
    //   nelts_per_cache_line = (8 * constants::bytes_per_gpu_cache_line) / dtype.nbits.
    //
    // This is in order to enable a unit test where we check agreement between a float16
    // GPU kernel, and a float32 reference kernel derived from the same DedispersionPlan.
    // In this case, we want the reference kernel to have dtype float32, but use a value
    // of 'nelts_per_segment' which matched to the float16 GPU kernel.
    
    int nelts_per_segment = 0;

    // The 'ringbuf_locations' array has shape (nsegments_per_tree, 4), where:
    //   nsegments_per_tree = pow2(dd_rank + amb_rank) * xdiv(ntime,nelts_per_segment)
    //
    // The DedispersionKernelParams::ringbuf_locations array is always on the host (even for a
    // GPU kernel). The copy from host to GPU happens in GpuDedispersionKernel::allocate()).
    //
    // The 'ringbuf_locations' array only gets used if (input_is_ringbuf || output_is_ringbuf).
    
    // Only used if (input_is_ringbuf || output_is_ringbuf)
    ksgpu::Array<uint> ringbuf_locations;
    long ringbuf_nseg = 0;
    
    // Throws an exception if anything is wrong.
    void validate(bool gpu_kernel) const;
};


// Notes:
//
//  - The reference kernel allocates persistent state in its constructor
//    (i.e. no allocate() method is defined).
//
//  - The reference kernel uses float32, regardless of what dtype is specified.

struct ReferenceDedispersionKernel
{
    using Params = DedispersionKernelParams;
    Params params;  // reminder: contains 'ringbuf_locations' array

    ReferenceDedispersionKernel(const Params &params);
    
    // The 'in' and 'out' arrays are either "simple" buffers or ringbufs, depending on
    // values of Params::input_is_ringbuf and Params::output_is_ringbuf. Shapes are:
    //
    //   - simple buf has shape (params.beams_per_batch, pow2(amb_rank), pow2(dd_rank), ntime).
    //   - ringbuf has 1-d shape (params.ringbuf_nseg * params.nelts_per_segment,)

    void apply(ksgpu::Array<void> &in, ksgpu::Array<void> &out, long ibatch, long it_chunk);

    long nbatches = 0;     // same as (params.total_beams / params.beams_per_batch)
    std::vector<std::shared_ptr<ReferenceTree>> trees;        // length (nbatches)
    std::vector<std::shared_ptr<ReferenceLagbuf>> rlag_bufs;  // length (params.apply_input_residual_lags ? nbatches : 0)

    // Used internally
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
    
    Params params;   // reminder: contains 'ringbuf_locations' array on host (not GPU!)
    bool is_allocated = false;
    
    // Note: allocate() initializes or zeroes all arrays (i.e. no array is left uninitialized)
    void allocate();
    
    // launch(): asynchronously launch dedispersion kernel, and return without synchronizing stream.
    //
    // The 'in' and 'out' arrays are either "simple" buffers or ringbufs, depending on
    // values of Params::input_is_ringbuf and Params::output_is_ringbuf. Shapes are:
    //
    //   - simple buf has shape (params.beams_per_batch, pow2(amb_rank), pow2(dd_rank), ntime).
    //   - ringbuf has 1-d shape (params.ringbuf_nseg * params.nelts_per_segment,)
    
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

    // -------------------- Internals start here --------------------

    // The 'persistent_state' and 'gpu_ringbuf_locations' arrays are
    // allocated in GpuDedipsersionKernel::allocate(), not the constructor.
    
    // Shape (total_beams, pow2(params.amb_rank), ninner)
    // where ninner = cuda_kernel.pstate_32_per_small_tree * (32/nbits)
    ksgpu::Array<void> persistent_state;

    // FIXME should add run-time check that current cuda device is consistent.
    ksgpu::Array<uint> gpu_ringbuf_locations;

    struct RegistryKey
    {
	ksgpu::Dtype dtype;   // either float16 or float32
	int rank = -1;
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
    };

    // Low-level cuda kernel and associated metadata.
    RegistryValue registry_value;

    // Static member functions for querying registry.
    static RegistryValue query_registry(const RegistryKey &k);
    static RegistryKey get_random_registry_key();

    // Static member function for adding to the registry.
    // Called during library initialization, from source files with gpu kernels.
    static void register_kernel(const RegistryKey &key, const RegistryValue &val, bool debug);    
};


// Defined in GpuDedispersionKernel.cu
extern bool operator==(const GpuDedispersionKernel::RegistryKey &k1, const GpuDedispersionKernel::RegistryKey &k2);
extern std::ostream &operator<<(std::ostream &os, const GpuDedispersionKernel::RegistryKey &k);


}  // namespace pirate

#endif // _PIRATE_DEDISPERSION_KERNEL_HPP
