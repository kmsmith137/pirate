#ifndef _PIRATE_PEAK_FINDING_KERNEL_HPP
#define _PIRATE_PEAK_FINDING_KERNEL_HPP

#include <vector>

#include <ksgpu/Dtype.hpp>
#include <ksgpu/Array.hpp>
#include "trackers.hpp"  // BandwidthTracker


namespace pirate {
#if 0
}  // editor auto-indent
#endif


// -------------------------------------------------------------------------------------------------
//
// All classes below are defined in src_lib/PeakFindingKernel.cu.


struct PeakFindingKernelParams
{
    ksgpu::Dtype dtype;
    
    long dm_downsampling_factor = 0;
    long time_downsampling_factor = 0;
    long max_kernel_width = 0;
    long beams_per_batch = 0;
    long total_beams = 0;
    long ndm_in = 0;
    long nt_in = 0;

    void validate() const;  // throws an exception if anything is wrong
};


// Base class for ReferencePeakFindingKernel, GpuPeakFindingKernel.
struct PeakFindingKernel
{
    const PeakFindingKernelParams params;

    long Dcore = 0;
    long nbatches = 0;   // = (total_beams / beams_per_batch)
    long ndm_out = 0;    // = (params.ndm_in / params.dm_downsampling_factor)
    long nt_out = 0;     // = (params.nt_in / params.time_downsampling_factor)
    long nprofiles = 0;  // = (3 * log2(max_kernel_width) + 1)

    PeakFindingKernel(const PeakFindingKernelParams &params, long Dcore);

    void _check_args(const ksgpu::Array<void> &out_max,
		     const ksgpu::Array<void> &out_ssq,
		     const ksgpu::Array<void> &in,
		     const ksgpu::Array<void> &wt,
		     ksgpu::Dtype expected_dtype,
		     long ibatch);
};


// -------------------------------------------------------------------------------------------------
//
// ReferencePeakFindingKernel


struct ReferencePeakFindingKernel : PeakFindingKernel
{
    // The 'Dcore' argument needs some discussion:
    //
    //   - In the GPU kernel, there is an internal parameter 'Dcore', which the GPU
    //     kernel selects based on a nontrivial tradeoff (larger Dcore leads to less
    //     computation but more register usage). The value of Dcore slightly affects
    //     the output of the kernel.
    //
    //   - In the reference kernel, the caller can specify Dcore as an extra
    //     argument. In a unit testing context, the caller can match a GPU kernel.
    //
    //   - Dcore must be a divisor of params.time_downsampling_factor. If Dcore
    //     is unspecified, then it defaults to (params.time_downsampling_factor).
    
    ReferencePeakFindingKernel(const PeakFindingKernelParams &params, long Dcore=0);

    // Inherits from PeakFindingKernel base class:
    //  params, Dcore, nbatches, ndm_out, nt_out, nprofiles

    // Note: params.dtype is ignored in reference kernel (all Arrays must be float)
    // All arrays must be fully contiguous (this could be changed if needed).
    
    void apply(ksgpu::Array<void> &out_max,     // shape (beams_per_batch, nprofiles, ndm_out, nt_out)
	       ksgpu::Array<void> &out_ssq,     // shape (beams_per_batch, nprofiles, ndm_out, nt_out)
	       const ksgpu::Array<void> &in,    // shape (beams_per_batch, ndm_in, nt_in)
	       const ksgpu::Array<void> &wt,    // shape (beams_per_batch, nprofiles, ndm_in)
	       long ibatch);

    // The ReferenceKernel allocates persistent_state in the constructor (not a separate
    // allocate() method). We currently use a simple but suboptimal approach: just save the
    // last 'pstate_nt' time samples from the previous chunk of data.

    long pstate_nt = 0;          // initialized in constructor
    ksgpu::Array<float> pstate;  // shape (total_beams, ndm_in, pstate_nt)
};


// -------------------------------------------------------------------------------------------------
//
// GpuPeakFindingKernel


struct GpuPeakFindingKernel : PeakFindingKernel
{
    GpuPeakFindingKernel(const PeakFindingKernelParams &params);
    
    // Inherits from PeakFindingKernel base class:
    //  params, Dcore, nbatches, ndm_out, nt_out, nprofiles
    
    bool is_allocated = false;
        
    // Note: allocate() initializes or zeroes all arrays (i.e. no array is left uninitialized)
    void allocate();

    // Bandwidth per call to GpuPeakFindingKernel::launch().
    // To get bandwidth per time chunk, multiply by 'nbatches'.
    BandwidthTracker bw_per_launch;

    // Warning: GPU kernel assumes all weights are positive, and behavior
    // is undefined if there are negative weights.
    
    void launch(
        ksgpu::Array<void> &out_max,   // shape (beams_per_batch, nprofiles, ndm_out, nt_out)
	ksgpu::Array<void> &out_ssq,   // shape (beams_per_batch, nprofiles, ndm_out, nt_out)
	const ksgpu::Array<void> &in,  // shape (beams_per_batch, ndm_in, nt_in)
	const ksgpu::Array<void> &wt,  // shape (beams_per_batch, nprofiles, ndm_in)
	long ibatch,                   // 0 <= ibatch < nbatches
	cudaStream_t stream            // NULL stream is allowed, but is not the default
    );

    // -------------------- Internals start here --------------------

    // Allocated in GpuPeakFindingKernel::allocate(), not the constructor.
    ksgpu::Array<void> persistent_state;

    struct RegistryKey
    {
	ksgpu::Dtype dtype;   // either float16 or float32
	int M = 0;            // same as PeakFindingKernelParams::dm_downsampling_factor
	int E = 0;            // same as PeakFindingKernelParams::max_kernel_width
	int Dout = 0;         // same as PeakFindingKernelParams::time_downsampling_factor
    };

    struct RegistryValue
    {
	// We define two low-level cuda kernels: a "full peak-finding kernel" which is externally
	// useful, and a "reduce-only" kernel which serves an internal debugging purpose.
	//
	// The full peak-finding kernel is called as:
	//
	//   void pf_full(out_max, out_ssq, pstate, in, wt, Mout, Tout);
	//
	// and the reduce-only kernel is called as:
	//
	//   void pf_reduce(out_max, out_ssq, in_max, in_ssq, wt, Mout, Tout);
	//
	// where:
	//
	//  - 'out_max' and 'out_ssq' have shape (B, P, Mout, Tout).
	//  - 'in_max' and 'in_ssq' have shape (B, P, Mout*M, Tout*(Dout/Dcore))
	//  - 'pstate' has shape (B, Mout, P32).
	//  - 'in' has shape (B, Mout*M, Tout*Dout).
	//  - 'wt' has shape (B, P, Mout*M).
	//  - 'B' is the number of beams, and other params (P, M, ...) are defined below
	//
	// Kernels are launched with {BM,B,1} blocks and {32*W,1} threads, where BM = ceil(Mout/W).
	// Shared memory is statically allocated.
	
	int Dcore = 0;   // internal downsampling factor (see discussion above)
	int W = 0;       // warps per threadblock
	int P = 0;       // number of peak-finding kernels (= 3*log2(E) + 1)
	int P32 = 0;     // ring buffer 32-bit registers per output (beam, mout)

	// The "full peak-finding" and "reduce-only" kernels have the same call signatures
	// (five pointers and two ints), but the meaning of the args is different, see above.
	using cuda_kernel_t = void (*) (void *, void *, void *, void *, void *, int, int);
	
	cuda_kernel_t full_kernel = nullptr;
	cuda_kernel_t reduce_only_kernel = nullptr;
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


// Defined in GpuPeakFindingKernel.cu
extern bool operator==(const GpuPeakFindingKernel::RegistryKey &k1, const GpuPeakFindingKernel::RegistryKey &k2);
extern std::ostream &operator<<(std::ostream &os, const GpuPeakFindingKernel::RegistryKey &k);


}  // namespace pirate

#endif // _PIRATE_PEAK_FINDING_KERNEL_HPP
