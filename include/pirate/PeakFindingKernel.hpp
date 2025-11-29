#ifndef _PIRATE_PEAK_FINDING_KERNEL_HPP
#define _PIRATE_PEAK_FINDING_KERNEL_HPP

#include <vector>
#include <unordered_map>
#include <ksgpu/Dtype.hpp>
#include <ksgpu/Array.hpp>

#include "KernelRegistry.hpp"
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

    using Registry = KernelRegistry<RegistryKey, RegistryValue>;
    
    // Low-level cuda kernel and associated metadata (non-static).
    RegistryValue registry_value;

    // Static member function to access registry.
    static Registry &registry();
};


// Defined in GpuPeakFindingKernel.cu
extern bool operator==(const GpuPeakFindingKernel::RegistryKey &k1, const GpuPeakFindingKernel::RegistryKey &k2);
extern std::ostream &operator<<(std::ostream &os, const GpuPeakFindingKernel::RegistryKey &k);
extern std::ostream &operator<<(std::ostream &os, const GpuPeakFindingKernel::RegistryValue &v);


// -------------------------------------------------------------------------------------------------


// FIXME relocate this to a different .hpp?
struct FrequencySubbands
{
    FrequencySubbands(const std::vector<long> &subband_counts);

    // Length-(rank+1) vector, containing number of frequency subbands at each level.
    // This vector is used as an "identifier" for frequency subbands in low-level code.
    std::vector<long> subband_counts;   

    long pf_rank = -1;  // = subband_counts.size() - 1
    long F = 0;  // number of distinct frequency subbands
    long M = 0;  // number of "multiplets", i.e. (frequency_subband, fine_grained_dm) pairs

    std::vector<long> m_to_f;     // mapping (multiplet) -> (frequency_subband, fine_grained_dm)
    std::vector<long> m_to_d;     // mapping (multiplet) -> (frequency_subband, fine_grained_dm)
    std::vector<long> f_to_ilo;   // mapping (frequency_subband) -> (index pair 0 <= ilo < ihi <= 2**rank)
    std::vector<long> f_to_ihi;   // mapping (frequency_subband) -> (index pair 0 <= ilo < ihi <= 2**rank)

    // These members are used in the peak-finding kernel, whose 'out_argmax' array consists
    // of "tokens" of the form (t) | (p << 8) | (d << 14) | (f0 << 20) | (f1 << 26).

    std::vector<uint> m_to_token;                      // (d,f0,f1) part of token only
    std::unordered_map<uint,long> token_to_m;          // (token & token_m_mask) -> m

    static constexpr uint token_m_mask = 0xffffc000u;  // selects (d,f0,f1) part of token
};


// GpuPfWeightLayout: describes the layout of peak-finding weights on the GPU.
//
// Peak-finding weights are logically a 5-d array with shape:
//
//   (nbeams, Dbar, Tbar, P, F)        (*)
//
// where:
//
//  - Dbar = number of coarse DMs
//  - Tbar = number of coarse time samples
//  - P = number of peak-finding profiles
//  - F = number of frequency subbands F
//
// where the DM/time axis lengths (Dbar, Tbar) are related to the corresponding
// "tree" quantities by downsampling factors:
//
//   Dbar = Dtree / WDd   (must divide evenly)
//   Tbar = Ttree / WDt   (must divide evenly)
//
// The on-GPU memory layout is more complicated than (*)! Details of this layout
// are important in the autogenerated cuda kernels, but can be mostly opaque to
// the non-autogenerated C++ code. For full documentation of the layout, see
// 'class cuda_generator.PfWeightLayout' in the python code.
//
// When an (autogenerated) GPU kernel is retrieved from the registry, it comes
// with a GpuPfWeightLayout instance describing its expected memory layout. This
// instance has member functions to_gpu()) which are intended to hide details
// of the layout.

struct GpuPfWeightLayout
{
    ksgpu::Dtype dtype;
    
    long F = 0;     // number of distinct frequency subbands
    long P = 0;     // number of peak-finding kernels

    // Used internally to define layout.
    long Pouter = 0;
    long Pinner = 0;
    long Tinner = 0;
    long touter_byte_stride = 0;

    // Copies weights array from host to GPU (also converts fp32 -> fp16 if needed).
    // The source array has the straightforward layout (nbeams, Dbar, Tbar, P, F).
    // The destination array has the complicated layout assumed by the GPU kernel.
    // This function is intended to help hide details of the GPU layout.
    // Note: poorly optimized! (Intended for unit tests.)
    
    ksgpu::Array<void> to_gpu(const ksgpu::Array<float> &src);

    // Throws an exception if anything is wrong.
    void validate() const;
};


// -------------------------------------------------------------------------------------------------


struct PeakFindingKernelParams2
{
    FrequencySubbands fs;  // includes pf_rank, F, M
    ksgpu::Dtype dtype;

    long max_kernel_width = 0;
    long beams_per_batch = 0;
    long total_beams = 0;

    // Peak-finding input array has shape (beams_per_batch, ndm_out, fs.M, nt_in).
    // Output arrays have shape (beams_per_batch, ndm_out, nt_out).
    // Weight array has shape (beams_per_batch, ndm_wt, nt_wt, nprofiles, fs.F).

    long ndm_out = 0;
    long ndm_wt = 0;
    long nt_out = 0;
    long nt_in = 0;
    long nt_wt = 0;

    void validate() const;  // throws an exception if anything is wrong
};


struct ReferencePeakFindingKernel2
{
    // Parameters specified at construction.
    // params include: beams_per_batch, total_beams, ndm_out, ndm_wt, nt_out, nt_in, nt_wt, fs.F, fs.M  
    PeakFindingKernelParams2 params;
    long Dcore = 0;

    // Derived parameters, computed in constructor.
    long Dout = 0;             // = (nt_in/nt_out) = time downsampling factor of output array 
    long nbatches = 0;         // = (total_beams / beams_per_batch)
    long nprofiles = 0;        // = (3 * log2(max_kernel_width) + 1)

    // Note that the reference kernel uses float32, regardless of what dtype is specified.
    // All arrays must be fully contiguous (this could be changed if needed).
    
    ReferencePeakFindingKernel2(const PeakFindingKernelParams2 &params, long Dcore);

    void apply(ksgpu::Array<void> &out_max,      // shape (beams_per_batch, ndm_out, nt_out)
               ksgpu::Array<uint> &out_argmax,   // shape (beams_per_batch, ndm_out, nt_out)
               const ksgpu::Array<void> &in,     // shape (beams_per_batch, ndm_out, M, nt_in)
               const ksgpu::Array<void> &wt,     // shape (beams_per_batch, ndm_wt, nt_wt, nprofiles, F)
               long ibatch);
    
    void _init_tmp_arrays(const ksgpu::Array<float> &in, long ibatch);
    void _peak_find(ksgpu::Array<float> &out_max, ksgpu::Array<uint> &out_argmax, const ksgpu::Array<float> &wt);
    void _eval_tokens(ksgpu::Array<float> &out_max, const ksgpu::Array<uint> &in_tokens, const ksgpu::Array<float> &wt);

    // At "level" l (where 0 <= l < log2(E)), we have an array 'tmp_arr' containing input
    // array elements downsampled by 2^l (prepadded with data from the previous chunk).
    //
    //  - tmp_dt[l]: step size (in time) of temp array
    //  - tmp_nt[l]: number of time samples in temp array
    //  - tmp_arr[l]: array of shape is (B, D, M, tmp_nt[l]))
    //
    // Array element tmp_arr[l][b,d,m,j] is obtained by summing:
    //
    //    in[b,d,m, ilo:(ilo+2^l)]    where ilo = -tpad + j*tmp_dt[l]
    //
    // We also compute the following members, for convenience in computing
    // "triggers":
    //
    //   - tmp_iout[l]: "base" time index in tmp_arr
    //   - tmp_nout[l]: number of tmp time indices per output time index
    //   - tmp_sout[l]: spacing between tmp time indices that are logically contiguous
    //
    // To be totally precise about what these mean, when we compute triggers
    // for p=3*l+q and 0 <= tout < nt_out, we use a loop like this:
    //
    //   I = tmp_iout[l];   // base
    //   N = tmp_nout[l];   // count
    //   S = tmp_sout[l];   // spacing
    //
    //   for (n = 0; n < N; n++) {
    //       float x_0 = tmp_arr[l][b,d,m, I + tout*N + n - (q-1)*S];
    //       float x_1 = tmp_arr[l][b,d,m, I + tout*N + n - (q-2)*S];
    //           ...
    //       float x_end = tmp_arr[l][b,d,m, I + tout*N + n];

    long tpad = 0;  // prepadding (in "input time samples"), same for all levels
    long num_levels = 0;
    long expected_ibatch = 0;  // checked in apply()

    std::vector<long> tmp_dt;
    std::vector<long> tmp_nt;
    std::vector<long> tmp_iout;
    std::vector<long> tmp_nout;
    std::vector<long> tmp_sout;
    std::vector<ksgpu::Array<float>> tmp_arr;   // shape (B, D, M, tmp_nt[l])

    // The reference kernel allocates persistent state in the constructor (not a separate
    // allocate() method). We just save the last (tpad) samples from the previous chunk.

    ksgpu::Array<float> pstate;  // shape (total_beams, A, M, tpad)
};


struct GpuPeakFindingKernel2
{
    // -------------------- Internals start here --------------------

    struct RegistryKey
    {
        ksgpu::Dtype dtype;   // either float16 or float32
        std::vector<long> subband_counts;  // length (rank+1)
        long Tinner = 0;
        long Dout = 0;
        long E = 0;
    };

    struct RegistryValue
    {
        // cuda_kernel(const void *in, void *out_max, uint *out_argmax, const void *wt, void *pstate, uint Tin, uint WDd, uint WDt)
        //
        // in: shape (B*W, M, Tin)
        // out_max: shape (B*W, Tin/Dout)
        // out_argmax: shape (B*W, Tin/Dout)
        // wt: complicated format (see class PfWeightLayout), Dbar=(B*W*2^pf_rank/WDd), Tbar=(Tin/WDt)
        // pstate: (B*W, PW32) where PW32 = pstate 32-bit registers per warp
        //
        // WDd, WDt: coarse-graining factors for weight array.
        // WDd must be a multiple of 2**pf_rank.
        // If Tinner > 1, then WDt must equal (32*SW)/Tinner, and Tin must be a multiple of (32*SW).
        // If Tinner == 1, then WDt must be a multiple of (32*SW), and Tin must be a multiple of WDt.
        // Here, SW = (128/sizeof(dtype)) is the simd width.

        void (*cuda_kernel)(const void *, void *, uint *, const void *, void *, uint, uint, uint) = nullptr;

        // Layout of peak-finding weights in GPU memory, expected by the kernel.
        GpuPfWeightLayout pf_weight_layout;

        int Dcore = 0;   // internal downsampling factor (see discussion above)
    };

    using Registry = KernelRegistry<RegistryKey, RegistryValue>;

    // Static member function to access registry.
    static Registry &registry();
};

// Defined in PeakFindingKernel.cu
extern bool operator==(const GpuPeakFindingKernel2::RegistryKey &k1, const GpuPeakFindingKernel2::RegistryKey &k2);
extern std::ostream &operator<<(std::ostream &os, const GpuPeakFindingKernel2::RegistryKey &k);
extern std::ostream &operator<<(std::ostream &os, const GpuPeakFindingKernel2::RegistryValue &v);


// -------------------------------------------------------------------------------------------------
//
// Unit tests of peak-finding "microkernels".


extern void test_pf_weight_reader_microkernel();
extern void test_pf_output_microkernel();


// Everything after this point is KernelRegistry boilerplate.
// First, a registry for test_pf_weight_reader kernels.
struct TestPfWeightReader
{
    struct RegistryKey
    {
        ksgpu::Dtype dtype;     // either float16 or float32
        std::vector<long> subband_counts;  // length (rank+1)
        long Dcore = 0;
        long Tinner = 0;
        long P = 0;
    };

    struct RegistryValue
    {
        // The test kernel is called as (32 threads, 1 threadblock):
        //   void kernel(void *out, const void *in, uint Tin, uint WDt);
        //
        // where 'out' has shape (Tin/Dcore, Mouter * Minner, Pouter * Pinner)
        // and 'in' is managed by 'struct GpuPfWeightLayout' (see above).
        //
        // If Tinner > 1, then WDt must equal (32*SW)/Tinner, and Tin must be a multiple of (32*SW).
        // If Tinner == 1, then WDt must be a multiple of (32*SW), and Tin must be a multiple of WDt.
        
        void (*cuda_kernel)(void *, const void *, uint, uint) = nullptr;

        // Layout of peak-finding weights in GPU memory, expected by the kernel.
        GpuPfWeightLayout pf_weight_layout;

        long Mouter = 0;
        long Minner = 0;
    };

    using Registry = KernelRegistry<RegistryKey, RegistryValue>;

    // Static member function to access registry.
    static Registry &registry();        
};

// Defined in GpuPeakFindingKernel.cu
extern bool operator==(const TestPfWeightReader::RegistryKey &k1, const TestPfWeightReader::RegistryKey &k2);
extern std::ostream &operator<<(std::ostream &os, const TestPfWeightReader::RegistryKey &k);
extern std::ostream &operator<<(std::ostream &os, const TestPfWeightReader::RegistryValue &v);


// Registry for test_pf_output kernels.
struct TestPfOutput2
{
    
    struct RegistryKey
    {
        ksgpu::Dtype dtype;   // either float16 or float32
        int Dout = 0;
    };

    struct RegistryValue
    {
        // The test kernel is called as (32 threads, 1 threadblock):
        //   void kernel(void *zout, uint *aout, void *zin, uint *ain, uint Tin);
        //
        // where 'zout' and 'zin' have type RegistryKey::dtype, and:
        //   zout.shape == aout.shape == (Tin//Dout)
        //   zin.shape == ain.shape == (4, Tin)

        void (*cuda_kernel) (void *, uint *, void *, uint *, uint) = nullptr;
    };
    
    using Registry = KernelRegistry<RegistryKey, RegistryValue>;

    // Static member function to access registry.
    static Registry &registry();    
};

// Defined in GpuPeakFindingKernel.cu
extern bool operator==(const TestPfOutput2::RegistryKey &k1, const TestPfOutput2::RegistryKey &k2);
extern std::ostream &operator<<(std::ostream &os, const TestPfOutput2::RegistryKey &k);
extern std::ostream &operator<<(std::ostream &os, const TestPfOutput2::RegistryValue &v);


}  // namespace pirate

#endif // _PIRATE_PEAK_FINDING_KERNEL_HPP
