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
    // of "tokens" of the form (t) | (p << 8) | (m << 16).

    // For debugging/testing.
    static void validate_subband_counts(const std::vector<long> &subband_counts);
    
    void show_token(uint token, std::ostream &os = std::cout) const;
    void show(std::ostream &os = std::cout) const;
};


// GpuPfWeightLayout: describes the layout of peak-finding weights on the GPU.
//
// Peak-finding weights are logically a 5-d array with shape:
//
//   (nbeams, ndm_wt, nt_wt, P, F)        (*)
//
// where:
//
//  - ndm_wt = number of DMs in weights array (downsampled relative to tree)
//  - nt_wt = number of time samples in weights array (downsampled relative to tree)
//  - P = number of peak-finding profiles
//  - F = number of frequency subbands F
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

    std::vector<long> get_shape(long nbeams, long ndm_wt, long nt_wt) const;
    std::vector<long> get_strides(long nbeams, long ndm_wt, long nt_wt) const;

    // Copies weights array from host to GPU (also converts fp32 -> fp16 if needed).
    // The source array has the straightforward layout (nbeams, ndm_wt, nt_wt, P, F).
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
    std::vector<long> subband_counts;   // same meaning as FrequencySubbands.subband_counts
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
    PeakFindingKernelParams2 params;  // beams_per_batch, total_beams, ndm_out, ndm_wt, nt_out, nt_in, nt_wt
    FrequencySubbands fs;             // pf_rank, F, M
    long Dcore = 0;

    // Derived parameters, computed in constructor.
    long Dout = 0;             // = (nt_in/nt_out) = time downsampling factor of output array 
    long nbatches = 0;         // = (total_beams / beams_per_batch)
    long nprofiles = 0;        // = (3 * log2(max_kernel_width) + 1)

    // Note that the reference kernel uses float32, regardless of what dtype is specified.
    // All arrays must be fully contiguous (this could be changed if needed).
    
    ReferencePeakFindingKernel2(const PeakFindingKernelParams2 &params, long Dcore);

    // The reference kernel uses float32, regardless of what dtype is specified.
    void apply(ksgpu::Array<float> &out_max,     // shape (beams_per_batch, ndm_out, nt_out)
               ksgpu::Array<uint> &out_argmax,   // shape (beams_per_batch, ndm_out, nt_out)
               const ksgpu::Array<float> &in,    // shape (beams_per_batch, ndm_out, params.fs.M, nt_in)
               const ksgpu::Array<float> &wt,    // shape (beams_per_batch, ndm_wt, nt_wt, nprofiles, params.fs.F)
               long ibatch);

    void eval_tokens(ksgpu::Array<float> &out_max,  // shape (beams_per_batch, ndm_out, nt_out)
        const ksgpu::Array<uint> &in_tokens,        // shape (beams_per_batch. ndm_out, params.fs.M, nt_out)
        const ksgpu::Array<float> &wt);             // shape (beams_per_batch, ndm_wt, nt_wt, nprofiles, params.fs.F)

    // Make a mean-zero input array for testing.
    // Returns shape (nbeams_per_batch, ndm_out, params.fs.M, nt_in)
    ksgpu::Array<float> make_random_input_array();

    // Make an interesting weights array for testing.
    // Returns shape (nbeams_per_batch, ndm_wt, nt_wt, nprofiles, params.fs.F)
    ksgpu::Array<float> make_random_weights();

    // FIXME absorb these into apply()
    void _init_tmp_arrays(const ksgpu::Array<float> &in, long ibatch);
    void _peak_find(ksgpu::Array<float> &out_max, ksgpu::Array<uint> &out_argmax, const ksgpu::Array<float> &wt);

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

    // Helper for eval_tokens()
    static std::runtime_error _bad_token(uint token, const char *why);
};


struct GpuPeakFindingKernel2
{
    GpuPeakFindingKernel2(const PeakFindingKernelParams2 &params);

    void allocate();

    void launch(ksgpu::Array<void> &out_max,      // shape (beams_per_batch, ndm_out, nt_out)
                ksgpu::Array<uint> &out_argmax,   // shape (beams_per_batch, ndm_out, nt_out)
                const ksgpu::Array<void> &in,     // shape (beams_per_batch, ndm_out, M, nt_in)
                const ksgpu::Array<void> &wt,     // from GpuPfWeightLayout::to_gpu()
                long ibatch,                      // 0 <= ibatch < nbatches
                cudaStream_t stream);             // NULL stream is allowed, but is not the default);

    static void test(bool short_circuit=false);

    // ------------------------  Members  ------------------------

    PeakFindingKernelParams2 params;  // beams_per_batch, total_beams, ndm_out, ndm_wt, nt_out, nt_in, nt_wt
    FrequencySubbands fs;             // pf_rank, F, M

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

    // It's easiest to represent the persistent_state as dtype=uint, since we parameterize
    // its size using registry_value.PW32 = "number of 32-bit registers per warp".
    ksgpu::Array<uint> persistent_state;  // shape (total_beams, ndm_out, registry_value.PW32)
    bool is_allocated = false;
    long expected_ibatch = 0;

    // -------------------- Internals start here --------------------

    struct RegistryKey
    {
        ksgpu::Dtype dtype;   // either float16 or float32
        std::vector<long> subband_counts;  // length (rank+1)
        long Tinner = 0;      // for weights
        long Dout = 0;
        long E = 0;
    };

    struct RegistryValue
    {
        // cuda_kernel(const void *in, void *out_max, uint *out_argmax, const void *wt, void *pstate, uint nt_in, uint ndm_out_per_wt, uint nt_in_per_wt)
        //
        // in: shape (B*W, M, nt_in)
        // out_max: shape (B*W, nt_in/Dout)
        // out_argmax: shape (B*W, nt_in/Dout)
        // wt: complicated format (from class PfWeightLayout, see below)
        // pstate: (B*W, PW32) where PW32 = pstate 32-bit registers per warp
        // nt_in: number of input time samples
        // ndm_out_per_wt: dm downsampling factor for weight array (relative to *output*)
        // nt_in_per_wt: time downsampling factor for weight array (relative to *input*)
        //
        // Caller is responsible for checking:
        //   - ndm_out_per_wt and nt_in_per_wt are powers of two
        //   - If Tinner == 1, then nt_in_per_wt >= (32 * simd_width)
        //   - If Tinner > 1, then nt_in_per_wt == (32 * simd_width) / Tinner
        //   - nt_in is a mutliple of nt_in_per_wt
        //   - nt_in is a multiple of (32 * simd_width)
        //   - total warps (B*W) is a multiple of ndm_out_per_wt
        //
        // Launch with {32,W,1} threads/block and {B,1,1} threadblocks.

        void (*cuda_kernel)(const void *, void *, uint *, const void *, void *, uint, uint, uint) = nullptr;

        // Layout of peak-finding weights in GPU memory, expected by the kernel.
        GpuPfWeightLayout pf_weight_layout;

        long Dcore = 0;   // internal downsamplingq
        //  factor (see discussion above)
        long PW32 = -1;   // number of 32-bit registers per warp (= "one pf_rank")
    };

    // Non-static members for interacting with the kernel registry.
    RegistryKey registry_key;
    RegistryValue registry_value;

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
        // cuda_kernel(void *out, const void *in, uint nt_in, uint nt_in_per_wt)
        //
        // out: shape (nt_in/Dcore, Mouter*Minner, Pouter*Pinner)
        // in: shape (nt_in/(nt_in_per_wt*Tinner), Pouter, F, Tinner, Pinner)
        // nt_in: number of input time samples
        // nt_in_per_wt: time downsampling factor for weight array
        //
        // Caller is responsible for checking:
        //   - nt_in_per_wt is a power of two
        //   - If Tinner == 1, then nt_in_per_wt >= (32 * simd_width)
        //   - If Tinner > 1, then nt_in_per_wt == (32 * simd_width) / Tinner
        //   - nt_in is a mutliple of nt_in_per_wt
        //   - nt_in is a multiple of (32 * simd_width)
        //
        // Launch with 32 threads, 1 block.
        
        void (*cuda_kernel)(void *, const void *, uint, uint) = nullptr;

        // Layout of peak-finding weights in GPU memory, expected by the kernel.
        GpuPfWeightLayout pf_weight_layout;

        long Mouter = 0;
        long Minner = 0;
    };

    using Registry = KernelRegistry<RegistryKey, RegistryValue>;

    // Static member function to access registry.
    static Registry &registry();

    // Static member function: runs one randomized test iteration.
    static void test();
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
        // cuda_kernel(void *zout, uint *aout, void *zin, uint *ain, uint nt_in)
        //
        // zout: shape (nt_in//Dout) == (nt_in//4)
        // aout: shape (nt_in//Dout) == (nt_in//4)
        // zin: shape (4, nt_in)
        // ain: shape (4, nt_in)
        // nt_in: number of input time samples
        //
        // Caller is responsible for checking:
        //   - nt_in is a multiple of (32 * simd_width)

        void (*cuda_kernel) (void *, uint *, void *, uint *, uint) = nullptr;
    };
    
    using Registry = KernelRegistry<RegistryKey, RegistryValue>;

    // Static member function to access registry.
    static Registry &registry();

    // Static member function: runs one randomized test iteration.
    static void test();
};

// Defined in GpuPeakFindingKernel.cu
extern bool operator==(const TestPfOutput2::RegistryKey &k1, const TestPfOutput2::RegistryKey &k2);
extern std::ostream &operator<<(std::ostream &os, const TestPfOutput2::RegistryKey &k);
extern std::ostream &operator<<(std::ostream &os, const TestPfOutput2::RegistryValue &v);


}  // namespace pirate

#endif // _PIRATE_PEAK_FINDING_KERNEL_HPP
