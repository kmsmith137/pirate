#ifndef _PIRATE_PEAK_FINDING_KERNEL_HPP
#define _PIRATE_PEAK_FINDING_KERNEL_HPP

#include <vector>
#include <unordered_map>
#include <ksgpu/Dtype.hpp>
#include <ksgpu/Array.hpp>

#include "BumpAllocator.hpp"
#include "FrequencySubbands.hpp"
#include "KernelRegistry.hpp"
#include "trackers.hpp"  // BandwidthTracker


namespace pirate {
#if 0
}  // editor auto-indent
#endif

// The peak-finding kernel is the last step in the dedispersion transform.
// Its output is a pair of 2-d arrays (out_max, out_argmax), and it has an auxiliary
// input 'weights' array. These inputs and outputs are nontrivial to explain, hence
// this long comment.
//
// The 'out_max' array is a 2-d "SNR map" indexed by (coarse-grained dm, coarse-grained time).
// The array shape is (ndm_out, nt_out), where ndm_out and nt_out are members of 
// struct PeakFindingKernelParams.
//
// Each element of the 'out_max' array is a detection significance in "sigmas". Conceptually,
// it is obtained by taking a maximum over a 4-D "trial array" indexed by (frequency subband, 
// fine-grained DM, fine-grained arrival time, peak-finding profile). The purpose of
// searching over trial frequency subbands is to increase SNR for FRBs that do not span
// the full frequency range.
// 
// The 'out_argmax' array has the same shape (ndm_out, nt_out). Each element of the
// out_argmax array is an uint32 "token" which indicates which element of the 4-d trial
// array (from the previous paragraph) is responsible for the maximum SNR. 
// The tokens are defined as follows:
//
//   token = (t) | (p << 8) | (m << 16);   // 8+8+16 bits
//
//     where  0 <= t < (nt_in / nt_out)  indexes a fine-grained arrival time
//            0 <= p < P                 indexes a peak-finding profile (see below)
//            0 <= m < M                 indexes a "multiplet" (see below)
//
// It's convenient to combine the (frequency_subband, fine-grained DM) axes into
// a single axis, indexed by a "multiplet" 0 <= m < M. The FrequencySubband helper
// class contains information about the frequency subband scheme. In particular:
//
//   FrequencySubband::F = number of distinct frequency subbands
//   FrequencySubband::M = number of distinct multiplets (freq_subband, fine_dm)
//
// The peak-finding profile 0 <= p < P indexes a trial profile (in time) which is
// used to implement a (roughly) matched filter for a range of pulse widths. The
// details of these profiles will be described later (FIXME), but for now we note
// that the total number of profiles P is given by:
//
//   P = 1 + 3 * log2(max_kernel_width)
//
// where max_kernel_width is specified in DedispersionConfig, and is also a member
// of 'struct PeakFindingKernelParams'.
//
// Now we have fully described the 'out_max' and 'out_argmax' arrays.
// Next, we describe the weights array 'wt', which is an argument to the peak-finder.
//
// For each set of trial parameters, we compute an "unnormalized" peak-finding output
// in whatever normalization is convenient for the GPU kernel. The 'wt' array contains
// the multiplier which converts to detection significants in "sigmas". We make the
// approximation that the normalization only depends on the multiplet 0 <= m < M
// (see above) through its frequency subband 0 <= f < F. Then, the 'wt' array is
// a logical 4-d array (for each beam) with shape:
//
//   (ndm_wt, nt_wt, P, F)    (*)
//
// where (ndm_wt, nt_wt) are obtained by applying downsampling factors to (ndm_in, nt_in).
// These "weights" downsampling factors are independent of the downsampling factors used
// to obtain (ndm_out, nt_in), and are specified in DedispersionConfig.
//
// On the CPU, the weights array is represented as a 4-d array with shape (*), but
// on the GPU we use a complicated, non-contiguous representation which is convenient
// for the GPU kernel. The helper class 'GpuPfWeightLayout' is intended to hide the
// details of the GPU memory layout, by providing helper functions to convert from (*).


struct PeakFindingKernelParams
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


struct ReferencePeakFindingKernel
{
    // Parameters specified at construction.
    PeakFindingKernelParams params;  // beams_per_batch, total_beams, ndm_out, ndm_wt, nt_out, nt_in, nt_wt
    FrequencySubbands fs;             // pf_rank, F, M
    long Dcore = 0;

    // Derived parameters, computed in constructor.
    long Dout = 0;             // = (nt_in/nt_out) = time downsampling factor of output array 
    long nbatches = 0;         // = (total_beams / beams_per_batch)
    long nprofiles = 0;        // = (3 * log2(max_kernel_width) + 1)

    // Note that the reference kernel uses float32, regardless of what dtype is specified.
    // All arrays must be fully contiguous (this could be changed if needed).
    
    ReferencePeakFindingKernel(const PeakFindingKernelParams &params, long Dcore);

    // The reference kernel uses float32, regardless of what dtype is specified.
    void apply(ksgpu::Array<float> &out_max,     // shape (beams_per_batch, ndm_out, nt_out)
               ksgpu::Array<uint> &out_argmax,   // shape (beams_per_batch, ndm_out, nt_out)
               const ksgpu::Array<float> &in,    // shape (beams_per_batch, ndm_out, params.fs.M, nt_in)
               const ksgpu::Array<float> &wt,    // shape (beams_per_batch, ndm_wt, nt_wt, nprofiles, fs.F)
               long ibatch,                      // 0 <= ibatch < nbatches
               bool debug = false);              // enables verbose debugging output

    // eval_tokens() is used in testing, and takes some time to explain!
    // 
    // Inputs:
    //   - 'wt' array (same meaning as apply())
    //   - 'in_tokens' array (integer-valued, same format as 'out_argmax' in apply())
    //
    // Output:
    //   - 'out' array (same shape as 'out' in apply())
    //
    // The output array is computed by rerunning the peak-finding computation, but
    // instead of taking the "max" over the 4-d trials array, we take a single
    // trial (per output array element) specified by the 'in_tokens'.
    //
    // This is useful in a testing context as follows.
    // Suppose that we want to compute the reference/gpu peak-finding kernels.
    // We run both kernels, obtaining four arrays:
    //  (cpu_max, cpu_argmax, gpu_max, gpu_argmax)
    //
    // The 'cpu_max' and 'gpu_max' arrays should agree, and can be compared directly.
    // However, because of near-ties and roundoff error, the argmax arrays won't
    // necessarily be equal, and can't be compared exactly. However, the following
    // procedure is a complete test of 'gpu_argmax'.
    //
    //    eval_tokens(gpu_argmax) -> tmp
    //    assert_arrays_equal(tmp, cpu_max)
    //
    // For an example, see GpuPeakFindingKernel::test() in PeakFindingKernel.cu.
    //
    // Note: eval_tokens() reads temp arrays, which are computed in apply() and stored
    // in the ReferencePeakFinder. You'll notice that eval_tokens() doesn't take an 'in'
    // array -- the input data comes from the most recent call to apply().

    void eval_tokens(ksgpu::Array<float> &out,  // output array, shape (beams_per_batch, ndm_out, nt_out)
        const ksgpu::Array<uint> &in_tokens,    // input array, shape (beams_per_batch, ndm_out, params.fs.M, nt_out)
        const ksgpu::Array<float> &wt);         // input array, shape (beams_per_batch, ndm_wt, nt_wt, nprofiles, fs.F)

    // Make a mean-zero input array for testing.
    // Returns shape (nbeams_per_batch, ndm_out, params.fs.M, nt_in)
    ksgpu::Array<float> make_random_input_array();

    // Make an interesting weights array for testing.
    // Returns shape (nbeams_per_batch, ndm_wt, nt_wt, nprofiles, fs.F)
    //
    // The 'subband_variance' array has shape (fs.F,) and is an estimate for
    // the variance of the peak-finder input, in frequency subband 0 <= f < F. 
    ksgpu::Array<float> make_random_weights(const ksgpu::Array<float> &subband_variances);

    // At "level" l (where 0 <= l < log2(Wmax)), we have an array 'tmp_arr' containing input
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

    // The reference rl allocates persistent state in the constructor (not a separate
    // allocate() method). We just save the last (tpad) samples from the previous chunk.

    ksgpu::Array<float> pstate;  // shape (total_beams, A, M, tpad)

    // Helper for eval_tokens()
    static std::runtime_error _bad_token(uint token, const char *why);
};


// GpuPfWeightLayout: describes the layout of peak-finding weights on the GPU.
//
// Peak-finding weights are logically a 5-d array with shape:
//
//   (beams_per_batch, ndm_wt, nt_wt, P, F)        (*)
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
    
    ksgpu::Array<void> to_gpu(const ksgpu::Array<float> &src) const;

    // Throws an exception if anything is wrong.
    void validate() const;
};


// Note that the GpuPeakFindingKernel is not actually used in the GpuDedisperser!
//
// Instead, the GpuDedisperser uses a coalesced kernel (CoalescedDdKernel2) which
// combines the second half of the dedispersion transform and peak-finding, without
// a global memory read-write cycle in between.
//
// The GpuPeakFindingKernel is useful because it implements a subset of 
// CoalescedDdKernel2, which can be unit-tested in isolation. (When I was first
// writing all this code, passing unit tests for GpuPeakFindingKernel took a
// while, and CoalescedDdKernel2 was smooth sailing afterwards.)

struct GpuPeakFindingKernel
{
    GpuPeakFindingKernel(const PeakFindingKernelParams &params);

    void allocate(BumpAllocator &allocator);

    // The 'weights' array has logical shape (beams_per_batch, ndm_wt, nt_wt, P, F),
    // but is passed to the gpu kernel in a complicated, non-contiguous layout. To put
    // an array into the proper layout, call GpuPfWeightLayout::to_gpu().

    void launch(ksgpu::Array<void> &out_max,      // shape (beams_per_batch, ndm_out, nt_out)
                ksgpu::Array<uint> &out_argmax,   // shape (beams_per_batch, ndm_out, nt_out)
                const ksgpu::Array<void> &in,     // shape (beams_per_batch, ndm_out, M, nt_in)
                const ksgpu::Array<void> &wt,     // see comment above
                long ibatch,                      // 0 <= ibatch < nbatches
                cudaStream_t stream);             // NULL stream is allowed, but is not the default);

    // If short_circuit=true, then we run some ReferencePeakFindingKernel tests, 
    // but don't test the GPU peak-finder.
    static void test(bool short_circuit=false);

    // ------------------------  Members  ------------------------

    PeakFindingKernelParams params;  // beams_per_batch, total_beams, ndm_out, ndm_wt, nt_out, nt_in, nt_wt
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
        long Wmax = 0;
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

    // Static member functions to query registry.
    static long registry_size() { return registry().size(); }
    static void show_registry() { registry().show(); }
};

// Defined in PeakFindingKernel.cu
extern bool operator==(const GpuPeakFindingKernel::RegistryKey &k1, const GpuPeakFindingKernel::RegistryKey &k2);
extern std::ostream &operator<<(std::ostream &os, const GpuPeakFindingKernel::RegistryKey &k);
extern std::ostream &operator<<(std::ostream &os, const GpuPeakFindingKernel::RegistryValue &v);


// -------------------------------------------------------------------------------------------------
//
// Unit tests of peak-finding "microkernels".
//
// The code generator has a modular structure, where larger kernels are built up from "microkernels".
// Since debugging autogenerated kernels can be painful, it's useful to take it one step at a time,
// by individually testing the microkernels, before attempting to pass unit tests on a large kernel.


// Everything after this point is KernelRegistry boilerplate.
// First, a registry for test_pf_weight_reader kernels.
struct PfWeightReaderMicrokernel
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

    // Static member functions to query registry.
    static long registry_size() { return registry().size(); }
    static void show_registry() { registry().show(); }

    // Static member function: runs one randomized test iteration.
    static void test();
};

// Defined in GpuPeakFindingKernel.cu
extern bool operator==(const PfWeightReaderMicrokernel::RegistryKey &k1, const PfWeightReaderMicrokernel::RegistryKey &k2);
extern std::ostream &operator<<(std::ostream &os, const PfWeightReaderMicrokernel::RegistryKey &k);
extern std::ostream &operator<<(std::ostream &os, const PfWeightReaderMicrokernel::RegistryValue &v);


// Registry for test_pf_output kernels.
struct PfOutputMicrokernel
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

    // Static member functions to query registry.
    static long registry_size() { return registry().size(); }
    static void show_registry() { registry().show(); }

    // Static member function: runs one randomized test iteration.
    static void test();
};

// Defined in GpuPeakFindingKernel.cu
extern bool operator==(const PfOutputMicrokernel::RegistryKey &k1, const PfOutputMicrokernel::RegistryKey &k2);
extern std::ostream &operator<<(std::ostream &os, const PfOutputMicrokernel::RegistryKey &k);
extern std::ostream &operator<<(std::ostream &os, const PfOutputMicrokernel::RegistryValue &v);


}  // namespace pirate

#endif // _PIRATE_PEAK_FINDING_KERNEL_HPP
