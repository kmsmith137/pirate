#include "../include/pirate/CoalescedDdKernel2.hpp"
#include "../include/pirate/DedispersionConfig.hpp"  // used in CoalescedDdKernel2::time()
#include "../include/pirate/DedispersionPlan.hpp"    // used in CoalescedDdKernel2::time()
#include "../include/pirate/DedispersionTree.hpp"    // used in get_registry_dcore(dtype, tree)
#include "../include/pirate/MegaRingbuf.hpp"
#include "../include/pirate/inlines.hpp"
#include "../include/pirate/utils.hpp"
#include "../include/pirate/constants.hpp"     // cuda_max_static_shmem_bytes

#include <mutex>
#include <cstring>   // memcpy()
#include <sstream>
#include <iomanip>
#include <unordered_map>

#include <ksgpu/Array.hpp>
#include <ksgpu/cuda_utils.hpp>
#include <ksgpu/KernelTimer.hpp>

using namespace std;
using namespace ksgpu;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// Shared core of _make_registry_key() and get_registry_dcore() below -- keeps the
// Tinner formula and key-field mapping in one place.
static CoalescedDdKernel2::RegistryKey _make_key(
    const Dtype &dtype, long dd_rank, long max_kernel_width,
    long nt_in, long nt_out, long nt_wt, const vector<long> &subband_counts)
{
    CoalescedDdKernel2::RegistryKey key;
    key.dtype = dtype;
    key.dd_rank = dd_rank;
    key.Dout = xdiv(nt_in, nt_out);
    key.Wmax = max_kernel_width;
    key.subband_counts = subband_counts;

    // Recall the definition of Tinner (used for weight layout, see comments in
    // cuda_generator.PeakFinder.py):
    //
    //   Tinner = max(32*SW/nt_in_per_wt, 1)

    long SW = xdiv(32, dtype.nbits);      // simd width
    long nt_in_per_wt = xdiv(nt_in, nt_wt);
    key.Tinner = (nt_in_per_wt < 32*SW) ? xdiv(32*SW, nt_in_per_wt) : 1;

    return key;
}


// File-local. Warning: a RegistryKey contains no Dcore, so registry().get() may return
// a kernel whose Dcore does not match PeakFindingKernelParams::Dcore. (The
// CoalescedDdKernel2 constructor checks this by hand.)
static CoalescedDdKernel2::RegistryKey _make_registry_key(
    const DedispersionKernelParams &dd_params, const PeakFindingKernelParams &pf_params)
{
    // Note: does not call pf_params.validate(), since test_random() calls this function
    // (to peek the registry Dcore) while pf_params.Dcore is still unset.

    return _make_key(pf_params.dtype, dd_params.dd_rank, pf_params.max_kernel_width,
                     pf_params.nt_in, pf_params.nt_out, pf_params.nt_wt, pf_params.subband_counts);
}


// File-local. Maps a DedispersionTree to the registry key of the cdd2 kernel which
// implements it. Shared by get_registry_dcore() and _all_kernels_registered().
static CoalescedDdKernel2::RegistryKey _make_tree_key(
    const Dtype &dtype, const DedispersionTree &tree)
{
    return _make_key(dtype, tree.dd_rank, tree.pf.max_width,
                     tree.nt_ds, tree.nt_out, tree.nt_wt,
                     tree.frequency_subbands.subband_counts);
}


// Static member function. See doc-comment in CoalescedDdKernel2.hpp.
long CoalescedDdKernel2::get_registry_dcore(
    const ksgpu::Dtype &dtype, const DedispersionTree &tree)
{
    return registry().get(_make_tree_key(dtype,tree), /*init_kernel=*/ false).Dcore;
}


// File-local. Returns true if every tree in 'plan' has a cdd2 kernel compiled into this
// build. Called by time_one(), which uses it to skip (rather than throw) when kernels are
// missing: makefile_helper.py only emits the "production" cdd2 kernels when its
// 'production_kernels' flag is set, and even then only for the (dtype, subband_counts, ...)
// combinations listed there.
//
// The 'plan' argument must have been constructed with cdd2_kernel_required=false, since a
// cdd2_kernel_required=true plan queries the registry itself (and throws on a missing kernel).
static bool _all_kernels_registered(const Dtype &dtype, const DedispersionPlan &plan)
{
    xassert(!plan.cdd2_kernel_required);

    for (const DedispersionTree &tree: plan.trees)
        if (!CoalescedDdKernel2::registry().has_key(_make_tree_key(dtype,tree)))
            return false;

    return true;
}


CoalescedDdKernel2::CoalescedDdKernel2(const DedispersionKernelParams &dd_params_, const PeakFindingKernelParams &pf_params_) :
    dd_params(dd_params_), pf_params(pf_params_), fs(pf_params_.subband_counts)
{
    dd_params.validate();
    pf_params.validate();
    xassert(dd_params.dd_rank > 0);  // FIXME define _r0 for testing

    xassert(dd_params.nspec == 1);
    xassert(dd_params.apply_input_residual_lags);
    xassert(dd_params.input_is_ringbuf);
    xassert(!dd_params.output_is_ringbuf);
    xassert(dd_params.mega_ringbuf);
    xassert(dd_params.consumer_id >= 0);
    xassert(dd_params.consumer_id < dd_params.mega_ringbuf->num_consumers);
    xassert(dd_params.nspec == 1);

    this->nsegments_per_beam = pow2(dd_params.dd_rank+dd_params.amb_rank) * xdiv(dd_params.ntime,dd_params.nt_per_segment);
    xassert_shape_eq(dd_params.mega_ringbuf->consumer_quadruples.at(dd_params.consumer_id), ({nsegments_per_beam,4}));

    xassert_eq(pf_params.dtype, dd_params.dtype);
    xassert_eq(pf_params.beams_per_batch, dd_params.beams_per_batch);
    xassert_eq(pf_params.total_beams, dd_params.total_beams);
    xassert_eq(pf_params.nt_in, dd_params.ntime);

    // One output DM per warp of the second dedispersion stage, hence 'rank1' here rather
    // than fs.pf_rank. If pf_rank < rank1, the extra DM bits are folded into the
    // peak-finder's multiplet index and max-reduced away. See CoalescedDdKernel2.hpp.
    long rank1 = dd_params.dd_rank - (dd_params.dd_rank / 2);
    xassert_le(fs.pf_rank, rank1);
    xassert_eq(pf_params.ndm_out, pow2(dd_params.dd_rank + dd_params.amb_rank - rank1));

    // The initialization logic below is mostly cut-and-paste from either the
    // PeakFindingKernel or GpuDedispersionKernel constructor.

    this->dtype = dd_params.dtype;
    this->nbatches = xdiv(dd_params.total_beams, dd_params.beams_per_batch);
    this->Dout = xdiv(pf_params.nt_in, pf_params.nt_out);
    this->nprofiles = 3 * log2(pf_params.max_kernel_width) + 1;

    this->registry_key = _make_registry_key(dd_params, pf_params);
    this->registry_value = registry().get(registry_key);

    // Derived parameters chosen by the kernel.
    this->pf_weight_layout = registry_value.pf_weight_layout;
    this->expected_wt_shape = pf_weight_layout.get_shape(pf_params.beams_per_batch, pf_params.ndm_wt, pf_params.nt_wt);
    this->expected_wt_strides = pf_weight_layout.get_strides(pf_params.beams_per_batch, pf_params.ndm_wt, pf_params.nt_wt);
    this->Dcore = registry_value.Dcore;

    // Caller-specified Dcore (e.g. from DedispersionPlan) must match the compiled kernel.
    xassert_eq(pf_params.Dcore, Dcore);

    // Important: ensure that caller-specified 'nt_per_segment' matches GPU kernel.
    xassert_eq(dd_params.nt_per_segment, registry_value.nt_per_segment);

    long B = dd_params.beams_per_batch;
    long A = pow2(dd_params.amb_rank);
    long D = pow2(dd_params.dd_rank);
    long S = xdiv(dtype.nbits, 8);
    long bw_in = B * A * D * dd_params.ntime * S;
    long bw_out_max = B * pf_params.ndm_out * pf_params.nt_out * S;
    long bw_out_argmax = B * pf_params.ndm_out * pf_params.nt_out * 4;
    long quads_nbytes = nsegments_per_beam * 4 * 4;
    long pstate_nbytes_per_beam = A * registry_value.pstate32_per_small_tree * 4;

    resource_tracker.add_kernel("cdd2", bw_in + bw_out_max + bw_out_argmax);
    resource_tracker.add_gmem_bw("cdd2_quads", B * quads_nbytes);
    resource_tracker.add_gmem_bw("cdd2_pstate", 2 * B * pstate_nbytes_per_beam);
    resource_tracker.add_gmem_bw("cdd2_weights", expected_wt_shape[0] * expected_wt_strides[0] * S);
 
    resource_tracker.add_gmem_footprint("persistent_state", dd_params.total_beams * pstate_nbytes_per_beam, true);
    resource_tracker.add_gmem_footprint("quadruples", quads_nbytes, true);
}


void CoalescedDdKernel2::allocate(BumpAllocator &allocator)
{
    if (is_allocated)
        throw runtime_error("double call to CoalescedDdKernel2::allocate()");

    if (!(allocator.aflags & af_gpu))
        throw runtime_error("CoalescedDdKernel2::allocate(): allocator.aflags must contain af_gpu");
    if (!(allocator.aflags & af_zero))
        throw runtime_error("CoalescedDdKernel2::allocate(): allocator.aflags must contain af_zero");

    long nbytes_before = allocator.get_nbytes_allocated();

    // Allocate persistent_state.
    long ninner = registry_value.pstate32_per_small_tree * xdiv(32, dd_params.dtype.nbits);
    std::initializer_list<long> shape = { dd_params.total_beams, pow2(dd_params.amb_rank), ninner };
    this->persistent_state = allocator.allocate_array<void>(dd_params.dtype, shape);

    // Copy host -> GPU.
    const Array<uint> &src = dd_params.mega_ringbuf->consumer_quadruples.at(dd_params.consumer_id);
    this->gpu_ringbuf_quadruples = allocator.allocate_array<uint>({nsegments_per_beam, 4});
    this->gpu_ringbuf_quadruples.fill(src);

    // Shape/stride check (paranoid).
    xassert_shape_eq(gpu_ringbuf_quadruples, ({nsegments_per_beam,4}));
    xassert(gpu_ringbuf_quadruples.is_fully_contiguous());
    xassert(gpu_ringbuf_quadruples.on_gpu());

    long nbytes_allocated = allocator.get_nbytes_allocated() - nbytes_before;
    // cout << "CoalescedDdKernel2: " << nbytes_allocated << " bytes allocated" << endl;
    xassert_eq(nbytes_allocated, resource_tracker.get_gmem_footprint());

    this->is_allocated = true;
}


void CoalescedDdKernel2::launch(
    ksgpu::Array<void> &out_max,      // shape (beams_per_batch, ndm_out, nt_out)
    ksgpu::Array<uint> &out_argmax,   // shape (beams_per_batch, ndm_out, nt_out)
    const ksgpu::Array<void> &in,     // shape (mega_ringbuf->gpu_global_nseg * nt_per_segment * nspec,)
    const ksgpu::Array<void> &wt,     // from GpuPfWeightLayout::to_gpu()
    long ichunk,                      // time-chunk index 0, 1, ...
    long ibatch,                      // 0 <= ibatch < nbatches
    cudaStream_t stream)              // NULL stream is allowed, but is not the default);
{
    xassert(this->is_allocated);
    xassert((ibatch >= 0) && (ibatch < nbatches));
    xassert(ichunk >= 0);

    xassert(out_max.dtype == dtype);
    xassert(in.dtype == dtype);
    xassert(wt.dtype == dtype);

    // Validate 'in' array: shape (mega_ringbuf->gpu_global_nseg * nt_per_segment * nspec,)
    long global_nseg = dd_params.mega_ringbuf->gpu_global_nseg;
    xassert_shape_eq(in, ({ global_nseg * dd_params.nt_per_segment * dd_params.nspec }));

    // Validate 'out' and 'out_argmax' arrays: shape (beams_per_batch, ndm_out, nt_out)
    xassert_shape_eq(out_max, ({ pf_params.beams_per_batch, pf_params.ndm_out, pf_params.nt_out }));
    xassert_shape_eq(out_argmax, ({ pf_params.beams_per_batch, pf_params.ndm_out, pf_params.nt_out }));

    // Validate 'wt' array. These checks will pass if 'wt' is the output of GpuPfWeightLayout::to_gpu().

    if (!wt.shape_equals(expected_wt_shape)) {
        stringstream ss;
        ss << "CoalescedDdKernel2::launch(): wt.shape=" << wt.shape_str()
           << ", expected_wt_shape=" << ksgpu::tuple_str(expected_wt_shape);
        throw runtime_error(ss.str());
    }

    if (!wt.strides_equal(expected_wt_strides)) {
        stringstream ss;
        ss << "CoalescedDdKernel2::launch(): wt.strides=" << wt.stride_str()
           << ", expected_wt_strides=" << ksgpu::tuple_str(expected_wt_strides);
        throw runtime_error(ss.str());
    }


    xassert(out_max.is_fully_contiguous());
    xassert(out_argmax.is_fully_contiguous());
    xassert(in.is_fully_contiguous());
    // Weights array is not fully contiguous -- see above.

    xassert(out_max.on_gpu());
    xassert(out_argmax.on_gpu());
    xassert(in.on_gpu());
    xassert(wt.on_gpu());

    // The global persistent_state array has shape { total_beams, pow2(params.amb_rank), ninner }.
    // We want to select a "slice"" of beams corresponding to the current batch.
    long b0 = (ibatch) * dd_params.beams_per_batch;
    long b1 = (ibatch+1) * dd_params.beams_per_batch;
    Array<void> pstate = this->persistent_state.slice(0, b0, b1);

    ulong nt_cumul = ichunk * dd_params.ntime;
    long rb_frame0 = (ichunk * dd_params.total_beams) + (ibatch * dd_params.beams_per_batch);
    long ndm_out_per_wt = xdiv(pf_params.ndm_out, pf_params.ndm_wt);
    long nt_in_per_wt = xdiv(pf_params.nt_in, pf_params.nt_wt);

    // Output beam-strides (axis 0), expressed as multiples of 32 bits. We compute these directly
    // from Array::strides, as if the output arrays could be non-contiguous, even though the
    // is_fully_contiguous() asserts above currently require contiguity. ksgpu Array strides are in
    // multiples of dtype size, so we convert to 32-bit units per-array:
    //   - out_max has dtype 'this->dtype' (fp16 or fp32). For fp16, 32-bit alignment guarantees the
    //     beam stride is even, so the division below is exact.
    //   - out_argmax is Array<uint> (32-bit elements), so its stride is already in 32-bit units.
    xassert((out_max.strides[0] * dtype.nbits) % 32 == 0);
    uint out_max_bstride32 = uint(out_max.strides[0] * dtype.nbits / 32);
    uint out_argmax_bstride32 = uint(out_argmax.strides[0]);

    dim3 grid_dims = { uint(pow2(dd_params.amb_rank)), uint(dd_params.beams_per_batch), 1 };
    dim3 block_dims = { 32, uint(registry_value.warps_per_threadblock), 1 };

    registry_value.cuda_kernel<<< grid_dims, block_dims, registry_value.shmem_nbytes, stream >>>
        (in.data, gpu_ringbuf_quadruples.data, rb_frame0,  // void *grb_base_, uint *grb_quads_, long grb_frame0,
         out_max.data, out_argmax.data, wt.data,           // void *out_max_, uint *out_argmax, const void *wt_,
         out_max_bstride32, out_argmax_bstride32,          // uint out_max_bstride32, uint out_argmax_bstride32,
         pstate.data, dd_params.ntime,                     // void *pstate_, int ntime,
         nt_cumul, dd_params.input_is_downsampled_tree,    // ulong nt_cumul, bool input_is_downsampled_tree,
         ndm_out_per_wt, nt_in_per_wt);                    // uint ndm_out_per_wt, uint nt_in_per_wt

    CUDA_PEEK("coalesced_dd_kernel2 launch");
}


// Helper for CoalescedDdKernel2::test_random(). Reindexes the reference dedisperser's
// subband array into the layout that the kernel's peak-finding half sees:
//
//    dst[b, dm_out, (m << xdm_rank) | e, t] = src[b, (dm_out << xdm_rank) | e, m, t]
//
// i.e. the low 'xdm_rank' bits of the DM index become the low bits of the multiplet index.
// Both arrays are fully contiguous, so each (dm_out, m, e) is one memcpy along the time axis.
// (See cuda_generator/Dedisperser.emit_subband_extraction() for where the reindexing comes
// from, and cuda_generator/CoalescedDdKernel2.py for why the peak-finder is happy with it.)
static void _gather_m_ext(Array<float> &dst, const Array<float> &src, long xdm_rank)
{
    xassert_ge(xdm_rank, 1);
    xassert_eq(src.ndim, 4);

    long nbeams = src.shape[0];
    long E = pow2(xdm_rank);
    long ndm_out = xdiv(src.shape[1], E);
    long M = src.shape[2];
    long nt = src.shape[3];

    xassert_shape_eq(dst, ({ nbeams, ndm_out, M*E, nt }));
    xassert(src.is_fully_contiguous());
    xassert(dst.is_fully_contiguous());

    for (long b = 0; b < nbeams; b++) {
        for (long d = 0; d < ndm_out; d++) {
            for (long m = 0; m < M; m++) {
                for (long e = 0; e < E; e++) {
                    const float *p = src.data + ((((b*ndm_out + d)*E + e)*M) + m) * nt;
                    float *q = dst.data + ((((b*ndm_out + d)*M + m)*E) + e) * nt;
                    memcpy(q, p, nt * sizeof(float));
                }
            }
        }
    }
}


// Static member function: runs one randomized test iteration.
void CoalescedDdKernel2::test_random()
{
    RegistryKey key = registry().get_random_key();
    Dtype dtype = key.dtype;

    long simd_width = xdiv(32, key.dtype.nbits);
    long pf_rank = key.subband_counts.size() - 1;
    long dd_rank = key.dd_rank;
    long Tinner = key.Tinner;

    // The kernel's second-stage rank, and the number of "extra DM" bits it emits per warp.
    // (See cuda_generator/Dedisperser.emit_subband_extraction().)
    long rank1 = dd_rank - (dd_rank / 2);
    long xdm_rank = rank1 - pf_rank;

    long nt_in_per_wt = (Tinner > 1) ? xdiv(32*simd_width,Tinner) : ((32 * simd_width) << rand_int(0,3));
    long nt_in_divisor = max(32*simd_width, nt_in_per_wt);

    auto v = ksgpu::random_integers_with_bounded_product(5, 30000 / pow2(dd_rank));
    long nchunks = v[0];
    long nt_in_per_chunk = nt_in_divisor * v[1];
    long beams_per_batch = v[2];
    long num_batches = v[3];
    long total_beams = beams_per_batch * num_batches;
    long amb_rank = min(8L, long(log2(v[4] + 0.5)));
    long lg_ndm_out = amb_rank + dd_rank - rank1;   // one output DM per warp
    long lg_ndm_wt = rand_int(0, lg_ndm_out+1);
    bool is_downsampled_tree = rand_bool();

    // Uncomment one or more lines below, to make the test instance smaller.
    // nchunks = 1;
    // nt_in_per_chunk = max(32*simd_width, nt_in_per_wt);
    // nt_in_per_wt = (Tinner > 1) ? xdiv(32*simd_width,Tinner) : nt_in_per_chunk;
    // is_downsampled_tree = false;
    // lg_ndm_wt = 0;
    // beams_per_batch = 1;
    // num_batches = 1;
    // amb_rank = 0;
    //
    // *** YOU MUST ALSO UNCOMMENT THE NEXT TWO LINES ***
    // total_beams = beams_per_batch * num_batches;
    // lg_ndm_out = amb_rank + dd_rank - rank1;

    DedispersionKernelParams dd_params;
    dd_params.dtype = dtype;
    dd_params.dd_rank = key.dd_rank;
    dd_params.amb_rank = amb_rank;
    dd_params.beams_per_batch = beams_per_batch;
    dd_params.total_beams = total_beams;
    dd_params.ntime = nt_in_per_chunk;
    dd_params.nspec = 1;
    dd_params.input_is_ringbuf = true;
    dd_params.output_is_ringbuf = false;
    dd_params.apply_input_residual_lags = true;
    dd_params.input_is_downsampled_tree = is_downsampled_tree;
    dd_params.nt_per_segment = xdiv(1024, dtype.nbits);

    long nquads = pow2(key.dd_rank + amb_rank) * xdiv(nt_in_per_chunk, dd_params.nt_per_segment);
    dd_params.mega_ringbuf = MegaRingbuf::make_random_simplified(total_beams, beams_per_batch, nchunks, nquads);
    dd_params.consumer_id = 0;
    
    PeakFindingKernelParams pf_params;
    pf_params.subband_counts = key.subband_counts;
    pf_params.dtype = dtype;
    pf_params.max_kernel_width = key.Wmax;
    pf_params.beams_per_batch = beams_per_batch;
    pf_params.total_beams = total_beams;
    pf_params.ndm_out = pow2(lg_ndm_out);
    pf_params.ndm_wt = pow2(lg_ndm_wt);
    pf_params.nt_out = xdiv(nt_in_per_chunk, key.Dout);
    pf_params.nt_in = nt_in_per_chunk;
    pf_params.nt_wt = xdiv(nt_in_per_chunk, nt_in_per_wt);

    // Dcore is a property of the compiled GPU kernel (registry value, not part of the
    // key). Metadata-only peek: init_kernel=false skips GPU/kernel initialization.
    pf_params.Dcore = registry().get(_make_registry_key(dd_params, pf_params),
                                     /*init_kernel=*/ false).Dcore;

    CoalescedDdKernel2 cdd2_kernel(dd_params, pf_params);
    BumpAllocator allocator(af_gpu | af_zero, -1);  // dummy allocator
    cdd2_kernel.allocate(allocator);

    ReferenceDedispersionKernel ref_dd_kernel(dd_params, pf_params.subband_counts);

    // The reference peak-finder is a peak-finder-layer object, so it gets the same subband
    // counts as the GPU kernel's peak-finding half: 'xdm_rank' zeros prepended to the compact
    // counts. That FrequencySubbands has the same N and the same per-subband multiplet runs,
    // but 2^xdm_rank times as many multiplets, indexed by m_ext = (m << xdm_rank) | e. (It is
    // NOT the same band set -- see cuda_generator/CoalescedDdKernel2.py -- but the peak-finder
    // never uses the band geometry.) The two kernels are otherwise identically parameterized.
    PeakFindingKernelParams ref_pf_params = pf_params;
    ref_pf_params.subband_counts.insert(ref_pf_params.subband_counts.begin(), xdm_rank, 0);
    ReferencePeakFindingKernel ref_pf_kernel(ref_pf_params);

    FrequencySubbands &fs = cdd2_kernel.fs;             // compact
    const FrequencySubbands &fs_ext = ref_pf_kernel.fs; // extended (multiplet index is m_ext)
    GpuPfWeightLayout &wl = cdd2_kernel.pf_weight_layout;
    xassert(fs.pf_rank == pf_rank);
    xassert_eq(fs_ext.M, pow2(xdm_rank) * fs.M);
    xassert_eq(fs_ext.N, fs.N);

    // Print this monstrosity.
    cout << "CoalescedDdKernel2::test()\n"
         << "    dtype = " << dtype.str() << "\n"
         << "    dd_rank = " << dd_params.dd_rank << "\n"
         << "    amb_rank = " << dd_params.amb_rank << "\n"
         << "    pf_rank = " << pf_rank << "\n"
         << "    xdm_rank = " << xdm_rank << "\n"
         << "    is_downsampled_tree = " << is_downsampled_tree << "\n"
         << "    subbands = " << ksgpu::tuple_str(key.subband_counts) << "\n"
         << "    Wmax = " << key.Wmax << "\n"
         << "    Dcore = " << cdd2_kernel.Dcore << "\n"
         << "    Dout = " << key.Dout << "\n"
         << "    Tinner = " << key.Tinner << "\n"
         << "    M = " << fs.M << "\n"
         << "    M_ext = " << fs_ext.M << "\n"
         << "    N = " << fs.N << "\n"
         << "    num_profiles = " << ref_pf_kernel.nprofiles << "\n"
         << "    beams_per_batch = " << beams_per_batch << "\n"
         << "    total_beams = " << total_beams << "\n"
         << "    ndm_out = " << pf_params.ndm_out << "\n"
         << "    ndm_wt = "  << pf_params.ndm_wt << "\n"
         << "    nt_in_per_chunk = " << nt_in_per_chunk << "\n"
         << "    nt_out_per_chunk = " << pf_params.nt_out << "\n"
         << "    nt_wt_per_chunk = " << pf_params.nt_wt << "\n"
         << "    nchunks = " << nchunks << "\n" 
         << endl;

    long rb_nseg = dd_params.mega_ringbuf->gpu_global_nseg;
    long rb_nelts = rb_nseg * dd_params.nt_per_segment;
    Array<float> in_cpu({rb_nelts}, af_rhost);

    // Fill input ring buffer with fixed random data.
    // Some data may be "replayed" across multiple time chunks, but that's okay.
    {
        std::mt19937 &rng = ksgpu::default_rng();
        for (long i = 0; i < rb_nelts; i++)
            in_cpu.data[i] = rand_uniform(-1.0f, 1.0f, rng);
    }

    // Copy to GPU (converting dtype if necessary)
    Array<void> in_gpu = in_cpu.to_gpu(dtype);

    // Set up tmp/output buffers
    long B = dd_params.beams_per_batch;
    long A = pow2(dd_params.amb_rank);
    long T = nt_in_per_chunk;
    long D = pow2(dd_params.dd_rank);
    long M = fs.M;
    long ndm_out = pow2(lg_ndm_out);      // DM axis of out_max/out_argmax (one DM per warp)
    long Dpf = ndm_out << xdm_rank;       // DM axis of the reference dedisperser's 'sb_out'
    long Tout = pf_params.nt_out;

    Array<float> dd_cpu({B,A,D,T}, af_uhost);     // 'dd_out' for ref_dd_kernel
    Array<float> sb_cpu({B,Dpf,M,T}, af_uhost);   // 'sb_out' from ref_dd_kernel
    xassert(Dpf == ref_dd_kernel.Dpf);

    // Input to ref_pf_kernel: the same data as 'sb_cpu', reindexed by _gather_m_ext().
    // The two arrays are the same object when xdm_rank == 0, and the gather is skipped.
    Array<float> pf_in_cpu = (xdm_rank > 0)
        ? Array<float> ({B, ndm_out, fs_ext.M, T}, af_uhost)
        : sb_cpu;

    Array<float> max_cpu({B,ndm_out,Tout}, af_uhost | af_zero);
    Array<uint> argmax_cpu({B,ndm_out,Tout}, af_uhost | af_zero);

    Array<void> max_gpu(dtype, {B,ndm_out,Tout}, af_gpu | af_zero);
    Array<uint> argmax_gpu({B,ndm_out,Tout}, af_gpu | af_zero);

    Array<float> wt_cpu({B, pf_params.ndm_wt, pf_params.nt_wt, ref_pf_kernel.nprofiles, fs.N}, af_rhost | af_zero);
    
    // Tmp buffer for comparing "argmax" arrays, see below.
    Array<float> max_x({B,ndm_out,Tout}, af_uhost | af_zero);

    for (long ichunk = 0; ichunk < nchunks; ichunk++) {
        for (long ibatch = 0; ibatch < num_batches; ibatch++) {                
            ref_dd_kernel.apply(in_cpu, dd_cpu, sb_cpu, ichunk, ibatch);
            pf_params.fill_host_weights(wt_cpu, Array<double>(), /*randomize=*/true);

            if (xdm_rank > 0)
                _gather_m_ext(pf_in_cpu, sb_cpu, xdm_rank);

            Array<double> var_cpu;   // empty -> out_var feature disabled
            ref_pf_kernel.apply(max_cpu, argmax_cpu, var_cpu, pf_in_cpu, wt_cpu, ibatch);

            // CPU kernel done! Now run the GPU kernel.
            Array<void> wt_gpu = wl.to_gpu(wt_cpu);
            cdd2_kernel.launch(max_gpu, argmax_gpu, in_gpu, wt_gpu, ichunk, ibatch, NULL);

            // The "max" arrays can be compared straightforwardly.
            assert_arrays_equal(max_cpu, max_gpu, "max_cpu", "max_gpu", {"b","d","tout"});

            // For the "argmax" arrays, we have to do something weird.
            // On the CPU, evaluate triggers at the "argmax_gpu" values.
            Array<uint> argmax_x = argmax_gpu.to_host();
            ref_pf_kernel.eval_tokens(max_x, argmax_x, wt_cpu);

            // Then compare to "max_cpu", possibly at reduced precision.
            double eps = 4.0 * dtype.precision() * sqrt(dd_params.dd_rank + 1);
            assert_arrays_equal(max_cpu, max_x, "max_cpu", "max_x", {"b","d","tout"}, eps);
        }
    }
}


// Static member function
void CoalescedDdKernel2::time_one(const vector<long> &subband_counts, const string &name)
{
    for (Dtype dtype : {Dtype::native<float>(), Dtype::native<__half>()}) {
        DedispersionConfig config = DedispersionConfig::make_mini_chord(dtype);
        config.frequency_subband_counts = subband_counts;
        config.validate();

        // Check that this build actually contains the cdd2 kernels for 'config', and skip
        // with a message if not. (A cdd2_kernel_required=false plan does not query the registry,
        // so it can be constructed in any build -- unlike the real plan below, which
        // throws on a missing kernel.)
        if (!_all_kernels_registered(dtype, DedispersionPlan(config, /*cdd2_kernel_required=*/ false))) {
            cout << "\nCoalescedDdKernel2::time(): " << name << ", " << dtype.str()
                 << ": no cdd2 kernel in this build, skipping"
                 << " (see autogenerated_cdd2_kernels() in makefile_helper.py)" << endl;
            continue;
        }

        shared_ptr<DedispersionPlan> plan = make_shared<DedispersionPlan> (config);

        const DedispersionKernelParams &dd_params = plan->stage2_dd_kernel_params.at(0);
        const PeakFindingKernelParams &pf_params = plan->stage2_pf_params.at(0);
        shared_ptr<CoalescedDdKernel2> cdd2_kernel = make_shared<CoalescedDdKernel2> (dd_params, pf_params);

        BumpAllocator allocator(af_gpu | af_zero, -1);  // dummy allocator
        cdd2_kernel->allocate(allocator);

        long nbatches = xdiv(config.beams_per_gpu, config.beams_per_batch);
        long nstreams = config.num_active_batches;
        long niterations = 100;

        long S = nstreams;
        long B = config.beams_per_batch;
        long ndm_out = pf_params.ndm_out;
        long nt_out = pf_params.nt_out;
        long rb_nelts = plan->mega_ringbuf->gpu_global_nseg * plan->nelts_per_segment;

        vector<long> wt_shape = cdd2_kernel->pf_weight_layout.get_shape(B, pf_params.ndm_wt, pf_params.nt_wt);
        vector<long> wt_strides = cdd2_kernel->pf_weight_layout.get_strides(B, pf_params.ndm_wt, pf_params.nt_wt);

        // Prepend number of streams to 'wt_shape' and 'wt_stride'.
        wt_shape.insert(wt_shape.begin(), S);
        wt_strides.insert(wt_strides.begin(), wt_shape[1] * wt_strides[0]);

        // All streams share ONE input ringbuf. It is read-only in the kernel, and in
        // production there is a single MegaRingbuf shared by all streams -- so the
        // per-stream copies this used to allocate were both unnecessary and unrealistic.
        // (They also made the allocation num_active_batches times too large: ~27 GiB per
        // copy, see make_mini_chord().) The other arrays below ARE per-stream, since
        // concurrent launches write to them.
        Array<void> in_gpu(dtype, {rb_nelts}, af_gpu | af_zero);
        Array<void> wt_gpu(dtype, wt_shape, wt_strides, af_gpu | af_zero);
        Array<void> max_gpu(dtype, {S,B,ndm_out,nt_out}, af_gpu | af_zero);
        Array<uint> argmax_gpu({S,B,ndm_out,nt_out}, af_gpu | af_zero);

        cout << "\nCoalescedDdKernel2::time(): " << name << ", " << dtype.str() << endl;
        KernelTimer kt(niterations, nstreams);

        while (kt.next()) {
            Array<void> max_gpu_slice = max_gpu.slice(0, kt.istream);
            Array<uint> argmax_gpu_slice = argmax_gpu.slice(0, kt.istream);
            Array<void> wt_gpu_slice = wt_gpu.slice(0, kt.istream);
            long ichunk = kt.curr_iteration / nbatches;
            long ibatch = kt.curr_iteration % nbatches;

            cdd2_kernel->launch(
                max_gpu_slice,
                argmax_gpu_slice,
                in_gpu,
                wt_gpu_slice,
                ichunk,
                ibatch,
                kt.stream);

            if (kt.warmed_up && ((kt.curr_iteration % 10) == 9))
                cout << (1.0e-9 * cdd2_kernel->resource_tracker.get_gmem_bw() / kt.dt) << " GB/s\n";
        }
    }
}

// Static member function
void CoalescedDdKernel2::time_selected()
{
    // From makefile_helper.py
    time_one({0,0,0,0,1}, "no subbands");
    time_one({0,0,1,2,1}, "chime-thresh0.3");
    time_one({0,5,7,3,1}, "chime-thresh0.1");
    time_one({1,2,2,2,1}, "chord-thresh0.4");
    time_one({5,9,7,3,1}, "chord-thresh0.1");
}


// -------------------------------------------------------------------------------------------------
//
// Registry and related functions


struct Cdd2Registry : public CoalescedDdKernel2::Registry
{
    using Key = CoalescedDdKernel2::RegistryKey;
    using Val = CoalescedDdKernel2::RegistryValue;

    virtual void add(const Key &key, const Val &val, bool debug) override
    {
        // Just check that all members have been initialized.
        // (In the future, I may add more argument checking here.)
        
        xassert((key.dtype == Dtype::native<float>()) || (key.dtype == Dtype::native<__half>()));
        xassert_ge(key.subband_counts.size(), 1);
        xassert(key.dd_rank > 0);
        xassert(key.Tinner > 0);
        xassert(key.Dout > 0);
        xassert(key.Wmax > 0);

        xassert(val.cuda_kernel != nullptr);
        xassert(val.warps_per_threadblock > 0);
        xassert(val.pstate32_per_small_tree >= 0);
        xassert(val.nt_per_segment > 0);
        xassert(val.Dcore > 0);
        
        val.pf_weight_layout.validate();
        
        // Call add() in base class.
        CoalescedDdKernel2::Registry::add(key, val, debug);
    }


    // Setting shared memory size is "deferred" from when the kernel is registered, to when
    // the kernel is first used. Deferring is important, since cudaFuncSetAttribute() creates
    // hard-to-debug problems if called at library initialization time, but behaves normally
    // if deferred. (Here, "hard-to-debug" means that the call appears to succeed, but an
    // unrelated kernel launch will fail later with error 400 ("invalid resource handle").)

    virtual void deferred_initialization(Val &val) override
    {
        if (val.shmem_nbytes > constants::cuda_max_static_shmem_bytes) {
            CUDA_CALL(cudaFuncSetAttribute(
                val.cuda_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                val.shmem_nbytes
            ));
        }
    }
};


CoalescedDdKernel2::Registry &CoalescedDdKernel2::registry()
{
    // This kludge implements "construct on first use". It's necessary because the
    // registry is accessed at library initialization time (when kernel .cu files
    // call CoalescedDdKernel2::registry().add() to register themselves).
    //
    // Using a static variable in this way (instead of a global variable) ensures
    // that the registry is constructed before CoalescedDdKernel2::registry().add()
    // is called.
    //
    // This kludge is necessary because the registry is accessed at library initialization
    // time, by callers in other source files, and source files are executed in an
    // arbitrary order.
    
    static Cdd2Registry reg;
    return reg;  // note: thread-safe (as of c++11)
}

bool operator==(const CoalescedDdKernel2::RegistryKey &k1, const CoalescedDdKernel2::RegistryKey &k2)
{
    return (k1.dtype == k2.dtype)
        && (k1.dd_rank == k2.dd_rank)
        && (k1.subband_counts == k2.subband_counts)
        && (k1.Tinner == k2.Tinner)
        && (k1.Dout == k2.Dout)
        && (k1.Wmax == k2.Wmax);
}

ostream &operator<<(ostream &os, const CoalescedDdKernel2::RegistryKey &k)
{
    FrequencySubbands fs(k.subband_counts);
    os << "CoalescedDdKernel2(dtype=" << k.dtype.str()
       << ", dd_rank=" << k.dd_rank
       << ", subbands=" << tuple_str(k.subband_counts)
       << ", Tinner=" << k.Tinner
       << ", Dout=" << k.Dout
       << ", Wmax=" << k.Wmax
       << ", N=" << fs.N
       << ", M=" << fs.M
       << ")";
    return os;
}

ostream &operator<<(ostream &os, const CoalescedDdKernel2::RegistryValue &v)
{
    os << "(Dcore=" << v.Dcore
       << ", shmem=" << v.shmem_nbytes
       << ", warps=" << v.warps_per_threadblock
       << ", pstate32=" << v.pstate32_per_small_tree
       << ", nt_seg=" << v.nt_per_segment
       << ")";
    return os;
}


}  // namespace pirate
