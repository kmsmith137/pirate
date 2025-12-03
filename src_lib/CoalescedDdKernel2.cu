#include "../include/pirate/CoalescedDdKernel2.hpp"
#include "../include/pirate/inlines.hpp"
#include "../include/pirate/utils.hpp"

#include <mutex>
#include <sstream>
#include <iomanip>
#include <unordered_map>
#include <ksgpu/Array.hpp>
#include <ksgpu/cuda_utils.hpp>

using namespace std;
using namespace ksgpu;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


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

    xassert_eq(pf_params.dtype, dd_params.dtype);
    xassert_eq(pf_params.beams_per_batch, dd_params.beams_per_batch);
    xassert_eq(pf_params.total_beams, dd_params.total_beams);
    xassert_eq(pf_params.nt_in, dd_params.ntime);
    xassert_eq(pf_params.ndm_out, pow2(dd_params.dd_rank + dd_params.amb_rank - fs.pf_rank));

    // The initialization logic below is mostly cut-and-paste from either the
    // PeakFindingKernel or GpuDedispersionKernel constructor.

    this->dtype = dd_params.dtype;
    this->nbatches = xdiv(dd_params.total_beams, dd_params.beams_per_batch);
    this->Dout = xdiv(pf_params.nt_in, pf_params.nt_out);
    this->nprofiles = 3 * log2(pf_params.max_kernel_width) + 1;

    this->registry_key.dtype = pf_params.dtype;
    this->registry_key.dd_rank = dd_params.dd_rank;
    this->registry_key.Dout = xdiv(pf_params.nt_in, pf_params.nt_out);
    this->registry_key.W = pf_params.max_kernel_width;
    this->registry_key.subband_counts = fs.subband_counts;

    long SW = xdiv(32, pf_params.dtype.nbits);      // simd width
    long nt_in_per_wt = xdiv(pf_params.nt_in, pf_params.nt_wt);
    this->registry_key.Tinner = (nt_in_per_wt < 32*SW) ? xdiv(32*SW, nt_in_per_wt) : 1;

    // Call static member function CoalescedDdKernel2::registry().
    // this->registry_value = registry().get(registry_key);

    // Derived parameters chosen by the kernel.
    this->pf_weight_layout = registry_value.pf_weight_layout;
    this->expected_wt_shape = pf_weight_layout.get_shape(pf_params.beams_per_batch, pf_params.ndm_wt, pf_params.nt_wt);
    this->expected_wt_strides = pf_weight_layout.get_strides(pf_params.beams_per_batch, pf_params.ndm_wt, pf_params.nt_wt);
    this->Dcore = registry_value.Dcore;
    
    // Important: ensure that caller-specified 'nt_per_segment' matches GPU kernel.
    xassert_eq(dd_params.nt_per_segment, registry_value.nt_per_segment);

    // FIXME add bandwidth tracking later.
}

void CoalescedDdKernel2::allocate()
{
    if (is_allocated)
        throw runtime_error("double call to CoalescedDdKernel2::allocate()");

    // Note 'af_zero' flag here.
    long ninner = registry_value.pstate32_per_small_tree * xdiv(32, dd_params.dtype.nbits);
    std::initializer_list<long> shape = { dd_params.total_beams, pow2(dd_params.amb_rank), ninner };
    this->persistent_state = Array<void> (dd_params.dtype, shape, af_zero | af_gpu);

    // Copy host -> GPU.
    this->gpu_ringbuf_locations = dd_params.ringbuf_locations.to_gpu();

    // Shape/stride check.
    long nrb = pow2(dd_params.amb_rank + dd_params.dd_rank) * xdiv(dd_params.ntime, dd_params.nt_per_segment);
    xassert_shape_eq(gpu_ringbuf_locations, ({nrb,4}));
    xassert(gpu_ringbuf_locations.is_fully_contiguous());
    xassert(gpu_ringbuf_locations.on_gpu());

    this->is_allocated = true;
}


void CoalescedDdKernel2::launch(
    ksgpu::Array<void> &out_max,      // shape (beams_per_batch, ndm_out, nt_out)
    ksgpu::Array<uint> &out_argmax,   // shape (beams_per_batch, ndm_out, nt_out)
    const ksgpu::Array<void> &in,     // shape (ringbuf_nseg * nt_per_segment * nspec,)
    const ksgpu::Array<void> &wt,     // from GpuPfWeightLayout::to_gpu()
    long ibatch,                      // 0 <= ibatch < nbatches
    long it_chunk,                    // time-chunk index 0, 1, ...
    cudaStream_t stream)              // NULL stream is allowed, but is not the default);
{
    xassert(this->is_allocated);
    xassert((ibatch >= 0) && (ibatch < nbatches));
    xassert(it_chunk >= 0);

    xassert(out_max.dtype == dtype);
    xassert(in.dtype == dtype);
    xassert(wt.dtype == dtype);

    // Validate 'in' array: shape (ringbuf_nseg * nt_per_segment * nspec,)
    xassert_shape_eq(in, ({ dd_params.ringbuf_nseg * dd_params.nt_per_segment * dd_params.nspec }));

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

    ulong nt_cumul = it_chunk * dd_params.ntime;
    long rb_pos = (it_chunk * dd_params.total_beams) + (ibatch * dd_params.beams_per_batch);
    long ndm_out_per_wt = xdiv(pf_params.ndm_out, pf_params.ndm_wt);
    long nt_in_per_wt = xdiv(pf_params.nt_in, pf_params.nt_wt);

    dim3 grid_dims = { uint(pow2(dd_params.amb_rank)), uint(dd_params.beams_per_batch), 1 };
    dim3 block_dims = { 32, uint(registry_value.warps_per_threadblock), 1 };

    registry_value.cuda_kernel<<< grid_dims, block_dims, registry_value.shmem_nbytes, stream >>>
        (in.data, gpu_ringbuf_locations.data, rb_pos,    // void *grb_base_, uint *grb_loc_, long grb_pos,
         out_max.data, out_argmax.data, wt.data,         // void *out_max_, uint *out_argmax, const void *wt_,
         pstate.data, dd_params.ntime,                   // void *pstate_, int ntime,
         nt_cumul, dd_params.input_is_downsampled_tree,  // ulong nt_cumul, bool input_is_downsampled_tree,
         ndm_out_per_wt, nt_in_per_wt);                  // uint ndm_out_per_wt, uint nt_in_per_wt

    CUDA_PEEK("coalesced_dd_kernel2 launch");
    throw runtime_error("CoalescedDdKernel2::launch() not implemented");
}


// Static member function: runs one randomized test iteration.
void CoalescedDdKernel2::test()
{
    cout << "CoalescedDdKernel2::test() placeholder" << endl;
}


// Static member function: run timing for representative kernels.
void CoalescedDdKernel2::time()
{
    cout << "CoalescedDdKernel2::time() placeholder" << endl;
}


// -------------------------------------------------------------------------------------------------
//
// Registry and related functions


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
    
    static CoalescedDdKernel2::Registry reg;
    return reg;  // note: thread-safe (as of c++11)
}

bool operator==(const CoalescedDdKernel2::RegistryKey &k1, const CoalescedDdKernel2::RegistryKey &k2)
{
    return (k1.dtype == k2.dtype)
        && (k1.dd_rank == k2.dd_rank)
        && (k1.subband_counts == k2.subband_counts)
        && (k1.Tinner == k2.Tinner)
        && (k1.Dout == k2.Dout)
        && (k1.W == k2.W);
}

ostream &operator<<(ostream &os, const CoalescedDdKernel2::RegistryKey &k)
{
    FrequencySubbands fs(k.subband_counts);
    os << "CoalescedDdKernel2(dtype=" << k.dtype.str()
       << ", dd_rank=" << k.dd_rank
       << ", subbands=" << tuple_str(k.subband_counts)
       << ", Tinner=" << k.Tinner
       << ", Dout=" << k.Dout
       << ", W=" << k.W
       << ", F=" << fs.F
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
