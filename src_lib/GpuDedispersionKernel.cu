#include "../include/pirate/DedispersionKernel.hpp"
#include "../include/pirate/KernelRegistry.hpp"
#include "../include/pirate/constants.hpp"
#include "../include/pirate/inlines.hpp"   // pow2(), is_aligned(), simd_type
#include "../include/pirate/utils.hpp"     // bit_reverse_slow()

#include <mutex>
#include <sstream>
#include <ksgpu/xassert.hpp>
#include <ksgpu/cuda_utils.hpp>  // CUDA_CALL()
#include <ksgpu/rand_utils.hpp>  // rand_int()
#include <ksgpu/string_utils.hpp>

using namespace std;
using namespace ksgpu;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


GpuDedispersionKernel::GpuDedispersionKernel(const Params &params_) :
    params(params_)
{
    params.validate();
    xassert(params.dd_rank > 0);  // FIXME define _r0 for testing

    RegistryKey key;
    key.dtype = params.dtype;
    key.rank = params.dd_rank;
    key.input_is_ringbuf = params.input_is_ringbuf;
    key.output_is_ringbuf = params.output_is_ringbuf;
    key.apply_input_residual_lags = params.apply_input_residual_lags;
    key.nspec = params.nspec;

    this->registry_value = query_registry(key);
    this->nbatches = xdiv(params.total_beams, params.beams_per_batch);

    int ST = xdiv(params.dtype.nbits, 8);    
    this->bw_per_launch.kernel_launches = 1;
    this->bw_per_launch.nbytes_gmem += 2 * params.beams_per_batch * pow2(params.dd_rank+params.amb_rank) * params.ntime * params.nspec * ST;
    this->bw_per_launch.nbytes_gmem += 8 * params.beams_per_batch * pow2(params.amb_rank) * registry_value.pstate32_per_small_tree;
    // FIXME(?) not currently including ringbuf_locations.

    // Important: ensure that caller-specified 'nt_per_segment' matches GPU kernel.
    xassert_eq(params.nt_per_segment, registry_value.nt_per_segment);
}


void GpuDedispersionKernel::allocate()
{
    if (is_allocated)
	throw runtime_error("double call to GpuDedispersionKernel::allocate()");
    
    // Note 'af_zero' flag here.
    long ninner = registry_value.pstate32_per_small_tree * xdiv(32, params.dtype.nbits);
    std::initializer_list<long> shape = { params.total_beams, pow2(params.amb_rank), ninner };
    this->persistent_state = Array<void> (params.dtype, shape, af_zero | af_gpu);

    // Copy host -> GPU.
    if (params.input_is_ringbuf || params.output_is_ringbuf) {
	this->gpu_ringbuf_locations = params.ringbuf_locations.to_gpu();

	long nrb = pow2(params.amb_rank + params.dd_rank) * xdiv(params.ntime, params.nt_per_segment);
	xassert_shape_eq(gpu_ringbuf_locations, ({nrb,4}));
	xassert(gpu_ringbuf_locations.is_fully_contiguous());
	xassert(gpu_ringbuf_locations.on_gpu());
    }

    this->is_allocated = true;
}


void GpuDedispersionKernel::launch(Array<void> &in_arr, Array<void> &out_arr, long ibatch, long it_chunk, cudaStream_t stream)
{
    xassert(this->is_allocated);
    xassert((ibatch >= 0) && (ibatch < nbatches));
    xassert(it_chunk >= 0);

    DedispersionKernelIobuf in(params, in_arr, params.input_is_ringbuf, true);     // on_gpu=true
    DedispersionKernelIobuf out(params, out_arr, params.output_is_ringbuf, true);  // on_gpu=true

    // The global persistent_state array has shape { total_beams, pow2(params.amb_rank), ninner }.
    // We want to select a subset of beams corresponding to the current batch.
    long b0 = (ibatch) * params.beams_per_batch;
    long b1 = (ibatch+1) * params.beams_per_batch;
    Array<void> pstate = this->persistent_state.slice(0, b0, b1);
    
    // Only used if (params.input_is_ringbuf || params.output_is_ringbuf)
    long rb_pos = (it_chunk * params.total_beams) + (ibatch * params.beams_per_batch);

    dim3 grid_dims = { uint(pow2(params.amb_rank)), uint(params.beams_per_batch), 1 };
    dim3 block_dims = { 32, uint(registry_value.warps_per_threadblock), 1 };
    ulong nt_cumul = it_chunk * params.ntime;

    if (!params.input_is_ringbuf && !params.output_is_ringbuf) {
	// Case 1: neither input nor output are ringbufs.
	auto cuda_kernel = this->registry_value.cuda_kernel_no_rb;
	xassert(cuda_kernel != nullptr);
	    
	cuda_kernel<<< grid_dims, block_dims, registry_value.shmem_nbytes, stream >>>
	    (in.buf, in.beam_stride32, in.amb_stride32, in.act_stride32,
	     out.buf, out.beam_stride32, out.amb_stride32, out.act_stride32,
	     pstate.data, params.ntime, nt_cumul, params.input_is_downsampled_tree);
    }
    else if (params.input_is_ringbuf && !params.output_is_ringbuf) {
	// Case 2: input is ringbuf.
	auto cuda_kernel = this->registry_value.cuda_kernel_in_rb;
	xassert(cuda_kernel != nullptr);
	
	cuda_kernel<<< grid_dims, block_dims, registry_value.shmem_nbytes, stream >>>
	    (in.buf, gpu_ringbuf_locations.data, rb_pos,
	     out.buf, out.beam_stride32, out.amb_stride32, out.act_stride32,
	     pstate.data, params.ntime, nt_cumul, params.input_is_downsampled_tree);
    }	
    else if (!params.input_is_ringbuf && params.output_is_ringbuf) {
	// Case 3: output is ringbuf.
	auto cuda_kernel = this->registry_value.cuda_kernel_out_rb;
	xassert(cuda_kernel != nullptr);
	    
	cuda_kernel<<< grid_dims, block_dims, registry_value.shmem_nbytes, stream >>>
	    (in.buf, in.beam_stride32, in.amb_stride32, in.act_stride32,
	     out.buf, gpu_ringbuf_locations.data, rb_pos,
	     pstate.data, params.ntime, nt_cumul, params.input_is_downsampled_tree);
    }
    else
	throw runtime_error("DedispersionKernelParams::{input,output}_is_ringbuf flags are both set");
    
    CUDA_PEEK("dedispersion kernel");
}


// -------------------------------------------------------------------------------------------------
//
// Kernel registry.


template<typename F>
inline void _set_shmem(F kernel, uint nbytes)
{
    if ((kernel != nullptr) && (nbytes > 48*1024)) {
	CUDA_CALL(cudaFuncSetAttribute(
	    kernel,
	    cudaFuncAttributeMaxDynamicSharedMemorySize,
	    nbytes
	));
    }
}


struct DedispRegistry : public KernelRegistry<GpuDedispersionKernel::RegistryKey, GpuDedispersionKernel::RegistryValue>
{
    // Setting shared memory size is "deferred" from when the kernel is registered, to when
    // the kernel is first used. Deferring is important, since cudaFuncSetAttribute() creates
    // hard-to-debug problems if called at library initialization time, but behaves normally
    // if deferred. (Here, "hard-to-debug" means that the call appears to succeed, but an
    // unrelated kernel launch will fail later with error 400 ("invalid resource handle").)

    virtual void deferred_initialization(GpuDedispersionKernel::RegistryValue &val) override
    {
	_set_shmem(val.cuda_kernel_no_rb, val.shmem_nbytes);
	_set_shmem(val.cuda_kernel_in_rb, val.shmem_nbytes);
	_set_shmem(val.cuda_kernel_out_rb, val.shmem_nbytes);
    }
};

// Instead of declaring the registry as a static global variable, we declare it
// as a static local variable in the function dd_registry(). The registry will
// be initialized the first time that dd_registry() is called.
//
// This kludge is necessary because the registry is accessed at library initialization
// time, by callers in other source files, and source files are executed in an
// arbitrary order.

static DedispRegistry &dd_registry()
{
    static DedispRegistry reg;
    return reg;  // note: thread-safe (as of c++11)
}


// Static member function.
GpuDedispersionKernel::RegistryValue GpuDedispersionKernel::query_registry(const RegistryKey &k)
{
    return dd_registry().query(k);
}

// Static member function.
GpuDedispersionKernel::RegistryKey GpuDedispersionKernel::get_random_registry_key()
{
    return dd_registry().get_random_key();
}


// Static member function for adding to the registry.
// Called during library initialization, from source files with gpu kernels.
void GpuDedispersionKernel::register_kernel(const RegistryKey &key, const RegistryValue &val, bool debug)
{
    // Just check that all members have been initialized.
    // (In the future, I may add more argument checking here.)
    
    xassert((key.dtype == Dtype::native<float>()) || (key.dtype == Dtype::native<__half>()));
    xassert(key.nspec > 0);
    
    xassert(val.warps_per_threadblock > 0);
    xassert(val.nt_per_segment > 0);

    auto k1 = val.cuda_kernel_no_rb;
    auto k2 = val.cuda_kernel_in_rb;
    auto k3 = val.cuda_kernel_out_rb;

    if (!key.input_is_ringbuf && !key.output_is_ringbuf)
	xassert(k1 && !k2 && !k3);
    else if (key.input_is_ringbuf && !key.output_is_ringbuf)
	xassert(!k1 && k2 && !k3);
    else if (!key.input_is_ringbuf && key.output_is_ringbuf)
	xassert(!k1 && !k2 && k3);
    else
	throw runtime_error("DedispersionKernelParams::{input,output}_is_ringbuf flags are both set");
    
    return dd_registry().add(key, val, debug);
}


bool operator==(const GpuDedispersionKernel::RegistryKey &k1, const GpuDedispersionKernel::RegistryKey &k2)
{
    return (k1.dtype == k2.dtype) &&
	(k1.rank == k2.rank) &&
	(k1.nspec == k2.nspec) &&
	(k1.input_is_ringbuf == k2.input_is_ringbuf) &&
	(k1.output_is_ringbuf == k2.output_is_ringbuf) &&
	(k1.apply_input_residual_lags == k2.apply_input_residual_lags);
}


ostream &operator<<(ostream &os, const GpuDedispersionKernel::RegistryKey &k)
{
    os << "GpuDedispersionKernel(dtype=" << k.dtype
       << ", rank=" << k.rank
       << ", nspec=" << k.nspec
       << ", input_is_ringbuf=" << k.input_is_ringbuf
       << ", output_is_ringbuf=" << k.output_is_ringbuf
       << ", apply_input_residual_lags=" << k.apply_input_residual_lags
       << ")";

    return os;
}


}  // namespace pirate
