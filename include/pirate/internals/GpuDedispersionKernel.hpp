#ifndef _PIRATE_INTERNALS_GPU_DEDISPERSION_KERNEL_HPP
#define _PIRATE_INTERNALS_GPU_DEDISPERSION_KERNEL_HPP

#include "inlines.hpp"  // simd32_type
#include "dedispersion_inbufs.hpp"
#include "dedispersion_outbufs.hpp"

#include <gputils/Array.hpp>

namespace pirate {
#if 0
}  // editor auto-indent
#endif


template<typename T>
class GpuDedispersionKernel
{
public:
    // Each kernel has an rlag_type which determines whether residual lags are applied.
    // See long comment below for details.

    struct Params {
	int rank = 0;
	bool apply_input_residual_lags = false;   // see below
	bool is_downsampled_tree = false;
    };

    GpuDedispersionKernel(const Params &params);
    
    const Params params;
    
    // Kernel persistent state is an array of shape
    // { nbeams, nambient, state_nelts_per_small_tree }.
    long state_nelts_per_small_tree = 0;

    // Array interface to launch().
    //
    // The 'iobuf' amd 'rstate' arrays must have shapes
    //   int N = constants::bytes_per_gpu_cache_line / sizeof(T);
    //   iobuf.shape = { nbeams, nambient, 2^rank, ntime }
    //   rstate.shape = { nbeams, nambient, state_nelts_per_small_tree }
    //
    // This may be more array dimensions than you need -- if so, just call Array::reshape().
    //
    // The last axis of 'iobuf' (representing time) must be contiguous, and the 'rstate'
    // array must be fully contiguous. Remaining iobuf axis strides (beam, ambient, row)
    // are kernel arguments.
    //
    // The 'beam' index is always a "pure spectator" index.
    // The meaning of the 'ambient' index depends on the value of params.apply_input_residual_lags:
    //
    //   - apply_input_residual_lags == false:
    //
    //       Ambient index is a pure spectator.
    //
    //   - apply_input_residual_lags == true:
    //
    //       Ambient index represents a bit-reversed DM 0 <= d < 2^(ambient_rank).
    //       The "row" index represents a coarse frequency 0 <= f < 2^(rank).
    //       Before dedispersing the data, the following residual lag is applied:
    //
    //        int nelts_per_segment = constants::bytes_per_gpu_cache_line / sizeof(T);
    //        int lag = rb_lag(f, d, ambient_rank, rank, params.is_downsampled_tree);
    //        int residual_lag = lag % nelts_per_segment;
    
    void launch(gputils::Array<T> &iobuf,
		gputils::Array<T> &rstate,
		cudaStream_t stream = nullptr) const;
    
    // Bare pointer interface to launch().
    // All stride arguments are "T" strides, i.e. byte offset is obtained by multiplying by sizeof(T).
    
    void launch(T *iobuf, T *rstate,
		long nbeams, long beam_stride,
		long nambient, long ambient_stride,
		long row_stride, long ntime,   // number of rows is always 2^rank
		cudaStream_t stream = nullptr) const;

    void print(std::ostream &os=std::cout, int indent=0) const;

protected:
    // Used internally by GpuDedispersionKernel::launch().
    int warps_per_threadblock = 0;
    int shmem_nbytes = 0;

    // If T==float, then T32 is also 'float'.
    // If T==__half, then T32 is '__half2'.
    using T32 = typename simd32_type<T>::type;
    
    // __global__ CUDA kernels (defined in GpuDedispersionKernel.cu)
    // FIXME replace 'integer_constants' argument with constant memory.
    //
    // A few notes, relative to launch() above:
    //
    //   - The 'nbeams' and 'nambient' arguments to launch() are implicitly
    //     supplied to the kernel via (GridDims.y, GridDims.x).
    //
    //   - Instead of using 'ntime' as a kernel argument, we use
    //
    //       nt_cl = number of cache lines spanned by 'ntime' time samples
    //             = (ntime * sizeof(T)) / 128
    //
    //   - Some arguments (row_stride, nt_cl) are 32-bit in the cuda kernel,
    //     but 64-bit in launch() above. The necessary overflow checking
    //     is done in launch().

    // (inbuf, outbuf, rstate, nt_cl, integer_constants)
    void (*kernel_unlagged)(typename dedispersion_simple_inbuf<T,false>::device_args, typename dedispersion_simple_outbuf<T>::device_args, T32 *, int, uint *) = nullptr;
    void (*kernel_lagged)(typename dedispersion_simple_inbuf<T,true>::device_args, typename dedispersion_simple_outbuf<T>::device_args, T32 *, int, uint *) = nullptr;

    // FIXME only on current cuda device (at time of construction).
    // Should either add run-time check, or switch to using constant memory.
    gputils::Array<uint> integer_constants;
};


}  // namespace pirate

#endif // _PIRATE_INTERNALS_GPU_DEDISPERSION_KERNEL_HPP
