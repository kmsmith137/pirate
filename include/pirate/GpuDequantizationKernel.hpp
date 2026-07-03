#ifndef _PIRATE_GPU_DEQUANTIZATION_KERNEL_HPP
#define _PIRATE_GPU_DEQUANTIZATION_KERNEL_HPP

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <ksgpu/Array.hpp>
#include <ksgpu/Dtype.hpp>
#include "ResourceTracker.hpp"

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// Dequantization: convert an int4 array to float32 or float16, applying a
// per-(beam, freq, minichunk) affine transform during conversion. Implemented by
// two kernels sharing this operation: GpuDequantizationKernel (GPU) and
// ReferenceDequantizationKernel (CPU reference, always float32).
//
// Inputs:
//   scales_offsets: shape (nbeams, nfreq, ntime/256, 2), dtype float16
//   data:           shape (nbeams, nfreq, ntime),       dtype int4
//
// Output:
//   out:            shape (nbeams, nfreq, ntime),       dtype float32 or float16
//
// Output formula:
//
//                 { 0                                                 if data[b,f,t] == -8
//   out[b,f,t] = {
//                 { scales_offsets[b,f,t/256,0] * data[b,f,t]
//                 +    scales_offsets[b,f,t/256,1]                    otherwise
//
// The int4 'data' values are interpreted as signed two's complement (-8 to +7).
// data == -8 (bit pattern 0b1000) is the "missing sample" sentinel and is
// always mapped to 0 in the output, regardless of scale and offset; this
// matches the convention used by AssembledFrame::data.
//
// Nibble packing in 'data': low nibble = even index, high nibble = odd index.
// The last axis of 'scales_offsets' is {scale, offset}; one (scale, offset)
// pair is shared by 256 consecutive time samples of a single (beam, freq).


// Reinterpret a contiguous uint8 array of shape (nbeams, nfreq, ntime/2) as an
// int4 array of shape (nbeams, nfreq, ntime). Helper for the pybind11 wrappers of
// both dequantization kernels: numpy/cupy have no int4 dtype (dtypes must be at
// least 8 bits), so Python passes the packed data as uint8.
extern ksgpu::Array<void> dequantization_uint8_to_int4(const ksgpu::Array<void> &in_uint8,
                                                       long nbeams, long nfreq, long ntime);


struct GpuDequantizationKernel
{
    // Constructor args: (dtype, nbeams, nfreq, ntime)
    // dtype: output dtype (must be float32 or float16)
    // Throws exception if dtype is invalid, or if ntime is not divisible by 256.
    GpuDequantizationKernel(ksgpu::Dtype dtype, long nbeams, long nfreq, long ntime);

    // GPU kernel launch (async, does not sync stream).
    //   out:            shape (nbeams, nfreq, ntime),       dtype matches this->dtype, contiguous, on GPU
    //   scales_offsets: shape (nbeams, nfreq, ntime/256, 2), dtype float16, contiguous, on GPU
    //   data:           shape (nbeams, nfreq, ntime),       dtype int4,    contiguous, on GPU
    // The float32 kernel converts (scale, offset) from fp16 to fp32 before any
    // arithmetic; the float16 kernel performs the affine math natively in fp16.
    void launch(ksgpu::Array<void> &out,
                const ksgpu::Array<__half> &scales_offsets,
                const ksgpu::Array<void> &data,
                cudaStream_t stream) const;

    // Static test function (called via 'python -m pirate_frb test --gdqk').
    // Compares the GPU kernel against ReferenceDequantizationKernel.
    static void test_random();

    // Static timing function
    static void time_selected();

    // Members
    ksgpu::Dtype dtype;  // output dtype (float32 or float16)
    long nbeams;
    long nfreq;
    long ntime;

    ResourceTracker resource_tracker;

    // Kernel launch config
    dim3 nblocks;
    dim3 nthreads;
};


// CPU reference implementation of the dequantization operation described above.
// Always outputs float32 (there is no output-dtype argument).
struct ReferenceDequantizationKernel
{
    // Constructor args: (nbeams, nfreq, ntime).
    // Throws exception if any is non-positive, or if ntime is not divisible by 256.
    ReferenceDequantizationKernel(long nbeams, long nfreq, long ntime);

    //   out:            shape (nbeams, nfreq, ntime),       dtype float32, contiguous, on host
    //   scales_offsets: shape (nbeams, nfreq, ntime/256, 2), dtype float16, contiguous, on host
    //   data:           shape (nbeams, nfreq, ntime),       dtype int4,    contiguous, on host
    // Each (scale, offset) pair is converted from fp16 to fp32 immediately,
    // before any arithmetic.
    void apply(ksgpu::Array<float> &out,
               const ksgpu::Array<__half> &scales_offsets,
               const ksgpu::Array<void> &data);

    // Members
    long nbeams;
    long nfreq;
    long ntime;
};


}  // namespace pirate

#endif // _PIRATE_GPU_DEQUANTIZATION_KERNEL_HPP
