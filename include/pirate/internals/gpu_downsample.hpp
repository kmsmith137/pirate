#ifndef _PIRATE_INTERNALS_GPU_DOWNSAMPLE_HPP
#define _PIRATE_INTERNALS_GPU_DOWNSAMPLE_HPP

#include <ksgpu/Array.hpp>

namespace pirate {
#if 0
}  // editor auto-indent
#endif


extern void launch_downsample(ksgpu::Array<float> &dst_i, ksgpu::Array<float> &dst_w,
			      const ksgpu::Array<float> &src_i, const ksgpu::Array<float> &src_w,
			      int Df, int Dt, bool transpose_output, cudaStream_t stream=nullptr);


} // namespace pirate

#endif //  _PIRATE_INTERNALS_GPU_DOWNSAMPLE_HPP
