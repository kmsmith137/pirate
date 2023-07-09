#ifndef _PIRATE_INTERNALS_GPU_DOWNSAMPLE_HPP
#define _PIRATE_INTERNALS_GPU_DOWNSAMPLE_HPP

#include <gputils/Array.hpp>

namespace pirate {
#if 0
}  // editor auto-indent
#endif


extern void launch_downsample(gputils::Array<float> &dst_i, gputils::Array<float> &dst_w,
			      const gputils::Array<float> &src_i, const gputils::Array<float> &src_w,
			      int Df, int Dt, bool transpose_output, cudaStream_t stream=nullptr);


} // namespace pirate

#endif //  _PIRATE_INTERNALS_GPU_DOWNSAMPLE_HPP
