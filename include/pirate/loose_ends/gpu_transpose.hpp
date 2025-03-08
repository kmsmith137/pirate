#ifndef _PIRATE_LOOSE_ENDS_GPU_TRANSPOSE_HPP
#define _PIRATE_LOOSE_ENDS_GPU_TRANSPOSE_HPP

#include <ksgpu/Array.hpp>

namespace pirate {
#if 0
}  // editor auto-indent
#endif


extern void launch_transpose(ksgpu::Array<float> &dst, const ksgpu::Array<float> &src, cudaStream_t stream=nullptr);


} // namespace pirate

#endif //  _PIRATE_LOOSE_ENDS_GPU_TRANSPOSE_HPP
