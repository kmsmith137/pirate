#ifndef _PIRATE_INTERNALS_GPU_TRANSPOSE_HPP
#define _PIRATE_INTERNALS_GPU_TRANSPOSE_HPP

#include <gputils/Array.hpp>

namespace pirate {
#if 0
}  // editor auto-indent
#endif


extern void launch_transpose(gputils::Array<float> &dst, const gputils::Array<float> &src, cudaStream_t stream=nullptr);


} // namespace pirate

#endif //  _PIRATE_INTERNALS_GPU_TRANSPOSE_HPP
