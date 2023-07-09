#ifndef _PIRATE_INTERNALS_CPU_DOWNSAMPLE_HPP
#define _PIRATE_INTERNALS_CPU_DOWNSAMPLE_HPP

#include <cstdint>

namespace pirate {
#if 0
}  // editor auto-indent
#endif


extern void cpu_downsample(int src_bit_depth, const uint8_t *src, uint8_t *dst, ssize_t src_nbytes, ssize_t dst_nbytes);

// 'src_nbytes' must be a multiple of this.
extern ssize_t cpu_downsample_src_bytes_per_chunk(int src_bit_depth);


} // namespace pirate

#endif // _PIRATE_INTERNALS_CPU_DOWNSAMPLE_HPP
