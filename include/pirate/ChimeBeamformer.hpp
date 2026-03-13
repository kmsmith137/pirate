#ifndef _PIRATE_CHIME_BEAMFORMER_HPP
#define _PIRATE_CHIME_BEAMFORMER_HPP

#include <cuda_fp16.h>
#include <ksgpu/Array.hpp>

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// 'data': shape=(T,F,2,1024,2), axes (time,freq,pol,beam,ReIm)
// 'results_array': shape=(1024,F,T/384,16), axes (beam,cfreq,time,ufreq)

extern void launch_chime_frb_upchan(const __half *data, float *results_array, long T, long F, cudaStream_t stream=nullptr);
extern void launch_chime_frb_upchan(const ksgpu::Array<__half> &data, ksgpu::Array<float> &results_array, cudaStream_t stream=nullptr);

extern void time_chime_frb_upchan();


} // namespace pirate

#endif // _PIRATE_CHIME_BEAMFORMER_HPP
