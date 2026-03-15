#ifndef _PIRATE_CHIME_BEAMFORMER_HPP
#define _PIRATE_CHIME_BEAMFORMER_HPP

#include <cuda_fp16.h>
#include <ksgpu/Array.hpp>

namespace pirate {
#if 0
}  // editor auto-indent
#endif

// CHIME beamformer is split between two source files, in order to parallelize computation.
// (Compile times are high since we use cufftdx.)
//
//  - ChimeBeamformer1.cu: chime_frb_beamform() and friends
//  - ChimeBeamformer2.cu: chime_frb_upchan() and friends


// 'inputData':  shape (T,F,2,4,256), dtype uint8_t, axes (time,freq,pol,ew,ns)
// 'map':        shape (F,256), dtype uint, axes (freq,ns)
// 'co':         shape (F,4,4,2), dtype float, axes (freq,ewout,ewin,ReIm)
// 'outputData': shape (T,F,2,4,256), dtype float16+16, axes (time,freq,pol,ew,ns)
// 'gains':      shape (F,2,4,256), dtype float32+32, axes (freq,pol,ew,ns)
extern void launch_chime_frb_beamform(
    const uint8_t *inputData, const uint *map, const float *co,
    __half *outputData, const float *gains,
    long T, long F, cudaStream_t stream=nullptr);

extern void launch_chime_frb_beamform(
    const ksgpu::Array<uint8_t> &inputData,
    const ksgpu::Array<uint> &map,
    const ksgpu::Array<float> &co,
    ksgpu::Array<__half> &outputData,
    const ksgpu::Array<float> &gains,
    cudaStream_t stream=nullptr);

extern void cpu_chime_frb_beamform(
    const ksgpu::Array<uint8_t> &inputData,
    const ksgpu::Array<uint> &map,
    const ksgpu::Array<float> &co,
    ksgpu::Array<float> &outputData,
    const ksgpu::Array<float> &gains);

extern void test_chime_frb_beamform();
extern void time_chime_frb_beamform();


// 'data': shape=(T,F,2,B,2), axes (time,freq,pol,beam,ReIm)
// 'results_array': shape=(B,F,T/384,16), axes (beam,cfreq,time,ufreq)
extern void launch_chime_frb_upchan(const __half *data, float *results_array, long T, long F, long B, cudaStream_t stream=nullptr);
extern void launch_chime_frb_upchan(const ksgpu::Array<__half> &data, ksgpu::Array<float> &results_array, cudaStream_t stream=nullptr);
extern void cpu_chime_frb_upchan(const ksgpu::Array<float> &data, ksgpu::Array<float> &results_array);
extern void test_chime_frb_upchan();
extern void time_chime_frb_upchan();


} // namespace pirate

#endif // _PIRATE_CHIME_BEAMFORMER_HPP
