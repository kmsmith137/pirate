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


// -------------------------------------------------------------------------------------------------
//
// ChimeBeamformer1.cu: chime_frb_beamform() and friends
//
// For a precise description of what these functions compute, see comments in ChimeBeamformer1.cu.


// 'inputData':  shape (T,F,2,4,256), dtype uint8_t, axes (time,freq,pol,ew,ns)
// 'map':        shape (F,256), dtype uint, axes (freq,ns)
// 'co':         shape (F,4,4,2), dtype float, axes (freq,ewout,ewin,ReIm)
// 'outputData': shape (T,F,2,4,256), dtype float16+16, axes (time,freq,pol,ew,ns)
// 'gains':      shape (F,2,4,256), dtype float32+32, axes (freq,pol,ew,ns)
extern void launch_chime_frb_beamform(
    const uint8_t *inputData, const uint *map, const float *co,
    __half *outputData, const float *gains,
    long T, long F, cudaStream_t stream);

extern void launch_chime_frb_beamform(
    const ksgpu::Array<uint8_t> &inputData,
    const ksgpu::Array<uint> &map,
    const ksgpu::Array<float> &co,
    ksgpu::Array<__half> &outputData,
    const ksgpu::Array<float> &gains,
    cudaStream_t stream);

extern void cpu_chime_frb_beamform(
    const ksgpu::Array<uint8_t> &inputData,
    const ksgpu::Array<uint> &map,
    const ksgpu::Array<float> &co,
    ksgpu::Array<float> &outputData,
    const ksgpu::Array<float> &gains);

extern void test_chime_frb_beamform();
extern void time_chime_frb_beamform();


// Helper function to initialize the 'map' argument to chime_beamform (on the host, so
// you'll need to copy it to the GPU). Logic is cut-and-paste from kotekan.
// Note: this is for a single frequency, see calculate_cl_indices() below for multifrequency.
// 
//   host_map:          output array, must have room for 256 uint32 values, each in [0,511].
//   freq_now:          observing frequency [MHz]
//   northmost_beam:    zenith angle of the northernmost beam [degrees]. Production CHIME uses 60.0.

extern void calculate_cl_index(uint *host_map, double freq_now, double northmost_beam);


// Multifrequency version of calculate_cl_index()
//   freqs:             length-F 1-d array containing observing frequencies [MHz]
//   northmost_beam:    zenith angle of the northernmost beam [degrees]. Production CHIME uses 60.0.
//
// Returns a shape (F,256) array, which can be copied to the GPU and used as the 'map' argument
// to launch_chime_frb_beamform().

extern ksgpu::Array<uint> calculate_cl_indices(const ksgpu::Array<double> &freqs, double northmost_beam);


// -------------------------------------------------------------------------------------------------
//
// ChimeBeamformer2.cu: chime_frb_upchan() and friends
//
// For a precise description of what these functions compute, see comments in ChimeBeamformer2.cu.


// 'data': shape=(T,F,2,B,2), axes (time,freq,pol,beam,ReIm)
// 'results_array': shape=(B,F,T/384,16), axes (beam,cfreq,time,ufreq)
extern void launch_chime_frb_upchan(const __half *data, float *results_array, long T, long F, long B, cudaStream_t stream=nullptr);
extern void launch_chime_frb_upchan(const ksgpu::Array<__half> &data, ksgpu::Array<float> &results_array, cudaStream_t stream=nullptr);

extern void cpu_chime_frb_upchan(const ksgpu::Array<float> &data, ksgpu::Array<float> &results_array);
extern void test_chime_frb_upchan();
extern void time_chime_frb_upchan();


} // namespace pirate

#endif // _PIRATE_CHIME_BEAMFORMER_HPP
