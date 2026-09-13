#ifndef _PIRATE_CHIMEFRB_TRANSFORM_BASE_HPP
#define _PIRATE_CHIMEFRB_TRANSFORM_BASE_HPP

#include <string>
#include <initializer_list>
#include <cuda_runtime.h>
#include <ksgpu/Array.hpp>

namespace pirate {
namespace chimefrb {
#if 0
}}  // editor auto-indent
#endif


// GpuTransformBase: the base class of every chimefrb transform -- anything that a
// WiPipeline or RfiMaskPipeline (pirate_frb.chimefrb) can run. A transform processes one
// (nbeams, nfreq, ntime) block of intensity and weights, in place, on the GPU.
//
// The base class owns the part of the interface that is the same for every transform: the
// array geometry, the scratch size, and launch(), which checks its arguments and then calls
// the virtual launch_checked(). A C++ subclass overrides launch_checked() in C++ (the five
// ported RFI transforms); a python subclass overrides it in python, through a pybind11
// trampoline in src_pybind11/pirate_pybind11_chimefrb.cpp (the two pipeline classes,
// ExampleCupyTransform, and whatever a user writes). Which of the two arrays a transform
// modifies is the transform's business, stated in its docstring.
//
// The python side of the interface is in two python files: what every transform shares
// (the stream=None / scratch=None conventions of launch(), the yaml key check, __repr__)
// is injected onto this class from pirate_frb/chimefrb/GpuTransformBase.py; what a
// transform WRITTEN in python needs (its constructor, the methods it defines, the hook the
// trampoline calls) is the plain python subclass GpuPythonTransform in
// pirate_frb/chimefrb/GpuPythonTransform.py, which is the class python authors derive
// from. See notes/chimefrb.md for the porting rules the chimefrb classes follow.

struct GpuTransformBase
{
    // 'name' is the class name as python sees it, and prefixes every message this class
    // throws ("GpuStdDevClipper: ...", "MyTransform.launch(): ..."). A C++ subclass passes
    // its own name; the python side passes type(self).__name__.
    //
    // Throws std::runtime_error on nbeams, nfreq or ntime < 1, or scratch_nelts < 0.
    GpuTransformBase(const std::string &name, long nbeams, long nfreq, long ntime, long scratch_nelts);
    virtual ~GpuTransformBase() = default;

    const std::string name;
    const long nbeams, nfreq, ntime;   // shape of the arrays launch() takes
    const long scratch_nelts;          // float32 scratch elements launch() needs; may be 0

    // launch(): check the arguments, then call launch_checked(). Asynchronous on 'stream';
    // nothing synchronizes. Note: stream=NULL is allowed, but is not the default.
    //
    //   intensity  shape (nbeams, nfreq, ntime), float32, fully contiguous, in GPU memory.
    //
    //   weights    same shape and layout, and not the same array as 'intensity'. Weights
    //              are >= 0 (not checked), and a zero weight means "ignore this sample".
    //
    //   scratch    1-d, fully contiguous, in GPU memory, with at least scratch_nelts
    //              elements, aliasing neither data array. May be any array, even an empty
    //              one, when scratch_nelts == 0. Contents on entry are ignored and on exit
    //              are garbage.
    //
    //   stream     CUDA stream.
    //
    // Either or both data arrays may be modified in place, as the subclass sees fit. A
    // failed check throws std::runtime_error with a message that names the transform and
    // says what was expected and what was received.
    void launch(ksgpu::Array<float> &intensity, ksgpu::Array<float> &weights,
                ksgpu::Array<float> &scratch, cudaStream_t stream) const;

    // launch_checked(): the computation. Called by launch() after the checks above, with
    // 'scratch' cut to exactly scratch_nelts elements (an empty array when that is 0). May
    // assume everything launch() checked, and must launch asynchronously on 'stream'.
    virtual void launch_checked(ksgpu::Array<float> &intensity, ksgpu::Array<float> &weights,
                                ksgpu::Array<float> &scratch, cudaStream_t stream) const = 0;

protected:
    // carve_scratch(): the next sub-array of the given shape out of the caller's 1-d
    // scratch array, advancing 'pos'. A transform that lays its per-launch workspace out
    // inside the caller's scratch (GpuClipperBase, GpuSplineDetrender) carves the pieces in
    // a fixed order, so that its scratch_nelts is the sum of the pieces.
    static ksgpu::Array<float> carve_scratch(ksgpu::Array<float> &scratch, long &pos,
                                             std::initializer_list<long> shape);
};


}}  // namespace pirate::chimefrb

#endif  // _PIRATE_CHIMEFRB_TRANSFORM_BASE_HPP
