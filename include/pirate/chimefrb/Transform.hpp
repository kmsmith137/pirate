#ifndef _PIRATE_CHIMEFRB_TRANSFORM_HPP
#define _PIRATE_CHIMEFRB_TRANSFORM_HPP

#include <string>
#include <initializer_list>
#include <cuda_runtime.h>
#include <ksgpu/Array.hpp>

#include "../constants.hpp"   // bytes_per_gpu_cache_line
#include "../inlines.hpp"     // align_up()

namespace pirate {
namespace chimefrb {
#if 0
}}  // editor auto-indent
#endif


// A transform that needs per-launch workspace carves it out of the caller's 'scratch'
// array, one sub-array after another (see ScratchLayout below). Each sub-array starts on a
// 128-byte boundary -- constants::bytes_per_gpu_cache_line, the alignment the GPU wants for
// a coalesced load, and the one BumpAllocator and SlabAllocator hand out -- so a sub-array
// of 'nelts' float32 elements OCCUPIES this many. ScratchLayout::carve() is the only caller
// in C++; the twin of this function on the python side, for containers written in python,
// is padded_scratch_nelts() in pirate_frb/chimefrb/utils.py.
inline long padded_scratch_nelts(long nelts)
{
    return align_up(nelts, constants::bytes_per_gpu_cache_line / long(sizeof(float)));
}


// ScratchLayout: the cursor a transform uses to lay out its per-launch workspace inside the
// caller's 'scratch' array. It runs in two modes, so that a transform can describe its
// layout ONCE and have both the size and the sub-arrays come out of that one description:
//
//   ScratchLayout lay;                  // SIZING: carve() returns an empty Array and only
//   my_layout(lay);                     //   advances nelts(). Use it in the constructor,
//   scratch_nelts = lay.nelts();        //   where there is no array yet.
//
//   ScratchLayout lay(scratch);         // CARVING: carve() returns views into 'scratch'.
//   Scratch s = my_layout(lay);         //   Use it in launch_checked().
//
// Writing the layout twice -- once to add up a size, once to carve -- is what this replaces;
// the two drifting apart was a standing hazard, and for a CONDITIONAL layout (GpuClipperBase,
// whose pieces depend on axis and on (Df,Dt)) it meant writing the same conditionals twice in
// two different forms.
//
// The layout function must be a function of the transform's const members only, or the two
// passes can disagree; launch_checked() should assert nelts() == scratch_nelts to pin that.

class ScratchLayout
{
public:
    ScratchLayout() = default;                                  // sizing mode
    explicit ScratchLayout(ksgpu::Array<float> &scratch);        // carving mode

    // The next sub-array of the given shape, starting on a 128-byte boundary. In sizing mode
    // the result is an EMPTY Array -- assign it, do not dereference it.
    ksgpu::Array<float> carve(std::initializer_list<long> shape);

    // Float32 elements consumed so far, padding included. After the layout function has run
    // in sizing mode, this is the transform's scratch_nelts.
    long nelts() const { return _pos; }

private:
    ksgpu::Array<float> _scratch;    // empty in sizing mode
    bool _carving = false;
    long _pos = 0;
};


// GpuTransform: the base class of every chimefrb transform -- anything that a
// Pipeline or RfiMaskPipeline (pirate_frb.chimefrb) can run. A transform processes one
// (nbeams, nfreq, ntime) block of intensity and weights, in place, on the GPU.
//
// The base class owns the part of the interface that is the same for every transform: the
// array geometry, the scratch size, and launch(), which checks its arguments and then calls
// the virtual launch_checked(). A C++ subclass overrides launch_checked() in C++ (the five
// ported RFI transforms); a python subclass overrides it in python, through a pybind11
// trampoline in src_pybind11/pirate_pybind11_chimefrb.cpp (the two pipeline classes,
// ExamplePythonTransform, and whatever a user writes). Which of the two arrays a transform
// modifies is the transform's business, stated in its docstring.
//
// The python side of the interface is in two python files: what every transform shares
// (the stream=None / scratch=None conventions of launch(), the yaml key check, __repr__)
// is injected onto this class from pirate_frb/chimefrb/cpp_transforms.py; what a
// transform WRITTEN in python needs (its constructor, the methods it defines, the hook the
// trampoline calls) is the plain python subclass GpuPythonTransform in
// pirate_frb/chimefrb/GpuPythonTransform.py, which is the class python authors derive
// from. See notes/chimefrb.md for the porting rules the chimefrb classes follow.

struct GpuTransform
{
    // 'name' is the class name as python sees it, and prefixes every message this class
    // throws ("GpuStdDevClipper: ...", "MyTransform.launch(): ..."). A C++ subclass passes
    // its own name; the python side passes type(self).__name__.
    //
    // Throws std::runtime_error on nbeams, nfreq or ntime < 1, or scratch_nelts < 0.
    GpuTransform(const std::string &name, long nbeams, long nfreq, long ntime, long scratch_nelts);
    virtual ~GpuTransform() = default;

    const std::string name;
    const long nbeams, nfreq, ntime;   // shape of the arrays launch() takes
    // Float32 scratch elements launch() needs; may be 0. NOT const, but treat it as if it
    // were: the constructor sets it -- either from its argument, or, for a subclass that
    // lays out its own workspace, by running that layout in sizing mode (see ScratchLayout)
    // once the members it depends on exist -- and nothing changes it afterwards. A subclass
    // assigning it in its body bypasses the constructor's "scratch_nelts >= 0" check, which
    // is harmless: ScratchLayout::nelts() cannot be negative.
    long scratch_nelts;

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
    //              are garbage. The CALLER is responsible for passing a 128-byte-aligned
    //              array (this is not checked): the sub-arrays a transform carves out of it
    //              are aligned relative to its base, so a misaligned base misaligns them all.
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

};


}}  // namespace pirate::chimefrb

#endif  // _PIRATE_CHIMEFRB_TRANSFORM_HPP
