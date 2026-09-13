// Python bindings for the chimefrb port (pirate_frb.chimefrb subpackage). C++ classes
// are defined in include/pirate/chimefrb/*.hpp; see pirate_pybind11.cpp for the main
// module.
//
// Method injections, if any, live in pirate_frb/chimefrb/Reference<ClassName>.py:
//   - AssembledChunk: none
//   - GpuWiDownsampler: launch() converts stream=None to the current cupy stream
//   - GpuWrms: same, and lets the caller omit the scratch array
//   - GpuWeightUpsampler: launch() converts stream=None to the current cupy stream
//   - GpuTransformBase (injections in pirate_frb/chimefrb/GpuTransformBase.py, not a
//     Reference*.py): the python side shared by every transform, C++ or python --
//     launch() with stream=None and scratch=None handling, the check_yaml_keys()
//     classmethod, __repr__. The python side specific to transforms WRITTEN in python
//     (the constructor, the launch_checked() and yaml stubs, the hook the trampoline
//     below calls) is the plain python class GpuPythonTransform, not an injection.
//   - The five "transforms" -- GpuBadChannelMask, GpuIntensityClipper, GpuStdDevClipper,
//     GpuPolynomialDetrender, GpuSplineDetrender -- add to_yaml_dict() / from_yaml_dict()
//     (the yaml form) and from_json_dict() (the old rf_pipelines json form); see
//     pirate_frb/chimefrb/transform_io.py. GpuBadChannelMask's __init__ also normalizes
//     its range arguments to python floats.

#define PY_ARRAY_UNIQUE_SYMBOL PyArray_API_pirate
#define NO_IMPORT_ARRAY  // Secondary file: don't call _import_array()
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <optional>
#include <pybind11/stl.h>
#include <ksgpu/pybind11.hpp>

#include "../include/pirate/chimefrb/AssembledChunk.hpp"
#include "../include/pirate/chimefrb/BadChannelMask.hpp"
#include "../include/pirate/chimefrb/ClipperAxis.hpp"
#include "../include/pirate/chimefrb/ClipperBase.hpp"
#include "../include/pirate/chimefrb/IntensityClipper.hpp"
#include "../include/pirate/chimefrb/PolynomialDetrender.hpp"
#include "../include/pirate/chimefrb/SplineDetrender.hpp"
#include "../include/pirate/chimefrb/StdDevClipper.hpp"
#include "../include/pirate/chimefrb/TransformBase.hpp"
#include "../include/pirate/chimefrb/WeightUpsampler.hpp"
#include "../include/pirate/chimefrb/WiDownsampler.hpp"
#include "../include/pirate/chimefrb/Wrms.hpp"
#include "../include/pirate/SlabAllocator.hpp"

using namespace std;
using namespace ksgpu;
using namespace pirate::chimefrb;
namespace py = pybind11;


// type_caster<ClipperAxis>: python sees the three axis names, C++ sees the enum, with the
// conversion here -- the arrangement ksgpu uses for Dtype (python sees numpy.dtype) and for
// Array<T>. The mapping itself is axis_to_string() / axis_from_string() in ClipperAxis.hpp,
// where C++ callers (the time_selected() printouts) use it too.
//
// load() THROWS on a bad value rather than returning false. A soft failure makes pybind11
// report "incompatible function arguments" followed by every overload's signature, which is
// useless to read after a typo; ksgpu's Array caster made the same choice for the same
// reason. The isinstance check is what makes axis=1 read like axis='freqq' rather than like
// a pybind11 internal cast error.
namespace PYBIND11_NAMESPACE {
namespace detail {

template<>
struct type_caster<pirate::chimefrb::ClipperAxis>
{
    PYBIND11_TYPE_CASTER(pirate::chimefrb::ClipperAxis, const_name("str"));

    bool load(handle src, bool)
    {
        if (!isinstance<str>(src))
            throw std::runtime_error("chimefrb: expected axis to be the string 'freq', 'time'"
                                     " or 'none', got a " + std::string(str(src.get_type().attr("__name__"))));

        value = pirate::chimefrb::axis_from_string(src.cast<std::string>());
        return true;
    }

    static handle cast(pirate::chimefrb::ClipperAxis axis, return_value_policy, handle)
    {
        return str(pirate::chimefrb::axis_to_string(axis)).release();
    }
};

}}  // namespace PYBIND11_NAMESPACE::detail


namespace pirate {


// Helper for the two decode bindings: use the caller's 'out' array if there is one,
// otherwise allocate a fresh (nfreq, nt) float32 array.
static Array<float> _decode_dst(const AssembledChunk &self, optional<Array<float>> &out)
{
    if (out)
        return *out;
    return Array<float> ({self.nfreq(), self.nt()}, af_uhost);
}


// PyGpuTransformBase: the python side of GpuTransformBase (a pybind11 "trampoline"). When a
// python class subclasses GpuTransformBase, pybind11 constructs this class in its place, so
// that launch_checked() -- called by GpuTransformBase::launch() after the argument checks --
// reaches the python subclass. It hands everything to ONE python method,
// GpuPythonTransform._dispatch_launch_checked() (pirate_frb/chimefrb/GpuPythonTransform.py),
// which makes the stream current and calls the subclass's launch_checked(intensity,
// weights, scratch). Keeping the stream and scratch handling in python keeps it readable
// from python. A python class that subclasses GpuTransformBase directly, rather than
// GpuPythonTransform, has no such method, and its first launch() is refused here.
//
// The three arrays reach python as new cupy objects on the caller's memory (the ksgpu
// caster's DLPack export), so in-place writes in launch_checked() land in the caller's
// arrays. 'scratch' is passed as None when scratch_nelts == 0.
struct PyGpuTransformBase : public GpuTransformBase
{
    using GpuTransformBase::GpuTransformBase;

    void launch_checked(Array<float> &intensity, Array<float> &weights,
                        Array<float> &scratch, cudaStream_t stream) const override
    {
        py::gil_scoped_acquire gil;   // launch() is bound with the GIL released

        py::function f = py::get_override(static_cast<const GpuTransformBase *>(this),
                                          "_dispatch_launch_checked");
        if (!f)
            throw std::runtime_error(name + ": a transform written in python must derive from"
                                     " GpuPythonTransform (pirate_frb.chimefrb), not from"
                                     " GpuTransformBase directly; see GpuPythonTransform.py");

        py::object s = (scratch_nelts > 0) ? py::cast(scratch) : py::object(py::none());
        f(intensity, weights, s, reinterpret_cast<uintptr_t>(stream));
    }
};


void register_chimefrb_bindings(pybind11::module &m)
{
    py::class_<AssembledChunk, std::shared_ptr<AssembledChunk>>(m, "AssembledChunk",
        "One data file written by the old CHIME FRB search, in the format ch_frb_io calls\n"
        "\"assembled_chunk in msgpack format\".\n"
        "\n"
        "Read one with :meth:`from_msgpack`, then call :meth:`decode_intensity` to get\n"
        "physical units::\n"
        "\n"
        "    chunk = chimefrb.AssembledChunk.from_msgpack('chunk_01060317.msg')\n"
        "    intensity = chunk.decode_intensity(apply_rfimask=False)   # (nfreq, nt) float32\n"
        "\n"
        "Only uncompressed, format-version-2 files are supported; anything else raises.\n"
        "\n"
        "The ``data``, ``scales``, ``offsets`` and ``rfi_mask`` attributes are zero-copy\n"
        "numpy views into one buffer owned by the chunk. They stay valid for as long as the\n"
        "arrays themselves are alive -- no context manager is needed.")

        .def_static("from_msgpack", &AssembledChunk::from_msgpack,
            py::arg("filename"), py::arg("allocator") = std::shared_ptr<SlabAllocator>(),
            py::call_guard<py::gil_scoped_release>(),
            "Read and parse one file. Raises if it is truncated, corrupt, compressed, or not\n"
            "msgpack format version 2.\n"
            "\n"
            "If 'allocator' is given, the chunk's buffer comes from it. Note a SlabAllocator\n"
            "serves a single slab size, so all files sharing one allocator must have identical\n"
            "parameters.")

        .def_readonly("version", &AssembledChunk::version, "msgpack format version (always 2).")
        .def_readonly("compression", &AssembledChunk::compression,
            "Compression flag from the file header (always 0; compressed files are rejected).")
        .def_readonly("beam_id", &AssembledChunk::beam_id)
        .def_readonly("binning", &AssembledChunk::binning,
            "Downsampling level in the telescoping ring buffer: 1, 2, 4, ...")
        .def_readonly("nupfreq", &AssembledChunk::nupfreq,
            "Upchannelization factor: fine frequency channels per coarse channel.")
        .def_readonly("nt_per_packet", &AssembledChunk::nt_per_packet,
            "Time samples sharing one (scale, offset) pair.")
        .def_readonly("fpga_counts_per_sample", &AssembledChunk::fpga_counts_per_sample)
        .def_readonly("nt_coarse", &AssembledChunk::nt_coarse)
        .def_readonly("nscales", &AssembledChunk::nscales)
        .def_readonly("ndata", &AssembledChunk::ndata)
        .def_readonly("nrfifreq", &AssembledChunk::nrfifreq,
            "Frequency resolution of the RFI chain, which is coarser than nfreq. Zero if the\n"
            "file carries no RFI mask.")
        .def_readonly("fpga_begin", &AssembledChunk::fpga_begin)
        .def_readonly("fpga_end", &AssembledChunk::fpga_end)
        .def_readonly("frame0_nano", &AssembledChunk::frame0_nano,
            "ctime in nanoseconds of FPGA count zero.")
        .def_readonly("has_rfi_mask", &AssembledChunk::has_rfi_mask)
        .def_readonly("nfreq_coarse", &AssembledChunk::nfreq_coarse,
            "Coarse frequency channels. Not stored in the file: derived as nscales/nt_coarse.")
        .def_readonly("nt_per_chunk", &AssembledChunk::nt_per_chunk,
            "Time samples in the chunk. Not stored in the file: derived as\n"
            "nt_coarse*nt_per_packet.")

        .def_property_readonly("nfreq", &AssembledChunk::nfreq,
            "Fine frequency channels, nfreq_coarse*nupfreq. Channel 0 is the HIGHEST radio\n"
            "frequency (800 MHz), decreasing to 400 MHz -- the same convention pirate uses.")
        .def_property_readonly("nt", &AssembledChunk::nt, "Time samples; same as nt_per_chunk.")

        .def_readonly("data", &AssembledChunk::data,
            "Raw uint8 data, shape (nfreq, nt). Most callers want decode_intensity() instead.")
        .def_readonly("scales", &AssembledChunk::scales,
            "float32 array of shape (nfreq_coarse, nt_coarse).")
        .def_readonly("offsets", &AssembledChunk::offsets,
            "float32 array of shape (nfreq_coarse, nt_coarse).")
        // Returns None rather than an empty array when the file carried no mask: ksgpu's
        // type_caster has no numpy representation for a null-pointer Array.
        .def_property_readonly("rfi_mask",
            [](const AssembledChunk &self) -> py::object {
                if (!self.has_rfi_mask || (self.rfi_mask.size == 0))
                    return py::none();
                return py::cast(self.rfi_mask);
            },
            "Bit-packed RFI mask, shape (nrfifreq, nt/8) BYTES, LSB-first within each byte:\n"
            "bit i of byte j is time sample 8*j+i. A SET bit means GOOD data.\n"
            "\n"
            "Note the frequency axis is nrfifreq, NOT nfreq -- the RFI chain ran coarser than\n"
            "the data, so applying this to the data means broadcasting over nfreq/nrfifreq\n"
            "fine channels. None if has_rfi_mask is False.")

        .def("decode_intensity",
            [](const AssembledChunk &self, bool apply_rfimask, optional<Array<float>> out) {
                Array<float> dst = _decode_dst(self, out);
                self.decode_intensity(dst, apply_rfimask);
                return dst;
            },
            py::arg("apply_rfimask"), py::arg("out") = std::nullopt,
            py::call_guard<py::gil_scoped_release>(),
            "Decode to physical units: scales*data + offsets, as a (nfreq, nt) float32 array.\n"
            "\n"
            "'apply_rfimask' has no default on purpose -- on real data it changes ~46% of\n"
            "samples. When True, samples the RFI mask marks bad are zeroed, and the call\n"
            "raises if the file carried no mask. Pass the SAME value to decode_weights():\n"
            "masking the intensity but not the weights leaves a masked sample looking like a\n"
            "real measurement of zero.\n"
            "\n"
            "If 'out' is given it is written in place and returned; otherwise a new array is\n"
            "allocated. Requires nt_per_packet == 16.")

        .def("decode_weights",
            [](const AssembledChunk &self, bool apply_rfimask, optional<Array<float>> out) {
                Array<float> dst = _decode_dst(self, out);
                self.decode_weights(dst, apply_rfimask);
                return dst;
            },
            py::arg("apply_rfimask"), py::arg("out") = std::nullopt,
            py::call_guard<py::gil_scoped_release>(),
            "Per-sample weights as a (nfreq, nt) float32 array: 0 where the raw data is 0 or\n"
            "255 (the saturation sentinels), else 1.\n"
            "\n"
            "With apply_rfimask=False this reproduces ch_frb_io's assembled_chunk::decode()\n"
            "exactly, which is what a comparison against the old pipeline needs. See\n"
            "decode_intensity() for the apply_rfimask semantics.")

        .def("fraction_missing", &AssembledChunk::fraction_missing,
            "Fraction of (coarse channel, time block) pairs for which no packet ever arrived.\n"
            "\n"
            "This is packet loss, and it is the only thing that distinguishes packet loss from\n"
            "RFI flagging -- the raw data is zero in these blocks either way. It is NOT an\n"
            "extra masking source: decode_weights() already zeroes those samples.")
        ;

    // GpuWiDownsampler: Python injections in pirate_frb/chimefrb/ReferenceWiDownsampler.py:
    //   - launch: converts stream=None to current cupy stream
    py::class_<GpuWiDownsampler>(m, "GpuWiDownsampler",
        "Reduces an (intensity, weights) pair by a factor Df in frequency and Dt in time,\n"
        "using the normalization of the old CHIME FRB search::\n"
        "\n"
        "    out_w = sum of the cell's weights          (SUM, not mean)\n"
        "    out_i = (sum of w*i) / out_w,  or 0 where out_w <= 0\n"
        "\n"
        "At (Df, Dt) = (1, 1), out_i is the input intensity exactly (where out_w > 0), not\n"
        "(w*i)/w, which can differ in the last bit.\n"
        "\n"
        "The 'sum, not mean' is worth flagging: rf_kernels::wi_downsampler (which this\n"
        "ports) sums, while the python helper rf_pipelines.utils.wi_downsample() takes the\n"
        "mean, so the two differ by a factor (Df*Dt).\n"
        "\n"
        "One thing the old kernel does not have: 'transpose' writes the output with\n"
        "frequency as the fastest-varying axis. The clippers want the downsampled data in\n"
        "both layouts, so a caller may run this twice on the same source arrays.\n"
        "\n"
        "(Df, Dt) and the array shapes are runtime, so there is no table of supported\n"
        "configurations; see :meth:`launch` for the divisibility rules.")

        .def(py::init<long, long, bool, long>(),
            py::arg("Df"), py::arg("Dt"), py::arg("transpose"), py::arg("warps_per_block") = 32,
            "Create a GpuWiDownsampler.\n"
            "\n"
            "Args:\n"
            "    Df: frequency downsampling factor\n"
            "    Dt: time downsampling factor\n"
            "    transpose: if True, output axes are (beam, time, freq)\n"
            "    warps_per_block: performance knob, 4/8/16/32. Must not change the result.\n"
            "        The default is what measures fastest on an L40S; see time_selected().\n"
            "\n"
            "Raises:\n"
            "    RuntimeError: on Df < 1, Dt < 1, an unsupported warps_per_block, or\n"
            "        (Df, Dt, transpose) = (1, 1, False), which is the identity.")

        .def_readonly("Df", &GpuWiDownsampler::Df, "Frequency downsampling factor")
        .def_readonly("Dt", &GpuWiDownsampler::Dt, "Time downsampling factor")
        .def_readonly("transpose", &GpuWiDownsampler::transpose,
            "If True, output axes are (beam, time, freq)")
        .def_readonly("warps_per_block", &GpuWiDownsampler::warps_per_block,
            "Warps sharing one 32-by-32 output tile (4, 8, 16 or 32)")

        .def_static("time_selected", &GpuWiDownsampler::time_selected,
            py::call_guard<py::gil_scoped_release>(),
            "Run timing benchmarks, for the (Df, Dt, transpose) configurations used by the\n"
            "old search's production RFI chain (called via 'python -m pirate_frb time --cfrb')")

        .def("launch",
            [](const GpuWiDownsampler &self, Array<float> &out_i, Array<float> &out_w,
               const Array<float> &in_i, const Array<float> &in_w, uintptr_t stream_ptr) {
                cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
                self.launch(out_i, out_w, in_i, in_w, stream);
            },
            py::arg("out_i"), py::arg("out_w"), py::arg("in_i"), py::arg("in_w"),
            py::arg("stream_ptr"),
            py::call_guard<py::gil_scoped_release>(),   // async launch; body is pure C++
            "GPU kernel launch (async, does not sync stream).\n"
            "\n"
            "All four arrays are cupy float32 arrays, fully contiguous, on GPU, and the\n"
            "outputs must not alias the inputs. With (F_ds, T_ds) = (F/Df, T/Dt):\n"
            "\n"
            "Args:\n"
            "    out_i: shape (B, F_ds, T_ds), or (B, T_ds, F_ds) if transpose.\n"
            "        Downsampled intensity. Fully overwritten.\n"
            "    out_w: same shape as out_i. Downsampled weights. Fully overwritten.\n"
            "    in_i: shape (B, F, T). Intensity. Read only.\n"
            "    in_w: shape (B, F, T). Weights, which must be >= 0. Read only.\n"
            "    stream_ptr: CUDA stream pointer (integer, e.g. from cupy stream.ptr)\n"
            "\n"
            "Raises:\n"
            "    RuntimeError: unless F is divisible by 32*Df and T by 32*Dt (the 32 is the\n"
            "        output tile size, and the kernel has no edge predication).")
        ;

    // GpuWrms: Python injections in pirate_frb/chimefrb/ReferenceWrms.py:
    //   - launch: converts stream=None to current cupy stream, allocates scratch=None
    py::class_<GpuWrms>(m, "GpuWrms",
        "The weighted mean and variance of each row of an (R, L) array, refined by\n"
        "iterated sigma clipping. A port of rf_kernels::weighted_mean_rms, and the\n"
        "statistic both chimefrb clippers are built on.\n"
        "\n"
        "Does NOT apply any threshold to the weights -- that is a separate kernel. Its\n"
        "only outputs are, per row, a mean and a variance.\n"
        "\n"
        "The three clipper axes all arrive here as row reductions of a contiguous 2-D\n"
        "array, so this class knows nothing about frequencies, times or axes: the caller\n"
        "views GpuWiDownsampler's output as (R, L), and the whole-plane case is the same\n"
        "thing with one long row.\n"
        "\n"
        "Two things are easy to misread: 'niter' counts TOTAL passes, so niter=1 means no\n"
        "refinement at all; and var == 0 is the 'no usable statistic' signal, which a row\n"
        "gets when it has no weight or when its variance falls below the algorithm's\n"
        "float32 epsilon cutoffs. Once a row's variance is zero, every later refinement\n"
        "leaves it zero. See plans/chimefrb_wrms.md section 2 for the full algorithm.")

        .def(py::init<long, long, double, bool, long>(),
            py::arg("L"), py::arg("niter"), py::arg("iter_sigma"), py::arg("two_pass"),
            py::arg("threads_per_block") = 128,
            "Create a GpuWrms.\n"
            "\n"
            "Args:\n"
            "    L: samples per row. A constructor argument because it decides which of\n"
            "        two kernels runs: a row that fits in shared memory is refined\n"
            "        on-chip, a longer one is re-read from global once per refinement.\n"
            "    niter: TOTAL passes. 1 means no refinement.\n"
            "    iter_sigma: the threshold used BY THE REFINEMENTS, in units of the\n"
            "        current rms. Not the intensity_clipper's final-clip sigma, which is\n"
            "        a different number applied by a different kernel. Ignored at niter=1.\n"
            "    two_pass: use the stabler two-pass first pass.\n"
            "    threads_per_block: performance knob, 128/256/512/1024. Must not change\n"
            "        the result. The default is the smallest value, unlike\n"
            "        GpuWiDownsampler: see time_selected().\n"
            "\n"
            "Raises:\n"
            "    RuntimeError: on L < 1, niter < 1, iter_sigma < 0, or an unsupported\n"
            "        threads_per_block.")

        .def_readonly("L", &GpuWrms::L, "Samples per row")
        .def_readonly("niter", &GpuWrms::niter, "TOTAL passes; 1 means no refinement")
        .def_readonly("iter_sigma", &GpuWrms::iter_sigma,
            "Refinement clipping threshold, in units of the current rms")
        .def_readonly("two_pass", &GpuWrms::two_pass, "Use the stabler two-pass first pass")
        .def_readonly("threads_per_block", &GpuWrms::threads_per_block, "128, 256, 512 or 1024")

        .def_property_readonly("is_shared_memory_path", &GpuWrms::is_shared_memory_path,
            "True if this L uses the shared-memory kernel (the row is staged on-chip and\n"
            "the input is read exactly once), False if it uses the global-memory kernel\n"
            "(the row is re-read once per refinement). A test or a timing run wants to say\n"
            "which path it measured.")

        .def_static("max_shared_L", &GpuWrms::max_shared_L,
            "The largest L that uses the shared-memory kernel. One source of truth for\n"
            "the threshold, so that a test drawing L either side of it cannot drift from\n"
            "the kernel's own idea of where it is.")

        .def("scratch_nelts", &GpuWrms::scratch_nelts, py::arg("R"),
            "Number of float32 scratch elements launch() needs for R rows. Zero on the\n"
            "shared-memory path.")

        .def_static("time_selected", &GpuWrms::time_selected,
            py::call_guard<py::gil_scoped_release>(),
            "Run timing benchmarks, for the configurations the old search's production\n"
            "RFI chain uses (called via 'python -m pirate_frb time --cfrb')")

        .def("launch",
            [](const GpuWrms &self, Array<float> &mean, Array<float> &var,
               const Array<float> &in_i, const Array<float> &in_w,
               Array<float> &scratch, uintptr_t stream_ptr) {
                cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
                self.launch(mean, var, in_i, in_w, scratch, stream);
            },
            py::arg("mean"), py::arg("var"), py::arg("in_i"), py::arg("in_w"),
            py::arg("scratch"), py::arg("stream_ptr"),
            py::call_guard<py::gil_scoped_release>(),   // async launch; body is pure C++
            "GPU kernel launch (async, does not sync stream).\n"
            "\n"
            "All arrays are cupy float32 arrays, fully contiguous, on GPU, and the outputs\n"
            "must not alias the inputs.\n"
            "\n"
            "Args:\n"
            "    mean: shape (R,). Weighted mean of each row. Fully overwritten.\n"
            "    var: shape (R,). Weighted variance, or 0 for a row with no usable\n"
            "        statistic. Fully overwritten, and never negative.\n"
            "    in_i: shape (R, L). Intensity. Read only.\n"
            "    in_w: shape (R, L). Weights, which must be >= 0. Read only.\n"
            "    scratch: shape (scratch_nelts(R),), or empty when that is zero.\n"
            "    stream_ptr: CUDA stream pointer (integer, e.g. from cupy stream.ptr)")
        ;

    // GpuTransformBase: the base class of every transform. Python injections in
    // pirate_frb/chimefrb/GpuTransformBase.py, which also carries the class docstring
    // (option 2 in notes/docstrings.md): launch() with stream=None and scratch=None
    // handling, the check_yaml_keys() classmethod, and __repr__. The python-only side is
    // GpuPythonTransform (see the trampoline above).
    py::class_<GpuTransformBase, PyGpuTransformBase>(m, "GpuTransformBase")
        .def(py::init<const std::string &, long, long, long, long>(),
            py::arg("name"), py::arg("nbeams"), py::arg("nfreq"), py::arg("ntime"), py::arg("scratch_nelts"),
            "The C++ constructor, called by GpuPythonTransform.__init__() with the python class's\n"
            "name. Not for direct use: a bare GpuTransformBase has no computation to launch.")

        .def_readonly("nbeams", &GpuTransformBase::nbeams)
        .def_readonly("nfreq", &GpuTransformBase::nfreq)
        .def_readonly("ntime", &GpuTransformBase::ntime)
        .def_readonly("scratch_nelts", &GpuTransformBase::scratch_nelts)

        .def("launch",
            [](const GpuTransformBase &self, Array<float> &intensity, Array<float> &weights,
               Array<float> &scratch, uintptr_t stream_ptr) {
                cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
                self.launch(intensity, weights, scratch, stream);
            },
            py::arg("intensity"), py::arg("weights"), py::arg("scratch"), py::arg("stream_ptr"),
            py::call_guard<py::gil_scoped_release>(),   // async launch; a python subclass re-acquires
            "The raw form of launch(): stream_ptr is an integer. Python callers use the injected\n"
            "launch(intensity, weights, scratch, stream=None), which wraps this.")
        ;

    // GpuSplineDetrender: Python injections in pirate_frb/chimefrb/ReferenceSplineDetrender.py
    // (the yaml and legacy-json methods; see the top of this file).
    py::class_<GpuSplineDetrender, GpuTransformBase>(m, "GpuSplineDetrender",
        "A port of rf_kernels::spline_detrender, the frequency-direction detrender of the\n"
        "old CHIME FRB search's RFI chain.\n"
        "\n"
        "Per (beam, time sample), independently: fit a piecewise-cubic spline in frequency\n"
        "-- ``nbins`` equal bins, C^1 at the bin edges, 2*(nbins+1) coefficients -- to the\n"
        "intensity by weighted least squares, and subtract it from every channel. The fit\n"
        "is regularized by ``epsilon * (sum of weights) / nbins`` times the integrated\n"
        "squared slope, so a constant baseline is removed exactly and the penalty weakens\n"
        "with the sample's total weight. Weights are real-valued and nonnegative ({0,1} at\n"
        "full resolution, integer counts after downsampling, in the production chain), and\n"
        "are read but never modified. A sample whose weights are all zero is left untouched.\n"
        "\n"
        "This reproduces the old code's ESTIMATOR, including its bin geometry to the\n"
        "fraction of a channel, but not its arithmetic: the solve is equilibrated, and a\n"
        "zero-weight channel contributes exactly zero even if its intensity is NaN. There\n"
        "is no mask expansion and no conditioning statistic. Validated against\n"
        ":class:`ReferenceSplineDetrender`, a transcription of the old code's own reference;\n"
        "the GPU kernels are those of :class:`pirate_frb.kernels.GpuDetrenderLps2d`,\n"
        "instantiated with the Hermite basis.\n"
        "\n"
        "Stateless across launches: the instance holds only read-only tables, and its\n"
        "workspace comes from the caller's scratch array, so one instance may be used from\n"
        "any number of streams at once (with one scratch per stream).\n"
        "\n"
        "launch() modifies ``intensity`` only; ``weights`` is read.\n"
        "\n"
        "Attributes (read-only):\n"
        "\n"
        "- ``nbeams``, ``nfreq``, ``ntime``, ``nbins``, ``epsilon`` -- the constructor arguments.\n"
        "- ``scratch_nelts`` (int) -- float32 scratch elements launch() needs.\n"
        "- ``N_phi`` (int) -- number of spline coefficients, 2*(nbins+1).\n"
        "- ``nfrange``, ``channels_per_range``, ``solve_threads`` -- launch geometry, derived\n"
        "  in the constructor; of interest to timing runs only.\n")

        .def(py::init<long, long, long, long, double>(),
            py::arg("nbeams"), py::arg("nfreq"), py::arg("ntime"), py::arg("nbins"), py::arg("epsilon"),
            "Create a GpuSplineDetrender.\n"
            "\n"
            "Args:\n"
            "    nbeams, nfreq, ntime: the array shape launch() will be given; ntime a\n"
            "        positive multiple of 32.\n"
            "    nbins: equal bins; the spline is C^1 across bin edges.\n"
            "    epsilon: regularization strength (the production chain uses 3e-4).\n"
            "\n"
            "Raises:\n"
            "    RuntimeError: on nbeams < 1, ntime not a positive multiple of 32, nbins < 1,\n"
            "        nfreq < nbins, or epsilon <= 0.")

        .def_readonly("nbins", &GpuSplineDetrender::nbins)
        .def_readonly("epsilon", &GpuSplineDetrender::epsilon)
        .def_readonly("N_phi", &GpuSplineDetrender::N_phi)
        .def_readonly("nfrange", &GpuSplineDetrender::nfrange)
        .def_readonly("channels_per_range", &GpuSplineDetrender::channels_per_range)
        .def_readonly("solve_threads", &GpuSplineDetrender::solve_threads)

        .def("bin_edges", &GpuSplineDetrender::bin_edges,
            "The bin edges as channel indices, length nbins+1, running from 0 to nfreq.")

        .def_static("time_selected", &GpuSplineDetrender::time_selected,
            py::call_guard<py::gil_scoped_release>(),
            "Run timing benchmarks at the two shapes the old search's production RFI chain\n"
            "uses this detrender at (called via 'python -m pirate_frb time --cfrb')")

        ;

    // GpuPolynomialDetrender: Python injections in
    // pirate_frb/chimefrb/ReferencePolynomialDetrender.py (the yaml and legacy-json methods;
    // see the top of this file).
    py::class_<GpuPolynomialDetrender, GpuTransformBase>(m, "GpuPolynomialDetrender",
        "A port of rf_pipelines::polynomial_detrender along the time axis, the only axis\n"
        "the old CHIME FRB search's production RFI chain ran it on.\n"
        "\n"
        "Per (beam, channel, chunk of ``nt_chunk`` samples), independently: fit a polynomial\n"
        "of degree ``polydeg`` in time to the intensity by weighted least squares (Legendre\n"
        "basis, no regularization), and subtract it at every sample of the chunk. Before\n"
        "the solve, a CONDITIONING GATE decides whether the row is fit at all: the normal\n"
        "matrix is Cholesky-factored pivot by pivot, and the row passes only if at every\n"
        "pivot the Schur complement exceeds ``epsilon`` times the diagonal entry. A row that\n"
        "fails has ALL its weights set to zero and its intensity left untouched -- the\n"
        "transform's only effect on the weights, and not a corner case: at the production\n"
        "setting (degree 4, epsilon 0.01, 1024-sample chunks) a channel whose weighted\n"
        "samples form one contiguous run shorter than about half the chunk is erased for\n"
        "that chunk.\n"
        "\n"
        "This reproduces the old code's ESTIMATOR and gate DECISION, not its arithmetic:\n"
        "the solve is rescaled to unit diagonal first, and a row within float32 roundoff\n"
        "of the threshold may be decided either way. A NaN intensity at a zero-weight\n"
        "sample contributes exactly zero (the old code let it poison the row). Validated\n"
        "against :class:`ReferencePolynomialDetrender`, a transcription of the old kernel,\n"
        "which also implements the old code's 'freq' variant.\n"
        "\n"
        "Stateless: one instance may be used from any number of streams at once.\n"
        "\n"
        "launch() modifies both arrays: ``intensity`` on rows that pass the gate, ``weights``\n"
        "on rows that fail it.\n"
        "\n"
        "Attributes (read-only):\n"
        "\n"
        "- ``nbeams``, ``nfreq``, ``ntime``, ``polydeg``, ``epsilon``, ``nt_chunk``,\n"
        "  ``warps_per_block`` -- the constructor arguments.\n"
        "- ``scratch_nelts`` (int) -- always 0: launch() needs no scratch.\n")

        .def(py::init<long, long, long, long, double, long, long>(),
            py::arg("nbeams"), py::arg("nfreq"), py::arg("ntime"), py::arg("polydeg"),
            py::arg("epsilon"), py::arg("nt_chunk"), py::arg("warps_per_block") = 16,
            "Create a GpuPolynomialDetrender.\n"
            "\n"
            "Args:\n"
            "    nbeams, nfreq, ntime: the array shape launch() will be given; ntime a\n"
            "        positive multiple of nt_chunk (one fit per beam, channel and chunk).\n"
            "    polydeg: degree of the fit, 0..8 (the production chain uses 4).\n"
            "    epsilon: gate threshold, > 0 (the production chain uses 0.01).\n"
            "    nt_chunk: samples per independent fit; a positive multiple of 64 (the\n"
            "        production chain uses 1024, one assembled chunk).\n"
            "    warps_per_block: 4, 8 or 16. A performance knob; does not change the result.\n"
            "\n"
            "Raises:\n"
            "    RuntimeError: on an argument outside those ranges.")

        .def_readonly("polydeg", &GpuPolynomialDetrender::polydeg)
        .def_readonly("epsilon", &GpuPolynomialDetrender::epsilon)
        .def_readonly("nt_chunk", &GpuPolynomialDetrender::nt_chunk)
        .def_readonly("warps_per_block", &GpuPolynomialDetrender::warps_per_block)

        .def_static("time_selected", &GpuPolynomialDetrender::time_selected,
            py::call_guard<py::gil_scoped_release>(),
            "Run timing benchmarks at the production configuration, at the two channel\n"
            "counts the old chain uses (called via 'python -m pirate_frb time --cfrb')")

        ;

    // GpuClipperBase: bound without a constructor. It exists so that the attributes the
    // clippers share (axis, statistic parameters, derived geometry) are bound, and
    // documented, once; the clippers are bound as its subclasses.
    py::class_<GpuClipperBase, GpuTransformBase>(m, "GpuClipperBase",
        "What the chimefrb clippers have in common, on top of GpuTransformBase: axis,\n"
        "(Df, Dt), the per-row weighted mean and variance (downsample, transpose if\n"
        "axis == FREQ, then GpuWrms), and the scratch layout. Not constructible on its own:\n"
        "GpuIntensityClipper and GpuStdDevClipper derive from it.\n"
        "\n"
        "CHUNKING: every clipper requires ntime == nt_chunk (the array holds exactly one\n"
        "chunk). The old code's ntime = N*nt_chunk behaviour is implemented by the numpy\n"
        "references, not here.")

        .def_readonly("nt_chunk", &GpuClipperBase::nt_chunk,
            "Samples per chunk. Currently required to equal ntime.")
        .def_readonly("axis", &GpuClipperBase::axis,
            "Which axis is reduced: 'freq', 'time' or 'none'")
        .def_readonly("Df", &GpuClipperBase::Df, "Frequency downsampling factor")
        .def_readonly("Dt", &GpuClipperBase::Dt, "Time downsampling factor")
        .def_readonly("niter", &GpuClipperBase::niter,
            "TOTAL passes of the statistic; 1 means no refinement (always 1 for\n"
            "GpuStdDevClipper)")
        .def_readonly("iter_sigma", &GpuClipperBase::iter_sigma,
            "The statistic's REFINEMENT threshold, in rms units; ignored at niter=1")
        .def_readonly("two_pass", &GpuClipperBase::two_pass,
            "Use the stabler two-pass first pass of the statistic")

        .def_readonly("F_ds", &GpuClipperBase::F_ds, "nfreq // Df")
        .def_readonly("T_ds", &GpuClipperBase::T_ds, "nt_chunk // Dt")
        .def_readonly("wrms_L", &GpuClipperBase::wrms_L,
            "Samples per statistic row. Exposed so that a test can rebuild the statistic\n"
            "from GpuWiDownsampler and GpuWrms exactly, which is how the clippers are\n"
            "checked: the statistic comes from the GPU, and what follows it from numpy.")
        .def_readonly("wrms_R", &GpuClipperBase::wrms_R,
            "Statistic rows: nbeams*F_ds (TIME), nbeams*T_ds (FREQ), or nbeams (NONE)")
        ;

    // GpuIntensityClipper: Python injections in
    // pirate_frb/chimefrb/ReferenceIntensityClipper.py (the yaml and legacy-json methods; see
    // the top of this file).
    py::class_<GpuIntensityClipper, GpuClipperBase>(m, "GpuIntensityClipper",
        "Zeroes the weights of samples that sit more than 'sigma' standard deviations from\n"
        "a weighted mean. A port of rf_kernels::intensity_clipper, the old CHIME FRB\n"
        "search's principal RFI flagger (48 of its production chain's 120 nodes).\n"
        "\n"
        "Three steps, of which only the last is new code: downsample by (Df, Dt), compute\n"
        "the weighted mean and variance over ``axis`` (GpuWrms), then mask every\n"
        "downsampled cell with ``|I_ds - mean| >= sigma*sqrt(var)``, zeroing all Df*Dt\n"
        "full-resolution weights of a masked cell.\n"
        "\n"
        "Three things are worth knowing before calling it:\n"
        "\n"
        "``sigma`` and ``iter_sigma`` are DIFFERENT NUMBERS. sigma is the final clip;\n"
        "iter_sigma is used inside the statistic's refinements. In the production chain\n"
        "they are 5 and 3.\n"
        "\n"
        "A row whose variance was rejected (var == 0) gets a threshold of zero, and the\n"
        "strict '<' in the survivor test then masks EVERY sample in the row. That is\n"
        "intentional in the original: no usable statistic means no usable data.\n"
        "\n"
        "CHUNKING: the old code applies this transform to one 'nt_chunk' block of the\n"
        "stream at a time, and the result does depend on where those boundaries fall. This\n"
        "class requires ntime == nt_chunk (the array holds exactly ONE chunk), which is all\n"
        "the production RFI chain needs. Processing ntime = N*nt_chunk samples in one call\n"
        "would be a useful generalization and is deliberately left for later;\n"
        "ReferenceIntensityClipper does implement it, so the semantics are pinned down and\n"
        "tested.\n"
        "\n"
        "launch() modifies ``weights`` only; ``intensity`` is read.")

        .def(py::init<long, long, long, long, ClipperAxis, double, long, long, long, double, bool, long>(),
            py::arg("nbeams"), py::arg("nfreq"), py::arg("ntime"), py::arg("nt_chunk"), py::arg("axis"),
            py::arg("sigma"), py::arg("Df"), py::arg("Dt"), py::arg("niter"), py::arg("iter_sigma"),
            py::arg("two_pass"), py::arg("warps_per_block") = 16,
            "Create a GpuIntensityClipper.\n"
            "\n"
            "Args:\n"
            "    nbeams, nfreq, ntime: the full-resolution array shape, fixed at construction.\n"
            "    nt_chunk: samples per chunk; must equal ntime for now (see CHUNKING above).\n"
            "    axis: 'freq', 'time' or 'none' -- which axis the statistic reduces along.\n"
            "    sigma: the FINAL clip threshold, in units of the row's rms.\n"
            "    Df, Dt: downsampling factors for frequency and time.\n"
            "    niter: TOTAL passes of the statistic. 1 means no refinement.\n"
            "    iter_sigma: the threshold used BY THE REFINEMENTS, in the same units.\n"
            "        Ignored at niter=1. Unlike rf_kernels::intensity_clipper, 0 has no\n"
            "        special meaning here (there it means 'use sigma').\n"
            "    two_pass: use the stabler two-pass first pass of the statistic.\n"
            "    warps_per_block: performance knob, 4/8/16/32, for the final clip kernel.\n"
            "        Must not change the result. The default is the MIDDLE of the menu,\n"
            "        unlike either of the other two chimefrb kernels. See time_selected().\n"
            "\n"
            "Raises:\n"
            "    RuntimeError: unless ntime == nt_chunk, nfreq is divisible by 32*Df and\n"
            "        nt_chunk by 32*Dt; also on nbeams < 1, Df < 1, Dt < 1, niter < 1,\n"
            "        sigma < 0, iter_sigma < 0, or an unsupported warps_per_block.")

        // The shared attributes are bound on GpuClipperBase and GpuTransformBase.
        .def_readonly("sigma", &GpuIntensityClipper::sigma,
            "FINAL clip threshold, in units of the row's rms (not iter_sigma)")
        .def_readonly("warps_per_block", &GpuIntensityClipper::warps_per_block,
            "Performance knob for the final clip kernel (4, 8, 16 or 32)")

        .def_static("time_selected", &GpuIntensityClipper::time_selected,
            py::call_guard<py::gil_scoped_release>(),
            "Run timing benchmarks, for the four configurations the old search's\n"
            "production RFI chain uses (called via 'python -m pirate_frb time --cfrb')")

        ;

    // GpuStdDevClipper: Python injections in pirate_frb/chimefrb/ReferenceStdDevClipper.py
    // (the yaml and legacy-json methods; see the top of this file).
    py::class_<GpuStdDevClipper, GpuClipperBase>(m, "GpuStdDevClipper",
        "Zeroes whole channels (axis 'time') or whole time samples (axis 'freq') whose variance\n"
        "is an outlier among its peers. A port of rf_kernels::std_dev_clipper, the most\n"
        "numerous transform in the old CHIME FRB search's RFI chain (60 of its 120 nodes).\n"
        "Where GpuIntensityClipper catches samples that are too bright, this catches rows\n"
        "whose NOISE LEVEL is wrong.\n"
        "\n"
        "Per row, a weighted variance (GpuWrms at niter=1 -- variances, not standard\n"
        "deviations, despite the name). Per beam, the mean vbar and standard deviation s of\n"
        "the nonzero variances (s divides by n, not n-1), computed before anything is\n"
        "clipped; every row with ``|v - vbar| >= sigma*s`` then has all its weights zeroed.\n"
        "\n"
        "Two things are worth knowing before calling it:\n"
        "\n"
        "If at most ONE row of a beam has a usable variance, the whole beam's chunk is\n"
        "zeroed -- including, when exactly one channel has data, that channel. This is the\n"
        "old code's behaviour, reproduced deliberately.\n"
        "\n"
        "If every usable variance in a beam is exactly equal, the outcome is decided by\n"
        "float32 roundoff (exact arithmetic clips every row; a rounded mean usually clips\n"
        "none), and this class and the old code round differently. Real data cannot produce\n"
        "it; it is documented rather than fixed.\n"
        "\n"
        "Axis 'none' is not supported, as in the old code. Like every GpuClipperBase, it\n"
        "requires ntime == nt_chunk (the array holds exactly one chunk).\n"
        "\n"
        "launch() modifies ``weights`` only; ``intensity`` is read.")

        .def(py::init<long, long, long, long, ClipperAxis, double, long, long, bool, long>(),
            py::arg("nbeams"), py::arg("nfreq"), py::arg("ntime"), py::arg("nt_chunk"), py::arg("axis"),
            py::arg("sigma"), py::arg("Df"), py::arg("Dt"), py::arg("two_pass"), py::arg("warps_per_block") = 16,
            "Create a GpuStdDevClipper.\n"
            "\n"
            "Args:\n"
            "    nbeams, nfreq, ntime: the full-resolution array shape, fixed at construction.\n"
            "    nt_chunk: samples per chunk; must equal ntime for now.\n"
            "    axis: 'time' or 'freq' ('none' is not supported; see above).\n"
            "    sigma: the clip threshold, in units of the standard deviation OF THE\n"
            "        VARIANCES. Nothing can be clipped if sigma >= sqrt(n-1), for n usable\n"
            "        rows in a beam.\n"
            "    Df, Dt: downsampling factors for frequency and time.\n"
            "    two_pass: use the stabler two-pass form of the per-row variance.\n"
            "    warps_per_block: performance knob, 4/8/16/32, for the kernel that zeroes the\n"
            "        weights. Must not change the result. See time_selected().\n"
            "\n"
            "Raises:\n"
            "    RuntimeError: on axis NONE; unless ntime == nt_chunk, nfreq is divisible by\n"
            "        32*Df and nt_chunk by 32*Dt; and on nbeams < 1, Df < 1, Dt < 1,\n"
            "        sigma < 0, or an unsupported warps_per_block.")

        // The shared attributes are bound on GpuClipperBase and GpuTransformBase.
        .def_readonly("sigma", &GpuStdDevClipper::sigma,
            "Clip threshold, in units of the standard deviation of the variances")
        .def_readonly("warps_per_block", &GpuStdDevClipper::warps_per_block,
            "Performance knob for the kernel that zeroes the weights (4, 8, 16 or 32)")

        .def_static("time_selected", &GpuStdDevClipper::time_selected,
            py::call_guard<py::gil_scoped_release>(),
            "Run timing benchmarks, for the two configurations the old search's production\n"
            "RFI chain uses (called via 'python -m pirate_frb time --cfrb')")

        ;

    // GpuBadChannelMask: Python injections in pirate_frb/chimefrb/ReferenceBadChannelMask.py
    // (the yaml and legacy-json methods; see the top of this file). Its __init__ also
    // normalizes 'mask_ranges' and 'freq_range' to python floats.
    py::class_<GpuBadChannelMask, GpuTransformBase>(m, "GpuBadChannelMask",
        "Zeroes the weights of whole frequency channels. A port of rf_pipelines::badchannel_mask,\n"
        "which the old CHIME FRB search used at the start of its RFI chain to remove channels\n"
        "known in advance to be bad.\n"
        "\n"
        "As in the old transform, the channels are given as a list of (lo, hi) frequency ranges\n"
        "in MHz, together with the band ``freq_range`` that the ``nfreq`` channels span, channel\n"
        "0 at the top. The constructor converts them to a per-channel ``keep`` array with the old\n"
        "code's arithmetic, including its quirks; ``badchannel_keep()`` is the python\n"
        "transcription of that arithmetic, and its docstring states the rule.\n"
        "\n"
        "A masked channel is set to +0.0, whatever it held (the old code stores a literal 0),\n"
        "and every other weight is left bit-identical. The kernel writes only the zeros, so its\n"
        "cost is proportional to the number of masked channels. Nothing depends on time, so\n"
        "ntime may be anything.\n"
        "\n"
        "launch() modifies ``weights`` only; ``intensity`` is checked and never touched.\n"
        "\n"
        "Attributes (read-only):\n"
        "\n"
        "- ``nbeams``, ``nfreq``, ``ntime``, ``mask_ranges``, ``freq_range``,\n"
        "  ``warps_per_block`` -- the constructor arguments.\n"
        "- ``nmasked`` (int) -- number of masked channels.\n"
        "- ``keep`` (cupy uint8 array, shape (nfreq,)) -- 1 = keep, 0 = mask.\n"
        "- ``scratch_nelts`` (int) -- always 0: launch() needs no scratch.\n")

        .def(py::init<long, long, long, const std::vector<std::pair<double,double>> &,
                      std::pair<double,double>, long>(),
            py::arg("nbeams"), py::arg("nfreq"), py::arg("ntime"), py::arg("mask_ranges"),
            py::arg("freq_range"), py::arg("warps_per_block") = 4,
            py::call_guard<py::gil_scoped_release>(),   // copies 'keep' to the GPU; pure C++
            "Create a GpuBadChannelMask.\n"
            "\n"
            "Args:\n"
            "    nbeams, nfreq, ntime: the array shape launch() will be given.\n"
            "    mask_ranges: sequence of (lo, hi) pairs in MHz, each with lo < hi, in any\n"
            "        order; overlaps are fine.\n"
            "    freq_range: (lo, hi) of the band in MHz; (400, 800) for CHIME.\n"
            "    warps_per_block: performance knob, 4, 8, 16 or 32; must not change the\n"
            "        result. See time_selected().\n"
            "\n"
            "Raises:\n"
            "    RuntimeError: on nbeams, nfreq or ntime < 1; a range with lo >= hi, or one\n"
            "        lying entirely outside the band or strictly covering it (to mask every\n"
            "        channel, pass the band itself); freq_range with lo >= hi; or an\n"
            "        unsupported warps_per_block.")

        .def_readonly("mask_ranges", &GpuBadChannelMask::mask_ranges)
        .def_readonly("freq_range", &GpuBadChannelMask::freq_range)
        .def_readonly("warps_per_block", &GpuBadChannelMask::warps_per_block)
        .def_readonly("nmasked", &GpuBadChannelMask::nmasked)
        .def_readonly("keep", &GpuBadChannelMask::keep)

        .def_static("time_selected", &GpuBadChannelMask::time_selected,
            py::call_guard<py::gil_scoped_release>(),
            "Run timing benchmarks at the production array size, for masks from one channel to\n"
            "all of them (called via 'python -m pirate_frb time --cfrb')")

        ;

    // GpuWeightUpsampler: Python injections in pirate_frb/chimefrb/ReferenceWeightUpsampler.py:
    //   - launch: converts stream=None to current cupy stream
    py::class_<GpuWeightUpsampler>(m, "GpuWeightUpsampler",
        "Pushes a low-resolution weight mask back up to full resolution. A port of\n"
        "rf_kernels::weight_upsampler, which rf_pipelines::wi_sub_pipeline runs after its\n"
        "sub-pipeline: every full-resolution weight whose (Df x Dt) cell has a low-resolution\n"
        "weight ``w_lo <= w_cutoff`` is set to +0.0, and every other weight is left\n"
        "bit-identical.\n"
        "\n"
        "The comparison is the old code's exactly: strict (a weight equal to the cutoff is\n"
        "masked), in float32 against ``float32(w_cutoff)``, and false for NaN, so a NaN\n"
        "low-resolution weight masks its cell. Denormal weights are not supported: they may\n"
        "compare as zero on the GPU.\n"
        "\n"
        "The kernel writes only the zeros and never reads the full-resolution weights, so its\n"
        "cost is one read of the low-resolution array plus the masked cells. There is no\n"
        "chunking rule, and no shape needs to be a multiple of anything.")

        .def(py::init<long, long, double, long>(),
            py::arg("Df"), py::arg("Dt"), py::arg("w_cutoff") = 0.0,
            py::arg("warps_per_block") = 32,
            "Create a GpuWeightUpsampler.\n"
            "\n"
            "Args:\n"
            "    Df: frequency upsampling factor, >= 1\n"
            "    Dt: time upsampling factor, >= 1\n"
            "    w_cutoff: a cell is kept iff its low-resolution weight exceeds this. The\n"
            "        production chain uses 0.\n"
            "    warps_per_block: performance knob, 4/8/16/32. Must not change the result.\n"
            "        See time_selected().\n"
            "\n"
            "Raises:\n"
            "    RuntimeError: on Df < 1, Dt < 1, w_cutoff < 0 or NaN, or an unsupported\n"
            "        warps_per_block.")

        .def_readonly("Df", &GpuWeightUpsampler::Df, "Frequency upsampling factor")
        .def_readonly("Dt", &GpuWeightUpsampler::Dt, "Time upsampling factor")
        .def_readonly("w_cutoff", &GpuWeightUpsampler::w_cutoff,
            "A cell is kept iff its low-resolution weight exceeds ``float32(w_cutoff)``")
        .def_readonly("warps_per_block", &GpuWeightUpsampler::warps_per_block,
            "Performance knob for the kernel (4, 8, 16 or 32)")

        .def_static("time_selected", &GpuWeightUpsampler::time_selected,
            py::call_guard<py::gil_scoped_release>(),
            "Run timing benchmarks at the production configuration, for masks from nothing to\n"
            "everything (called via 'python -m pirate_frb time --cfrb')")

        .def("launch",
            [](const GpuWeightUpsampler &self, Array<float> &w_hires,
               const Array<float> &w_lores, uintptr_t stream_ptr) {
                cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
                self.launch(w_hires, w_lores, stream);
            },
            py::arg("w_hires"), py::arg("w_lores"), py::arg("stream_ptr"),
            py::call_guard<py::gil_scoped_release>(),   // async launch; body is pure C++
            "GPU kernel launch (async, does not sync stream).\n"
            "\n"
            "Note the order: the array that is modified comes first, as in the old code. At\n"
            "(Df, Dt) = (1, 1) the two shapes agree, so a swapped call is not caught.\n"
            "\n"
            "Args:\n"
            "    w_hires: cupy float32 array, shape (B, F_lo*Df, T_lo*Dt), fully contiguous, on\n"
            "        GPU. MODIFIED IN PLACE: every weight in a masked cell becomes +0.0, and\n"
            "        every other weight is left bit-identical. Never read.\n"
            "    w_lores: cupy float32 array, shape (B, F_lo, T_lo), fully contiguous, on GPU,\n"
            "        with any B, F_lo, T_lo. Read only. Must not be the same array as w_hires.\n"
            "    stream_ptr: CUDA stream pointer (integer, e.g. from cupy stream.ptr)")
        ;
}


}  // namespace pirate
