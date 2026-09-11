// Python bindings for the chimefrb port (pirate_frb.chimefrb subpackage). C++ classes
// are defined in include/pirate/chimefrb/*.hpp; see pirate_pybind11.cpp for the main
// module.
//
// Method injections, if any, live in pirate_frb/chimefrb/<ClassName>.py:
//   - AssembledChunk: none
//   - GpuWiDownsampler: launch() converts stream=None to the current cupy stream
//   - GpuWrms: same, and lets the caller omit the scratch array
//   - GpuIntensityClipper: same
//   - GpuStdDevClipper: same

#define PY_ARRAY_UNIQUE_SYMBOL PyArray_API_pirate
#define NO_IMPORT_ARRAY  // Secondary file: don't call _import_array()
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <optional>
#include <pybind11/stl.h>
#include <ksgpu/pybind11.hpp>

#include "../include/pirate/chimefrb/AssembledChunk.hpp"
#include "../include/pirate/chimefrb/ClipperAxis.hpp"
#include "../include/pirate/chimefrb/ClipperBase.hpp"
#include "../include/pirate/chimefrb/IntensityClipper.hpp"
#include "../include/pirate/chimefrb/StdDevClipper.hpp"
#include "../include/pirate/chimefrb/WiDownsampler.hpp"
#include "../include/pirate/chimefrb/Wrms.hpp"
#include "../include/pirate/SlabAllocator.hpp"

using namespace std;
using namespace ksgpu;
using namespace pirate::chimefrb;
namespace py = pybind11;


namespace pirate {


// Helper for the two decode bindings: use the caller's 'out' array if there is one,
// otherwise allocate a fresh (nfreq, nt) float32 array.
static Array<float> _decode_dst(const AssembledChunk &self, optional<Array<float>> &out)
{
    if (out)
        return *out;
    return Array<float> ({self.nfreq(), self.nt()}, af_uhost);
}


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

    py::enum_<ClipperAxis>(m, "ClipperAxis",
        "Which axis a chimefrb clipper reduces along.\n"
        "\n"
        "The numeric values match rf_kernels::axis_type, and the AXIS_FREQ/AXIS_TIME/\n"
        "AXIS_NONE constants in pirate_frb.chimefrb: a spot-check driver casts an integer\n"
        "straight to the old enum, so the three must not drift apart.")
        .value("FREQ", ClipperAxis::FREQ,
            "One statistic per downsampled time sample, reducing over frequency")
        .value("TIME", ClipperAxis::TIME,
            "One statistic per downsampled frequency, reducing over time")
        .value("NONE", ClipperAxis::NONE,
            "One statistic per beam, reducing over the whole plane")
        ;

    // GpuClipperBase: bound without a constructor. It exists so that the attributes the
    // clippers share (geometry, axis, statistic parameters) are bound, and documented,
    // once; the clippers are bound as its subclasses.
    py::class_<GpuClipperBase>(m, "GpuClipperBase",
        "What the chimefrb clippers have in common: array geometry, axis, (Df, Dt), the\n"
        "per-row weighted mean and variance (downsample, transpose if axis == FREQ, then\n"
        "GpuWrms), argument checking, and the scratch layout. Not constructible on its own:\n"
        "GpuIntensityClipper and GpuStdDevClipper derive from it.\n"
        "\n"
        "CHUNKING: every clipper requires its array to hold exactly one nt_chunk. The old\n"
        "code's T = N*nt_chunk behaviour is implemented by the numpy references, not here.")

        .def_readonly("B", &GpuClipperBase::B, "Beams")
        .def_readonly("F", &GpuClipperBase::F, "Full-resolution frequency channels")
        .def_readonly("nt_chunk", &GpuClipperBase::nt_chunk,
            "Full-resolution time samples. The array must hold exactly one chunk.")
        .def_readonly("axis", &GpuClipperBase::axis, "The ClipperAxis being reduced")
        .def_readonly("Df", &GpuClipperBase::Df, "Frequency downsampling factor")
        .def_readonly("Dt", &GpuClipperBase::Dt, "Time downsampling factor")
        .def_readonly("niter", &GpuClipperBase::niter,
            "TOTAL passes of the statistic; 1 means no refinement (always 1 for\n"
            "GpuStdDevClipper)")
        .def_readonly("iter_sigma", &GpuClipperBase::iter_sigma,
            "The statistic's REFINEMENT threshold, in rms units; ignored at niter=1")
        .def_readonly("two_pass", &GpuClipperBase::two_pass,
            "Use the stabler two-pass first pass of the statistic")

        .def_readonly("F_ds", &GpuClipperBase::F_ds, "F // Df")
        .def_readonly("T_ds", &GpuClipperBase::T_ds, "nt_chunk // Dt")
        .def_readonly("wrms_L", &GpuClipperBase::wrms_L,
            "Samples per statistic row. Exposed so that a test can rebuild the statistic\n"
            "from GpuWiDownsampler and GpuWrms exactly, which is how the clippers are\n"
            "checked: the statistic comes from the GPU, and what follows it from numpy.")
        .def_readonly("wrms_R", &GpuClipperBase::wrms_R,
            "Statistic rows: B*F_ds (TIME), B*T_ds (FREQ), or B (NONE)")
        .def_readonly("scratch_nelts", &GpuClipperBase::scratch_nelts,
            "Number of float32 scratch elements launch() needs. Never zero.")
        ;

    // GpuIntensityClipper: Python injections in
    // pirate_frb/chimefrb/ReferenceIntensityClipper.py:
    //   - launch: converts stream=None to current cupy stream, allocates scratch=None
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
        "class requires the array to hold exactly ONE chunk, which is all the production\n"
        "RFI chain needs. Processing T = N*nt_chunk samples in one call would be a useful\n"
        "generalization and is deliberately left for later; ReferenceIntensityClipper does\n"
        "implement it, so the semantics are pinned down and tested.")

        .def(py::init<long, long, long, ClipperAxis, double, long, long, long, double, bool, long>(),
            py::arg("B"), py::arg("F"), py::arg("nt_chunk"), py::arg("axis"), py::arg("sigma"),
            py::arg("Df"), py::arg("Dt"), py::arg("niter"), py::arg("iter_sigma"),
            py::arg("two_pass"), py::arg("warps_per_block") = 16,
            "Create a GpuIntensityClipper.\n"
            "\n"
            "Args:\n"
            "    B, F, nt_chunk: the full-resolution array shape. All three are\n"
            "        constructor arguments, so the geometry is fixed at construction: F\n"
            "        and nt_chunk fix the statistic's row length, which GpuWrms needs at\n"
            "        construction, and B fixes the row count and the scratch size.\n"
            "    axis: a ClipperAxis.\n"
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
            "    RuntimeError: unless F is divisible by 32*Df and nt_chunk by 32*Dt; also\n"
            "        on B < 1, Df < 1, Dt < 1, niter < 1, sigma < 0, iter_sigma < 0, or an\n"
            "        unsupported warps_per_block.")

        // The shared attributes (B, F, ..., scratch_nelts) are bound on GpuClipperBase.
        .def_readonly("sigma", &GpuIntensityClipper::sigma,
            "FINAL clip threshold, in units of the row's rms (not iter_sigma)")
        .def_readonly("warps_per_block", &GpuIntensityClipper::warps_per_block,
            "Performance knob for the final clip kernel (4, 8, 16 or 32)")

        .def_static("time_selected", &GpuIntensityClipper::time_selected,
            py::call_guard<py::gil_scoped_release>(),
            "Run timing benchmarks, for the four configurations the old search's\n"
            "production RFI chain uses (called via 'python -m pirate_frb time --cfrb')")

        .def("launch",
            [](const GpuIntensityClipper &self, const Array<float> &intensity,
               Array<float> &weights, Array<float> &scratch, uintptr_t stream_ptr) {
                cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
                self.launch(intensity, weights, scratch, stream);
            },
            py::arg("intensity"), py::arg("weights"), py::arg("scratch"), py::arg("stream_ptr"),
            py::call_guard<py::gil_scoped_release>(),   // async launch; body is pure C++
            "GPU kernel launch (async, does not sync stream).\n"
            "\n"
            "All arrays are cupy float32 arrays, fully contiguous, and on GPU.\n"
            "\n"
            "Args:\n"
            "    intensity: shape (B, F, nt_chunk). Read only, never modified.\n"
            "    weights: shape (B, F, nt_chunk). MODIFIED IN PLACE: zeroed where the clip\n"
            "        fires, bit-identical everywhere else. Must be >= 0 on entry.\n"
            "    scratch: shape (scratch_nelts,). Contents ignored on entry, garbage on exit.\n"
            "    stream_ptr: CUDA stream pointer (integer, e.g. from cupy stream.ptr)\n"
            "\n"
            "Raises:\n"
            "    RuntimeError: on a shape mismatch. A time axis that is a multiple of\n"
            "        nt_chunk gets a message saying so: T = N*nt_chunk is a generalization\n"
            "        we have not implemented, not a caller error.")
        ;

    // GpuStdDevClipper: Python injections in pirate_frb/chimefrb/ReferenceStdDevClipper.py:
    //   - launch: converts stream=None to current cupy stream, allocates scratch=None
    py::class_<GpuStdDevClipper, GpuClipperBase>(m, "GpuStdDevClipper",
        "Zeroes whole channels (AXIS_TIME) or whole time samples (AXIS_FREQ) whose variance\n"
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
        "AXIS_NONE is not supported, as in the old code. Like every GpuClipperBase, the\n"
        "array must hold exactly one nt_chunk.")

        .def(py::init<long, long, long, ClipperAxis, double, long, long, bool, long>(),
            py::arg("B"), py::arg("F"), py::arg("nt_chunk"), py::arg("axis"), py::arg("sigma"),
            py::arg("Df"), py::arg("Dt"), py::arg("two_pass"), py::arg("warps_per_block") = 16,
            "Create a GpuStdDevClipper.\n"
            "\n"
            "Args:\n"
            "    B, F, nt_chunk: the full-resolution array shape, fixed at construction.\n"
            "    axis: ClipperAxis.TIME or ClipperAxis.FREQ.\n"
            "    sigma: the clip threshold, in units of the standard deviation OF THE\n"
            "        VARIANCES. Nothing can be clipped if sigma >= sqrt(n-1), for n usable\n"
            "        rows in a beam.\n"
            "    Df, Dt: downsampling factors for frequency and time.\n"
            "    two_pass: use the stabler two-pass form of the per-row variance.\n"
            "    warps_per_block: performance knob, 4/8/16/32, for the kernel that zeroes the\n"
            "        weights. Must not change the result. See time_selected().\n"
            "\n"
            "Raises:\n"
            "    RuntimeError: on axis NONE; unless F is divisible by 32*Df and nt_chunk by\n"
            "        32*Dt; and on B < 1, Df < 1, Dt < 1, sigma < 0, or an unsupported\n"
            "        warps_per_block.")

        // The shared attributes (B, F, ..., scratch_nelts) are bound on GpuClipperBase.
        .def_readonly("sigma", &GpuStdDevClipper::sigma,
            "Clip threshold, in units of the standard deviation of the variances")
        .def_readonly("warps_per_block", &GpuStdDevClipper::warps_per_block,
            "Performance knob for the kernel that zeroes the weights (4, 8, 16 or 32)")

        .def_static("time_selected", &GpuStdDevClipper::time_selected,
            py::call_guard<py::gil_scoped_release>(),
            "Run timing benchmarks, for the two configurations the old search's production\n"
            "RFI chain uses (called via 'python -m pirate_frb time --cfrb')")

        .def("launch",
            [](const GpuStdDevClipper &self, const Array<float> &intensity,
               Array<float> &weights, Array<float> &scratch, uintptr_t stream_ptr) {
                cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
                self.launch(intensity, weights, scratch, stream);
            },
            py::arg("intensity"), py::arg("weights"), py::arg("scratch"), py::arg("stream_ptr"),
            py::call_guard<py::gil_scoped_release>(),   // async launch; body is pure C++
            "GPU kernel launch (async, does not sync stream).\n"
            "\n"
            "All arrays are cupy float32 arrays, fully contiguous, and on GPU.\n"
            "\n"
            "Args:\n"
            "    intensity: shape (B, F, nt_chunk). Read only, never modified.\n"
            "    weights: shape (B, F, nt_chunk). MODIFIED IN PLACE: whole rows are zeroed\n"
            "        where the clip fires, bit-identical everywhere else. Must be >= 0.\n"
            "    scratch: shape (scratch_nelts,). Contents ignored on entry, garbage on exit.\n"
            "    stream_ptr: CUDA stream pointer (integer, e.g. from cupy stream.ptr)")
        ;
}


}  // namespace pirate
