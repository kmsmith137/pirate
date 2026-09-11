// Python bindings for the chimefrb port (pirate_frb.chimefrb subpackage). C++ classes
// are defined in include/pirate/chimefrb/*.hpp; see pirate_pybind11.cpp for the main
// module.
//
// Method injections, if any, live in pirate_frb/chimefrb/<ClassName>.py:
//   - AssembledChunk: none
//   - GpuWiDownsampler: launch() converts stream=None to the current cupy stream
//   - GpuWrms: same, and lets the caller omit the scratch array

#define PY_ARRAY_UNIQUE_SYMBOL PyArray_API_pirate
#define NO_IMPORT_ARRAY  // Secondary file: don't call _import_array()
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <optional>
#include <pybind11/stl.h>
#include <ksgpu/pybind11.hpp>

#include "../include/pirate/chimefrb/AssembledChunk.hpp"
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
}


}  // namespace pirate
