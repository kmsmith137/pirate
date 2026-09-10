// Python bindings for reading old CHIME FRB data files (pirate_frb.chimefrb subpackage).
// C++ classes are defined in include/pirate/chimefrb/AssembledChunk.hpp +
// src_lib/chimefrb/AssembledChunk.cpp; see pirate_pybind11.cpp for the main module.
// (AssembledChunk has no method injections.)

#define PY_ARRAY_UNIQUE_SYMBOL PyArray_API_pirate
#define NO_IMPORT_ARRAY  // Secondary file: don't call _import_array()
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <optional>
#include <pybind11/stl.h>
#include <ksgpu/pybind11.hpp>

#include "../include/pirate/chimefrb/AssembledChunk.hpp"
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
}


}  // namespace pirate
