// Python bindings for CASM beamformer (pirate_frb.casm subpackage).
// See pirate_pybind11.cu for the main module definition.

#define PY_ARRAY_UNIQUE_SYMBOL PyArray_API_pirate
#define NO_IMPORT_ARRAY  // Secondary file: don't call _import_array()
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <pybind11/stl.h>
#include <ksgpu/pybind11.hpp>

#include "../include/pirate/CasmBeamformer.hpp"

using namespace std;
using namespace ksgpu;
using namespace pirate;
namespace py = pybind11;


namespace pirate {

void register_casm_bindings(pybind11::module &m)
{
    // CasmBeamformer: GPU beamformer for CASM telescope
    // Note: Python injections in pirate_frb/casm/CasmBeamformer.py add a stream
    // argument to launch_beamformer(); the class docstring stays here (option 1
    // in notes/docstrings.md).
    py::class_<CasmBeamformer> (m, "CasmBeamformer",
        "GPU beamformer for CASM/CHORD-style arrays.\n\n"
        "Constructed once with time-independent parameters (beam locations, feed\n"
        "layout, frequencies); then launch_beamformer() is called on successive\n"
        "chunks of int4+4 electric-field data to produce downsampled per-beam\n"
        "intensities, applying per-feed complex weights (gains / masking).\n\n"
        "Array shapes: E-field (T, F, 2, 256), feed_weights (F, 2, 256, 2),\n"
        "output intensities (T/downsampling_factor, F, B). See launch_beamformer().")

        // Constructor with optional ew_feed_spacings argument (defaults to None/empty Array)
        .def(py::init([](const Array<float> &frequencies,
                         const Array<int> &feed_indices,
                         const Array<float> &beam_locations,
                         int downsampling_factor,
                         float ns_feed_spacing,
                         py::object ew_feed_spacings_obj) {
            // The .cast<>() is python API, so it must happen BEFORE the GIL release
            // below (a py::call_guard would be a bug here). The C++ constructor
            // (host precompute + cudaMalloc + blocking H2D copies) runs GIL-free.
            bool have_ew = !ew_feed_spacings_obj.is_none();
            Array<float> ew_feed_spacings;
            if (have_ew)
                ew_feed_spacings = ew_feed_spacings_obj.cast<Array<float>>();

            py::gil_scoped_release nogil;

            if (have_ew)
                return new CasmBeamformer(frequencies, feed_indices, beam_locations,
                                          downsampling_factor, ns_feed_spacing, ew_feed_spacings);
            return new CasmBeamformer(frequencies, feed_indices, beam_locations,
                                      downsampling_factor, ns_feed_spacing);
        }),
        py::arg("frequencies"),
        py::arg("feed_indices"),
        py::arg("beam_locations"),
        py::arg("downsampling_factor"),
        py::arg("ns_feed_spacing") = CasmBeamformer::default_ns_feed_spacing,
        py::arg("ew_feed_spacings") = py::none())

        // Internal launch_beamformer binding that accepts stream_ptr
        // Python wrapper in pirate_frb/casm/CasmBeamformer.py handles stream=None default
        .def("launch_beamformer",
             [](CasmBeamformer &self, const Array<uint8_t> &e_in, 
                const Array<float> &feed_weights, Array<float> &i_out,
                uintptr_t stream_ptr) {
                 cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
                 self.launch_beamformer(e_in, feed_weights, i_out, stream);
             },
             py::arg("e_in"), py::arg("feed_weights"), py::arg("i_out"), py::arg("stream_ptr"),
             py::call_guard<py::gil_scoped_release>())   // async launch, pure C++ body

        .def_static("get_max_beams", &CasmBeamformer::get_max_beams)
        .def_static("test_microkernels", &CasmBeamformer::test_microkernels,
             py::call_guard<py::gil_scoped_release>())
        .def_static("run_timings", &CasmBeamformer::run_timings, py::arg("ncu_hack"),
             py::call_guard<py::gil_scoped_release>())
    ;
}

}  // namespace pirate

