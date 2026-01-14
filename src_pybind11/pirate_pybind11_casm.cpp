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
    // Note: Python method injections in pirate_frb/pybind11_injections.py add stream argument support
    py::class_<CasmBeamformer> (m, "CasmBeamformer")
        
        // Constructor with optional ew_feed_spacings argument (defaults to None/empty Array)
        .def(py::init([](const Array<float> &frequencies,
                         const Array<int> &feed_indices,
                         const Array<float> &beam_locations,
                         int downsampling_factor,
                         float ns_feed_spacing,
                         py::object ew_feed_spacings_obj) {
            if (ew_feed_spacings_obj.is_none()) {
                return new CasmBeamformer(frequencies, feed_indices, beam_locations,
                                          downsampling_factor, ns_feed_spacing);
            } else {
                Array<float> ew_feed_spacings = ew_feed_spacings_obj.cast<Array<float>>();
                return new CasmBeamformer(frequencies, feed_indices, beam_locations,
                                          downsampling_factor, ns_feed_spacing, ew_feed_spacings);
            }
        }),
        py::arg("frequencies"),
        py::arg("feed_indices"),
        py::arg("beam_locations"),
        py::arg("downsampling_factor"),
        py::arg("ns_feed_spacing") = CasmBeamformer::default_ns_feed_spacing,
        py::arg("ew_feed_spacings") = py::none())

        // Internal launch_beamformer binding that accepts stream_ptr
        // Python wrapper in pybind11_injections.py handles stream=None default
        .def("launch_beamformer",
             [](CasmBeamformer &self, const Array<uint8_t> &e_in, 
                const Array<float> &feed_weights, Array<float> &i_out,
                uintptr_t stream_ptr) {
                 cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
                 self.launch_beamformer(e_in, feed_weights, i_out, stream);
             },
             py::arg("e_in"), py::arg("feed_weights"), py::arg("i_out"), py::arg("stream_ptr"))

        .def_static("get_max_beams", &CasmBeamformer::get_max_beams)
        .def_static("test_microkernels", &CasmBeamformer::test_microkernels)
        .def_static("run_timings", &CasmBeamformer::run_timings, py::arg("ncu_hack"))
    ;
}

}  // namespace pirate

