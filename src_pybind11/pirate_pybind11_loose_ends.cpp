// Python bindings for "loose ends" functions (pirate_frb.loose_ends subpackage).
// See pirate_pybind11.cu for the main module definition.

#define PY_ARRAY_UNIQUE_SYMBOL PyArray_API_pirate
#define NO_IMPORT_ARRAY  // Secondary file: don't call _import_array()
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <pybind11/stl.h>
#include <ksgpu/pybind11.hpp>

#include "../include/pirate/loose_ends/tests.hpp"
#include "../include/pirate/loose_ends/timing.hpp"
#include "../include/pirate/utils.hpp"   // revisit_512gb_inner

using namespace std;
using namespace ksgpu;
using namespace pirate;
namespace py = pybind11;

// Declared extern here, defined in src_lib/scratch.cu.
namespace pirate { extern void scratch(); }


namespace pirate {

void register_loose_ends_bindings(pybind11::module &m)
{
    // Timing functions
    m.def("time_cpu_downsample", &time_cpu_downsample, py::arg("nthreads"), py::call_guard<py::gil_scoped_release>());
    m.def("time_gpu_downsample", &time_gpu_downsample, py::call_guard<py::gil_scoped_release>());
    m.def("time_gpu_transpose", &time_gpu_transpose, py::call_guard<py::gil_scoped_release>());
    
    // "Zombie" test functions (code written during protoyping that may never get used)
    m.def("test_avx2_m64_outbuf", &test_avx2_m64_outbuf, py::call_guard<py::gil_scoped_release>());
    m.def("test_cpu_downsampler", &test_cpu_downsampler, py::call_guard<py::gil_scoped_release>());
    m.def("test_gpu_downsample", &test_gpu_downsample, py::call_guard<py::gil_scoped_release>());
    m.def("test_gpu_transpose", &test_gpu_transpose, py::call_guard<py::gil_scoped_release>());
    m.def("test_gpu_reduce2", &test_gpu_reduce2, py::call_guard<py::gil_scoped_release>());

    // Called by 'python -m pirate_frb scratch'. Defined in src_lib/utils.cu.
    m.def("scratch", &scratch);

    // Called by 'python -m pirate_frb revisit_512gb'. Defined in src_lib/utils.cpp.
    m.def("revisit_512gb_inner", &revisit_512gb_inner,
          py::arg("nbytes"), py::arg("use_hugepages"),
          py::call_guard<py::gil_scoped_release>());   // minutes: mmap/prefault + cudaHostRegister
}

}  // namespace pirate

