// For an explanation of PY_ARRAY_UNIQUE_SYMBOL, see comments in ksgpu/src_pybind11/ksgpu_pybind11.cu.
#define PY_ARRAY_UNIQUE_SYMBOL PyArray_API_pirate

#include <iostream>
#include <ksgpu/pybind11.hpp>
#include "../include/pirate/internals/utils.hpp"  // dedisperse_non_incremental()

using namespace std;
using namespace ksgpu;
namespace py = pybind11;


PYBIND11_MODULE(pirate_pybind11, m)  // extension module gets compiled to pirate_pybind11.so
{
    m.doc() = "pirate: Perimeter Institute RAdio Transient Engine";

    // Note: looks like _import_array() will fail if different numpy versions are
    // found at compile-time versus runtime.

    if (_import_array() < 0) {
	PyErr_Print();
	PyErr_SetString(PyExc_ImportError, "pirate: numpy.core.multiarray failed to import");
	return;
    }

    m.def("dedisperse_non_incremental",
	  &pirate::dedisperse_non_incremental,
	  "This function is just included as a placeholder -- sensible python API coming soon",
	  py::arg("arr"));
}
