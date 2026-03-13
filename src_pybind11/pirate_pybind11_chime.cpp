// Python bindings for CHIME beamformer (pirate_frb.chime subpackage).
// See pirate_pybind11.cu for the main module definition.

#define PY_ARRAY_UNIQUE_SYMBOL PyArray_API_pirate
#define NO_IMPORT_ARRAY  // Secondary file: don't call _import_array()
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <pybind11/stl.h>
#include <ksgpu/pybind11.hpp>

#include "../include/pirate/ChimeBeamformer.hpp"

using namespace std;
using namespace ksgpu;
using namespace pirate;
namespace py = pybind11;


namespace pirate {

void register_chime_bindings(pybind11::module &m)
{
    m.def("time_chime_frb_upchan", &time_chime_frb_upchan,
          "Run timing benchmark for the CHIME FRB upchannelization kernel.");
}

}  // namespace pirate
