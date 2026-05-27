// Python bindings for the pirate_frb.utils subpackage.
// See pirate_pybind11.cpp for the top-level module definition and to
// trace how register_utils_bindings() is wired in.

#define PY_ARRAY_UNIQUE_SYMBOL PyArray_API_pirate
#define NO_IMPORT_ARRAY   // Secondary file: don't call _import_array()
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <pybind11/stl.h>
#include <ksgpu/pybind11.hpp>

#include "../include/pirate/ToyIPC.hpp"

using namespace std;
using namespace pirate;
namespace py = pybind11;


namespace pirate {

void register_utils_bindings(pybind11::module &m)
{
    // ToyIPC: pybind11_injections.py has no entries for this class.
    // Note: ToyIPC::create() is used internally, but appears as a
    // constructor to Python (matching FrbServer / GpuDedisperser).
    py::class_<ToyIPC, std::shared_ptr<ToyIPC>>(m, "ToyIPC",
        "Toy CUDA-IPC producer.\n\n"
        "Allocates a shape (5,2) float32 ring buffer on the GPU, opens a\n"
        "gRPC bidi stream to the given server_address, and sends the\n"
        "cudaIpcMemHandle_t as the first message on the stream. A worker\n"
        "thread reads CONSUMED notifications from the server and updates\n"
        "rb_start. The one python-callable entry point is send().\n\n"
        "Pairs with pirate_frb.utils.ToyGrouper.")
        .def(py::init(&ToyIPC::create),
             py::arg("server_address"),
             py::arg("cuda_device_id"),
             "Args:\n"
             "    server_address: 'ip:port' string the ToyGrouper is listening on\n"
             "        (e.g. '127.0.0.1:6817')\n"
             "    cuda_device_id: CUDA device index. The ToyGrouper must use the\n"
             "        same device, or cudaIpcOpenMemHandle will refuse the handle.")
        .def("send", &ToyIPC::send,
             py::call_guard<py::gil_scoped_release>(),
             "Produce one ring-buffer slot.\n\n"
             "Blocks until at least one slot is free, then fills slot (rb_end%5)\n"
             "with two random floats (also printed to stdout), sends a PRODUCED\n"
             "notification, and bumps rb_end. Releases the GIL while blocked.")
        .def("stop",
             [](ToyIPC &self) { self.stop(); },
             "Stop the worker thread and cancel the gRPC stream. Idempotent.")
        .def_property_readonly("rb_start", &ToyIPC::get_rb_start,
             "Snapshot of rb_start = (last-consumed slot) + 1.")
        .def_property_readonly("rb_end", &ToyIPC::get_rb_end,
             "Snapshot of rb_end = (last-produced slot) + 1.")
    ;
}

}   // namespace pirate
