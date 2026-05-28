// Python bindings for the pirate_frb.utils subpackage.
// See pirate_pybind11.cpp for the top-level module definition and to
// trace how register_utils_bindings() is wired in.

#define PY_ARRAY_UNIQUE_SYMBOL PyArray_API_pirate
#define NO_IMPORT_ARRAY   // Secondary file: don't call _import_array()
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <cuda_runtime.h>

#include <pybind11/stl.h>
#include <ksgpu/pybind11.hpp>

#include "../include/pirate/ToyIPC.hpp"
#include "../include/pirate/utils.hpp"

using namespace std;
using namespace pirate;
namespace py = pybind11;


namespace pirate {

void register_utils_bindings(pybind11::module &m)
{
    // safe_memcpy_{h2g,g2h}_{sync,async}: host<->device cudaMemcpy* wrappers
    // that split at absolute cuda_host_register_chunk_size-aligned host
    // addresses. Use these whenever the host pointer COULD live in a
    // pirate hugepage-backed BumpAllocator -- the chunked cudaHostRegister
    // layout means an unsplit cudaMemcpyAsync that crosses a chunk seam
    // (and, empirically, even a 55-MiB intra-chunk copy in some Python
    // call paths) returns cudaErrorInvalidValue. See plans/python_h2g_chunking.md
    // and the doc-comment block at the top of include/pirate/utils.hpp.

    m.def("safe_memcpy_h2g_async",
          [](uintptr_t dst_ptr, uintptr_t src_ptr, long nbytes, uintptr_t stream_ptr) {
              auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
              safe_memcpy_h2g_async(reinterpret_cast<void *>(dst_ptr),
                                    reinterpret_cast<const void *>(src_ptr),
                                    nbytes, stream);
          },
          py::arg("dst_ptr"), py::arg("src_ptr"), py::arg("nbytes"), py::arg("stream_ptr"),
          "Host->device cudaMemcpyAsync that splits the host range at\n"
          "absolute cuda_host_register_chunk_size-aligned boundaries.");

    m.def("safe_memcpy_g2h_async",
          [](uintptr_t dst_ptr, uintptr_t src_ptr, long nbytes, uintptr_t stream_ptr) {
              auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
              safe_memcpy_g2h_async(reinterpret_cast<void *>(dst_ptr),
                                    reinterpret_cast<const void *>(src_ptr),
                                    nbytes, stream);
          },
          py::arg("dst_ptr"), py::arg("src_ptr"), py::arg("nbytes"), py::arg("stream_ptr"),
          "Device->host cudaMemcpyAsync, same splitting as safe_memcpy_h2g_async.");

    m.def("safe_memcpy_h2g_sync",
          [](uintptr_t dst_ptr, uintptr_t src_ptr, long nbytes) {
              safe_memcpy_h2g_sync(reinterpret_cast<void *>(dst_ptr),
                                   reinterpret_cast<const void *>(src_ptr),
                                   nbytes);
          },
          py::arg("dst_ptr"), py::arg("src_ptr"), py::arg("nbytes"),
          "Synchronous host->device cudaMemcpy with chunk-boundary splitting.");

    m.def("safe_memcpy_g2h_sync",
          [](uintptr_t dst_ptr, uintptr_t src_ptr, long nbytes) {
              safe_memcpy_g2h_sync(reinterpret_cast<void *>(dst_ptr),
                                   reinterpret_cast<const void *>(src_ptr),
                                   nbytes);
          },
          py::arg("dst_ptr"), py::arg("src_ptr"), py::arg("nbytes"),
          "Synchronous device->host cudaMemcpy with chunk-boundary splitting.");

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
