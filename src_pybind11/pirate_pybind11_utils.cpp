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
}

}   // namespace pirate
