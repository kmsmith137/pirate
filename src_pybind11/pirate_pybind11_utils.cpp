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
#include "../include/pirate/avx2_utils.hpp"   // test/time_avx2_simulate_4bit_noise

using namespace std;
using namespace pirate;
namespace py = pybind11;


namespace pirate {

void register_utils_bindings(pybind11::module &m)
{
    // atomic_print(): the output funnel shared by C++ and python. Python code
    // calls this (C++ code uses class AtomicPrint, which routes here), so that
    // both languages serialize on the same process-global mutex and every
    // line is emitted with a single write(2). See include/pirate/utils.hpp.
    //
    // The GIL is released before we block on the mutex, so a slow write can
    // never stall the interpreter; the str -> std::string conversion happens
    // during argument parsing, before the call guard takes effect. C++ threads
    // never touch the GIL in this path, so there is no lock-order cycle.
    m.def("atomic_print",
          [](const string &s, int fd) { atomic_print(s, fd); },
          py::arg("s"), py::arg("fd") = 1,
          py::call_guard<py::gil_scoped_release>(),
          "Emit 's' as one atomic line (appending '\\n' if absent) on file "
          "descriptor 'fd' (1=stdout, 2=stderr). Thread- and process-safe: "
          "concurrent writers cannot interleave mid-line. Empty 's' is a "
          "no-op; pass '\\n' for a blank line.");

    // Concurrency smoke test for the above (see 'python -m pirate_frb test --aout').
    m.def("test_atomic_print", &test_atomic_print,
          py::arg("fd"), py::arg("nthreads"), py::arg("nlines"),
          py::call_guard<py::gil_scoped_release>());

    // avx2_simulate_4bit_noise() test + timing (see 'python -m pirate_frb test/time --sim').
    m.def("test_avx2_simulate_4bit_noise", &test_avx2_simulate_4bit_noise, py::call_guard<py::gil_scoped_release>());
    m.def("time_avx2_simulate_4bit_noise", &time_avx2_simulate_4bit_noise, py::arg("nthreads"), py::call_guard<py::gil_scoped_release>());

    // safe_memcpy_{h2g,g2h}_{sync,async}: host<->device cudaMemcpy* wrappers
    // that split at absolute cuda_host_register_chunk_size-aligned host
    // addresses. Use these whenever the host pointer COULD live in a
    // pirate hugepage-backed BumpAllocator -- the chunked cudaHostRegister
    // layout means an unsplit cudaMemcpyAsync that crosses a chunk seam
    // (and, empirically, even a 55-MiB intra-chunk copy in some Python
    // call paths) returns cudaErrorInvalidValue. See the doc-comment block
    // at the top of include/pirate/utils.hpp.

    m.def("safe_memcpy_h2g_async",
          [](uintptr_t dst_ptr, uintptr_t src_ptr, long nbytes, uintptr_t stream_ptr) {
              auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
              safe_memcpy_h2g_async(reinterpret_cast<void *>(dst_ptr),
                                    reinterpret_cast<const void *>(src_ptr),
                                    nbytes, stream);
          },
          py::arg("dst_ptr"), py::arg("src_ptr"), py::arg("nbytes"), py::arg("stream_ptr"),
          py::call_guard<py::gil_scoped_release>(),
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
          py::call_guard<py::gil_scoped_release>(),
          "Device->host cudaMemcpyAsync, same splitting as safe_memcpy_h2g_async.");

    m.def("safe_memcpy_h2g_sync",
          [](uintptr_t dst_ptr, uintptr_t src_ptr, long nbytes) {
              safe_memcpy_h2g_sync(reinterpret_cast<void *>(dst_ptr),
                                   reinterpret_cast<const void *>(src_ptr),
                                   nbytes);
          },
          py::arg("dst_ptr"), py::arg("src_ptr"), py::arg("nbytes"),
          py::call_guard<py::gil_scoped_release>(),
          "Synchronous host->device cudaMemcpy with chunk-boundary splitting.");

    m.def("safe_memcpy_g2h_sync",
          [](uintptr_t dst_ptr, uintptr_t src_ptr, long nbytes) {
              safe_memcpy_g2h_sync(reinterpret_cast<void *>(dst_ptr),
                                   reinterpret_cast<const void *>(src_ptr),
                                   nbytes);
          },
          py::arg("dst_ptr"), py::arg("src_ptr"), py::arg("nbytes"),
          py::call_guard<py::gil_scoped_release>(),
          "Synchronous device->host cudaMemcpy with chunk-boundary splitting.");
}

}   // namespace pirate
