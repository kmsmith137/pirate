// Python bindings for GPU kernel classes (pirate_frb.kernels subpackage).
// See pirate_pybind11.cu for the main module definition.

#define PY_ARRAY_UNIQUE_SYMBOL PyArray_API_pirate
#define NO_IMPORT_ARRAY  // Secondary file: don't call _import_array()
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <pybind11/stl.h>
#include <ksgpu/pybind11.hpp>

#include "../include/pirate/CoalescedDdKernel2.hpp"
#include "../include/pirate/DedispersionKernel.hpp"
#include "../include/pirate/GpuDequantizationKernel.hpp"
#include "../include/pirate/LaggedDownsamplingKernel.hpp"
#include "../include/pirate/PeakFindingKernel.hpp"
#include "../include/pirate/ReferenceLagbuf.hpp"
#include "../include/pirate/ReferenceTree.hpp"
#include "../include/pirate/RingbufCopyKernel.hpp"
#include "../include/pirate/TreeGriddingKernel.hpp"

using namespace std;
using namespace ksgpu;
using namespace pirate;
namespace py = pybind11;


namespace pirate {

void register_kernel_bindings(pybind11::module &m)
{
    py::class_<CoalescedDdKernel2>(m, "CoalescedDdKernel2")
          .def_static("test_random", &CoalescedDdKernel2::test_random)
          .def_static("time_selected", &CoalescedDdKernel2::time_selected)
          .def_static("registry_size", &CoalescedDdKernel2::registry_size)
          .def_static("show_registry", &CoalescedDdKernel2::show_registry)
    ;

    py::class_<GpuDedispersionKernel>(m, "GpuDedispersionKernel")
          .def_static("test_random", &GpuDedispersionKernel::test_random)
          .def_static("time_selected", &GpuDedispersionKernel::time_selected)
          .def_static("registry_size", &GpuDedispersionKernel::registry_size)
          .def_static("show_registry", &GpuDedispersionKernel::show_registry)
    ;

    py::class_<GpuDequantizationKernel>(m, "GpuDequantizationKernel",
        "GPU kernel to convert int4 array to float32 or float16.\n\n"
        "Input: contiguous array of shape (nbeams, nfreq, ntime), dtype int4.\n"
        "Output: contiguous array of shape (nbeams, nfreq, ntime), dtype float32 or float16.\n\n"
        "The int4 values are interpreted as signed two's complement (-8 to +7).\n"
        "Nibble packing: low nibble = even index, high nibble = odd index.\n\n"
        "IMPORTANT: Since numpy/cupy don't support int4 dtype (dtypes must be at least 8 bits),\n"
        "the Python wrappers for apply_reference() and launch() accept a uint8 array of\n"
        "shape (nbeams, nfreq, ntime//2), which is reinterpreted as int4 with shape\n"
        "(nbeams, nfreq, ntime). The uint8 array must be fully contiguous.")
          .def(py::init<Dtype, long, long, long>(),
               py::arg("dtype"), py::arg("nbeams"), py::arg("nfreq"), py::arg("ntime"),
               "Create a GpuDequantizationKernel.\n\n"
               "Args:\n"
               "    dtype: Output dtype (must be float32 or float16)\n"
               "    nbeams: Number of beams\n"
               "    nfreq: Number of frequency channels\n"
               "    ntime: Number of time samples (must be divisible by 256)\n\n"
               "Raises:\n"
               "    RuntimeError: If dtype is invalid or ntime is not divisible by 256")
          .def_readonly("dtype", &GpuDequantizationKernel::dtype,
               "Output dtype (float32 or float16)")
          .def_readonly("nbeams", &GpuDequantizationKernel::nbeams,
               "Number of beams")
          .def_readonly("nfreq", &GpuDequantizationKernel::nfreq,
               "Number of frequency channels")
          .def_readonly("ntime", &GpuDequantizationKernel::ntime,
               "Number of time samples")
          .def_readonly("resource_tracker", &GpuDequantizationKernel::resource_tracker,
               "ResourceTracker for memory/bandwidth accounting")
          .def("apply_reference",
               [](const GpuDequantizationKernel &self, Array<float> &out, const Array<void> &in_uint8) {
                   Array<void> in_int4 = self.convert_uint8_to_int4(in_uint8);
                   self.apply_reference(out, in_int4);
               },
               py::arg("out"), py::arg("in_uint8"),
               "Reference implementation (CPU, always outputs float32).\n\n"
               "Args:\n"
               "    out: Output array, shape (nbeams, nfreq, ntime), dtype float32,\n"
               "         fully contiguous, on host\n"
               "    in_uint8: Input array, shape (nbeams, nfreq, ntime//2), dtype uint8,\n"
               "              fully contiguous, on host. This is reinterpreted as int4\n"
               "              with shape (nbeams, nfreq, ntime).\n\n"
               "Note: The input is passed as uint8 because numpy/cupy don't support int4\n"
               "(all dtypes must be at least 8 bits). Each uint8 element contains two\n"
               "int4 values: low nibble = even index, high nibble = odd index.")
          .def("launch",
               [](const GpuDequantizationKernel &self, Array<void> &out, const Array<void> &in_uint8, uintptr_t stream_ptr) {
                   Array<void> in_int4 = self.convert_uint8_to_int4(in_uint8);
                   cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
                   self.launch(out, in_int4, stream);
               },
               py::arg("out"), py::arg("in_uint8"), py::arg("stream_ptr"),
               "GPU kernel launch (async, does not sync stream).\n\n"
               "Args:\n"
               "    out: Output array, shape (nbeams, nfreq, ntime), dtype matches\n"
               "         kernel's dtype (float32 or float16), fully contiguous, on GPU\n"
               "    in_uint8: Input array, shape (nbeams, nfreq, ntime//2), dtype uint8,\n"
               "              fully contiguous, on GPU. This is reinterpreted as int4\n"
               "              with shape (nbeams, nfreq, ntime).\n"
               "    stream_ptr: CUDA stream pointer (integer, e.g. from cupy stream.ptr)\n\n"
               "Note: The input is passed as uint8 because numpy/cupy don't support int4\n"
               "(all dtypes must be at least 8 bits). Each uint8 element contains two\n"
               "int4 values: low nibble = even index, high nibble = odd index.")
          .def_static("test_random", &GpuDequantizationKernel::test_random,
               "Run randomized tests (called via 'python -m pirate_frb test --gdqk')")
          .def_static("time_selected", &GpuDequantizationKernel::time_selected,
               "Run timing benchmarks")
    ;

    py::class_<GpuLaggedDownsamplingKernel>(m, "GpuLaggedDownsamplingKernel")
          .def_static("test_random", &GpuLaggedDownsamplingKernel::test_random)
          .def_static("time_selected", &GpuLaggedDownsamplingKernel::time_selected)
    ;

    py::class_<GpuPeakFindingKernel>(m, "GpuPeakFindingKernel")
          .def_static("test_random", &GpuPeakFindingKernel::test_random, py::arg("short_circuit") = false)
          .def_static("registry_size", &GpuPeakFindingKernel::registry_size)
          .def_static("show_registry", &GpuPeakFindingKernel::show_registry)
    ;

    py::class_<GpuRingbufCopyKernel>(m, "GpuRingbufCopyKernel")
          .def_static("test_random", &GpuRingbufCopyKernel::test_random)
    ;

    py::class_<GpuTreeGriddingKernel>(m, "GpuTreeGriddingKernel")
          .def_static("test_random", &GpuTreeGriddingKernel::test_random)
          .def_static("time_selected", &GpuTreeGriddingKernel::time_selected)
    ;

    py::class_<PfOutputMicrokernel>(m, "PfOutputMicrokernel")
          .def_static("test_random", &PfOutputMicrokernel::test_random)
          .def_static("registry_size", &PfOutputMicrokernel::registry_size)
          .def_static("show_registry", &PfOutputMicrokernel::show_registry)
    ;

    py::class_<PfWeightReaderMicrokernel>(m, "PfWeightReaderMicrokernel")
          .def_static("test_random", &PfWeightReaderMicrokernel::test_random)
          .def_static("registry_size", &PfWeightReaderMicrokernel::registry_size)
          .def_static("show_registry", &PfWeightReaderMicrokernel::show_registry)
    ;

    py::class_<ReferenceLagbuf>(m, "ReferenceLagbuf")
          .def_static("test_random", &ReferenceLagbuf::test_random)
    ;

    py::class_<ReferenceTree>(m, "ReferenceTree")
          .def_static("test_basics", &ReferenceTree::test_basics)
          .def_static("test_subbands", &ReferenceTree::test_subbands)
    ;
}

}  // namespace pirate
