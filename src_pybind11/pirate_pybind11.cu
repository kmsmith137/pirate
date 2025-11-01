// FIXME split into multiple source files soon (single-core compile time is ~15 seconds).

// For an explanation of PY_ARRAY_UNIQUE_SYMBOL, see comments in ksgpu/src_pybind11/ksgpu_pybind11.cu.
#define PY_ARRAY_UNIQUE_SYMBOL PyArray_API_pirate
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

// Needed in order to wrap methods with STL arguments (e.g. const vector<int> &vcpu_list).
#include <pybind11/stl.h>

#include <ksgpu/pybind11.hpp>

#include "../include/pirate/CasmBeamformer.hpp"
#include "../include/pirate/FakeCorrelator.hpp"
#include "../include/pirate/FakeServer.hpp"
#include "../include/pirate/tests.hpp"
#include "../include/pirate/timing.hpp"
#include "../include/pirate/utils.hpp"  // show_kernels()

using namespace std;
using namespace ksgpu;
using namespace pirate;
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

    
    py::class_<FakeCorrelator>(m, "FakeCorrelator")
	.def(py::init<long, bool, bool, bool>(),
	     py::arg("send_bufsize"), py::arg("use_zerocopy"), py::arg("use_mmap"), py::arg("use_hugepages"))

	.def("add_endpoint", &FakeCorrelator::add_endpoint,
	     py::arg("ip_addr"), py::arg("num_tcp_connections"), py::arg("total_gpbs"), py::arg("vcpu_list"))

	.def("run", &FakeCorrelator::run)  // no args
    ;
    
    
    py::class_<FakeServer>(m, "FakeServer")
	.def(py::init<const std::string &, bool>(),
	     py::arg("server_name"), py::arg("use_hugepages"))

	.def("add_tcp_receiver", &FakeServer::add_tcp_receiver,
	     py::arg("ip_addr"), py::arg("num_tcp_connections"), py::arg("recv_bufsize"),
	     py::arg("use_epoll"), py::arg("vcpu_list"), py::arg("cpu"), py::arg("inic"))

	.def("add_chime_dedisperser", &FakeServer::add_chime_dedisperser,
	     py::arg("device"), py::arg("beams_per_gpu"), py::arg("num_active_batches"),
	     py::arg("beams_per_batch"), py::arg("use_copy_engine"), py::arg("vcpu_list"),
	     py::arg("cpu"))
	
	.def("add_memcpy_thread", &FakeServer::add_memcpy_thread,
	     py::arg("src_device"), py::arg("dst_device"), py::arg("blocksize"),
	     py::arg("use_copy_engine"), py::arg("vcpu_list"), py::arg("cpu"))
	
	.def("add_ssd_writer", &FakeServer::add_ssd_writer,
	     py::arg("root_dir"), py::arg("nbytes_per_file"), py::arg("vcpu_list"),
	     py::arg("cpu"), py::arg("issd"))

	.def("add_downsampling_thread", &FakeServer::add_downsampling_thread,
	     py::arg("src_bit_depth"), py::arg("src_nelts"), py::arg("vcpu_list"),
	     py::arg("cpu"))

	 // Called by python code, to control server.
	.def("abort", &FakeServer::abort, py::arg("abort_msg"))
	.def("join_threads", &FakeServer::join_threads)
	.def("show_stats", &FakeServer::show_stats)
	.def("start", &FakeServer::start)
	.def("stop", &FakeServer::stop)
    ;

    m.def("test_casm_microkernels", &test_casm_microkernels);
    m.def("test_dedispersion_basics", &test_dedispersion_basics);
    m.def("test_gpu_lagged_downsampling_kernel", &test_gpu_lagged_downsampling_kernel);
    m.def("test_gpu_dedispersion_kernels", &test_gpu_dedispersion_kernels);
    m.def("test_gpu_peak_finding_kernel", &test_gpu_peak_finding_kernel, py::arg("reduce_only"));
    m.def("test_gpu_ringbuf_copy_kernel", &test_gpu_ringbuf_copy_kernel);
    m.def("test_gpu_tree_gridding_kernel", &test_gpu_tree_gridding_kernel);
    m.def("test_dedisperser", static_cast<void (*)()> (&test_dedisperser));

    m.def("time_cpu_downsample", &time_cpu_downsample, py::arg("nthreads"));
    m.def("time_gpu_dedispersion_kernels", &time_gpu_dedispersion_kernels);
    m.def("time_gpu_downsample", &time_gpu_downsample);
    m.def("time_gpu_lagged_downsampling_kernels", &time_gpu_lagged_downsampling_kernels);
    m.def("time_gpu_peak_finding_kernels", &time_gpu_peak_finding_kernels);
    m.def("time_gpu_transpose", &time_gpu_transpose);

    m.def("show_kernels", &show_kernels);
}
