// For an explanation of PY_ARRAY_UNIQUE_SYMBOL, see comments in ksgpu/src_pybind11/ksgpu_pybind11.cu.
#define PY_ARRAY_UNIQUE_SYMBOL PyArray_API_pirate
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

// Needed in order to wrap methods with STL arguments (e.g. const vector<int> &vcpu_list).
#include <pybind11/stl.h>

#include <ksgpu/pybind11.hpp>

#include "../include/pirate/FakeCorrelator.hpp"
#include "../include/pirate/FakeServer.hpp"


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

	.def("add_receiver", &FakeServer::add_receiver,
	     py::arg("ip_addr"), py::arg("num_tcp_connections"), py::arg("recv_bufsize"),
	     py::arg("use_epoll"), py::arg("network_sync_cadence"), py::arg("vcpu_list"))
	
	.def("add_memcpy_worker", &FakeServer::add_memcpy_worker,
	     py::arg("src_device"), py::arg("dst_device"), py::arg("nbytes_per_iteration"),
	     py::arg("blocksize"), py::arg("vcpu_list"))

	.def("add_downsampling_worker", &FakeServer::add_downsampling_worker,
	     py::arg("src_bit_depth"), py::arg("src_nelts"), py::arg("vcpu_list"))

	.def("add_sleepy_worker", &FakeServer::add_sleepy_worker, py::arg("sleep_usec"))

	.def("run", &FakeServer::run, py::arg("num_iterations"))
    ;
}
