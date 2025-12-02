// FIXME split into multiple source files soon (single-core compile time is ~15 seconds).

// For an explanation of PY_ARRAY_UNIQUE_SYMBOL, see comments in ksgpu/src_pybind11/ksgpu_pybind11.cu.
#define PY_ARRAY_UNIQUE_SYMBOL PyArray_API_pirate
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

// Needed in order to wrap methods with STL arguments (e.g. const vector<int> &vcpu_list).
#include <pybind11/stl.h>

#include <ksgpu/pybind11.hpp>

#include "../include/pirate/CasmBeamformer.hpp"
#include "../include/pirate/Dedisperser.hpp"
#include "../include/pirate/DedispersionKernel.hpp"
#include "../include/pirate/FakeCorrelator.hpp"
#include "../include/pirate/FakeServer.hpp"
#include "../include/pirate/LaggedDownsamplingKernel.hpp"
#include "../include/pirate/PeakFindingKernel.hpp"
#include "../include/pirate/RingbufCopyKernel.hpp"
#include "../include/pirate/TreeGriddingKernel.hpp"
#include "../include/pirate/tests.hpp"
#include "../include/pirate/timing.hpp"

#include "../include/pirate/loose_ends/tests.hpp"
#include "../include/pirate/loose_ends/timing.hpp"

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
    
    py::class_<CasmBeamformer> (m, "CasmBeamformer")
        
        // FIXME the constructor syntax I wanted was:
        //
        //    CasmBeamformer(
        //      const Array<float> &frequencies,     // shape (F,)
        //      const Array<int> &feed_indices,      // shape (256,2)
        //      const Array<float> &beam_locations,  // shape (B,2)
        //      int downsampling_factor,
        //      float ns_feed_spacing = default_ns_feed_spacing,
        //      const Array<float> &ew_feed_spacings = Array<float>()
        //   )
        //
        // but I got an import-time error when I tried to specify a default
        // argument for 'ew_feed_spacings'. For this reason, I ended up wrapping
        // two constructors (with and without the 'ew_feed_spacings' arg).
        //
        // FIXME CasmBeamformer should have docstrings (low-priority since
        // pybind11 interface is not "vendorized").
        
        .def(py::init<const Array<float> &, const Array<int> &, const Array<float> &, int, float>(),
             py::arg("frequencies"), py::arg("feed_indices"),
             py::arg("beam_locations"), py::arg("downsampling_factor"),
             py::arg("ns_feed_spacing") = CasmBeamformer::default_ns_feed_spacing)
        
        .def(py::init<const Array<float> &, const Array<int> &, const Array<float> &, int, float, const Array<float> &>(),
             py::arg("frequencies"), py::arg("feed_indices"),
             py::arg("beam_locations"), py::arg("downsampling_factor"),
             py::arg("ns_feed_spacing"),
             py::arg("ew_feed_spacings"))

        // FIXME figure out how to python-wrap 'stream' argument.
        // (For now, the python interface always uses the default stream!
        // Currently, we only use the python interface for testing, so this is okay.)
        
        .def("launch_beamformer",
             [](CasmBeamformer &self, const Array<uint8_t> &e_in, const Array<float> &feed_weights, Array<float> &i_out) {
                 self.launch_beamformer(e_in, feed_weights, i_out, nullptr);    // stream=nullptr
             }, py::arg("e_in"), py::arg("feed_weights"), py::arg("i_out"))

        .def_static("get_max_beams", &CasmBeamformer::get_max_beams)
        .def_static("test_microkernels", &CasmBeamformer::test_microkernels)
        .def_static("run_timings", &CasmBeamformer::run_timings, py::arg("ncu_hack"))
    ;
    
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

    py::class_<GpuDedispersionKernel>(m, "GpuDedispersionKernel")
          .def_static("test", &GpuDedispersionKernel::test)
          .def_static("registry_size", &GpuDedispersionKernel::registry_size)
          .def_static("show_registry", &GpuDedispersionKernel::show_registry)
    ;

    py::class_<GpuDedisperser>(m, "GpuDedisperser")
          .def_static("test", &GpuDedisperser::test)
    ;

    py::class_<GpuLaggedDownsamplingKernel>(m, "GpuLaggedDownsamplingKernel")
          .def_static("test", &GpuLaggedDownsamplingKernel::test)
    ;

    py::class_<GpuPeakFindingKernel>(m, "GpuPeakFindingKernel")
          .def_static("test", &GpuPeakFindingKernel::test, py::arg("short_circuit") = false)
          .def_static("registry_size", &GpuPeakFindingKernel::registry_size)
          .def_static("show_registry", &GpuPeakFindingKernel::show_registry)
    ;

    py::class_<GpuRingbufCopyKernel>(m, "GpuRingbufCopyKernel")
          .def_static("test", &GpuRingbufCopyKernel::test)
    ;

    py::class_<GpuTreeGriddingKernel>(m, "GpuTreeGriddingKernel")
          .def_static("test", &GpuTreeGriddingKernel::test)
    ;

    py::class_<ReferenceDedisperserBase>(m, "ReferenceDedisperser")
          .def_static("test_dedispersion_basics", &ReferenceDedisperserBase::test_dedispersion_basics)
    ;

    py::class_<PfWeightReaderMicrokernel>(m, "PfWeightReaderMicrokernel")
          .def_static("test", &PfWeightReaderMicrokernel::test)
          .def_static("registry_size", &PfWeightReaderMicrokernel::registry_size)
          .def_static("show_registry", &PfWeightReaderMicrokernel::show_registry)
    ;

    py::class_<PfOutputMicrokernel>(m, "PfOutputMicrokernel")
          .def_static("test", &PfOutputMicrokernel::test)
          .def_static("registry_size", &PfOutputMicrokernel::registry_size)
          .def_static("show_registry", &PfOutputMicrokernel::show_registry)
    ;

    m.def("time_cpu_downsample", &time_cpu_downsample, py::arg("nthreads"));
    m.def("time_gpu_dedispersion_kernels", &time_gpu_dedispersion_kernels);
    m.def("time_gpu_downsample", &time_gpu_downsample);
    m.def("time_gpu_lagged_downsampling_kernels", &time_gpu_lagged_downsampling_kernels);
    m.def("time_gpu_transpose", &time_gpu_transpose);
    
    // "Zombie" tests (code that I wrote during protoyping that may never get used)
    m.def("test_avx2_m64_outbuf", &test_avx2_m64_outbuf);
    m.def("test_cpu_downsampler", &test_cpu_downsampler);
    m.def("test_gpu_downsample", &test_gpu_downsample);
    m.def("test_gpu_transpose", &test_gpu_transpose);
    m.def("test_gpu_reduce2", &test_gpu_reduce2);

}
