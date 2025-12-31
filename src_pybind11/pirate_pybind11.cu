// For an explanation of PY_ARRAY_UNIQUE_SYMBOL, see comments in ksgpu/src_pybind11/ksgpu_pybind11.cu.
#define PY_ARRAY_UNIQUE_SYMBOL PyArray_API_pirate
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

// Needed in order to wrap methods with STL arguments (e.g. const vector<int> &vcpu_list).
#include <pybind11/stl.h>

#include <ksgpu/pybind11.hpp>

#include "../include/pirate/CasmBeamformer.hpp"
#include "../include/pirate/CoalescedDdKernel2.hpp"
#include "../include/pirate/ResourceTracker.hpp"
#include "../include/pirate/Dedisperser.hpp"
#include "../include/pirate/DedispersionConfig.hpp"
#include "../include/pirate/DedispersionPlan.hpp"
#include "../include/pirate/DedispersionKernel.hpp"
#include "../include/pirate/FakeCorrelator.hpp"
#include "../include/pirate/FakeServer.hpp"
#include "../include/pirate/FrequencySubbands.hpp"
#include "../include/pirate/GpuDequantizationKernel.hpp"
#include "../include/pirate/LaggedDownsamplingKernel.hpp"
#include "../include/pirate/PeakFindingKernel.hpp"
#include "../include/pirate/ReferenceLagbuf.hpp"
#include "../include/pirate/ReferenceTree.hpp"
#include "../include/pirate/RingbufCopyKernel.hpp"
#include "../include/pirate/TreeGriddingKernel.hpp"

#include "../include/pirate/loose_ends/tests.hpp"
#include "../include/pirate/loose_ends/timing.hpp"

using namespace std;
using namespace ksgpu;
using namespace pirate;
namespace py = pybind11;

// Declared extern here, defined in src_lib/scratch.cu.
namespace pirate { extern void scratch(); }


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

    py::class_<ResourceTracker>(m, "ResourceTracker")
          .def(py::init<>())
          .def("add_kernel", &ResourceTracker::add_kernel,
               py::arg("key"), py::arg("gmem_bw_nbytes"))
          .def("add_memcpy_h2g", &ResourceTracker::add_memcpy_h2g,
               py::arg("key"), py::arg("nbytes"))
          .def("add_memcpy_g2h", &ResourceTracker::add_memcpy_g2h,
               py::arg("key"), py::arg("nbytes"))
          .def("add_gmem_bw", &ResourceTracker::add_gmem_bw,
               py::arg("key"), py::arg("gmem_bw_nbytes"))
          .def("add_hmem_bw", &ResourceTracker::add_hmem_bw,
               py::arg("key"), py::arg("hmem_bw_nbytes"))
          .def("add_gmem_footprint", &ResourceTracker::add_gmem_footprint,
               py::arg("key"), py::arg("gmem_footprint_nbytes"), py::arg("align") = false)
          .def("add_hmem_footprint", &ResourceTracker::add_hmem_footprint,
               py::arg("key"), py::arg("hmem_footprint_nbytes"), py::arg("align") = false)
          .def("get_gmem_bw", &ResourceTracker::get_gmem_bw,
               py::arg("key") = "")
          .def("get_gmem_footprint", &ResourceTracker::get_gmem_footprint,
               py::arg("key") = "")
          .def("get_hmem_footprint", &ResourceTracker::get_hmem_footprint,
               py::arg("key") = "")
          .def("clone", &ResourceTracker::clone)
          .def("__iadd__", &ResourceTracker::operator+=, py::return_value_policy::reference)
          .def("to_yaml_string", &ResourceTracker::to_yaml_string,
               py::arg("multiplier"), py::arg("fine_grained"))
    ;

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

    py::class_<GpuDedisperser>(m, "GpuDedisperser")
          .def(py::init<const std::shared_ptr<DedispersionPlan> &>(), py::arg("plan"))
          .def_readonly("resource_tracker", &GpuDedisperser::resource_tracker)
          .def_static("test_random", &GpuDedisperser::test_random)
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

    py::class_<GpuDequantizationKernel>(m, "GpuDequantizationKernel")
          .def_static("test_random", &GpuDequantizationKernel::test_random)
          .def_static("time_selected", &GpuDequantizationKernel::time_selected)
    ;

    py::class_<ReferenceLagbuf>(m, "ReferenceLagbuf")
          .def_static("test_random", &ReferenceLagbuf::test_random)
    ;

    py::class_<ReferenceTree>(m, "ReferenceTree")
          .def_static("test_basics", &ReferenceTree::test_basics)
          .def_static("test_subbands", &ReferenceTree::test_subbands)
    ;

    py::class_<PfWeightReaderMicrokernel>(m, "PfWeightReaderMicrokernel")
          .def_static("test_random", &PfWeightReaderMicrokernel::test_random)
          .def_static("registry_size", &PfWeightReaderMicrokernel::registry_size)
          .def_static("show_registry", &PfWeightReaderMicrokernel::show_registry)
    ;

    py::class_<PfOutputMicrokernel>(m, "PfOutputMicrokernel")
          .def_static("test_random", &PfOutputMicrokernel::test_random)
          .def_static("registry_size", &PfOutputMicrokernel::registry_size)
          .def_static("show_registry", &PfOutputMicrokernel::show_registry)
    ;

    // DedispersionConfig::PeakFindingConfig (nested struct)
    py::class_<DedispersionConfig::PeakFindingConfig>(m, "PeakFindingConfig")
          .def_readonly("max_width", &DedispersionConfig::PeakFindingConfig::max_width)
          .def_readonly("dm_downsampling", &DedispersionConfig::PeakFindingConfig::dm_downsampling)
          .def_readonly("time_downsampling", &DedispersionConfig::PeakFindingConfig::time_downsampling)
          .def_readonly("wt_dm_downsampling", &DedispersionConfig::PeakFindingConfig::wt_dm_downsampling)
          .def_readonly("wt_time_downsampling", &DedispersionConfig::PeakFindingConfig::wt_time_downsampling)
    ;

    // DedispersionConfig::EarlyTrigger (nested struct)
    py::class_<DedispersionConfig::EarlyTrigger>(m, "EarlyTrigger")
          .def_readonly("ds_level", &DedispersionConfig::EarlyTrigger::ds_level)
          .def_readonly("delta", &DedispersionConfig::EarlyTrigger::delta_rank)
    ;
    
    py::class_<DedispersionConfig>(m, "DedispersionConfig")
          .def_static("from_yaml", static_cast<DedispersionConfig (*)(const std::string &)>(&DedispersionConfig::from_yaml),
                      py::arg("filename"))
          .def_static("make_random",
               [](int max_rank, int max_early_triggers, bool gpu_valid, bool verbose) {
                   DedispersionConfig::RandomArgs args;
                   args.max_rank = max_rank;
                   args.max_early_triggers = max_early_triggers;
                   args.gpu_valid = gpu_valid;
                   args.verbose = verbose;
                   return DedispersionConfig::make_random(args);
               },
               py::arg("max_rank") = 10,
               py::arg("max_early_triggers") = 5,
               py::arg("gpu_valid") = true,
               py::arg("verbose") = false)
          .def("to_yaml_string", &DedispersionConfig::to_yaml_string,
               py::arg("verbose") = false)
          .def("validate", &DedispersionConfig::validate)
          .def("add_early_trigger", &DedispersionConfig::add_early_trigger,
               py::arg("ds_level"), py::arg("tree_rank"))
          .def("get_nelts_per_segment", &DedispersionConfig::get_nelts_per_segment)
          .def("frequency_to_index", &DedispersionConfig::frequency_to_index, py::arg("f"))
          .def("index_to_frequency", &DedispersionConfig::index_to_frequency, py::arg("index"))
          .def("delay_to_frequency", &DedispersionConfig::delay_to_frequency, py::arg("delay"))
          .def("frequency_to_delay", &DedispersionConfig::frequency_to_delay, py::arg("f"))
          .def("get_total_nfreq", &DedispersionConfig::get_total_nfreq)
          .def("test", &DedispersionConfig::test)
          .def("make_channel_map", &DedispersionConfig::make_channel_map)
          // dtype: exposed via getter/setter using ksgpu::Dtype::str() / ksgpu::Dtype::from_str()
          .def_property("dtype",
               [](const DedispersionConfig &self) { return self.dtype.str(); },
               [](DedispersionConfig &self, const std::string &s) { self.dtype = ksgpu::Dtype::from_str(s); })
          // Frequency channel configuration
          .def_readonly("zone_nfreq", &DedispersionConfig::zone_nfreq)
          .def_readonly("zone_freq_edges", &DedispersionConfig::zone_freq_edges)
          // Core dedispersion parameters
          .def_readonly("time_sample_ms", &DedispersionConfig::time_sample_ms)
          .def_readonly("tree_rank", &DedispersionConfig::tree_rank)
          .def_readonly("num_downsampling_levels", &DedispersionConfig::num_downsampling_levels)
          .def_readonly("time_samples_per_chunk", &DedispersionConfig::time_samples_per_chunk)
          // Frequency sub-band configuration
          .def_readonly("frequency_subband_counts", &DedispersionConfig::frequency_subband_counts)
          // Peak-finding parameters (one per downsampling level)
          .def_readonly("peak_finding_params", &DedispersionConfig::peak_finding_params)
          // Early triggers
          .def_readonly("early_triggers", &DedispersionConfig::early_triggers)
          // GPU configuration
          .def_readonly("beams_per_gpu", &DedispersionConfig::beams_per_gpu)
          .def_readonly("beams_per_batch", &DedispersionConfig::beams_per_batch)
          .def_readonly("num_active_batches", &DedispersionConfig::num_active_batches)
          // Testing parameter
          .def_readonly("max_gpu_clag", &DedispersionConfig::max_gpu_clag)
    ;

    // DedispersionPlan: construct via shared_ptr
    py::class_<DedispersionPlan, std::shared_ptr<DedispersionPlan>>(m, "DedispersionPlan")
          .def(py::init<const DedispersionConfig &>(), py::arg("config"))
          .def_readonly("nfreq", &DedispersionPlan::nfreq)
          .def_readonly("nt_in", &DedispersionPlan::nt_in)
          .def_readonly("num_downsampling_levels", &DedispersionPlan::num_downsampling_levels)
          .def_readonly("beams_per_gpu", &DedispersionPlan::beams_per_gpu)
          .def_readonly("beams_per_batch", &DedispersionPlan::beams_per_batch)
          .def_readonly("num_active_batches", &DedispersionPlan::num_active_batches)
          .def_readonly("ntrees", &DedispersionPlan::ntrees)
          .def_readonly("nbits", &DedispersionPlan::nbits)
          .def("to_yaml_string", &DedispersionPlan::to_yaml_string, py::arg("verbose") = false)
    ;

    py::class_<FrequencySubbands>(m, "FrequencySubbands")
          .def(py::init<const std::vector<long> &>(), py::arg("subband_counts"))
          .def("to_string", &FrequencySubbands::to_string)
    ;

    m.def("time_cpu_downsample", &time_cpu_downsample, py::arg("nthreads"));
    m.def("time_gpu_downsample", &time_gpu_downsample);
    m.def("time_gpu_transpose", &time_gpu_transpose);
    
    // "Zombie" tests (code that I wrote during protoyping that may never get used)
    m.def("test_avx2_m64_outbuf", &test_avx2_m64_outbuf);
    m.def("test_cpu_downsampler", &test_cpu_downsampler);
    m.def("test_gpu_downsample", &test_gpu_downsample);
    m.def("test_gpu_transpose", &test_gpu_transpose);
    m.def("test_gpu_reduce2", &test_gpu_reduce2);

    // Called by 'python -m pirate_frb scratch'. Defined in src_lib/utils.cu.
    m.def("scratch", &scratch);

}
