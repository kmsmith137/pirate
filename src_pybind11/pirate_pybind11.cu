// Main pybind11 source file containing the PYBIND11_MODULE definition.
// Core dedispersion bindings are defined here.
// Kernel and infrastructure bindings are in pirate_pybind11_kernels.cu.
//
// For an explanation of PY_ARRAY_UNIQUE_SYMBOL, see comments in ksgpu/src_pybind11/ksgpu_pybind11.cu.

#define PY_ARRAY_UNIQUE_SYMBOL PyArray_API_pirate
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

// Needed in order to wrap methods with STL arguments (e.g. const vector<int> &vcpu_list).
#include <pybind11/stl.h>

#include <ksgpu/pybind11.hpp>

#include "../include/pirate/BumpAllocator.hpp"
#include "../include/pirate/CudaStreamPool.hpp"
#include "../include/pirate/Dedisperser.hpp"
#include "../include/pirate/DedispersionConfig.hpp"
#include "../include/pirate/DedispersionPlan.hpp"
#include "../include/pirate/DedispersionTree.hpp"
#include "../include/pirate/FrequencySubbands.hpp"
#include "../include/pirate/ResourceTracker.hpp"

using namespace std;
using namespace ksgpu;
using namespace pirate;
namespace py = pybind11;

// Defined in pirate_pybind11_kernels.cu.
namespace pirate { void register_kernel_bindings(pybind11::module &m); }


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

    // CudaStreamPool: always accessed via shared_ptr.
    // Stream members are exposed to python as CudaStreamWrapper objects.
    py::class_<CudaStreamPool, std::shared_ptr<CudaStreamPool>>(m, "CudaStreamPool")
        .def(py::init([](int num_compute_streams, int compute_stream_priority) {
            return CudaStreamPool::create(num_compute_streams, compute_stream_priority);
        }), 
          py::arg("num_compute_streams"),
          py::arg("compute_stream_priority") = 0
        )
        .def_readonly("pool_id", &CudaStreamPool::pool_id)
        .def_readonly("num_compute_streams", &CudaStreamPool::num_compute_streams)
        .def_property_readonly("low_priority_g2h_stream",
            [](const CudaStreamPool &self) { return self.low_priority_g2h_stream; })
        .def_property_readonly("low_priority_h2g_stream",
            [](const CudaStreamPool &self) { return self.low_priority_h2g_stream; })
        .def_property_readonly("high_priority_g2h_stream",
            [](const CudaStreamPool &self) { return self.high_priority_g2h_stream; })
        .def_property_readonly("high_priority_h2g_stream",
            [](const CudaStreamPool &self) { return self.high_priority_h2g_stream; })
        .def_property_readonly("compute_streams",
            [](const CudaStreamPool &self) { return self.compute_streams; })
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
          .def("get_hmem_bw", &ResourceTracker::get_hmem_bw,
               py::arg("key") = "")
          .def("get_h2g_bw", &ResourceTracker::get_h2g_bw,
               py::arg("key") = "")
          .def("get_g2h_bw", &ResourceTracker::get_g2h_bw,
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

    // GpuDedisperser::Params (nested struct, exposed as GpuDedisperserParams)
    py::class_<GpuDedisperser::Params>(m, "GpuDedisperserParams")
          .def(py::init<>())
          .def_readwrite("plan", &GpuDedisperser::Params::plan)
          .def_readwrite("stream_pool", &GpuDedisperser::Params::stream_pool)
          .def_readwrite("nbatches_wt", &GpuDedisperser::Params::nbatches_wt)
          .def_readwrite("nbatches_out", &GpuDedisperser::Params::nbatches_out)
          .def_readwrite("detect_deadlocks", &GpuDedisperser::Params::detect_deadlocks)
    ;

    py::class_<GpuDedisperser>(m, "GpuDedisperser")
          .def(py::init<const GpuDedisperser::Params &>(), py::arg("params"))
          .def_readonly("resource_tracker", &GpuDedisperser::resource_tracker)
          .def_static("test_random", &GpuDedisperser::test_random)
          .def_static("test_one", &GpuDedisperser::test_one,
               py::arg("config"), 
               py::arg("nchunks"), 
               py::arg("nbatches_out") = 0,
               py::arg("host_only") = false)
          .def_static("time_one", &GpuDedisperser::time_one,
               py::arg("config"), py::arg("niterations"), py::arg("use_hugepages"))
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
          .def_readwrite("zone_nfreq", &DedispersionConfig::zone_nfreq)
          .def_readwrite("zone_freq_edges", &DedispersionConfig::zone_freq_edges)
          // Core dedispersion parameters
          .def_readwrite("time_sample_ms", &DedispersionConfig::time_sample_ms)
          .def_readwrite("tree_rank", &DedispersionConfig::tree_rank)
          .def_readwrite("num_downsampling_levels", &DedispersionConfig::num_downsampling_levels)
          .def_readwrite("time_samples_per_chunk", &DedispersionConfig::time_samples_per_chunk)
          // Frequency sub-band configuration
          .def_readwrite("frequency_subband_counts", &DedispersionConfig::frequency_subband_counts)
          // Peak-finding parameters (one per downsampling level)
          .def_readwrite("peak_finding_params", &DedispersionConfig::peak_finding_params)
          // Early triggers
          .def_readwrite("early_triggers", &DedispersionConfig::early_triggers)
          // GPU configuration
          .def_readwrite("beams_per_gpu", &DedispersionConfig::beams_per_gpu)
          .def_readwrite("beams_per_batch", &DedispersionConfig::beams_per_batch)
          .def_readwrite("num_active_batches", &DedispersionConfig::num_active_batches)
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

    // DedispersionTree: simple data class representing output of dedisperser
    // for one choice of (downsampling level, early trigger).
    py::class_<DedispersionTree>(m, "DedispersionTree")
          .def_readonly("ds_level", &DedispersionTree::ds_level)
          .def_readonly("amb_rank", &DedispersionTree::amb_rank)
          .def_readonly("pri_dd_rank", &DedispersionTree::pri_dd_rank)
          .def_readonly("early_dd_rank", &DedispersionTree::early_dd_rank)
          .def_readonly("nt_ds", &DedispersionTree::nt_ds)
          .def_readonly("frequency_subbands", &DedispersionTree::frequency_subbands)
          .def_readonly("pf", &DedispersionTree::pf)
          .def_readonly("nprofiles", &DedispersionTree::nprofiles)
          .def_readonly("ndm_out", &DedispersionTree::ndm_out)
          .def_readonly("ndm_wt", &DedispersionTree::ndm_wt)
          .def_readonly("nt_out", &DedispersionTree::nt_out)
          .def_readonly("nt_wt", &DedispersionTree::nt_wt)
          .def_readonly("dm_min", &DedispersionTree::dm_min)
          .def_readonly("dm_max", &DedispersionTree::dm_max)
          .def_readonly("trigger_frequency", &DedispersionTree::trigger_frequency)
    ;

    py::class_<FrequencySubbands>(m, "FrequencySubbands")
          // Constructors
          .def(py::init<>())  // default constructor
          .def(py::init<const std::vector<long> &>(), py::arg("subband_counts"))
          .def(py::init<const std::vector<long> &, double, double>(),
               py::arg("subband_counts"), py::arg("fmin"), py::arg("fmax"))
          // Data members (readonly since they are computed from subband_counts)
          .def_readonly("subband_counts", &FrequencySubbands::subband_counts)
          .def_readonly("pf_rank", &FrequencySubbands::pf_rank)
          .def_readonly("F", &FrequencySubbands::F)
          .def_readonly("M", &FrequencySubbands::M)
          .def_readonly("m_to_f", &FrequencySubbands::m_to_f)
          .def_readonly("m_to_d", &FrequencySubbands::m_to_d)
          .def_readonly("f_to_ilo", &FrequencySubbands::f_to_ilo)
          .def_readonly("f_to_ihi", &FrequencySubbands::f_to_ihi)
          .def_readonly("f_to_mbase", &FrequencySubbands::f_to_mbase)
          .def_readonly("i_to_f", &FrequencySubbands::i_to_f)
          .def_readonly("fmin", &FrequencySubbands::fmin)
          .def_readonly("fmax", &FrequencySubbands::fmax)
          // Inline methods
          .def("m_to_ilo", &FrequencySubbands::m_to_ilo, py::arg("m"))
          .def("m_to_ihi", &FrequencySubbands::m_to_ihi, py::arg("m"))
          // Methods with ostream arguments: wrap to return strings
          .def("show", [](const FrequencySubbands &self) {
              std::ostringstream os;
              self.show(os);
              return os.str();
          })
          .def("show_compact", [](const FrequencySubbands &self) {
              std::stringstream ss;
              self.show_compact(ss);
              return ss.str();
          })
          .def("show_token", [](const FrequencySubbands &self, uint token) {
              std::ostringstream os;
              self.show_token(token, os);
              return os.str();
          }, py::arg("token"))
          .def("to_string", &FrequencySubbands::to_string)
          // Static methods
          .def_static("from_threshold", &FrequencySubbands::from_threshold,
               py::arg("fmin"), py::arg("fmax"), py::arg("threshold"), py::arg("pf_rank") = 4)
          .def_static("restrict_subband_counts", &FrequencySubbands::restrict_subband_counts,
               py::arg("subband_counts"), py::arg("et_delta_rank"), py::arg("new_pf_rank"))
          .def_static("validate_subband_counts", &FrequencySubbands::validate_subband_counts,
               py::arg("subband_counts"))
          .def_static("make_random_subband_counts",
               static_cast<std::vector<long> (*)(long)>(&FrequencySubbands::make_random_subband_counts),
               py::arg("pf_rank"))
          .def_static("make_random_subband_counts",
               static_cast<std::vector<long> (*)()>(&FrequencySubbands::make_random_subband_counts))
          .def_static("make_random", &FrequencySubbands::make_random)
    ;

    // BumpAllocator: Thread-safe bump allocator for GPU/host memory
    // Wrapped with shared_ptr for proper lifetime management when arrays reference the allocator.
    // Note: Python injections in pirate_frb/pybind11_injections.py handle aflags and dtype conversions.
    py::class_<BumpAllocator, std::shared_ptr<BumpAllocator>>(m, "BumpAllocator",
        "Thread-safe bump allocator supporting GPU/host memory.\n\n"
        "Modes:\n"
        "  - capacity >= 0: Pre-allocates base region, allocations share this memory\n"
        "  - capacity < 0: Dummy mode, each allocation gets independent memory")
        .def(py::init<int, long>(),
            py::arg("aflags"), py::arg("capacity"),
            "Create allocator.\n\n"
            "Args:\n"
            "    aflags: Memory allocation flags (af_gpu, af_rhost, etc.)\n"
            "    capacity: Bytes to pre-allocate (>= 0) or -1 for dummy mode")
        .def_property_readonly("nbytes_allocated",
            [](const BumpAllocator &self) { return self.nbytes_allocated.load(); },
            "Bytes allocated so far (aligned to 128-byte cache lines)")
        .def_readonly("aflags", &BumpAllocator::aflags,
            "Memory allocation flags")
        .def_readonly("capacity", &BumpAllocator::capacity,
            "Total capacity in bytes, or -1 for dummy mode")
        .def("_allocate_array_raw",
            [](std::shared_ptr<BumpAllocator> self, ksgpu::Dtype dtype, const std::vector<long> &shape) {
                return self->_allocate_array_internal(dtype, shape.size(), shape.data(), nullptr);
            },
            py::arg("dtype"), py::arg("shape"),
            "Internal: allocate array with dtype and shape (use allocate_array() instead)")
    ;

    // Register kernel and infrastructure bindings from pirate_pybind11_kernels.cu.
    register_kernel_bindings(m);
}
