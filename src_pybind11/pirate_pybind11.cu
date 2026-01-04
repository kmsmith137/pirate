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

    py::class_<GpuDedisperser, std::shared_ptr<GpuDedisperser>>(m, "GpuDedisperser")
          .def(py::init([](const GpuDedisperser::Params &params) {
              return GpuDedisperser::create(params);
          }), py::arg("params"))
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
    py::class_<DedispersionConfig::PeakFindingConfig>(m, "PeakFindingConfig",
        "Configuration for peak finding at a single downsampling level.\n\n"
        "Defines the maximum width for peak detection and downsampling factors\n"
        "for both the coarse-grained and weights arrays relative to tree resolution.\n"
        "All members must be powers of two.")
          .def(py::init<>(),
               "Create a PeakFindingConfig with default (zero) values.")
          .def(py::init([](long max_width, long dm_downsampling, long time_downsampling,
                          long wt_dm_downsampling, long wt_time_downsampling) {
                   DedispersionConfig::PeakFindingConfig pf;
                   pf.max_width = max_width;
                   pf.dm_downsampling = dm_downsampling;
                   pf.time_downsampling = time_downsampling;
                   pf.wt_dm_downsampling = wt_dm_downsampling;
                   pf.wt_time_downsampling = wt_time_downsampling;
                   return pf;
               }),
               py::arg("max_width"),
               py::arg("dm_downsampling"),
               py::arg("time_downsampling"),
               py::arg("wt_dm_downsampling"),
               py::arg("wt_time_downsampling"),
               "Create a PeakFindingConfig.\n\n"
               "Args:\n"
               "    max_width: Maximum width of peak-finding kernel (in tree time samples)\n"
               "    dm_downsampling: DM downsampling factor relative to tree\n"
               "    time_downsampling: Time downsampling factor relative to tree\n"
               "    wt_dm_downsampling: DM downsampling factor for weights (>= dm_downsampling)\n"
               "    wt_time_downsampling: Time downsampling for weights (>= time_downsampling)")
          .def_readwrite("max_width", &DedispersionConfig::PeakFindingConfig::max_width,
               "Maximum width of peak-finding kernel (in tree time samples)")
          .def_readwrite("dm_downsampling", &DedispersionConfig::PeakFindingConfig::dm_downsampling,
               "DM downsampling factor of coarse-grained array relative to tree")
          .def_readwrite("time_downsampling", &DedispersionConfig::PeakFindingConfig::time_downsampling,
               "Time downsampling factor of coarse-grained array relative to tree")
          .def_readwrite("wt_dm_downsampling", &DedispersionConfig::PeakFindingConfig::wt_dm_downsampling,
               "DM downsampling factor of weights array (must be >= dm_downsampling)")
          .def_readwrite("wt_time_downsampling", &DedispersionConfig::PeakFindingConfig::wt_time_downsampling,
               "Time downsampling factor of weights array (must be >= time_downsampling)")
          .def("__repr__", [](const DedispersionConfig::PeakFindingConfig &self) {
               std::ostringstream os;
               os << "PeakFindingConfig(max_width=" << self.max_width
                  << ", dm_downsampling=" << self.dm_downsampling
                  << ", time_downsampling=" << self.time_downsampling
                  << ", wt_dm_downsampling=" << self.wt_dm_downsampling
                  << ", wt_time_downsampling=" << self.wt_time_downsampling << ")";
               return os.str();
          })
    ;

    // DedispersionConfig::EarlyTrigger (nested struct)
    py::class_<DedispersionConfig::EarlyTrigger>(m, "EarlyTrigger",
        "Early trigger configuration for reduced-latency dedispersion.\n\n"
        "Early triggers search a subset [fmid, fmax] of the full frequency range\n"
        "at reduced latency. Each trigger is parameterized by a downsampling level\n"
        "and delta_rank, which specifies how 'early' the trigger is.")
          .def(py::init<>(),
               "Create an EarlyTrigger with default values (ds_level=-1, delta_rank=0).")
          .def(py::init([](long ds_level, long delta_rank) {
                   DedispersionConfig::EarlyTrigger et;
                   et.ds_level = ds_level;
                   et.delta_rank = delta_rank;
                   return et;
               }),
               py::arg("ds_level"),
               py::arg("delta_rank"),
               "Create an EarlyTrigger.\n\n"
               "Args:\n"
               "    ds_level: Downsampling level (0 <= ds_level < num_downsampling_levels)\n"
               "    delta_rank: Specifies 'early-ness' of trigger (must be > 0)")
          .def_readwrite("ds_level", &DedispersionConfig::EarlyTrigger::ds_level,
               "Downsampling level (0 <= ds_level < num_downsampling_levels)")
          .def_readwrite("delta_rank", &DedispersionConfig::EarlyTrigger::delta_rank,
               "Specifies 'early-ness' of trigger (must be > 0)")
          .def("__repr__", [](const DedispersionConfig::EarlyTrigger &self) {
               std::ostringstream os;
               os << self;  // Use C++ operator<<
               return os.str();
          })
          .def("__eq__", [](const DedispersionConfig::EarlyTrigger &self, 
                           const DedispersionConfig::EarlyTrigger &other) {
               return self == other;  // Use C++ operator==
          }, py::arg("other"))
    ;
    
    py::class_<DedispersionConfig>(m, "DedispersionConfig",
        "Configuration for dedispersion processing.\n\n"
        "Specifies frequency channels, time samples, dedispersion tree parameters,\n"
        "downsampling levels, peak-finding configuration, early triggers, and GPU settings.\n"
        "Can be loaded from YAML files or constructed programmatically.\n\n"
        "Example:\n"
        "    # Load from YAML\n"
        "    config = DedispersionConfig.from_yaml('config.yaml')\n\n"
        "    # Create and configure programmatically\n"
        "    config = DedispersionConfig()\n"
        "    config.zone_nfreq = [1024]\n"
        "    config.zone_freq_edges = [400.0, 800.0]\n"
        "    config.time_sample_ms = 0.983\n"
        "    config.tree_rank = 13")
          .def(py::init<>(),
               "Create an empty DedispersionConfig.\n\n"
               "All fields are initialized to default values and should be set programmatically.")
          .def_static("from_yaml", static_cast<DedispersionConfig (*)(const std::string &)>(&DedispersionConfig::from_yaml),
                      py::arg("filename"),
                      "Load DedispersionConfig from a YAML file.\n\n"
                      "Args:\n"
                      "    filename: Path to YAML configuration file\n\n"
                      "Returns:\n"
                      "    DedispersionConfig object initialized from file")
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
               py::arg("verbose") = false,
               "Generate a random DedispersionConfig for testing.\n\n"
               "Args:\n"
               "    max_rank: Maximum tree rank (default=10)\n"
               "    max_early_triggers: Max number of early triggers (0 to disable, default=5)\n"
               "    gpu_valid: Generate GPU-valid configuration (default=True)\n"
               "    verbose: Print debug info (default=False)\n\n"
               "Returns:\n"
               "    Randomly generated DedispersionConfig")
          .def("to_yaml_string", &DedispersionConfig::to_yaml_string,
               py::arg("verbose") = false,
               "Convert configuration to YAML string.\n\n"
               "Args:\n"
               "    verbose: Include explanatory comments (default=False)\n\n"
               "Returns:\n"
               "    YAML string representation")
          .def("validate", &DedispersionConfig::validate,
               "Validate the configuration.\n\n"
               "Checks that all parameters are consistent and within valid ranges.\n"
               "Throws an exception if validation fails.")
          .def("add_early_trigger", &DedispersionConfig::add_early_trigger,
               py::arg("ds_level"), py::arg("tree_rank"),
               "Add a single early trigger.\n\n"
               "Early triggers are automatically kept sorted by ds_level (increasing)\n"
               "and delta_rank (decreasing).\n\n"
               "Args:\n"
               "    ds_level: Downsampling level (0 <= ds_level < num_downsampling_levels)\n"
               "    tree_rank: Tree rank for this trigger (must be > 0)")
          .def("add_early_triggers", 
               [](DedispersionConfig &self, long ds_level, const std::vector<long> &tree_ranks) {
                   for (long tree_rank : tree_ranks) {
                       self.add_early_trigger(ds_level, tree_rank);
                   }
               },
               py::arg("ds_level"), py::arg("tree_ranks"),
               "Add multiple early triggers at the same downsampling level.\n\n"
               "Args:\n"
               "    ds_level: Downsampling level for all triggers\n"
               "    tree_ranks: List of tree ranks to add")
          .def("get_nelts_per_segment", &DedispersionConfig::get_nelts_per_segment,
               "Get the number of elements per segment.\n\n"
               "Returns:\n"
               "    Number of elements per segment")
          .def("frequency_to_index", &DedispersionConfig::frequency_to_index, py::arg("f"),
               "Convert frequency to fractional frequency channel index.\n\n"
               "Args:\n"
               "    f: Frequency in MHz\n\n"
               "Returns:\n"
               "    Fractional channel index (0.0 at lowest frequency)")
          .def("index_to_frequency", &DedispersionConfig::index_to_frequency, py::arg("index"),
               "Convert fractional channel index to frequency.\n\n"
               "Args:\n"
               "    index: Fractional channel index\n\n"
               "Returns:\n"
               "    Frequency in MHz")
          .def("delay_to_frequency", &DedispersionConfig::delay_to_frequency, py::arg("delay"),
               "Convert dispersion delay to frequency.\n\n"
               "Delay is scaled so d=0 at f_max and d=2^tree_rank at f_min.\n\n"
               "Args:\n"
               "    delay: Dispersion delay in tree units\n\n"
               "Returns:\n"
               "    Frequency in MHz")
          .def("frequency_to_delay", &DedispersionConfig::frequency_to_delay, py::arg("f"),
               "Convert frequency to dispersion delay.\n\n"
               "Delay is scaled so d=0 at f_max and d=2^tree_rank at f_min.\n\n"
               "Args:\n"
               "    f: Frequency in MHz\n\n"
               "Returns:\n"
               "    Dispersion delay in tree units")
          .def("dm_per_unit_delay", &DedispersionConfig::dm_per_unit_delay,
               "Get DM corresponding to one time sample delay across the full band.\n\n"
               "Returns:\n"
               "    DM in pc cm^-3 for one time sample delay")
          .def("get_total_nfreq", &DedispersionConfig::get_total_nfreq,
               "Get total number of frequency channels across all zones.\n\n"
               "Returns:\n"
               "    Sum of zone_nfreq")
          .def("test", &DedispersionConfig::test,
               "Test that frequency/delay conversions are self-consistent.\n\n"
               "Samples random values and checks that forward/inverse transforms\n"
               "are correct. Throws an exception if test fails.")
          .def("make_channel_map", &DedispersionConfig::make_channel_map,
               "Create channel map array defining tree-to-frequency mapping.\n\n"
               "Returns:\n"
               "    Array of length (2^tree_rank + 1) with channel boundaries")
          // dtype: now uses direct readwrite, Python injection provides flexible setter
          .def_readwrite("dtype", &DedispersionConfig::dtype,
               "Data type for dedispersion (e.g. 'float32', 'float16')")
          // Frequency channel configuration
          .def_readwrite("zone_nfreq", &DedispersionConfig::zone_nfreq,
               "Number of frequency channels in each zone (list of length nzones)")
          .def_readwrite("zone_freq_edges", &DedispersionConfig::zone_freq_edges,
               "Frequency edges in MHz (list of length nzones+1, monotone increasing)")
          // Core dedispersion parameters
          .def_readwrite("time_sample_ms", &DedispersionConfig::time_sample_ms,
               "Time sample length in milliseconds")
          .def_readwrite("tree_rank", &DedispersionConfig::tree_rank,
               "Tree rank: number of tree channels is 2^tree_rank")
          .def_readwrite("num_downsampling_levels", &DedispersionConfig::num_downsampling_levels,
               "Number of downsampling levels (ds_level in 0..num_downsampling_levels-1)")
          .def_readwrite("time_samples_per_chunk", &DedispersionConfig::time_samples_per_chunk,
               "Number of time samples processed per chunk")
          // Frequency sub-band configuration
          .def_readwrite("frequency_subband_counts", &DedispersionConfig::frequency_subband_counts,
               "Frequency subband counts (set to [1] to disable subbanding)")
          // Peak-finding parameters (one per downsampling level)
          .def_readwrite("peak_finding_params", &DedispersionConfig::peak_finding_params,
               "Peak-finding configuration for each downsampling level")
          // Early triggers
          .def_readwrite("early_triggers", &DedispersionConfig::early_triggers,
               "List of early triggers (sorted by ds_level, then delta_rank)")
          // GPU configuration
          .def_readwrite("beams_per_gpu", &DedispersionConfig::beams_per_gpu,
               "Number of beams processed per GPU")
          .def_readwrite("beams_per_batch", &DedispersionConfig::beams_per_batch,
               "Number of beams per batch")
          .def_readwrite("num_active_batches", &DedispersionConfig::num_active_batches,
               "Number of active batches")
          // Testing parameter
          .def_readwrite("max_gpu_clag", &DedispersionConfig::max_gpu_clag,
               "Testing parameter: limit on-GPU ring buffer clag (default=10000)")
    ;

    // DedispersionPlan: construct via shared_ptr
    py::class_<DedispersionPlan, std::shared_ptr<DedispersionPlan>>(m, "DedispersionPlan",
        "Dedispersion execution plan.\n\n"
        "Created from a DedispersionConfig, this class contains all the derived parameters\n"
        "and data structures needed to execute dedispersion on the GPU. The plan includes:\n\n"
        "  - Stage1 and Stage2 dedispersion trees\n"
        "  - Kernel parameters for all processing stages\n"
        "  - Memory buffer layouts and ring buffer configuration\n"
        "  - Derived parameters like tree dimensions and output array shapes\n\n"
        "The plan is immutable once constructed and is shared between dedisperser instances.\n\n"
        "Example:\n"
        "    config = DedispersionConfig.from_yaml('config.yaml')\n"
        "    plan = DedispersionPlan(config)\n"
        "    print(f'Plan has {plan.ntrees} trees')\n"
        "    for i, tree in enumerate(plan.trees):\n"
        "        print(f'Tree {i}: ds_level={tree.ds_level}, dm_range=[{tree.dm_min:.1f}, {tree.dm_max:.1f}]')")
          .def(py::init<const DedispersionConfig &>(), 
               py::arg("config"),
               "Create a DedispersionPlan from a configuration.\n\n"
               "Args:\n"
               "    config: DedispersionConfig object (must be validated)")
          .def_readonly("config", &DedispersionPlan::config,
               "The DedispersionConfig used to create this plan")
          .def_readonly("dtype", &DedispersionPlan::dtype,
               "Data type for dedispersion (same as config.dtype)")
          .def_readonly("nfreq", &DedispersionPlan::nfreq,
               "Total number of frequency channels (same as config.get_total_nfreq())")
          .def_readonly("nt_in", &DedispersionPlan::nt_in,
               "Number of input time samples per chunk (same as config.time_samples_per_chunk)")
          .def_readonly("num_downsampling_levels", &DedispersionPlan::num_downsampling_levels,
               "Number of downsampling levels (same as config.num_downsampling_levels)")
          .def_readonly("beams_per_gpu", &DedispersionPlan::beams_per_gpu,
               "Number of beams processed per GPU (same as config.beams_per_gpu)")
          .def_readonly("beams_per_batch", &DedispersionPlan::beams_per_batch,
               "Number of beams per batch (same as config.beams_per_batch)")
          .def_readonly("num_active_batches", &DedispersionPlan::num_active_batches,
               "Number of active batches (same as config.num_active_batches)")
          .def_readonly("ntrees", &DedispersionPlan::ntrees,
               "Total number of stage2 trees (num_downsampling_levels + number of early triggers)")
          .def_readonly("nbits", &DedispersionPlan::nbits,
               "Number of bits per element (same as config.dtype.nbits)")
          .def_readonly("trees", &DedispersionPlan::trees,
               "Vector of DedispersionTree objects representing stage2 output trees.\n"
               "Length is ntrees. Each tree corresponds to one (downsampling level, early trigger) pair.")
          .def_readonly("stage1_dd_rank", &DedispersionPlan::stage1_dd_rank,
               "Active dedispersion rank of each stage1 tree.\n"
               "Vector of length num_downsampling_levels. Stage1 trees are internal to dedispersion.")
          .def_readonly("stage1_amb_rank", &DedispersionPlan::stage1_amb_rank,
               "Ambient rank of each stage1 tree (= number of coarse frequency channels).\n"
               "Vector of length num_downsampling_levels.")
          .def_readonly("nelts_per_segment", &DedispersionPlan::nelts_per_segment,
               "Number of elements per GPU memory segment.\n"
               "Currently always constants::bytes_per_gpu_cache_line / sizeof(dtype)")
          .def_readonly("nbytes_per_segment", &DedispersionPlan::nbytes_per_segment,
               "Number of bytes per GPU memory segment.\n"
               "Currently always constants::bytes_per_gpu_cache_line")
          .def("to_yaml_string", &DedispersionPlan::to_yaml_string, 
               py::arg("verbose") = false,
               "Convert plan to YAML string representation.\n\n"
               "Args:\n"
               "    verbose: Include detailed information about all trees and parameters\n\n"
               "Returns:\n"
               "    YAML string representation of the plan")
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
               [](py::object pf_rank) -> std::vector<long> {
                   if (pf_rank.is_none()) {
                       return FrequencySubbands::make_random_subband_counts();
                   } else {
                       return FrequencySubbands::make_random_subband_counts(pf_rank.cast<long>());
                   }
               },
               py::arg("pf_rank") = py::none(),
               "Generate random subband counts.\n\n"
               "Args:\n"
               "    pf_rank: Peak-finding rank, or None to choose randomly\n\n"
               "Returns:\n"
               "    Random subband_counts vector")
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
