// Main pybind11 source file containing the PYBIND11_MODULE definition.
// Main dedispersion bindings (DedispersionConfig, DedispersionPlan, GpuDedisperser) are defined here.
// Other bindings are organized into separate files by subpackage:
//   - pirate_pybind11_core.cu: core classes (pirate_frb.core)
//   - pirate_pybind11_kernels.cu: GPU kernels (pirate_frb.kernels)
//   - pirate_pybind11_casm.cu: CASM beamformer (pirate_frb.casm)
//   - pirate_pybind11_loose_ends.cu: prototype functions (pirate_frb.loose_ends)
//
// For an explanation of PY_ARRAY_UNIQUE_SYMBOL, see comments in ksgpu/src_pybind11/ksgpu_pybind11.cu.

#define PY_ARRAY_UNIQUE_SYMBOL PyArray_API_pirate
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

// Needed in order to wrap methods with STL arguments (e.g. const vector<int> &vcpu_list).
#include <pybind11/stl.h>

#include <ksgpu/pybind11.hpp>

#include "../include/pirate/constants.hpp"
#include "../include/pirate/CoalescedDdKernel2.hpp"    // GpuDedisperser.Dcore property
#include "../include/pirate/CudaStreamPool.hpp"
#include "../include/pirate/Dedisperser.hpp"
#include "../include/pirate/DedispersionConfig.hpp"
#include "../include/pirate/DedispersionPlan.hpp"
#include "../include/pirate/PeakFindingKernel.hpp"     // ReferenceDedisperser.pf_kernels

using namespace std;
using namespace ksgpu;
using namespace pirate;
namespace py = pybind11;

// Defined in separate source files.
namespace pirate {
    void register_core_bindings(pybind11::module &m);
    void register_avar_bindings(pybind11::module &m);
    void register_kernel_bindings(pybind11::module &m);
    void register_casm_bindings(pybind11::module &m);
    void register_chime_bindings(pybind11::module &m);
    void register_loose_ends_bindings(pybind11::module &m);
    void register_simpulse_bindings(pybind11::module &m);
    void register_utils_bindings(pybind11::module &m);
}


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

    // Register bindings from other files. simpulse is registered BEFORE core because
    // AssembledFrame.randomize (in core) has a default argument of type
    // shared_ptr<const SinglePulse>, which pybind can only convert once SinglePulse is registered.
    register_simpulse_bindings(m);
    register_core_bindings(m);
    register_avar_bindings(m);
    register_kernel_bindings(m);
    register_casm_bindings(m);
    register_chime_bindings(m);
    register_loose_ends_bindings(m);
    register_utils_bindings(m);

    // pirate::constants, exposed to python as pirate_frb.constants.<name>. Bound as a (never-
    // instantiated) class with read-only static properties, so assignment raises AttributeError.
    // When adding a constant here, also update include/pirate/constants.hpp's reminder.
    py::class_<constants>(m, "constants",
        "Compile-time constants (pirate::constants). Read-only; assignment raises AttributeError.")
        .def_readonly_static("max_tree_rank", &constants::max_tree_rank,
            "Maximum dedispersion tree rank (upper bound on DedispersionConfig.toplevel_tree_rank).")
        .def_readonly_static("max_primary_trees", &constants::max_primary_trees,
            "Maximum number of primary trees.")
        .def_readonly_static("max_pf_width", &constants::max_pf_width,
            "Maximum peak-finding kernel width (PrimaryTree::max_width), in tree time samples.")
        .def_readonly_static("k_dm", &constants::k_dm,
            "Dispersion constant K_DM, in (ms MHz^2) per (pc cm^{-3}): dispersion delay (ms) = "
            "k_dm * DM * (f_lo^{-2} - f_hi^{-2}), with frequencies in MHz.")
        .def_readonly_static("inactive_file_stream_capacity", &constants::inactive_file_stream_capacity,
            "Number of inactive (expired/cancelled) FileStreams retained by an FrbServer for "
            "ShowStreams history; the oldest are dropped beyond this.")
        .def_readonly_static("grouper_ping_timeout_ms", &constants::grouper_ping_timeout_ms,
            "Timeout (ms) for FrbGrouperClient.ping(): the early channel-level connectivity "
            "check done before bump allocation, to fail fast if the grouper isn't running.")
        .def_readonly_static("grouper_connect_timeout_ms", &constants::grouper_connect_timeout_ms,
            "Timeout (ms) for the real FrbGrouperClient reconnect (done in grouper_send_thread "
            "just before the Handshake).")
    ;

    // Main dedispersion classes defined here

    py::class_<DedispersionConfig>(m, "DedispersionConfig",
        "Configuration for dedispersion processing.\n\n"
        "Specifies frequency channels, time samples, dedispersion tree parameters,\n"
        "primary trees (with peak-finding configuration and early triggers), and GPU settings.\n"
        "Can be loaded from YAML files or constructed programmatically.\n\n"
        "See configs/dedispersion/chord_sb2_et.yml for an example with per-field documentation.\n\n"
        "Example usage::\n\n"
        "    # Load from YAML\n"
        "    config = DedispersionConfig.from_yaml('config.yaml')\n\n"
        "    # Create and configure programmatically\n"
        "    config = DedispersionConfig()\n"
        "    config.zone_nfreq = [1024]\n"
        "    config.zone_freq_edges = [400.0, 800.0]\n"
        "    config.time_sample_ms = 0.983\n"
        "    config.toplevel_tree_rank = 13")
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
               [](int max_toplevel_rank, int max_early_triggers, bool gpu_valid, bool verbose) {
                   DedispersionConfig::RandomArgs args;
                   args.max_toplevel_rank = max_toplevel_rank;
                   args.max_early_triggers = max_early_triggers;
                   args.gpu_valid = gpu_valid;
                   args.verbose = verbose;
                   return DedispersionConfig::make_random(args);
               },
               py::arg("max_toplevel_rank") = 10,
               py::arg("max_early_triggers") = 5,
               py::arg("gpu_valid") = true,
               py::arg("verbose") = false,
               "Generate a random DedispersionConfig for testing.\n\n"
               "Args:\n"
               "    max_toplevel_rank: Bounds toplevel_tree_rank (default=10)\n"
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
          .def("clone", &DedispersionConfig::clone,
               "Return a deep copy of this config (does not mutate the original).\n\n"
               "Useful before overriding fields (e.g. the beam geometry) without\n"
               "affecting the caller's config object.")
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
               "Delay is scaled so d=0 at f_max and d=2^toplevel_tree_rank at f_min.\n\n"
               "Args:\n"
               "    delay: Dispersion delay in tree units\n\n"
               "Returns:\n"
               "    Frequency in MHz")
          .def("frequency_to_delay", &DedispersionConfig::frequency_to_delay, py::arg("f"),
               "Convert frequency to dispersion delay.\n\n"
               "Delay is scaled so d=0 at f_max and d=2^toplevel_tree_rank at f_min.\n\n"
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
               "    Array of length (2^toplevel_tree_rank + 1) with channel boundaries")
          .def("make_random_freq_variances", &DedispersionConfig::make_random_freq_variances,
               py::arg("noisy") = false,
               "Random per-channel input variances for testing (one random value in [0,1] per zone).\n\n"
               "Args:\n"
               "    noisy: if true, print the per-zone variances\n\n"
               "Returns:\n"
               "    Array of length nfreq (constant within each frequency zone)")
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
          .def_readwrite("toplevel_tree_rank", &DedispersionConfig::toplevel_tree_rank,
               "Toplevel tree rank: number of tree channels is 2^toplevel_tree_rank.\n"
               "Individual trees have rank (toplevel_tree_rank - early_trigger_level - (p ? 1 : 0)).")
          .def_readwrite("time_samples_per_chunk", &DedispersionConfig::time_samples_per_chunk,
               "Number of time samples processed per chunk")
          // Frequency sub-band configuration
          .def_readwrite("frequency_subband_counts", &DedispersionConfig::frequency_subband_counts,
               "Frequency subband counts (set to [1] to disable subbanding)")
          // Primary trees (one per DM range searched)
          .def_readwrite("primary_trees", &DedispersionConfig::primary_trees,
               "List of PrimaryTree objects, one per DM range searched (ordered low to high DM).\n"
               "Each primary tree holds its peak-finding configuration and num_early_triggers.")
          .def_property_readonly("num_primary_trees", &DedispersionConfig::num_primary_trees,
               "Number of primary trees (= len(primary_trees))")
          // GPU configuration
          .def_readwrite("beams_per_gpu", &DedispersionConfig::beams_per_gpu,
               "Number of beams processed per GPU")
          .def_readwrite("beams_per_batch", &DedispersionConfig::beams_per_batch,
               "Number of beams per batch")
          .def_readwrite("num_active_batches", &DedispersionConfig::num_active_batches,
               "Number of active batches")
          .def_readwrite("future_write_max_samples", &DedispersionConfig::future_write_max_samples,
               "Max time samples a WriteFiles RPC may extend into the future (0 = no future writes)")
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
        "        print(f'Tree {i}: primary_tree_index={tree.primary_tree_index}, dm_range=[{tree.dm_min:.1f}, {tree.dm_max:.1f}]')")
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
          .def_readonly("num_primary_trees", &DedispersionPlan::num_primary_trees,
               "Number of primary trees (same as config.num_primary_trees)")
          .def_readonly("beams_per_gpu", &DedispersionPlan::beams_per_gpu,
               "Number of beams processed per GPU (same as config.beams_per_gpu)")
          .def_readonly("beams_per_batch", &DedispersionPlan::beams_per_batch,
               "Number of beams per batch (same as config.beams_per_batch)")
          .def_readonly("num_active_batches", &DedispersionPlan::num_active_batches,
               "Number of active batches (same as config.num_active_batches)")
          .def_readonly("ntrees", &DedispersionPlan::ntrees,
               "Total number of stage2 trees (num_primary_trees + number of early triggers)")
          .def_readonly("nbits", &DedispersionPlan::nbits,
               "Number of bits per element (same as config.dtype.nbits)")
          .def_readonly("trees", &DedispersionPlan::trees,
               "Vector of DedispersionTree objects representing stage2 output trees.\n"
               "Length is ntrees. Each tree corresponds to one (primary tree, early trigger) pair.")
          .def_readonly("stage1_dd_rank", &DedispersionPlan::stage1_dd_rank,
               "Active dedispersion rank of each stage1 tree.\n"
               "Vector of length num_primary_trees. Stage1 trees are internal to dedispersion.")
          .def_readonly("stage1_amb_rank", &DedispersionPlan::stage1_amb_rank,
               "Ambient rank of each stage1 tree (= number of coarse frequency channels).\n"
               "Vector of length num_primary_trees.")
          .def_readonly("nelts_per_segment", &DedispersionPlan::nelts_per_segment,
               "Number of elements per GPU memory segment.\n"
               "Currently always constants::bytes_per_gpu_cache_line / sizeof(dtype)")
          .def_readonly("nbytes_per_segment", &DedispersionPlan::nbytes_per_segment,
               "Number of bytes per GPU memory segment.\n"
               "Currently always constants::bytes_per_gpu_cache_line")
          .def("to_yaml_string", &DedispersionPlan::to_yaml_string,
               py::arg("verbose") = false,
               py::arg("zones") = false,
               "Convert plan to YAML string representation.\n\n"
               "Args:\n"
               "    verbose: Include explanatory comments on fields\n"
               "    zones: Include the per-clag mega_ringbuf host/gpu zone breakdown\n\n"
               "Returns:\n"
               "    YAML string representation of the plan")
    ;

    // Returned by GpuDedisperser.acquire_output(). Must be registered
    // before the GpuDedisperser class_ block, so pybind11 knows how to
    // convert the return value when acquire_output's lambda is bound below.
    py::class_<GpuDedisperser::Outputs>(m, "GpuDedisperserOutputs",
        "Helper class, representing dedispersion outputs for one beam batch.\n"
        "Return value from FrbGrouper.get_output(ichunk, ibatch)\n"
        "\n"
        "Example usage::\n"
        "\n"
        "    with grouper.get_output(ichunk, ibatch) as outputs:\n"
        "        # 'outputs' has type GpuDedisperserOutputs\n"
        "        # Loop over dedispersion trees.\n"
        "        for itree, tree_out in enumerate(outputs.out_max):\n"
        "            # 'tree_out' has shape (beams_per_batch, coarse_ndm, coarse_ntime)\n"
        "\n"
        "Attributes (all read-only):\n"
        "\n"
        "- ``ichunk_zero_based`` (int) -- chunk index of this output, relative to the first dedisperser output.\n"
        "- ``ichunk_fpga_based`` (int) -- chunk index relative to FPGA seq 0 (= ichunk_zero_based + the producer's initial_chunk).\n"
        "- ``ibeam`` (int) -- beam index (NOT beam_id) of the first beam in this output.\n"
        "- ``out_max`` (list) -- length-ntrees list of peak-finding maximum-value arrays.\n"
        "- ``out_argmax`` (list) -- length-ntrees list of peak-finding argmax-token arrays.")
        // Member docstrings omitted on purpose -- documented in the class docstring's
        // bullet list above (see notes/docstrings.md).
        .def_readonly("ichunk_zero_based", &GpuDedisperser::Outputs::ichunk_zero_based)
        .def_readonly("ichunk_fpga_based", &GpuDedisperser::Outputs::ichunk_fpga_based)
        .def_readonly("ibeam", &GpuDedisperser::Outputs::ibeam)
        .def_readonly("out_max", &GpuDedisperser::Outputs::out_max)
        .def_readonly("out_argmax", &GpuDedisperser::Outputs::out_argmax);

    // GpuDedisperser. No class docstring here: it lives in the Python injector, since the
    // primary Python interface is the injected get_input()/get_output() context managers
    // (option 2 in notes/docstrings.md). Those context managers wrap the low-level
    // acquire/release methods below, which are bound with a leading underscore
    // (_acquire_input, ...) to mark them internal -- Python callers should use the context
    // managers, not call these directly.
    py::class_<GpuDedisperser, std::shared_ptr<GpuDedisperser>>(m, "GpuDedisperser")
          .def(py::init([](std::shared_ptr<DedispersionPlan> plan,
                          std::shared_ptr<CudaStreamPool> stream_pool,
                          int cuda_device_id,
                          long num_consumers,
                          long nbatches_out,
                          long nbatches_wt,
                          long initial_chunk) {
              GpuDedisperser::Params params;
              params.plan = plan;
              params.stream_pool = stream_pool;
              params.cuda_device_id = cuda_device_id;
              params.num_consumers = num_consumers;
              params.nbatches_out = nbatches_out;
              params.nbatches_wt = nbatches_wt;
              params.initial_chunk = initial_chunk;
              return GpuDedisperser::create(params);
          }),
          py::arg("plan"),
          py::arg("stream_pool"),
          py::arg("cuda_device_id"),
          py::arg("num_consumers") = -1,
          py::arg("nbatches_out") = 0,
          py::arg("nbatches_wt") = 0,
          py::arg("initial_chunk") = 0)
          .def_readonly("config", &GpuDedisperser::config)
          .def_readonly("plan", &GpuDedisperser::plan)
          .def_readonly("dtype", &GpuDedisperser::dtype)
          .def_readonly("nfreq", &GpuDedisperser::nfreq)
          .def_readonly("nt_in", &GpuDedisperser::nt_in)
          .def_readonly("total_beams", &GpuDedisperser::total_beams)
          .def_readonly("beams_per_batch", &GpuDedisperser::beams_per_batch)
          .def_readonly("nstreams", &GpuDedisperser::nstreams)
          .def_readonly("nbatches", &GpuDedisperser::nbatches)
          .def_readonly("ntrees", &GpuDedisperser::ntrees)
          .def_readonly("trees", &GpuDedisperser::trees)
          .def_readonly("resource_tracker", &GpuDedisperser::resource_tracker)
          .def_readonly("stream_pool", &GpuDedisperser::stream_pool)
          .def_property_readonly("Dcore", [](const GpuDedisperser &self) {
               std::vector<long> ret;
               for (const auto &k : self.cdd2_kernels)
                   ret.push_back(k->Dcore);
               return ret;
          }, "Per-tree internal time-downsampling factors of the GPU peak-finding\n"
             "kernels (length ntrees). Pass to ReferenceDedisperser(Dcore=...) to make\n"
             "the reference kernels mimic this GpuDedisperser exactly.")
          .def("allocate", &GpuDedisperser::allocate,
               py::arg("gpu_allocator"), py::arg("host_allocator"),
               py::call_guard<py::gil_scoped_release>(),   // GPU/host buffer allocation + worker spawn
               "Allocate GPU and host memory buffers for dedispersion.\n\n"
               "Args:\n"
               "    gpu_allocator: BumpAllocator for GPU memory\n"
               "    host_allocator: BumpAllocator for host memory")
          // Low-level acquire/release (raw stream_ptr). Underscore-prefixed to mark them
          // internal: the Python interface is the get_input()/get_output() context managers
          // (pirate_frb/pybind11_injections.py), which wrap these.
          //
          // All four release the GIL. Each can block on progress driven by a DIFFERENT
          // python thread (e.g. acquire_output blocks until the producer thread calls
          // release_input_and_launch_dd_kernels for the same seq_id, and vice versa
          // for back-pressure), so holding the GIL here would deadlock a multithreaded
          // producer/consumer driver. The lambda bodies are pure C++ (safe to run
          // GIL-free); the returned Array/Outputs views are converted to python after
          // the GIL is reacquired.
          .def("_acquire_input",
               [](GpuDedisperser &self, long seq_id, uintptr_t stream_ptr) {
                   cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
                   return self.acquire_input(seq_id, stream);
               },
               py::arg("seq_id"), py::arg("stream_ptr"),
               py::call_guard<py::gil_scoped_release>(),
               "Acquire the input buffer for seq_id and return a\n"
               "ksgpu.Array view of it. After this call 'stream' sees an empty\n"
               "input buffer ready for writing; the returned view is valid until\n"
               "the matching _release_input_and_launch_dd_kernels() call.")
          .def("_release_input_and_launch_dd_kernels",
               [](GpuDedisperser &self, long seq_id, uintptr_t stream_ptr) {
                   cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
                   self.release_input_and_launch_dd_kernels(seq_id, stream);
               },
               py::arg("seq_id"), py::arg("stream_ptr"),
               py::call_guard<py::gil_scoped_release>())
          .def("_acquire_output",
               [](GpuDedisperser &self, long consumer_id, long seq_id, uintptr_t stream_ptr) {
                   cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
                   return self.acquire_output(consumer_id, seq_id, stream);
               },
               py::arg("consumer_id"), py::arg("seq_id"), py::arg("stream_ptr"),
               py::call_guard<py::gil_scoped_release>(),
               "Acquire the output buffer for (consumer_id, seq_id) and return\n"
               "an Outputs object holding list-of-Array views of out_max and out_argmax.\n"
               "After this call 'stream' sees a full output buffer ready for reading;\n"
               "the returned views are valid until the matching _release_output() call.\n"
               "consumer_id must be in [0, num_consumers).")
          .def("_release_output",
               [](GpuDedisperser &self, long consumer_id, long seq_id, uintptr_t stream_ptr) {
                   cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
                   self.release_output(consumer_id, seq_id, stream);
               },
               py::arg("consumer_id"), py::arg("seq_id"), py::arg("stream_ptr"),
               py::call_guard<py::gil_scoped_release>())
          .def_static("test_random", &GpuDedisperser::test_random,
               py::call_guard<py::gil_scoped_release>())
          .def_static("test_one", &GpuDedisperser::test_one,
               py::arg("config"),
               py::arg("nchunks"),
               py::arg("nbatches_out") = 0,
               py::arg("nbatches_wt") = 0,
               py::arg("host_only") = false,
               py::call_guard<py::gil_scoped_release>())
          .def("time", &GpuDedisperser::time,
               py::arg("gpu_allocator"), py::arg("cpu_allocator"), py::arg("niterations"),
               py::call_guard<py::gil_scoped_release>(),   // minutes-long benchmark loop
               "Run timing benchmark.\n\n"
               "Must call allocate() first.\n\n"
               "Args:\n"
               "    gpu_allocator: BumpAllocator for GPU memory (for raw data arrays)\n"
               "    cpu_allocator: BumpAllocator for host memory (for raw data arrays)\n"
               "    niterations: Number of timing iterations")
          .def("fill_all_weights", &GpuDedisperser::fill_all_weights,
               py::arg("itree"), py::arg("pf_weights"),
               py::call_guard<py::gil_scoped_release>(),   // heavy CPU permute + blocking H2D copies
               "Copy host-side peak-finding weights to the GPU for one tree, filling all\n"
               "nbatches_wt weight slots. Must call allocate() first.\n\n"
               "Args:\n"
               "    itree: tree index, in [0, ntrees)\n"
               "    pf_weights: host ksgpu.Array<float>, shape (nbatches_wt, beams_per_batch,\n"
               "        t.ndm_wt, t.nt_wt, t.nprofiles, t.frequency_subbands.N) with\n"
               "        t = plan.trees[itree]. Weights may differ per slot and per beam.")
          .def("fill_analytic_weights", &GpuDedisperser::fill_analytic_weights,
               py::arg("freq_variances"),
               py::call_guard<py::gil_scoped_release>(),   // D2D copies + cudaDeviceSynchronize
               "Fill the peak-finding weight arrays with NON-random analytic weights,\n"
               "derived from the per-channel noise variances. All weight slots and beams\n"
               "get identical weights (unlike fill_all_weights). This is the weighting a\n"
               "real search uses, so peak-finding out_max values come out as SNRs. Must\n"
               "call allocate() first; blocks (cudaDeviceSynchronize) before returning.\n\n"
               "Args:\n"
               "    freq_variances: host ksgpu.Array<double>, length nfreq (all positive).\n"
               "        Typically XEngineMetadata.get_channel_variances().")
    ;

    // ReferenceDedisperser: CPU reference dedisperser (testing / variance studies).
    // Bound as the abstract base ReferenceDedisperserBase, via a factory py::init that
    // returns a concrete subclass as shared_ptr<Base>; Python sees one type.
    py::class_<ReferenceDedisperserBase, std::shared_ptr<ReferenceDedisperserBase>>(m, "ReferenceDedisperser",
        "CPU reference dedisperser (for testing and variance studies).\n"
        "\n"
        "Constructed directly:\n"
        "    ReferenceDedisperser(plan, sophistication, enable_variances=False, Dcore=None)\n"
        "\n"
        "'sophistication' (0, 1, or 2) selects the reference implementation:\n"
        "  0 -- one-stage dedispersion (instead of two stages); in downsampled trees,\n"
        "       compute twice as many DMs as necessary then drop the bottom half; each\n"
        "       early trigger is a separate tree (disregarding some input channels).\n"
        "  1 -- the same two-stage tree/lag structure as the plan; lags applied with a\n"
        "       per-tree ReferenceLagbuf (not ring/staging buffers); lags split into\n"
        "       segments + residuals, but not further split into chunks.\n"
        "  2 -- as close to the GPU implementation as possible.\n"
        "All three produce the same peak-finding output, modulo float roundoff.\n"
        "\n"
        "'Dcore' is the per-tree internal time-downsampling factor of the peak-finder\n"
        "(see PeakFindingKernel.hpp). To perfectly mimic a GpuDedisperser, pass the\n"
        "Dcore values from its GpuPeakFindingKernels. For a host-only reference (the\n"
        "common case), leave Dcore=None to use plan.trees[i].pf.time_downsampling.\n"
        "\n"
        "If enable_variances=True, the per-tree out_var buffers are allocated and filled\n"
        "by dedisperse() (per-chunk peak-finding variances; see ReferencePeakFindingKernel).")
        .def(py::init([](std::shared_ptr<DedispersionPlan> plan, int sophistication,
                         bool enable_variances, py::object Dcore_obj) {
            ReferenceDedisperserBase::Params p;
            p.plan = plan;
            p.sophistication = sophistication;
            p.enable_variances = enable_variances;
            if (!Dcore_obj.is_none())
                p.Dcore = Dcore_obj.cast<std::vector<long>>();
            // empty p.Dcore -> make() fills it from tree.pf.time_downsampling (host-only default)

            // The .cast<>() above is python API, so it must happen BEFORE this GIL
            // release (a py::call_guard would be a bug here). make() -- plan walk
            // plus large host allocations -- runs GIL-free.
            py::gil_scoped_release nogil;
            return ReferenceDedisperserBase::make(p);
        }), py::arg("plan"), py::arg("sophistication"),
            py::arg("enable_variances") = false, py::arg("Dcore") = py::none())
        // params fields (nested) exposed as read-only properties:
        .def_property_readonly("sophistication",   [](const ReferenceDedisperserBase &d){ return d.params.sophistication; })
        .def_property_readonly("Dcore",            [](const ReferenceDedisperserBase &d){ return d.params.Dcore; })
        .def_property_readonly("enable_variances", [](const ReferenceDedisperserBase &d){ return d.params.enable_variances; })
        // derived convenience members:
        .def_readonly("ntrees",          &ReferenceDedisperserBase::ntrees)
        .def_readonly("nfreq",           &ReferenceDedisperserBase::nfreq)
        .def_readonly("nt_in",           &ReferenceDedisperserBase::nt_in)
        .def_readonly("total_beams",     &ReferenceDedisperserBase::total_beams)
        .def_readonly("beams_per_batch", &ReferenceDedisperserBase::beams_per_batch)
        .def_readonly("nbatches",        &ReferenceDedisperserBase::nbatches)
        .def_readonly("config",          &ReferenceDedisperserBase::config)
        .def_readonly("trees",           &ReferenceDedisperserBase::trees)
        // Input/weights buffers (write into these zero-copy views before dedisperse()):
        .def_readonly("input_array",     &ReferenceDedisperserBase::input_array)
        .def_readonly("wt_arrays",       &ReferenceDedisperserBase::wt_arrays)
        // Per-tree ReferencePeakFindingKernels (e.g. for eval_tokens() in unit tests):
        .def_readonly("pf_kernels",      &ReferenceDedisperserBase::pf_kernels)
        // Outputs (read after dedisperse()). out_var is empty unless enable_variances was set:
        .def_readonly("out_max",         &ReferenceDedisperserBase::out_max)
        .def_readonly("out_argmax",      &ReferenceDedisperserBase::out_argmax)
        // out_var elements are empty when variances are disabled, and the Array->numpy caster
        // rejects zero-size arrays -- so map empty -> None (the list is always length ntrees).
        .def_property_readonly("out_var", [](const ReferenceDedisperserBase &d) {
            py::list out;
            for (const auto &v : d.out_var) {
                if (v.size == 0)
                    out.append(py::none());
                else
                    out.append(py::cast(v));
            }
            return out;
        })
        .def("dedisperse",               &ReferenceDedisperserBase::dedisperse,
             py::arg("ichunk"), py::arg("ibatch"),
             py::call_guard<py::gil_scoped_release>(),   // heavy CPU dedispersion + peak-finding
             "Dedisperse one (ichunk, ibatch). Fills out_max/out_argmax (and out_var if enabled).")
    ;
}
