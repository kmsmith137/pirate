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
#include <optional>
#include <pybind11/stl.h>

#include <ksgpu/pybind11.hpp>

#include "../include/pirate/constants.hpp"
#include "../include/pirate/CoalescedDdKernel2.hpp"    // GpuDedisperser::cdd2_kernels
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
    void register_varmap_bindings(pybind11::module &m);
    void register_kernel_bindings(pybind11::module &m);
    void register_casm_bindings(pybind11::module &m);
    void register_chime_bindings(pybind11::module &m);
    void register_loose_ends_bindings(pybind11::module &m);
    void register_simpulse_bindings(pybind11::module &m);
    void register_utils_bindings(pybind11::module &m);
}


// The two decode methods return their results through reference arguments, which python
// wants as a tuple.
static py::tuple _plan_decode_argmax(const DedispersionPlan &plan, uint token, long itree,
                                     long Dcore, long idm_coarse, long itime_coarse)
{
    long fmin, fmax, tlo, thi, p;
    plan.decode_argmax(token, itree, Dcore, idm_coarse, itime_coarse, fmin, fmax, tlo, thi, p);
    return py::make_tuple(fmin, fmax, tlo, thi, p);
}


static py::tuple _plan_decode_argmax2(const DedispersionPlan &plan, long itree,
                                      long fmin, long fmax, long tlo, long thi, long p)
{
    double freq_lo_MHz, freq_hi_MHz, dm, timestamp_samp, width_samp;
    plan.decode_argmax2(itree, fmin, fmax, tlo, thi, p,
                        freq_lo_MHz, freq_hi_MHz, dm, timestamp_samp, width_samp);
    return py::make_tuple(freq_lo_MHz, freq_hi_MHz, dm, timestamp_samp, width_samp);
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
    register_varmap_bindings(m);
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
        .def_readonly_static("frb_dm0", &constants::frb_dm0,
            "DM-scale offset (pc cm^-3) for the log-uniform DM distribution of simulated FRBs.")
        .def_readonly_static("inactive_file_stream_capacity", &constants::inactive_file_stream_capacity,
            "Number of inactive (expired/cancelled) FileStreams retained by an FrbServer for "
            "ShowStreams history; the oldest are dropped beyond this.")
        .def_readonly_static("assembled_frame_allocator_queue_size", &constants::assembled_frame_allocator_queue_size,
            "Steady-state bound on the AssembledFrameAllocator's pre-init queue "
            "(its worker's throttle, and the memory headroom held ahead of consumption).")
        .def_readonly_static("assembled_frame_allocator_initial_size", &constants::assembled_frame_allocator_initial_size,
            "Number of frame sets the AssembledFrameAllocator's worker pre-allocates at "
            "startup when throw_exception_if_empty=True; doubles as a fail-fast "
            "pool-size check.")
        .def_readonly_static("default_server_max_unprocessed_chunks", &constants::default_server_max_unprocessed_chunks,
            "Default keep-up bound, in time chunks: FrbServer error-stops if "
            "rb_assembled - rb_processed exceeds FrbServer's max_unprocessed_chunks "
            "param, which defaults to this constant. The FakeXEngine pacing lookahead "
            "(pacing_budget_chunks) defaults to this constant minus one.")
        .def_readonly_static("grouper_ping_timeout_ms", &constants::grouper_ping_timeout_ms,
            "Timeout (ms) for FrbGrouperClient.ping(): the early channel-level connectivity "
            "check done before bump allocation, to fail fast if the grouper isn't running.")
        .def_readonly_static("grouper_connect_timeout_ms", &constants::grouper_connect_timeout_ms,
            "Timeout (ms) for the real FrbGrouperClient reconnect (done in grouper_send_thread "
            "just before the Handshake).")
        .def_readonly_static("default_poll_cadence_ms", &constants::default_poll_cadence_ms,
            "Default cadence (ms) for stop/cancel poll loops, e.g. the timeout passed to "
            "FrbServer.poll_from_python() in run_server.py to keep Ctrl-C responsive. See "
            "include/pirate/constants.hpp for the full list of use sites.")
        .def_readonly_static("default_print_cadence_sec", &constants::default_print_cadence_sec,
            "Default seconds between console/status prints in a report-then-pause loop "
            "(e.g. run_rpc_status, Hwtest). Not a stop-poll cadence. See constants.hpp.")
        .def_readonly_static("default_shutdown_timeout_sec", &constants::default_shutdown_timeout_sec,
            "Default seconds to wait for a graceful shutdown step (thread/process join, "
            "SIGTERM->SIGKILL grace) before escalating. See constants.hpp.")
        .def_readonly_static("grpc_reconnect_backoff_ms", &constants::grpc_reconnect_backoff_ms,
            "Cap (ms) on a gRPC channel's reconnect backoff (initial == max). Used by "
            "FrbGrouper channel args and FrbSifterClient. See constants.hpp.")
        .def_readonly_static("grpc_forced_shutdown_deadline_ms", &constants::grpc_forced_shutdown_deadline_ms,
            "Deadline (ms) for grpc::Server::Shutdown() during teardown, to avoid the "
            "deadline-less Shutdown()'s ~seconds internal block. See constants.hpp.")
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
          .def_static("from_yaml_string", &DedispersionConfig::from_yaml_string,
                      py::arg("yaml_string"),
                      "Load DedispersionConfig from a YAML string: the inverse of\n"
                      "to_yaml_string(). Use this wherever a config travels as a string\n"
                      "rather than a file, e.g. one embedded in a variance-map file by\n"
                      "pirate_frb.varmap.asdf_io.")
          .def_static("make_random",
               [](int max_toplevel_rank, int max_early_triggers, bool gpu_valid, bool verbose,
                  bool force_float32, bool no_host_mega_ringbuf) {
                   DedispersionConfig::RandomArgs args;
                   args.max_toplevel_rank = max_toplevel_rank;
                   args.max_early_triggers = max_early_triggers;
                   args.gpu_valid = gpu_valid;
                   args.verbose = verbose;
                   args.force_float32 = force_float32;
                   args.no_host_mega_ringbuf = no_host_mega_ringbuf;
                   return DedispersionConfig::make_random(args);
               },
               py::arg("max_toplevel_rank") = 10,
               py::arg("max_early_triggers") = 5,
               py::arg("gpu_valid") = true,
               py::arg("verbose") = false,
               py::arg("force_float32") = false,
               py::arg("no_host_mega_ringbuf") = false,
               "Generate a random DedispersionConfig for testing.\n\n"
               "Args:\n"
               "    max_toplevel_rank: Bounds toplevel_tree_rank (default=10)\n"
               "    max_early_triggers: Max number of early triggers (0 to disable, default=5)\n"
               "    gpu_valid: Generate GPU-valid configuration (default=True)\n"
               "    verbose: Print debug info (default=False)\n"
               "    force_float32: Draw only float32 configs (default=False)\n"
               "    no_host_mega_ringbuf: Leave max_gpu_clag at its default, keeping the\n"
               "        MegaRingbuf pure-GPU (default=False)\n\n"
               "The last two narrow the draw so that the config is usable by the GPU\n"
               "brute-force variance-map sweep (pirate_frb.varmap.brute_force).\n\n"
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
          // dtype: reads return numpy.dtype, writes accept strings/numpy dtypes/None,
          // via ksgpu's type_caster<ksgpu::Dtype> (no wrapper needed on either side).
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
    py::class_<DedispersionPlan, std::shared_ptr<DedispersionPlan>> plan_cls(m, "DedispersionPlan",
        "Everything the dedisperser needs, derived from a DedispersionConfig.\n\n"
        "Most of a plan is low-level GPU data (kernel params, buffer layouts, the ring\n"
        "buffer), which python callers rarely touch. The exception is ``trees``, the\n"
        "per-(primary tree, early trigger) output geometry, and the decoding methods built\n"
        "on it -- and those are available with no GPU, from a \"minimal\" plan::\n\n"
        "    plan = DedispersionPlan(config, mega_ringbuf=False, gpu_kernels=False)\n\n"
        "Both constructor flags default to True, which builds a complete plan and needs a\n"
        "CUDA device; see the constructor docstring for what each one turns off.\n\n"
        "The plan is immutable once constructed and is shared between dedisperser instances.\n\n"
        "Example::\n\n"
        "    config = DedispersionConfig.from_yaml('config.yaml')\n"
        "    plan = DedispersionPlan(config)\n"
        "    print(f'Plan has {plan.ntrees} trees')\n"
        "    for i, tree in enumerate(plan.trees):\n"
        "        print(f'Tree {i}: primary_tree_index={tree.primary_tree_index}, dm_range=[{tree.dm_min:.1f}, {tree.dm_max:.1f}]')");

    plan_cls
          // The C++ constructor takes a DedispersionPlan::Params, but that struct has only
          // two members, so python spells them as keyword arguments instead of wrapping it.
          .def(py::init([](const DedispersionConfig &config, bool mega_ringbuf, bool gpu_kernels) {
              DedispersionPlan::Params params;
              params.mega_ringbuf = mega_ringbuf;
              params.gpu_kernels = gpu_kernels;
              return new DedispersionPlan(config, params);
          }),
               py::arg("config"), py::arg("mega_ringbuf") = true, py::arg("gpu_kernels") = true,
               "Create a DedispersionPlan from a configuration.\n\n"
               "The two flags select how much of the plan gets initialized. The default\n"
               "(both True) is a 'complete' plan, which needs a CUDA device. Turning a flag\n"
               "off leaves members uninitialized, in exchange for a plan that is cheaper --\n"
               "and, with ``mega_ringbuf=False``, constructible with no CUDA device at all.\n\n"
               "Both flags False gives a \"minimal\" plan: config-derived scalars, stage1\n"
               "ranks and ``trees`` only. This is how GPU-less code (``pirate_frb.varmap``,\n"
               "the grouper) gets at the dedispersion trees::\n\n"
               "    plan = DedispersionPlan(config, mega_ringbuf=False, gpu_kernels=False)   # \"minimal\" plan\n\n"
               "Args:\n"
               "    config: DedispersionConfig object (must be validated)\n"
               "    mega_ringbuf: if False, then ``DedispersionPlan.mega_ringbuf`` is None.\n"
               "        (Constructing a MegaRingbuf allocates page-locked host memory, so it\n"
               "        needs a CUDA device; this is the flag that decides whether a plan can\n"
               "        be built without one.)\n"
               "    gpu_kernels: if False, then all gpu kernel params\n"
               "        (``tree_gridding_kernel_params`` ... ``h2h_copy_kernel_params``) are\n"
               "        uninitialized. Requires ``mega_ringbuf=True``.")
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
               "Length is ntrees. Each tree is one (primary tree, early trigger) pair.\n"
               "Ordered by primary tree, then by DECREASING early-trigger level.\n"
               "\n"
               "Note this is a fresh list of COPIES on every attribute access, so code that\n"
               "needs one tree repeatedly should cache it.")
          .def("dedispersion_tree_index", &DedispersionPlan::dedispersion_tree_index,
               py::arg("primary_tree_index"), py::arg("early_trigger_level"),
               "Returns the 'itree' of the tree with this (primary_tree_index,\n"
               "early_trigger_level) pair, i.e. the inverse of reading those two members of\n"
               "trees[itree]. Throws if either argument is out of range.")
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
          // Low-level kernel parameters. These are for callers that drive the GPU kernels by
          // hand instead of using a GpuDedisperser (e.g. varmap.brute_force._GpuSweep).
          // They encode the ring-buffer lag structure, so pass them through rather than
          // reconstructing them.
          .def_readonly("mega_ringbuf", &DedispersionPlan::mega_ringbuf,
               "The MegaRingbuf: the ring buffer through which stage 1 feeds stage 2.")
          .def_readonly("tree_gridding_kernel_params", &DedispersionPlan::tree_gridding_kernel_params,
               "TreeGriddingKernelParams for the (single) tree gridding kernel.")
          .def_readonly("lds_params", &DedispersionPlan::lds_params,
               "LaggedDownsamplingKernelParams for this plan. Meaningful only when\n"
               "num_primary_trees > 1, though it is filled (and valid) either way.")
          .def_readonly("stage1_dd_buf_params", &DedispersionPlan::stage1_dd_buf_params,
               "DedispersionBufferParams for the stage-1 input buffers: nbuf ==\n"
               "num_primary_trees, with entry ipri the input of primary tree ipri.")
          .def_readonly("stage1_dd_kernel_params", &DedispersionPlan::stage1_dd_kernel_params,
               "List of DedispersionKernelParams, length num_primary_trees.")
          .def_readonly("stage2_dd_kernel_params", &DedispersionPlan::stage2_dd_kernel_params,
               "List of DedispersionKernelParams, length ntrees.")
          .def("to_yaml_string", &DedispersionPlan::to_yaml_string,
               py::arg("verbose") = false,
               py::arg("zones") = false,
               "Convert plan to YAML string representation.\n\n"
               "Args:\n"
               "    verbose: Include explanatory comments on fields\n"
               "    zones: Include the per-clag mega_ringbuf host/gpu zone breakdown\n\n"
               "Returns:\n"
               "    YAML string representation of the plan")
          .def_static("from_yaml_string", &DedispersionPlan::from_yaml_string,
               py::arg("config"), py::arg("plan_yaml"),
               "Rebuild a producer's plan, for a consumer that may be running a different\n"
               "pirate_frb build.\n\n"
               "Returns a \"minimal\" plan (see the constructor) built from ``config``,\n"
               "cross-checked field by field against the yaml; a disagreement raises, naming\n"
               "the field and both values. Nothing is adopted from the yaml: a plan is a pure\n"
               "function of its config.\n\n"
               "Decoding the producer's out_argmax tokens additionally needs its per-tree\n"
               "Dcore, which is a property of its compiled kernels rather than of the plan\n"
               "and travels as its own handshake field (see FrbGrouper.dcores).\n\n"
               "Args:\n"
               "    config: the producer's DedispersionConfig (from the same handshake).\n"
               "    plan_yaml: the producer's ``to_yaml_string()`` output.")
          .def("decode_argmax", &_plan_decode_argmax,
               py::arg("token"), py::arg("itree"), py::arg("Dcore"),
               py::arg("idm_coarse"), py::arg("itime_coarse"),
               "Decode an ``out_argmax`` token into the winning trial parameters, i.e. the\n"
               "(subband, peak-finding profile, fine-grained dm, fine-grained arrival time)\n"
               "responsible for the coarse-grained maximum in ``trees[itree].out_max``.\n\n"
               "Raises on out-of-range indices or a malformed token. See DedispersionPlan.hpp\n"
               "for the full spec.\n\n"
               "Args:\n"
               "    token: uint32 token from tree ``itree``'s ``out_argmax`` array.\n"
               "    itree: tree index, in ``[0, ntrees)``.\n"
               "    Dcore: internal time-downsampling factor of the peak-finding kernel that\n"
               "        WROTE the token -- ``GpuDedisperser.Dcores[itree]``,\n"
               "        ``ReferenceDedisperser.Dcores[itree]``, or ``FrbGrouper.dcores[itree]``.\n"
               "        It is a property of that kernel, not of this plan, which is why it is\n"
               "        an argument; a wrong (but legal) value silently mis-decodes fine times.\n"
               "    idm_coarse: dm index into ``out_max`` / ``out_argmax``, in\n"
               "        ``[0, trees[itree].ndm_out)``.\n"
               "    itime_coarse: time index into ``out_max`` / ``out_argmax``, in\n"
               "        ``[0, trees[itree].nt_out)``.\n\n"
               "Returns:\n"
               "    The tuple ``(fmin, fmax, tlo, thi, p)``, all TOPLEVEL-relative. ``fmin``\n"
               "    and ``fmax`` are tree-freq channels of the toplevel gridding, spanning the\n"
               "    winning frequency subband. ``tlo`` and ``thi`` are full-resolution time\n"
               "    samples with ``t=0`` at the start of the current chunk; they are EXCLUSIVE\n"
               "    trailing edges, and are frequently negative, since dedispersion delays\n"
               "    usually exceed the chunk length (they then refer to earlier chunks). ``p``\n"
               "    is the winning peak-finding profile index.")
          .def("decode_argmax2", &_plan_decode_argmax2,
               py::arg("itree"), py::arg("fmin"), py::arg("fmax"), py::arg("tlo"),
               py::arg("thi"), py::arg("p"),
               "Convert ``decode_argmax()`` output to physical parameters.\n\n"
               "The arguments ``fmin``, ``fmax``, ``tlo``, ``thi``, ``p`` are the tuple\n"
               "returned by ``decode_argmax()`` on the same ``itree``. See\n"
               "DedispersionPlan.hpp for the full spec.\n\n"
               "Returns:\n"
               "    The tuple ``(freq_lo_MHz, freq_hi_MHz, dm, timestamp_samp, width_samp)`` --\n"
               "    the low/high radio frequency of the winning subband, the dispersion measure\n"
               "    in pc/cm^3, the arrival time, and the winning peak-finder width in toplevel\n"
               "    time samples. ``timestamp_samp`` is the estimated arrival time of the pulse\n"
               "    center at the lowest radio frequency, in toplevel samples with ``t=0`` at\n"
               "    the START OF THE CURRENT CHUNK (not at ``fpga_seq=0``); the caller adds the\n"
               "    chunk's absolute FPGA start. It is NOT confined to ``[0, nt_in)`` -- an\n"
               "    early-trigger tree extrapolates to the band bottom, so the time can lie past\n"
               "    the chunk end, and the finite peak-finder kernel width can push an event\n"
               "    detected near the chunk start slightly before it.")
          .def("compute_steady_state_it0", &DedispersionPlan::compute_steady_state_it0,
               py::arg("itree"),
               "Time index at which each of tree ``itree``'s DM channels becomes\n"
               "\"steady-state\".\n\n"
               "A dedispersion output element ``(ichunk, ibeam, idm, it)`` of the tree is\n"
               "steady-state, i.e. unaffected by the zero-padding before the start of the\n"
               "acquisition, iff ``ichunk*nt_out + it >= result[idm]``. Earlier elements are\n"
               "computed from sums whose footprint extends past the start of the acquisition,\n"
               "so their ``out_max`` values are artificially low -- warmup artifacts, not real\n"
               "triggers.\n\n"
               "Returns:\n"
               "    A 1-d int64 array of shape ``(trees[itree].ndm_out,)``, in host memory.\n"
               "    Needs no CUDA device.")
          .def("n_index_mapping", &DedispersionPlan::n_index_mapping,
               py::arg("iparent"), py::arg("ichild"),
               "Subband index mapping between two trees of this plan: a list of length\n"
               "``trees[ichild].frequency_subbands.N``, whose entry ``n_c`` is the parent\n"
               "subband searching the same toplevel band.\n\n"
               "Bands are matched by toplevel range ``(n_to_toplevel_flo, n_to_toplevel_fhi)``,\n"
               "so trees of different rank are comparable. Raises if the child searches a band\n"
               "the parent does not; the message says so if the arguments look reversed.\n\n"
               "The usual pair is an early-trigger tree and its ``(primary_tree_index, 0)``\n"
               "parent, but equal ``primary_tree_index`` is not required, and the identity\n"
               "case ``(t, t)`` works. Calling both ways round tests set EQUALITY.")
          .def("m_index_mapping", &DedispersionPlan::m_index_mapping,
               py::arg("iparent"), py::arg("ichild"),
               "As ``n_index_mapping()``, but over multiplets: a list of length\n"
               "``trees[ichild].frequency_subbands.M``, whose entry ``m_c`` is the parent\n"
               "multiplet with the same band and the same fine-DM index within it.\n\n"
               "Additionally raises unless matched bands have the same subband level.\n\n"
               "Does NOT check ``nprofiles`` or the coarse-DM count ``2**(tree_rank -\n"
               "pf_rank)``: those are not subband geometry, and a caller building a row map\n"
               "over (dm, multiplet, profile) must check them itself.")
    ;

    // Returned by GpuDedisperser.acquire_output(). Must be registered
    // before the GpuDedisperser class_ block, so pybind11 knows how to
    // convert the return value when acquire_output's lambda is bound below.
    // No class docstring here: GpuDedisperserOutputs's docstring lives in the
    // Python injector (pirate_frb/core/GpuDedisperserOutputs.py), which is also
    // where out_max/out_argmax get their cached_property accessors (option 2 in
    // notes/docstrings.md).
    py::class_<GpuDedisperser::Outputs>(m, "GpuDedisperserOutputs", py::dynamic_attr())
        // Member docstrings omitted on purpose -- documented in the class docstring
        // (which lives in the injector; see notes/docstrings.md).
        .def_readonly("ichunk_zero_based", &GpuDedisperser::Outputs::ichunk_zero_based)
        .def_readonly("ichunk_fpga_based", &GpuDedisperser::Outputs::ichunk_fpga_based)
        .def_readonly("ibeam", &GpuDedisperser::Outputs::ibeam)
        // out_max/out_argmax are exposed here under underscore names; the Python
        // injector (pirate_frb/core/GpuDedisperserOutputs.py) wraps them in
        // @cached_property accessors named out_max/out_argmax, so the
        // vector<Array> -> list conversion runs once per Outputs instead of on
        // every attribute read. py::dynamic_attr() (above) gives each instance the
        // __dict__ that cached_property caches into.
        //
        // CACHING CONTRACT: correct only because acquire_output() returns a fresh
        // Outputs BY VALUE, so pybind mints a new Python object (empty __dict__)
        // per batch. Returning a reference to a persistent Outputs would make the
        // cache serve a prior batch's recycled arrays -- see acquire_output() in
        // Dedisperser.hpp / FrbGrouper.hpp.
        .def_readonly("_out_max", &GpuDedisperser::Outputs::out_max)
        .def_readonly("_out_argmax", &GpuDedisperser::Outputs::out_argmax);

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
          .def_readonly("Dcores", &GpuDedisperser::Dcores,
               "Per-tree internal time-downsampling factors of the GPU peak-finding kernels\n"
               "(length ntrees). Compiled into the cdd2 kernels, so they cannot be predicted\n"
               "from the plan: pass Dcores[itree] to DedispersionPlan.decode_argmax() to\n"
               "decode this dedisperser's out_argmax tokens, and Dcores= to\n"
               "ReferenceDedisperser to make its tokens identical to these kernels'.")
          .def("allocate", &GpuDedisperser::allocate,
               py::arg("gpu_allocator"), py::arg("host_allocator"),
               py::call_guard<py::gil_scoped_release>(),   // GPU/host buffer allocation + worker spawn
               "Allocate GPU and host memory buffers for dedispersion.\n\n"
               "Args:\n"
               "    gpu_allocator: BumpAllocator for GPU memory\n"
               "    host_allocator: BumpAllocator for host memory")
          // Low-level acquire/release (raw stream_ptr). Underscore-prefixed to mark them
          // internal: the Python interface is the get_input()/get_output() context managers
          // (pirate_frb/GpuDedisperser.py), which wrap these.
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
        "    ReferenceDedisperser(plan, sophistication, tree_domain_input=False)\n"
        "\n"
        "'sophistication' (0, 1, or 2) selects the reference implementation:\n"
        "\n"
        "* 0 -- one-stage dedispersion (instead of two stages); in downsampled trees,\n"
        "  compute twice as many DMs as necessary then drop the bottom half; each\n"
        "  early trigger is a separate tree (disregarding some input channels).\n"
        "* 1 -- the same two-stage tree/lag structure as the plan; lags applied with a\n"
        "  per-tree ReferenceLagbuf (not ring/staging buffers); lags split into\n"
        "  segments + residuals, but not further split into chunks.\n"
        "* 2 -- as close to the GPU implementation as possible.\n"
        "\n"
        "All three produce the same peak-finding output, modulo float roundoff.\n"
        "\n"
        "Dcores (list of int, length ntrees) sets the peak-finders' internal time\n"
        "downsampling, which is the time granularity of their out_argmax tokens. The\n"
        "default is one profile evaluation per output bin at the coarsest level\n"
        "(trees[i].time_downsampling). To make the tokens identical to a GpuDedisperser's,\n"
        "pass its ``Dcores``: the GPU values are compiled into the cdd2 kernels and cannot\n"
        "be predicted from the plan.\n"
        "\n"
        "FOOTGUN: in a tree with K = ``pf_kernels[itree].xdm_rank`` > 0, the fourth byte of an\n"
        "``out_argmax[itree]`` token is an extra-DM index mu, and the winning input DM row is\n"
        "(d << K) | mu rather than the output row d. That is how the tokens are made identical\n"
        "to a GpuDedisperser's. K is zero except in early-trigger trees, so code which\n"
        "predates early triggers can look correct and not be. Use ``plan.decode_argmax()``\n"
        "rather than parsing tokens by hand.\n"
        "\n"
        "``out_sb[itree]`` is free of that footgun and is what variance calculations should\n"
        "use. It is the tree's subband array after dedisperse(), shape ``(beams_per_batch,\n"
        "Dpf, M, t.nt_ds)`` with ``Dpf = 2^(r-R)`` the full coarse-DM count -- the same\n"
        "layout GpuSbDedispersionKernel writes, whatever K is. Pair it with\n"
        "ReferencePfSquare (reshaping the (Dpf, M) pair into its 'ndm' row count) to get a\n"
        "CPU sweep that mirrors the GPU one. It is a view into internal storage, which the\n"
        "next dedisperse() overwrites.\n"
        "\n"
        "If tree_domain_input=True, the tree gridding kernel is skipped: input_array has\n"
        "shape (beams_per_batch, 2^toplevel_tree_rank, nt_in) and is interpreted as an\n"
        "already-gridded toplevel tree-domain array. Used by unit tests that inject probes\n"
        "into specific tree-freq channels (see test_decode_argmax).")
        .def(py::init([](std::shared_ptr<DedispersionPlan> plan, int sophistication,
                         bool tree_domain_input, std::optional<std::vector<long>> Dcores) {
            ReferenceDedisperserBase::Params p;
            p.plan = plan;
            p.sophistication = sophistication;
            p.tree_domain_input = tree_domain_input;
            if (Dcores.has_value())
                p.Dcores = *Dcores;

            // make() -- plan walk plus large host allocations -- runs GIL-free.
            py::gil_scoped_release nogil;
            return ReferenceDedisperserBase::make(p);
        }), py::arg("plan"), py::arg("sophistication"),
            py::arg("tree_domain_input") = false,
            py::arg("Dcores") = py::none())
        // params fields (nested) exposed as read-only properties:
        .def_property_readonly("sophistication",   [](const ReferenceDedisperserBase &d){ return d.params.sophistication; })
        .def_property_readonly("tree_domain_input", [](const ReferenceDedisperserBase &d){ return d.params.tree_domain_input; })
        .def_readonly("Dcores", &ReferenceDedisperserBase::Dcores,
            "Per-tree peak-finder internal time-downsampling factors (length ntrees):\n"
            "the constructor's Dcores argument, or the default described there.")
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
        // Outputs (read after dedisperse()):
        .def_readonly("out_max",         &ReferenceDedisperserBase::out_max)
        .def_readonly("out_argmax",      &ReferenceDedisperserBase::out_argmax)
        .def_readonly("out_sb",          &ReferenceDedisperserBase::out_sb)
        .def("dedisperse",               &ReferenceDedisperserBase::dedisperse,
             py::arg("ichunk"), py::arg("ibatch"),
             py::call_guard<py::gil_scoped_release>(),   // heavy CPU dedispersion + peak-finding
             "Dedisperse one (ichunk, ibatch). Fills out_max/out_argmax and out_sb.")
    ;
}
