// Python bindings for core classes (pirate_frb.core subpackage).
// See pirate_pybind11.cu for the main module definition.

#define PY_ARRAY_UNIQUE_SYMBOL PyArray_API_pirate
#define NO_IMPORT_ARRAY  // Secondary file: don't call _import_array()
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <pybind11/stl.h>
#include <ksgpu/pybind11.hpp>

#include "../include/pirate/AssembledFrame.hpp"
#include "../include/pirate/BumpAllocator.hpp"
#include "../include/pirate/CudaStreamPool.hpp"
#include "../include/pirate/SlabAllocator.hpp"
#include "../include/pirate/DedispersionConfig.hpp"  // for PeakFindingConfig, EarlyTrigger
#include "../include/pirate/DedispersionTree.hpp"
#include "../include/pirate/FrequencySubbands.hpp"
#include "../include/pirate/ResourceTracker.hpp"
#include "../include/pirate/system_utils.hpp"  // set_thread_affinity, get_thread_affinity
#include "../include/pirate/XEngineMetadata.hpp"
#include "../include/pirate/FakeXEngine.hpp"
#include "../include/pirate/Receiver.hpp"
#include "../include/pirate/FrbServer.hpp"
#include "../include/pirate/FileWriter.hpp"
#include "../include/pirate/HwtestSender.hpp"
#include "../include/pirate/Hwtest.hpp"

using namespace std;
using namespace ksgpu;
using namespace pirate;
namespace py = pybind11;


namespace pirate {

void register_core_bindings(pybind11::module &m)
{
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

    // SlabAllocator: Thread-safe pool allocator for fixed-size slabs
    // Note: Python injections in pirate_frb/pybind11_injections.py handle aflags conversion.
    // Note: get_slab() returns shared_ptr<void>, which is not wrapped (per pybind11 guidelines).
    py::class_<SlabAllocator, std::shared_ptr<SlabAllocator>>(m, "SlabAllocator",
        "Thread-safe pool allocator for fixed-size memory slabs.\n\n"
        "Pre-allocates a large memory region subdivided into fixed-size slabs.\n"
        "Slabs are returned to the pool when their reference count drops to zero.\n\n"
        "Modes:\n"
        "  - capacity >= 0: Pre-allocates base region, slabs share this memory\n"
        "  - capacity < 0: Dummy mode, each get_slab() allocates fresh memory")
        .def(py::init(static_cast<std::shared_ptr<SlabAllocator>(*)(int, long)>(&SlabAllocator::create)),
            py::arg("aflags"), py::arg("capacity"),
            "Create allocator with new memory.\n\n"
            "Args:\n"
            "    aflags: Memory allocation flags (af_gpu, af_rhost, etc.)\n"
            "    capacity: Bytes to pre-allocate (>= 0) or < 0 for dummy mode")
        .def(py::init(static_cast<std::shared_ptr<SlabAllocator>(*)(BumpAllocator &, long)>(&SlabAllocator::create)),
            py::arg("bump_allocator"), py::arg("nbytes"),
            "Create allocator using memory from a BumpAllocator.\n\n"
            "Args:\n"
            "    bump_allocator: Source of memory (must not be in dummy mode)\n"
            "    nbytes: Bytes to allocate from the BumpAllocator")
        .def("num_free_slabs", &SlabAllocator::num_free_slabs,
            "Number of slabs currently available. Throws in dummy mode.")
        .def("num_total_slabs", &SlabAllocator::num_total_slabs,
            "Total number of slabs in the pool. Throws in dummy mode.")
        .def("get_slab_size", &SlabAllocator::get_slab_size,
            "Established slab size in bytes, or -1 if not yet established.")
        .def("is_dummy", &SlabAllocator::is_dummy,
            "True if in dummy mode (capacity < 0).")
        .def_readonly("aflags", &SlabAllocator::aflags,
            "Memory allocation flags")
        .def_readonly("capacity", &SlabAllocator::capacity,
            "Total capacity in bytes, or < 0 for dummy mode")
    ;

    // AssembledFrame: data frame containing beamformed data for one (time_chunk, beam_id) pair.
    // The 'data' member is int4 with shape (nfreq, ntime), but exposed to Python as uint8
    // with shape (nfreq, ntime/2) since numpy doesn't support sub-byte dtypes.
    py::class_<AssembledFrame, std::shared_ptr<AssembledFrame>>(m, "AssembledFrame",
        "Data frame containing beamformed data for one (time_chunk, beam_id) pair.")
        .def_readonly("nfreq", &AssembledFrame::nfreq)
        .def_readonly("ntime", &AssembledFrame::ntime)
        .def_readonly("beam_id", &AssembledFrame::beam_id)
        .def_readonly("time_chunk_index", &AssembledFrame::time_chunk_index)
        .def_property_readonly("data",
            [](const AssembledFrame &self) {
                // Convert int4 array (nfreq, ntime) to uint8 array (nfreq, ntime/2).
                Array<uint8_t> arr;
                arr.data = static_cast<uint8_t *>(self.data.data);
                arr.ndim = 2;
                arr.shape[0] = self.nfreq;
                arr.shape[1] = self.ntime / 2;
                arr.size = self.nfreq * (self.ntime / 2);
                arr.strides[0] = self.ntime / 2;
                arr.strides[1] = 1;
                arr.dtype = Dtype::native<uint8_t>();
                arr.aflags = self.data.aflags;
                arr.base = self.data.base;
                arr.check_invariants("AssembledFrame::data getter");
                return arr;
            },
            "Data as uint8 array with shape (nfreq, ntime/2).\n\n"
            "The underlying data is int4 (nfreq, ntime), packed as uint8.")
        .def_static("test_asdf", &AssembledFrame::test_asdf,
            "Unit test for ASDF file I/O.\n\n"
            "Creates a random AssembledFrame, writes to temp file, reads back,\n"
            "and verifies the data matches.")
    ;

    // AssembledFrameAllocator: allocates AssembledFrames for multiple consumers.
    // Designed for multi-threaded use, but works fine with a single Python consumer.
    py::class_<AssembledFrameAllocator, std::shared_ptr<AssembledFrameAllocator>>(m, "AssembledFrameAllocator",
        "Allocates AssembledFrames for multiple consumers.\n\n"
        "Each consumer calls initialize() once, then get_frame() in a loop.\n"
        "All consumers receive the same sequence of frames (same shared_ptr).")
        .def(py::init<const std::shared_ptr<SlabAllocator> &, int>(),
            py::arg("slab_allocator"), py::arg("num_consumers"))
        .def_readonly("nfreq", &AssembledFrameAllocator::nfreq)
        .def_readonly("time_samples_per_chunk", &AssembledFrameAllocator::time_samples_per_chunk)
        .def_readonly("beam_ids", &AssembledFrameAllocator::beam_ids)
        .def("initialize", &AssembledFrameAllocator::initialize,
            py::arg("consumer_id"), py::arg("nfreq"),
            py::arg("time_samples_per_chunk"), py::arg("beam_ids"),
            "Initialize a consumer. Must be called once per consumer before get_frame().\n\n"
            "All consumers must provide the same nfreq, time_samples_per_chunk, and beam_ids.")
        .def("get_frame", &AssembledFrameAllocator::get_frame,
            py::arg("consumer_id"),
            "Get the next frame for this consumer.\n\n"
            "Frames cycle through beam_ids for each time_chunk_index.")
        .def("num_free_frames", &AssembledFrameAllocator::num_free_frames,
            py::arg("permissive") = false,
            "Number of frames currently available in the pool.\n\n"
            "Throws in dummy mode or if not initialized.")
        .def("num_total_frames", &AssembledFrameAllocator::num_total_frames,
            py::arg("blocking") = false,
            "Total number of frames in the pool.\n\n"
            "Throws in dummy mode or if not initialized.")
    ;

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

    // Thread affinity functions
    m.def("set_thread_affinity", &set_thread_affinity,
          py::arg("vcpu_list"),
          "Set the calling thread's CPU affinity to the specified vCPUs.\n\n"
          "Args:\n"
          "    vcpu_list: List of vCPU indices. If empty, this is a no-op.\n\n"
          "Raises:\n"
          "    RuntimeError: If a vCPU index is out of range or the system call fails.");

    m.def("get_thread_affinity", &get_thread_affinity,
          "Get the calling thread's CPU affinity mask.\n\n"
          "Returns:\n"
          "    List of vCPU indices that the thread is allowed to run on.");

    // XEngineMetadata: documents file format for communication between X-engine and FRB nodes.
    // Skipped methods: to_yaml() [YAML::Emitter arg], from_yaml() [YamlFile arg]
    py::class_<XEngineMetadata>(m, "XEngineMetadata",
        "Metadata for X-engine to FRB node communication.\n\n"
        "Used in two contexts:\n"
        "  1. Sent by X-engine nodes to FRB nodes at start of TCP stream\n"
        "  2. Configuration for the fake correlator in testing")
          .def(py::init<>())
          .def_readwrite("version", &XEngineMetadata::version,
               "Version number of the metadata format")
          .def_readwrite("zone_nfreq", &XEngineMetadata::zone_nfreq,
               "Number of frequency channels in each zone")
          .def_readwrite("zone_freq_edges", &XEngineMetadata::zone_freq_edges,
               "Frequency band edges in MHz (length nzones+1, monotone increasing)")
          .def_readwrite("freq_channels", &XEngineMetadata::freq_channels,
               "Which frequency channels are present (optional)")
          .def_readwrite("nbeams", &XEngineMetadata::nbeams,
               "Number of beams")
          .def_readwrite("beam_ids", &XEngineMetadata::beam_ids,
               "Beam identifiers (defaults to [0, 1, ..., nbeams-1] if empty)")
          .def("validate", &XEngineMetadata::validate,
               "Validate that all members have sensible values")
          .def("get_total_nfreq", &XEngineMetadata::get_total_nfreq,
               "Returns sum of zone_nfreq (total frequency channels)")
          .def("to_yaml_string", &XEngineMetadata::to_yaml_string,
               py::arg("verbose") = false,
               "Serialize to YAML string.\n\n"
               "Args:\n"
               "    verbose: Include explanatory comments")
          .def_static("from_yaml_string", &XEngineMetadata::from_yaml_string,
               py::arg("s"),
               "Parse XEngineMetadata from a YAML string")
          .def_static("from_yaml_file", &XEngineMetadata::from_yaml_file,
               py::arg("filename"),
               "Parse XEngineMetadata from a YAML file")
    ;

    // FakeXEngine: simulates multiple upstream X-engine nodes sending data to a receiver.
    // Skipped members: mutex, cv, error, workers, barrier (internal state)
    // Skipped methods: _throw_if_stopped, make_worker_metadata, worker_main, _worker_main, _send_all (private)
    py::class_<FakeXEngine>(m, "FakeXEngine",
        "Simulates multiple upstream X-engine nodes sending data to a receiver.\n\n"
        "Creates multiple worker threads, each sending data over a TCP connection\n"
        "following the X->FRB network protocol.\n\n"
        "Threads are assigned round-robin to IP addresses. nthreads must be a\n"
        "multiple of len(ip_addrs).\n\n"
        "Usage:\n"
        "    xmd = XEngineMetadata.from_yaml_file('...')\n"
        "    fxe = FakeXEngine(xmd, ['10.0.0.2:5000', '10.0.1.2:5000'], 64)\n"
        "    fxe.start()  # creates threads and begins sending\n"
        "    # ... wait ...\n"
        "    fxe.stop()   # signals threads to stop")
          .def(py::init<const XEngineMetadata &, const std::vector<std::string> &, int>(),
               py::arg("xmd"), py::arg("ip_addrs"), py::arg("nthreads"),
               "Create a FakeXEngine (does not start sending until start() is called).\n\n"
               "Args:\n"
               "    xmd: X-engine metadata defining frequency channels and beams\n"
               "    ip_addrs: List of receiver addresses in 'ip:port' format\n"
               "    nthreads: Number of worker threads (must be a multiple of len(ip_addrs))")
          .def("start", &FakeXEngine::start,
               "Create worker threads and begin sending data.\n\n"
               "Raises:\n"
               "    RuntimeError: If already started or stopped")
          .def("stop", [](FakeXEngine &self) { self.stop(); },
               "Signal worker threads to stop. Safe to call multiple times.")
          .def_property_readonly("is_stopped",
               [](FakeXEngine &self) {
                   std::lock_guard<std::mutex> lock(self.mutex);
                   return self.is_stopped;
               },
               "True if stop() has been called (e.g. due to connection reset).")
          .def_readonly("xmd", &FakeXEngine::xmd,
               "X-engine metadata")
          .def_readonly("ip_addrs", &FakeXEngine::ip_addrs,
               "Receiver addresses in 'ip:port' format")
          .def_readonly("nthreads", &FakeXEngine::nthreads,
               "Number of worker threads")
          .def_property_readonly_static("protocol_magic",
               [](py::object) { return FakeXEngine::protocol_magic; },
               "Protocol magic number (0xf4bf4b01)")
          .def_property_readonly_static("send_timeout_ms",
               [](py::object) { return FakeXEngine::send_timeout_ms; },
               "Timeout for send operations in milliseconds")
    ;

    // Receiver: listens for TCP connections and reads data.
    // Skipped members: mutex, cv, is_started, is_stopped, error, listener_thread, reader_thread, pending_sockets (internal)
    // Skipped methods: _listener_main, _reader_main, listener_main, reader_main (private)
    // Note: Uses shared_ptr holder so Receivers can be passed to FrbServer.
    py::class_<Receiver, std::shared_ptr<Receiver>>(m, "Receiver",
        "Listens for TCP connections and reads data.\n\n"
        "A thread-backed class with two worker threads:\n"
        "  - listener: accepts incoming connections\n"
        "  - reader: reads data from all open connections using epoll")
          .def(py::init([](const std::string &address, long time_samples_per_chunk,
                           std::shared_ptr<AssembledFrameAllocator> allocator, long consumer_id) {
               Receiver::Params params;
               params.address = address;
               params.time_samples_per_chunk = time_samples_per_chunk;
               params.allocator = allocator;
               params.consumer_id = consumer_id;
               return std::make_shared<Receiver>(params);
          }),
               py::arg("address"), py::arg("time_samples_per_chunk"),
               py::arg("allocator"), py::arg("consumer_id"),
               "Create a Receiver (does not start worker threads).\n\n"
               "Args:\n"
               "    address: Address to bind to (e.g. '127.0.0.1:5000')\n"
               "    time_samples_per_chunk: Number of time samples per assembled chunk\n"
               "    allocator: AssembledFrameAllocator for output frames\n"
               "    consumer_id: Consumer ID for the allocator")
          .def("start", &Receiver::start,
               "Start the worker threads.\n\n"
               "Raises:\n"
               "    RuntimeError: If called twice or after stop().")
          .def("get_status", [](Receiver &self) {
               long num_conn, num_bytes;
               self.get_status(num_conn, num_bytes);
               return py::make_tuple(num_conn, num_bytes);
          }, "Returns (num_connections, num_bytes) tuple.")
          .def("stop", [](Receiver &self) { self.stop(); },
               "Signal worker threads to stop. Safe to call multiple times.")
          .def("get_metadata", &Receiver::get_metadata,
               py::arg("blocking"),
               "Get metadata from peers.\n\n"
               "Args:\n"
               "    blocking: If True, wait until metadata is available.\n\n"
               "Returns:\n"
               "    XEngineMetadata object (default-constructed if non-blocking and not available).")
          .def_property_readonly("address", [](const Receiver &self) { return self.params.address; },
               "Address bound to (e.g. '127.0.0.1:5000')")
    ;

    // FrbServer: gRPC server that queries Receivers and responds to RPCs.
    // Skipped members: params, rpc_service, rpc_server, mutex, is_started, is_stopped (internal)
    // Note: Params struct is not exposed; constructor takes receivers and address directly.
    // Note: FrbServer::create() is used internally, but appears as a constructor to Python.
    py::class_<FrbServer, std::shared_ptr<FrbServer>>(m, "FrbServer",
        "gRPC server that queries Receivers via RPC.\n\n"
        "Wraps multiple Receivers and exposes their status via gRPC.")
          .def(py::init([](std::vector<std::shared_ptr<Receiver>> receivers,
                           std::shared_ptr<FileWriter> file_writer,
                           const std::string &rpc_server_address) {
               FrbServer::Params params;
               params.receivers = std::move(receivers);
               params.file_writer = std::move(file_writer);
               params.rpc_server_address = rpc_server_address;
               return FrbServer::create(params);
          }),
               py::arg("receivers"), py::arg("file_writer"), py::arg("rpc_server_address"),
               "Create an FrbServer.\n\n"
               "Args:\n"
               "    receivers: List of Receiver objects to query\n"
               "    file_writer: FileWriter for saving frames to disk\n"
               "    rpc_server_address: gRPC server address (e.g. 'localhost:50051')")
          .def("start", &FrbServer::start,
               "Start all Receivers.\n\n"
               "Raises:\n"
               "    RuntimeError: If called twice or after stop().")
          .def("stop", [](FrbServer &self) { self.stop(); },
               "Stop the server and all Receivers. Safe to call multiple times.")
    ;

    // FileWriter: writes AssembledFrames to disk via SSD and NFS queues.
    // Skipped members: Params, WriteStatus, RpcSubscriber, process_frame, add_subscriber (internal)
    py::class_<FileWriter, std::shared_ptr<FileWriter>>(m, "FileWriter",
        "Writes AssembledFrames to disk via SSD and NFS queues.\n\n"
        "Creates worker threads for writing to local SSD and copying to NFS.")
          .def(py::init([](const std::string &ssd_root, const std::string &nfs_root,
                           int num_ssd_threads, int num_nfs_threads) {
               FileWriter::Params params;
               params.ssd_root = ssd_root;
               params.nfs_root = nfs_root;
               params.num_ssd_threads = num_ssd_threads;
               params.num_nfs_threads = num_nfs_threads;
               return std::make_shared<FileWriter>(params);
          }),
               py::arg("ssd_root"), py::arg("nfs_root"),
               py::arg("num_ssd_threads") = 4, py::arg("num_nfs_threads") = 2,
               "Create a FileWriter.\n\n"
               "Args:\n"
               "    ssd_root: Absolute path to local SSD directory\n"
               "    nfs_root: Absolute path to NFS directory\n"
               "    num_ssd_threads: Number of threads for SSD writes (default 4)\n"
               "    num_nfs_threads: Number of threads for NFS copies (default 2)")
          .def("stop", [](FileWriter &self) { self.stop(); },
               "Stop the writer. Safe to call multiple times.")
    ;
    
    // HwtestSender: simulates a correlator sending data over TCP.
    // Skipped members: mutex, cv, is_stopped, is_started, error, workers, endpoints (internal state)
    // Skipped methods: _throw_if_stopped, worker_main, _worker_main, _send_all (private)
    py::class_<HwtestSender>(m, "HwtestSender")
        .def(py::init<long, bool, bool, bool>(),
             py::arg("send_bufsize"), py::arg("use_zerocopy"), py::arg("use_mmap"), py::arg("use_hugepages"))

        .def("add_endpoint", &HwtestSender::add_endpoint,
             py::arg("ip_addr"), py::arg("num_tcp_connections"), py::arg("total_gbps"), py::arg("vcpu_list"),
             "Add an endpoint. Must be called before start().")

        .def("start", &HwtestSender::start,
             py::call_guard<py::gil_scoped_release>(),
             "Create worker threads and begin sending data.")

        .def("stop", [](HwtestSender &self) { self.stop(); },
             py::call_guard<py::gil_scoped_release>(),
             "Signal worker threads to stop. Safe to call multiple times.")

        .def("join", &HwtestSender::join,
             py::call_guard<py::gil_scoped_release>(),
             "Block until all worker threads have exited.")

        .def("wait", &HwtestSender::wait,
             py::arg("timeout_ms"),
             py::call_guard<py::gil_scoped_release>(),
             "Wait until all workers have exited, or timeout expires.\n"
             "Returns True if all workers exited, False on timeout.")
    ;

    py::class_<Hwtest, std::shared_ptr<Hwtest>>(m, "Hwtest")
        .def(py::init([](const std::string &server_name, bool use_hugepages) {
                 return Hwtest::create(server_name, use_hugepages);
             }),
             py::arg("server_name"), py::arg("use_hugepages"))

        .def("add_tcp_receiver", &Hwtest::add_tcp_receiver,
             py::arg("ip_addr"), py::arg("num_tcp_connections"), py::arg("recv_bufsize"),
             py::arg("vcpu_list"), py::arg("cpu"), py::arg("inic"))

        .def("add_chime_dedisperser", &Hwtest::add_chime_dedisperser,
             py::arg("device"), py::arg("vcpu_list"), py::arg("cpu"))

        .def("add_memcpy_thread", &Hwtest::add_memcpy_thread,
             py::arg("src_device"), py::arg("dst_device"), py::arg("blocksize"),
             py::arg("use_copy_engine"), py::arg("vcpu_list"), py::arg("cpu"))

        .def("add_ssd_writer", &Hwtest::add_ssd_writer,
             py::arg("root_dir"), py::arg("nbytes_per_file"), py::arg("write_asdf"),
             py::arg("vcpu_list"), py::arg("cpu"), py::arg("issd"))

        .def("add_downsampling_thread", &Hwtest::add_downsampling_thread,
             py::arg("src_bit_depth"), py::arg("src_nelts"), py::arg("vcpu_list"),
             py::arg("cpu"))

         // Called by python code, to control server.
        .def("join", &Hwtest::join,
             py::call_guard<py::gil_scoped_release>())
        .def("show_stats", &Hwtest::show_stats,
             py::call_guard<py::gil_scoped_release>())
        .def("start", &Hwtest::start,
             py::call_guard<py::gil_scoped_release>())
        .def("stop", [](Hwtest &self) { self.stop(); },
             py::call_guard<py::gil_scoped_release>())
    ;
}

}  // namespace pirate

