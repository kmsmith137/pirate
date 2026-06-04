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
#include "../include/pirate/DedispersionPlan.hpp"     // FrbServer.plan property
#include "../include/pirate/Dedisperser.hpp"           // FrbServer.dedisperser property
#include "../include/pirate/DedispersionTree.hpp"
#include "../include/pirate/FrequencySubbands.hpp"
#include "../include/pirate/ResourceTracker.hpp"
#include "../include/pirate/system_utils.hpp"  // set_thread_affinity, get_thread_affinity
#include "../include/pirate/XEngineMetadata.hpp"
#include "../include/pirate/FakeXEngine.hpp"
#include "../include/pirate/Receiver.hpp"
#include "../include/pirate/FrbServer.hpp"
#include "../include/pirate/FrbGrouper.hpp"
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
        .def(py::init<int, long, bool, int, int>(),
            py::arg("aflags"), py::arg("capacity"),
            py::arg("is_async") = false,
            py::arg("nthreads") = 0,
            py::arg("cuda_device") = -1,
            "Create allocator.\n\n"
            "Args:\n"
            "    aflags: Memory allocation flags (af_gpu, af_rhost, etc.)\n"
            "    capacity: Bytes to pre-allocate (>= 0) or -1 for dummy mode\n"
            "    is_async: If True, constructor returns immediately and the\n"
            "        allocation/zeroing happens on worker threads. Public\n"
            "        methods block until init complete. nthreads and\n"
            "        cuda_device are then required.\n"
            "    nthreads: Number of worker threads (>= 2 for case 1 and 2;\n"
            "        ignored for case 3 / af_gpu).\n"
            "    cuda_device: CUDA device id (>= 0 required in async mode).")
        .def_property_readonly("nbytes_allocated",
            [](const BumpAllocator &self) { return self.nbytes_allocated.load(); },
            "Bytes allocated so far (aligned to 128-byte cache lines)")
        .def_readonly("aflags", &BumpAllocator::aflags,
            "Memory allocation flags")
        .def_readonly("capacity", &BumpAllocator::capacity,
            "Total capacity in bytes, or -1 for dummy mode")
        .def("wait_until_initialized", &BumpAllocator::wait_until_initialized,
            "In async mode: block until init complete, or rethrow async-init "
            "exception. In sync mode: no-op.")
        .def("is_initialized", &BumpAllocator::is_initialized,
            "Non-blocking poll: True iff init has completed and the allocator "
            "has not been stopped.")
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
        .def(py::init(static_cast<std::shared_ptr<SlabAllocator>(*)(const std::shared_ptr<BumpAllocator> &, long)>(&SlabAllocator::create)),
            py::arg("bump_allocator"), py::arg("nbytes"),
            "Create allocator using memory from a BumpAllocator.\n\n"
            "Args:\n"
            "    bump_allocator: Source of memory (must not be in dummy mode)\n"
            "    nbytes: Bytes to allocate from the BumpAllocator\n\n"
            "If the BumpAllocator is async, the SlabAllocator is itself async:\n"
            "constructor returns immediately, and the bump_allocator.allocate_bytes()\n"
            "call is deferred to the first get_slab().")
        .def("wait_until_initialized", &SlabAllocator::wait_until_initialized,
            "If backed by an async BumpAllocator: block until it's initialized\n"
            "(or rethrow async-init exception). Otherwise no-op.")
        .def("is_initialized", &SlabAllocator::is_initialized,
            "Non-blocking poll: True iff the underlying BumpAllocator is ready\n"
            "to serve allocations (delegates to bump_allocator.is_initialized()).\n"
            "Always True in dummy mode.")
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
        .def_property_readonly("metadata",
            [](const AssembledFrame &self) {
                return std::const_pointer_cast<XEngineMetadata>(self.metadata);
            },
            "Shared XEngineMetadata for this frame. Read-only by convention --\n"
            "do not mutate the returned object, since it is shared with all\n"
            "sibling frames from the same allocator.\n\n"
            "Always frequency-scrubbed: metadata.freq_channels is empty.")
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
        .def_property_readonly("scales_offsets",
            [](const AssembledFrame &self) {
                // The Array<void> already carries dtype Dtype(df_float, 16);
                // ksgpu's pybind11 type_caster maps that to numpy np.float16.
                return self.scales_offsets;
            },
            "Scales/offsets as float16 array with shape (nfreq, mpc, 2),\n"
            "where mpc = ntime / 256. The last axis is {scale, offset}.")
        .def("write_asdf", &AssembledFrame::write_asdf,
            py::arg("filename"), py::arg("sync") = true, py::arg("verbose") = false,
            "Write this AssembledFrame to an ASDF file.\n\n"
            "If sync=True (default), fsync() the file after writing to avoid\n"
            "runaway page cache usage.\n\n"
            "If verbose=True, emit comments throughout the YAML header. The\n"
            "comments are detailed enough that the header serves as self-\n"
            "contained documentation of the file format. Intended for use\n"
            "with 'pirate_frb show_file_format'.")
        .def_static("make_random", &AssembledFrame::make_random,
            py::arg("xmd"), py::arg("ntime"), py::arg("beam_id"), py::arg("time_chunk_index"),
            "Create a random AssembledFrame backed by the given XEngineMetadata.\n\n"
            "Throws if xmd is None or if beam_id is not in xmd.beam_ids.\n"
            "Data is filled with random bytes. nfreq is taken from xmd.get_total_nfreq().\n\n"
            "xmd.freq_channels is ignored. Callers should typically pass a\n"
            "frequency-scrubbed xmd so the returned frame matches the\n"
            "'always frequency-scrubbed' invariant on AssembledFrame.metadata.")
        .def_static("from_asdf", &AssembledFrame::from_asdf,
            py::arg("filename"),
            "Read an AssembledFrame back from an ASDF file.\n\n"
            "Note: XEngineMetadata is projected through ASDF -- after reading, the\n"
            "metadata's beam_ids / beam_positions_{x,y} are length-1 (just this\n"
            "frame's beam) and freq_channels is empty. See XEngineMetadata.hpp.")
        .def("randomize", &AssembledFrame::randomize,
            py::call_guard<py::gil_scoped_release>(),
            "Fill the frame's data buffer with uniformly random bytes.\n"
            "Each int4 sample is uniform over [-8, +7]. Intended for testing.\n\n"
            "Thread-safe with respect to the RNG (uses ksgpu's per-thread\n"
            "default RNG), but the caller must ensure that no other thread is\n"
            "concurrently reading or writing the same frame's data buffer --\n"
            "only the RNG is protected, not the destination buffer.")
    ;

    // AssembledFrameSet: container of (nbeams) AssembledFrames for one time chunk.
    py::class_<AssembledFrameSet, std::shared_ptr<AssembledFrameSet>>(m, "AssembledFrameSet",
        "Container of (nbeams) AssembledFrames for one time chunk.")
        .def_readonly("nfreq", &AssembledFrameSet::nfreq)
        .def_readonly("ntime", &AssembledFrameSet::ntime)
        .def_readonly("nbeams", &AssembledFrameSet::nbeams)
        .def_readonly("time_chunk_index", &AssembledFrameSet::time_chunk_index)
        .def_property_readonly("metadata",
            [](const AssembledFrameSet &self) {
                return std::const_pointer_cast<XEngineMetadata>(self.metadata);
            },
            "Shared XEngineMetadata for this set. Read-only by convention.\n\n"
            "Always frequency-scrubbed: metadata.freq_channels is empty.")
        .def_readonly("frames", &AssembledFrameSet::frames,
            "Length-nbeams list of AssembledFrame, in metadata.beam_ids order.")
        .def("get_frame", &AssembledFrameSet::get_frame,
            py::arg("ibeam"),
            "Bounds-checked accessor for frames[ibeam].")
        .def("validate", &AssembledFrameSet::validate,
            "Defensive consistency check; throws on inconsistency. Cheap.")
        // NOTE: there is no AssembledFrameSet.randomize(); to randomize a
        // whole set, use FakeXEngine.randomize_frames(fset) (parallelized
        // over the randomizer-thread pool).
    ;

    // AssembledFrameAllocator: allocates AssembledFrameSets for multiple consumers.
    // Designed for multi-threaded use, but works fine with a single Python consumer.
    py::class_<AssembledFrameAllocator, std::shared_ptr<AssembledFrameAllocator>>(m, "AssembledFrameAllocator",
        "Allocates AssembledFrameSets for multiple consumers.\n\n"
        "Each consumer calls initialize_metadata() / initialize_initial_chunk()\n"
        "once (or waits for someone else to), then get_frame_set() in a loop.\n"
        "All consumers receive the same sequence of sets (same shared_ptr).")
        .def(py::init<const std::shared_ptr<SlabAllocator> &, int, long>(),
            py::arg("slab_allocator"),
            py::arg("num_consumers"),
            py::arg("time_samples_per_chunk"))
        .def_readonly("nfreq", &AssembledFrameAllocator::nfreq)
        .def_readonly("time_samples_per_chunk", &AssembledFrameAllocator::time_samples_per_chunk)
        .def_readonly("beam_ids", &AssembledFrameAllocator::beam_ids)
        .def_property_readonly("metadata",
            [](AssembledFrameAllocator &self) {
                // Route through get_metadata(blocking=false) so the read is
                // properly lock-synchronized; pybind11 maps a null shared_ptr
                // to Python None automatically.
                auto m = self.get_metadata(/*blocking=*/false);
                return std::const_pointer_cast<XEngineMetadata>(m);
            },
            "Shared XEngineMetadata, set the first time initialize() is called.\n"
            "None if no consumer has called initialize() yet. Read-only by\n"
            "convention. (Note: freq_channels is cleared on the canonical copy.)")
        .def("initialize_metadata", &AssembledFrameAllocator::initialize_metadata,
            py::arg("metadata"),
            "Set the canonical XEngineMetadata on the allocator. The first call\n"
            "stores the metadata (with freq_channels cleared); subsequent calls\n"
            "validate consistency via XEngineMetadata.check_sender_consistency.\n"
            "Typically called by each Receiver's reader thread as it parses a\n"
            "peer's YAML, so many calls per allocator are expected.")
        .def("initialize_initial_chunk", &AssembledFrameAllocator::initialize_initial_chunk,
            py::arg("target_time_chunk"),
            "Establish (on the first call from any caller) the canonical\n"
            "initial_time_chunk for the whole pipeline. The first set\n"
            "returned by get_frame_set() has time_chunk_index = initial_time_chunk.\n"
            "Returns the established value (target_time_chunk on the first call,\n"
            "previously-established value on subsequent calls).")
        .def("wait_for_initial_chunk", &AssembledFrameAllocator::wait_for_initial_chunk,
            "Block until some caller has invoked initialize_initial_chunk(),\n"
            "then return the established initial_time_chunk.")
        .def("get_frame_set", &AssembledFrameAllocator::get_frame_set,
            py::arg("consumer_id"),
            py::call_guard<py::gil_scoped_release>(),
            "Get the next AssembledFrameSet (one time chunk, all beams) for\n"
            "this consumer. The N-th call returns time_chunk_index =\n"
            "initial_time_chunk + N.\n\n"
            "Releases the GIL: in dummy mode the calling thread does the\n"
            "per-set allocation + memset synchronously, and in non-dummy mode\n"
            "it may block on the worker thread -- neither should stall other\n"
            "Python threads (e.g. a sender thread running concurrently with a\n"
            "frame-provider thread).")
        .def("get_metadata",
            [](AssembledFrameAllocator &self, bool blocking) {
                auto m = self.get_metadata(blocking);
                return std::const_pointer_cast<XEngineMetadata>(m);
            },
            py::arg("blocking"),
            "Get the canonical XEngineMetadata pointer.\n\n"
            "Args:\n"
            "    blocking: If True, wait until any consumer has called initialize().\n\n"
            "Returns:\n"
            "    XEngineMetadata object (or None if non-blocking and not yet set).\n"
            "    Note: freq_channels is cleared on the canonical copy.")
        .def("num_free_frames", &AssembledFrameAllocator::num_free_frames,
            py::arg("permissive") = false,
            "Number of frames currently available in the pool.\n\n"
            "Throws in dummy mode or if not initialized.")
        .def("num_total_frames", &AssembledFrameAllocator::num_total_frames,
            py::arg("blocking") = false,
            "Total number of frames in the pool.\n\n"
            "Throws in dummy mode or if not initialized.")
        .def("is_initialized", &AssembledFrameAllocator::is_initialized,
            "Non-blocking poll: True iff the underlying memory is ready to\n"
            "serve allocations (delegates through SlabAllocator to BumpAllocator).")
        .def("stop", [](AssembledFrameAllocator &self) { self.stop(); },
            py::call_guard<py::gil_scoped_release>(),
            "Stop the allocator. After stop(), entry points throw and any\n"
            "thread blocked in get_frame_set() (waiting for a free slab) wakes\n"
            "and raises. Idempotent and safe to call from any thread.")
        .def_static("slab_nbytes", &AssembledFrameAllocator::slab_nbytes,
            py::arg("nfreq"), py::arg("time_samples_per_chunk"),
            "Total backing bytes for one AssembledFrame's slab (one beam), with\n"
            "scales_offsets and data each cache-line aligned. Matches\n"
            "_create_frame_set exactly -- use it (times nbeams) to size a slab\n"
            "pool for AssembledFrameSets. Static: callable without an instance.\n"
            "Throws unless time_samples_per_chunk is a positive multiple of 256.")
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
    py::class_<XEngineMetadata, std::shared_ptr<XEngineMetadata>>(m, "XEngineMetadata",
        "Metadata sent over the wire by X-engine nodes to FRB nodes at the\n"
        "start of every TCP stream. Also used for bookkeeping in several\n"
        "places (allocator's canonical copy, per-frame ASDF projection, etc).\n\n"
        "See configs/xengine/xengine_metadata_v2.yml for field-by-field\n"
        "documentation. The freq_channels member has two distinct meanings\n"
        "depending on context -- see the 'freq_channels' attribute docstring\n"
        "and XEngineMetadata.hpp for the 'frequency-scrubbed' convention.\n\n"
        "Note: YAML is the full-fidelity serialization. ASDF (via\n"
        "AssembledFrame.write_asdf) drops/projects 4 members per-frame --\n"
        "see the C++ header for details.")
          .def(py::init<>())
          .def_readwrite("version", &XEngineMetadata::version,
               "Version number of the metadata format")
          .def_readwrite("zone_nfreq", &XEngineMetadata::zone_nfreq,
               "Number of frequency channels in each zone")
          .def_readwrite("zone_freq_edges", &XEngineMetadata::zone_freq_edges,
               "Frequency band edges in MHz (length nzones+1, monotone increasing)")
          .def_readwrite("freq_channels", &XEngineMetadata::freq_channels,
               "Which frequency channels are present. MEANINGFUL when the\n"
               "metadata is associated with one specific X-engine sender\n"
               "(the wire-protocol case, or a FakeXEngine Worker). Otherwise\n"
               "(bookkeeping contexts where no specific sender is distinguished,\n"
               "e.g. the allocator's canonical copy, FrbServer, AssembledFrame's\n"
               "metadata) the convention is to set this to an empty list, which\n"
               "we call 'frequency-scrubbed'. See XEngineMetadata.hpp.")
          .def_readwrite("beamset", &XEngineMetadata::beamset,
               "Integer identifier for this set of beams")
          .def_readwrite("beam_ids", &XEngineMetadata::beam_ids,
               "Beam identifiers (length nbeams)")
          .def_readwrite("beam_positions_x", &XEngineMetadata::beam_positions_x,
               "Direction cosine b.x in grid frame, length nbeams")
          .def_readwrite("beam_positions_y", &XEngineMetadata::beam_positions_y,
               "Direction cosine b.y in grid frame, length nbeams")
          .def_readwrite("unix_ns_at_seq_0", &XEngineMetadata::unix_ns_at_seq_0,
               "UNIX nanoseconds at FPGA seq=0")
          .def_readwrite("dt_ns_per_seq", &XEngineMetadata::dt_ns_per_seq,
               "Nanoseconds per FPGA seq tick")
          .def_readwrite("seq_per_frb_time_sample", &XEngineMetadata::seq_per_frb_time_sample,
               "FPGA seq ticks per FRB time sample")
          .def_readwrite("tel_origin_itrs_lat_deg", &XEngineMetadata::tel_origin_itrs_lat_deg,
               "Telescope ITRS latitude in degrees")
          .def_readwrite("tel_origin_itrs_lon_deg", &XEngineMetadata::tel_origin_itrs_lon_deg,
               "Telescope ITRS longitude in degrees")
          .def_readwrite("tel_grid_x_axis", &XEngineMetadata::tel_grid_x_axis,
               "Grid x-axis unit vector (topocentric, length 3)")
          .def_readwrite("tel_grid_y_axis", &XEngineMetadata::tel_grid_y_axis,
               "Grid y-axis unit vector (topocentric, length 3)")
          .def_readwrite("tel_dish_elev_axis", &XEngineMetadata::tel_dish_elev_axis,
               "Dish elevation-axis unit vector (topocentric, length 3)")
          .def_readwrite("tel_dish_vert_axis", &XEngineMetadata::tel_dish_vert_axis,
               "Dish vertical-axis unit vector (topocentric, length 3)")
          .def_readwrite("tel_dish_coelev_deg", &XEngineMetadata::tel_dish_coelev_deg,
               "Dish coelevation in degrees (angle from vertical, north positive)")
          .def_readwrite("tel_dish_separation_x_m", &XEngineMetadata::tel_dish_separation_x_m,
               "Dish separation along grid x-axis in meters")
          .def_readwrite("tel_dish_separation_y_m", &XEngineMetadata::tel_dish_separation_y_m,
               "Dish separation along grid y-axis in meters")
          .def_readwrite("noise_variance", &XEngineMetadata::noise_variance,
               "Per-zone noise variance, length nzones (=len(zone_nfreq))")
          .def("validate", &XEngineMetadata::validate,
               "Validate that all members have sensible values")
          .def("get_total_nfreq", &XEngineMetadata::get_total_nfreq,
               "Returns sum of zone_nfreq (total frequency channels)")
          .def("get_nbeams", &XEngineMetadata::get_nbeams,
               "Returns the number of beams (= len(beam_ids))")
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
          .def_static("make_fiducial", &XEngineMetadata::make_fiducial,
               py::arg("zone_nfreq"), py::arg("zone_freq_edges"), py::arg("beam_ids"),
               py::arg("time_sample_ms"),
               "Return a fully-valid XEngineMetadata with placeholder telescope and\n"
               "timekeeping values. noise_variance defaults to {1.0, ...} of length\n"
               "nzones, beamset defaults to 0, and beam_positions_{x,y} are arranged\n"
               "on a deterministic 2D grid spanning [-0.1, +0.1]. The timekeeping\n"
               "fields are chosen so the resulting time sample length matches\n"
               "time_sample_ms (rounded to the closest integer seq tick count).\n"
               "Throws if time_sample_ms < 0.5. Caller may further patch fields\n"
               "before calling validate().")
          .def_static("make_random", &XEngineMetadata::make_random,
               "Return a fully-valid XEngineMetadata with all fields randomized\n"
               "within validity bounds (small scale: 1-4 zones, 1-8 beams, etc.).\n"
               "Used for fuzz-style coverage of code paths that consume metadata.")
    ;

    // FakeXEngine: simulates multiple upstream X-engine nodes sending data to a receiver.
    // Driven externally by a controller thread that submits SEND_JUNK commands
    // via enqueue_send_junk() and synchronizes via wait_until_processed().
    // Skipped members: mutex, cv, error, workers (internal state)
    // Skipped methods: _throw_if_stopped, make_worker_metadata, worker_main, _worker_main, _send_all (private)
    py::class_<FakeXEngine>(m, "FakeXEngine",
        "Simulates multiple upstream X-engine nodes sending data to a receiver.\n\n"
        "Creates 'nworkers' worker threads in the constructor; each worker\n"
        "waits on a per-worker command queue. An external controller thread\n"
        "drives the workers by calling enqueue_send_junk(worker_id, minichunk_index)\n"
        "and wait_until_processed(worker_id, minichunk_index).\n\n"
        "Workers are assigned round-robin to IP addresses. nworkers must be a\n"
        "multiple of len(ip_addrs). Worker threads inherit the vcpu affinity\n"
        "of the thread that calls the constructor -- Python callers MUST\n"
        "instantiate FakeXEngine inside a ThreadAffinity context manager.\n\n"
        "Usage:\n"
        "    xmd = XEngineMetadata.from_yaml_file('...')\n"
        "    with ThreadAffinity(vcpu_list):\n"
        "        fxe = FakeXEngine(xmd, ['10.0.0.2:5000', '10.0.1.2:5000'], 64,\n"
        "                          time_samples_per_chunk=32768)\n"
        "        # Spawn a controller thread (under the same affinity) that\n"
        "        # calls fxe.enqueue_send_junk / fxe.wait_until_processed in a loop.\n"
        "    # ... wait ...\n"
        "    fxe.stop()   # signals workers and any in-flight entry points to exit")
          .def(py::init<const std::shared_ptr<const XEngineMetadata> &,
                        const std::vector<std::string> &,
                        int, long, bool, bool, const std::string &>(),
               py::arg("xmd"), py::arg("ip_addrs"), py::arg("nworkers"),
               py::arg("time_samples_per_chunk"),
               py::arg("debug") = false,
               py::arg("paced") = true,
               py::arg("rpc_address") = "",
               "Create a FakeXEngine and spawn 'nworkers' worker threads.\n\n"
               "Workers inherit the vcpu affinity of the calling thread, so the\n"
               "Python caller MUST invoke this constructor inside a ThreadAffinity\n"
               "context manager.\n\n"
               "Args:\n"
               "    xmd: X-engine metadata defining frequency zones and beams.\n"
               "        Passed as a shared_ptr -- the FakeXEngine retains a\n"
               "        non-mutating reference. xmd.freq_channels is ignored --\n"
               "        per-worker freq_channels is built internally by\n"
               "        round-robin assignment of channels to workers.\n"
               "    ip_addrs: List of receiver addresses in 'ip:port' format\n"
               "    nworkers: Number of worker threads (must be a multiple of len(ip_addrs))\n"
               "    time_samples_per_chunk: Receiver-side chunk size (must equal\n"
               "        the FrbServer's AssembledFrameAllocator.time_samples_per_chunk).\n"
               "        Must be positive and a multiple of 256.\n"
               "    debug: Adds real-time debugging checks with nontrivial\n"
               "        cpu/network cost, and unbounded memory usage. Useful for\n"
               "        unit tests, but don't use in production!\n"
               "    paced (default True): Spawn a pacing thread that subscribes\n"
               "        to the FrbServer's MonitorRingbuf push stream and gates\n"
               "        each worker's sends to stay <=5 chunks ahead of\n"
               "        server-side rb_processed. Requires rpc_address.\n"
               "    rpc_address: 'ip:port' of the FrbServer's gRPC endpoint.\n"
               "        Required (non-empty) when paced=True; ignored (silently\n"
               "        accepted) when paced=False.")
          .def("enqueue_send_junk", &FakeXEngine::enqueue_send_junk,
               py::arg("worker_id"), py::arg("minichunk_index"),
               py::call_guard<py::gil_scoped_release>(),
               "Submit a SEND_JUNK(minichunk_index) command to worker_id's queue.\n\n"
               "Non-blocking. minichunk_index must be exactly one greater than\n"
               "the last state-advancing command (SEND_JUNK / SKIP_MINICHUNK /\n"
               "SEND_MINICHUNK) submitted to this worker -- with one exception:\n"
               "the very first state-advancing command on a worker may pick any\n"
               "minichunk_index >= 0 (for NOTE-2 nonzero-initial-chunk tests).\n"
               "DISCONNECT commands do not participate in the chain.\n\n"
               "The sequentiality check fires synchronously at submit time;\n"
               "a bad index raises RuntimeError to this caller rather than\n"
               "tearing down the FakeXEngine.\n\n"
               "Wire effect: the worker sends one minichunk worth of all-zero\n"
               "data. The protocol handshake is sent ahead of the first\n"
               "SEND_JUNK or SEND_MINICHUNK on this worker.\n\n"
               "Raises:\n"
               "    RuntimeError: If the FakeXEngine is stopped, arguments are\n"
               "    out of range, or minichunk_index is not the +1 successor of\n"
               "    the most recently enqueued state-advancing command on this\n"
               "    worker (with the first-command exemption).")
          .def("enqueue_skip_minichunk", &FakeXEngine::enqueue_skip_minichunk,
               py::arg("worker_id"), py::arg("minichunk_index"),
               py::call_guard<py::gil_scoped_release>(),
               "Submit a SKIP_MINICHUNK(minichunk_index) command to worker_id's queue.\n\n"
               "Non-blocking. Same queue-time sequentiality rules as\n"
               "enqueue_send_junk (strict +1 from the most recently enqueued\n"
               "state-advancing command on this worker, with the first-command\n"
               "exemption).\n\n"
               "Wire effect: NONE. Advances last_processed_minichunk past\n"
               "minichunk_index without putting any bytes on the wire.\n"
               "A worker whose only commands are SKIPs never opens its TCP\n"
               "connection -- useful for 'silent peer' tests.\n\n"
               "Raises:\n"
               "    RuntimeError: If the FakeXEngine is stopped, arguments are\n"
               "    out of range, or minichunk_index violates the +1 chain.")
          .def("enqueue_send_minichunk", &FakeXEngine::enqueue_send_minichunk,
               py::arg("worker_id"), py::arg("minichunk_index"), py::arg("frame_set"),
               py::call_guard<py::gil_scoped_release>(),
               "Submit a SEND_MINICHUNK(minichunk_index, frame_set) command.\n\n"
               "Non-blocking. Same queue-time sequentiality rules as\n"
               "enqueue_send_junk (strict +1 from the most recently enqueued\n"
               "state-advancing command on this worker, with the first-command\n"
               "exemption).\n\n"
               "Wire effect: gather the per-(beam, freq) int4 data for the\n"
               "minichunk at offset (minichunk_index - frame_set.time_chunk_index\n"
               "* minichunks_per_chunk) within the set, then send one minichunk.\n\n"
               "The caller is responsible for keeping the AssembledFrameSet (and\n"
               "its frames' data buffers) alive throughout the SEND_MINICHUNK\n"
               "processing. The worker holds a shared_ptr while the Command sits\n"
               "in the queue + is being processed, so the typical pattern (hold\n"
               "the same Python reference until wait_until_processed returns for\n"
               "this minichunk_index) is sufficient.\n\n"
               "REAPER RACE WARNING: the worker reads the frames' data buffers\n"
               "WITHOUT taking frame.mutex, so it races the AssembledFrame reaper.\n"
               "The reaper currently lives in FrbServer, not FakeXEngine, and the\n"
               "design assumption is that FakeXEngine and FrbServer are never\n"
               "colocated in the same process -- so the race does not occur in\n"
               "practice. If we ever want to colocate the two, the gather loop\n"
               "needs lock acquisition (one per beam).\n\n"
               "Raises:\n"
               "    RuntimeError: If the FakeXEngine is stopped, arguments are\n"
               "    out of range, frame_set is None, or minichunk_index violates\n"
               "    the +1 chain.")
          .def("enqueue_disconnect", &FakeXEngine::enqueue_disconnect,
               py::arg("worker_id"),
               py::call_guard<py::gil_scoped_release>(),
               "Submit a DISCONNECT command to worker_id's queue.\n\n"
               "Non-blocking (fire-and-forget). The worker closes its TCP socket\n"
               "on receipt. last_processed_minichunk and last_queued_minichunk\n"
               "are NOT touched; the next enqueue_send_junk or\n"
               "enqueue_send_minichunk on this worker transparently reopens the\n"
               "connection and re-sends the protocol handshake.\n\n"
               "SKIP_MINICHUNK commands continue to work normally while\n"
               "disconnected (they advance last_processed_minichunk without\n"
               "touching the socket). If you want the next reconnect to start at\n"
               "a higher minichunk_index, bridge the gap yourself with\n"
               "enqueue_skip_minichunk calls.\n\n"
               "DISCONNECT does NOT participate in the +1 sequentiality chain,\n"
               "so it can be submitted at any time relative to the state-\n"
               "advancing commands.\n\n"
               "DISCONNECT on an already-disconnected (or never-connected)\n"
               "worker is a no-op.\n\n"
               "Raises:\n"
               "    RuntimeError: If the FakeXEngine is stopped or worker_id is\n"
               "    out of range.")
          .def("is_connected", &FakeXEngine::is_connected,
               py::arg("worker_id"),
               "Return True iff this worker has an open TCP connection to its\n"
               "receiver right now.\n\n"
               "The result is a snapshot -- by the time the caller observes it,\n"
               "the worker thread may have already flipped the flag. Does NOT\n"
               "throw on a stopped FakeXEngine (the last-known per-worker state\n"
               "is still meaningful for diagnostics). Throws only on worker_id\n"
               "out of range.\n\n"
               "O(1) atomic load; the GIL is NOT released for this call.")
          .def("wait_until_processed", &FakeXEngine::wait_until_processed,
               py::arg("worker_id"), py::arg("minichunk_index"),
               py::call_guard<py::gil_scoped_release>(),
               "Block until worker_id has finished processing minichunk_index\n"
               "(or a later one). 'Processed' means a state-advancing command\n"
               "(SEND_JUNK / SKIP_MINICHUNK / SEND_MINICHUNK) with the given\n"
               "minichunk_index has been fully handled by the worker -- bytes\n"
               "actually on the wire for SEND_*, or just state-advance for SKIP.\n\n"
               "Returns immediately if minichunk_index is negative (since per-\n"
               "worker last_processed_minichunk starts at -1, this lets the\n"
               "controller call wait_until_processed(w, n-2) unconditionally for\n"
               "n in {0, 1}).\n\n"
               "Raises:\n"
               "    RuntimeError: If the FakeXEngine is stopped.")
          .def("enqueue_wait_for_acks", &FakeXEngine::enqueue_wait_for_acks,
               py::arg("worker_id"),
               py::call_guard<py::gil_scoped_release>(),
               "Enqueue a WAIT_FOR_ACKS command on worker_id's queue. Non-\n"
               "blocking (fire-and-forget) at the API level; when the worker\n"
               "eventually pops the command, it drains all outstanding\n"
               "debug-mode acks (blocking, with a 1-second per-call deadline).\n\n"
               "Useful when you want to issue a barrier without waiting for it\n"
               "to complete. To wait for all acks too, call synchronize() or\n"
               "poll get_minichunk_status().\n\n"
               "Raises:\n"
               "    RuntimeError: If the FakeXEngine was not constructed with\n"
               "    debug=True (no acks to wait for), if worker_id is out\n"
               "    of range, or if the FakeXEngine is stopped.")
          .def("synchronize", &FakeXEngine::synchronize,
               py::arg("worker_id"),
               py::call_guard<py::gil_scoped_release>(),
               "Block the calling thread until worker_id's command_queue is\n"
               "fully drained. If the FakeXEngine was constructed with\n"
               "debug=True, this additionally enqueues a WAIT_FOR_ACKS\n"
               "before waiting -- so the wait also covers all outstanding acks.\n\n"
               "SEMANTICS: this is a 'drain everything in the queue' barrier,\n"
               "NOT a 'drain everything that was in the queue at the moment I\n"
               "was called' barrier. If another thread is concurrently\n"
               "enqueueing commands on the same worker, synchronize() waits\n"
               "for those commands too (until command_queue stays empty long\n"
               "enough for synchronize() to observe it under the lock).\n\n"
               "Raises:\n"
               "    RuntimeError: If the FakeXEngine is stopped or worker_id\n"
               "    is out of range.")
          .def("randomize_frames", &FakeXEngine::randomize_frames,
               py::arg("fset"),
               py::call_guard<py::gil_scoped_release>(),
               "Fill every AssembledFrame in 'fset' with random data,\n"
               "distributing the per-beam AssembledFrame.randomize() calls\n"
               "across the randomizer-thread pool. Blocks until done.\n\n"
               "'fset' must have exactly nbeams frames. The caller keeps\n"
               "ownership and must keep 'fset' alive until this returns; on\n"
               "return no randomizer thread is still touching it. NOT safe to\n"
               "call concurrently with itself (single in-flight job).\n\n"
               "Raises:\n"
               "    RuntimeError: If the FakeXEngine is stopped, or fset's\n"
               "    frame count does not match nbeams.")
          .def("get_minichunk_status", &FakeXEngine::get_minichunk_status,
               py::arg("worker_id"), py::arg("minichunk_index"),
               "Return the status byte for a previously-enqueued state-\n"
               "advancing command. One of FakeXEngine.STATUS_DROPPED /\n"
               "STATUS_ASSEMBLED / STATUS_SENT / STATUS_QUEUED /\n"
               "STATUS_SKIPPED.\n\n"
               "Snapshot semantic: the value may already be stale by the time\n"
               "the caller observes it (the worker thread or an ack may have\n"
               "advanced the state). Does NOT throw on a stopped FakeXEngine.\n\n"
               "Example:\n"
               "    fxe = FakeXEngine(xmd, ip_addrs, 1,\n"
               "                      time_samples_per_chunk=32768, debug=True)\n"
               "    fxe.enqueue_send_junk(0, 0)\n"
               "    fxe.synchronize(0)\n"
               "    s = fxe.get_minichunk_status(0, 0)\n"
               "    assert s == FakeXEngine.STATUS_ASSEMBLED\n\n"
               "The FakeXEngine's per-worker status vector grows without bound\n"
               "per state-advancing command (when debug=True), so don't use\n"
               "debug mode in production.\n\n"
               "Raises:\n"
               "    RuntimeError: If debug was not True at construction;\n"
               "    if worker_id is out of range; if no state-advancing\n"
               "    commands have been enqueued yet on this worker; or if\n"
               "    minichunk_index is out of range.")
          .def("get_debug_counters", &FakeXEngine::get_debug_counters,
               "Snapshot the four ack-prediction outcome counters as a\n"
               "tuple (length 4). Indices:\n"
               "    [0] unambiguous, DROPPED   (predicted + got DROPPED)\n"
               "    [1] unambiguous, ASSEMBLED (predicted + got ASSEMBLED)\n"
               "    [2] ambiguous,   DROPPED   (no prediction; got DROPPED)\n"
               "    [3] ambiguous,   ASSEMBLED (no prediction; got ASSEMBLED)\n\n"
               "Each ack byte processed in debug=True mode bumps exactly\n"
               "one entry. With debug=False, all four entries are zero\n"
               "(no acks ever arrive). Cheap relaxed-atomic loads; the\n"
               "snapshot is only 'consistent' if no acks are arriving\n"
               "concurrently -- typically call after synchronize() has\n"
               "drained all in-flight acks. Does NOT throw.")
          .def("get_worker_freq_channels",
               &FakeXEngine::get_worker_freq_channels,
               py::arg("worker_id"),
               "Return the round-robin subset of total frequency channels\n"
               "assigned to workers[worker_id]: channels worker_id,\n"
               "worker_id + nworkers, worker_id + 2*nworkers, ... (intersected\n"
               "with [0, total_nfreq)). Same content as Worker::xmd.freq_channels.\n\n"
               "Raises:\n"
               "    RuntimeError: If worker_id is out of range.")
          .def("stop", [](FakeXEngine &self) { self.stop(); },
               "Signal worker threads to stop. Any in-flight wait_until_processed\n"
               "/ enqueue_send_junk calls throw RuntimeError. Safe to call\n"
               "multiple times.")
          .def_property_readonly("is_stopped",
               [](FakeXEngine &self) {
                   // O(1) atomic load -- the property's "true" value just
                   // means stop() has been called; worker threads may
                   // still be exiting. Wait for the destructor (or a
                   // join) if you need to know they're fully done.
                   return self.is_stopped_cache.load(std::memory_order_acquire);
               },
               "True if stop() has been called (e.g. due to connection reset).\n"
               "Note: workers may still be in the process of exiting at the moment\n"
               "this returns true. Rely on the destructor's thread join if you need\n"
               "to know they've fully finished.")
          .def_readonly("xmd", &FakeXEngine::xmd,
               "X-engine metadata")
          .def_readonly("ip_addrs", &FakeXEngine::ip_addrs,
               "Receiver addresses in 'ip:port' format")
          .def_readonly("nworkers", &FakeXEngine::nworkers,
               "Number of worker threads")
          .def_readonly("time_samples_per_chunk", &FakeXEngine::time_samples_per_chunk,
               "Receiver-side chunk size in samples")
          .def_readonly("minichunks_per_chunk", &FakeXEngine::minichunks_per_chunk,
               "= time_samples_per_chunk / 256")
          .def_readonly("debug", &FakeXEngine::debug,
               "True if the FakeXEngine was constructed with debug=True.\n"
               "Read-only after construction.")
          .def_property_readonly_static("protocol_magic",
               [](py::object) { return FakeXEngine::protocol_magic; },
               "Protocol magic number (0xf4bf4b02)")
          .def_property_readonly_static("send_timeout_ms",
               [](py::object) { return FakeXEngine::send_timeout_ms; },
               "Timeout for send operations in milliseconds")
    ;

    // Constants (exposed as plain Python ints, addressable as
    // FakeXEngine.FLAG_ACK / FakeXEngine.STATUS_X).
    {
        py::object cls_obj = m.attr("FakeXEngine");
        cls_obj.attr("FLAG_ACK")          = py::int_(FakeXEngine::FLAG_ACK);
        cls_obj.attr("STATUS_DROPPED")    = py::int_(FakeXEngine::STATUS_DROPPED);
        cls_obj.attr("STATUS_ASSEMBLED")  = py::int_(FakeXEngine::STATUS_ASSEMBLED);
        cls_obj.attr("STATUS_SENT")       = py::int_(FakeXEngine::STATUS_SENT);
        cls_obj.attr("STATUS_QUEUED")     = py::int_(FakeXEngine::STATUS_QUEUED);
        cls_obj.attr("STATUS_SKIPPED")    = py::int_(FakeXEngine::STATUS_SKIPPED);
    }

    // Receiver: listens for TCP connections and reads data.
    // Skipped members: mutex, cv, is_started, is_stopped, error, listener_thread, reader_thread, pending_sockets (internal)
    // Skipped methods: _listener_main, _reader_main, listener_main, reader_main (private)
    // Note: Uses shared_ptr holder so Receivers can be passed to FrbServer.
    py::class_<Receiver, std::shared_ptr<Receiver>>(m, "Receiver",
        "Listens for TCP connections and reads data.\n\n"
        "A thread-backed class with two worker threads:\n"
        "  - listener: accepts incoming connections\n"
        "  - reader: reads data from all open connections using epoll")
          .def(py::init([](const std::string &address,
                           std::shared_ptr<AssembledFrameAllocator> allocator,
                           long consumer_id,
                           bool misbehaving_reads) {
               Receiver::Params params;
               params.address = address;
               params.allocator = allocator;
               params.consumer_id = consumer_id;
               params.misbehaving_reads = misbehaving_reads;
               return std::make_shared<Receiver>(params);
          }),
               py::arg("address"),
               py::arg("allocator"),
               py::arg("consumer_id"),
               py::arg("misbehaving_reads") = false,
               "Create a Receiver (does not start worker threads).\n\n"
               "Args:\n"
               "    address: Address to bind to (e.g. '127.0.0.1:5000')\n"
               "    allocator: AssembledFrameAllocator for output frames\n"
               "        (time_samples_per_chunk is taken from the allocator).\n"
               "    consumer_id: Consumer ID for the allocator\n"
               "    misbehaving_reads: If True, peer sockets accepted by\n"
               "        this Receiver will have set_misbehaving_reads()\n"
               "        called on them, which truncates each read() to a\n"
               "        log-uniform smaller size with probability 0.5.\n"
               "        Test-only -- do NOT enable in production.")
          .def("start", &Receiver::start,
               "Start the worker threads.\n\n"
               "Raises:\n"
               "    RuntimeError: If called twice or after stop().")
          .def("get_status", [](Receiver &self) {
               long num_conn, num_bytes;
               self.get_status(num_conn, num_bytes);
               return py::make_tuple(num_conn, num_bytes);
          }, "Returns (num_connections, num_bytes) tuple.")
          .def("get_frame_set", &Receiver::get_frame_set,
               py::call_guard<py::gil_scoped_release>(),
               "Retrieve the next assembled frame set (one time chunk, all beams)\n"
               "from the queue, blocking until a set is available. The GIL is\n"
               "released for the duration of the call.\n\n"
               "Raises:\n"
               "    RuntimeError: If the Receiver is stopped.")
          .def("stop", [](Receiver &self) { self.stop(); },
               "Signal worker threads to stop. Safe to call multiple times.")
          .def_property_readonly("address", [](const Receiver &self) { return self.params.address; },
               "Address bound to (e.g. '127.0.0.1:5000')")
    ;

    // FrbServer: gRPC server that queries Receivers and responds to RPCs.
    // Skipped members: params, rpc_service, rpc_server, mutex, is_started, is_stopped (internal)
    // Note: Params struct is not exposed; constructor takes receivers and address directly.
    // Note: FrbServer::create() is used internally, but appears as a constructor to Python.
    py::class_<FrbServer, std::shared_ptr<FrbServer>>(m, "FrbServer",
        "gRPC server that queries Receivers via RPC.\n\n"
        "Wraps multiple Receivers and exposes their status via gRPC. Also\n"
        "builds a DedispersionPlan (via a dedicated processing thread) once\n"
        "X-engine metadata arrives -- accessible via the 'plan' property.")
          .def(py::init([](const DedispersionConfig &config_prefilled,
                           std::vector<std::shared_ptr<Receiver>> receivers,
                           std::shared_ptr<FileWriter> file_writer,
                           const std::string &rpc_server_address,
                           int ringbuf_nchunks,
                           int min_data_mtu,
                           std::shared_ptr<BumpAllocator> host_allocator,
                           std::shared_ptr<BumpAllocator> gpu_allocator,
                           int cuda_device_id,
                           double processing_delay_sec,
                           bool randomize_weights,
                           const std::string &grouper_ip_addr,
                           bool no_dedispersion) {
               FrbServer::Params params;
               params.config_prefilled = config_prefilled;
               params.receivers = std::move(receivers);
               params.file_writer = std::move(file_writer);
               params.rpc_server_address = rpc_server_address;
               params.ringbuf_nchunks = ringbuf_nchunks;
               params.min_data_mtu = min_data_mtu;
               params.host_allocator = std::move(host_allocator);
               params.gpu_allocator = std::move(gpu_allocator);
               params.cuda_device_id = cuda_device_id;
               params.processing_delay_sec = processing_delay_sec;
               params.randomize_weights = randomize_weights;
               params.grouper_ip_addr = grouper_ip_addr;
               params.no_dedispersion = no_dedispersion;
               return FrbServer::create(params);
          }),
               py::arg("config_prefilled"),
               py::arg("receivers"), py::arg("file_writer"),
               py::arg("rpc_server_address"), py::arg("ringbuf_nchunks"),
               py::arg("min_data_mtu"),
               py::arg("host_allocator"),
               py::arg("gpu_allocator"),
               py::arg("cuda_device_id"),
               py::arg("processing_delay_sec") = 0.0,
               py::arg("randomize_weights") = true,
               py::arg("grouper_ip_addr") = "",
               py::arg("no_dedispersion") = false,
               "Create an FrbServer.\n\n"
               "Args:\n"
               "    config_prefilled: DedispersionConfig. Four members\n"
               "        (zone_nfreq, zone_freq_edges, time_sample_ms,\n"
               "        beams_per_gpu) will be overwritten by the processing\n"
               "        thread once X-engine metadata arrives; the rest\n"
               "        (including time_samples_per_chunk) are taken as-is.\n"
               "        The 'filled' config is available as server.plan.config\n"
               "        once the plan has been built.\n"
               "    receivers: List of Receiver objects to query\n"
               "    file_writer: FileWriter for saving frames to disk\n"
               "    rpc_server_address: gRPC server address (e.g. 'localhost:50051')\n"
               "    ringbuf_nchunks: Logical ring buffer length in time chunks\n"
               "    min_data_mtu: Minimum data-NIC MTU expected on the sender\n"
               "        side; surfaced via the GetConfig RPC.\n"
               "    host_allocator: BumpAllocator with af_rhost | af_zero, used by\n"
               "        the processing thread to back GpuDedisperser host buffers.\n"
               "    gpu_allocator: BumpAllocator with af_gpu | af_zero, used by\n"
               "        the processing thread to back GpuDedisperser GPU buffers.\n"
               "    cuda_device_id: CUDA device index. The processing thread and\n"
               "        the GpuDedisperser worker thread both call cudaSetDevice\n"
               "        with this value.\n"
               "    processing_delay_sec (default 0): Artificial per-frame\n"
               "        delay (in seconds) injected by the processing thread,\n"
               "        for simulating slow GPU work in pacing tests.\n"
               "    randomize_weights (default True): if True, the processing\n"
               "        thread does a one-time randomization of the dedisperser's\n"
               "        peak-finding weights during initialization. Placeholder\n"
               "        until a real variance calculation is implemented.\n"
               "    grouper_ip_addr (default ''): ip:port of the FrbGrouper to\n"
               "        feed (the FrbServer is the gRPC client). Empty => disabled\n"
               "        (GpuDedisperser built with num_consumers=0). Must be a\n"
               "        loopback address (CUDA IPC requires the same node/GPU).\n"
               "    no_dedispersion (default False): if True, the processing thread\n"
               "        skips ALL GPU work -- data is not even copied host->device,\n"
               "        and no dequantization/dedispersion kernels run. The\n"
               "        receive/assemble/ringbuf/reaper pipeline still runs in full.\n"
               "        Implies no grouper, so grouper_ip_addr must be '' (the\n"
               "        constructor asserts this).")
          .def("start", &FrbServer::start,
               "Start all Receivers.\n\n"
               "Raises:\n"
               "    RuntimeError: If called twice or after stop().")
          .def("stop", [](FrbServer &self) { self.stop(); },
               "Stop the server and all Receivers. Safe to call multiple times.")
          .def_property_readonly("plan", [](FrbServer &self) {
               std::lock_guard<std::mutex> lock(self.mutex);
               return self.plan;
          }, "DedispersionPlan built by the processing thread, or None if not yet constructed.")
          .def_property_readonly("dedisperser", [](FrbServer &self) {
               std::lock_guard<std::mutex> lock(self.mutex);
               return self.dedisperser;
          }, "GpuDedisperser built by the processing thread (alongside .plan),\n"
             "or None if not yet constructed.")
          .def_property_readonly("is_stopped", [](FrbServer &self) {
               std::lock_guard<std::mutex> lock(self.mutex);
               return self.is_stopped;
          }, "True if the server has been stopped, either by a stop() call or because\n"
             "a worker/reaper/processing thread threw an exception. When this transitions\n"
             "to True due to an internal error, the C++ side prints the exception message\n"
             "to stderr (see FrbServer.cpp's thread-main wrappers).")
          .def_property_readonly("host_allocator", [](FrbServer &self) {
               return self.params.host_allocator;
          }, "BumpAllocator for host (dd_host) memory. May be async; call\n"
             "wait_until_initialized() before start() if so.")
          .def_property_readonly("gpu_allocator", [](FrbServer &self) {
               return self.params.gpu_allocator;
          }, "BumpAllocator for GPU memory. May be async; call\n"
             "wait_until_initialized() before start() if so.")
          .def_property_readonly("frame_allocator", [](FrbServer &self) {
               return self.frame_allocator;
          }, "AssembledFrameAllocator (backs the receivers). Underlying memory\n"
             "is from the ring-buffer host BumpAllocator; may be async.")
    ;

    // FrbGrouper: gRPC *server* side of the FrbGrouper service. Downstream
    // consumer of an FrbServer's GpuDedisperser::output_ringbuf over CUDA IPC.
    // py::dynamic_attr() lets the Python injection attach dedispersion_plan_yaml.
    // Injections (pirate_frb/pybind11_injections.py) add __enter__/__exit__ and
    // a get_output() context manager.
    py::class_<FrbGrouper, std::shared_ptr<FrbGrouper>>(m, "FrbGrouper", py::dynamic_attr(),
        "gRPC server that receives a GpuDedisperser output ring buffer from an\n"
        "FrbServer over CUDA IPC. Use as a context manager (see injections):\n"
        "    with pirate_frb.rpc.FrbGrouper('127.0.0.1:7000') as g:\n"
        "        with g.get_output(seq_id) as outputs: ...")
          .def(py::init([](const std::string &addr){ return FrbGrouper::create(addr); }),
               py::arg("listen_address"))
          // open() must stay interruptible by Ctrl-C while it blocks waiting for
          // a client. We can't just release the GIL for the whole call (Python
          // signal handlers wouldn't run); instead we drive the wait in 0.5s
          // steps, releasing the GIL during each wait and reacquiring it between
          // steps to poll for pending Python signals (PyErr_CheckSignals() != 0
          // => the SIGINT handler raised KeyboardInterrupt, propagated via
          // error_already_set).
          .def("open", [](FrbGrouper &self) {
               self.start_listening();                 // GIL held; quick (bind + spawn)
               for (;;) {
                   bool ready;
                   { py::gil_scoped_release nogil; ready = self.wait_for_handshake(500); }
                   if (ready)
                       break;
                   if (PyErr_CheckSignals() != 0)       // e.g. Ctrl-C
                       throw py::error_already_set();
               }
          }, "Start listening + block until a client connects and its handshake\n"
             "is processed. Interruptible by Ctrl-C.")
          .def("close", &FrbGrouper::close, py::call_guard<py::gil_scoped_release>(),
               "Stop the session + join the send thread + shut down the gRPC server.")
          .def("stop", [](FrbGrouper &self){ self.stop(); },
               "Put the FrbGrouper into the stopped state (idempotent).")
          .def("acquire_output", &FrbGrouper::acquire_output,
               py::arg("seq_id"), py::call_guard<py::gil_scoped_release>(),
               "Block until produced_seq_id has been received for 'seq_id'; return\n"
               "a per-batch slice (nbeams == beams_per_batch) of output_ringbuf.")
          .def("release_output", &FrbGrouper::release_output, py::arg("seq_id"),
               "Record that the caller is done with 'seq_id' (emits CONSUMED).")
          .def_property_readonly("is_stopped", &FrbGrouper::is_stopped_pub)
          .def_readonly("cuda_device_id", &FrbGrouper::cuda_device_id)
          .def_readonly("dtype", &FrbGrouper::dtype)
          .def_readonly("nt_in", &FrbGrouper::nt_in)
          .def_readonly("total_beams", &FrbGrouper::total_beams,
               "Total beams per chunk (= beams_per_gpu). NOT the output-ringbuf\n"
               "leading axis, which is num_batch_slots * beams_per_batch.")
          .def_readonly("beams_per_batch", &FrbGrouper::beams_per_batch)
          .def_readonly("nbatches", &FrbGrouper::nbatches,
               "Beam-batches per time chunk (= total_beams / beams_per_batch). NOT\n"
               "num_batch_slots (the output ring-buffer depth).")
          .def_readonly("num_batch_slots", &FrbGrouper::num_batch_slots)
          .def_readonly("ntrees", &FrbGrouper::ntrees)
          .def_readonly("ndm_out", &FrbGrouper::ndm_out)
          .def_readonly("nt_out", &FrbGrouper::nt_out)
          .def_readonly("dedispersion_config", &FrbGrouper::dedispersion_config)
          .def_readonly("xengine_metadata", &FrbGrouper::xengine_metadata)
          .def_readonly("xengine_metadata_yaml_string", &FrbGrouper::xengine_metadata_yaml_string)
          .def_readonly("dedispersion_config_yaml_string", &FrbGrouper::dedispersion_config_yaml_string)
          .def_readonly("dedispersion_plan_yaml_string", &FrbGrouper::dedispersion_plan_yaml_string)
          // NOTE: FrbGrouper::dedispersion_plan_yaml (YAML::Node) is intentionally
          // NOT wrapped; the injection adds a Python dedispersion_plan_yaml attribute
          // parsed from dedispersion_plan_yaml_string. output_ringbuf is private
          // (reached only via acquire_output).
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

