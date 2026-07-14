// Python bindings for core classes (pirate_frb.core subpackage).
// See pirate_pybind11.cu for the main module definition.

#define PY_ARRAY_UNIQUE_SYMBOL PyArray_API_pirate
#define NO_IMPORT_ARRAY  // Secondary file: don't call _import_array()
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <pybind11/stl.h>
#include <ksgpu/pybind11.hpp>

#include "../include/pirate/AssembledFrame.hpp"
#include "../include/pirate/SimulatedFrameFactory.hpp"
#include "../include/pirate/simpulse.hpp"   // SinglePulse (AssembledFrame.randomize arg)
#include "../include/pirate/BumpAllocator.hpp"
#include "../include/pirate/CudaStreamPool.hpp"
#include "../include/pirate/SlabAllocator.hpp"
#include "../include/pirate/DedispersionConfig.hpp"  // for PrimaryTree
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
#include "../include/pirate/Hwtest.hpp"

using namespace std;
using namespace ksgpu;
using namespace pirate;
namespace py = pybind11;


namespace pirate {


// -------------------------------------------------------------------------------------------------
//
// Vectorized ("batch") decode_argmax*() helpers, bound as methods on both DedispersionPlan
// (in pirate_pybind11.cpp, which declares these prototypes) and FrbGrouper (below, forwarding
// to its internal incomplete_plan). Non-static, since the two class bindings live in
// different source files.
//
// Inputs are 1-d contiguous nonempty host arrays, one event per element. (Python callers
// should short-circuit the zero-event case: the Array -> numpy caster rejects zero-size
// arrays.) Outputs are freshly-allocated numpy arrays. Implemented as loops over the scalar
// DedispersionPlan methods, so per-element validation (index ranges, malformed tokens)
// comes from decode_argmax() itself.


template<typename T>
static void _check_batch_arg(const char *fname, const char *arg_name, const ksgpu::Array<T> &a, long n)
{
    if ((a.ndim != 1) || !a.is_fully_contiguous() || !a.on_host() || (a.size != n)) {
        stringstream ss;
        ss << fname << ": argument '" << arg_name << "' must be a 1-d contiguous host array"
           << " of length " << n << ", got shape " << a.shape_str();
        throw runtime_error(ss.str());
    }
}


py::tuple _decode_argmax_batch(
    const DedispersionPlan &plan, const Array<uint> &tokens,
    const Array<long> &itrees, const Array<long> &idms, const Array<long> &itimes)
{
    const char *fname = "decode_argmax_batch";
    long n = tokens.size;

    if (n <= 0)
        throw runtime_error("decode_argmax_batch: empty input arrays (callers should "
                            "short-circuit the zero-event case)");

    _check_batch_arg(fname, "tokens", tokens, n);
    _check_batch_arg(fname, "itrees", itrees, n);
    _check_batch_arg(fname, "idms", idms, n);
    _check_batch_arg(fname, "itimes", itimes, n);

    Array<long> fmins({n}, af_uhost);
    Array<long> fmaxs({n}, af_uhost);
    Array<long> tlos({n}, af_uhost);
    Array<long> this_({n}, af_uhost);
    Array<long> ps({n}, af_uhost);

    for (long i = 0; i < n; i++)
        plan.decode_argmax(tokens.data[i], itrees.data[i], idms.data[i], itimes.data[i],
                           fmins.data[i], fmaxs.data[i], tlos.data[i], this_.data[i], ps.data[i]);

    return py::make_tuple(fmins, fmaxs, tlos, this_, ps);
}


py::tuple _decode_argmax2_batch(
    const DedispersionPlan &plan, const Array<long> &itrees,
    const Array<long> &fmins, const Array<long> &fmaxs,
    const Array<long> &tlos, const Array<long> &this_, const Array<long> &ps)
{
    const char *fname = "decode_argmax2_batch";
    long n = itrees.size;

    if (n <= 0)
        throw runtime_error("decode_argmax2_batch: empty input arrays (callers should "
                            "short-circuit the zero-event case)");

    _check_batch_arg(fname, "itrees", itrees, n);
    _check_batch_arg(fname, "fmins", fmins, n);
    _check_batch_arg(fname, "fmaxs", fmaxs, n);
    _check_batch_arg(fname, "tlos", tlos, n);
    _check_batch_arg(fname, "this", this_, n);
    _check_batch_arg(fname, "ps", ps, n);

    Array<double> freqs_lo({n}, af_uhost);
    Array<double> freqs_hi({n}, af_uhost);
    Array<double> dms({n}, af_uhost);
    Array<double> timestamps({n}, af_uhost);
    Array<double> widths({n}, af_uhost);

    for (long i = 0; i < n; i++)
        plan.decode_argmax2(itrees.data[i], fmins.data[i], fmaxs.data[i],
                            tlos.data[i], this_.data[i], ps.data[i],
                            freqs_lo.data[i], freqs_hi.data[i], dms.data[i],
                            timestamps.data[i], widths.data[i]);

    return py::make_tuple(freqs_lo, freqs_hi, dms, timestamps, widths);
}


// -------------------------------------------------------------------------------------------------


void register_core_bindings(pybind11::module &m)
{
    // BumpAllocator: Thread-safe bump allocator for GPU/host memory
    // Wrapped with shared_ptr for proper lifetime management when arrays reference the allocator.
    // Note: Python injections in pirate_frb/core/BumpAllocator.py handle aflags and dtype conversions.
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
            // Sync-mode construction registers/zeros up to 100s of GB; release the GIL.
            py::call_guard<py::gil_scoped_release>(),
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
            [](const BumpAllocator &self) { return self.get_nbytes_allocated(); },
            "Bytes allocated so far (aligned to 128-byte cache lines)")
        .def_readonly("aflags", &BumpAllocator::aflags,
            "Memory allocation flags")
        .def_readonly("capacity", &BumpAllocator::capacity,
            "Total capacity in bytes, or -1 for dummy mode")
        .def("wait_until_initialized", &BumpAllocator::wait_until_initialized,
            py::call_guard<py::gil_scoped_release>(),
            "In async mode: block until init complete, or rethrow async-init "
            "exception. In sync mode: no-op. Releases the GIL while blocking.")
        .def("is_initialized", &BumpAllocator::is_initialized,
            "Non-blocking poll: True iff init has completed and the allocator "
            "has not been stopped.")
        .def("_allocate_array_raw",
            [](std::shared_ptr<BumpAllocator> self, ksgpu::Dtype dtype, const std::vector<long> &shape) {
                return self->_allocate_array_internal(dtype, shape.size(), shape.data(), nullptr);
            },
            py::arg("dtype"), py::arg("shape"),
            // Blocks on async init; in dummy mode does a fresh (possibly zeroed) allocation.
            py::call_guard<py::gil_scoped_release>(),
            "Internal: allocate array with dtype and shape (use allocate_array() instead)")
    ;

    // SlabAllocator: Thread-safe pool allocator for fixed-size slabs
    // Note: Python injections in pirate_frb/core/SlabAllocator.py handle aflags conversion.
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
            // Allocates (and for af_rhost, cuda-registers) the full capacity; release the GIL.
            py::call_guard<py::gil_scoped_release>(),
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
            py::call_guard<py::gil_scoped_release>(),
            "If backed by an async BumpAllocator: block until it's initialized\n"
            "(or rethrow async-init exception). Otherwise no-op. Releases the\n"
            "GIL while blocking.")
        .def("is_initialized", &SlabAllocator::is_initialized,
            "Non-blocking poll: True iff the underlying BumpAllocator is ready\n"
            "to serve allocations (delegates to bump_allocator.is_initialized()).\n"
            "Always True in dummy mode.")
        .def("num_free_slabs", &SlabAllocator::num_free_slabs,
            "Number of slabs currently available. Throws in dummy mode.")
        .def("num_total_slabs", &SlabAllocator::num_total_slabs,
            py::arg("blocking") = false,
            py::call_guard<py::gil_scoped_release>(),
            "Total number of slabs in the pool. Throws in dummy mode.\n\n"
            "If blocking=True, waits until the slab size has been established\n"
            "(by the first get_slab()); releases the GIL while blocking.")
        .def("get_slab_size", &SlabAllocator::get_slab_size,
            "Established slab size in bytes. Throws if the slab size has not\n"
            "been established yet (by the first get_slab()).")
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
        "Data frame containing beamformed data for one (time_chunk, beam_id) pair.\n\n"
        "The data is stored in its raw binary format (int4, special value -8 indicates\n"
        "masked data, float16 offsets/scales at lower resolution). Most python callers\n"
        "will want to convert to float32 by calling dequantize().")
        .def_readonly("nfreq", &AssembledFrame::nfreq,
            "Number of frequency channels.")
        .def_readonly("ntime", &AssembledFrame::ntime,
            "Number of time samples in this frame's time chunk.")
        .def_readonly("beam_id", &AssembledFrame::beam_id,
            "Beam id of this frame.")
        .def_readonly("time_chunk_index", &AssembledFrame::time_chunk_index,
            "Index of this frame's time chunk (one \"chunk\" is ntime samples).")
        .def_property_readonly("fpga_seq_start", &AssembledFrame::fpga_seq_start,
            "FPGA sequence number at the start of this frame's time chunk\n"
            "(= time_chunk_index * ntime * metadata.seq_per_frb_time_sample).")
        .def_property_readonly("fpga_seq_end", &AssembledFrame::fpga_seq_end,
            "FPGA sequence number one-past-the-end of this frame's time chunk\n"
            "(= start of the next chunk).")
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
                // Snapshot the lock-protected member first, to avoid racing the
                // reaper thread (same pattern as write_asdf()).
                Array<void> local_data;
                {
                    std::lock_guard<std::mutex> guard(self.mutex);
                    local_data = self.data;
                }
                if (local_data.size == 0)
                    throw std::runtime_error("AssembledFrame.data: frame has been reaped (arrays released)");

                // Convert int4 array (nfreq, ntime) to uint8 array (nfreq, ntime/2).
                Array<uint8_t> arr;
                arr.data = static_cast<uint8_t *>(local_data.data);
                arr.ndim = 2;
                arr.shape[0] = self.nfreq;
                arr.shape[1] = self.ntime / 2;
                arr.size = self.nfreq * (self.ntime / 2);
                arr.strides[0] = self.ntime / 2;
                arr.strides[1] = 1;
                arr.dtype = Dtype::native<uint8_t>();
                arr.aflags = local_data.aflags;
                arr.base = local_data.base;
                arr.check_invariants("AssembledFrame::data getter");
                return arr;
            },
            "Raw int4 data (pre-scaled) as uint8 array with shape (nfreq, ntime/2).\n\n"
            "Most python callers will want to call dequantize(), rather than working directly\n"
            "with the raw data.")
        .def_property_readonly("scales_offsets",
            [](const AssembledFrame &self) {
                // Snapshot under the lock, to avoid racing the reaper thread (see
                // the 'data' getter above). The Array<void> already carries dtype
                // Dtype(df_float, 16); ksgpu's pybind11 type_caster maps that to
                // numpy np.float16.
                std::lock_guard<std::mutex> guard(self.mutex);
                return self.scales_offsets;
            },
            "Raw scales/offsets as float16 array with shape (nfreq, ntime/256, 2), "
            "where the last axis is {scale, offset}.\n\n"
            "Most python callers will want to call dequantize(), rather than working directly\n"
            "with the scales/offsets.")
        .def("dequantize", &AssembledFrame::dequantize,
            py::call_guard<py::gil_scoped_release>(),   // pinned alloc + full CPU dequantization loop
            "Convert raw data/scales/offsets to a float32 array of shape (nfreq, ntime).\n\n"
            "NOTE: we currently convert masked samples (represented by raw int4 value -8) to\n"
            "zeroes. We don't currently define a separate method to return the boolean mask,\n"
            "but if you need this let me know.")
        .def("write_asdf", &AssembledFrame::write_asdf,
            py::arg("filename"), py::arg("sync") = true, py::arg("verbose") = false,
            py::call_guard<py::gil_scoped_release>(),   // 100s of MB of file I/O + fsync
            "Write this AssembledFrame to an ASDF file.\n\n"
            "If sync=True (default), fsync() the file after writing to avoid\n"
            "runaway page cache usage.\n\n"
            "If verbose=True, emit comments throughout the YAML header. The\n"
            "comments are detailed enough that the header serves as self-\n"
            "contained documentation of the file format. Intended for use\n"
            "with 'pirate_frb show_file_format'.")
        .def_static("make_uninitialized", &AssembledFrame::make_uninitialized,
            py::arg("xmd"), py::arg("ntime"), py::arg("beam_id"), py::arg("time_chunk_index"),
            py::call_guard<py::gil_scoped_release>(),   // pinned-host (cudaHostAlloc-scale) allocations
            "Create an AssembledFrame backed by the given XEngineMetadata, with\n"
            "freshly-allocated data / scales_offsets arrays whose CONTENTS ARE\n"
            "UNSPECIFIED -- the caller must fill them (e.g. via randomize()).\n\n"
            "Throws if xmd is None or if beam_id is not in xmd.beam_ids.\n"
            "nfreq is taken from xmd.get_total_nfreq().\n\n"
            "xmd.freq_channels is ignored. Callers should typically pass a\n"
            "frequency-scrubbed xmd so the returned frame matches the\n"
            "'always frequency-scrubbed' invariant on AssembledFrame.metadata.")
        .def_static("from_asdf", &AssembledFrame::from_asdf,
            py::arg("filename"),
            py::call_guard<py::gil_scoped_release>(),   // file I/O + pinned-host allocation
            "Read an AssembledFrame back from an ASDF file.\n\n"
            "Note: XEngineMetadata is \"projected\" through ASDF -- after reading, the\n"
            "metadata's beam_ids / beam_positions_{x,y} are length-1 (just this\n"
            "frame's beam) and freq_channels is empty.")
        .def("randomize", &AssembledFrame::randomize,
            // normalize/gaussian required (be explicit); sp/dt_sp default to "no pulse".
            py::arg("normalize"), py::arg("gaussian"),
            py::arg("sp") = std::shared_ptr<const simpulse::SinglePulse>(),
            py::arg("dt_sp") = (long) 0,
            py::call_guard<py::gil_scoped_release>(),
            "Fill the frame with random int4 data (intended for testing), optionally\n"
            "injecting a simulated FRB pulse.\n\n"
            "If gaussian=False, data is uniform over [-8, +7]. If gaussian=True, data is\n"
            "simulated Gaussian noise quantized to int4 and clamped to [-7, +7]\n"
            "(the -8 sentinel is never produced).\n\n"
            "If normalize=False, scales are uniform in [0,1] and offsets in [-1,1].\n"
            "If normalize=True, then offsets are zero, and scales are chosen to match\n"
            "metadata.noise_variance.\n\n"
            "sp (default None): a simulated FRB to add on top of the noise.")
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
        .def("randomize", &AssembledFrameSet::randomize,
            py::arg("normalize"), py::arg("gaussian"),
            py::call_guard<py::gil_scoped_release>(),
            "Fill every frame in the set with random int4 test data, by calling\n"
            "AssembledFrame.randomize(normalize, gaussian) on each frame in turn\n"
            "(serial, single-threaded). See AssembledFrame.randomize for the meaning\n"
            "of normalize and gaussian. To randomize a stream of sets in parallel,\n"
            "use a SimulatedFrameFactory.")
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
        // nfreq / beam_ids are written by initialize_metadata() under the
        // allocator's lock, so (unlike the ctor-constant
        // time_samples_per_chunk below) a raw def_readonly would read them
        // unsynchronized -- a torn vector copy in the worst case. Route
        // through the lock-synchronized getters, like the 'metadata'
        // property.
        .def_property_readonly("nfreq", &AssembledFrameAllocator::get_nfreq)
        .def_readonly("time_samples_per_chunk", &AssembledFrameAllocator::time_samples_per_chunk)
        .def_property_readonly("beam_ids", &AssembledFrameAllocator::get_beam_ids)
        .def_property_readonly("metadata",
            [](AssembledFrameAllocator &self) {
                // Route through get_metadata(blocking=false) so the read is
                // properly lock-synchronized; pybind11 maps a null shared_ptr
                // to Python None automatically.
                auto m = self.get_metadata(/*blocking=*/false);
                return std::const_pointer_cast<XEngineMetadata>(m);
            },
            "Shared XEngineMetadata, set the first time initialize_metadata() is\n"
            "called. None if no consumer has called initialize_metadata() yet.\n"
            "Read-only by convention. (Note: freq_channels is cleared on the\n"
            "canonical copy.)")
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
            py::call_guard<py::gil_scoped_release>(),
            "Block until some caller has invoked initialize_initial_chunk(),\n"
            "then return the established initial_time_chunk.\n\n"
            "Releases the GIL: the waker may be another python thread (which\n"
            "could never run if we held the GIL while blocked), or a Receiver\n"
            "reader thread whose arrival time is unbounded.")
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
            py::call_guard<py::gil_scoped_release>(),
            "Get the canonical XEngineMetadata pointer.\n\n"
            "Args:\n"
            "    blocking: If True, wait until any consumer has called\n"
            "        initialize_metadata().\n\n"
            "Returns:\n"
            "    XEngineMetadata object (or None if non-blocking and not yet set).\n"
            "    Note: freq_channels is cleared on the canonical copy.\n\n"
            "Releases the GIL: with blocking=True, the initializer may be another\n"
            "python thread, or a Receiver reader thread whose arrival time is\n"
            "unbounded.")
        .def("num_free_frames", &AssembledFrameAllocator::num_free_frames,
            py::arg("permissive") = false,
            "Number of frames currently available in the pool.\n\n"
            "Throws in dummy mode or if not initialized.")
        .def("num_total_frames", &AssembledFrameAllocator::num_total_frames,
            py::arg("blocking") = false,
            py::call_guard<py::gil_scoped_release>(),
            "Total number of frames in the pool.\n\n"
            "Throws in dummy mode or if not initialized. Releases the GIL\n"
            "(with blocking=True, waits for the slab size to be established).")
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

    // SimulatedFrameFactory::Event: one recorded FRB-injection event. Bound as a top-level
    // python class "SimulatedFrameFactoryEvent". NOTE: this name is deliberately NOT re-exported
    // into pirate_frb.core -- it is only reachable as pirate_frb.pirate_pybind11.SimulatedFrameFactoryEvent
    // (see SimulatedFrameFactory._pop_events / the pop_events() injection).
    py::class_<SimulatedFrameFactory::Event>(m, "SimulatedFrameFactoryEvent",
        "One simulated-FRB injection event recorded by SimulatedFrameFactory (returned by\n"
        "_pop_events()). fpga_timestamp is the arrival at the LOWEST frequency of the FULL band\n"
        "(not the pulse's subband), as an FPGA sequence number.")
        .def_readonly("beam_id", &SimulatedFrameFactory::Event::beam_id)
        .def_readonly("fpga_timestamp", &SimulatedFrameFactory::Event::fpga_timestamp)
        .def_readonly("dm", &SimulatedFrameFactory::Event::dm)
        .def_readonly("snr", &SimulatedFrameFactory::Event::snr)
        .def_readonly("width_ms", &SimulatedFrameFactory::Event::width_ms)
        .def_readonly("subband_freq_lo_MHz", &SimulatedFrameFactory::Event::subband_freq_lo_MHz)
        .def_readonly("subband_freq_hi_MHz", &SimulatedFrameFactory::Event::subband_freq_hi_MHz)
    ;

    // SimulatedFrameFactory: hands a consumer a stream of pre-randomized
    // AssembledFrameSets. Bound with a kwargs constructor that fills Params.
    py::class_<SimulatedFrameFactory, std::shared_ptr<SimulatedFrameFactory>>(m, "SimulatedFrameFactory",
        "Produces a stream of pre-randomized AssembledFrameSets for an external\n"
        "consumer, staying a few frames ahead (bounded by frame_set_queue_size and\n"
        "the allocator's slab pool). Owns a producer thread plus a randomizer-thread\n"
        "pool. The allocator (num_consumers=1) must already be initialized\n"
        "(initialize_metadata + initialize_initial_chunk), and the factory MUST be\n"
        "constructed inside a ThreadAffinity context manager -- the spawned threads\n"
        "inherit the caller's vcpu affinity. stop() propagates to the allocator.")
        .def(py::init([](std::shared_ptr<AssembledFrameAllocator> allocator,
                         long num_randomizer_threads, bool normalized, bool gaussian,
                         long frame_set_queue_size, bool simulate_frbs,
                         double frb_dm0, double frb_max_dm, double frb_max_width_ms,
                         double frb_snr, std::vector<double> frb_subband_fmin_MHz,
                         std::vector<double> frb_subband_fmax_MHz, double frb_gap_sec,
                         long num_frb_simulator_threads, long single_pulse_queue_size) {
                 SimulatedFrameFactory::Params p;
                 p.allocator = std::move(allocator);
                 p.num_randomizer_threads = num_randomizer_threads;
                 p.normalized = normalized;
                 p.gaussian = gaussian;
                 p.frame_set_queue_size = frame_set_queue_size;
                 p.simulate_frbs = simulate_frbs;
                 p.frb_dm0 = frb_dm0;
                 p.frb_max_dm = frb_max_dm;
                 p.frb_max_width_ms = frb_max_width_ms;
                 p.frb_snr = frb_snr;
                 p.frb_subband_fmin_MHz = std::move(frb_subband_fmin_MHz);
                 p.frb_subband_fmax_MHz = std::move(frb_subband_fmax_MHz);
                 p.frb_gap_sec = frb_gap_sec;
                 p.num_frb_simulator_threads = num_frb_simulator_threads;
                 p.single_pulse_queue_size = single_pulse_queue_size;
                 return std::make_unique<SimulatedFrameFactory>(p);
             }),
             py::arg("allocator"),
             py::arg("num_randomizer_threads"),
             py::arg("normalized") = true,
             py::arg("gaussian") = true,
             py::arg("frame_set_queue_size") = 4,
             py::arg("simulate_frbs") = false,
             py::arg("frb_dm0") = -1.0,
             py::arg("frb_max_dm") = -1.0,
             py::arg("frb_max_width_ms") = -1.0,
             py::arg("frb_snr") = -1.0,
             py::arg("frb_subband_fmin_MHz") = std::vector<double>(),
             py::arg("frb_subband_fmax_MHz") = std::vector<double>(),
             py::arg("frb_gap_sec") = 0.0,
             py::arg("num_frb_simulator_threads") = 0,
             py::arg("single_pulse_queue_size") = 0,
             "num_randomizer_threads (>= 1): size of the randomizer-thread pool that\n"
             "parallelizes per-beam randomize() within a set; the caller sizes it\n"
             "(run_fake_xengine uses min(nbeams, num_vcpus/2)). normalized (default\n"
             "True): calibrate scales/offsets to the metadata's per-zone noise variance\n"
             "(else arbitrary uniform-junk). gaussian (default True): simulated Gaussian\n"
             "int4 data clamped to [-7,+7] (else uniform [-8,+7]). frame_set_queue_size\n"
             "(default 4): output-queue depth (how many randomized sets the producer may\n"
             "stay ahead of the consumer).\n\n"
             "simulate_frbs (default False): inject simulated FRBs -- the producer keeps\n"
             "one 'active' pulse per beam, placed at a random phase within the chunk\n"
             "where it is assigned, and replaced once entirely in the past. Requires\n"
             "normalized=True and gaussian=True, plus the remaining args:\n"
             "frb_dm0/frb_max_dm (DM in [0, frb_max_dm], log(DM + frb_dm0) uniform),\n"
             "frb_max_width_ms (intrinsic width log-uniform over [min(dt/3, wmax), wmax]),\n"
             "frb_snr (matched-filter SNR over the pulse's subband),\n"
             "frb_subband_fmin_MHz/frb_subband_fmax_MHz (equal-length lists; each pulse\n"
             "picks one subband uniformly at random; each subband must overlap the band),\n"
             "frb_gap_sec (>= 0, default 0: extra padding in seconds between consecutive\n"
             "FRBs on a beam, rounded to whole time samples),\n"
             "num_frb_simulator_threads (>= 1) and single_pulse_queue_size (>= 1;\n"
             "~nbeams recommended, since up to nbeams pulses can be popped per chunk).\n"
             "Injected FRBs are recorded as events, retrievable via pop_events().")
        .def_readonly("nbeams", &SimulatedFrameFactory::nbeams)
        .def_property_readonly("num_randomizer_threads",
            [](const SimulatedFrameFactory &f) { return f.params.num_randomizer_threads; })
        .def_property_readonly("normalized",
            [](const SimulatedFrameFactory &f) { return f.params.normalized; })
        .def_property_readonly("gaussian",
            [](const SimulatedFrameFactory &f) { return f.params.gaussian; })
        .def_property_readonly("frame_set_queue_size",
            [](const SimulatedFrameFactory &f) { return f.params.frame_set_queue_size; })
        .def_property_readonly("simulate_frbs",
            [](const SimulatedFrameFactory &f) { return f.params.simulate_frbs; })
        .def_property_readonly("frb_dm0",
            [](const SimulatedFrameFactory &f) { return f.params.frb_dm0; })
        .def_property_readonly("frb_max_dm",
            [](const SimulatedFrameFactory &f) { return f.params.frb_max_dm; })
        .def_property_readonly("frb_max_width_ms",
            [](const SimulatedFrameFactory &f) { return f.params.frb_max_width_ms; })
        .def_property_readonly("frb_snr",
            [](const SimulatedFrameFactory &f) { return f.params.frb_snr; })
        .def_property_readonly("frb_subband_fmin_MHz",
            [](const SimulatedFrameFactory &f) { return f.params.frb_subband_fmin_MHz; })
        .def_property_readonly("frb_subband_fmax_MHz",
            [](const SimulatedFrameFactory &f) { return f.params.frb_subband_fmax_MHz; })
        .def_property_readonly("frb_gap_sec",
            [](const SimulatedFrameFactory &f) { return f.params.frb_gap_sec; })
        .def_property_readonly("num_frb_simulator_threads",
            [](const SimulatedFrameFactory &f) { return f.params.num_frb_simulator_threads; })
        .def_property_readonly("single_pulse_queue_size",
            [](const SimulatedFrameFactory &f) { return f.params.single_pulse_queue_size; })
        .def("get_frame_set", &SimulatedFrameFactory::get_frame_set,
            py::call_guard<py::gil_scoped_release>(),
            "Block until a randomized AssembledFrameSet is available and return it\n"
            "(FIFO / allocator order).\n\n"
            "Returns:\n"
            "    The next AssembledFrameSet, or None if the factory was cleanly\n"
            "    stopped (normal end of production -- the consumer loop's signal\n"
            "    to wind down).\n\n"
            "Raises:\n"
            "    RuntimeError: Re-raises the factory's saved root-cause error if\n"
            "    it was stopped by an error.")
        .def("stop", [](SimulatedFrameFactory &self) { self.stop(); },
            py::call_guard<py::gil_scoped_release>(),
            "Stop the factory (and its allocator); wakes all threads. Idempotent.\n"
            "This is a clean stop (normal termination): a blocked or later\n"
            "get_frame_set() call returns None rather than raising.")

        // One awkward aspect of the current code: we have two very similar ways to represent an event
        // list, either a python FrbSifterEvents object, or a C++ vector<SimulatedFrameFactory::Event>.
        // The _pop_events() method returns the "C++ representation", where it will be converted to
        // the "python representation" by the pop_events() method injection.
        //
        // This design is awkward but I think it's the least bad option. (The only real alternative seems
        // to be rewriting the FrbSifterClient in C++, and I prefer the current python implementation.)

        .def("_pop_events", &SimulatedFrameFactory::pop_events,
            py::call_guard<py::gil_scoped_release>(),
            "Return the recorded FRB-injection events (a list of SimulatedFrameFactoryEvent) and\n"
            "clear the internal list, so each event is returned exactly once. This method is\n"
            "wrapped by the pop_events() method injection, which returns an 'FrbSifterEvents'\n"
            "(whereas _pop_events() returns vector<SimulatedFrameFactoryEvent>).")
    ;

    // CudaStreamPool: always accessed via shared_ptr.
    // Stream members are exposed to python as CudaStreamWrapper objects.
    // No class docstring here: CudaStreamPool's docstring lives in the Python
    // injector (pirate_frb/core/CudaStreamPool.py), since its Python interface is
    // the injected stream accessors that return ksgpu.CudaStreamWrapper objects
    // (option 2 in notes/docstrings.md).
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

    py::class_<FrequencySubbands>(m, "FrequencySubbands")
          // Constructors
          .def(py::init<>())  // default constructor
          .def(py::init<const std::vector<long> &>(), py::arg("subband_counts"))
          .def(py::init<const std::vector<long> &, double, double>(),
               py::arg("subband_counts"), py::arg("fmin"), py::arg("fmax"))
          // Data members (readonly since they are computed from subband_counts)
          .def_readonly("subband_counts", &FrequencySubbands::subband_counts)
          .def_readonly("pf_rank", &FrequencySubbands::pf_rank)
          .def_readonly("N", &FrequencySubbands::N)
          .def_readonly("M", &FrequencySubbands::M)
          .def_readonly("m_to_n", &FrequencySubbands::m_to_n)
          .def_readonly("m_to_d", &FrequencySubbands::m_to_d)
          .def_readonly("n_to_flo", &FrequencySubbands::n_to_flo)
          .def_readonly("n_to_fhi", &FrequencySubbands::n_to_fhi)
          .def_readonly("n_to_mbase", &FrequencySubbands::n_to_mbase)
          .def_readonly("f_to_freq", &FrequencySubbands::f_to_freq)
          .def_readonly("fmin", &FrequencySubbands::fmin)
          .def_readonly("fmax", &FrequencySubbands::fmax)
          // Inline methods
          .def("m_to_flo", &FrequencySubbands::m_to_flo, py::arg("m"))
          .def("m_to_fhi", &FrequencySubbands::m_to_fhi, py::arg("m"))
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
               py::arg("subband_counts"), py::arg("early_trigger_level"), py::arg("new_pf_rank"))
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

    // DedispersionConfig::PrimaryTree (nested struct)
    py::class_<DedispersionConfig::PrimaryTree>(m, "PrimaryTree",
        "Configuration of a single primary tree (one DM range searched).\n\n"
        "A primary tree is expanded into (num_early_triggers+1) dedispersion trees:\n"
        "the main full-band tree, plus one early-trigger tree for each\n"
        "early_trigger_level = 1..num_early_triggers. The remaining members define the maximum\n"
        "width for peak detection and downsampling factors for both the coarse-grained\n"
        "and weights arrays relative to tree resolution. All numeric members must be\n"
        "powers of two.")
          .def(py::init<>(),
               "Create a PrimaryTree with default (zero) values.")
          .def(py::init([](long num_early_triggers, long max_width, long dm_downsampling,
                          long time_downsampling, long wt_dm_downsampling, long wt_time_downsampling) {
                   DedispersionConfig::PrimaryTree pt;
                   pt.num_early_triggers = num_early_triggers;
                   pt.max_width = max_width;
                   pt.dm_downsampling = dm_downsampling;
                   pt.time_downsampling = time_downsampling;
                   pt.wt_dm_downsampling = wt_dm_downsampling;
                   pt.wt_time_downsampling = wt_time_downsampling;
                   return pt;
               }),
               py::arg("num_early_triggers"),
               py::arg("max_width"),
               py::arg("dm_downsampling"),
               py::arg("time_downsampling"),
               py::arg("wt_dm_downsampling"),
               py::arg("wt_time_downsampling"),
               "Create a PrimaryTree.\n\n"
               "Args:\n"
               "    num_early_triggers: Number of early triggers (early_trigger_level = 1..num_early_triggers)\n"
               "    max_width: Maximum width of peak-finding kernel (in tree time samples)\n"
               "    dm_downsampling: DM downsampling factor relative to tree\n"
               "    time_downsampling: Time downsampling factor relative to tree\n"
               "    wt_dm_downsampling: DM downsampling factor for weights (>= dm_downsampling)\n"
               "    wt_time_downsampling: Time downsampling for weights (>= time_downsampling)")
          .def_readwrite("num_early_triggers", &DedispersionConfig::PrimaryTree::num_early_triggers,
               "Number of early triggers (early_trigger_level = 1..num_early_triggers, can be zero)")
          .def_readwrite("max_width", &DedispersionConfig::PrimaryTree::max_width,
               "Maximum width of peak-finding kernel (in tree time samples)")
          .def_readwrite("dm_downsampling", &DedispersionConfig::PrimaryTree::dm_downsampling,
               "DM downsampling factor of coarse-grained array relative to tree")
          .def_readwrite("time_downsampling", &DedispersionConfig::PrimaryTree::time_downsampling,
               "Time downsampling factor of coarse-grained array relative to tree")
          .def_readwrite("wt_dm_downsampling", &DedispersionConfig::PrimaryTree::wt_dm_downsampling,
               "DM downsampling factor of weights array (must be >= dm_downsampling)")
          .def_readwrite("wt_time_downsampling", &DedispersionConfig::PrimaryTree::wt_time_downsampling,
               "Time downsampling factor of weights array (must be >= time_downsampling)")
          .def("__repr__", [](const DedispersionConfig::PrimaryTree &self) {
               std::ostringstream os;
               os << "PrimaryTree(num_early_triggers=" << self.num_early_triggers
                  << ", max_width=" << self.max_width
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
    // for one choice of (primary tree, early trigger).
    py::class_<DedispersionTree>(m, "DedispersionTree")
          .def_readonly("primary_tree_index", &DedispersionTree::primary_tree_index)
          .def_readonly("early_trigger_level", &DedispersionTree::early_trigger_level)
          .def_readonly("amb_rank", &DedispersionTree::amb_rank)
          .def_readonly("dd_rank", &DedispersionTree::dd_rank)
          .def_readonly("nt_ds", &DedispersionTree::nt_ds)
          .def("total_rank", &DedispersionTree::total_rank,
               "Total tree rank (amb_rank + dd_rank).")
          .def_readonly("frequency_subbands", &DedispersionTree::frequency_subbands)
          .def_readonly("pf", &DedispersionTree::pf)
          .def_readonly("Dcore", &DedispersionTree::Dcore,
               "Internal time-downsampling of this tree's peak-finding kernel (sets out_argmax\n"
               "token granularity). From the cdd2 kernel registry; equals pf.time_downsampling\n"
               "if the kernel is not compiled into this build.")
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
        "See configs/xengine_metadata.yml for an example with per-field documentation.")
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
               "(the wire-protocol case, or a FakeXEngine Worker).\n\nOtherwise\n"
               "(bookkeeping contexts where no specific sender is distinguished,\n"
               "e.g. the allocator's canonical copy, FrbServer, AssembledFrame's\n"
               "metadata) the convention is to set this to an empty list, which\n"
               "we call 'frequency-scrubbed'.")
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
          .def("get_channel_freq_edges", &XEngineMetadata::get_channel_freq_edges,
               "Per-channel frequency edges (length get_total_nfreq()+1), expanded\n"
               "from the zone structure (equal channel width within each zone)")
          .def("get_channel_variances", &XEngineMetadata::get_channel_variances,
               "Per-channel noise variance (length get_total_nfreq()):\n"
               "noise_variance[z] broadcast across zone z's channels")
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
    py::class_<FakeXEngine, std::shared_ptr<FakeXEngine>>(m, "FakeXEngine",
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
               "The sequentiality check fires synchronously at submit time,\n"
               "so a bad index raises RuntimeError to this caller. Per the\n"
               "strict stoppable-class policy, the error also stops the\n"
               "FakeXEngine (see notes/stoppable_class.md).\n\n"
               "Wire effect: the worker sends one minichunk worth of all-zero\n"
               "data. The protocol handshake is sent ahead of the first\n"
               "SEND_JUNK or SEND_MINICHUNK on this worker.\n\n"
               "Returns:\n"
               "    True if the command was enqueued; False (command NOT\n"
               "    enqueued) if the FakeXEngine was cleanly stopped -- the\n"
               "    caller's signal to wind down.\n\n"
               "Raises:\n"
               "    RuntimeError: Re-raises the saved root-cause error if the\n"
               "    FakeXEngine was stopped by an error. Also raised if\n"
               "    arguments are out of range, or minichunk_index is not the\n"
               "    +1 successor of the most recently enqueued state-advancing\n"
               "    command on this worker (with the first-command exemption).")
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
               "Returns:\n"
               "    True if the command was enqueued; False (command NOT\n"
               "    enqueued) if the FakeXEngine was cleanly stopped.\n\n"
               "Raises:\n"
               "    RuntimeError: Re-raises the saved root-cause error if the\n"
               "    FakeXEngine was stopped by an error. Also raised if\n"
               "    arguments are out of range, or minichunk_index violates\n"
               "    the +1 chain.")
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
               "Returns:\n"
               "    True if the command was enqueued; False (command NOT\n"
               "    enqueued) if the FakeXEngine was cleanly stopped.\n\n"
               "Raises:\n"
               "    RuntimeError: Re-raises the saved root-cause error if the\n"
               "    FakeXEngine was stopped by an error. Also raised if\n"
               "    arguments are out of range, frame_set is None, or\n"
               "    minichunk_index violates the +1 chain.")
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
               "Returns:\n"
               "    True if the command was enqueued; False (command NOT\n"
               "    enqueued) if the FakeXEngine was cleanly stopped.\n\n"
               "Raises:\n"
               "    RuntimeError: Re-raises the saved root-cause error if the\n"
               "    FakeXEngine was stopped by an error. Also raised if\n"
               "    worker_id is out of range.")
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
               "Returns True immediately if minichunk_index is negative (since\n"
               "per-worker last_processed_minichunk starts at -1, this lets the\n"
               "controller call wait_until_processed(w, n-2) unconditionally for\n"
               "n in {0, 1}).\n\n"
               "Returns:\n"
               "    True once the condition is reached; False if the FakeXEngine\n"
               "    was cleanly stopped first -- the caller's signal to wind\n"
               "    down.\n\n"
               "Raises:\n"
               "    RuntimeError: Re-raises the saved root-cause error if the\n"
               "    FakeXEngine was stopped by an error. Also raised if\n"
               "    worker_id is out of range.")
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
               "Returns:\n"
               "    True if the command was enqueued; False (command NOT\n"
               "    enqueued) if the FakeXEngine was cleanly stopped.\n\n"
               "Raises:\n"
               "    RuntimeError: Re-raises the saved root-cause error if the\n"
               "    FakeXEngine was stopped by an error. Also raised if the\n"
               "    FakeXEngine was not constructed with debug=True (no acks\n"
               "    to wait for), or if worker_id is out of range.")
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
               "Returns:\n"
               "    True once the queue (and, in debug mode, the ack queue) is\n"
               "    fully drained; False if the FakeXEngine was cleanly stopped\n"
               "    first.\n\n"
               "Raises:\n"
               "    RuntimeError: Re-raises the saved root-cause error if the\n"
               "    FakeXEngine was stopped by an error. Also raised if\n"
               "    worker_id is out of range.")
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
               "Signal worker threads to stop. This is a clean stop (normal\n"
               "termination): any in-flight or later enqueue_* /\n"
               "wait_until_processed / synchronize calls return False rather\n"
               "than raising. Safe to call multiple times.")
          .def_property_readonly("is_stopped",
               [](FakeXEngine &self) {
                   // O(1) atomic load -- the property's "true" value just
                   // means stop() has been called; worker threads may
                   // still be exiting. Wait for the destructor (or a
                   // join) if you need to know they're fully done.
                   return self.is_stopped_cache.load(std::memory_order_acquire);
               },
               "True if stop() has been called -- explicitly (clean stop), or\n"
               "internally on an error (e.g. the receiver closed the connection).\n"
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
    // Skipped members: mutex, cv, is_started, is_stopped, error, listener_thread,
    //   reader_thread, assembler_thread, reader_peer_queue, assembler_peer_queue (internal)
    // Skipped methods: _listener_main, _reader_main, listener_main, reader_main (private)
    // Note: Uses shared_ptr holder so Receivers can be passed to FrbServer.
    py::class_<Receiver, std::shared_ptr<Receiver>>(m, "Receiver",
        "Listens for TCP connections and reads data.\n\n"
        "A thread-backed class with three worker threads:\n"
        "  - listener: accepts incoming connections\n"
        "  - reader: reads data from all open connections using epoll\n"
        "  - assembler: copies minichunks from per-peer ring buffers into\n"
        "    AssembledFrames")
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
          .def("wait_until_listening", &Receiver::wait_until_listening,
               py::arg("timeout_sec") = -1.0,
               py::call_guard<py::gil_scoped_release>(),
               "Wait until the listener thread has bound the listening socket\n"
               "(i.e. a client's connect() will succeed); returns True once it is.\n"
               "If timeout_sec >= 0, give up after that many seconds and return\n"
               "False; a negative timeout (default) waits forever. The GIL is\n"
               "released for the duration of the call -- pass a finite timeout and\n"
               "poll to stay responsive to signals / detect a dead peer. Useful\n"
               "when the caller must not attempt a connection before the Receiver\n"
               "is accepting (e.g. FakeXEngine, whose lazy connect raises on\n"
               "ECONNREFUSED). Requires start() to have been called.\n\n"
               "Raises:\n"
               "    RuntimeError: If the Receiver is stopped.")
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

    // FrbGrouperClient: gRPC *client* side of the FrbGrouper service -- the
    // producer (FrbServer) end. Constructed in run_server, ping()'d early to fail
    // fast if the grouper isn't up, then passed into the FrbServer constructor.
    // shared_ptr holder so it can be handed to FrbServer (like Receiver/FileWriter).
    // Registered BEFORE FrbServer so FrbServer's grouper_client=None default arg
    // can resolve this type's holder.
    py::class_<FrbGrouperClient, std::shared_ptr<FrbGrouperClient>>(m, "FrbGrouperClient",
        "Producer-side client for an FrbGrouper (frb_grouper.proto). Construct with the\n"
        "grouper's 'ip:port', call ping() early to fail fast if it isn't running, then\n"
        "pass it to FrbServer(grouper_client=...). The FrbServer opens the real\n"
        "connection + Handshake later, from its grouper send thread.")
          .def(py::init([](const std::string &grouper_ip_addr) {
                   return std::make_shared<FrbGrouperClient>(grouper_ip_addr);
               }),
               py::arg("grouper_ip_addr"))
          .def("ping", &FrbGrouperClient::ping,
               py::arg("timeout_ms") = constants::grouper_ping_timeout_ms,
               py::call_guard<py::gil_scoped_release>(),   // blocks up to timeout_ms when grouper is down
               "Channel-level connectivity check: bring a throwaway channel to READY, then\n"
               "drop it (no Session RPC, no Handshake). Raises RuntimeError if the grouper is\n"
               "not reachable within timeout_ms.")
          .def_readonly("grouper_ip_addr", &FrbGrouperClient::grouper_ip_addr,
               "The grouper's 'ip:port' (loopback).")
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
                           std::shared_ptr<FrbGrouperClient> grouper_client,
                           bool no_dedispersion,
                           long nbatches_wt,
                           bool quiet) {
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
               params.grouper_client = std::move(grouper_client);
               params.no_dedispersion = no_dedispersion;
               params.nbatches_wt = nbatches_wt;
               params.quiet = quiet;
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
               py::arg("grouper_client") = std::shared_ptr<FrbGrouperClient>(),
               py::arg("no_dedispersion") = false,
               py::arg("nbatches_wt") = 0,
               py::arg("quiet") = false,
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
               "    rpc_server_address: gRPC server address (e.g. '127.0.0.1:6000')\n"
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
               "    grouper_client (default None): an FrbGrouperClient (already\n"
               "        pinged) for the FrbGrouper to feed (the FrbServer is the\n"
               "        gRPC client). None => disabled (GpuDedisperser built with\n"
               "        num_consumers=0). Its address must be loopback (CUDA IPC\n"
               "        requires the same node/GPU).\n"
               "    no_dedispersion (default False): if True, the processing thread\n"
               "        skips ALL GPU work -- data is not even copied host->device,\n"
               "        and no dequantization/dedispersion kernels run. The\n"
               "        receive/assemble/ringbuf/reaper pipeline still runs in full.\n"
               "        Implies no grouper, so grouper_client must be None (the\n"
               "        constructor asserts this).\n"
               "    nbatches_wt (default 0): weight-ring depth of the internal\n"
               "        GpuDedisperser. 0 = num_active_batches. If nonzero, must be\n"
               "        >= num_active_batches. Only used by unit tests.\n"
               "    quiet (default False): if True, suppress the per-chunk\n"
               "        'FrbServer: beamset=...' stdout line (one per assembled\n"
               "        chunk). Everything else is unaffected.")
          .def("start", &FrbServer::start,
               "Start all Receivers.\n\n"
               "Raises:\n"
               "    RuntimeError: If called twice or after stop().")
          .def("stop", [](FrbServer &self) { self.stop(); },
               py::call_guard<py::gil_scoped_release>(),   // stop() blocks: grpc Shutdown() waits for in-flight
                                                           // handlers, plus downstream stop() cascades. (It does
                                                           // NOT join the backing threads -- the destructor does.)
               "Stop the server and all Receivers. Safe to call multiple times.")
          .def("poll_from_python", &FrbServer::poll_from_python, py::arg("timeout_ms"),
               py::call_guard<py::gil_scoped_release>(),
               "Block until the server is stopped, or until ``timeout_ms`` elapses.\n\n"
               "Releases the GIL while blocked. Call in a loop with a short timeout\n"
               "(e.g. constants.default_poll_cadence_ms) so that Ctrl-C stays\n"
               "responsive: KeyboardInterrupt is raised between calls, never during one.\n\n"
               "Args:\n"
               "    timeout_ms: Maximum time to block, in milliseconds (must be >= 0).\n\n"
               "Returns:\n"
               "    True if the server is stopped (a clean stop); False if the timeout\n"
               "    expired first.\n\n"
               "Raises:\n"
               "    Exception: If the server stopped due to an internal error, the\n"
               "        saved root-cause exception is re-raised here.")
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
             "a receiver/reaper/processing thread threw an exception. To retrieve the\n"
             "error itself, call poll_from_python(), which re-raises the saved\n"
             "exception (nothing is printed to stderr).")
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
    // py::dynamic_attr() lets the Python injection attach derived attributes (the
    // parsed-yaml dicts, steady_state_it0, full_steady_ichunk, etc.).
    // Injections (pirate_frb/rpc/FrbGrouper.py) add __enter__/__exit__ and
    // a get_output() context manager.
    // The class docstring (including the read-only-attribute bullet list) lives in
    // the Python injector pirate_frb/rpc/FrbGrouper.py, NOT here: FrbGrouper's
    // primary interface is the context-manager / get_output() usage defined there,
    // and ksgpu.inject_methods copies the injector's docstring onto this class
    // (overriding any docstring set here). Per-class policy: notes/docstrings.md.
    py::class_<FrbGrouper, std::shared_ptr<FrbGrouper>>(m, "FrbGrouper", py::dynamic_attr())
          .def(py::init([](const std::string &ip_addr){ return FrbGrouper::create(ip_addr); }),
               py::arg("ip_addr"))
          // open() must stay interruptible by Ctrl-C while it blocks waiting for
          // a client. We can't just release the GIL for the whole call (Python
          // signal handlers wouldn't run); instead we drive the wait in
          // constants::default_poll_cadence_ms steps, releasing the GIL during each
          // wait and reacquiring it between steps to poll for pending Python signals
          // (PyErr_CheckSignals() != 0 => the SIGINT handler raised KeyboardInterrupt,
          // propagated via error_already_set).
          .def("_open", [](FrbGrouper &self) {
               self.start_listening();                 // GIL held; quick (bind + spawn)
               for (;;) {
                   bool ready;
                   { py::gil_scoped_release nogil; ready = self.wait_for_handshake(constants::default_poll_cadence_ms); }
                   if (ready)
                       break;
                   if (PyErr_CheckSignals() != 0)       // e.g. Ctrl-C
                       throw py::error_already_set();
               }
          }, "Start listening + block until a client connects and its handshake\n"
             "is processed. Interruptible by Ctrl-C.")
          .def("_close", &FrbGrouper::close, py::call_guard<py::gil_scoped_release>(),
               "Stop the session + join the send thread + shut down the gRPC server.")
          .def("_stop", [](FrbGrouper &self){ self.stop(); },
               "Put the FrbGrouper into the stopped state (idempotent).")
          .def("_acquire_output", &FrbGrouper::acquire_output,
               py::arg("seq_id"), py::call_guard<py::gil_scoped_release>(),
               "Block until produced_seq_id has been received for 'seq_id'; return\n"
               "a per-batch slice (nbeams == beams_per_batch) of output_ringbuf.")
          .def("_release_output", &FrbGrouper::release_output, py::arg("seq_id"),
               "Record that the caller is done with 'seq_id' (emits CONSUMED).")
          // Vectorized decode of out_argmax tokens, forwarding to the grouper's internal
          // "incomplete" DedispersionPlan (deserialized at handshake). The plan's per-tree
          // Dcore values come from the PRODUCER, so tokens decode correctly even if this
          // process runs a different pirate_frb build. Valid only after the handshake.
          .def("decode_argmax_batch",
               [](const FrbGrouper &self, const Array<uint> &tokens, const Array<long> &itrees,
                  const Array<long> &idms, const Array<long> &itimes) {
                   xassert(self.incomplete_plan);   // populated at handshake
                   return _decode_argmax_batch(*self.incomplete_plan, tokens, itrees, idms, itimes);
               },
               py::arg("tokens"), py::arg("itrees"), py::arg("idms"), py::arg("itimes"),
               "Vectorized decode of out_argmax tokens (see DedispersionPlan.decode_argmax\n"
               "for the scalar spec). Inputs are 1-d nonempty arrays, one event per element\n"
               "(tokens: uint32; itrees/idms/itimes: int64). Returns TOPLEVEL-relative\n"
               "(fmins, fmaxs, tlos, this, ps), each an int64 array. Uses the producer's\n"
               "plan from the handshake; valid only after the handshake.")
          .def("decode_argmax2_batch",
               [](const FrbGrouper &self, const Array<long> &itrees, const Array<long> &fmins,
                  const Array<long> &fmaxs, const Array<long> &tlos, const Array<long> &this_,
                  const Array<long> &ps) {
                   xassert(self.incomplete_plan);   // populated at handshake
                   return _decode_argmax2_batch(*self.incomplete_plan, itrees, fmins, fmaxs,
                                                tlos, this_, ps);
               },
               py::arg("itrees"), py::arg("fmins"), py::arg("fmaxs"),
               py::arg("tlos"), py::arg("this"), py::arg("ps"),
               "Vectorized decode_argmax2(): converts decode_argmax_batch() outputs to\n"
               "physical params. Returns (freqs_lo_MHz, freqs_hi_MHz, dms, timestamps_samp,\n"
               "widths_samp), each a float64 array. Timestamps are CHUNK-RELATIVE toplevel\n"
               "sample counts (extrapolated to the full-band lowest frequency); the caller\n"
               "converts to absolute FPGA counts. Valid only after the handshake.")
          .def("_compute_steady_state_it0", &FrbGrouper::_compute_steady_state_it0,
               py::arg("itree"),
               "Forwards DedispersionPlan.compute_steady_state_it0(), using the producer's\n"
               "plan from the handshake (see the 'steady_state_it0' bullet in the FrbGrouper\n"
               "class docstring). Valid only after the handshake.")
          // Member docstrings are intentionally omitted here: each member is documented
          // in the bullet list in the class docstring, which lives in the Python injector
          // (pirate_frb/rpc/FrbGrouper.py). Kept as a plain list, not a napoleon
          // "Attributes" section, so the rendering is compact and the members are not
          // re-registered as separate sphinx objects / sidebar entries.
          //
          // Synchronization convention for the raw def_readonly members below:
          // they are written by the Session handler thread WITHOUT the mutex,
          // and published via the mutexed handshake_done flag. They are safe
          // to read only after wait_for_handshake() has returned true on the
          // reading thread (the documented flow); reading them from a second
          // Python thread while another is still blocked in open() would race
          // with the handler's writes.
          .def_property_readonly("is_stopped", &FrbGrouper::is_stopped_pub)
          .def_readonly("cuda_device_id", &FrbGrouper::cuda_device_id)
          .def_readonly("dtype", &FrbGrouper::dtype)
          .def_readonly("nt_in", &FrbGrouper::nt_in)
          .def_readonly("total_beams", &FrbGrouper::total_beams)
          .def_readonly("beams_per_batch", &FrbGrouper::beams_per_batch)
          .def_readonly("nbatches", &FrbGrouper::nbatches)
          .def_readonly("num_batch_slots", &FrbGrouper::num_batch_slots)
          .def_readonly("initial_chunk", &FrbGrouper::initial_chunk)
          .def_readonly("ntrees", &FrbGrouper::ntrees)
          .def_readonly("ndm_out", &FrbGrouper::ndm_out)
          .def_readonly("nt_out", &FrbGrouper::nt_out)
          .def_readonly("dedispersion_config", &FrbGrouper::dedispersion_config)
          .def_readonly("xengine_metadata", &FrbGrouper::xengine_metadata)
          .def_readonly("xengine_metadata_yaml_string", &FrbGrouper::xengine_metadata_yaml_string)
          .def_readonly("dedispersion_config_yaml_string", &FrbGrouper::dedispersion_config_yaml_string)
          .def_readonly("dedispersion_plan_yaml_string", &FrbGrouper::dedispersion_plan_yaml_string)
          .def_readonly("grouper_ip_addr", &FrbGrouper::grouper_ip_addr)
          .def_readonly("search_ip_addr", &FrbGrouper::search_ip_addr)
          // NOTE: FrbGrouper::dedispersion_plan_yaml (YAML::Node) is intentionally
          // NOT wrapped; the injection adds a Python dedispersion_plan_yaml attribute
          // parsed from dedispersion_plan_yaml_string. output_ringbuf is private
          // (reached only via _acquire_output).
    ;

    // FileWriter: writes AssembledFrames to disk via SSD and NFS queues.
    // Skipped members: Params, WriteStatus, RpcSubscriber, process_frame, add_subscriber (internal)
    py::class_<FileWriter, std::shared_ptr<FileWriter>>(m, "FileWriter",
        "Writes AssembledFrames to disk via SSD and NFS queues.\n\n"
        "Creates worker threads for writing to local SSD and copying to NFS.")
          .def(py::init([](const std::string &ssd_root, const std::string &nfs_root,
                           int num_ssd_threads, int num_nfs_threads,
                           long max_subscriber_backlog) {
               FileWriter::Params params;
               params.ssd_root = ssd_root;
               params.nfs_root = nfs_root;
               params.num_ssd_threads = num_ssd_threads;
               params.num_nfs_threads = num_nfs_threads;
               params.max_subscriber_backlog = max_subscriber_backlog;
               return std::make_shared<FileWriter>(params);
          }),
               py::arg("ssd_root"), py::arg("nfs_root"),
               py::arg("num_ssd_threads") = 4, py::arg("num_nfs_threads") = 2,
               py::arg("max_subscriber_backlog") = constants::max_file_subscriber_backlog,
               "Create a FileWriter.\n\n"
               "Args:\n"
               "    ssd_root: Absolute path to local SSD directory\n"
               "    nfs_root: Absolute path to NFS directory\n"
               "    num_ssd_threads: Number of threads for SSD writes (default 4)\n"
               "    num_nfs_threads: Number of threads for NFS copies (default 2)\n"
               "    max_subscriber_backlog: Max queued-but-unsent notifications per\n"
               "        SubscribeFiles subscriber; a subscriber that falls this far\n"
               "        behind is stopped with a 'fell behind' error (mainly\n"
               "        overridden by tests)")
          .def("stop", [](FileWriter &self) { self.stop(); },
               "Stop the writer. Safe to call multiple times.")
    ;
    
    // HwtestSender: simulates a correlator sending data over TCP.
    // Skipped members: mutex, cv, is_stopped, is_started, error, workers, endpoints (internal state)
    // Skipped methods: _throw_if_stopped, worker_main, _worker_main, _send_all (private)
    py::class_<HwtestSender, std::shared_ptr<HwtestSender>>(m, "HwtestSender")
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

