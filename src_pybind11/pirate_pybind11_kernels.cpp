// Python bindings for GPU kernel classes (pirate_frb.kernels subpackage).
// See pirate_pybind11.cu for the main module definition.

#define PY_ARRAY_UNIQUE_SYMBOL PyArray_API_pirate
#define NO_IMPORT_ARRAY  // Secondary file: don't call _import_array()
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <pybind11/stl.h>
#include <ksgpu/pybind11.hpp>

#include "../include/pirate/BumpAllocator.hpp"
#include "../include/pirate/CoalescedDdKernel2.hpp"
#include "../include/pirate/DedispersionKernel.hpp"
#include "../include/pirate/Detrender1d.hpp"
#include "../include/pirate/Detrender2d.hpp"
#include "../include/pirate/GpuDequantizationKernel.hpp"
#include "../include/pirate/LaggedDownsamplingKernel.hpp"
#include "../include/pirate/MegaRingbuf.hpp"
#include "../include/pirate/PeakFindingKernel.hpp"
#include "../include/pirate/ReferenceLagbuf.hpp"
#include "../include/pirate/ReferenceTree.hpp"
#include "../include/pirate/RingbufCopyKernel.hpp"
#include "../include/pirate/TreeGriddingKernel.hpp"

using namespace std;
using namespace ksgpu;
using namespace pirate;
namespace py = pybind11;


namespace pirate {

void register_kernel_bindings(pybind11::module &m)
{
    py::class_<CoalescedDdKernel2>(m, "CoalescedDdKernel2")
          .def_static("test_random", &CoalescedDdKernel2::test_random, py::call_guard<py::gil_scoped_release>())
          .def_static("time_selected", &CoalescedDdKernel2::time_selected, py::call_guard<py::gil_scoped_release>())
          .def_static("registry_size", &CoalescedDdKernel2::registry_size)
          .def_static("show_registry", &CoalescedDdKernel2::show_registry)
    ;

    // MegaRingbuf is bound as an opaque handle: it is reachable only via a DedispersionPlan
    // (or a DedispersionKernelParams), and python callers pass it around rather than inspect
    // it. Only the few members a caller needs in order to size the ring buffer are exposed.
    py::class_<MegaRingbuf, std::shared_ptr<MegaRingbuf>>(m, "MegaRingbuf",
        "The ring buffer through which stage-1 dedispersion feeds stage 2.\n\n"
        "Not constructible from python: obtain one from DedispersionPlan.mega_ringbuf. See\n"
        "MegaRingbuf.hpp for how it works.")
          .def_readonly("gpu_global_nseg", &MegaRingbuf::gpu_global_nseg,
               "Segments in the GPU ring buffer, per beam. The buffer a kernel expects is a\n"
               "1-d array of (gpu_global_nseg * nt_per_segment) elements.")
          .def_readonly("host_global_nseg", &MegaRingbuf::host_global_nseg,
               "Segments in the HOST ring buffer, per beam. Zero unless some consumer's\n"
               "chunk_lag exceeds the previous consumer's by more than max_gpu_clag, which\n"
               "never happens at the default max_gpu_clag = 10000.")
          .def_readonly("max_clag", &MegaRingbuf::max_clag,
               "Largest lag, in chunks, of any zone")
          .def_readonly("num_producers", &MegaRingbuf::num_producers)
          .def_readonly("num_consumers", &MegaRingbuf::num_consumers)
          .def_readonly("is_finalized", &MegaRingbuf::is_finalized)
    ;

    // The kernel-parameter structs below are bound READ-ONLY, with no constructor: a caller
    // gets them from a DedispersionPlan (plan.stage1_dd_kernel_params etc.) and passes them
    // straight to a kernel constructor. They encode the ring-buffer lag structure, which must
    // not be re-derived on the python side.
    py::class_<DedispersionKernelParams>(m, "DedispersionKernelParams",
        "Construction parameters for a (Reference|Gpu|GpuSb)DedispersionKernel.\n\n"
        "Not constructible from python: obtain one from DedispersionPlan.stage1_dd_kernel_params\n"
        "or .stage2_dd_kernel_params.")
          .def_readonly("dtype", &DedispersionKernelParams::dtype)
          .def_readonly("dd_rank", &DedispersionKernelParams::dd_rank,
               "log2(number of 'active' tree channels)")
          .def_readonly("amb_rank", &DedispersionKernelParams::amb_rank,
               "log2(number of 'spectator' tree channels)")
          .def_readonly("total_beams", &DedispersionKernelParams::total_beams)
          .def_readonly("beams_per_batch", &DedispersionKernelParams::beams_per_batch)
          .def_readonly("ntime", &DedispersionKernelParams::ntime,
               "Time samples per chunk (includes the tree's time downsampling, if any)")
          .def_readonly("nspec", &DedispersionKernelParams::nspec, "'Inner' spectator index")
          .def_readonly("input_is_ringbuf", &DedispersionKernelParams::input_is_ringbuf)
          .def_readonly("output_is_ringbuf", &DedispersionKernelParams::output_is_ringbuf)
          .def_readonly("apply_input_residual_lags", &DedispersionKernelParams::apply_input_residual_lags)
          .def_readonly("input_is_downsampled_tree", &DedispersionKernelParams::input_is_downsampled_tree)
          .def_readonly("nt_per_segment", &DedispersionKernelParams::nt_per_segment)
          .def_readonly("mega_ringbuf", &DedispersionKernelParams::mega_ringbuf,
               "The MegaRingbuf, if either input_is_ringbuf or output_is_ringbuf")
          .def_readonly("producer_id", &DedispersionKernelParams::producer_id)
          .def_readonly("consumer_id", &DedispersionKernelParams::consumer_id)
          .def("validate", &DedispersionKernelParams::validate,
               "Raises RuntimeError if any parameter is invalid.")
    ;

    py::class_<TreeGriddingKernelParams>(m, "TreeGriddingKernelParams",
        "Construction parameters for a (Reference|Gpu)TreeGriddingKernel.\n\n"
        "Not constructible from python: obtain one from\n"
        "DedispersionPlan.tree_gridding_kernel_params.")
          .def_readonly("dtype", &TreeGriddingKernelParams::dtype)
          .def_readonly("nfreq", &TreeGriddingKernelParams::nfreq, "Input frequency channels")
          .def_readonly("nchan", &TreeGriddingKernelParams::nchan, "Output tree channels")
          .def_readonly("ntime", &TreeGriddingKernelParams::ntime, "Time samples per chunk")
          .def_readonly("beams_per_batch", &TreeGriddingKernelParams::beams_per_batch)
          .def_readonly("channel_map", &TreeGriddingKernelParams::channel_map,
               "Length (nchan+1) host array of frequency-channel edges, monotonically decreasing")
    ;

    py::class_<GpuSbDedispersionKernel>(m, "GpuSbDedispersionKernel",
        "Stage-2 dedispersion with frequency subbands: the GPU counterpart of\n"
        "ReferenceDedispersionKernel's 'sb_out' output, which the production dedisperser\n"
        "(CoalescedDdKernel2) never materializes.\n\n"
        "Constructed from a plan's stage-2 kernel params and the tree's subbands::\n\n"
        "    k = GpuSbDedispersionKernel(plan.stage2_dd_kernel_params[itree],\n"
        "                                plan.trees[itree].frequency_subbands)\n"
        "    k.allocate(bump_allocator)\n"
        "    k.launch(sb_out, ringbuf, ichunk, ibatch)\n\n"
        "float32 only. The constructor raises RuntimeError if no kernel is compiled for this\n"
        "(dd_rank, subband_counts) pair; the message names the missing key, which can be added\n"
        "to autogenerated_sbdd_kernels() in makefile_helper.py.")
          .def(py::init<const DedispersionKernelParams &, const FrequencySubbands &>(),
               py::arg("dd_params"), py::arg("frequency_subbands"),
               py::call_guard<py::gil_scoped_release>())
          .def("allocate", &GpuSbDedispersionKernel::allocate, py::arg("allocator"),
               py::call_guard<py::gil_scoped_release>(),
               "Allocate (and zero) persistent state from a BumpAllocator. Must be called\n"
               "before launch().")
          .def("launch",
               [](GpuSbDedispersionKernel &self, Array<float> &sb_out, const Array<float> &in,
                  long ichunk, long ibatch, uintptr_t stream_ptr) {
                   self.launch(sb_out, in, ichunk, ibatch,
                               reinterpret_cast<cudaStream_t> (stream_ptr));
               },
               py::arg("sb_out"), py::arg("in_"), py::arg("ichunk"), py::arg("ibatch"),
               py::arg("stream_ptr"),
               py::call_guard<py::gil_scoped_release>(),   // async launch; body is pure C++
               "GPU kernel launch (async, does not sync stream).\n\n"
               "Args:\n"
               "    sb_out: Array, shape (beams_per_batch, Dpf, fs.M, ntime), float32, fully\n"
               "        contiguous, on GPU. The kernel derives all its strides from (M, ntime),\n"
               "        so a padded or sliced buffer is rejected.\n"
               "    in_: the ring buffer, a 1-d float32 GPU array of length\n"
               "        (mega_ringbuf.gpu_global_nseg * nt_per_segment)\n"
               "    ichunk: time-chunk index 0, 1, ...\n"
               "    ibatch: 0 <= ibatch < nbatches\n"
               "    stream_ptr: CUDA stream pointer (integer, e.g. from cupy stream.ptr)")
          .def_readonly("dd_params", &GpuSbDedispersionKernel::dd_params)
          .def_readonly("fs", &GpuSbDedispersionKernel::fs)
          .def_readonly("Dpf", &GpuSbDedispersionKernel::Dpf,
               "= 2^(amb_rank + dd_rank - fs.pf_rank), the 'sb_out' DM axis")
          .def_readonly("nbatches", &GpuSbDedispersionKernel::nbatches)
          .def_readonly("is_allocated", &GpuSbDedispersionKernel::is_allocated)
          .def_readonly("resource_tracker", &GpuSbDedispersionKernel::resource_tracker)
          .def_static("test_random", &GpuSbDedispersionKernel::test_random, py::call_guard<py::gil_scoped_release>())
          .def_static("registry_size", &GpuSbDedispersionKernel::registry_size)
          .def_static("show_registry", &GpuSbDedispersionKernel::show_registry)
    ;

    py::class_<GpuDedispersionKernel>(m, "GpuDedispersionKernel",
        "One stage of GPU tree dedispersion; inputs and outputs are plain buffers or ring\n"
        "buffers according to the params.\n\n"
        "Constructed from a plan's kernel params::\n\n"
        "    k = GpuDedispersionKernel(plan.stage1_dd_kernel_params[ipri])\n"
        "    k.allocate(bump_allocator)\n"
        "    k.launch(in_, out, ichunk, ibatch)\n\n"
        "Unlike GpuSbDedispersionKernel, this kernel has no 'sb_out' (subband) output.")
          .def(py::init<const DedispersionKernelParams &>(), py::arg("params"),
               py::call_guard<py::gil_scoped_release>())
          .def("allocate", &GpuDedispersionKernel::allocate, py::arg("allocator"),
               py::call_guard<py::gil_scoped_release>(),
               "Allocate (and zero) persistent state from a BumpAllocator. Must be called\n"
               "before launch().")
          .def("launch",
               [](GpuDedispersionKernel &self, Array<void> &in_, Array<void> &out,
                  long ichunk, long ibatch, uintptr_t stream_ptr) {
                   self.launch(in_, out, ichunk, ibatch,
                               reinterpret_cast<cudaStream_t> (stream_ptr));
               },
               py::arg("in_"), py::arg("out"), py::arg("ichunk"), py::arg("ibatch"),
               py::arg("stream_ptr"),
               py::call_guard<py::gil_scoped_release>(),   // async launch; body is pure C++
               "GPU kernel launch (async, does not sync stream).\n\n"
               "Args:\n"
               "    in_, out: on-GPU arrays with the kernel's dtype. A 'simple' buffer has\n"
               "        shape (beams_per_batch, 2^amb_rank, 2^dd_rank, ntime) -- or one more\n"
               "        axis of length nspec, if nspec > 1. A ring buffer is 1-d of length\n"
               "        (mega_ringbuf.gpu_global_nseg * nt_per_segment * nspec). Which one\n"
               "        each is comes from params.{input,output}_is_ringbuf.\n"
               "    ichunk: time-chunk index 0, 1, ...\n"
               "    ibatch: 0 <= ibatch < nbatches\n"
               "    stream_ptr: CUDA stream pointer (integer, e.g. from cupy stream.ptr)")
          .def_readonly("params", &GpuDedispersionKernel::params)
          .def_readonly("nbatches", &GpuDedispersionKernel::nbatches)
          .def_readonly("is_allocated", &GpuDedispersionKernel::is_allocated)
          .def_readonly("resource_tracker", &GpuDedispersionKernel::resource_tracker)
          .def_static("test_random", &GpuDedispersionKernel::test_random, py::call_guard<py::gil_scoped_release>())
          .def_static("time_selected", &GpuDedispersionKernel::time_selected, py::call_guard<py::gil_scoped_release>())
          .def_static("registry_size", &GpuDedispersionKernel::registry_size)
          .def_static("show_registry", &GpuDedispersionKernel::show_registry)
    ;

    // Detrender1d: Python injections in pirate_frb/kernels/Detrender1d.py:
    //   - launch: converts stream=None to current cupy stream
    py::class_<Detrender1d> detrender_1d(m, "Detrender1d",
        "The 1-d time detrender: a masked, adaptively centered moving local polynomial fit.\n\n"
        "For each output sample t, a degree-n polynomial is fit to the valid samples of the\n"
        "window [t-W, t+W] and evaluated back at t; the fit is subtracted from the data, and\n"
        "the sample is dropped ('mask expansion') if its window is too ill-conditioned to\n"
        "determine the fit. Operates in place on a (data, mask) pair, independently for each\n"
        "row (one row per (beam, freq) pair).\n\n"
        "Only the middle T samples of each row are written, i.e. buffer samples [W, W+T).\n"
        "The 2W padding samples are read but not written, and the caller is responsible for\n"
        "the buffer shift between chunks. Where the expanded mask is false, the residual is\n"
        "written as zero.\n\n"
        "(n, W, T) are compile-time parameters of the cuda kernel, so only the configurations\n"
        "listed in the constructor's error message exist; the number of rows M is runtime.\n\n"
        "The algorithm is specified in notes/detrending.tex, section 'Time detrending\n"
        "algorithm 1: local polynomial subtraction'. pirate_frb.detrending_1d is the\n"
        "pure-numpy reference that this kernel is validated against.");

    detrender_1d.attr("eps") = Detrender1d::eps;

    detrender_1d
          .def(py::init<long, long, long>(),
               py::arg("n"), py::arg("W"), py::arg("T") = 2048,
               "Create a Detrender1d.\n\n"
               "Args:\n"
               "    n: polynomial degree\n"
               "    W: window half-width (the window is 2W+1 samples)\n"
               "    T: output samples per row (chunk size)\n\n"
               "Raises:\n"
               "    RuntimeError: if no kernel is compiled for (n, W, T). The message lists\n"
               "        the available configurations.")
          .def_readonly("n", &Detrender1d::n, "Polynomial degree")
          .def_readonly("W", &Detrender1d::W, "Window half-width (the window is 2W+1 samples)")
          .def_readonly("T", &Detrender1d::T, "Output samples per row (chunk size)")
          .def_readonly("nbuf", &Detrender1d::nbuf, "Buffer samples per row, = T + 2W")
          .def_static("configs", &Detrender1d::configs,
               "The compiled (n, W, T) configurations, i.e. the arguments the constructor\n"
               "accepts. Returned as a list of (n, W, T) tuples.")
          .def_static("time_selected", &Detrender1d::time_selected,
               py::call_guard<py::gil_scoped_release>(),
               "Run timing benchmarks, for every compiled configuration "
               "(called via 'python -m pirate_frb time --dt1d')")
          .def("launch",
               [](const Detrender1d &self, Array<float> &data, Array<unsigned char> &mask,
                  uintptr_t stream_ptr) {
                   cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
                   self.launch(data, mask, stream);
               },
               py::arg("data"), py::arg("mask"), py::arg("stream_ptr"),
               py::call_guard<py::gil_scoped_release>(),   // async launch; body is pure C++
               "GPU kernel launch (async, does not sync stream).\n\n"
               "Args:\n"
               "    data: Array, shape (M, nbuf), dtype float32, fully contiguous, on GPU.\n"
               "          Modified in place.\n"
               "    mask: Array, shape (M, nbuf), dtype uint8, fully contiguous, on GPU,\n"
               "          {0,1}-valued. Modified in place, and the output mask is the\n"
               "          authoritative one (it can only lose samples).\n"
               "    stream_ptr: CUDA stream pointer (integer, e.g. from cupy stream.ptr)")
    ;

    // Detrender2d::Params. Unlike FakeXEngine (whose Params is deliberately not bound),
    // this one is: callers that configure a detrender without constructing it -- e.g.
    // slow_avar.BruteForceVarianceMap, which needs the same parameters for both the GPU
    // kernel and the numpy reference -- need the struct as a value.
    py::class_<Detrender2d::Params>(m, "Detrender2dParams",
        "Construction parameters for a Detrender2d (which see for what they mean).")
          .def(py::init([](long nfreq, const std::vector<long> &knots, long M, long n_phi,
                           long n, long W, long T, double eta, double eps) {
              Detrender2d::Params params;
              params.nfreq = nfreq;
              params.knots = knots;
              params.M = M;
              params.n_phi = n_phi;
              params.n = n;
              params.W = W;
              params.T = T;
              params.eta = eta;
              params.eps = eps;
              return params;
          }),
          py::arg("nfreq"), py::arg("knots"), py::arg("M"),
          py::arg("n_phi") = 2, py::arg("n") = 2, py::arg("W") = 4, py::arg("T") = 2048,
          py::arg("eta") = 1.0e-3, py::arg("eps") = 3.0e-5)
          .def_readwrite("nfreq", &Detrender2d::Params::nfreq, "Number of frequency channels")
          .def_readwrite("knots", &Detrender2d::Params::knots,
               "Non-decreasing list of channel indices, running from 0 to nfreq, with the\n"
               "first and last values repeated exactly n_phi+1 times and no interior value\n"
               "repeated more than n_phi+1 times. An interior value repeated exactly n_phi+1\n"
               "times is a zone boundary.")
          .def_readwrite("M", &Detrender2d::Params::M, "Number of spectator (beam) rows")
          .def_readwrite("n_phi", &Detrender2d::Params::n_phi, "Spline degree in frequency")
          .def_readwrite("n", &Detrender2d::Params::n, "Degree of the time polynomial")
          .def_readwrite("W", &Detrender2d::Params::W,
               "Window half-width (the window is 2W+1 samples)")
          .def_readwrite("T", &Detrender2d::Params::T, "Output samples per row (chunk size)")
          .def_readwrite("eta", &Detrender2d::Params::eta, "Regularization strength")
          .def_readwrite("eps", &Detrender2d::Params::eps, "Mask-expansion threshold on r_min")
          .def("validate", &Detrender2d::Params::validate,
               "Raises RuntimeError if any parameter is invalid; see the Detrender2d\n"
               "constructor, which calls this.")
          .def_static("from_yaml",
               static_cast<Detrender2d::Params (*)(const std::string &)> (&Detrender2d::Params::from_yaml),
               py::arg("filename"),
               "Load Detrender2dParams from a YAML file.\n\n"
               "The yaml keys are spelled out rather than matching the member names:\n"
               "num_beams (M), spline_degree_freq (n_phi), poly_degree_time (n),\n"
               "time_halfwidth (W), time_samples_per_chunk (T),\n"
               "regularization_strength (eta), conditioning_threshold (eps), plus nfreq\n"
               "and knots. The last two tuning parameters are optional and default to\n"
               "1.0e-3 and 3.0e-5; every other key is required.\n\n"
               "Raises:\n"
               "    RuntimeError: on a missing/unknown key, or if the parameters are\n"
               "        invalid (validate() is called before returning).")
          .def_static("from_yaml_string", &Detrender2d::Params::from_yaml_string,
               py::arg("yaml_string"),
               "Load Detrender2dParams from a YAML string: the inverse of to_yaml_string().\n"
               "Same keys and same errors as from_yaml(); use this when the yaml travels as\n"
               "a string rather than a file, e.g. one embedded in a variance-map file.")
          .def("to_yaml_string", &Detrender2d::Params::to_yaml_string,
               py::arg("verbose") = false,
               "Convert to a YAML string. If 'verbose', include explanatory comments and\n"
               "the derived basis-function and zone counts.")
    ;

    // Detrender2d: Python injections in pirate_frb/kernels/Detrender2d.py:
    //   - launch: converts stream=None to current cupy stream
    py::class_<Detrender2d> detrender_2d(m, "Detrender2d",
        "The 2-d spline detrender: a regularized fit of a B-spline in frequency times a\n"
        "local polynomial in time, subtracted from the data.\n\n"
        "Constructed either from a Detrender2dParams, or from the same fields as kwargs::\n\n"
        "    det = Detrender2d(nfreq=4096, knots=knots, M=1, W=4, T=2048)\n\n"
        "For each output sample t, the baseline over a window of 2W+1 time samples is\n"
        "modelled as sum_{jq} alpha_jq phi_j(f) p_q(s), with {phi_j} the B-spline basis of\n"
        "the caller's knot vector and {p_q} orthonormal polynomials on the window. It is\n"
        "fitted by weighted least squares over the unmasked samples, with a first-difference\n"
        "regulator eta*D_1 on the frequency coefficients, and evaluated back at the window\n"
        "centre.\n\n"
        "Zones -- interior knots of multiplicity n_phi+1 -- decouple the fit exactly, and\n"
        "mask expansion is per zone: a zone whose conditioning statistic r_min falls below\n"
        "eps has all of its channels dropped for that time sample. There is no per-channel\n"
        "expansion.\n\n"
        "Operates in place on a (data, mask) pair of shape (M, nfreq, nbuf). Only the middle\n"
        "T samples of each row are written, i.e. buffer samples [W, W+T). The 2W padding\n"
        "samples are read but not written, and the caller is responsible for the buffer\n"
        "shift between chunks. Where the expanded mask is false, the residual is written as\n"
        "zero.\n\n"
        "n_phi is the ONLY compile-time parameter of the cuda kernel, so only the values\n"
        "listed in the constructor's error message exist. Everything else is runtime,\n"
        "including the time-polynomial degree n, the window half-width W and the chunk\n"
        "length T. T must be a positive multiple of 32, W at most 16, and n at most 2 --\n"
        "the last matching the numpy reference, since a larger time-polynomial degree would\n"
        "be a configuration no test validates.\n\n"
        "FOOTGUN: no constant-offset subtraction is performed, so float32 data with a large\n"
        "DC level relative to its structure loses mantissa bits for nothing. In the intended\n"
        "pipeline the 1-d time detrender runs first and leaves the data roughly zero-mean.\n\n"
        "THREAD SAFETY: an instance owns per-launch scratch arrays, so one instance must not\n"
        "be used concurrently from two streams.\n\n"
        "The algorithm is specified in notes/detrending.tex, section '2-d detrending'.\n"
        "pirate_frb.detrending_spline is the pure-numpy reference that this kernel is\n"
        "validated against.");

    detrender_2d
          .def(py::init<const Detrender2d::Params &>(), py::arg("params"),
               "Create a Detrender2d from a Detrender2dParams.\n\n"
               "Raises:\n"
               "    RuntimeError: if no kernel is compiled for n_phi, if T is not a positive\n"
               "        multiple of 32, if n is outside [0,2], if W is outside [0,16] or\n"
               "        gives 2W+1 < n+1, or if the knot vector is invalid. The message says\n"
               "        which.")
          .def(py::init([](long nfreq, const std::vector<long> &knots, long M, long n_phi,
                           long n, long W, long T, double eta, double eps) {
              Detrender2d::Params params;
              params.nfreq = nfreq;
              params.knots = knots;
              params.M = M;
              params.n_phi = n_phi;
              params.n = n;
              params.W = W;
              params.T = T;
              params.eta = eta;
              params.eps = eps;
              return new Detrender2d(params);
          }),
               py::arg("nfreq"), py::arg("knots"), py::arg("M"),
               py::arg("n_phi") = 2, py::arg("n") = 2, py::arg("W") = 4, py::arg("T") = 2048,
               py::arg("eta") = 1.0e-3, py::arg("eps") = 3.0e-5,
               "Create a Detrender2d, from the Detrender2dParams fields as kwargs.\n\n"
               "Args:\n"
               "    nfreq: number of frequency channels\n"
               "    knots: non-decreasing list of channel indices, running from 0 to nfreq,\n"
               "        with the first and last values repeated exactly n_phi+1 times and no\n"
               "        interior value repeated more than n_phi+1 times. An interior value\n"
               "        repeated exactly n_phi+1 times is a zone boundary.\n"
               "    M: number of spectator (beam) rows\n"
               "    n_phi: spline degree in frequency\n"
               "    n: degree of the time polynomial\n"
               "    W: window half-width (the window is 2W+1 samples)\n"
               "    T: output samples per row (chunk size)\n"
               "    eta: regularization strength (dimensionless)\n"
               "    eps: mask-expansion threshold on r_min\n\n"
               "Raises:\n"
               "    RuntimeError: see the Detrender2dParams overload.")
          .def_readonly("params", &Detrender2d::params, "The Detrender2dParams used to create this")
          .def_property_readonly("nfreq", [](const Detrender2d &d) { return d.params.nfreq; })
          .def_property_readonly("M", [](const Detrender2d &d) { return d.params.M; })
          .def_property_readonly("n_phi", [](const Detrender2d &d) { return d.params.n_phi; })
          .def_property_readonly("n", [](const Detrender2d &d) { return d.params.n; })
          .def_property_readonly("W", [](const Detrender2d &d) { return d.params.W; })
          .def_property_readonly("T", [](const Detrender2d &d) { return d.params.T; })
          .def_property_readonly("eta", [](const Detrender2d &d) { return d.params.eta; })
          .def_property_readonly("eps", [](const Detrender2d &d) { return d.params.eps; })
          .def_readonly("nbuf", &Detrender2d::nbuf, "Buffer samples per row, = T + 2W")
          .def_readonly("N_phi", &Detrender2d::N_phi, "Number of B-spline basis functions")
          .def_readonly("nzone", &Detrender2d::nzone, "Number of zones")
          .def_readonly("nfrange", &Detrender2d::nfrange, "Number of internal freq-ranges")
          .def_readonly("channels_per_range", &Detrender2d::channels_per_range,
               "Internal freq-range width, derived from (nfreq, knots, T). It is part of the\n"
               "frequency summation order, which is why results are bit-reproducible across\n"
               "chunkings at a fixed T but only to roundoff across different T.")
          .def_static("configs", &Detrender2d::configs,
               "The compiled n_phi values. n, W and T are not among them: all three are\n"
               "runtime arguments. Returned as a list of ints.")
          .def_static("time_selected", &Detrender2d::time_selected,
               py::call_guard<py::gil_scoped_release>(),
               "Run timing benchmarks, for every compiled configuration "
               "(called via 'python -m pirate_frb time --dt2d')")
          .def("launch",
               [](const Detrender2d &self, Array<float> &data, Array<unsigned char> &mask,
                  uintptr_t stream_ptr) {
                   cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
                   self.launch(data, mask, stream);
               },
               py::arg("data"), py::arg("mask"), py::arg("stream_ptr"),
               py::call_guard<py::gil_scoped_release>(),   // async launch; body is pure C++
               "GPU kernel launch (async, does not sync stream).\n\n"
               "Args:\n"
               "    data: Array, shape (M, nfreq, nbuf), dtype float32, fully contiguous,\n"
               "          on GPU. Modified in place over buffer samples [W, W+T).\n"
               "    mask: Array, shape (M, nfreq, nbuf), dtype uint8, fully contiguous, on\n"
               "          GPU, {0,1}-valued. Modified in place over the same range, and the\n"
               "          output mask is the authoritative one (it can only lose samples).\n"
               "    stream_ptr: CUDA stream pointer (integer, e.g. from cupy stream.ptr)")
    ;

    // GpuDequantizationKernel: Python injections in pirate_frb/kernels/GpuDequantizationKernel.py:
    //   - launch: converts stream=None to current cupy stream
    py::class_<GpuDequantizationKernel>(m, "GpuDequantizationKernel",
        "GPU kernel to convert int4 array to float32 or float16, applying a\n"
        "per-(beam, freq, minichunk) affine transform during conversion.\n\n"
        "Inputs:\n"
        "    scales_offsets: shape (nbeams, nfreq, ntime//256, 2), dtype float16\n"
        "    data:           shape (nbeams, nfreq, ntime),         dtype int4\n"
        "Output:\n"
        "    out:            shape (nbeams, nfreq, ntime),         dtype float32 or float16\n\n"
        "Output formula:\n"
        "    out[b,f,t] = 0                                        if data[b,f,t] == -8\n"
        "    out[b,f,t] = scales_offsets[b,f,t//256,0] * data[b,f,t]\n"
        "               + scales_offsets[b,f,t//256,1]             otherwise\n\n"
        "The int4 'data' values are interpreted as signed two's complement (-8 to +7).\n"
        "data == -8 (bit pattern 0b1000) is the 'missing sample' sentinel and is always\n"
        "mapped to 0 in the output, regardless of scale and offset; this matches the\n"
        "convention used by AssembledFrame.data.\n\n"
        "Nibble packing in 'data': low nibble = even index, high nibble = odd index.\n"
        "The last axis of 'scales_offsets' is (scale, offset); one pair is shared by\n"
        "256 consecutive time samples of a single (beam, freq).\n\n"
        "IMPORTANT: Since numpy/cupy don't support int4 dtype (dtypes must be at least 8 bits),\n"
        "the Python wrapper for launch() accepts the data array as uint8 of shape\n"
        "(nbeams, nfreq, ntime//2), which is reinterpreted as int4 with shape\n"
        "(nbeams, nfreq, ntime). The uint8 array must be fully contiguous.\n\n"
        "For a CPU reference implementation, see ReferenceDequantizationKernel.")
          .def(py::init<Dtype, long, long, long>(),
               py::arg("dtype"), py::arg("nbeams"), py::arg("nfreq"), py::arg("ntime"),
               "Create a GpuDequantizationKernel.\n\n"
               "Args:\n"
               "    dtype: Output dtype (must be float32 or float16)\n"
               "    nbeams: Number of beams\n"
               "    nfreq: Number of frequency channels\n"
               "    ntime: Number of time samples (must be divisible by 256)\n\n"
               "Raises:\n"
               "    RuntimeError: If dtype is invalid or ntime is not divisible by 256")
          .def_readonly("dtype", &GpuDequantizationKernel::dtype,
               "Output dtype (float32 or float16)")
          .def_readonly("nbeams", &GpuDequantizationKernel::nbeams,
               "Number of beams")
          .def_readonly("nfreq", &GpuDequantizationKernel::nfreq,
               "Number of frequency channels")
          .def_readonly("ntime", &GpuDequantizationKernel::ntime,
               "Number of time samples")
          .def_readonly("resource_tracker", &GpuDequantizationKernel::resource_tracker,
               "ResourceTracker for memory/bandwidth accounting")
          .def("launch",
               [](const GpuDequantizationKernel &self,
                  Array<void> &out,
                  const Array<void> &scales_offsets,
                  const Array<void> &data_uint8,
                  uintptr_t stream_ptr) {
                   // Array<void> on the Python boundary (numpy/cupy float16);
                   // cast<__half>() does a runtime dtype check.
                   const Array<__half> &scoff = scales_offsets.cast<__half>(
                       "GpuDequantizationKernel.launch: scales_offsets");
                   Array<void> data_int4 = dequantization_uint8_to_int4(data_uint8, self.nbeams, self.nfreq, self.ntime);
                   cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
                   self.launch(out, scoff, data_int4, stream);
               },
               py::arg("out"), py::arg("scales_offsets"), py::arg("data_uint8"), py::arg("stream_ptr"),
               py::call_guard<py::gil_scoped_release>(),   // async launch; body is pure C++ (Array::cast, not python)
               "GPU kernel launch (async, does not sync stream).\n\n"
               "Args:\n"
               "    out: Output array, shape (nbeams, nfreq, ntime), dtype matches\n"
               "         kernel's dtype (float32 or float16), fully contiguous, on GPU\n"
               "    scales_offsets: Array, shape (nbeams, nfreq, ntime//256, 2), dtype float16,\n"
               "                    fully contiguous, on GPU. Last axis is (scale, offset).\n"
               "    data_uint8: Array, shape (nbeams, nfreq, ntime//2), dtype uint8,\n"
               "                fully contiguous, on GPU. Reinterpreted as int4 with shape\n"
               "                (nbeams, nfreq, ntime).\n"
               "    stream_ptr: CUDA stream pointer (integer, e.g. from cupy stream.ptr)\n\n"
               "The float32 kernel converts (scale, offset) from fp16 to fp32 before any\n"
               "arithmetic; the float16 kernel performs the affine math natively in fp16.\n"
               "data == -8 is mapped to 0 in the output (see class docstring).\n\n"
               "Note: The data array is passed as uint8 because numpy/cupy don't support int4\n"
               "(all dtypes must be at least 8 bits). Each uint8 element contains two\n"
               "int4 values: low nibble = even index, high nibble = odd index.")
          .def_static("test_random", &GpuDequantizationKernel::test_random,
               py::call_guard<py::gil_scoped_release>(),
               "Run randomized tests (called via 'python -m pirate_frb test --gdqk')")
          .def_static("time_selected", &GpuDequantizationKernel::time_selected,
               py::call_guard<py::gil_scoped_release>(),
               "Run timing benchmarks")
    ;

    // ReferenceDequantizationKernel: CPU reference for GpuDequantizationKernel (always
    // outputs float32). No Python injections; apply() takes no dtype/stream argument.
    py::class_<ReferenceDequantizationKernel>(m, "ReferenceDequantizationKernel",
        "CPU reference implementation of the int4 -> float32 dequantization performed by\n"
        "GpuDequantizationKernel (see that class for the affine-transform / missing-sample\n"
        "conventions). Always outputs float32; there is no output-dtype argument.\n\n"
        "IMPORTANT: Since numpy/cupy don't support int4 dtype (dtypes must be at least 8 bits),\n"
        "the Python wrapper for apply() accepts the data array as uint8 of shape\n"
        "(nbeams, nfreq, ntime//2), which is reinterpreted as int4 with shape\n"
        "(nbeams, nfreq, ntime). The uint8 array must be fully contiguous.")
          .def(py::init<long, long, long>(),
               py::arg("nbeams"), py::arg("nfreq"), py::arg("ntime"),
               "Create a ReferenceDequantizationKernel.\n\n"
               "Args:\n"
               "    nbeams: Number of beams\n"
               "    nfreq: Number of frequency channels\n"
               "    ntime: Number of time samples (must be divisible by 256)\n\n"
               "Raises:\n"
               "    RuntimeError: If any argument is non-positive or ntime is not divisible by 256")
          .def_readonly("nbeams", &ReferenceDequantizationKernel::nbeams,
               "Number of beams")
          .def_readonly("nfreq", &ReferenceDequantizationKernel::nfreq,
               "Number of frequency channels")
          .def_readonly("ntime", &ReferenceDequantizationKernel::ntime,
               "Number of time samples")
          .def("apply",
               [](ReferenceDequantizationKernel &self,
                  Array<float> &out,
                  const Array<void> &scales_offsets,
                  const Array<void> &data_uint8) {
                   // Array<void> on the Python boundary (numpy float16);
                   // cast<__half>() does a runtime dtype check.
                   const Array<__half> &scoff = scales_offsets.cast<__half>(
                       "ReferenceDequantizationKernel.apply: scales_offsets");
                   Array<void> data_int4 = dequantization_uint8_to_int4(data_uint8, self.nbeams, self.nfreq, self.ntime);
                   self.apply(out, scoff, data_int4);
               },
               py::arg("out"), py::arg("scales_offsets"), py::arg("data_uint8"),
               py::call_guard<py::gil_scoped_release>(),   // O(nbeams*nfreq*ntime) CPU loop
               "Reference implementation (CPU, always outputs float32).\n\n"
               "Args:\n"
               "    out: Output array, shape (nbeams, nfreq, ntime), dtype float32,\n"
               "         fully contiguous, on host\n"
               "    scales_offsets: Array, shape (nbeams, nfreq, ntime//256, 2), dtype float16,\n"
               "                    fully contiguous, on host. Last axis is (scale, offset).\n"
               "    data_uint8: Array, shape (nbeams, nfreq, ntime//2), dtype uint8,\n"
               "                fully contiguous, on host. Reinterpreted as int4 with shape\n"
               "                (nbeams, nfreq, ntime).\n\n"
               "Each (scale, offset) pair is converted from fp16 to fp32 immediately,\n"
               "before any arithmetic. data == -8 is mapped to 0 in the output (see\n"
               "class docstring).\n\n"
               "Note: The data array is passed as uint8 because numpy/cupy don't support int4\n"
               "(all dtypes must be at least 8 bits). Each uint8 element contains two\n"
               "int4 values: low nibble = even index, high nibble = odd index.")
    ;

    py::class_<GpuLaggedDownsamplingKernel>(m, "GpuLaggedDownsamplingKernel")
          .def_static("test_random", &GpuLaggedDownsamplingKernel::test_random, py::call_guard<py::gil_scoped_release>())
          .def_static("time_selected", &GpuLaggedDownsamplingKernel::time_selected, py::call_guard<py::gil_scoped_release>())
    ;

    py::class_<GpuPeakFindingKernel>(m, "GpuPeakFindingKernel")
          .def_static("test_random", &GpuPeakFindingKernel::test_random, py::arg("short_circuit") = false,
               py::call_guard<py::gil_scoped_release>())
          .def_static("registry_size", &GpuPeakFindingKernel::registry_size)
          .def_static("show_registry", &GpuPeakFindingKernel::show_registry)
    ;

    py::class_<GpuPfSquare>(m, "GpuPfSquare",
        "Convolves the peak-finding profiles with an input array and accumulates the sum of\n"
        "squares over time, in place, into a float64 accumulator.\n\n"
        "This is what a peak-finding kernel does NOT give you: out_max is a max over trials\n"
        "and a coarse-grained one at that, whereas the variance map is a sum over every\n"
        "output time. Every axis except time is a spectator, so 'ndm' is just a row count::\n\n"
        "    k = GpuPfSquare(max_kernel_width, total_beams, beams_per_batch, ndm, nt_in)\n"
        "    k.allocate(bump_allocator)\n"
        "    k.launch(acc, in_, ibatch)         # acc += sum_t (h_p * in)[t]^2\n")
          .def(py::init<long, long, long, long, long>(),
               py::arg("max_kernel_width"), py::arg("total_beams"), py::arg("beams_per_batch"),
               py::arg("ndm"), py::arg("nt_in"),
               py::call_guard<py::gil_scoped_release>(),
               "Create a GpuPfSquare.\n\n"
               "Args:\n"
               "    max_kernel_width: largest peak-finding boxcar, a power of two in\n"
               "        [1, constants.max_pf_width]\n"
               "    total_beams, beams_per_batch: beam geometry (total_beams must be a\n"
               "        multiple of beams_per_batch)\n"
               "    ndm: independent time series per beam. Any positive value -- it need not\n"
               "        be a power of two, so a (Dpf, M) pair can simply be flattened.\n"
               "    nt_in: time samples per chunk; a multiple of 32, and at least\n"
               "        max(2*max_kernel_width, 32)")
          .def("allocate", &GpuPfSquare::allocate, py::arg("allocator"),
               py::call_guard<py::gil_scoped_release>(),
               "Allocate (and zero) persistent state from a BumpAllocator. Must be called\n"
               "before launch().")
          .def("launch",
               [](GpuPfSquare &self, Array<double> &acc, const Array<float> &in_,
                  long ibatch, uintptr_t stream_ptr) {
                   self.launch(acc, in_, ibatch, reinterpret_cast<cudaStream_t> (stream_ptr));
               },
               py::arg("acc"), py::arg("in_"), py::arg("ibatch"), py::arg("stream_ptr"),
               py::call_guard<py::gil_scoped_release>(),   // async launch; body is pure C++
               "GPU kernel launch (async, does not sync stream).\n\n"
               "Args:\n"
               "    acc: Array, shape (beams_per_batch, ndm, nprofiles), float64, fully\n"
               "        contiguous, on GPU. ACCUMULATED INTO (+=), never overwritten, so the\n"
               "        caller zeroes it when a new accumulation starts.\n"
               "    in_: Array, shape (beams_per_batch, ndm, nt_in), float32, fully\n"
               "        contiguous, on GPU.\n"
               "    ibatch: 0 <= ibatch < nbatches. Calls must run 0, 1, ..., nbatches-1, 0,\n"
               "        ... -- the kernel checks, since it carries per-beam input history\n"
               "        across chunks.\n"
               "    stream_ptr: CUDA stream pointer (integer, e.g. from cupy stream.ptr)")
          .def_readonly("max_kernel_width", &GpuPfSquare::max_kernel_width)
          .def_readonly("total_beams", &GpuPfSquare::total_beams)
          .def_readonly("beams_per_batch", &GpuPfSquare::beams_per_batch)
          .def_readonly("ndm", &GpuPfSquare::ndm)
          .def_readonly("nt_in", &GpuPfSquare::nt_in)
          .def_readonly("nprofiles", &GpuPfSquare::nprofiles,
               "= 1 + 3*log2(max_kernel_width)")
          .def_readonly("nbatches", &GpuPfSquare::nbatches)
          .def_readonly("tpad", &GpuPfSquare::tpad,
               "Input samples of history carried across chunks, = max(2*max_kernel_width, 32)")
          .def_readonly("is_allocated", &GpuPfSquare::is_allocated)
          .def_readonly("resource_tracker", &GpuPfSquare::resource_tracker)
          .def_static("test_random", &GpuPfSquare::test_random, py::call_guard<py::gil_scoped_release>())
    ;

    py::class_<GpuRingbufCopyKernel>(m, "GpuRingbufCopyKernel")
          .def_static("test_random", &GpuRingbufCopyKernel::test_random, py::call_guard<py::gil_scoped_release>())
    ;

    py::class_<GpuTreeGriddingKernel>(m, "GpuTreeGriddingKernel",
        "Rebins input frequency channels into 'tree' channels by weighted sums (the GPU\n"
        "counterpart of ReferenceTreeGriddingKernel).\n\n"
        "Constructed from a plan's gridding params::\n\n"
        "    k = GpuTreeGriddingKernel(plan.tree_gridding_kernel_params)\n"
        "    k.allocate(bump_allocator)\n"
        "    k.launch(out, in_)")
          .def(py::init<const TreeGriddingKernelParams &>(), py::arg("params"),
               py::call_guard<py::gil_scoped_release>())
          .def("allocate", &GpuTreeGriddingKernel::allocate, py::arg("allocator"),
               py::call_guard<py::gil_scoped_release>(),
               "Copy the channel map to the GPU (as 32.32 fixed point). Must be called before\n"
               "launch().")
          .def("launch",
               [](GpuTreeGriddingKernel &self, Array<void> &out, const Array<void> &in_,
                  uintptr_t stream_ptr) {
                   self.launch(out, in_, reinterpret_cast<cudaStream_t> (stream_ptr));
               },
               py::arg("out"), py::arg("in_"), py::arg("stream_ptr"),
               py::call_guard<py::gil_scoped_release>(),   // async launch; body is pure C++
               "GPU kernel launch (async, does not sync stream).\n\n"
               "Args:\n"
               "    out: Array, shape (beams_per_batch, nchan, ntime), kernel dtype, on GPU\n"
               "    in_: Array, shape (beams_per_batch, nfreq, ntime), kernel dtype, on GPU\n"
               "    stream_ptr: CUDA stream pointer (integer, e.g. from cupy stream.ptr)")
          .def_readonly("params", &GpuTreeGriddingKernel::params)
          .def_readonly("is_allocated", &GpuTreeGriddingKernel::is_allocated)
          .def_readonly("resource_tracker", &GpuTreeGriddingKernel::resource_tracker)
          .def_static("test_random", &GpuTreeGriddingKernel::test_random, py::call_guard<py::gil_scoped_release>())
          .def_static("time_selected", &GpuTreeGriddingKernel::time_selected, py::call_guard<py::gil_scoped_release>())
    ;

    // ReferenceTreeGriddingKernel
    // Note: dtype is omitted since the reference kernel always uses float32 arrays.
    py::class_<ReferenceTreeGriddingKernel>(m, "ReferenceTreeGriddingKernel",
        "Reference implementation of tree gridding kernel.\n\n"
        "Rebins input frequency channels into output tree channels using weighted sums.\n"
        "Always uses float32 arrays regardless of dtype parameter in TreeGriddingKernelParams.")
          .def(py::init([](long nfreq, long nchan, long ntime,
                           long beams_per_batch, const Array<double> &channel_map) {
              TreeGriddingKernelParams params;
              params.dtype = Dtype::native<float>();  // Reference kernel always uses float32
              params.nfreq = nfreq;
              params.nchan = nchan;
              params.ntime = ntime;
              params.beams_per_batch = beams_per_batch;
              params.channel_map = channel_map;
              return new ReferenceTreeGriddingKernel(params);
          }),
          py::arg("nfreq"), py::arg("nchan"), py::arg("ntime"),
          py::arg("beams_per_batch"), py::arg("channel_map"))
          .def_property_readonly("nfreq", [](const ReferenceTreeGriddingKernel &self) { return self.params.nfreq; })
          .def_property_readonly("nchan", [](const ReferenceTreeGriddingKernel &self) { return self.params.nchan; })
          .def_property_readonly("ntime", [](const ReferenceTreeGriddingKernel &self) { return self.params.ntime; })
          .def_property_readonly("beams_per_batch", [](const ReferenceTreeGriddingKernel &self) { return self.params.beams_per_batch; })
          .def_property_readonly("channel_map", [](const ReferenceTreeGriddingKernel &self) { return self.params.channel_map; })
          .def("apply",
               [](ReferenceTreeGriddingKernel &self, const Array<float> &in) {
                   Dtype dtype = Dtype::native<float> ();
                   long beams = self.params.beams_per_batch;
                   long nchan = self.params.nchan;
                   long ntime = self.params.ntime;
                   Array<float> out(dtype, {beams, nchan, ntime}, af_rhost);
                   self.apply(out, in);
                   return out;
               },
               py::arg("in"),
               py::call_guard<py::gil_scoped_release>(),   // O(B*N*F*T) CPU rebin + pinned-host alloc
               "Rebins input frequency channels into output tree channels.\n\n"
               "Args:\n"
               "    in: Input array, shape (beams_per_batch, nfreq, ntime)\n\n"
               "Returns:\n"
               "    Output array, shape (beams_per_batch, nchan, ntime)")
    ;

    py::class_<PfOutputMicrokernel>(m, "PfOutputMicrokernel")
          .def_static("test_random", &PfOutputMicrokernel::test_random, py::call_guard<py::gil_scoped_release>())
          .def_static("registry_size", &PfOutputMicrokernel::registry_size)
          .def_static("show_registry", &PfOutputMicrokernel::show_registry)
    ;

    py::class_<PfWeightReaderMicrokernel>(m, "PfWeightReaderMicrokernel")
          .def_static("test_random", &PfWeightReaderMicrokernel::test_random, py::call_guard<py::gil_scoped_release>())
          .def_static("registry_size", &PfWeightReaderMicrokernel::registry_size)
          .def_static("show_registry", &PfWeightReaderMicrokernel::show_registry)
    ;

    py::class_<ReferenceLagbuf>(m, "ReferenceLagbuf")
          .def_static("test_random", &ReferenceLagbuf::test_random, py::call_guard<py::gil_scoped_release>())
    ;

    py::class_<ReferenceTree>(m, "ReferenceTree",
        "Reference implementation of tree dedispersion.\n\n"
        "Processes input incrementally in chunks of shape\n"
        "(num_beams, 2^amb_rank, 2^dd_rank, ntime * nspec).")
          .def(py::init([](long num_beams, long amb_rank, long dd_rank, long ntime,
                           long nspec, const std::vector<long> &subband_counts) {
              ReferenceTree::Params params;
              params.num_beams = num_beams;
              params.amb_rank = amb_rank;
              params.dd_rank = dd_rank;
              params.ntime = ntime;
              params.nspec = nspec;
              params.subband_counts = subband_counts;
              return new ReferenceTree(params);
          }),
          py::arg("num_beams"), py::arg("amb_rank"), py::arg("dd_rank"), py::arg("ntime"),
          py::arg("nspec") = 1, py::arg("subband_counts") = std::vector<long>{1})
          .def_property_readonly("num_beams", [](const ReferenceTree &self) { return self.params.num_beams; })
          .def_property_readonly("amb_rank", [](const ReferenceTree &self) { return self.params.amb_rank; })
          .def_property_readonly("dd_rank", [](const ReferenceTree &self) { return self.params.dd_rank; })
          .def_property_readonly("ntime", [](const ReferenceTree &self) { return self.params.ntime; })
          .def_property_readonly("nspec", [](const ReferenceTree &self) { return self.params.nspec; })
          .def_property_readonly("subband_counts", [](const ReferenceTree &self) { return self.params.subband_counts; })
          .def_readonly("frequency_subbands", &ReferenceTree::frequency_subbands)
          .def("dedisperse",
               [](ReferenceTree &self, Array<float> &buf, py::object out_obj) {
                   // The .cast<>() is python API, so it must happen BEFORE the GIL
                   // release below (a py::call_guard would be a bug here). The heavy
                   // CPU dedispersion then runs GIL-free.
                   Array<float> out;   // stays empty if out_obj is None (ok if M=1)
                   if (!out_obj.is_none())
                       out = out_obj.cast<Array<float>>();

                   py::gil_scoped_release nogil;
                   self.dedisperse(buf, out);
               },
               py::arg("buf"), py::arg("out") = py::none(),
               "Dedisperses buf in place, writes subbands to out.\n\n"
               "Footgun: the two results use different DM orderings. The natural\n"
               "(non-bit-reversed) DM ordering is a property of the subband array 'out',\n"
               "not of the in-place buffer.\n\n"
               "Args:\n"
               "    buf: Input/output array, shape (num_beams, 2^amb_rank, 2^dd_rank, ntime*nspec).\n"
               "         On return, its DM axis is BIT-REVERSED.\n"
               "    out: Output array for subbands, indexed by (beam, coarse DM, multiplet, time)\n"
               "         with the DM axis in natural order (optional if M=1)")
          .def_static("test_basics", &ReferenceTree::test_basics, py::call_guard<py::gil_scoped_release>())
          .def_static("test_subbands", &ReferenceTree::test_subbands, py::call_guard<py::gil_scoped_release>())
    ;

    // Exposed for unit tests only (see PfVarianceConvolver.test_kernels_match_reference).
    // The reference peak-finder computes in float32 regardless of the configured dtype.
    // shared_ptr holder: ReferenceDedisperser.pf_kernels returns shared_ptr elements.
    py::class_<ReferencePeakFindingKernel, std::shared_ptr<ReferencePeakFindingKernel>>(m, "ReferencePeakFindingKernel",
        "Reference (CPU, float32) peak-finding kernel; exposed for unit tests.")
          .def(py::init([](const std::vector<long> &subband_counts, long max_kernel_width,
                           long beams_per_batch, long total_beams, long ndm_out, long ndm_wt,
                           long nt_out, long nt_in, long nt_wt, long Dcore) {
              PeakFindingKernelParams params;
              params.subband_counts = subband_counts;
              params.dtype = Dtype::native<float> ();
              params.max_kernel_width = max_kernel_width;
              params.beams_per_batch = beams_per_batch;
              params.total_beams = total_beams;
              params.ndm_out = ndm_out;
              params.ndm_wt = ndm_wt;
              params.nt_out = nt_out;
              params.nt_in = nt_in;
              params.nt_wt = nt_wt;
              params.Dcore = Dcore;
              return new ReferencePeakFindingKernel(params);
          }),
          py::arg("subband_counts"), py::arg("max_kernel_width"),
          py::arg("beams_per_batch"), py::arg("total_beams"),
          py::arg("ndm_out"), py::arg("ndm_wt"),
          py::arg("nt_out"), py::arg("nt_in"), py::arg("nt_wt"), py::arg("Dcore"))
          .def_property_readonly("P", [](const ReferencePeakFindingKernel &self) { return self.nprofiles; })
          .def_property_readonly("M", [](const ReferencePeakFindingKernel &self) { return self.fs.M; })
          .def_property_readonly("N", [](const ReferencePeakFindingKernel &self) { return self.fs.N; })
          .def_property_readonly("Dout", [](const ReferencePeakFindingKernel &self) { return self.Dout; })
          .def_property_readonly("Dcore", [](const ReferencePeakFindingKernel &self) { return self.Dcore; })
          .def_readonly("samples_per_chunk", &ReferencePeakFindingKernel::samples_per_chunk,
               "Length-nprofiles list: the count that apply() divides by to turn its sum of\n"
               "squares into out_var. Multiply out_var by this to recover the raw sum of\n"
               "squares. Equal to nt_in for every profile when Dcore == 1, and smaller\n"
               "otherwise -- the peak-finder then evaluates on a sublattice of output times.")
          .def("apply",
               [](ReferencePeakFindingKernel &self, Array<float> &out_max, Array<uint> &out_argmax,
                  const Array<float> &in_, const Array<float> &wt, long ibatch) {
                   Array<double> out_var;   // empty -> out_var feature disabled
                   self.apply(out_max, out_argmax, out_var, in_, wt, ibatch);
               },
               py::arg("out_max"), py::arg("out_argmax"), py::arg("in_"),
               py::arg("wt"), py::arg("ibatch"),
               py::call_guard<py::gil_scoped_release>())
          .def("apply",
               [](ReferencePeakFindingKernel &self, Array<float> &out_max, Array<uint> &out_argmax,
                  const Array<float> &in_, const Array<float> &wt, long ibatch, Array<double> &out_var) {
                   self.apply(out_max, out_argmax, out_var, in_, wt, ibatch);
               },
               py::arg("out_max"), py::arg("out_argmax"), py::arg("in_"),
               py::arg("wt"), py::arg("ibatch"), py::arg("out_var"),
               py::call_guard<py::gil_scoped_release>())
          .def("eval_tokens",
               [](ReferencePeakFindingKernel &self, Array<float> &out,
                  const Array<uint> &in_tokens, const Array<float> &wt) {
                   self.eval_tokens(out, in_tokens, wt);
               },
               py::arg("out"), py::arg("in_tokens"), py::arg("wt"),
               py::call_guard<py::gil_scoped_release>())
    ;
}

}  // namespace pirate
