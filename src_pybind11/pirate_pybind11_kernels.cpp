// Python bindings for GPU kernel classes (pirate_frb.kernels subpackage).
// See pirate_pybind11.cu for the main module definition.

#define PY_ARRAY_UNIQUE_SYMBOL PyArray_API_pirate
#define NO_IMPORT_ARRAY  // Secondary file: don't call _import_array()
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <pybind11/stl.h>
#include <ksgpu/pybind11.hpp>

#include "../include/pirate/CoalescedDdKernel2.hpp"
#include "../include/pirate/DedispersionKernel.hpp"
#include "../include/pirate/Detrender1d.hpp"
#include "../include/pirate/Detrender2d.hpp"
#include "../include/pirate/GpuDequantizationKernel.hpp"
#include "../include/pirate/LaggedDownsamplingKernel.hpp"
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

    py::class_<GpuSbDedispersionKernel>(m, "GpuSbDedispersionKernel")
          .def_static("test_random", &GpuSbDedispersionKernel::test_random, py::call_guard<py::gil_scoped_release>())
          .def_static("registry_size", &GpuSbDedispersionKernel::registry_size)
          .def_static("show_registry", &GpuSbDedispersionKernel::show_registry)
    ;

    py::class_<GpuDedispersionKernel>(m, "GpuDedispersionKernel")
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
        "The algorithm is specified in notes/tree_dedispersion.tex, section 'Time detrending\n"
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

    // Detrender2d: Python injections in pirate_frb/kernels/Detrender2d.py:
    //   - launch: converts stream=None to current cupy stream
    py::class_<Detrender2d> detrender_2d(m, "Detrender2d",
        "The 2-d spline detrender: a regularized fit of a B-spline in frequency times a\n"
        "local polynomial in time, subtracted from the data.\n\n"
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
        "The algorithm is specified in notes/tree_dedispersion.tex, section '2-d detrending'.\n"
        "pirate_frb.detrending_spline is the pure-numpy reference that this kernel is\n"
        "validated against.");

    detrender_2d
          .def(py::init<long, const std::vector<long> &, long, long, long, long, long, double, double, long>(),
               py::arg("nfreq"), py::arg("knots"), py::arg("M"),
               py::arg("n_phi") = 2, py::arg("n") = 2, py::arg("W") = 4, py::arg("T") = 2048,
               py::arg("eta") = 1.0e-3, py::arg("eps") = 3.0e-5,
               py::arg("channels_per_range") = 0,
               "Create a Detrender2d.\n\n"
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
               "    eps: mask-expansion threshold on r_min\n"
               "    channels_per_range: internal freq-range width; 0 (the default) derives it\n"
               "        from (nfreq, knots, T).  Exposed only because it is part of the\n"
               "        frequency summation order: two instances with different values agree\n"
               "        to roundoff but not bit-for-bit, so pass it explicitly when two\n"
               "        instances must agree exactly (e.g. comparing T=512 against T=2048).\n\n"
               "Raises:\n"
               "    RuntimeError: if no kernel is compiled for n_phi, if T is not a positive\n"
               "        multiple of 32, if n is outside [0,2], if W is outside [0,16] or\n"
               "        gives 2W+1 < n+1, or if the knot vector is invalid. The message says\n"
               "        which.")
          .def_readonly("nfreq", &Detrender2d::nfreq, "Number of frequency channels")
          .def_readonly("M", &Detrender2d::M, "Number of spectator (beam) rows")
          .def_readonly("n_phi", &Detrender2d::n_phi, "Spline degree in frequency")
          .def_readonly("n", &Detrender2d::n, "Degree of the time polynomial")
          .def_readonly("W", &Detrender2d::W, "Window half-width (the window is 2W+1 samples)")
          .def_readonly("T", &Detrender2d::T, "Output samples per row (chunk size)")
          .def_readonly("nbuf", &Detrender2d::nbuf, "Buffer samples per row, = T + 2W")
          .def_readonly("eta", &Detrender2d::eta, "Regularization strength")
          .def_readonly("eps", &Detrender2d::eps, "Mask-expansion threshold on r_min")
          .def_readonly("N_phi", &Detrender2d::N_phi, "Number of B-spline basis functions")
          .def_readonly("nzone", &Detrender2d::nzone, "Number of zones")
          .def_readonly("nfrange", &Detrender2d::nfrange, "Number of internal freq-ranges")
          .def_readonly("channels_per_range", &Detrender2d::channels_per_range,
               "Freq-range width actually used (derived unless requested)")
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

    py::class_<GpuPfSquare>(m, "GpuPfSquare")
          .def_static("test_random", &GpuPfSquare::test_random, py::call_guard<py::gil_scoped_release>())
    ;

    py::class_<GpuRingbufCopyKernel>(m, "GpuRingbufCopyKernel")
          .def_static("test_random", &GpuRingbufCopyKernel::test_random, py::call_guard<py::gil_scoped_release>())
    ;

    py::class_<GpuTreeGriddingKernel>(m, "GpuTreeGriddingKernel")
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
               "Args:\n"
               "    buf: Input/output array, shape (num_beams, 2^amb_rank, 2^dd_rank, ntime*nspec)\n"
               "    out: Output array for subbands (optional if M=1)")
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
