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

    // GpuDequantizationKernel: Python injections in pybind11_injections.py:
    //   - __init__: converts dtype argument via ksgpu.Dtype()
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
               "Run randomized tests (called via 'python -m pirate_frb test --gdqk')")
          .def_static("time_selected", &GpuDequantizationKernel::time_selected,
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
               "Rebins input frequency channels into output tree channels.\n\n"
               "Args:\n"
               "    in: Input array, shape (beams_per_batch, nfreq, ntime)\n\n"
               "Returns:\n"
               "    Output array, shape (beams_per_batch, nchan, ntime)")
    ;

    py::class_<PfOutputMicrokernel>(m, "PfOutputMicrokernel")
          .def_static("test_random", &PfOutputMicrokernel::test_random)
          .def_static("registry_size", &PfOutputMicrokernel::registry_size)
          .def_static("show_registry", &PfOutputMicrokernel::show_registry)
    ;

    py::class_<PfWeightReaderMicrokernel>(m, "PfWeightReaderMicrokernel")
          .def_static("test_random", &PfWeightReaderMicrokernel::test_random)
          .def_static("registry_size", &PfWeightReaderMicrokernel::registry_size)
          .def_static("show_registry", &PfWeightReaderMicrokernel::show_registry)
    ;

    py::class_<ReferenceLagbuf>(m, "ReferenceLagbuf")
          .def_static("test_random", &ReferenceLagbuf::test_random)
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
          .def_static("test_basics", &ReferenceTree::test_basics)
          .def_static("test_subbands", &ReferenceTree::test_subbands)
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
              return new ReferencePeakFindingKernel(params, Dcore);
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
               py::arg("wt"), py::arg("ibatch"))
          .def("apply",
               [](ReferencePeakFindingKernel &self, Array<float> &out_max, Array<uint> &out_argmax,
                  const Array<float> &in_, const Array<float> &wt, long ibatch, Array<double> &out_var) {
                   self.apply(out_max, out_argmax, out_var, in_, wt, ibatch);
               },
               py::arg("out_max"), py::arg("out_argmax"), py::arg("in_"),
               py::arg("wt"), py::arg("ibatch"), py::arg("out_var"))
          .def("eval_tokens",
               [](ReferencePeakFindingKernel &self, Array<float> &out,
                  const Array<uint> &in_tokens, const Array<float> &wt) {
                   self.eval_tokens(out, in_tokens, wt);
               },
               py::arg("out"), py::arg("in_tokens"), py::arg("wt"))
    ;
}

}  // namespace pirate
