// Python bindings for the FRB pulse-simulation code (pirate_frb.simpulse subpackage).
// C++ classes/functions are defined in include/pirate/simpulse.hpp + src_lib/simpulse.cpp;
// see pirate_pybind11.cpp for the main module.
//
// Method injections (in pirate_frb/pybind11_injections.py): SinglePulse.get_signal_to_noise
// (scalar-vs-vector dispatch over the private _get_signal_to_noise_{scalar,vector} bindings) and
// SinglePulse.get_sparse (convenience wrapper around get_n_sparse + add_to_timestream_sparse).

#define PY_ARRAY_UNIQUE_SYMBOL PyArray_API_pirate
#define NO_IMPORT_ARRAY  // Secondary file: don't call _import_array()
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <pybind11/stl.h>
#include <ksgpu/pybind11.hpp>

#include "../include/pirate/simpulse.hpp"

using namespace std;
using namespace ksgpu;
using namespace pirate::simpulse;
namespace py = pybind11;


namespace pirate {

void register_simpulse_bindings(pybind11::module &m)
{
    m.def("dispersion_delay", &dispersion_delay, py::arg("dm"), py::arg("freq_MHz"),
          "Dispersion delay in seconds. 'dm' is the dispersion measure (pc cm^{-3}).");

    m.def("scattering_time", &scattering_time, py::arg("sm"), py::arg("freq_MHz"),
          "Scattering time in seconds. 'sm' is the scattering time in milliseconds at 1 GHz.");

    // -------------------------------------------------------------------------------- SinglePulse

    py::class_<SinglePulse>(m, "SinglePulse",
        "One dispersed, scattered FRB pulse, in a fixed frequency channelization.\n"
        "\n"
        "Frequency channels are ordered LOW to HIGH and may have UNEQUAL widths (channel i spans\n"
        "``[freq_edges_MHz[i], freq_edges_MHz[i+1]]``). NOTE: bonsai/rf_pipelines and pirate intensity\n"
        "arrays use the OPPOSITE, high-to-low, ordering.\n"
        "\n"
        "Attributes (read-only; the construction parameters, fixed after construction):\n"
        "\n"
        "- ``pulse_nt`` (int) -- number of under-the-hood samples representing the pulse.\n"
        "- ``freq_edges_MHz`` (array) -- sorted, length (nfreq+1); channel i spans edges[i]..edges[i+1].\n"
        "- ``dm`` (float) -- dispersion measure (pc cm^{-3}).\n"
        "- ``sm`` (float) -- scattering measure (scattering time in ms at 1 GHz).\n"
        "- ``intrinsic_width`` (float) -- frequency-independent Gaussian width in seconds.\n"
        "- ``fluence`` (float) -- integrated flux at the central frequency.\n"
        "- ``spectral_index`` (float) -- exponent alpha in F(nu) = F(nu_0) (nu/nu_0)^alpha.\n"
        "- ``undispersed_arrival_time`` (float) -- arrival time as freq->infty, in seconds.\n"
        "\n"
        "Also derived (read-only): ``nfreq``, ``freq_lo_MHz``, ``freq_hi_MHz`` (from freq_edges_MHz).\n")

        .def(py::init([](long pulse_nt, const Array<double> &freq_edges_MHz,
                         double dm, double sm, double intrinsic_width, double fluence,
                         double spectral_index, double undispersed_arrival_time) {
                 SinglePulse::Params p;
                 p.pulse_nt = pulse_nt;
                 p.freq_edges_MHz = freq_edges_MHz;
                 p.dm = dm;
                 p.sm = sm;
                 p.intrinsic_width = intrinsic_width;
                 p.fluence = fluence;
                 p.spectral_index = spectral_index;
                 p.undispersed_arrival_time = undispersed_arrival_time;
                 return new SinglePulse(p);
             }),
             py::arg("pulse_nt"), py::arg("freq_edges_MHz"),
             py::arg("dm"), py::arg("sm"), py::arg("intrinsic_width"), py::arg("fluence"),
             py::arg("spectral_index"), py::arg("undispersed_arrival_time"))

        // Read-only views of the construction parameters (SinglePulse::params).
        .def_property_readonly("pulse_nt", [](const SinglePulse &s) { return s.params.pulse_nt; })
        .def_property_readonly("freq_edges_MHz", [](const SinglePulse &s) { return s.params.freq_edges_MHz; })
        .def_property_readonly("dm", [](const SinglePulse &s) { return s.params.dm; })
        .def_property_readonly("sm", [](const SinglePulse &s) { return s.params.sm; })
        .def_property_readonly("intrinsic_width", [](const SinglePulse &s) { return s.params.intrinsic_width; })
        .def_property_readonly("fluence", [](const SinglePulse &s) { return s.params.fluence; })
        .def_property_readonly("spectral_index", [](const SinglePulse &s) { return s.params.spectral_index; })
        .def_property_readonly("undispersed_arrival_time", [](const SinglePulse &s) { return s.params.undispersed_arrival_time; })
        // Derived read-only attributes, computed from freq_edges_MHz.
        .def_property_readonly("nfreq", [](const SinglePulse &s) { return s.params.freq_edges_MHz.size - 1; })
        .def_property_readonly("freq_lo_MHz", [](const SinglePulse &s) { return s.params.freq_edges_MHz.data[0]; })
        .def_property_readonly("freq_hi_MHz", [](const SinglePulse &s) { return s.params.freq_edges_MHz.data[s.params.freq_edges_MHz.size - 1]; })

        .def("get_endpoints", [](const SinglePulse &self) {
                 double t0, t1;
                 self.get_endpoints(t0, t1);
                 return py::make_tuple(t0, t1);
             },
             "Returns (t0, t1): the earliest and latest arrival times in [freq_lo_MHz, freq_hi_MHz].")

        .def("add_to_timestream", &SinglePulse::add_to_timestream,
             py::arg("out"), py::arg("out_t0"), py::arg("out_t1"), py::arg("weight") = 1.0,
             "Add the pulse to a 2-d (nfreq, out_nt) float32 array, in place.\n"
             "\n"
             "'out' is written in place (it must be float32; time samples must be contiguous). 'out_t0'\n"
             "and 'out_t1' are the endpoints of the sampled region, in seconds, relative to the same\n"
             "origin as undispersed_arrival_time. Frequencies are ordered low to high (row i is the\n"
             "i-th lowest-frequency channel).\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "out : Array<float>\n"
             "    2-d float32 array of shape (nfreq, out_nt), modified in place.\n"
             "out_t0 : float\n"
             "    Start time of the sampled region (seconds).\n"
             "out_t1 : float\n"
             "    End time of the sampled region (seconds).\n"
             "weight : float\n"
             "    Scaling factor for the pulse amplitude (default 1).")

        .def("get_n_sparse", &SinglePulse::get_n_sparse,
             py::arg("out_t0"), py::arg("out_t1"), py::arg("out_nt"),
             "Total number of samples needed for the sparse representation (sum over channels).")

        .def("add_to_timestream_sparse", &SinglePulse::add_to_timestream_sparse,
             py::arg("out"), py::arg("out_i0"), py::arg("out_n"),
             py::arg("out_t0"), py::arg("out_t1"), py::arg("out_nt"), py::arg("weight") = 1.0,
             "Sparse version of add_to_timestream(), writing the packed samples + per-channel offsets\n"
             "in place. 'out' is a 1-d float32 array of length >= get_n_sparse(); 'out_i0' and 'out_n'\n"
             "are length-nfreq int64 arrays (channel i occupies dense indices [out_i0[i], out_i0[i]+\n"
             "out_n[i]), packed at sum(out_n[:i])). Usually called via the get_sparse() convenience.")

        // get_signal_to_noise() overloads, bound under private names; the public dispatch (scalar
        // vs vector, channel-weight defaulting) is the get_signal_to_noise injection.
        .def("_get_signal_to_noise_scalar",
             py::overload_cast<double, double, double>(&SinglePulse::get_signal_to_noise, py::const_),
             py::arg("sample_dt"), py::arg("sample_rms"), py::arg("sample_t0"))

        .def("_get_signal_to_noise_vector",
             [](const SinglePulse &self, double sample_dt, const Array<double> &sample_rms,
                py::object channel_weights, double sample_t0) {
                 if (channel_weights.is_none())
                     return self.get_signal_to_noise(sample_dt, sample_rms, Array<double>(), sample_t0);
                 Array<double> cw = channel_weights.cast<Array<double>>();
                 return self.get_signal_to_noise(sample_dt, sample_rms, cw, sample_t0);
             },
             py::arg("sample_dt"), py::arg("sample_rms"), py::arg("channel_weights"), py::arg("sample_t0"))

        .def("__repr__", &SinglePulse::str)
    ;
}

}  // namespace pirate
