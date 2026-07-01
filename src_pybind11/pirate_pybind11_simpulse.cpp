// Python bindings for the FRB pulse-simulation code (pirate_frb.simpulse subpackage).
// C++ classes/functions are defined in include/pirate/simpulse.hpp + src_lib/simpulse.cpp;
// see pirate_pybind11.cpp for the main module.
//
// Method injection (in pirate_frb/pybind11_injections.py): SinglePulse.get_signal_to_noise
// (scalar-vs-vector dispatch over the private _get_signal_to_noise_{scalar,vector} bindings).

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
        "One dispersed, scattered FRB pulse, on a fixed frequency channelization and a fixed,\n"
        "zero-based time sampling (dt = 1e-3 * time_sample_ms seconds; sample it spans\n"
        "[it*dt, (it+1)*dt]).\n"
        "\n"
        "The constructor precomputes the pulse as a SPARSE array of per-channel time samples\n"
        "(sparse_i0 / sparse_n / sparse_offset / sparse_data); add_to_timestream() scatters it into a\n"
        "dense (nfreq, out_nt) array. Frequency channels are ordered LOW to HIGH and may have UNEQUAL\n"
        "widths (channel i spans ``[freq_edges_MHz[i], freq_edges_MHz[i+1]]``). NOTE: bonsai/rf_pipelines\n"
        "and pirate intensity arrays use the OPPOSITE, high-to-low, ordering.\n"
        "\n"
        "Attributes (read-only). Construction parameters:\n"
        "\n"
        "- ``pulse_nt`` (int) -- number of under-the-hood samples representing the pulse.\n"
        "- ``time_sample_ms`` (float) -- time-sample duration in ms (dt = 1e-3*time_sample_ms sec).\n"
        "- ``freq_edges_MHz`` (array) -- sorted, length (nfreq+1); channel i spans edges[i]..edges[i+1].\n"
        "- ``dm`` (float) -- dispersion measure (pc cm^{-3}).\n"
        "- ``sm`` (float) -- scattering measure (scattering time in ms at 1 GHz).\n"
        "- ``intrinsic_width`` (float) -- frequency-independent Gaussian width in seconds.\n"
        "- ``fluence`` (float) -- integrated flux at the central frequency.\n"
        "- ``spectral_index`` (float) -- exponent alpha in F(nu) = F(nu_0) (nu/nu_0)^alpha.\n"
        "- ``undispersed_arrival_time`` (float) -- arrival time as freq->infty, in seconds.\n"
        "\n"
        "Precomputed sparse representation (arrays): ``sparse_i0`` / ``sparse_n`` / ``sparse_offset``\n"
        "(length nfreq, int) and ``sparse_data`` (float). Also ``nt_min`` (smallest out_nt with no\n"
        "clipping) and the derived ``nfreq`` / ``freq_lo_MHz`` / ``freq_hi_MHz``.\n")

        .def(py::init([](long pulse_nt, double time_sample_ms, const Array<double> &freq_edges_MHz,
                         double dm, double sm, double intrinsic_width, double fluence,
                         double spectral_index, double undispersed_arrival_time) {
                 SinglePulse::Params p;
                 p.pulse_nt = pulse_nt;
                 p.time_sample_ms = time_sample_ms;
                 p.freq_edges_MHz = freq_edges_MHz;
                 p.dm = dm;
                 p.sm = sm;
                 p.intrinsic_width = intrinsic_width;
                 p.fluence = fluence;
                 p.spectral_index = spectral_index;
                 p.undispersed_arrival_time = undispersed_arrival_time;
                 return new SinglePulse(p);
             }),
             py::arg("pulse_nt"), py::arg("time_sample_ms"), py::arg("freq_edges_MHz"),
             py::arg("dm"), py::arg("sm"), py::arg("intrinsic_width"), py::arg("fluence"),
             py::arg("spectral_index"), py::arg("undispersed_arrival_time"))

        // Read-only views of the construction parameters (SinglePulse::params).
        .def_property_readonly("pulse_nt", [](const SinglePulse &s) { return s.params.pulse_nt; })
        .def_property_readonly("time_sample_ms", [](const SinglePulse &s) { return s.params.time_sample_ms; })
        .def_property_readonly("freq_edges_MHz", [](const SinglePulse &s) { return s.params.freq_edges_MHz; })
        .def_property_readonly("dm", [](const SinglePulse &s) { return s.params.dm; })
        .def_property_readonly("sm", [](const SinglePulse &s) { return s.params.sm; })
        .def_property_readonly("intrinsic_width", [](const SinglePulse &s) { return s.params.intrinsic_width; })
        .def_property_readonly("fluence", [](const SinglePulse &s) { return s.params.fluence; })
        .def_property_readonly("spectral_index", [](const SinglePulse &s) { return s.params.spectral_index; })
        .def_property_readonly("undispersed_arrival_time", [](const SinglePulse &s) { return s.params.undispersed_arrival_time; })

        // Precomputed sparse representation.
        .def_readonly("sparse_i0", &SinglePulse::sparse_i0)
        .def_readonly("sparse_n", &SinglePulse::sparse_n)
        .def_readonly("sparse_offset", &SinglePulse::sparse_offset)
        .def_readonly("sparse_data", &SinglePulse::sparse_data)
        .def_readonly("nt_min", &SinglePulse::nt_min)

        // Derived read-only attributes, computed from freq_edges_MHz.
        .def_property_readonly("nfreq", [](const SinglePulse &s) { return s.params.freq_edges_MHz.size - 1; })
        .def_property_readonly("freq_lo_MHz", [](const SinglePulse &s) { return s.params.freq_edges_MHz.data[0]; })
        .def_property_readonly("freq_hi_MHz", [](const SinglePulse &s) { return s.params.freq_edges_MHz.data[s.params.freq_edges_MHz.size - 1]; })

        .def("add_to_timestream", &SinglePulse::add_to_timestream,
             py::arg("out"), py::arg("weight") = 1.0,
             "Add the pulse to a 2-d (nfreq, out_nt) float32 array, in place, scaled by 'weight'.\n"
             "\n"
             "The grid is zero-based (sample it spans [it*dt, (it+1)*dt] seconds); samples at index\n"
             ">= out_nt are clipped (size out_nt >= nt_min for no clipping). 'out' must be float32,\n"
             "with contiguous time samples, ordered low to high in frequency.")

        // get_signal_to_noise() overloads, bound under private names; the public dispatch (scalar
        // vs vector, channel-weight defaulting) is the get_signal_to_noise injection.
        .def("_get_signal_to_noise_scalar",
             py::overload_cast<double>(&SinglePulse::get_signal_to_noise, py::const_),
             py::arg("sample_rms"))

        .def("_get_signal_to_noise_vector",
             [](const SinglePulse &self, const Array<double> &sample_rms, py::object channel_weights) {
                 if (channel_weights.is_none())
                     return self.get_signal_to_noise(sample_rms, Array<double>());
                 Array<double> cw = channel_weights.cast<Array<double>>();
                 return self.get_signal_to_noise(sample_rms, cw);
             },
             py::arg("sample_rms"), py::arg("channel_weights"))

        .def("__repr__", &SinglePulse::str)
    ;
}

}  // namespace pirate
