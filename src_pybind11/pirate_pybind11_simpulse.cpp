// Python bindings for the FRB pulse-simulation code (pirate_frb.simpulse subpackage).
// C++ classes/functions are defined in include/pirate/simpulse.hpp + src_lib/simpulse.cpp;
// see pirate_pybind11.cpp for the main module. (SinglePulse has no method injections.)

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

    py::class_<SinglePulse, std::shared_ptr<SinglePulse>>(m, "SinglePulse",
        "One dispersed, scattered FRB pulse, on a fixed frequency channelization and a fixed,\n"
        "zero-based time sampling (dt = 1e-3 * time_sample_ms seconds; sample it spans\n"
        "[it*dt, (it+1)*dt]).\n"
        "\n"
        "The constructor precomputes the pulse as a SPARSE array of per-channel time samples\n"
        "(freq_it0 / freq_nt / freq_sd_off / sparse_data); add_to_timestream() scatters it into a\n"
        "dense (nfreq, out_nt) array. Frequency channels are ordered LOW to HIGH and may have UNEQUAL\n"
        "widths (channel i spans ``[freq_edges_MHz[i], freq_edges_MHz[i+1]]``).\n"
        "\n"
        "Attributes (read-only). Construction parameters:\n"
        "\n"
        "- ``internal_nt`` (int) -- number of under-the-hood samples (a power of two). Default 1024.\n"
        "- ``time_sample_ms`` (float) -- time-sample duration in ms (dt = 1e-3*time_sample_ms sec).\n"
        "- ``freq_edges_MHz`` (array) -- sorted, length (nfreq+1); channel i spans edges[i]..edges[i+1].\n"
        "- ``freq_variances`` (array) -- per-channel noise variance, length nfreq (all positive).\n"
        "- ``dm`` (float) -- dispersion measure (pc cm^{-3}).\n"
        "- ``sm`` (float) -- scattering measure (scattering time in ms at 1 GHz).\n"
        "- ``intrinsic_width`` (float) -- frequency-independent Gaussian width in seconds.\n"
        "- ``snr`` (float) -- target signal-to-noise (perfect matched filter); sets the normalization.\n"
        "- ``spectral_index`` (float) -- exponent alpha in F(nu) = F(nu_0) (nu/nu_0)^alpha.\n"
        "- ``undispersed_arrival_time_sec`` (float) -- arrival time as freq->infty, in seconds.\n"
        "- ``allow_negative_arrival_times`` (bool) -- if False (default), a pulse with samples at t<0\n"
        "  is an error; if True, the t<0 part is clipped. Default False.\n"
        "\n"
        "Precomputed sparse representation (arrays): ``freq_it0`` / ``freq_nt`` / ``freq_sd_off``\n"
        "(length nfreq, int) and ``sparse_data`` (float). Also ``nt_min`` (smallest out_nt with no\n"
        "clipping) and the derived ``nfreq`` / ``freq_lo_MHz`` / ``freq_hi_MHz``.\n")

        .def(py::init([](double dm, double sm, double intrinsic_width, double spectral_index,
                         double undispersed_arrival_time_sec, double time_sample_ms, double snr,
                         const Array<double> &freq_edges_MHz, const Array<double> &freq_variances,
                         bool allow_negative_arrival_times, long internal_nt) {
                 SinglePulse::Params p;
                 p.dm = dm;
                 p.sm = sm;
                 p.intrinsic_width = intrinsic_width;
                 p.spectral_index = spectral_index;
                 p.undispersed_arrival_time_sec = undispersed_arrival_time_sec;
                 p.time_sample_ms = time_sample_ms;
                 p.snr = snr;
                 p.freq_edges_MHz = freq_edges_MHz;
                 p.freq_variances = freq_variances;
                 p.allow_negative_arrival_times = allow_negative_arrival_times;
                 p.internal_nt = internal_nt;
                 return new SinglePulse(p);
             }),
             // Argument order matches the C++ Params members. Only the last two (allow_negative_arrival_times,
             // internal_nt) have defaults -- the required arrays sit mid-struct, so everything before them
             // (including snr) must be a required argument.
             py::arg("dm"), py::arg("sm"), py::arg("intrinsic_width"), py::arg("spectral_index"),
             py::arg("undispersed_arrival_time_sec"), py::arg("time_sample_ms"), py::arg("snr"),
             py::arg("freq_edges_MHz"), py::arg("freq_variances"),
             py::arg("allow_negative_arrival_times") = false, py::arg("internal_nt") = 1024)

        // Read-only views of the construction parameters (SinglePulse::params).
        .def_property_readonly("internal_nt", [](const SinglePulse &s) { return s.params.internal_nt; })
        .def_property_readonly("time_sample_ms", [](const SinglePulse &s) { return s.params.time_sample_ms; })
        .def_property_readonly("freq_edges_MHz", [](const SinglePulse &s) { return s.params.freq_edges_MHz; })
        .def_property_readonly("freq_variances", [](const SinglePulse &s) { return s.params.freq_variances; })
        .def_property_readonly("dm", [](const SinglePulse &s) { return s.params.dm; })
        .def_property_readonly("sm", [](const SinglePulse &s) { return s.params.sm; })
        .def_property_readonly("intrinsic_width", [](const SinglePulse &s) { return s.params.intrinsic_width; })
        .def_property_readonly("snr", [](const SinglePulse &s) { return s.params.snr; })
        .def_property_readonly("spectral_index", [](const SinglePulse &s) { return s.params.spectral_index; })
        .def_property_readonly("undispersed_arrival_time_sec", [](const SinglePulse &s) { return s.params.undispersed_arrival_time_sec; })
        .def_property_readonly("allow_negative_arrival_times", [](const SinglePulse &s) { return s.params.allow_negative_arrival_times; })

        // Precomputed sparse representation.
        .def_readonly("freq_it0", &SinglePulse::freq_it0)
        .def_readonly("freq_nt", &SinglePulse::freq_nt)
        .def_readonly("freq_sd_off", &SinglePulse::freq_sd_off)
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
             ">= out_nt are clipped (size out_nt >= nt_min for no clipping). 'out' must be a host (CPU)\n"
             "float32 array with contiguous time samples, ordered low to high in frequency.")

        .def("__repr__", &SinglePulse::str)
    ;
}

}  // namespace pirate
