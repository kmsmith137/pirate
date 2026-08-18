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
        "zero-based time sampling.\n"
        "\n"
        "Time sample ``it`` spans ``[it*dt, (it+1)*dt]`` seconds, where\n"
        "``dt = 1e-3 * time_sample_ms``. Sample indices may be negative, for a pulse whose\n"
        "arrival extends to ``t < 0``. Frequency channels are ordered LOW to HIGH and may have\n"
        "UNEQUAL widths: channel ``i`` spans ``[freq_edges_MHz[i], freq_edges_MHz[i+1]]``.\n"
        "\n"
        "The constructor precomputes the pulse as a SPARSE array of per-channel time samples,\n"
        "and ``add_to_timestream()`` then scatters it into a dense ``(nfreq, out_nt)`` array.\n"
        "The precompute is the expensive part (a per-channel inverse FFT), so a pulse is worth\n"
        "reusing: ``shift_samples()`` retimes one without recomputing it.\n"
        "\n"
        "Constructor::\n"
        "\n"
        "    sp = pirate_frb.simpulse.SinglePulse(\n"
        "        dm = 100.0,                           # pc cm^-3\n"
        "        sm = 0.0,                             # scattering time (ms) at 1 GHz\n"
        "        intrinsic_width = 2.0e-3,             # seconds\n"
        "        spectral_index = 0.0,                 # alpha, in F(nu) ~ nu^alpha\n"
        "        undispersed_arrival_time_sec = 0.05,\n"
        "        time_sample_ms = 1.0,\n"
        "        snr = 30.0,\n"
        "        freq_edges_MHz = np.linspace(400., 800., nfreq+1),\n"
        "        freq_variances = np.ones(nfreq))\n"
        "\n"
        "Three more arguments are optional: ``subband_freq_lo_MHz`` (default 0) and\n"
        "``subband_freq_hi_MHz`` (default 1e9), which restrict the pulse to channels\n"
        "overlapping that subband, and ``internal_nt`` (default 1024). Every constructor\n"
        "argument is readable afterwards as a same-named attribute.\n"
        "\n"
        "Attributes (read-only) -- the construction parameters:\n"
        "\n"
        "- ``dm`` (float) -- dispersion measure, in pc cm^-3.\n"
        "- ``sm`` (float) -- scattering measure, defined here as the scattering time in\n"
        "  MILLISECONDS (not seconds) at 1 GHz.\n"
        "- ``intrinsic_width`` (float) -- frequency-independent Gaussian width, in seconds.\n"
        "- ``spectral_index`` (float) -- the exponent alpha in\n"
        "  ``F(nu) = F(nu_0) * (nu/nu_0)**alpha``.\n"
        "- ``undispersed_arrival_time_sec`` (float) -- arrival time in the limit\n"
        "  ``nu -> infinity``, in seconds. This is the one parameter that is not immutable:\n"
        "  ``shift_samples()`` updates it.\n"
        "- ``time_sample_ms`` (float) -- time-sample duration in ms, i.e. ``dt`` above.\n"
        "- ``snr`` (float) -- target signal-to-noise assuming a perfect matched filter. Sets\n"
        "  the overall normalization of ``sparse_data``.\n"
        "- ``freq_edges_MHz`` (float array) -- channel edges, strictly increasing, of length\n"
        "  ``nfreq+1``.\n"
        "- ``freq_variances`` (float array) -- per-channel noise variance, length ``nfreq``,\n"
        "  all positive. Enters the SNR normalization.\n"
        "- ``subband_freq_lo_MHz``, ``subband_freq_hi_MHz`` (float) -- the pulse is restricted\n"
        "  to channels overlapping this subband. Channels outside it are \"inactive\", with\n"
        "  ``freq_nt == 0``.\n"
        "- ``internal_nt`` (int) -- number of under-the-hood samples (a power of two).\n"
        "\n"
        "Attributes (read-only) -- the precomputed sparse representation, plus derived sizes:\n"
        "\n"
        "- ``freq_it0`` (int array, length ``nfreq``) -- first grid sample index of each\n"
        "  channel's pulse.\n"
        "- ``freq_nt`` (int array, length ``nfreq``) -- number of samples in each channel's\n"
        "  pulse; 0 for an inactive channel.\n"
        "- ``freq_sd_off`` (int array, length ``nfreq``) -- offset of each channel's samples\n"
        "  into ``sparse_data``, i.e. the exclusive prefix sum of ``freq_nt``.\n"
        "- ``sparse_data`` (float array, length ``sum(freq_nt)``) -- the packed samples, with\n"
        "  the spectral weighting and the SNR normalization already applied.\n"
        "- ``it_start``, ``it_end`` (int) -- grid range bracketing the pulse, namely\n"
        "  ``min(freq_it0)`` and ``max(freq_it0 + freq_nt)`` over the ACTIVE channels. Every\n"
        "  channel, inactive ones included, satisfies\n"
        "  ``it_start <= freq_it0 <= freq_it0 + freq_nt <= it_end``.\n"
        "- ``nfreq`` (int) -- number of frequency channels, ``len(freq_edges_MHz) - 1``.\n"
        "- ``freq_lo_MHz``, ``freq_hi_MHz`` (float) -- first and last entries of\n"
        "  ``freq_edges_MHz``.\n")

        .def(py::init([](double dm, double sm, double intrinsic_width, double spectral_index,
                         double undispersed_arrival_time_sec, double time_sample_ms, double snr,
                         const Array<double> &freq_edges_MHz, const Array<double> &freq_variances,
                         double subband_freq_lo_MHz, double subband_freq_hi_MHz, long internal_nt) {
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
                 p.subband_freq_lo_MHz = subband_freq_lo_MHz;
                 p.subband_freq_hi_MHz = subband_freq_hi_MHz;
                 p.internal_nt = internal_nt;
                 return new SinglePulse(p);
             }),
             // Argument order matches the C++ Params members. The trailing args with sensible defaults
             // (subband_*, internal_nt) are optional; the required arrays sit mid-struct, so everything
             // before them (including snr) must be a required argument.
             py::arg("dm"), py::arg("sm"), py::arg("intrinsic_width"), py::arg("spectral_index"),
             py::arg("undispersed_arrival_time_sec"), py::arg("time_sample_ms"), py::arg("snr"),
             py::arg("freq_edges_MHz"), py::arg("freq_variances"),
             py::arg("subband_freq_lo_MHz") = 0.0, py::arg("subband_freq_hi_MHz") = 1.0e9,
             py::arg("internal_nt") = 1024,
             // Per-channel inverse FFTs + interpolation; body is pure C++ (copies
             // pre-converted Arrays into Params).
             py::call_guard<py::gil_scoped_release>(),
             // Sphinx does not render this docstring (autoclass_content is 'class', and
             // autodoc cannot introspect a pybind __init__), which is why the class
             // docstring carries the 'Constructor::' block. Kept for help() / repl users.
             "Precompute one pulse. See the class docstring: every argument is documented\n"
             "there, in the attribute list, since each is readable afterwards as a same-named\n"
             "attribute.")

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
        .def_property_readonly("subband_freq_lo_MHz", [](const SinglePulse &s) { return s.params.subband_freq_lo_MHz; })
        .def_property_readonly("subband_freq_hi_MHz", [](const SinglePulse &s) { return s.params.subband_freq_hi_MHz; })

        // Precomputed sparse representation.
        .def_readonly("freq_it0", &SinglePulse::freq_it0)
        .def_readonly("freq_nt", &SinglePulse::freq_nt)
        .def_readonly("freq_sd_off", &SinglePulse::freq_sd_off)
        .def_readonly("sparse_data", &SinglePulse::sparse_data)
        .def_readonly("it_start", &SinglePulse::it_start)
        .def_readonly("it_end", &SinglePulse::it_end)

        // Derived read-only attributes, computed from freq_edges_MHz.
        .def_property_readonly("nfreq", [](const SinglePulse &s) { return s.params.freq_edges_MHz.size - 1; })
        .def_property_readonly("freq_lo_MHz", [](const SinglePulse &s) { return s.params.freq_edges_MHz.data[0]; })
        .def_property_readonly("freq_hi_MHz", [](const SinglePulse &s) { return s.params.freq_edges_MHz.data[s.params.freq_edges_MHz.size - 1]; })

        .def("add_to_timestream", &SinglePulse::add_to_timestream,
             py::arg("out"), py::arg("out_it0"), py::arg("weight") = 1.0f,
             py::call_guard<py::gil_scoped_release>(),   // O(pulse samples) CPU scatter
             "Add the pulse into a dense 2-d array of (frequency, time) samples, in place.\n"
             "\n"
             "``out`` must be a host (CPU) float32 array of shape ``(nfreq, out_nt)``, ordered low\n"
             "to high in frequency, with contiguous time samples (``out.strides[1] == 1``). Its\n"
             "column ``it`` represents grid sample index ``out_it0 + it``, so ``out`` spans sample\n"
             "indices ``[out_it0, out_it0 + out_nt)``.\n"
             "\n"
             "Args:\n"
             "    out: array to add the pulse into, modified in place.\n"
             "    out_it0: grid sample index of ``out``'s first column. May be negative.\n"
             "    weight: scale factor applied to the pulse as it is added.\n"
             "\n"
             "Raises:\n"
             "    RuntimeError: if ``out`` does not span the pulse's full time range, i.e. unless\n"
             "        ``out_it0 <= it_start`` and ``out_it0 + out_nt >= it_end``.")

        .def("shift_samples", &SinglePulse::shift_samples, py::arg("delta_it"),
             "Shift the pulse forward in time by ``delta_it`` samples (may be negative).\n"
             "\n"
             "Adds ``delta_it`` to every ``freq_it0`` and to ``it_start`` / ``it_end``, and adds\n"
             "``1e-3 * delta_it * time_sample_ms`` to ``undispersed_arrival_time_sec``. The sample\n"
             "VALUES are untouched (``sparse_data``, ``freq_nt`` and ``freq_sd_off`` are all\n"
             "unchanged), so this is a cheap way to reuse one precomputed pulse at many arrival\n"
             "times.\n"
             "\n"
             "Args:\n"
             "    delta_it: number of grid samples to shift by.")

        .def("__repr__", &SinglePulse::str)
    ;
}

}  // namespace pirate
