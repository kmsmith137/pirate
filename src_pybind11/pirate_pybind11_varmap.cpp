// Python bindings for the variance-map C++ ports (pirate_frb.fast_varmap subpackage).
// These are ports of pirate_frb/varmap; see pirate_pybind11.cpp for the main module.
//
// A DELIBERATELY THIN SURFACE. Only what pirate_frb/fast_varmap/test_fast_varmap.py and
// pirate_frb/__main__.py actually call is bound: SparseTile::predict_dbits (a static, so the
// class is bound as its holder and nothing else), the SparseTileTriple gridding+iterate
// sweep, all of PfVarianceConvolver, and the two compute_detrender_free_* functions. The
// SparseTile constructor / accessors / unpack() and SparseTileTriple::get_singleton() were
// bound for tests that no longer exist, and are not bound any more -- add them back if a
// python caller needs them, rather than keeping unreachable bindings alive.

#define PY_ARRAY_UNIQUE_SYMBOL PyArray_API_pirate
#define NO_IMPORT_ARRAY  // Secondary file: don't call _import_array()
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <pybind11/stl.h>
#include <ksgpu/pybind11.hpp>

#include "../include/pirate/varmap.hpp"
#include "../include/pirate/DedispersionPlan.hpp"
#include "../include/pirate/DedispersionConfig.hpp"

using namespace std;
using namespace ksgpu;
using namespace pirate;
namespace py = pybind11;


namespace pirate {

void register_varmap_bindings(pybind11::module &m)
{
    // ---------------------------------------------------------------------------- SparseTile

    // Bound only as the holder for the predict_dbits static; no python caller constructs,
    // reads or unpacks a SparseTile.
    py::class_<SparseTile>(m, "SparseTile")
        .def_static("predict_dbits", &SparseTile::predict_dbits,
                    py::arg("kmax"), py::arg("f0"), py::arg("nf"))
    ;

    // ------------------------------------------------------------------------ SparseTileTriple

    py::class_<SparseTileTriple>(m, "SparseTileTriple")
        .def_static("make_tree_gridding_output", [](const Array<double> &cm, long ifreq,
                                                    long flo, long fhi) {
                 // Validated here rather than in the C++: cm.data is dereferenced on the host,
                 // so a cupy or non-contiguous array would segfault.
                 xassert(cm.on_host());
                 xassert(cm.is_fully_contiguous());
                 return SparseTileTriple::make_tree_gridding_output(cm.data, cm.size, ifreq, flo, fhi);
             }, py::arg("channel_map"), py::arg("ifreq"),
                // The python spells the no-upper-clip default 'None'; the C++ spells it -1, and
                // this binding follows the C++ (a python caller who wants no clip just omits it).
                py::arg("flo") = 0, py::arg("fhi") = -1)
        .def_readonly("r", &SparseTileTriple::r)
        .def_readonly("k", &SparseTileTriple::k)
        .def_readonly("f0", &SparseTileTriple::f0)
        .def_readonly("nf", &SparseTileTriple::nf)
        .def("iterate", &SparseTileTriple::iterate)
        .def("unpack", &SparseTileTriple::unpack, py::arg("ntime"))
    ;

    // ----------------------------------------------------------------------- PfVarianceConvolver

    py::class_<PfVarianceConvolver>(m, "PfVarianceConvolver")
        .def(py::init<>())
        .def_readonly("Pmax", &PfVarianceConvolver::Pmax)
        .def_readonly("Tmax_last", &PfVarianceConvolver::Tmax_last)
        .def_property_readonly("Tmax", [](const PfVarianceConvolver &self) { return self.Tmax; })
        .def_property_readonly("A", [](const PfVarianceConvolver &self) {
            Array<double> a({self.Pmax, self.Tmax_last}, af_uhost);
            memcpy(a.data, self.A.data(), self.A.size() * sizeof(double));
            return a;
        })
        .def("variance", [](const PfVarianceConvolver &self, const Array<double> &x, long P) {
            xassert(x.on_host());   // x.data is dereferenced on the host
            xassert(x.ndim == 2 && x.is_fully_contiguous());
            long S = x.shape[0], nt = x.shape[1];
            Array<double> out({S, P}, af_uhost);
            self.variance(x.data, S, nt, P, out.data);
            return out;
        }, py::arg("x"), py::arg("P"), py::call_guard<py::gil_scoped_release>())
    ;

    // ------------------------------------------- detrender-free variance vectors

    // The on_host() / is_fully_contiguous() checks belong here rather than in the C++ functions:
    // the host code dereferences freq_variances.data, so a cupy array would segfault. (SdPlan
    // repeats them, since it is also reachable from C++.)
    //
    // gil_scoped_release: these are the longest pure-CPU calls in this module -- seconds at CHORD
    // scale -- and neither touches a python object once it has the array.

    m.def("compute_detrender_free_varfine",
          [](const DedispersionPlan &plan, const Array<double> &freq_variances) {
              xassert(freq_variances.on_host());
              xassert(freq_variances.is_fully_contiguous());
              return compute_detrender_free_varfine(plan, freq_variances);
          }, py::arg("plan"), py::arg("freq_variances"),
             py::call_guard<py::gil_scoped_release>());

    m.def("compute_detrender_free_varcoarse",
          [](const DedispersionPlan &plan, const Array<double> &freq_variances) {
              xassert(freq_variances.on_host());
              xassert(freq_variances.is_fully_contiguous());
              return compute_detrender_free_varcoarse(plan, freq_variances);
          }, py::arg("plan"), py::arg("freq_variances"),
             py::call_guard<py::gil_scoped_release>());
}

}  // namespace pirate
