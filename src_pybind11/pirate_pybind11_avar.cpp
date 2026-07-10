// Python bindings for the analytic-variance ("avar") classes (pirate_frb.fast_avar subpackage).
// These are C++ ports of pirate_frb/slow_avar; see pirate_pybind11.cpp for the main module.

#define PY_ARRAY_UNIQUE_SYMBOL PyArray_API_pirate
#define NO_IMPORT_ARRAY  // Secondary file: don't call _import_array()
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <pybind11/stl.h>
#include <ksgpu/pybind11.hpp>

#include "../include/pirate/SparseTile.hpp"
#include "../include/pirate/PfVariance.hpp"
#include "../include/pirate/DedispersionPlan.hpp"

using namespace std;
using namespace ksgpu;
using namespace pirate;
namespace py = pybind11;


namespace pirate {

void register_avar_bindings(pybind11::module &m)
{
    // ---------------------------------------------------------------------------- SparseTile

    py::class_<SparseTile>(m, "SparseTile")
        .def(py::init([](long r, long k, long f0, long nf, long nt, long dbits,
                         const Array<double> &data, const Array<long> &tshifts, long t0, double scale) {
                 xassert(data.is_fully_contiguous());
                 xassert(tshifts.is_fully_contiguous());
                 return new SparseTile(r, k, f0, nf, nt, dbits, data.data, tshifts.data, t0, scale);
             }),
             py::arg("r"), py::arg("k"), py::arg("f0"), py::arg("nf"), py::arg("nt"),
             py::arg("dbits"), py::arg("data"), py::arg("tshifts"), py::arg("t0") = 0, py::arg("scale") = 1.0)
        .def_readonly("r", &SparseTile::r)
        .def_readonly("k", &SparseTile::k)
        .def_readonly("f0", &SparseTile::f0)
        .def_readonly("nf", &SparseTile::nf)
        .def_readonly("nt", &SparseTile::nt)
        .def_readonly("dbits", &SparseTile::dbits)
        .def_readonly("t0", &SparseTile::t0)
        .def_readonly("scale", &SparseTile::scale)
        .def_property_readonly("tshifts", [](const SparseTile &self) {
            Array<long> out({self.k}, af_rhost);
            for (long j = 0; j < self.k; j++)
                out.data[j] = self.tshifts[j];
            return out;
        })
        .def("unpack", &SparseTile::unpack, py::arg("ntime"))
    ;

    // ------------------------------------------------------------------------ SparseTileTriple

    py::class_<SparseTileTriple>(m, "SparseTileTriple")
        .def_static("make_tree_gridding_output", [](const Array<double> &cm, long ifreq) {
                 xassert(cm.is_fully_contiguous());
                 return SparseTileTriple::make_tree_gridding_output(cm.data, cm.size, ifreq);
             }, py::arg("channel_map"), py::arg("ifreq"))
        .def_readonly("r", &SparseTileTriple::r)
        .def_readonly("k", &SparseTileTriple::k)
        .def_readonly("f0", &SparseTileTriple::f0)
        .def_readonly("nf", &SparseTileTriple::nf)
        .def_readonly("ntiles", &SparseTileTriple::ntiles)
        .def("iterate", &SparseTileTriple::iterate)
        .def("get_singleton", [](const SparseTileTriple &self, long f) -> py::object {
            SparseTile out;
            if (self.get_singleton(f, out))
                return py::cast(out);
            return py::none();
        }, py::arg("f"))
        .def("unpack", &SparseTileTriple::unpack, py::arg("ntime"))
    ;

    // ----------------------------------------------------------------------- PfVarianceConvolver

    py::class_<PfVarianceConvolver>(m, "PfVarianceConvolver")
        .def(py::init<>())
        .def_readonly("Pmax", &PfVarianceConvolver::Pmax)
        .def_readonly("Tmax_last", &PfVarianceConvolver::Tmax_last)
        .def_property_readonly("Tmax", [](const PfVarianceConvolver &self) { return self.Tmax; })
        .def_property_readonly("A", [](const PfVarianceConvolver &self) {
            Array<double> a({self.Pmax, self.Tmax_last}, af_rhost);
            memcpy(a.data, self.A.data(), self.A.size() * sizeof(double));
            return a;
        })
        .def("variance", [](const PfVarianceConvolver &self, const Array<double> &x, long P) {
            xassert(x.ndim == 2 && x.is_fully_contiguous());
            long S = x.shape[0], nt = x.shape[1];
            Array<double> out({S, P}, af_rhost);
            self.variance(x.data, S, nt, P, out.data);
            return out;
        }, py::arg("x"), py::arg("P"), py::call_guard<py::gil_scoped_release>())
    ;

    // ------------------------------------------------------------------------------ PfVariance

    py::class_<PfVariance>(m, "PfVariance")
        .def(py::init<long, long>(), py::arg("rank"), py::arg("P"))
        .def_readonly("rank", &PfVariance::rank)
        .def_readonly("P", &PfVariance::P)
        .def_static("from_tile", &PfVariance::from_tile,
                    py::arg("tile"), py::arg("P"), py::arg("convolver"))
        .def("get_all_dbits", &PfVariance::get_all_dbits)
        .def("add", &PfVariance::add, py::arg("src"), py::arg("upper_half") = false, py::arg("scale") = 1.0)
        .def("unpack", &PfVariance::unpack, py::arg("dbits"))
    ;

    // ------------------------------------------------------------------------ PfAvarApproximation

    py::class_<PfAvarApproximation>(m, "PfAvarApproximation")
        .def(py::init([](std::shared_ptr<DedispersionPlan> plan, const Array<double> &freq_variances) {
                 return new PfAvarApproximation(plan, freq_variances);
             }), py::arg("plan"), py::arg("freq_variances"),
                 py::call_guard<py::gil_scoped_release>())   // seconds of CPU (per-channel sweep)
        .def_readonly("nfreq", &PfAvarApproximation::nfreq)
        .def_readonly("ntrees", &PfAvarApproximation::ntrees)
        .def_property_readonly("tree_variance", [](const PfAvarApproximation &self) {
            return self.tree_variance;
        })
        .def_property_readonly("per_tf", [](const PfAvarApproximation &self) {
            return self.per_tf;
        })
    ;
}

}  // namespace pirate
