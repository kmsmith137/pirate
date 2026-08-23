# Low-rank representations of the variance map A of a DedispersionTree (notes/variance_map.tex).
#
# Everything to do with variance maps lives here: the representation, the distance function,
# the brute-force sweep, the analytic (detrender-free) map, the file format, and the
# 'pirate_frb variance_map' CLI. pirate_frb.slow_avar now holds only the older analytic
# (PfAvarExact / TmpVmap) machinery and the SparseTile primitives it is built on. Two things
# here reach into it: detrender_free.py, which is built on the SparseTile primitives, and
# tests.py, which uses PfAvarExact's inner loop as an independent oracle for both the sweep
# and detrender_free.py. Both import from the LEAF modules rather than the slow_avar package,
# to avoid dragging in check_mc / check_approximation.
#
# NOTE THE FILE FORMAT: asdf_io.py's format is not the older one that predates this package.
# The reader refuses an old-format file by name rather than misreading it, and nothing can
# read one any more. It is also at version 2, which holds one entry per PRIMARY tree; a
# version-1 file (one entry per dedispersion tree) is likewise refused by name.
#
# A VarianceMap is a PRIMARY tree's map. An early-trigger tree's map is a subset of its
# parent's ROWS, so it is derived rather than stored -- see VarianceMultiMap, and the
# appendix "Variance maps of a config's trees are row-restrictions of one another" in
# notes/variance_map.tex.
from .distance import YTRUE_FLOOR, f, fprime, AdmissibilityResult, DistanceEstimate
from .VarianceMap import VarianceMap, coarse_grain_vector
from .VarianceMultiMap import VarianceMultiMap, restrict_fine_vector
from .lp import (LpConfig, solve_covering_lps, q_step, w_step, covering_lp_data,
                 majorizer_weights, repair_rows, repair_cols, repair_additive,
                 fix_nonneg, apply_repair, violation_stats, check_nonneg,
                 blocking_is_exact,
                 solve_cover_lp, solve_cover_lp_cuts)
from .brute_force import compute_variance_multimap
from .detrender_free import SdMatrix, build_sd_matrices, compute_detrender_free_base_map
from .basis import (basis_svd, basis_envelope_column, basis_greedy_envelope,
                    greedy_envelope_tree, basis_pivoted_qr, basis_random, svd_init,
                    spectrum_effective_rank, shape_cover_statistic)
from .report import (row_dict, frontier, format_table, format_row, save_json, load_json)
