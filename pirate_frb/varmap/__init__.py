# Low-rank representations of the variance map A of a DedispersionTree (notes/variance_map.tex).
#
# This package supersedes the variance-map half of pirate_frb.slow_avar (VarMapDistance,
# varmap_eval, variance_map_io, brute_force*). Those stay in place, unmodified, for as long as
# they are the reference the new code is checked against, so varmap is purely additive until
# that equivalence is established.
#
# Not superseded, and callable from here: slow_avar's SparseTile, PfVariance, TmpVmap and
# check_* modules.
from .distance import YTRUE_FLOOR, f, fprime, AdmissibilityResult, DistanceEstimate
from .VarianceMap import VarianceMap
from .VarianceMultiMap import VarianceMultiMap
from .lp import (LpConfig, solve_covering_lps, q_step, w_step, covering_lp_data,
                 majorizer_weights, repair_rows, repair_cols, repair_additive,
                 fix_nonneg, apply_repair, violation_stats, check_nonneg,
                 blocking_is_exact,
                 solve_cover_lp, solve_cover_lp_cuts)
from .basis import (basis_svd, basis_envelope_column, basis_greedy_envelope,
                    greedy_envelope_tree, basis_pivoted_qr, basis_random, svd_init,
                    spectrum_effective_rank, shape_cover_statistic)
from .report import (row_dict, frontier, format_table, format_row, save_json, load_json)
