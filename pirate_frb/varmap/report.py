"""Reporting for varmap: the experiment record, the rank frontier, and the table/json helpers.

One contract is what this module exists to enforce: EVERY EXPERIMENT SHOULD REPORT ITS
NUMBERS THROUGH ONE PLACE, so that results taken months apart are comparable. Its companion
-- that the distance function must not be changed silently -- lives on varmap/distance.py,
where D is defined.

The record is a plain DICT, and that is deliberate rather than lazy. A frontier is then a list
of dicts, which prints as a table and serializes to json with no extra machinery, and an
experiment that wants to record something nobody anticipated just puts it in 'extra'. A result
class would buy type checking and cost exactly the property that makes these files readable
five months later.

WHAT IS AND IS NOT COMPUTED HERE. row_dict() reads the map; it does not score it. D comes in as
an argument because the caller decides how it was obtained -- get_distance() on an admissible
map, estimate_distance() on a sampled one, the inflated distance on a map that failed. The
elementwise half of D (is the map really admissible?) likewise comes in as an optional
AdmissibilityResult, because for anything built by a Q-step it is redundant: admissibility is
the covering constraint that step solves, enforced exactly by its repair. Paying for a full
elementwise pass per row of a results table would be the single most expensive habit available
here, so it is opt-in.
"""

import json
import time

import numpy as np


# The columns a results table shows by default, in this order, when the caller does not say.
# Anything else in the record still reaches json; this is only what prints.
_COLUMNS = ('name', 'K', 'factor_rank', 'D', 'max_r', 'max_diff', 'D_inflated',
            'admissible', 'algo_seconds')


def _plain(v):
    """A numpy scalar as a python one, so that json.dump() does not choke on a record."""
    return v.item() if isinstance(v, np.generic) else v


def row_dict(vmap, D, *, name=None, adm=None, extra=None, apply_cost=True):
    """The experiment record for one scored map, as a plain dict.

    Every field but 'D' is read off the map itself -- the schema of an experiment record is a
    reporting concern, and the map already carries what it needs.

    Parameters
    ----------
    vmap : VarianceMap
        The approximation being reported.
    D : float or None
        Its distance, from get_distance() (or estimate_distance(), or the inflated distance --
        this function does not care, and does not check, which is why the caller passes it).
    adm : AdmissibilityResult or None
        The result of measure_admissibility(), when one was taken. It contributes 'max_r',
        'max_diff', 'argmax_r', 'nviol', 'viol_frac' and -- if it was taken with inflate=True
        -- 'inflation' and 'D_inflated'. Without it the record's 'admissible' is the map's own
        FLAG, which is a claim rather than a measurement; with it, it is the measurement.
    extra : dict or None
        Merged in last, for experiment bookkeeping (config name, timings, git hash). Numpy
        scalars in it are converted, so it can be filled straight from a step's info dict.
    apply_cost : bool
        Include apply_cost(), which counts the nonzeros of Q and is therefore an O(nbeta*K)
        pass. Once per row of a results table is what it is for; pass False in a loop.

    Note 'factor_rank' is deliberately not called 'rank': it is the FACTORIZATION's rank, an
    upper bound on the numerical rank of the product, and not the same thing.
    """

    r = dict(name=name,
             itree=int(vmap.itree),
             factor_rank=(None if (vmap.factor_rank is None) else int(vmap.factor_rank)),
             is_factored=bool(vmap.is_factored),
             nalpha=int(vmap.nalpha),
             nbeta=int(vmap.nbeta),
             nfreq=int(vmap.nfreq),
             is_coarse_grained=bool(vmap.is_coarse_grained),
             L=(None if (vmap.L is None) else int(vmap.L)),
             D=(None if (D is None) else float(D)),
             admissible=bool(vmap.is_admissible))

    r['nscored'] = int(vmap.nscored) if (vmap.y_true is not None) else None
    if apply_cost:
        r['apply_cost'] = int(vmap.apply_cost())

    if adm is not None:
        # The measurement wins over the flag: that is the whole reason for taking one.
        r.update(admissible=bool(adm.admissible),
                 max_r=float(adm.max_r),
                 max_diff=float(adm.max_diff),
                 argmax_r=[int(adm.argmax_r[0]), int(adm.argmax_r[1])],
                 nviol=int(adm.nviol),
                 viol_frac=float(adm.viol_frac))
        if adm.inflation is not None:
            r.update(inflation=float(adm.inflation), D_inflated=float(adm.D_inflated))

    if extra:
        r.update({str(k): _plain(v) for (k, v) in extra.items()})
    return r


def frontier(ref, algorithm, ranks, *, name=None, measure=False, inflate=False, extra=None,
             **kwargs):
    """The rank-versus-distance frontier: call ``algorithm(ref, K)`` for each K and report the
    map it returns. Returns a list of records, one per K.

    'algorithm' takes the reference map and a rank and returns a VarianceMap -- e.g.
    ``lambda ref, K: basis.svd_init(ref, K)``. An algorithm that naturally produces every rank
    in one run (agglomerative merging, say) should precompute and close over the result, since
    this function assumes nothing about how the rank is reached.

    Each record gains 'K' (the rank ASKED for, which need not be the 'factor_rank' delivered)
    and 'algo_seconds'.

    'measure' runs measure_admissibility() against 'ref' as well, which is a full elementwise
    pass and is redundant for anything built by a Q-step -- so it is off by default. Turn it on
    for a basis whose admissibility is a question rather than a theorem, and note that it is
    then also what rescues the row: a map that fails is reported with D = inf, and with
    'inflate' also on, with the distance it would have after being scaled up to admissibility,
    which is what distinguishes "a 2% rescale fixes this" from "hopeless".

    Extra kwargs go to row_dict().
    """

    rows = []
    for K in ranks:
        t0 = time.time()
        vmap = algorithm(ref, int(K))
        dt = time.time() - t0

        adm = vmap.measure_admissibility(ref, inflate=inflate) if measure else None
        scoreable = adm.admissible if (adm is not None) else vmap.is_admissible
        D = (adm.vmap if (adm is not None) else vmap).get_distance() if scoreable else np.inf

        r = row_dict(vmap, D, name=name, adm=adm,
                     extra=dict(extra or {}, K=int(K), algo_seconds=dt), **kwargs)
        rows.append(r)
    return rows


####################################   formatting   ####################################


def format_table(rows, columns=None):
    """A markdown table of the records returned by row_dict() / frontier()."""

    if len(rows) == 0:
        return '(no rows)'
    columns = columns if (columns is not None) else [c for c in _COLUMNS if c in rows[0]]

    def fmt(v):
        if isinstance(v, float):
            return 'inf' if np.isinf(v) else f'{v:.6g}'
        return str(v)

    cells = [[fmt(r.get(c, '')) for c in columns] for r in rows]
    width = [max(len(str(c)), max(len(row[i]) for row in cells))
             for i, c in enumerate(columns)]

    out = ['| ' + ' | '.join(str(c).ljust(width[i]) for i, c in enumerate(columns)) + ' |',
           '|' + '|'.join('-' * (width[i]+2) for i in range(len(columns))) + '|']
    out += ['| ' + ' | '.join(row[i].ljust(width[i]) for i in range(len(columns))) + ' |'
            for row in cells]
    return '\n'.join(out)


def format_row(r):
    """One-line summary of a single record.

    The admissibility fields are OPTIONAL: measuring is a separate pass that most rows do not
    pay for, so a row that skipped it prints what it has rather than raising.
    """

    K = r.get('factor_rank')
    out = [f"{r.get('name')}: rank={K}"]
    D = r.get('D')
    out.append('D=inf' if (D is None or not np.isfinite(D)) else f'D={D:.6g}')
    if r.get('max_r') is not None:
        a, f = r.get('argmax_r', (None, None))
        out.append(f"max_r={r['max_r']:.6g} argmax=(row={a},F={f})")
        if r.get('max_diff') is not None:
            out.append(f"max_diff={r['max_diff']:.6g}")
    else:
        out.append(f"admissible={r.get('admissible')} (flag, not measured)")
    if r.get('D_inflated') is not None:
        out.append(f"D_inflated={r['D_inflated']:.6g}")
    return ' '.join(out)


####################################   json   ####################################


def save_json(rows, path):
    """Write records to json. Infinities become the strings 'inf' / '-inf', which is what makes
    the file loadable by anything rather than only by us."""

    def clean(v):
        v = _plain(v)
        if isinstance(v, float) and not np.isfinite(v):
            return 'inf' if (v > 0) else '-inf'
        if isinstance(v, tuple):
            return list(v)
        return v

    with open(path, 'w') as fp:
        json.dump([{k: clean(v) for k, v in r.items()} for r in rows], fp, indent=2)


def load_json(path):
    """Inverse of save_json(): the string infinities become floats again."""

    def restore(v):
        return {'inf': np.inf, '-inf': -np.inf}.get(v, v) if isinstance(v, str) else v

    with open(path) as fp:
        return [{k: restore(v) for k, v in r.items()} for r in json.load(fp)]
