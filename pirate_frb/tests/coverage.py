"""Coverage analysis of the randomization utilities the unit tests draw from.

NOT A TEST: nothing here asserts. It is a diagnostic, reached as 'pirate_frb dev coverage',
and it answers one question -- how often does a randomized unit test actually get the
structure it needs? Many tests in this suite draw a config or a shape and REPORT rather than
assert what they covered, precisely because a fair draw misses some cases (notes/unit_tests.md
item 8: a test that asserts it was handed a structure is asserting a property of the draw,
not of the code). This is where those rates live.

EVERY LINE PRINTED HERE HAS A CONSUMER. If you add one, say which test needs the structure;
if a line no longer has a test behind it, delete it.

ORGANIZED BY RANDOMIZATION FUNCTION, NOT BY TEST, and that is the load-bearing choice. A row
is measured on the draws of exactly one named randomizer and is printed under it, so a rate
cannot drift onto a population no test samples -- which is a mistake this file has made
before, reporting a rate measured on one config draw as though it described a test that draws
from another.

THREE KINDS OF ROW, because they fail differently and are fixed in different places:

  rate   a probability over draws. A rate at 0% or 100% on a row that should be neither is
         the failure signal -- the structure has become unreachable, or the "corner" has
         become the only case.
  dist   percentiles of a drawn quantity. This is the kind that catches a cost blowup or a
         starved range, neither of which shows up as a probability.
  enum   a static property of what this build COMPILED -- the kernel registries. Not a draw
         at all, and split out because the fix lives in makefile_helper.py, never in a test.

Every row carries an expected band. The bands are deliberately WIDE: they are tripwires
against a structure becoming unreachable, not assertions about a number. Each was set from a
measured value, then widened. A row outside its band is not necessarily a bug -- it is a
number that has moved far enough to be worth a look. The run ends with 'NN rows, M outside
band' and, when M > 0, NAMES the offending rows with their section and expected band, so that
nobody has to scroll back through a few hundred lines to find which one moved. That count is
the hook for asserting on this some day.

ONE SECTION NEEDS A GPU, AND ONLY ONE. Every plan this file builds itself passes
mega_ringbuf=False, gpu_kernels=False, and the registry section reads static key tables
rather than instantiating kernels -- so the report runs on a laptop. The exception is the
sweep-loop section, whose whole subject is a rejection rate defined by a cost model evaluated
on a real _SweepGeometry, and that builds a full plan. It is skipped, with a message, when
there is no device.

WHAT IS NOT HERE. The per-kernel-test shape draws -- TestInstanceLDS::make_random() and its
siblings behind test --gldk, --gtgk, --grck, --gdqk -- are not reported, because each is still
written inline in its own test_random() rather than being a function this file could call.
random_kernel_shape() in utils.hpp is the shared half of that draw and could be sampled, but
sampling it in isolation would report a budget the callers each divide differently, which is
not the number anyone needs. notes/unit_tests.md item 8 anticipates the refactor: "You may
need to refactor your randomization logic into its own function, so that it is callable from
coverage."
"""

import time
from collections import Counter

import numpy as np

from ..utils import atomic_print, integer_log2, print_separator


####################################################################################################
#
# Report primitives.


class _Report:
    """Accumulates rows, prints them as it goes, and counts how many left their band.

    Printing as we go (rather than at the end) matters: the expensive sections take tens of
    seconds, and a report that appears all at once looks like a hang.
    """

    def __init__(self):
        self.nrows = 0
        # (section title, kind, label, measured value, band) for every row that left its
        # band. The section title is part of the identity: section A prints the same ten row
        # labels under four different make_random() settings, so a label alone would not say
        # which one moved.
        self.bad = []
        self.section_title = '(no section)'
        self.t0 = time.time()

    def section(self, title, subtitle=None, consumer=None):
        self.section_title = title
        print_separator(title)
        if subtitle is not None:
            atomic_print(f'  {subtitle}')
        if consumer is not None:
            atomic_print(f'  consumers: {consumer}')
        atomic_print('')

    def rate(self, label, count, n, band, consumer):
        """A probability, printed as a percentage with its (count/n). 'band' is (lo, hi) in
        percent; a rate outside it is flagged."""

        pct = (100.0 * count / n) if (n > 0) else float('nan')
        lo, hi = band
        ok = (n > 0) and (lo <= pct <= hi)
        self._emit('rate', label, f'{pct:5.1f}% ({count}/{n})', f'[{lo:g},{hi:g}]', ok, consumer)

    def dist(self, label, values, band, consumer, fmt='{:.3g}'):
        """Percentiles of a drawn quantity. 'band' is (stat, lo, hi) where 'stat' is one
        of 'p10', 'p50', 'p90', 'max', 'min' -- the one percentile worth a tripwire. Reporting the whole
        spread but banding one number keeps the row readable."""

        v = np.asarray(list(values), dtype=float)
        if v.size == 0:
            self._emit('dist', label, 'no draws', '--', False, consumer)
            return
        p10, p50, p90, vmax = np.percentile(v, [10, 50, 90]).tolist() + [float(v.max())]
        stat, lo, hi = band
        got = {'p10': p10, 'p50': p50, 'p90': p90,
               'max': vmax, 'min': float(v.min())}[stat]
        text = (f'p10 {fmt.format(p10)}  p50 {fmt.format(p50)}'
                f'  p90 {fmt.format(p90)}  max {fmt.format(vmax)}')
        self._emit('dist', label, text, f'{stat} in [{lo:g},{hi:g}]', lo <= got <= hi, consumer)

    def enum(self, label, value, band, consumer, text=None):
        """A count or a set size that is a property of the build. 'band' is (lo, hi)."""

        lo, hi = band
        self._emit('enum', label, text if (text is not None) else str(value),
                   f'[{lo:g},{hi:g}]', lo <= value <= hi, consumer)

    def note(self, text):
        """A histogram or similar, printed for context. Carries no band and is not counted:
        it is here to EXPLAIN a row above it when that row moves."""

        atomic_print(f'          {text}')

    def _emit(self, kind, label, value, band, ok, consumer):
        self.nrows += 1
        if not ok:
            self.bad.append((self.section_title, kind, label, value.strip(), band))
        atomic_print(f'  {kind:4s}  {label:<52s} {value:<30s} {band:<15s}'
                     f' {"ok" if ok else "OUT":3s}  {consumer}')

    def finish(self):
        atomic_print('')
        print_separator('summary')
        atomic_print(f'  {self.nrows} rows, {len(self.bad)} outside band'
                     f'   ({time.time() - self.t0:.1f} s)\n')
        if not self.bad:
            return

        # NAME THEM. The rows scroll past by the hundred, so a bare count means scrolling back
        # to find which one moved -- and in section A the same label appears four times, under
        # four different settings, so the section has to be named too.
        atomic_print('  Outside band:\n')
        last = None
        for (section, kind, label, value, band) in self.bad:
            if section != last:
                atomic_print(f'    {section}')
                last = section
            atomic_print(f'      {kind:4s}  {label:<52s} {value}')
            atomic_print(f'            expected {band}')
        atomic_print('\n  A row outside its band is a number that has MOVED, not necessarily a\n'
                     '  bug. Read it against the consumer named on the row: the question is\n'
                     '  always whether that test still gets the structure it needs.\n')


def _hist(values, top=None):
    c = Counter(values)
    items = sorted(c.items())
    if (top is not None) and (len(items) > top):
        items = sorted(c.items(), key=lambda kv: -kv[1])[:top]
        items = sorted(items)
        return dict(items) | {'...': f'{len(c) - top} more'}
    return dict(items)


def _plan(config):
    """A "minimal" DedispersionPlan: every section here needs the trees and nothing else, and
    a minimal plan needs no GPU."""

    from ..pirate_pybind11 import DedispersionPlan
    return DedispersionPlan(config, mega_ringbuf=False, gpu_kernels=False)


####################################################################################################
#
# Section A: DedispersionConfig::make_random(), at the settings its callers actually use.


# The geometry rows below are the same at every setting, and that is the point: a setting
# that starves one of them shows up as the SAME row reading differently, rather than as a row
# someone forgot to add. WHAT EACH ROW GATES, once, here rather than repeated in every row:
#
#   npri > 1              The multi-tree paths. test_asdf_io's duplicate-gamma tripwire and
#                         its "covers EVERY primary tree" rejection, test_multimap's
#                         short-list rejection, test_max_width_monotone's chains.
#   max_width varies      Makes the profile restriction P_gamma < P_0 something other than a
#                         no-op (test_multimap_vs_base, test_varfine). On the gpu_valid path
#                         it needs the cdd2 registry to stock two Wmax for the DOWNSAMPLED
#                         tree's key, so it is partly a property of what this build compiled.
#   early triggers        Proposition 1: the early-trigger trees, restrict_fine_vector(),
#                         apply_fine()'s expansion (test_varfine, test_apply_restriction).
#   R == 0                Degenerate subband geometry, N = M = 1, no coarse DM axis to speak
#                         of (test_multimap_vs_sweep, test_base_varmap_coarse).
#   nfreq < 2^r           The production-like regime -- chord_sb2_et.yml grids 28160 channels
#                         onto 65536 tree-freqs -- and the one that widens footprints.
#   N > 1                 D is a mean over FINE rows, so a plain mean over groups is biased
#                         only when the group sizes differ; and the greedy merge's size
#                         weighting differs from group-blind only then (test_estimate_distance,
#                         test_basis_constructors).
#   K > 0                 The extra-DM-bit path. Nothing in varmap READS K, but K > 0 is
#                         precisely where 2^(r-R) and ndm_out diverge, so it is the only case
#                         where a row count taken from the wrong one is visible
#                         (test_multimap_vs_sweep). For test_decode_argmax it is sharper
#                         still: the argmax token's 'mu' byte is a no-op at K == 0, so a run
#                         that draws no K > 0 tree says nothing about the token format.
#   rank over bound       toplevel_tree_rank > the max_toplevel_rank that was asked for. Was
#                         possible for ODD caps until the clamp landed, since the cap entered
#                         only through max_stage2_rank = (cap+1)/2 while the toplevel rank was
#                         drawn up to 2*dd_rank. Must be 0 everywhere.
#   dense map MiB         Cost tripwire. nalpha * nfreq * 8 bytes for the parent tree, the
#                         array the varmap tests actually allocate.
#   distinct sbc          The tripwire on the cdd2 registry: it was 3 before (4,2,1) was
#                         stocked, which is why the non-contiguous multiplet-map rate under
#                         section C was zero.
#
# BANDS ARE PER SETTING, because the settings reach genuinely different populations and a
# band copied across them would be meaningless. Where a band admits 0%, the setting simply
# does not reach that structure and no test drawing from it needs to -- see _SETTING_NOTES.
_DEFAULT_BANDS = {
    'npri':    (20, 90),
    'maxw':    (5, 70),
    'early':   (5, 80),
    'r0':      (0, 70),
    'wide':    (5, 80),
    'nsub':    (10, 95),
    'K':       (10, 100),
    'f32':     (10, 100),
    'mib':     (0.0, 512.0),
    'sbc':     (2, 200),
}


def _draw_config_rows(rep, configs, consumer, bands=None):
    b = dict(_DEFAULT_BANDS, **(bands or {}))
    n = len(configs)

    npri, ranks, wtds, mw, pfr = [], [], [], [], []
    n_vary = n_early = n_r0 = n_wide = n_multi = n_f32 = n_K = 0
    mib, sbc = [], set()

    for c in configs:
        pts = c.primary_trees
        npri.append(int(c.num_primary_trees))
        ranks.append(int(c.toplevel_tree_rank))
        wtds.extend(int(pt.wt_time_downsampling) for pt in pts)
        mw.extend(int(pt.max_width) for pt in pts)
        n_vary += int(len(set(int(pt.max_width) for pt in pts)) > 1)
        n_early += int(max(int(pt.num_early_triggers) for pt in pts) > 0)
        n_f32 += int(np.dtype(c.dtype) == np.float32)
        sbc.add(tuple(int(x) for x in c.frequency_subband_counts))

        plan = _plan(c)
        trees = plan.trees
        t0 = trees[int(plan.dedispersion_tree_index(0, 0))]
        fs = t0.frequency_subbands
        R, r = int(fs.pf_rank), int(t0.tree_rank)
        pfr.append(R)
        nfreq = int(c.get_total_nfreq())
        n_r0 += int(R == 0)
        n_wide += int(nfreq < (1 << r))
        n_multi += int(int(fs.N) > 1)
        # K = the peak-finder's extra-DM rank, from dm_downsampling = 2^(pf_rank + K).
        n_K += int(any((integer_log2(t.dm_downsampling)
                        - int(t.frequency_subbands.pf_rank)) > 0 for t in trees))
        # nalpha as varmap.tests._nalpha_of() computes it, times nfreq, times 8 bytes.
        nalpha = (1 << (r - R)) * int(fs.M) * int(t0.nprofiles)
        mib.append(nalpha * nfreq * 8.0 / 2**20)

    rep.rate('npri > 1', sum(1 for x in npri if x > 1), n, b['npri'], consumer)
    rep.rate('max_width varies across primary trees', n_vary, n, b['maxw'], consumer)
    rep.rate('some primary tree has early triggers', n_early, n, b['early'], consumer)
    rep.rate('R == 0 (N = M = 1)', n_r0, n, b['r0'], consumer)
    rep.rate('nfreq < 2^r (wide footprints)', n_wide, n, b['wide'], consumer)
    rep.rate('N > 1 (coarse-graining groups differ in size)', n_multi, n, b['nsub'], consumer)
    rep.rate('K > 0 in some tree (extra-DM bits)', n_K, n, b['K'], consumer)
    rep.rate('dtype is float32', n_f32, n, b['f32'], consumer)
    rep.dist('implied dense map, MiB', mib, ('p90',) + b['mib'], consumer)
    rep.enum('distinct subband_counts drawn', len(sbc), b['sbc'], consumer)

    rep.note(f'npri {_hist(npri)}   tree pf_rank {_hist(pfr)}'
             f'   config pf_rank {_hist([len(x) - 1 for x in sbc])} over the'
             f' {len(sbc)} distinct vectors   toplevel_tree_rank {_hist(ranks)}')
    rep.note(f'max_width {_hist(mw)}   wt_time_downsampling {_hist(wtds, top=6)}')


# What is KNOWN to be unreachable at a setting, and why. A band that admits 0% needs one of
# these lines behind it, or it is just a band nobody set.
_SETTING_NOTES = {
    'novalid': [
        'R == 0 IS UNREACHABLE HERE, and the tree-level pf_rank is 1 or 2 on every draw.',
        'That is the documented complement of the gpu_valid=True path, not a',
        'defect: varmap._random_config() draws gpu_valid PER CONFIG precisely because the two',
        'settings reach disjoint corners -- R == 0 only at True, subband_counts[0] == 0 and',
        'R == 3 only at False. See its docstring. Section C reports the mixture.',
    ],
    'loopback': [
        'NO EARLY TRIGGERS AT ALL. max_toplevel_rank=6 pins the toplevel',
        'rank at 5 or 6 against a stage-1 dd_rank of 3, leaving no room for one. The loopback',
        'tests do not need one -- they check assembly and transport, not the tree -- but it',
        'does mean test --net and test --serv never carry an early-trigger tree, and neither',
        'does anything downstream of them.',
    ],
}


# (name, note key, make_random kwargs, full consumer for the header, short per-row tag, bands)
_MAKE_RANDOM_SETTINGS = [
    ('defaults', None, dict(),
     'test --dd (GpuDedisperser::test_random), show random_config', 'test --dd',
     dict()),
    ('gpu_valid=False, max_toplevel_rank=8, max_early_triggers=4', 'novalid',
     dict(gpu_valid=False, max_toplevel_rank=8, max_early_triggers=4),
     'test --dd (the make_random()/validate() stress loop)', 'test --dd',
     dict(r0=(0, 70), K=(5, 100))),
    ('max_toplevel_rank=6, tspc_multiple=256, max_beams_per_gpu=8', 'loopback',
     dict(max_toplevel_rank=6, tspc_multiple=256, max_beams_per_gpu=8),
     'test --net, test --serv (tests/utils.py:make_random_subscale_config)',
     'test --net, --serv',
     dict(early=(0, 80))),
    ('max_toplevel_rank drawn in 6..10', None, None,
     'test --amax (test_decode_argmax._make_random_config)', 'test --amax',
     dict()),
]


def _sec_make_random(rep, ndraw):
    import random

    from ..pirate_pybind11 import DedispersionConfig

    for (name, note_key, kwargs, consumer, tag, bands) in _MAKE_RANDOM_SETTINGS:
        rep.section(f'DedispersionConfig::make_random({name})',
                    subtitle=f'{ndraw} draws', consumer=consumer)
        if kwargs is None:
            # The --amax setting redraws the cap per config, so the bound is per config too.
            caps = [random.randint(6, 10) for _ in range(ndraw)]
            configs = [DedispersionConfig.make_random(max_toplevel_rank=m) for m in caps]
        else:
            configs = [DedispersionConfig.make_random(**kwargs) for _ in range(ndraw)]
            caps = [kwargs.get('max_toplevel_rank', 10)] * ndraw

        _draw_config_rows(rep, configs, tag, bands)

        # Computed here rather than in _draw_config_rows(), because only the caller knows what
        # bound was asked for.
        n_over = sum(1 for (c, m) in zip(configs, caps) if int(c.toplevel_tree_rank) > m)
        rep.rate('toplevel_tree_rank exceeded the requested bound', n_over, ndraw, (0, 0), tag)

        for line in _SETTING_NOTES.get(note_key, []):
            rep.note(line)
        atomic_print('')


####################################################################################################
#
# Section B: the kernel registries. A property of what this build compiled, not of any draw.


# Which caller passes which max_toplevel_rank. Spelled out so the reachability rows below
# name a test rather than a number.
_CAP_CONSUMER = {
    6:  'test --net/--serv, test --amax (low end of its draw)',
    7:  'test --varmap (_random_config)',
    8:  'test --dd validate loop, the gpu-vs-cpu sweep, test --amax',
    9:  'test_multimap_vs_sweep',
    10: 'make_random() default: GpuDedisperser::test_random() (test --dd)',
    12: 'test --sb (test_subband_property)',
}


def _sec_registries(rep):
    """What the compiled kernel list makes reachable.

    THE ONLY SECTION HERE THAT IS NOT A DRAW, and the only one whose fix is a build change.
    A kernel test that picks a random registry key can never reach a (dtype, rank, Wmax, ...)
    combination the build did not stock, so a combination that is absent is one no amount of
    test iterations will cover. Nothing else in the suite surfaces that.
    """

    from .. import kernels as K

    rep.section('Kernel registries (a property of this build, not of a draw)',
                subtitle='marginals over every registered key',
                consumer='test --cdd2, --gddk, --sbdd, --gpfk, --pfwr, --pfom')

    specs = [
        ('CoalescedDdKernel2', K.CoalescedDdKernel2, (40, 400), 'test --cdd2, test --dd'),
        ('GpuDedispersionKernel', K.GpuDedispersionKernel, (8, 200), 'test --gddk'),
        ('GpuSbDedispersionKernel', K.GpuSbDedispersionKernel, (8, 200), 'test --sbdd'),
        ('GpuPeakFindingKernel', K.GpuPeakFindingKernel, (8, 200), 'test --gpfk'),
        ('PfWeightReaderMicrokernel', K.PfWeightReaderMicrokernel, (8, 200), 'test --pfwr'),
        ('PfOutputMicrokernel', K.PfOutputMicrokernel, (2, 100), 'test --pfom'),
    ]

    keys = {}
    for (name, cls, band, consumer) in specs:
        keys[name] = cls.registry_keys()
        rep.enum(f'{name} size', len(keys[name]), band, consumer)

    atomic_print('')
    for (name, _cls, _band, _consumer) in specs:
        kk = keys[name]
        if not kk:
            continue
        fields = [f for f in kk[0] if f != 'subband_counts']
        parts = [f'{f} {_hist([str(np.dtype(k[f])) if f == "dtype" else k[f] for k in kk], top=8)}'
                 for f in fields]
        rep.note(f'{name}:')
        for s in parts:
            rep.note(f'    {s}')
        if 'subband_counts' in kk[0]:
            sb = sorted(set(tuple(k['subband_counts']) for k in kk))
            rep.note(f'    subband_counts: {len(sb)} distinct, pf_rank'
                     f' {_hist([len(s) - 1 for s in sb])}')
    atomic_print('')

    # Derived rows: the two facts the marginals above imply but do not say out loud.

    # K > 0 is the extra-DM-bit folding path. The generated code differs substantially
    # between K == 0 and K > 0, so a registry stocked entirely at one value would leave the
    # other completely untested however many iterations run.
    cdd2 = keys['CoalescedDdKernel2']
    n_k = sum(1 for k in cdd2 if (-(-k['dd_rank'] // 2) - (len(k['subband_counts']) - 1)) > 0)
    rep.rate('cdd2 keys with K > 0', n_k, len(cdd2), (15, 85),
             'test --cdd2; K > 0 is the extra-DM-bit path')

    gpfk = keys['GpuPeakFindingKernel']
    rep.rate('peak-finding keys with K > 0', sum(1 for k in gpfk if k['K'] > 0), len(gpfk),
             (15, 85), 'test --gpfk, test --pfwr')

    # THE "45 UNREACHABLE KERNELS" ROW. make_random() draws its base cdd2 key only from those
    # with dd_rank <= max_stage2_rank = (max_toplevel_rank+1)/2, so a config-driven test can
    # never reach the rest -- however many iterations it runs, and whatever else is randomized.
    # Reported at the caps callers actually pass. Note 7 and 8 read the same, as do 9 and 10:
    # the cap enters only through max_stage2_rank = (m+1)/2, so an odd cap is an exact alias
    # for the even one above it.
    for m in (6, 7, 8, 9, 10, 12):
        reach = sum(1 for k in cdd2 if k['dd_rank'] <= (m + 1) // 2)
        rep.rate(f'cdd2 keys a config can reach at max_toplevel_rank={m}',
                 reach, len(cdd2), (5, 100), _CAP_CONSUMER[m])

    atomic_print('')


####################################################################################################
#
# Section C: varmap.tests._random_config(), the varmap suite's own wrapper.


def _sec_varmap_config(rep, ndraw, nstraddle):
    """_random_config() is make_random() plus a SIZE FLOOR and a drawn gpu_valid, and both
    move the population enough that it needs its own section rather than a row under A."""

    from ..varmap.tests import _random_config, _rng

    rep.section('varmap.tests._random_config()',
                subtitle=f'{ndraw} draws (max_toplevel_rank=7, max_early_triggers=2,'
                         f' gpu_valid drawn, nalpha floor 32)',
                consumer='test --varmap: every test outside run_once() draws its geometry here')

    consumer = 'test --varmap'
    rng = _rng()
    configs = [_random_config(rng) for _ in range(ndraw)]

    # The size floor lifts npri and the early-trigger rate relative to a bare make_random()
    # at the same cap, and drawing gpu_valid mixes the two disjoint corners -- so these bands
    # are not the ones in section A even where the row is.
    _draw_config_rows(rep, configs, consumer,
                      dict(early=(10, 70), r0=(5, 60), K=(20, 100), mib=(0.0, 64.0)))

    # The non-contiguous multiplet map is the only case where a wrong gather is visible at
    # all -- with a prefix map an off-by-one moves nothing. It is defined only on configs that
    # HAVE an early trigger, so it is reported against that count: a low absolute rate then
    # reads as "few early triggers" or "few of them clamp", which are different problems.
    n_early, n_nc, n_pairs, n_nc_pairs = 0, 0, 0, 0
    for config in configs:
        nets = [int(pt.num_early_triggers) for pt in config.primary_trees]
        if max(nets) == 0:
            continue
        n_early += 1
        plan, found = _plan(config), False
        for g in range(int(config.num_primary_trees)):
            iparent = int(plan.dedispersion_tree_index(g, 0))
            for e in range(1, nets[g] + 1):
                ichild = int(plan.dedispersion_tree_index(g, e))
                m_map = np.asarray(plan.m_index_mapping(iparent, ichild))
                n_pairs += 1
                if not np.array_equal(m_map, np.arange(m_map.size)):
                    n_nc_pairs += 1
                    found = True
        n_nc += int(found)

    rep.rate('non-contiguous multiplet map, given an early trigger', n_nc, n_early, (2, 80),
             consumer)
    rep.note(f'over {n_pairs} (parent, child) pairs, {n_nc_pairs} non-contiguous')

    # The straddle row costs an SdPlan per config, orders of magnitude more than a draw, so
    # it is measured on a subsample and reported against that count.
    n_str, n_tot = _straddle_rate(configs[:nstraddle])
    rep.rate('a straddled (channel, subband) entry', n_str, n_tot, (3, 90),
             'test --varmap, fast_varmap.test_cpp_detrender_free')
    atomic_print('')


def _straddle_rate(configs):
    """(n_with_a_straddle, n_examined). Its own function because it is the one statistic in
    section C that costs real work."""

    import contextlib
    import io

    from ..varmap.detrender_free import SdPlan

    n = 0
    for config in configs:
        with contextlib.redirect_stdout(io.StringIO()):
            n += int(int(SdPlan(config, init_sd_matrices=False).stats['n_straddled']) > 0)
    return n, len(configs)


####################################################################################################
#
# Section D: the LP/basis cell draw.


def _sec_lp_cell(rep, ndraw):
    """_draw_lp_cell_config() is a floor AND a ceiling, both on SIZE, and LP_CELL_BUDGET is
    the cost knob for the whole LP and basis tier. Nothing else reports whether it still does
    what its comment claims."""

    from ..varmap.tests import LP_CELL_BUDGET, _draw_lp_cell_config, _rng, _tree

    rep.section('varmap.tests._draw_lp_cell_config()',
                subtitle=f'{ndraw} cells, budget nbeta*nfreq <= {LP_CELL_BUDGET:g}',
                consumer='test --varmap: the LP tier (test_lp_*) and the basis tier'
                         ' (test_svd, test_column_algebra, test_map_steps, ...)')

    rng = _rng()
    nbeta, nfreq, prod, mind = [], [], [], []
    for _ in range(ndraw):
        config = _draw_lp_cell_config(rng)
        tree = _tree(config, 0)
        fs = tree.frequency_subbands
        R, rr = int(fs.pf_rank), int(tree.tree_rank)
        b = (1 << (rr - R - 1)) * int(fs.N) * int(tree.nprofiles)
        f = int(config.get_total_nfreq())
        nbeta.append(b)
        nfreq.append(f)
        prod.append(b * f)
        mind.append(min(b, f))

    rep.dist('nbeta', nbeta, ('p50', 16, 4096), 'the LP tier: cost tracks nbeta',
             fmt='{:.0f}')
    rep.dist('nfreq', nfreq, ('p50', 16, 4096), 'the LP tier', fmt='{:.0f}')
    rep.dist('nbeta * nfreq', prod, ('max', 0, LP_CELL_BUDGET),
             f'LP_CELL_BUDGET = {LP_CELL_BUDGET:g}', fmt='{:.0f}')

    # K reaches 12 across these tests and a factorization has at most min(nbeta, nfreq)
    # modes, so this floor is what lets every caller treat its K as fitting the cell.
    rep.rate('min(nbeta, nfreq) >= 12', sum(1 for x in mind if x >= 12), ndraw, (100, 100),
             'test_lp_rescue (K = 12), and _draw_K() in the basis tests')

    # MEASURED ON THE RIGHT POPULATION, which is the whole reason this row moved here: it
    # used to be reported under _random_config(), whose cells this test never sees.
    rep.rate('nfreq a multiple of 8 (row blocking is bit-exact)',
             sum(1 for f in nfreq if f % 8 == 0), ndraw, (2, 60),
             'test_lp_repairs: blocking_is_exact() runs only on these draws')
    atomic_print('')


####################################################################################################
#
# Section E: the brute-force sweep draws.


def _sec_sweep(rep, ncase):
    """Acceptance rates for the sweep tests' rejection loops.

    THE ONE THING NOTHING ELSE MEASURES. Each of these loops draws a config, builds a full
    DedispersionPlan inside _SweepGeometry to price it, and discards it if it is over budget --
    so a filter that has quietly tightened costs plan builds, and one that has gone to zero
    raises rather than hangs but only after max_attempts. The rates live in comments today,
    which is exactly where a measured number goes stale.

    Only the two draws that are NAMED FUNCTIONS are reported. test_multimap_vs_sweep and
    test_sweep_column_norms_random build their arguments inline, so there is nothing to call;
    factoring them out is what it would take to add them here.

    THE ONE SECTION THAT NEEDS A GPU: _SweepGeometry builds a full DedispersionPlan, and the
    acceptance rate is defined by a cost model evaluated on it. Skipped without a device.
    """

    from ..varmap.tests import _draw_gpu_vs_cpu_case, _draw_restriction_config, _rng

    rep.section('varmap.tests: the sweep rejection loops',
                subtitle=f'{ncase} accepted cases each (every attempt builds a DedispersionPlan)',
                consumer='test --varmap: test_sweep_gpu_vs_cpu_random,'
                         ' test_restriction_vs_sweep')

    rng = _rng()
    try:
        _draw_gpu_vs_cpu_case(rng=rng)
    except Exception as e:
        rep.note(f'SKIPPED (needs a CUDA device): {type(e).__name__}: {e}')
        atomic_print('')
        return

    for (label, fn, band, consumer) in [
        ('_draw_gpu_vs_cpu_case', lambda: _draw_gpu_vs_cpu_case(rng=rng)[3], (20, 100),
         'test_sweep_gpu_vs_cpu_random'),
        ('_draw_restriction_config', lambda: _draw_restriction_config(rng)[2], (0.5, 40),
         'test_restriction_vs_sweep; the tightest filter in varmap/tests.py'),
    ]:
        t0 = time.time()
        attempts = [fn() for _ in range(ncase)]
        dt = time.time() - t0
        rep.rate(f'{label}: acceptance', ncase, sum(attempts), band, consumer)
        rep.dist(f'{label}: attempts per case', attempts, ('max', 1, 500),
                 'a case near max_attempts is one filter tightening from raising',
                 fmt='{:.0f}')
        rep.note(f'{dt / ncase * 1e3:.0f} ms per accepted case')
    atomic_print('')


####################################################################################################
#
# Section F: the detrending draws.


def _sec_detrending(rep, ndraw):
    from ..detrending_spline import masks as msk
    from ..detrending_spline.SplineDetrender import ETA_DEFAULT
    from ..detrending_spline.reduce import CHANNEL_BLOCK
    from ..utils import random_nfreq

    rep.section('detrending_spline.masks.random_knots()',
                subtitle=f'{ndraw} knot vectors, n_phi drawn in 0..3 as run_all() does',
                consumer='test --dts: every test in the spline suite')

    rng = np.random.default_rng()
    kvs, nz, nfr, kinds = [], [], [], []
    for _ in range(ndraw):
        n_phi = int(rng.integers(0, 4))
        kv = msk.random_knots(rng, n_phi=n_phi)
        kvs.append(kv)
        nz.append(int(kv.nzone))
        nfr.append(int(kv.nfreq))
        kinds.append(int(len(kv.knots)))

    # nzone == 1 makes the partition-of-unity and cross-zone-independence checks vacuous:
    # they are skipped outright, so a collapse here silently removes them.
    rep.rate('nzone > 1', sum(1 for x in nz if x > 1), ndraw, (10, 90),
             'test_basis (partition of unity), test_zone_expansion (cross-zone)')
    rep.rate('K == 1 (one knot interval, h_max = nfreq)',
             sum(1 for x in kinds if x <= 2), ndraw, (2, 60),
             'test_conditioning, test_dtype_agreement: the h_max extreme')
    # BANDED TO FLAG, deliberately. random_knots()'s own default is still uniform on
    # [64, 10001), while 22 of its 25 call sites now pass a log-uniform random_nfreq(). The
    # three that do not -- test_knots, test_basis, test_regulator -- therefore never see a
    # small band, which is exactly the regime where zones are a few channels wide and the
    # basis support is clamped at both ends. One-line fix at each call site.
    rep.dist("nfreq (random_knots' own default draw)", nfr, ('p10', 32, 600),
             'test_knots, test_basis, test_regulator (the 3 call sites that omit nfreq)',
             fmt='{:.0f}')
    rep.rate('nfreq < CHANNEL_BLOCK (one partial block)',
             sum(1 for x in nfr if x < CHANNEL_BLOCK), ndraw, (0, 80),
             'test_chunk_invariance: the degenerate case of the frequency reduction')
    rep.note(f'nzone {_hist(nz, top=8)}')
    atomic_print('')

    # random_nfreq() is the shared helper that most spline call sites now pass, so its shape
    # is what the suite actually sees -- reported at one representative cap.
    rep.section('utils.random_nfreq(rng, hi)',
                subtitle=f'{ndraw} draws at hi=2500, the cap test_chunk_invariance uses',
                consumer='test --dts, --dt1d, --dt1k: 22 spline call sites and the'
                         ' detrending shape draws')
    v = [random_nfreq(rng, 2500) for _ in range(ndraw)]
    rep.dist('nfreq', v, ('p10', 32, 300),
             'the point of the log-uniform draw is weight below 200', fmt='{:.0f}')
    rep.rate('nfreq < 200 (few-channel zones, clamped basis support)',
             sum(1 for x in v if x < 200), ndraw, (10, 70),
             'the regime a uniform draw reached 1.4% of the time')
    atomic_print('')

    # mask_distribution() already computes exactly the right numbers and is called from
    # NOWHERE; its output is frozen in a comment block in masks.py, which is where a measured
    # number goes stale. This is its consumer.
    rep.section('detrending_spline.masks.random_mask_2d()',
                subtitle='via masks.mask_distribution(), on 3 drawn knot vectors',
                consumer='test --dts: test_time_rank_deficiency, test_n1_degeneracy,'
                         ' test_2d_conditioning')
    for kv in kvs[:3]:
        d = msk.mask_distribution(kv, rng, ETA_DEFAULT, n=1, W=2, ntime=9, ndraw=60)
        rep.note(f'nfreq {kv.nfreq:6d} nzone {kv.nzone}: '
                 + '  '.join(f'{k} {v:.3g}' if isinstance(v, float) else f'{k} {v}'
                             for (k, v) in d.items()))
    atomic_print('')


####################################################################################################
#
# Entry point.


_SECTIONS = ('config', 'reg', 'varmap', 'dt')


def report_coverage(select=(), scale=1.0):
    """Print the coverage report. 'select' is a subset of _SECTIONS (empty = all).

    'scale' multiplies every draw count. The defaults are chosen so a full run is well under
    a minute; scale it up when a rate is close to a band edge and you want to know whether it
    really moved.
    """

    sel = set(select) if select else set(_SECTIONS)
    rep = _Report()

    def n(x):
        return max(1, int(round(x * scale)))

    atomic_print('')
    # Draw counts. Chosen so that a full run is a few tens of seconds -- cheap enough to run
    # on a whim, which is the point of a diagnostic nobody is forced to look at. The straddle
    # subsample and the sweep case count are much smaller because each costs a plan build.
    if 'config' in sel:
        _sec_make_random(rep, n(600))
    if 'reg' in sel:
        _sec_registries(rep)
    if 'varmap' in sel:
        _sec_varmap_config(rep, n(600), n(150))
        _sec_lp_cell(rep, n(300))
        _sec_sweep(rep, n(40))
    if 'dt' in sel:
        _sec_detrending(rep, n(600))

    rep.finish()
