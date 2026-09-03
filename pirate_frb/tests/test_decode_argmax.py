"""
Black-box tests of decode_argmax() (run via 'test --amax').

Note decode_argmax(), decode_argmax2() and compute_steady_state_it0() are DedispersionPlan
methods; DedispersionTree has none of them. The VECTORIZED (batch) bindings live on
FrbGrouper, which is what the production event path uses, and are tested in test_server.py
('test --serv').

Strategy: for a FIXED token, eval_tokens() is a
LINEAR functional of the input array -- the actual dedispersion + peak-finding
computation for that one trial, with the max-reduce removed -- and all its
coefficients are nonnegative (profile coefficients in {1/2, 1}; dedispersion and
downsampling factors are positive; the test sets peak-finding weights to 1).
So for a nonnegative indicator probe, the functional is nonzero IFF the probe
overlaps the trial's support. Probes are injected directly into the toplevel
tree-domain space (ReferenceDedisperser(tree_domain_input=True)), which is the
space decode_argmax() reports in.

Per sampled tuple (itree, token, idm, itout), with decoded (fmin, fmax, tlo, thi)
-- where tlo/thi are EXCLUSIVE trailing edges (one past the last summed sample):

  P1: one-hot at (fmin, tlo - 1)  -> eval_tokens cell must be nonzero
  P2: one-hot at (fmax, thi - 1)  -> eval_tokens cell must be nonzero
  P3: indicator of the complement region {f < fmin} + {f > fmax} +
      {f == fmin, t >= tlo} + {f == fmax, t >= thi}
                                  -> eval_tokens cell must be exactly zero

P1+P2+P3 pins the support's channel range and both edge trailing times exactly
(completeness argument in the plan). A constant-probe membership sweep
supplements this with exhaustive fmin/fmax coverage (all tokens/cells at once
per probed channel), and a bare-ReferencePeakFindingKernel sweep covers the
token time-quantization formula (dt = min(Dcore, 2^level)) densely.
"""

import random
import numpy as np
import yaml

from ..pirate_pybind11 import (DedispersionConfig, DedispersionPlan, DedispersionTree,
                               ReferenceDedisperser, ReferencePeakFindingKernel)
from ..utils import atomic_print


####################################################################################################
#
# Yaml tests. DedispersionPlan::from_yaml() is how FrbGrouper recovers the producer's geometry
# at handshake, and how a variance-map file recovers the geometry it was written with. Unlike
# a naive parser it does not transcribe: it rebuilds the plan from the config and CHECKS the
# yaml against it, adopting nothing.


_TREE_INT_MEMBERS = ('primary_tree_index', 'early_trigger_level', 'amb_rank', 'dd_rank',
                     'nt_ds', 'dm_downsampling', 'time_downsampling', 'nprofiles',
                     'ndm_out', 'ndm_wt', 'nt_out', 'nt_wt')
_TREE_PF_MEMBERS = ('max_width', 'wt_dm_downsampling', 'wt_time_downsampling')

# The plan-level scalars the perturbation loop below bumps. (Not every key to_yaml() emits:
# 'dtype' is a string, and the two rank vectors are handled separately.)
_PLAN_INT_KEYS = ('ntrees', 'nfreq', 'nt_in', 'toplevel_tree_rank', 'num_primary_trees',
                  'beams_per_gpu', 'beams_per_batch', 'num_active_batches')


def _test_plan_yaml(config):
    """Round-trip of a "minimal" plan's yaml, which needs no GPU.

    This is the variance-map file's route: a GPU-less process writes plan.to_yaml_string()
    and a reader recovers it with from_yaml_string(). WHAT THIS CAN AND CANNOT SHOW:
    from_yaml_string() builds its trees from the config, so comparing the rebuilt plan's
    trees to the original's would compare the constructor to itself. The three meaningful
    checks are (i) the yaml is ACCEPTED, in particular by the loose comparison on the lossy
    display doubles; (ii) it re-emits byte-identically; and (iii) the perturbation loop,
    which is what proves the verifier actually reads every field -- a field emitted but never
    checked would pass (i) and (ii) both.
    """

    p = DedispersionPlan(config, mega_ringbuf=False, gpu_kernels=False)
    s = p.to_yaml_string()

    # A minimal plan has no MegaRingbuf, so its yaml is the plan yaml minus that one key.
    doc = yaml.safe_load(s)
    assert 'mega_ringbuf' not in doc
    assert doc['ntrees'] == p.ntrees

    p2 = DedispersionPlan.from_yaml_string(config, s)
    assert p2.to_yaml_string() == s

    # A plan is a pure function of its config, so a COMPLETE plan's 'trees' must be identical
    # to this minimal one's. In particular the plan yaml does not depend on which cdd2
    # kernels this build compiled, which is what lets a GPU-less writer and a GPU reader
    # exchange one. (Only 'trees' is compared: a complete plan's yaml also has a
    # mega_ringbuf section. test_decode_argmax's configs always admit a complete plan.)
    full = yaml.safe_load(DedispersionPlan(config).to_yaml_string())
    assert full['trees'] == doc['trees']

    def expect_throw(bad_doc, what):
        try:
            DedispersionPlan.from_yaml_string(config, yaml.safe_dump(bad_doc))
        except RuntimeError as e:
            assert what in str(e), (what, str(e))
            return
        raise AssertionError(f'from_yaml_string() should have thrown ({what})')

    # Perturb one field at a time and require the verifier to notice, naming the field.
    for key in _PLAN_INT_KEYS:
        bad = yaml.safe_load(s)
        bad[key] = int(bad[key]) + 1
        expect_throw(bad, key)

    for key in ('stage1_dd_rank', 'stage1_amb_rank'):
        bad = yaml.safe_load(s)
        bad[key][0] = int(bad[key][0]) + 1
        expect_throw(bad, key)

    for key in ('tree_index',) + _TREE_INT_MEMBERS + _TREE_PF_MEMBERS:
        bad = yaml.safe_load(s)
        bad['trees'][0][key] = int(bad['trees'][0][key]) + 1
        expect_throw(bad, key)

    bad = yaml.safe_load(s)
    bad['trees'][0]['frequency_subband_counts'][0] += 1
    expect_throw(bad, 'frequency_subband_counts')

    # The display doubles are compared at 1e-4 relative tolerance (yaml-cpp emits ~6
    # significant digits). '2x + 1' is the perturbation because it exceeds the tolerance for
    # any value, including dm_min = 0 on a primary_tree_index == 0 tree.
    for key in ('dm_min', 'dm_max', 'trigger_frequency'):
        bad = yaml.safe_load(s)
        bad['trees'][0][key] = 2.0 * float(bad['trees'][0][key]) + 1.0
        expect_throw(bad, key)


def _test_grouper_plan_rebuild(config, plan, dcores, tuples):
    """Rebuild the producer's plan from its yaml, as FrbGrouper does at handshake.

    Distinct from _test_plan_yaml() above in one way: the plan is a COMPLETE one, so its yaml
    carries a mega_ringbuf section (which from_yaml() ignores). Decoding through the rebuilt
    plan must agree with decoding through the original -- with the SAME 'dcores', since those
    do not travel in the yaml (the grouper gets them from their own handshake field).
    """

    cfg2 = DedispersionConfig.from_yaml_string(config.to_yaml_string())
    plan_yaml = plan.to_yaml_string()
    assert 'mega_ringbuf' in yaml.safe_load(plan_yaml)

    p2 = DedispersionPlan.from_yaml_string(cfg2, plan_yaml)
    assert p2.ntrees == plan.ntrees

    for (itree, token, idm, itout) in tuples:
        assert (p2.decode_argmax(token, itree, dcores[itree], idm, itout)
                == _decode(plan, token, itree, dcores[itree], idm, itout)), (itree, token)

    # Negative test: a missing per-tree key must throw (naming the key).
    bad = yaml.safe_load(plan_yaml)
    bad['trees'][0]['nt_out_renamed'] = bad['trees'][0].pop('nt_out')
    try:
        DedispersionPlan.from_yaml_string(cfg2, yaml.safe_dump(bad))
        raise AssertionError("DedispersionPlan.from_yaml_string() should have thrown"
                             " (missing nt_out)")
    except RuntimeError:
        pass

    return p2


def _decode(plan, token, itree, dcore, idm_coarse, itime_coarse):
    """Scalar decode. Decoding is a DedispersionPlan method; the vectorized form lives on
    FrbGrouper (see test_server.py).

    'dcore' is the producing peak-finding kernel's core factor, not a plan property -- here
    it is the value the scout ReferenceDedisperser was built with."""
    return plan.decode_argmax(token, itree, dcore, idm_coarse, itime_coarse)


####################################################################################################


def _make_random_config():
    """Random config with nbatches == 1 and enough beams to pack P1/P2/P3 probes.

    THE THREE BEAM ASSIGNMENTS ARE ALWAYS ACCEPTED, so this draws once rather than
    retrying: validate() asks only that num_active_batches * beams_per_batch <=
    beams_per_gpu, which (nbeams, nbeams, 1) satisfies by construction. Measured over
    4500 draws spanning max_toplevel_rank 6-10 and nbeams 3-12, the acceptance rate is
    exactly 1.

    nbeams is how many probes one pipeline run can carry -- three per tuple, so
    _probe_tuples() packs floor(nbeams/3) tuples into a run. Drawing it small makes the
    test slower, not weaker.

    max_toplevel_rank matters more than it looks, because make_random() ties the toplevel
    rank to the cdd2 base key it picked: R lies in [2*dd_rank - 1, 2*dd_rank], so a cap of
    6 admits dd_rank = 3 AND NOTHING ELSE -- 12 of the registry's 106 keys, every one of
    them with subband_counts (1,) or (2,1), so a cap of 6 cannot produce a tree with a wide
    subband split at all. 7-8 brings in dd_rank = 4, 9-10 dd_rank = 5.
    """
    nbeams = random.randint(3, 12)
    config = DedispersionConfig.make_random(max_toplevel_rank=random.randint(6, 10))
    config.beams_per_gpu = nbeams
    config.beams_per_batch = nbeams
    config.num_active_batches = 1
    config.validate()
    return config


def _num_chunks(plan, r_top, nt_in):
    """Chunk count covering max dispersion depth + peak-finding reach, in full-res samples.

    This is a CORRECTNESS requirement of probe P3, not just a settling nicety: P3's
    completeness needs the simulated span to cover the trial support's full extent.
    """
    depth = 0
    for tree in plan.trees:
        wmax = tree.primary_tree.max_width
        tpad = max(2 * wmax, 4)
        ds = 2 ** (r_top - tree.early_trigger_level) + 4 * wmax + tpad   # downsampled samples
        depth = max(depth, ds * 2 ** tree.primary_tree_index)            # full-res samples
    return depth // nt_in + 2


def _draw_dcores(plan):
    """Per-tree peak-finder core factors: a random power of two dividing each tree's Dout.

    DRAWN rather than taken from the cdd2 registry, which pins Dcore = min(Dout, 8) and so
    differs from Dout only at dd_rank >= 7. Dcore is a property of the peak-finding kernel,
    and a ReferencePeakFindingKernel accepts any legal value, so drawing it exercises the
    token time-quantization rule dt = min(Dcore, 2^level) at every Dcore a kernel could have.
    """
    return [2 ** random.randrange(int(tree.time_downsampling).bit_length())
            for tree in plan.trees]


def _fresh_rdd(plan, dcores):
    """ReferenceDedisperser in tree-domain-input mode, with all pf weights = 1.

    'dcores' must be the same vector the scout was built with, or its tokens would quantize
    time differently from the ones _decode() is checked against."""
    rdd = ReferenceDedisperser(plan, sophistication=0, tree_domain_input=True, Dcores=dcores)
    for w in rdd.wt_arrays:
        w[...] = 1.0
    return rdd


def _eval_tokens(rdd, plan, itree, tokens_by_beam):
    """Run eval_tokens for one tree, with per-beam tokens filled over all cells.

    tokens_by_beam: dict beam -> token. Other beams get token 0 (always valid).
    Returns the (B, ndm_out, nt_out) output array.
    """
    tree = plan.trees[itree]
    B = plan.beams_per_batch
    toks = np.zeros((B, tree.ndm_out, tree.nt_out), dtype=np.uint32)
    for b, token in tokens_by_beam.items():
        toks[b, :, :] = token
    out = np.zeros((B, tree.ndm_out, tree.nt_out), dtype=np.float32)
    rdd.pf_kernels[itree].eval_tokens(out, toks, rdd.wt_arrays[itree])
    return out


####################################################################################################
#
# Membership sweep: constant probe on a single channel fstar. For any token, the trial's
# support contains channel fstar iff fmin <= fstar <= fmax, so eval_tokens is strictly
# positive at EVERY cell (in-band) or exactly zero at EVERY cell (out-of-band). One probed
# channel validates fmin/fmax for all tokens and all (idm, itout) cells simultaneously.


def _membership_sweep(plan, dcores, tree_bands, C, chans):
    B = plan.beams_per_batch
    nt_in = plan.nt_in

    for i0 in range(0, len(chans), B):
        batch = chans[i0 : i0 + B]

        rdd = _fresh_rdd(plan, dcores)
        ia = rdd.input_array
        for c in range(C):
            ia[...] = 0.0
            for b, f in enumerate(batch):
                ia[b, f, :] = 1.0
            rdd.dedisperse(c, 0)

        for itree in range(plan.ntrees):
            for token, fmn, fmx in tree_bands[itree]:
                out = _eval_tokens(rdd, plan, itree, {b: token for b in range(len(batch))})
                for b, f in enumerate(batch):
                    if fmn <= f <= fmx:
                        assert (out[b] > 0).all(), \
                            f"membership: expected nonzero (itree={itree}, token={token:#x}, " \
                            f"band=[{fmn},{fmx}], fstar={f})"
                    else:
                        assert (out[b] == 0).all(), \
                            f"membership: expected zero (itree={itree}, token={token:#x}, " \
                            f"band=[{fmn},{fmx}], fstar={f})"


####################################################################################################
#
# P1/P2/P3 probes for sampled (itree, token, idm, itout) tuples.


def _token(m, mu, p=0, t=0):
    """An argmax token, (t) | (p << 8) | (m << 16) | (mu << 24); see PeakFindingKernel.hpp."""
    return t | (p << 8) | (m << 16) | (mu << 24)


def _sample_tuples(plan, kinfo, interesting_ms, ntuples):
    """Return stratified-ish random tuples.

    Biases (m, mu) toward subband/fine-dm extremes, p and t toward their extremes, cells
    toward corners."""

    def _pick(lo_hi_n):
        return random.choice(lo_hi_n)

    tuples = []
    for _ in range(ntuples):
        itree = random.randrange(plan.ntrees)
        M, P, Dout, Dcore, K = kinfo[itree]
        tree = plan.trees[itree]

        m, mu = random.choice(interesting_ms[itree] + [(random.randrange(M), random.randrange(1 << K))])
        p = _pick([0, P - 1, random.randrange(P)])
        lpf = (p - 1) // 3 if p else 0
        dt = min(Dcore, 2 ** lpf)
        nsamp = Dout // dt
        t = _pick([0, nsamp - 1, random.randrange(nsamp)]) * dt
        token = _token(m, mu, p, t)

        idm = _pick([0, tree.ndm_out - 1, random.randrange(tree.ndm_out)])
        itout = _pick([0, tree.nt_out - 1, random.randrange(tree.nt_out)])
        tuples.append((itree, token, idm, itout))

    return tuples


def _probe_tuples(plan, dcores, r_top, C, tuples):
    B = plan.beams_per_batch
    nt_in = plan.nt_in
    nchan = 2 ** r_top
    c_eval = C - 1
    per_run = max(B // 3, 1)

    for i0 in range(0, len(tuples), per_run):
        run_tuples = tuples[i0 : i0 + per_run]
        dec = [_decode(plan, tok, it, dcores[it], idm, ito)
               for (it, tok, idm, ito) in run_tuples]

        # Global (multi-chunk) positions of the decoded trailing edges (EXCLUSIVE: the
        # last summed sample is tlo-1 / thi-1); the warmup formula in _num_chunks()
        # guarantees these land inside the simulated span.
        for (fmin, fmax, tlo, thi, p), (it, tok, idm, ito) in zip(dec, run_tuples):
            assert 0 <= fmin < fmax < nchan
            assert tlo <= thi <= nt_in
            assert c_eval * nt_in + tlo - 1 >= 0, "test bug: warmup depth insufficient"

        rdd = _fresh_rdd(plan, dcores)
        ia = rdd.input_array
        for c in range(C):
            ia[...] = 0.0
            t0 = c * nt_in
            for k, (fmin, fmax, tlo, thi, p) in enumerate(dec):
                glo = c_eval * nt_in + tlo
                ghi = c_eval * nt_in + thi

                # P1/P2: one-hot probes at the last summed samples (beam slots 3k, 3k+1).
                if t0 <= glo - 1 < t0 + nt_in:
                    ia[3*k, fmin, glo - 1 - t0] = 1.0
                if t0 <= ghi - 1 < t0 + nt_in:
                    ia[3*k + 1, fmax, ghi - 1 - t0] = 1.0

                # P3: complement-region indicator (beam slot 3k+2).
                ia[3*k + 2, :fmin, :] = 1.0
                ia[3*k + 2, fmax + 1:, :] = 1.0
                lo = glo - t0
                if lo < nt_in:
                    ia[3*k + 2, fmin, max(lo, 0):] = 1.0
                hi = ghi - t0
                if hi < nt_in:
                    ia[3*k + 2, fmax, max(hi, 0):] = 1.0

            rdd.dedisperse(c, 0)

        # Evaluate, grouping tuples by tree (eval_tokens is per-tree).
        by_tree = {}
        for k, (it, tok, idm, ito) in enumerate(run_tuples):
            by_tree.setdefault(it, []).append((k, tok, idm, ito))

        for it, items in by_tree.items():
            tokens_by_beam = {}
            for k, tok, idm, ito in items:
                for b in range(3*k, 3*k + 3):
                    tokens_by_beam[b] = tok
            out = _eval_tokens(rdd, plan, it, tokens_by_beam)

            for k, tok, idm, ito in items:
                msg = f"itree={it}, token={tok:#x}, idm={idm}, itout={ito}, decode={dec[k]}"
                assert out[3*k, idm, ito] > 0, f"P1 failed (tlo-1 not in support): {msg}"
                assert out[3*k + 1, idm, ito] > 0, f"P2 failed (thi-1 not in support): {msg}"
                assert out[3*k + 2, idm, ito] == 0, f"P3 failed (support outside decoded region): {msg}"


####################################################################################################


def _check_bad_tokens(plan, kinfo):
    """decode_argmax() must throw on malformed tokens and out-of-range indices."""

    itree = random.randrange(plan.ntrees)
    M, P, Dout, Dcore, K = kinfo[itree]
    tree = plan.trees[itree]

    def expect_throw(*args):
        try:
            _decode(plan, *args)
        except RuntimeError:
            return
        raise AssertionError(f"decode_argmax{args} should have thrown")

    # m and mu have a byte each, so each has its own smallest out-of-range value: m < fs.M
    # and mu < 2^K, checked separately -- a reader never needs K to parse a token.
    fsM = int(tree.frequency_subbands.M)
    expect_throw(fsM << 16, itree, Dcore, 0, 0)        # m out of range
    expect_throw((1 << K) << 24, itree, Dcore, 0, 0)   # mu out of range
    expect_throw(P << 8, itree, Dcore, 0, 0)           # p out of range
    if Dout < 256:
        expect_throw(Dout, itree, Dcore, 0, 0)         # t out of range

    for p in range(P):
        lpf = (p - 1) // 3 if p else 0
        if min(Dcore, 2 ** lpf) > 1:
            expect_throw((p << 8) | 1, itree, Dcore, 0, 0)   # t not divisible by dt
            break

    expect_throw(0, plan.ntrees, Dcore, 0, 0)          # itree out of range
    expect_throw(0, itree, Dcore, tree.ndm_out, 0)     # idm_coarse out of range
    expect_throw(0, itree, Dcore, 0, tree.nt_out)      # itime_coarse out of range

    # Dcore is a caller-supplied kernel property, so decode_argmax() range-checks it too:
    # it must be a power of two dividing Dout.
    expect_throw(0, itree, 3, 0, 0)                    # Dcore not a power of two
    expect_throw(0, itree, 2 * Dout, 0, 0)             # Dcore does not divide Dout


def _test_pf_kernel_quantization(ntrials=None):
    """Kernel-level sweep of the token time-quantization formula, with arbitrary Dcore.

    A bare ReferencePeakFindingKernel (no dedispersion, single full-band multiplet) has
    Tlag = Dsub = 0, so the LAST pf-input sample summed by token (p, t) at cell tout is
    T = tout*Dout + t + dt - 1 with dt = min(Dcore, 2^level) -- the same arithmetic as
    decode_argmax(), which reports the exclusive edge (T + 1, before the toplevel time
    conversion). Verify with one-hot / tail probes (fresh kernel per probe, since
    pstate carries the previous probe's tail).
    """

    wmax = random.choice([1, 2, 4, 8, 16, 32])
    dout = random.choice([4, 8, 16, 32])
    dcore = 2 ** random.randrange(dout.bit_length())    # power of two <= Dout
    # A multiple of 32 (the fp32 segment length) is automatically a multiple of Dout <= 32,
    # so one draw satisfies both. Drawn rather than pinned at 512 mainly for nt_out: at
    # Dout = 32 the low end of this range is a two-cell output array, which is where an
    # off-by-one in the tout arithmetic has somewhere to go.
    nt_in = 32 * random.randint(2, 32)
    nt_out = nt_in // dout
    ntrials = random.randint(4, 12) if ntrials is None else ntrials
    P = (3 * wmax.bit_length() - 2) if wmax > 1 else 1  # = 3*log2(Wmax) + 1

    wt = np.ones((1, 1, nt_out, P, 1), dtype=np.float32)

    for _ in range(ntrials):
        p = random.randrange(P)
        lpf = (p - 1) // 3 if p else 0
        dt = min(dcore, 2 ** lpf)
        t = random.randrange(dout // dt) * dt
        tout = random.randrange(nt_out)
        T_exp = tout * dout + t + dt - 1     # decode_argmax's trailing-sample formula
        token = (p << 8) | t

        for tail_probe in (False, True):
            kern = ReferencePeakFindingKernel(
                subband_counts=[1], max_kernel_width=wmax,
                beams_per_batch=1, total_beams=1, dm_downsampling=1, time_downsampling=dout,
                ndm_out=1, ndm_wt=1, nt_out=nt_out, nt_wt=nt_out, Dcore=dcore)
            assert (kern.P, kern.Dout, kern.Dcore) == (P, dout, dcore)

            inp = np.zeros((1, 1, 1, nt_in), dtype=np.float32)
            if tail_probe:
                inp[..., T_exp + 1:] = 1.0   # strictly after the claimed trailing sample
            else:
                inp[..., T_exp] = 1.0        # one-hot at the claimed trailing sample

            out_max = np.zeros((1, 1, nt_out), dtype=np.float32)
            out_argmax = np.zeros((1, 1, nt_out), dtype=np.uint32)
            kern.apply(out_max, out_argmax, inp, wt, 0)

            toks = np.full((1, 1, nt_out), token, dtype=np.uint32)
            out = np.zeros((1, 1, nt_out), dtype=np.float32)
            kern.eval_tokens(out, toks, wt)

            msg = f"Wmax={wmax}, Dout={dout}, Dcore={dcore}, p={p}, t={t}, tout={tout}, T_exp={T_exp}"
            if tail_probe:
                assert out[0, 0, tout] == 0, f"trial reads past its trailing sample: {msg}"
            else:
                assert out[0, 0, tout] > 0, f"trailing sample not in trial support: {msg}"


####################################################################################################


def test_decode_argmax():
    """One iteration of the decode_argmax test suite (see module docstring)."""

    _test_pf_kernel_quantization()

    config = _make_random_config()
    _test_plan_yaml(config)
    plan = DedispersionPlan(config)
    r_top = config.toplevel_tree_rank
    nt_in = plan.nt_in
    B = plan.beams_per_batch
    nchan = 2 ** r_top
    C = _num_chunks(plan, r_top, nt_in)

    atomic_print(f"test_decode_argmax: r_top={r_top}, nt_in={nt_in}, ntrees={plan.ntrees}, "
                 f"nbeams={B}, nchunks={C}")

    # Per-tree (M, P, Dout, Dcore, K), from a scout ReferenceDedisperser: the kernels that
    # emit the tokens are the authority on Dcore and K. The sweeps below enumerate every
    # (m, mu) pair, 0 <= m < M and 0 <= mu < 2^K, which is exactly the set of legal token
    # multiplet fields. (The out-of-range checks are per field: see _check_bad_tokens().)
    dcores = _draw_dcores(plan)
    scout = ReferenceDedisperser(plan, sophistication=0, tree_domain_input=True, Dcores=dcores)
    kinfo = [(k.M, k.P, k.Dout, k.Dcore, k.xdm_rank) for k in scout.pf_kernels]
    del scout
    atomic_print(f'test_decode_argmax: Dcore by tree = {dcores}')

    # Check the kernel's K against the tree's own statement of it, dm_downsampling =
    # 2^(pf_rank + K) -- this is where the plan -> kernel-params conversion is tested. Also
    # report the per-tree K: a run which happened to generate no K > 0 tree covers strictly
    # less, and that should be visible rather than silent.
    xdm_ranks = []
    for itree in range(plan.ntrees):
        tree = plan.trees[itree]
        fs = tree.frequency_subbands
        K = kinfo[itree][4]
        xdm_ranks.append(K)
        assert kinfo[itree][0] == int(fs.M)
        assert (1 << (K + int(fs.pf_rank))) == int(tree.dm_downsampling), (itree, K)
    atomic_print(f'test_decode_argmax: xdm_rank by tree = {xdm_ranks}')

    # The Dcores argument must reach the peak-finding kernels: everything below decodes
    # tokens with 'dcores' and compares against what these kernels actually emitted.
    for itree in range(plan.ntrees):
        assert kinfo[itree][3] == dcores[itree], itree

    _check_bad_tokens(plan, kinfo)

    # Per tree: one token per distinct decoded band (fmin, fmax), i.e. per subband. (Several
    # (m, mu) pairs map to one band -- the fine dms of a multiplet run, and every extra-DM
    # index mu -- so the dict dedupes them.)
    # Also collect the first/last (m, mu) pair of each band, used to bias the P1/P2/P3
    # tuple sampling toward the extremes.
    tree_bands = []
    interesting_ms = []
    for itree in range(plan.ntrees):
        M, K = kinfo[itree][0], kinfo[itree][4]
        first, last = {}, {}
        for m in range(M):
            for mu in range(1 << K):
                fmin, fmax, _, _, _ = _decode(plan, _token(m, mu), itree, dcores[itree], 0, 0)
                first.setdefault((fmin, fmax), (m, mu))
                last[(fmin, fmax)] = (m, mu)
        tree_bands.append([(_token(m, mu), fmn, fmx) for (fmn, fmx), (m, mu) in first.items()])
        interesting_ms.append(sorted(set(first.values()) | set(last.values())))

    # Membership sweep channels: subband edges +-1 (off-by-one killers) + a few random.
    chans = set()
    for bands in tree_bands:
        for _, fmn, fmx in bands:
            chans.update(c for c in (fmn - 1, fmn, fmx, fmx + 1) if 0 <= c < nchan)
    chans.update(random.sample(range(nchan), min(4, nchan)))
    if len(chans) > 2 * B:   # cap at 2 pipeline runs; random subsampling covers the rest across iterations
        chans = random.sample(sorted(chans), 2 * B)
    _membership_sweep(plan, dcores, tree_bands, C, sorted(chans))

    # P1/P2/P3 probes on sampled tuples (2 pipeline runs).
    # One "pipeline run" carries floor(B/3) tuples (three probe beams each); the run count
    # is drawn, since it is a cost knob rather than a property of the geometry.
    tuples = _sample_tuples(plan, kinfo, interesting_ms,
                            ntuples=random.randint(1, 4) * max(B // 3, 1))
    _probe_tuples(plan, dcores, r_top, C, tuples)

    # Rebuild the plan from its yaml, as FrbGrouper does (reuses the sampled tuples).
    _test_grouper_plan_rebuild(config, plan, dcores, tuples)
