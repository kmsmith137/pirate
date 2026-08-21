"""
Black-box tests of decode_argmax() (run via 'test --amax').

Note decode_argmax(), decode_argmax2() and compute_steady_state_it0() are DedispersionTree
methods; DedispersionPlan has none of them. The VECTORIZED (batch) bindings live on
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
# Round-trip test of the per-tree yaml, which is how FrbGrouper recovers the producer's
# DedispersionTrees at handshake. This is the LOAD-BEARING guard keeping the "dumb" yaml
# parser in sync with to_yaml(): DedispersionTree::from_yaml() transcribes members verbatim
# (no re-derivation, no consistency asserts), so any to_yaml/parser drift must fail HERE, via
# member-by-member comparison.


_TREE_INT_MEMBERS = ('primary_tree_index', 'early_trigger_level', 'amb_rank', 'dd_rank',
                     'nt_ds', 'Dcore', 'nprofiles', 'ndm_out', 'ndm_wt', 'nt_out', 'nt_wt')
_TREE_PF_MEMBERS = ('num_early_triggers', 'max_width', 'dm_downsampling', 'time_downsampling',
                    'wt_dm_downsampling', 'wt_time_downsampling')


def _test_tree_yaml(config):
    """Round-trip test of DedispersionTree.to_yaml_string() / from_yaml_string().

    Both DedispersionPlan.to_yaml() and FrbGrouper's handshake go through these, so
    this is the same load-bearing guard as _test_grouper_tree_rebuild() below, on a
    standalone tree yaml -- which is how pirate_frb.varmap stores geometry.

    Also checks that trees are constructible with no DedispersionPlan (hence, on a machine
    with no GPU -- which this test cannot verify, but see the constructor's doc-comment).
    """

    ntrees = config.num_dedispersion_trees

    for itree in range(ntrees):
        # Dcore_from_cdd2_registry=False: this test is about geometry and yaml, and a random
        # config's cdd2 kernel is generally not compiled into the build.
        tree = DedispersionTree(config, itree, Dcore_from_cdd2_registry=False)
        s = tree.to_yaml_string(config, itree)
        t2 = DedispersionTree.from_yaml_string(s, config)

        for f in _TREE_INT_MEMBERS:
            assert getattr(tree, f) == getattr(t2, f), (itree, f)
        for f in _TREE_PF_MEMBERS:
            assert getattr(tree.pf, f) == getattr(t2.pf, f), (itree, 'pf.' + f)

        # dm_min/dm_max/trigger_frequency round-trip LOSSILY: to_yaml() uses yaml-cpp's
        # default ~6-significant-digit precision for doubles. They are print/display values.
        for f in ('dm_min', 'dm_max', 'trigger_frequency'):
            a, b = getattr(tree, f), getattr(t2, f)
            assert abs(a-b) <= 1.0e-5 * max(1.0, abs(a)), (itree, f, a, b)

        fs, fs2 = tree.frequency_subbands, t2.frequency_subbands
        assert list(fs.subband_counts) == list(fs2.subband_counts), itree
        assert np.array_equal(np.asarray(fs.m_to_n), np.asarray(fs2.m_to_n)), itree

        # Re-emitting the parsed tree must reproduce the string exactly. This is what would
        # catch a field emitted but not parsed (the parsed tree would carry a default).
        assert t2.to_yaml_string(config, itree) == s, itree

    # 'itree' is validated, rather than running off the end of the enumeration.
    for bad in (-1, ntrees):
        try:
            DedispersionTree(config, bad, Dcore_from_cdd2_registry=False)
            raise AssertionError(f'DedispersionTree accepted itree={bad} (ntrees={ntrees})')
        except RuntimeError:
            pass


def _decode(plan, token, itree, idm_coarse, itime_coarse):
    """Scalar decode through the plan's tree. DedispersionPlan has no decode_argmax()
    method: decoding goes through DedispersionTree (here) or, vectorized, through
    FrbGrouper (see test_server.py)."""
    return plan.trees[itree].decode_argmax(token, idm_coarse, itime_coarse)


def _test_grouper_tree_rebuild(config, plan, tuples):
    """Rebuild the producer's DedispersionTrees from the plan yaml, as FrbGrouper does.

    FrbGrouper receives (config yaml, plan yaml) at handshake and reconstructs one
    DedispersionTree per per-tree block of the plan yaml -- it never builds a
    DedispersionPlan. This checks that route end to end: every member must survive, and
    decoding through the rebuilt trees must match the freshly-built plan exactly. That is
    the property token decoding on the consumer side rests on.

    Distinct from _test_tree_yaml() above, which round-trips a STANDALONE tree yaml; here
    the trees are extracted from a whole plan yaml, which is the shape FrbGrouper sees.
    """

    cfg_yaml = config.to_yaml_string()
    plan_yaml = plan.to_yaml_string()

    cfg2 = DedispersionConfig.from_yaml_string(cfg_yaml)
    doc = yaml.safe_load(plan_yaml)
    assert doc['ntrees'] == plan.ntrees == cfg2.num_dedispersion_trees

    trees = [DedispersionTree.from_yaml_string(yaml.safe_dump(tn), cfg2)
             for tn in doc['trees']]

    for itree, (t1, t2) in enumerate(zip(plan.trees, trees)):
        for name in ['primary_tree_index', 'early_trigger_level', 'amb_rank', 'dd_rank',
                     'nt_ds', 'Dcore', 'nprofiles', 'ndm_out', 'ndm_wt', 'nt_out', 'nt_wt']:
            assert getattr(t2, name) == getattr(t1, name), f"tree {itree}: {name}"
        for name in ['max_width', 'dm_downsampling', 'time_downsampling',
                     'wt_dm_downsampling', 'wt_time_downsampling', 'num_early_triggers']:
            assert getattr(t2.pf, name) == getattr(t1.pf, name), f"tree {itree}: pf.{name}"
        for name in ['dm_min', 'dm_max', 'trigger_frequency']:
            x1, x2 = getattr(t1, name), getattr(t2, name)
            # Lossy yaml round-trip (to_yaml uses ~6-significant-digit doubles).
            assert abs(x2 - x1) <= 1.0e-4 * max(abs(x1), 1.0), f"tree {itree}: {name}"
        fs1, fs2 = t1.frequency_subbands, t2.frequency_subbands
        assert list(fs2.subband_counts) == list(fs1.subband_counts), f"tree {itree}: subband_counts"
        assert (fs2.N, fs2.M) == (fs1.N, fs1.M), f"tree {itree}: fs.N/M"

    # Every rebuilt tree must agree with the config it travelled with. This is the check
    # FrbGrouper runs at handshake; here it covers the passing path on random configs,
    # without needing a server.
    for t in trees:
        t.check_consistency(cfg2)

    # Decoding through the rebuilt trees must agree exactly with the full plan.
    for (itree, token, idm, itout) in tuples:
        assert (trees[itree].decode_argmax(token, idm, itout)
                == _decode(plan, token, itree, idm, itout)), (itree, token)

    # Negative test: a missing per-tree key must throw (naming the key).
    bad = yaml.safe_load(plan_yaml)['trees'][0]
    bad['Dcore_renamed'] = bad.pop('Dcore')
    try:
        DedispersionTree.from_yaml_string(yaml.safe_dump(bad), cfg2)
        raise AssertionError("DedispersionTree.from_yaml_string() should have thrown"
                             " (missing Dcore)")
    except RuntimeError:
        pass

    return trees


####################################################################################################


def _make_random_config(max_toplevel_rank=6, nbeams=6):
    """Random config with nbatches == 1 and enough beams to pack P1/P2/P3 probes."""

    for _ in range(200):
        config = DedispersionConfig.make_random(max_toplevel_rank=max_toplevel_rank)
        config.beams_per_gpu = nbeams
        config.beams_per_batch = nbeams
        config.num_active_batches = 1
        try:
            config.validate()
        except RuntimeError:
            continue
        return config

    raise RuntimeError("test_decode_argmax: failed to generate a valid config in 200 attempts")


def _num_chunks(plan, r_top, nt_in):
    """Chunk count covering max dispersion depth + peak-finding reach, in full-res samples.

    This is a CORRECTNESS requirement of probe P3, not just a settling nicety: P3's
    completeness needs the simulated span to cover the trial support's full extent.
    """
    depth = 0
    for tree in plan.trees:
        wmax = tree.pf.max_width
        tpad = max(2 * wmax, 4)
        ds = 2 ** (r_top - tree.early_trigger_level) + 4 * wmax + tpad   # downsampled samples
        depth = max(depth, ds * 2 ** tree.primary_tree_index)            # full-res samples
    return depth // nt_in + 2


def _fresh_rdd(plan):
    """ReferenceDedisperser in tree-domain-input mode, with all pf weights = 1."""
    rdd = ReferenceDedisperser(plan, sophistication=0, tree_domain_input=True)
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


def _membership_sweep(plan, tree_bands, C, chans):
    B = plan.beams_per_batch
    nt_in = plan.nt_in

    for i0 in range(0, len(chans), B):
        batch = chans[i0 : i0 + B]

        rdd = _fresh_rdd(plan)
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


def _sample_tuples(plan, kinfo, interesting_ms, ntuples):
    """Return stratified-ish random tuples.

    Biases m toward subband/fine-dm extremes, p and t toward their extremes, cells
    toward corners. Note 'm' here is the token's m-field, i.e. m_ext = (m << K) | mu when
    the tree has xdm_rank() = K > 0, so sweeping it sweeps the extra-DM index too."""

    def _pick(lo_hi_n):
        return random.choice(lo_hi_n)

    tuples = []
    for _ in range(ntuples):
        itree = random.randrange(plan.ntrees)
        M, P, Dout, Dcore = kinfo[itree]
        tree = plan.trees[itree]

        m = random.choice(interesting_ms[itree] + [random.randrange(M)])
        p = _pick([0, P - 1, random.randrange(P)])
        lpf = (p - 1) // 3 if p else 0
        dt = min(Dcore, 2 ** lpf)
        nsamp = Dout // dt
        t = _pick([0, nsamp - 1, random.randrange(nsamp)]) * dt
        token = (m << 16) | (p << 8) | t

        idm = _pick([0, tree.ndm_out - 1, random.randrange(tree.ndm_out)])
        itout = _pick([0, tree.nt_out - 1, random.randrange(tree.nt_out)])
        tuples.append((itree, token, idm, itout))

    return tuples


def _probe_tuples(plan, r_top, C, tuples):
    B = plan.beams_per_batch
    nt_in = plan.nt_in
    nchan = 2 ** r_top
    c_eval = C - 1
    per_run = max(B // 3, 1)

    for i0 in range(0, len(tuples), per_run):
        run_tuples = tuples[i0 : i0 + per_run]
        dec = [_decode(plan, tok, it, idm, ito) for (it, tok, idm, ito) in run_tuples]

        # Global (multi-chunk) positions of the decoded trailing edges (EXCLUSIVE: the
        # last summed sample is tlo-1 / thi-1); the warmup formula in _num_chunks()
        # guarantees these land inside the simulated span.
        for (fmin, fmax, tlo, thi, p), (it, tok, idm, ito) in zip(dec, run_tuples):
            assert 0 <= fmin < fmax < nchan
            assert tlo <= thi <= nt_in
            assert c_eval * nt_in + tlo - 1 >= 0, "test bug: warmup depth insufficient"

        rdd = _fresh_rdd(plan)
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
    M, P, Dout, Dcore = kinfo[itree]
    tree = plan.trees[itree]

    def expect_throw(*args):
        try:
            _decode(plan, *args)
        except RuntimeError:
            return
        raise AssertionError(f"decode_argmax{args} should have thrown")

    # M is the peak-finder's M_ext (see test_decode_argmax), so this is the smallest
    # out-of-range m-field, not (fs.M << 16), which is a VALID token when xdm_rank() > 0.
    expect_throw(M << 16, itree, 0, 0)          # m out of range
    expect_throw(P << 8, itree, 0, 0)           # p out of range
    if Dout < 256:
        expect_throw(Dout, itree, 0, 0)         # t out of range

    for p in range(P):
        lpf = (p - 1) // 3 if p else 0
        if min(Dcore, 2 ** lpf) > 1:
            expect_throw((p << 8) | 1, itree, 0, 0)   # t not divisible by dt
            break

    # (No itree-out-of-range case: decoding is a DedispersionTree method, so a caller
    # indexes plan.trees directly and an out-of-range itree is an IndexError from the list
    # rather than a decode error. The C++ range check on batch decode is covered in
    # test_server.py.)
    expect_throw(0, itree, tree.ndm_out, 0)     # idm_coarse out of range
    expect_throw(0, itree, 0, tree.nt_out)      # itime_coarse out of range


def _test_pf_kernel_quantization(ntrials=8):
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
    nt_in = 512                                         # multiple of 32 (fp32) and of Dout
    nt_out = nt_in // dout
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
                beams_per_batch=1, total_beams=1, ndm_out=1, ndm_wt=1,
                nt_out=nt_out, nt_in=nt_in, nt_wt=nt_out, Dcore=dcore)
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
    _test_tree_yaml(config)
    plan = DedispersionPlan(config)
    r_top = config.toplevel_tree_rank
    nt_in = plan.nt_in
    B = plan.beams_per_batch
    nchan = 2 ** r_top
    C = _num_chunks(plan, r_top, nt_in)

    atomic_print(f"test_decode_argmax: r_top={r_top}, nt_in={nt_in}, ntrees={plan.ntrees}, "
                 f"nbeams={B}, nchunks={C}")

    # Per-tree (M_ext, P, Dout, Dcore), from a scout ReferenceDedisperser.
    #
    # NOTE the first element is M_ext = (fs.M << K) with K = tree.xdm_rank(), i.e. the range
    # of the argmax token's m-field, NOT the tree's multiplet count fs.M. The tokens built
    # below from 0 <= m < M_ext sweep the (multiplet, extra-DM) pairs m_ext = (m << K) | mu,
    # which is precisely what decode_argmax() has to take apart -- and (M_ext << 16) is the
    # first out-of-range value, not (fs.M << 16).
    scout = ReferenceDedisperser(plan, sophistication=0, tree_domain_input=True)
    kinfo = [(k.M_ext, k.P, k.Dout, k.Dcore) for k in scout.pf_kernels]
    del scout

    # Cross-check that relation, and report the per-tree K: a run which happened to generate
    # no K > 0 tree covers strictly less, and that should be visible rather than silent.
    xdm_ranks = []
    for itree in range(plan.ntrees):
        tree = plan.trees[itree]
        xdm_ranks.append(tree.xdm_rank())
        assert kinfo[itree][0] == (tree.frequency_subbands.M << tree.xdm_rank())
    atomic_print(f'test_decode_argmax: xdm_rank by tree = {xdm_ranks}')

    # Cross-check: tree.Dcore (the decode-facing copy) must match the reference
    # peak-finders' Dcore (which flows through stage2_pf_params).
    for itree in range(plan.ntrees):
        assert plan.trees[itree].Dcore == kinfo[itree][3]

    # cdd2_kernel_required=False: no registry query; default Dcore = pf.time_downsampling.
    p0 = DedispersionPlan(config, cdd2_kernel_required=False)
    assert not p0.cdd2_kernel_required
    for tr in p0.trees:
        assert tr.Dcore == tr.pf.time_downsampling

    _check_bad_tokens(plan, kinfo)

    # Per tree: one token per distinct decoded band (fmin, fmax), i.e. per subband. (Several
    # m-field values map to one band -- the fine dms of a multiplet run, and with
    # xdm_rank() > 0 the extra-DM index as well -- so the dict dedupes them.)
    # Also collect the first/last m-field value of each band, used to bias the P1/P2/P3
    # tuple sampling toward the extremes.
    tree_bands = []
    interesting_ms = []
    for itree in range(plan.ntrees):
        M = kinfo[itree][0]
        first, last = {}, {}
        for m in range(M):
            fmin, fmax, _, _, _ = _decode(plan, m << 16, itree, 0, 0)
            first.setdefault((fmin, fmax), m)
            last[(fmin, fmax)] = m
        tree_bands.append([(m << 16, fmn, fmx) for (fmn, fmx), m in first.items()])
        interesting_ms.append(sorted(set(first.values()) | set(last.values())))

    # Membership sweep channels: subband edges +-1 (off-by-one killers) + a few random.
    chans = set()
    for bands in tree_bands:
        for _, fmn, fmx in bands:
            chans.update(c for c in (fmn - 1, fmn, fmx, fmx + 1) if 0 <= c < nchan)
    chans.update(random.sample(range(nchan), min(4, nchan)))
    if len(chans) > 2 * B:   # cap at 2 pipeline runs; random subsampling covers the rest across iterations
        chans = random.sample(sorted(chans), 2 * B)
    _membership_sweep(plan, tree_bands, C, sorted(chans))

    # P1/P2/P3 probes on sampled tuples (2 pipeline runs).
    tuples = _sample_tuples(plan, kinfo, interesting_ms, ntuples=2 * max(B // 3, 1))
    _probe_tuples(plan, r_top, C, tuples)

    # Round-trip test of the per-tree yaml, as FrbGrouper does (reuses the sampled tuples).
    _test_grouper_tree_rebuild(config, plan, tuples)

    # Vectorized decode bindings (batch == scalar loop; full plan == incomplete plan).
