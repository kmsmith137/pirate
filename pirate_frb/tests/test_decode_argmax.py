"""
Black-box tests of DedispersionPlan.decode_argmax() (run via 'test --amax').

Strategy (see plans/argmax_decoding.md): for a FIXED token, eval_tokens() is a
LINEAR functional of the input array -- the actual dedispersion + peak-finding
computation for that one trial, with the max-reduce removed -- and all its
coefficients are nonnegative (profile coefficients in {1/2, 1}; dedispersion and
downsampling factors are positive; the test sets peak-finding weights to 1).
So for a nonnegative indicator probe, the functional is nonzero IFF the probe
overlaps the trial's support. Probes are injected directly into the toplevel
tree-domain space (ReferenceDedisperser(tree_domain_input=True)), which is the
space decode_argmax() reports in.

Per sampled tuple (itree, token, idm, itout), with decoded (fmin, fmax, tlo, thi):

  P1: one-hot at (fmin, tlo)      -> eval_tokens cell must be nonzero
  P2: one-hot at (fmax, thi)      -> eval_tokens cell must be nonzero
  P3: indicator of the complement region {f < fmin} + {f > fmax} +
      {f == fmin, t > tlo} + {f == fmax, t > thi}
                                  -> eval_tokens cell must be exactly zero

P1+P2+P3 pins the support's channel range and both edge trailing times exactly
(completeness argument in the plan). A constant-probe membership sweep
supplements this with exhaustive fmin/fmax coverage (all tokens/cells at once
per probed channel), and a bare-ReferencePeakFindingKernel sweep covers the
token time-quantization formula (dt = min(Dcore, 2^level)) densely.
"""

import random
import numpy as np

from ..pirate_pybind11 import (DedispersionConfig, DedispersionPlan,
                               ReferenceDedisperser, ReferencePeakFindingKernel)


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
    """Stratified-ish random tuples: bias m toward subband/fine-dm extremes, p and t
    toward their extremes, cells toward corners."""

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
        dec = [plan.decode_argmax(tok, it, idm, ito) for (it, tok, idm, ito) in run_tuples]

        # Global (multi-chunk) positions of the decoded trailing times; the warmup
        # formula in _num_chunks() guarantees these land inside the simulated span.
        for (fmin, fmax, tlo, thi, p), (it, tok, idm, ito) in zip(dec, run_tuples):
            assert 0 <= fmin < fmax < nchan
            assert tlo <= thi < nt_in
            assert c_eval * nt_in + tlo >= 0, "test bug: warmup depth insufficient"

        rdd = _fresh_rdd(plan)
        ia = rdd.input_array
        for c in range(C):
            ia[...] = 0.0
            t0 = c * nt_in
            for k, (fmin, fmax, tlo, thi, p) in enumerate(dec):
                glo = c_eval * nt_in + tlo
                ghi = c_eval * nt_in + thi

                # P1/P2: one-hot probes (beam slots 3k, 3k+1).
                if t0 <= glo < t0 + nt_in:
                    ia[3*k, fmin, glo - t0] = 1.0
                if t0 <= ghi < t0 + nt_in:
                    ia[3*k + 1, fmax, ghi - t0] = 1.0

                # P3: complement-region indicator (beam slot 3k+2).
                ia[3*k + 2, :fmin, :] = 1.0
                ia[3*k + 2, fmax + 1:, :] = 1.0
                lo = glo + 1 - t0
                if lo < nt_in:
                    ia[3*k + 2, fmin, max(lo, 0):] = 1.0
                hi = ghi + 1 - t0
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
                assert out[3*k, idm, ito] > 0, f"P1 failed (tlo not in support): {msg}"
                assert out[3*k + 1, idm, ito] > 0, f"P2 failed (thi not in support): {msg}"
                assert out[3*k + 2, idm, ito] == 0, f"P3 failed (support outside decoded region): {msg}"


####################################################################################################


def _check_bad_tokens(plan, kinfo):
    """decode_argmax() must throw on malformed tokens and out-of-range indices."""

    itree = random.randrange(plan.ntrees)
    M, P, Dout, Dcore = kinfo[itree]
    tree = plan.trees[itree]

    def expect_throw(*args):
        try:
            plan.decode_argmax(*args)
        except RuntimeError:
            return
        raise AssertionError(f"decode_argmax{args} should have thrown")

    expect_throw(M << 16, itree, 0, 0)          # m out of range
    expect_throw(P << 8, itree, 0, 0)           # p out of range
    if Dout < 256:
        expect_throw(Dout, itree, 0, 0)         # t out of range

    for p in range(P):
        lpf = (p - 1) // 3 if p else 0
        if min(Dcore, 2 ** lpf) > 1:
            expect_throw((p << 8) | 1, itree, 0, 0)   # t not divisible by dt
            break

    expect_throw(0, plan.ntrees, 0, 0)          # itree out of range
    expect_throw(0, itree, tree.ndm_out, 0)     # idm_coarse out of range
    expect_throw(0, itree, 0, tree.nt_out)      # itime_coarse out of range


def _test_pf_kernel_quantization(ntrials=8):
    """Kernel-level sweep of the token time-quantization formula, with arbitrary Dcore.

    A bare ReferencePeakFindingKernel (no dedispersion, single full-band multiplet) has
    Tlag = Dsub = 0, so the trailing pf-input sample of token (p, t) at cell tout is
    T = tout*Dout + t + dt - 1 with dt = min(Dcore, 2^level) -- the same arithmetic as
    decode_argmax(). Verify with one-hot / tail probes (fresh kernel per probe, since
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
    plan = DedispersionPlan(config)
    r_top = config.toplevel_tree_rank
    nt_in = plan.nt_in
    B = plan.beams_per_batch
    nchan = 2 ** r_top
    C = _num_chunks(plan, r_top, nt_in)

    print(f"test_decode_argmax: r_top={r_top}, nt_in={nt_in}, ntrees={plan.ntrees}, "
          f"nbeams={B}, nchunks={C}")

    # Per-tree (M, P, Dout, Dcore), from a scout ReferenceDedisperser.
    scout = ReferenceDedisperser(plan, sophistication=0, tree_domain_input=True)
    kinfo = [(k.M, k.P, k.Dout, k.Dcore) for k in scout.pf_kernels]
    del scout

    _check_bad_tokens(plan, kinfo)

    # Per tree: one token per distinct decoded band (fmin, fmax), i.e. per subband.
    # Also collect the first/last multiplet of each band (fine-dm extremes), used to
    # bias the P1/P2/P3 tuple sampling.
    tree_bands = []
    interesting_ms = []
    for itree in range(plan.ntrees):
        M = kinfo[itree][0]
        first, last = {}, {}
        for m in range(M):
            fmin, fmax, _, _, _ = plan.decode_argmax(m << 16, itree, 0, 0)
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
