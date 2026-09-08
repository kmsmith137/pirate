"""Tests for SinglePulse injection / semantics (dispatched from 'pirate_frb test --sim').

test_pulse_injection(): builds frames of calibrated Gaussian noise with an injected pulse,
dequantizes, and checks:
  - the -8 sentinel is never produced;
  - the pulse is present at the right (freq, time) with roughly the right amplitude (a matched
    filter against the expected pulse); a reversed frequency mapping or wrong dt_sp would give ~0;
  - the off-pulse residual variance matches the per-zone noise variance;
  - the consistency + precondition checks throw on misuse.
  The pulse is injected at three (randomized) arrival times per call -- straddling t < 0
  (a negative arrival time, early samples clipped), fully inside the frame, and straddling
  t = ntime -- so inject_single_pulse()'s [0, ntime) clipping is exercised on both ends.

test_pulse_invariants(): checks SinglePulse's sparse-representation invariants, using a pulse
whose arrival extends to t < 0 (mixed-sign freq_it0) to stress the negative-time regime: the
it_start/it_end bracketing invariant, integer-shift equivalence, shift_samples(), and the
add_to_timestream(out, out_it0) span contract.
"""
import contextlib
import io
import os
import tempfile

import numpy as np

from ..core import AssembledFrame, AssembledFrameAllocator, BumpAllocator, SlabAllocator, XEngineMetadata
from ..make_simulated_acq import (
    _dedispersed_reference_frequency_MHz,
    _make_arg_parser,
    _normalize_burst_parameters,
    _toa_to_infinite_arrivals,
    main as make_simulated_acq_main,
    make_simulated_acq,
)
from ..simpulse import SinglePulse, dispersion_delay
from ..utils import atomic_print


def _unpack_int4(frame):
    """(nfreq, ntime) signed int4 values of 'frame', sign-extended to int16."""
    b = np.asarray(frame.data).view(np.uint8).ravel()
    lo = (b & 0xF).astype(np.int16)
    hi = ((b >> 4) & 0xF).astype(np.int16)
    lo[lo >= 8] -= 16
    hi[hi >= 8] -= 16
    v = np.empty(2 * b.size, np.int16)
    v[0::2] = lo   # low nibble = even index
    v[1::2] = hi
    return v.reshape(frame.nfreq, frame.ntime)


def _dequantize(frame):
    """out[f,t] = scale[f, t//256] * v[f,t] + offset[f, t//256]."""
    v = _unpack_int4(frame).astype(np.float64)
    so = np.asarray(frame.scales_offsets).astype(np.float64)   # (nfreq, mpc, 2) = {scale, offset}
    scale = np.repeat(so[:, :, 0], 256, axis=1)                # (nfreq, ntime)
    offset = np.repeat(so[:, :, 1], 256, axis=1)
    return scale * v + offset


def _expected_pulse(sp, nfreq, ntime, dt_sp):
    """Dense (nfreq, ntime) expected pulse (post-scaled units) from sp's sparse arrays.

    Direct mapping (pulse channel f -> frame row f); frame time it maps to pulse time it + dt_sp.
    """
    it0 = np.asarray(sp.freq_it0)
    nt = np.asarray(sp.freq_nt)
    off = np.asarray(sp.freq_sd_off)
    sd = np.asarray(sp.sparse_data)
    out = np.zeros((nfreq, ntime), np.float64)
    for f in range(nfreq):
        for k in range(int(nt[f])):
            t = int(it0[f]) + k - dt_sp
            if 0 <= t < ntime:
                out[f, t] = sd[int(off[f]) + k]
    return out


def _make_frame(xmd, nbeams, nfreq, ntime):
    """Return one AssembledFrame from a fresh allocator.

    Returns (frame, allocator); keep the allocator alive (it owns the frame's
    slab)."""
    per_frame = nfreq * (ntime // 256) * 4 + nfreq * (ntime // 2)
    bump = BumpAllocator("af_rhost", 2 * nbeams * per_frame)
    slab = SlabAllocator(bump)
    alloc = AssembledFrameAllocator(slab, num_consumers=1, time_samples_per_chunk=ntime, throw_exception_if_empty=False)
    alloc.initialize_metadata(xmd)
    alloc.initialize_initial_chunk(0)
    fset = alloc.get_frame_set(0)
    return fset.frames[0], alloc


def _single_pulse(edges, variances, time_sample_ms, dm=10.0, snr=40.0, uat_sec=0.1):
    return SinglePulse(dm=dm, sm=0.0, intrinsic_width=2.0e-3, spectral_index=0.0,
                       undispersed_arrival_time_sec=uat_sec, time_sample_ms=time_sample_ms, snr=snr,
                       freq_edges_MHz=edges, freq_variances=variances)


def test_pulse_injection():
    atomic_print("  test_pulse_injection()...")

    nfreq = 64
    flo, fhi = 400.0, 800.0
    beam_ids = [0, 1]
    ntime = 2048
    dt_sp = 0

    # make_fiducial defaults noise_variance to 1.0 per zone; use that for V.
    xmd = XEngineMetadata.make_fiducial([nfreq], [flo, fhi], beam_ids, 0.983)
    V = float(list(xmd.noise_variance)[0])
    xmd.validate()

    # The pulse must use the frame's ACTUAL dt (make_fiducial rounds seq_per_frb_time_sample).
    frame_dt_ms = xmd.dt_ns_per_seq * xmd.seq_per_frb_time_sample / 1.0e6
    edges = np.linspace(flo, fhi, nfreq + 1)
    variances = np.full(nfreq, V)

    def _inject_and_check(sp, label):
        """Inject 'sp' into a fresh frame, dequantize, and run the checks.

        The checks are pulse-content + noise. 'expected' is clipped to [0, ntime)
        exactly like inject_single_pulse(), so this
        works whether or not the pulse straddles a frame edge."""
        frame, _alloc = _make_frame(xmd, len(beam_ids), nfreq, ntime)
        frame.randomize(normalize=True, gaussian=True, sp=sp, dt_sp=dt_sp)

        v = _unpack_int4(frame)
        assert not (v == -8).any(), f"[{label}] gaussian+pulse produced a -8 sentinel"

        deq = _dequantize(frame)
        expected = _expected_pulse(sp, nfreq, ntime, dt_sp)
        assert expected.any(), f"[{label}] expected pulse is all-zero (bad test setup)"

        # Matched-filter amplitude ~1 iff the (in-frame part of the) pulse landed at the right
        # (freq, time) with the right scale. Reversed frequency mapping or a wrong dt_sp -> ~0.
        # Loose bounds: int4 quantization and any saturation of bright samples pull it somewhat
        # below 1. Each placement keeps >= ceil(L/2) samples in-frame, so the estimate is robust.
        amp = float(np.sum(deq * expected) / np.sum(expected * expected))
        assert 0.3 < amp < 2.0, f"[{label}] matched-filter amplitude {amp:.3f} out of range (mapping/scale bug?)"

        # Off-pulse residual (= dequantized noise) variance should match the per-zone noise variance V.
        resid = deq - expected
        rvar = float(resid[expected == 0.0].var())
        assert abs(rvar / V - 1.0) < 0.15, f"[{label}] off-pulse residual variance {rvar:.4f} != V={V}"

        atomic_print(f"    {label}: it=[{sp.it_start},{sp.it_end}) vs frame [0,{ntime}), "
                     f"matched-filter amp={amp:.3f}, off-pulse var={rvar:.4f} (V={V:.3f}) -- ok")

    # Learn the pulse's grid span L (with dt_sp=0, frame-time == pulse-time) from a uat=0
    # reference. Since uat += K*dt shifts every freq_it0 -- hence it_start -- by exactly K
    # integer samples, we can then place it_start at any target by choosing uat = K*dt.
    sp0 = _single_pulse(edges, variances, frame_dt_ms, uat_sec=0.0)
    it_start0, L = int(sp0.it_start), int(sp0.it_end) - int(sp0.it_start)
    assert 2 <= L <= ntime, f"pulse span L={L} does not fit ntime={ntime}; adjust dm"

    # Three placements exercising inject_single_pulse()'s [0, ntime) clipping: straddling t < 0
    # (negative arrival -> early samples clipped), fully inside, and straddling t = ntime. Each
    # keeps >= ceil(L/2) samples in-frame (matched filter stays robust). SinglePulse's own
    # negative-time invariants are covered separately by test_pulse_invariants().
    lo, hi = L // 2, L - L // 2   # hi == ceil(L/2)
    targets = {
        "straddle t<0":     np.random.randint(-lo, 0),                        # it_start in [-lo, -1]
        "inside":           np.random.randint(0, ntime - L + 1),              # it_start in [0, ntime-L]
        "straddle t=ntime": np.random.randint(ntime - L + 1, ntime - hi + 1),  # it_end in (ntime, ntime+lo]
    }
    for label, s in targets.items():
        s = int(s)
        sp = _single_pulse(edges, variances, frame_dt_ms, uat_sec=(s - it_start0) * frame_dt_ms * 1.0e-3)
        assert int(sp.it_start) == s, f"[{label}] placement failed: it_start={sp.it_start} != {s}"
        _inject_and_check(sp, label)

    # ---- consistency / precondition checks must throw ----
    # Use a plain (fully-in-frame) pulse + a fresh frame; the checks throw before touching buffers.
    sp = _single_pulse(edges, variances, frame_dt_ms)
    frame, _alloc = _make_frame(xmd, len(beam_ids), nfreq, ntime)

    def _expect_throw(desc, fn):
        try:
            fn()
        except RuntimeError:
            return
        raise AssertionError(f"expected RuntimeError: {desc}")

    _expect_throw("gaussian=False with sp",  lambda: frame.randomize(True, False, sp=sp, dt_sp=0))
    _expect_throw("normalize=False with sp", lambda: frame.randomize(False, True, sp=sp, dt_sp=0))

    sp_bad_nf = _single_pulse(np.linspace(flo, fhi, nfreq // 2 + 1), np.full(nfreq // 2, V), frame_dt_ms)
    _expect_throw("nfreq mismatch", lambda: frame.randomize(True, True, sp=sp_bad_nf, dt_sp=0))

    sp_bad_dt = _single_pulse(edges, variances, frame_dt_ms * 1.5)
    _expect_throw("time_sample_ms mismatch", lambda: frame.randomize(True, True, sp=sp_bad_dt, dt_sp=0))

    sp_bad_var = _single_pulse(edges, np.full(nfreq, V * 2.0), frame_dt_ms)
    _expect_throw("freq_variances mismatch", lambda: frame.randomize(True, True, sp=sp_bad_var, dt_sp=0))

    atomic_print("    consistency/precondition checks all threw -- ok")

    # ---- generalized multi-pulse path ----
    def _pulse_at_start(start, burst_snr=12.0):
        ref = _single_pulse(edges, variances, frame_dt_ms, snr=burst_snr, uat_sec=0.0)
        arrival = (int(start) - int(ref.it_start)) * frame_dt_ms * 1.0e-3
        out = _single_pulse(edges, variances, frame_dt_ms, snr=burst_snr, uat_sec=arrival)
        assert int(out.it_start) == int(start)
        return out

    # Zero pulses is Gaussian noise only, and the old one-pulse method above remains supported.
    noise_frame, _alloc = _make_frame(xmd, len(beam_ids), nfreq, ntime)
    noise_frame.randomize_many(normalize=True, gaussian=True, pulses=[], dt_sp=0)
    assert not (_unpack_int4(noise_frame) == -8).any()

    # Two separated pulses must both survive one frame randomization.
    sep_a = _pulse_at_start(64)
    sep_b = _pulse_at_start(ntime - L - 64)
    assert sep_a.it_end < sep_b.it_start
    sep_frame, _alloc = _make_frame(xmd, len(beam_ids), nfreq, ntime)
    sep_frame.randomize_many(normalize=True, gaussian=True, pulses=[sep_a, sep_b], dt_sp=0)
    sep_deq = _dequantize(sep_frame)
    for label, burst in (("separated A", sep_a), ("separated B", sep_b)):
        expected = _expected_pulse(burst, nfreq, ntime, 0)
        amp = float(np.sum(sep_deq * expected) / np.sum(expected * expected))
        assert 0.3 < amp < 1.7, f"[{label}] matched-filter amplitude {amp:.3f}"

    # Identical overlapping pulses are a sharp replacement-vs-sum check: replacing one pulse would
    # give amplitude ~0.5 against the expected sum. The residual variance checks that overlap gets
    # one noise contribution and one quantization, rather than independent noise per pulse.
    overlap = _pulse_at_start(ntime // 2 - L // 2, burst_snr=6.0)
    overlap_expected = 2.0 * _expected_pulse(overlap, nfreq, ntime, 0)
    overlap_frame, _alloc = _make_frame(xmd, len(beam_ids), nfreq, ntime)
    overlap_frame.randomize_many(normalize=True, gaussian=True, pulses=[overlap, overlap], dt_sp=0)
    overlap_deq = _dequantize(overlap_frame)
    overlap_amp = float(np.sum(overlap_deq * overlap_expected) /
                        np.sum(overlap_expected * overlap_expected))
    assert 0.70 < overlap_amp < 1.30, f"overlap sum amplitude {overlap_amp:.3f}"
    overlap_mask = overlap_expected != 0.0
    overlap_rvar = float((overlap_deq - overlap_expected)[overlap_mask].var())
    assert 0.45 < overlap_rvar / V < 1.55, \
        f"overlap residual variance {overlap_rvar:.4f} indicates noise was not added once"

    # A pulse crossing a chunk seam must be visible with each chunk's dt_sp offset.
    seam = _pulse_at_start(ntime - L // 2, burst_snr=20.0)
    assert seam.it_start < ntime < seam.it_end
    for ichunk in (0, 1):
        seam_frame, _alloc = _make_frame(xmd, len(beam_ids), nfreq, ntime)
        seam_frame.randomize_many(
            normalize=True, gaussian=True, pulses=[seam], dt_sp=ichunk * ntime
        )
        expected = _expected_pulse(seam, nfreq, ntime, ichunk * ntime)
        assert expected.any(), f"seam pulse missing from chunk {ichunk} test setup"
        amp = float(np.sum(_dequantize(seam_frame) * expected) / np.sum(expected * expected))
        assert 0.3 < amp < 1.8, f"seam chunk {ichunk} amplitude {amp:.3f}"

    # Every pulse is checked before scales_offsets or data is changed.
    untouched, _alloc = _make_frame(xmd, len(beam_ids), nfreq, ntime)
    untouched.randomize_many(normalize=True, gaussian=True, pulses=[], dt_sp=0)
    data_before = np.asarray(untouched.data).copy()
    so_before = np.asarray(untouched.scales_offsets).copy()
    _expect_throw(
        "bad pulse in collection",
        lambda: untouched.randomize_many(
            normalize=True, gaussian=True, pulses=[sp, sp_bad_dt], dt_sp=0
        ),
    )
    assert np.array_equal(np.asarray(untouched.data), data_before)
    assert np.array_equal(np.asarray(untouched.scales_offsets), so_before)
    atomic_print(
        f"    multi-pulse: separated + overlap + seam -- ok "
        f"(overlap amp={overlap_amp:.3f}, residual var={overlap_rvar:.3f})"
    )

def test_multi_burst_cli():
    """Argument broadcasting, timestamp conversion, and generated-ASDF compatibility."""
    atomic_print("  test_multi_burst_cli()...")

    nfreq = 64
    flo, fhi = 400.0, 800.0
    beam_ids = [0, 1]
    xmd = XEngineMetadata.make_fiducial([nfreq], [flo, fhi], beam_ids, 0.983)
    xmd.validate()
    # ---- Python argument handling and TOA conversion ----
    toas, widths, burst_snrs = _normalize_burst_parameters(
        toa=[5, 7, 12], arrival_sec=None, width_ms=[2], snr=[30]
    )
    assert toas == [5.0, 7.0, 12.0]
    assert widths == [2.0, 2.0, 2.0]
    assert burst_snrs == [30.0, 30.0, 30.0]

    toas, widths, burst_snrs = _normalize_burst_parameters(
        toa=[5, 7, 12], arrival_sec=None, width_ms=[1, 2, 1], snr=[30, 20, 40]
    )
    assert widths == [1.0, 2.0, 1.0] and burst_snrs == [30.0, 20.0, 40.0]

    for option, kwargs in (
        ("--width-ms", dict(width_ms=[1, 2], snr=[30])),
        ("--snr", dict(width_ms=[1], snr=[30, 20])),
    ):
        try:
            _normalize_burst_parameters(toa=[5, 7, 12], arrival_sec=None, **kwargs)
        except ValueError as exc:
            assert option in str(exc) and "one value or 3 values" in str(exc)
        else:
            raise AssertionError(f"invalid {option} length did not fail")

    parsed = _make_arg_parser().parse_args([
        "metadata.yml", "out", "--dm", "100", "--toa", "5", "7", "12",
        "--width-ms", "1", "2", "1", "--snr", "30", "20", "40",
    ])
    assert parsed.toa == [5.0, 7.0, 12.0]
    assert parsed.width_ms == [1.0, 2.0, 1.0]
    assert parsed.snr == [30.0, 20.0, 40.0]

    # main() validates list lengths before it attempts to read metadata or make an output directory.
    invalid_out = os.path.join(tempfile.gettempdir(), "pirate_invalid_multi_burst_should_not_exist")
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr):
        try:
            make_simulated_acq_main([
                "/metadata/does/not/exist.yml", invalid_out,
                "--toa", "1", "2", "3", "--snr", "10", "20",
            ])
        except SystemExit as exc:
            assert exc.code == 2
        else:
            raise AssertionError("invalid CLI list length did not fail")
    assert "--snr expects either one value or 3 values" in stderr.getvalue()
    assert not os.path.exists(invalid_out)

    reference_frequency_MHz = _dedispersed_reference_frequency_MHz(xmd)
    assert reference_frequency_MHz == flo
    arrivals, delay = _toa_to_infinite_arrivals([5.0, 7.0, 12.0], 100.0, 300.0)
    assert abs(delay - float(dispersion_delay(100.0, 300.0))) < 1.0e-12
    assert abs(delay - 4.6098) < 5.0e-4
    assert np.allclose(arrivals, np.asarray([5.0, 7.0, 12.0]) - delay)

    # A short generated multi-burst acquisition remains valid AssembledFrame ASDF.
    with tempfile.TemporaryDirectory(prefix="pirate_multi_burst_") as tmpdir:
        metadata_path = os.path.join(tmpdir, "metadata.yml")
        outdir = os.path.join(tmpdir, "acq")
        with open(metadata_path, "w", encoding="utf-8") as f:
            f.write(xmd.to_yaml_string())
        make_simulated_acq(
            metadata_path,
            outdir,
            nchunks=2,
            ntime=256,
            dm=1.0,
            toa=[0.15, 0.30],
            width_ms=[1.0],
            snr=[12.0, 9.0],
        )
        for ichunk in (0, 1):
            filename = os.path.join(outdir, f"frame_b{beam_ids[0]}_t{ichunk}.asdf")
            loaded = AssembledFrame.from_asdf(filename)
            assert loaded.nfreq == nfreq and loaded.ntime == 256
            assert loaded.beam_id == beam_ids[0] and loaded.time_chunk_index == ichunk

    atomic_print("    CLI broadcasting + metadata-derived TOA + ASDF round-trip -- ok")


def test_pulse_invariants():
    """SinglePulse's sparse-representation invariants, stressed in the negative-time regime.

    Uses a pulse whose arrival extends to t < 0 (mixed-sign freq_it0) to check: the
    it_start/it_end bracketing invariant; that a pulse and its integer-sample-shifted copy are
    identical up to the shift (nothing is discarded at t < 0); shift_samples(); and
    add_to_timestream()'s (out, out_it0) span contract.
    """
    atomic_print("  test_pulse_invariants()...")

    nfreq = 64
    dt_ms = 1.0
    edges = np.linspace(400.0, 800.0, nfreq + 1)
    variances = np.full(nfreq, 1.0)

    # dm=1, uat=0: with intrinsic_width=2 ms, the pulse starts at t ~ (dispersion delay - 8 ms),
    # negative in the top (high-freq) channels and positive (by many samples) in the bottom
    # channels -- exercising both signs of freq_it0.
    common = dict(dm=1.0, sm=0.0, intrinsic_width=2.0e-3, spectral_index=0.0,
                  time_sample_ms=dt_ms, snr=20.0, freq_edges_MHz=edges, freq_variances=variances)

    sp_a = SinglePulse(undispersed_arrival_time_sec=0.0, **common)
    it0_a = np.asarray(sp_a.freq_it0)
    nt_a = np.asarray(sp_a.freq_nt)
    assert (nt_a > 0).all()   # no subband restriction -> every channel active
    assert (it0_a < 0).any() and (it0_a >= 0).any(), \
        f"test setup: expected mixed-sign freq_it0, got range [{it0_a.min()}, {it0_a.max()}]"

    # it_start/it_end bracket every channel: it_start <= freq_it0 <= freq_it0+freq_nt <= it_end.
    assert sp_a.it_start == int(it0_a.min())
    assert sp_a.it_end == int((it0_a + nt_a).max())
    assert (sp_a.it_start <= it0_a).all() and ((it0_a + nt_a) <= sp_a.it_end).all()

    # Pulse B: identical params but arrival shifted later by an integer K samples. Nothing is
    # discarded at t < 0, so B is exactly A shifted: same freq_nt/sparse_data, and freq_it0
    # (hence it_start/it_end) offset by exactly K.
    K = 100
    sp_b = SinglePulse(undispersed_arrival_time_sec=K * dt_ms * 1.0e-3, **common)
    assert np.array_equal(np.asarray(sp_b.freq_it0), it0_a + K), "freq_it0 not shifted by K"
    assert np.array_equal(np.asarray(sp_b.freq_nt), nt_a)
    assert sp_b.it_start == sp_a.it_start + K and sp_b.it_end == sp_a.it_end + K
    sd_a, sd_b = np.asarray(sp_a.sparse_data), np.asarray(sp_b.sparse_data)
    assert np.allclose(sd_a, sd_b, rtol=1.0e-5, atol=1.0e-6 * np.abs(sd_b).max()), \
        "sparse_data mismatch between shifted copies of the same pulse"

    # shift_samples(K) shifts an existing pulse in place: it must reproduce sp_a shifted by K
    # (freq_it0/it_start/it_end += K, uat += K*dt), with counts and sample VALUES untouched.
    sp_c = SinglePulse(undispersed_arrival_time_sec=0.0, **common)
    sp_c.shift_samples(K)
    assert np.array_equal(np.asarray(sp_c.freq_it0), it0_a + K), "shift_samples didn't shift freq_it0 by K"
    assert sp_c.it_start == sp_a.it_start + K and sp_c.it_end == sp_a.it_end + K
    assert np.array_equal(np.asarray(sp_c.freq_nt), nt_a)                             # counts unchanged
    assert np.array_equal(np.asarray(sp_c.sparse_data), np.asarray(sp_a.sparse_data))  # values unchanged
    assert abs(sp_c.undispersed_arrival_time_sec - K * dt_ms * 1.0e-3) < 1.0e-12

    # add_to_timestream(out, out_it0): render A over an 'out' spanning [it_start, it_end), and B
    # over the same-size window shifted by K. The two dense renders must be identical (same pulse).
    nt = sp_a.it_end - sp_a.it_start
    dense_a = np.zeros((nfreq, nt), np.float32); sp_a.add_to_timestream(dense_a, sp_a.it_start)
    dense_b = np.zeros((nfreq, nt), np.float32); sp_b.add_to_timestream(dense_b, sp_b.it_start)
    assert (dense_a != 0).any()
    assert np.allclose(dense_a, dense_b, rtol=1.0e-5, atol=1.0e-6 * np.abs(dense_a).max())

    # A wider 'out' with a different out_it0 (still covering the range) places the pulse at
    # column (freq_it0 - out_it0) and scales by 'weight'; the uncovered columns stay zero.
    wide = np.zeros((nfreq, nt + 30), np.float32)
    sp_a.add_to_timestream(wide, sp_a.it_start - 10, weight=2.0)
    tol = 1.0e-6 * np.abs(dense_a).max() * 2
    assert np.allclose(wide[:, 10:10+nt], 2.0 * dense_a, rtol=1.0e-5, atol=tol)
    assert np.allclose(wide[:, :10], 0.0) and np.allclose(wide[:, 10+nt:], 0.0)

    # add_to_timestream() raises unless 'out' covers the full pulse range [it_start, it_end).
    def _expect_throw(desc, fn):
        try:
            fn()
        except RuntimeError:
            return
        raise AssertionError(f"expected RuntimeError: {desc}")
    _expect_throw("out starts too late",
                  lambda: sp_a.add_to_timestream(np.zeros((nfreq, nt), np.float32), sp_a.it_start + 1))
    _expect_throw("out ends too early",
                  lambda: sp_a.add_to_timestream(np.zeros((nfreq, nt - 1), np.float32), sp_a.it_start))

    atomic_print(f"    it=[{sp_a.it_start},{sp_a.it_end}): shift-equivalence + "
                 f"add_to_timestream span contract -- ok")
