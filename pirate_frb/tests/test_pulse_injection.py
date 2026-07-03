"""Tests for SinglePulse injection / semantics (dispatched from 'pirate_frb test --sim').

test_pulse_injection(): builds a frame of calibrated Gaussian noise with an injected pulse,
dequantizes, and checks:
  - the -8 sentinel is never produced;
  - the pulse is present at the right (freq, time) with roughly the right amplitude (a matched
    filter against the expected pulse); a reversed frequency mapping or wrong dt_sp would give ~0;
  - the off-pulse residual variance matches the per-zone noise variance;
  - the consistency + precondition checks throw on misuse.

test_negative_arrival_times(): checks that SinglePulse handles pulses whose arrival extends to
t < 0 (negative freq_it0) -- the it_start/it_end invariant, integer-shift equivalence, and the
add_to_timestream(out, out_it0) span contract.
"""

import numpy as np

from ..core import AssembledFrameAllocator, SlabAllocator, XEngineMetadata
from ..simpulse import SinglePulse


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
    """One AssembledFrame from a fresh allocator. Returns (frame, allocator); keep the allocator
    alive (it owns the frame's slab)."""
    per_frame = nfreq * (ntime // 256) * 4 + nfreq * (ntime // 2)
    slab = SlabAllocator("af_rhost", 2 * nbeams * per_frame)
    alloc = AssembledFrameAllocator(slab, num_consumers=1, time_samples_per_chunk=ntime)
    alloc.initialize_metadata(xmd)
    alloc.initialize_initial_chunk(0)
    fset = alloc.get_frame_set(0)
    return fset.frames[0], alloc


def _single_pulse(edges, variances, time_sample_ms, dm=10.0, snr=40.0):
    return SinglePulse(dm=dm, sm=0.0, intrinsic_width=2.0e-3, spectral_index=0.0,
                       undispersed_arrival_time_sec=0.1, time_sample_ms=time_sample_ms, snr=snr,
                       freq_edges_MHz=edges, freq_variances=variances)


def test_pulse_injection():
    print("  test_pulse_injection()...")

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
    sp = _single_pulse(edges, variances, frame_dt_ms)
    assert 0 <= sp.it_start and sp.it_end <= ntime, \
        f"pulse range [{sp.it_start},{sp.it_end}) does not fit in [0,{ntime}); lower dm"

    frame, _alloc = _make_frame(xmd, len(beam_ids), nfreq, ntime)
    frame.randomize(normalize=True, gaussian=True, sp=sp, dt_sp=dt_sp)

    v = _unpack_int4(frame)
    assert not (v == -8).any(), "gaussian+pulse produced a -8 sentinel"

    deq = _dequantize(frame)
    expected = _expected_pulse(sp, nfreq, ntime, dt_sp)
    assert expected.any(), "expected pulse is all-zero (bad test setup)"

    # Matched-filter amplitude ~1 iff the pulse landed at the right (freq, time) with the right
    # scale. Reversed frequency mapping or a wrong dt_sp -> ~0. Loose bounds: int4 quantization
    # and any saturation of bright samples pull it somewhat below 1.
    amp = float(np.sum(deq * expected) / np.sum(expected * expected))
    assert 0.3 < amp < 2.0, f"matched-filter amplitude {amp:.3f} out of range (mapping/scale bug?)"

    # Off-pulse residual (= dequantized noise) variance should match the per-zone noise variance V.
    resid = deq - expected
    rvar = float(resid[expected == 0.0].var())
    assert abs(rvar / V - 1.0) < 0.15, f"off-pulse residual variance {rvar:.4f} != V={V}"

    print(f"    injected pulse: it=[{sp.it_start},{sp.it_end}), matched-filter amp={amp:.3f}, "
          f"off-pulse var={rvar:.4f} (V={V:.3f}) -- ok")

    # ---- consistency / precondition checks must throw ----
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

    print("    consistency/precondition checks all threw -- ok")


def test_negative_arrival_times():
    """SinglePulse handles pulses whose arrival extends to t < 0 (negative freq_it0).

    Checks the it_start/it_end invariant, that a pulse and its integer-sample-shifted copy are
    identical up to the shift (nothing is discarded at t < 0), and add_to_timestream()'s
    (out, out_it0) span contract.
    """
    print("  test_negative_arrival_times()...")

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

    print(f"    it=[{sp_a.it_start},{sp_a.it_end}): shift-equivalence + "
          f"add_to_timestream span contract -- ok")
