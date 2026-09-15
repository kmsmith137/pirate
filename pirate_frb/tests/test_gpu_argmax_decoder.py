"""CPU-oracle checks for the readable CuPy argmax decoder."""

import inspect
import os

import numpy as np

from ..GpuArgmaxDecoder import (
    ArgmaxDecodeStatus,
    GpuArgmaxDecodeError,
    GpuArgmaxDecoder,
)
from ..BowtiePeakfinding import EdgeFlag, GpuRawCandidates
from ..pirate_pybind11 import DedispersionConfig, DedispersionPlan


def _make_plan(config_basename="toy.yml"):
    """Return reconstructed geometry and explicit synthetic-producer Dcores.

    These tests author their own tokens for deliberately selected, valid
    time granularities. Production readers must use the Dcores saved by the
    actual dedisperser; reconstructing a plan alone cannot supply them.
    """

    filename = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "data", config_basename,
    ))
    config = DedispersionConfig.from_yaml(filename)
    producer = DedispersionPlan(config)
    dcores = tuple(
        min(1 << (itree % 3), int(tree.nt_ds) // int(tree.nt_out))
        for itree, tree in enumerate(producer.trees)
    )
    plan = DedispersionPlan.from_yaml_string(config, producer.to_yaml_string())
    return plan, dcores


def _valid_rows(plan, dcores):
    """Cover every tree/profile/mu and both ends of its ragged output map."""

    rows = []
    serial = 0
    for itree, tree in enumerate(plan.trees):
        dout = int(tree.nt_ds) // int(tree.nt_out)
        dcore = dcores[itree]
        extra_dm_count = int(tree.dm_downsampling) >> int(tree.frequency_subbands.pf_rank)
        for profile in range(int(tree.nprofiles)):
            level = (profile - 1) // 3 if profile else 0
            dt = min(dcore, 1 << level)
            fine_times = sorted({0, ((dout - 1) // dt) * dt})
            multiplets = sorted({0, int(tree.frequency_subbands.M) - 1})
            for fine_time in fine_times:
                for multiplet in multiplets:
                    for extra_dm in range(extra_dm_count):
                        edge = serial & 1
                        rows.append((
                            np.uint32(fine_time | (profile << 8)
                                      | (multiplet << 16) | (extra_dm << 24)),
                            itree,
                            (int(tree.ndm_out) - 1) if edge else 0,
                            (int(tree.nt_out) - 1) if edge else 0,
                            (-3, 0, 5)[serial % 3],
                        ))
                        serial += 1
    return rows


def _raw_from_rows(cp, rows):
    """Upload test provenance while retaining the production raw schema."""

    n = len(rows)
    return GpuRawCandidates(
        beam_id=cp.asarray(
            [100 + (i & 1) for i in range(n)], dtype=cp.int32
        ),
        source_chunk_index=cp.asarray(
            [row[4] for row in rows], dtype=cp.int64
        ),
        tree=cp.asarray([row[1] for row in rows], dtype=cp.int32),
        idm=cp.asarray([row[2] for row in rows], dtype=cp.int32),
        itime=cp.asarray([row[3] for row in rows], dtype=cp.int32),
        snr=cp.linspace(10.0, 20.0, n, dtype=cp.float32),
        argmax_token=cp.asarray([row[0] for row in rows], dtype=cp.uint32),
        edge_flags=cp.zeros(n, dtype=cp.uint8),
    )


def test_gpu_argmax_decoder_cpu_parity(cuda_device_id=0):
    """Integer fields are exact; physical fields meet the approved tolerance."""

    import cupy as cp

    with cp.cuda.Device(cuda_device_id):
        for config_basename in ("toy.yml", "chord_sb2_et.yml"):
            plan, dcores = _make_plan(config_basename)
            rows = _valid_rows(plan, dcores)
            raw = _raw_from_rows(cp, rows)
            decoded = GpuArgmaxDecoder(
                plan, cuda_device_id=cuda_device_id, dcores=dcores
            ).decode(raw)

            # Scalar C++ calls form an independent test oracle. Production GPU
            # decoding remains vectorized; this loop is outside all timed paths.
            integer_rows = [
                plan.decode_argmax(int(token), itree, dcores[itree], idm, itime)
                for token, itree, idm, itime, _ in rows
            ]
            integer_oracle = tuple(np.asarray(integer_rows, dtype=np.int64).T)
            physical_rows = [
                plan.decode_argmax2(row[1], *decoded_row)
                for row, decoded_row in zip(rows, integer_rows)
            ]
            physical_oracle = tuple(np.asarray(physical_rows, dtype=np.float64).T)
            chunks = np.asarray([row[4] for row in rows], dtype=np.int64)
            assert any(int(row[0]) >> 24 for row in rows), "fixture must cover nonzero mu"

            for actual, expected in zip((
                decoded.fmin, decoded.fmax, decoded.tlo, decoded.thi,
                decoded.profile,
            ), integer_oracle):
                assert np.array_equal(cp.asnumpy(actual), expected)

            freq_lo, freq_hi, dm, relative_toa, width = physical_oracle
            expected_toa = chunks * int(plan.nt_in) + relative_toa
            comparisons = (
                (decoded.freq_lo_MHz, freq_lo, "frequency low"),
                (decoded.freq_hi_MHz, freq_hi, "frequency high"),
                (decoded.dm, dm, "DM"),
                (decoded.toa_sample_abs, expected_toa, "absolute TOA"),
                (decoded.width_samp, width, "width"),
                (
                    decoded.width_ms,
                    width * float(plan.config.time_sample_ms),
                    "width milliseconds",
                ),
            )
            for actual_gpu, expected, name in comparisons:
                actual = cp.asnumpy(actual_gpu)
                maximum_error = (
                    float(np.max(np.abs(actual - expected))) if actual.size else 0.0
                )
                assert maximum_error <= 1.0e-9, (
                    f"{name} CPU/GPU absolute error is {maximum_error}"
                )
            assert np.array_equal(
                cp.asnumpy(decoded.status),
                np.full(len(rows), int(ArgmaxDecodeStatus.OK), dtype=np.uint8),
            )


def test_gpu_argmax_decoder_absolute_chunks(cuda_device_id=0):
    """The same decoded arrival shifts by exactly plan.nt_in per source chunk."""

    import cupy as cp

    with cp.cuda.Device(cuda_device_id):
        plan, dcores = _make_plan()
        tree = plan.trees[0]
        token = np.uint32(0)
        rows = [
            (token, 0, int(tree.ndm_out) // 2, int(tree.nt_out) // 2, chunk)
            for chunk in (-2, 0, 3)
        ]
        decoded = GpuArgmaxDecoder(
            plan, cuda_device_id=cuda_device_id, dcores=dcores
        ).decode(_raw_from_rows(cp, rows))
        toa = cp.asnumpy(decoded.toa_sample_abs)
        assert np.allclose(
            toa - toa[1],
            np.asarray([-2, 0, 3], dtype=np.float64) * int(plan.nt_in),
            rtol=0.0,
            atol=1.0e-9,
        )


def test_gpu_argmax_decoder_preserves_startup_flag(cuda_device_id=0):
    """Decoding retains the raw uint8 startup bit without changing physics."""

    import cupy as cp

    with cp.cuda.Device(cuda_device_id):
        plan, dcores = _make_plan()
        rows = _valid_rows(plan, dcores)[:2]
        raw = _raw_from_rows(cp, rows)
        expected = np.asarray([
            int(EdgeFlag.STARTUP_INCOMPLETE),
            int(EdgeFlag.STARTUP_INCOMPLETE | EdgeFlag.DM_LOW),
        ], dtype=np.uint8)
        raw.edge_flags[:] = cp.asarray(expected)

        decoder = GpuArgmaxDecoder(plan, cuda_device_id=cuda_device_id, dcores=dcores)
        decoded = decoder.decode(raw)
        baseline = decoder.decode(_raw_from_rows(cp, rows))
        assert decoded.raw is raw
        assert decoded.raw.edge_flags.dtype == cp.uint8
        assert np.array_equal(cp.asnumpy(decoded.raw.edge_flags), expected)
        assert np.array_equal(
            cp.asnumpy(decoded.raw.snr), cp.asnumpy(baseline.raw.snr)
        )
        for name in ("dm", "toa_sample_abs", "width_samp", "width_ms"):
            assert np.array_equal(
                cp.asnumpy(getattr(decoded, name)),
                cp.asnumpy(getattr(baseline, name)),
            )


def test_gpu_argmax_decoder_invalid_empty_and_host_boundary(cuda_device_id=0):
    """Normal decoding has one scalar read; only an invalid row is diagnosed."""

    import cupy as cp

    with cp.cuda.Device(cuda_device_id):
        plan, dcores = _make_plan()
        decoder = GpuArgmaxDecoder(plan, cuda_device_id=cuda_device_id, dcores=dcores)
        empty = decoder.decode(GpuRawCandidates.empty())
        assert len(empty) == 0
        assert int(empty.status.size) == 0

        tree = plan.trees[0]
        bad = _raw_from_rows(cp, [(
            np.uint32(0xffffffff), 0, 0, 0, 0,
        )])
        try:
            decoder.decode(bad)
        except GpuArgmaxDecodeError as exc:
            assert exc.status == ArgmaxDecodeStatus.INVALID_SENTINEL
            assert exc.index == 0
        else:
            raise AssertionError("invalid argmax sentinel was accepted")

        # The valid path never materializes candidate columns on the host.
        source = inspect.getsource(GpuArgmaxDecoder.decode)
        assert "asnumpy" not in source and ".get(" not in source
        assert source.count(".item()") == 1
        row = [(
            np.uint32(0), 0, int(tree.ndm_out) // 2,
            int(tree.nt_out) // 2, 0,
        )]
        original_asnumpy = cp.asnumpy
        cp.asnumpy = lambda *args, **kwargs: (
            (_ for _ in ()).throw(AssertionError("unexpected host transfer"))
        )
        try:
            valid = decoder.decode(_raw_from_rows(cp, row))
            assert len(valid) == 1
        finally:
            cp.asnumpy = original_asnumpy


def test_gpu_argmax_decoder_token_field_bounds(cuda_device_id=0):
    """Each invalid byte is rejected independently, matching the C++ oracle."""
    import cupy as cp
    with cp.cuda.Device(cuda_device_id):
        plan, dcores = _make_plan()
        decoder = GpuArgmaxDecoder(plan, cuda_device_id=cuda_device_id, dcores=dcores)
        for itree, tree in enumerate(plan.trees):
            nm = int(tree.frequency_subbands.M)
            nmu = int(tree.dm_downsampling) >> int(tree.frequency_subbands.pf_rank)
            npf = int(tree.nprofiles)
            dout = int(tree.nt_ds) // int(tree.nt_out)
            cases = []
            if nm < 256:
                cases.append((nm << 16, ArgmaxDecodeStatus.INVALID_MULTIPLET))
            if nmu < 256:
                cases.append((nmu << 24, ArgmaxDecodeStatus.INVALID_EXTRA_DM))
            if npf < 256:
                cases.append((npf << 8, ArgmaxDecodeStatus.INVALID_PROFILE))
            if dout < 256:
                cases.append((dout, ArgmaxDecodeStatus.INVALID_FINE_TIME))
            for token, expected_status in cases:
                raw = _raw_from_rows(cp, [(np.uint32(token), itree, 0, 0, 0)])
                try:
                    decoder.decode(raw)
                except GpuArgmaxDecodeError as exc:
                    assert exc.status == expected_status, (itree, token, exc)
                else:
                    raise AssertionError(f"accepted invalid token {token:#x} in tree {itree}")
                try:
                    plan.decode_argmax(token, itree, dcores[itree], 0, 0)
                except RuntimeError:
                    pass
                else:
                    raise AssertionError("GPU and C++ token validity disagree")


def test_gpu_argmax_decoder_requires_producer_dcores(cuda_device_id=0):
    """The same plan can have producers with different token-time grids."""
    import cupy as cp
    with cp.cuda.Device(cuda_device_id):
        plan, dcores = _make_plan()
        try:
            GpuArgmaxDecoder(plan, cuda_device_id=cuda_device_id)
        except TypeError as exc:
            assert "dcores" in str(exc)
        else:
            raise AssertionError("missing producer Dcores were guessed")
        for invalid in (dcores[:-1], (True,) * plan.ntrees,
                        (0,) * plan.ntrees, (3,) * plan.ntrees,
                        (512,) * plan.ntrees):
            try:
                GpuArgmaxDecoder(plan, cuda_device_id=cuda_device_id, dcores=invalid)
            except ValueError:
                pass
            else:
                raise AssertionError(f"invalid producer Dcores accepted: {invalid}")
        # Profile 4: t=1 is legal for Dcore=1 but illegal for Dcore=2.
        assert int(plan.trees[0].nt_ds) // int(plan.trees[0].nt_out) >= 2
        raw = _raw_from_rows(cp, [(np.uint32(1 | (4 << 8)), 0, 0, 0, 0)])
        ones = (1,) * plan.ntrees
        accepted = GpuArgmaxDecoder(plan, cuda_device_id=cuda_device_id, dcores=ones).decode(raw)
        oracle = plan.decode_argmax(1 | (4 << 8), 0, 1, 0, 0)
        assert int(accepted.tlo[0].item()) == oracle[2]
        assert int(accepted.thi[0].item()) == oracle[3]
        twos = (2,) + ones[1:]
        try:
            GpuArgmaxDecoder(plan, cuda_device_id=cuda_device_id, dcores=twos).decode(raw)
        except GpuArgmaxDecodeError as exc:
            assert exc.status == ArgmaxDecodeStatus.INVALID_TIME_GRANULARITY
        else:
            raise AssertionError("decoder ignored the producer's Dcore")


def test_plan_batch_decoder_scalar_parity():
    """Native offline batches retain every 1.5 field and scalar rounding."""
    for basename in ("toy.yml", "chord_sb2_et.yml"):
        plan, values = _make_plan(basename)
        dcores = np.asarray(values, dtype=np.int64)
        rows = _valid_rows(plan, values)
        tokens = np.asarray([r[0] for r in rows], dtype=np.uint32)
        itrees = np.asarray([r[1] for r in rows], dtype=np.int64)
        idms = np.asarray([r[2] for r in rows], dtype=np.int64)
        itimes = np.asarray([r[3] for r in rows], dtype=np.int64)
        before = [a.copy() for a in (tokens, itrees, idms, itimes, dcores)]
        decoded = plan.decode_argmax_batch(tokens, itrees, idms, itimes, dcores=dcores)
        physical = plan.decode_argmax2_batch(itrees, *decoded)
        expected = np.asarray([
            plan.decode_argmax(int(t), int(it), int(dcores[it]), int(dm), int(tm))
            for t, it, dm, tm in zip(tokens, itrees, idms, itimes)
        ], dtype=np.int64)
        expected_physical = np.asarray([
            plan.decode_argmax2(int(it), *map(int, row))
            for it, row in zip(itrees, expected)
        ], dtype=np.float64)
        assert np.any(tokens >> np.uint32(24)), "must exercise nonzero extra-DM bytes"
        assert np.any(expected[:, 2] < 0), "must exercise negative chunk-relative edges"
        for actual, wanted in zip(decoded, expected.T):
            np.testing.assert_array_equal(actual, wanted)
            assert actual.dtype == np.int64 and actual.flags.c_contiguous
        for actual, wanted in zip(physical, expected_physical.T):
            np.testing.assert_array_equal(actual, wanted)
            assert actual.dtype == np.float64 and actual.flags.c_contiguous
        for actual, original in zip((tokens, itrees, idms, itimes, dcores), before):
            np.testing.assert_array_equal(actual, original)


def test_plan_batch_decoder_rejects_invalid_inputs():
    """Reject missing provenance, malformed buffers and invalid packed fields."""
    plan, values = _make_plan()
    dcores = np.asarray(values, dtype=np.int64)
    tokens = np.zeros(2, dtype=np.uint32)
    zeros = np.zeros(2, dtype=np.int64)

    def rejected(function):
        try:
            function()
        except (TypeError, ValueError, RuntimeError):
            return
        raise AssertionError("invalid native batch input was accepted")

    rejected(lambda: plan.decode_argmax_batch(tokens, zeros, zeros, zeros))
    for bad in (dcores[:-1], dcores[None, :], dcores.astype(np.float64),
                dcores.astype(np.int32), dcores.astype(np.bool_)):
        rejected(lambda: plan.decode_argmax_batch(tokens, zeros, zeros, zeros, dcores=bad))
    for invalid in (0, -1, 3, 512):
        bad = dcores.copy()
        bad[-1] = invalid  # validate even this unused tree's descriptor
        rejected(lambda: plan.decode_argmax_batch(tokens, zeros, zeros, zeros, dcores=bad))
    for bad in (np.full(2, -1, dtype=np.int64),
                np.full(2, plan.ntrees, dtype=np.int64)):
        rejected(lambda: plan.decode_argmax_batch(tokens, bad, zeros, zeros, dcores=dcores))
    for bad in (zeros[:-1], zeros[None, :], np.zeros(4, dtype=np.int64)[::2],
                zeros.astype(np.float64), np.full(2, -1, dtype=np.int64),
                np.full(2, plan.trees[0].ndm_out, dtype=np.int64)):
        rejected(lambda: plan.decode_argmax_batch(tokens, zeros, bad, zeros, dcores=dcores))
    for bad in (np.full(2, -1, dtype=np.int64),
                np.full(2, plan.trees[0].nt_out, dtype=np.int64)):
        rejected(lambda: plan.decode_argmax_batch(tokens, zeros, zeros, bad, dcores=dcores))
    tree = plan.trees[0]
    nmu = int(tree.dm_downsampling) >> int(tree.frequency_subbands.pf_rank)
    assert nmu < 256 and int(tree.frequency_subbands.M) < 256
    for bad_token in (0xffffffff, nmu << 24,
                      int(tree.frequency_subbands.M) << 16, int(tree.nprofiles) << 8):
        bad = np.full(2, bad_token, dtype=np.uint32)
        rejected(lambda: plan.decode_argmax_batch(bad, zeros, zeros, zeros, dcores=dcores))
    rejected(lambda: plan.decode_argmax_batch(tokens.astype(np.int64), zeros, zeros, zeros,
                                             dcores=dcores))
    rejected(lambda: plan.decode_argmax_batch(tokens[:0], zeros[:0], zeros[:0], zeros[:0],
                                             dcores=dcores))
    integers = plan.decode_argmax_batch(tokens, zeros, zeros, zeros, dcores=dcores)
    rejected(lambda: plan.decode_argmax2_batch(zeros, integers[0][:1], *integers[1:]))
    rejected(lambda: plan.decode_argmax2_batch(zeros.astype(np.float64), *integers))
    rejected(lambda: plan.decode_argmax2_batch(np.full(2, -1, dtype=np.int64), *integers))
    rejected(lambda: plan.decode_argmax2_batch(zeros, integers[1], integers[0], *integers[2:]))
    rejected(lambda: plan.decode_argmax2_batch(zeros[:0], *(a[:0] for a in integers)))


def test_plan_batch_decoder_respects_supplied_dcores():
    """Changing a legal producer granularity changes the decoded time edges."""
    plan, _ = _make_plan()
    assert all(int(t.nt_ds) // int(t.nt_out) >= 2 for t in plan.trees)
    token = np.array([4 << 8], dtype=np.uint32)  # profile with level 1
    zero = np.zeros(1, dtype=np.int64)
    one = np.ones(plan.ntrees, dtype=np.int64)
    two = 2 * one
    a = plan.decode_argmax_batch(token, zero, zero, zero, dcores=one)
    b = plan.decode_argmax_batch(token, zero, zero, zero, dcores=two)
    assert b[2][0] - a[2][0] == 1
    assert b[3][0] - a[3][0] == 1
    # Fine time 1 is legal with Dcore=1 but not Dcore=2 at this profile.
    token[0] |= np.uint32(1)
    plan.decode_argmax_batch(token, zero, zero, zero, dcores=one)
    try:
        plan.decode_argmax_batch(token, zero, zero, zero, dcores=two)
    except RuntimeError:
        pass
    else:
        raise AssertionError("batch decoder ignored producer fine-time quantization")
