"""CPU-oracle checks for the readable CuPy argmax decoder."""

import inspect
import os

import numpy as np

from ..GpuArgmaxDecoder import (
    ArgmaxDecodeStatus,
    GpuArgmaxDecodeError,
    GpuArgmaxDecoder,
)
from ..Peakfinders import EdgeFlag, GpuRawCandidates
from ..pirate_pybind11 import DedispersionConfig, DedispersionPlan


def _make_plan(config_basename="toy.yml"):
    """Reconstruct the exact producer plan used by the offline path."""

    filename = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "data", config_basename,
    ))
    config = DedispersionConfig.from_yaml(filename)
    producer = DedispersionPlan(config, gpu_runnable=True)
    return DedispersionPlan.make_incomplete_plan_from_yaml(
        config.to_yaml_string(), producer.to_yaml_string()
    )


def _valid_rows(plan):
    """Cover every tree/profile and both ends of its ragged output map."""

    rows = []
    serial = 0
    for itree, tree in enumerate(plan.trees):
        dout = int(tree.nt_ds) // int(tree.nt_out)
        dcore = int(tree.Dcore)
        for profile in range(int(tree.nprofiles)):
            level = (profile - 1) // 3 if profile else 0
            dt = min(dcore, 1 << level)
            fine_times = sorted({0, ((dout - 1) // dt) * dt})
            multiplets = sorted({0, int(tree.frequency_subbands.M) - 1})
            for fine_time in fine_times:
                for multiplet in multiplets:
                    edge = serial & 1
                    rows.append((
                        np.uint32(
                            fine_time | (profile << 8) | (multiplet << 16)
                        ),
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
        plan = _make_plan("chord_sb2_et.yml")
        rows = _valid_rows(plan)
        raw = _raw_from_rows(cp, rows)
        decoded = GpuArgmaxDecoder(
            plan, cuda_device_id=cuda_device_id
        ).decode(raw)

        tokens = np.asarray([row[0] for row in rows], dtype=np.uint32)
        trees = np.asarray([row[1] for row in rows], dtype=np.int64)
        idms = np.asarray([row[2] for row in rows], dtype=np.int64)
        itimes = np.asarray([row[3] for row in rows], dtype=np.int64)
        chunks = np.asarray([row[4] for row in rows], dtype=np.int64)
        integer_oracle = plan.decode_argmax_batch(
            tokens, trees, idms, itimes
        )
        physical_oracle = plan.decode_argmax2_batch(trees, *integer_oracle)

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
        plan = _make_plan()
        tree = plan.trees[0]
        token = np.uint32(0)
        rows = [
            (token, 0, int(tree.ndm_out) // 2, int(tree.nt_out) // 2, chunk)
            for chunk in (-2, 0, 3)
        ]
        decoded = GpuArgmaxDecoder(
            plan, cuda_device_id=cuda_device_id
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
        plan = _make_plan()
        rows = _valid_rows(plan)[:2]
        raw = _raw_from_rows(cp, rows)
        expected = np.asarray([
            int(EdgeFlag.STARTUP_INCOMPLETE),
            int(EdgeFlag.STARTUP_INCOMPLETE | EdgeFlag.DM_LOW),
        ], dtype=np.uint8)
        raw.edge_flags[:] = cp.asarray(expected)

        decoder = GpuArgmaxDecoder(plan, cuda_device_id=cuda_device_id)
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
        plan = _make_plan()
        decoder = GpuArgmaxDecoder(plan, cuda_device_id=cuda_device_id)
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
