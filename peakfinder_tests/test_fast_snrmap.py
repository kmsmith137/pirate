"""Focused NumPy regression tests for the fast analytic S/N-map simulator."""

from pathlib import Path

import numpy as np

from .experiment_common import prepare_plan, tree_for_dm
from .fast_snrmap import FastSnrMapSimulator
from .peakfinders import enumerate_plan_subbands


ROOT = Path(__file__).resolve().parents[1]
CONFIG = str(ROOT / "configs/dedispersion/chord_sb2.yml")
METADATA = str(ROOT / "configs/xengine_metadata.yml")


def _expect_raises(exception, operation, contains):
    try:
        operation()
    except exception as exc:
        assert contains in str(exc), str(exc)
        return
    raise AssertionError(f"expected {exception.__name__}")


def _make_simulator():
    config, xmd, plan, dcores = prepare_plan(CONFIG, METADATA)
    itree = tree_for_dm(plan, 100.0)
    _, subbands, _ = enumerate_plan_subbands(plan, itree, dcores=dcores)
    full_bands = [band for band in subbands if band.is_full_band]
    assert len(full_bands) == 1
    reference_freq_mhz = float(np.asarray(xmd.get_channel_freq_edges())[0])
    simulator = FastSnrMapSimulator(
        plan,
        itree,
        full_bands[0],
        xp=np,
        dcores=dcores,
        time_sample_s=float(config.time_sample_ms) / 1.0e3,
        nt_in=int(plan.nt_in),
        reference_freq_mhz=reference_freq_mhz,
        injected_dm=100.0,
        injected_snr=45.0,
        injected_width_s=0.001,
        base_seed=12345,
    )
    return simulator


def test_heterogeneous_injected_snrs():
    simulator = _make_simulator()
    target_idm = int(np.argmin(np.abs(simulator.dm_axis_cpu - 100.0)))
    first_toa = (3 * simulator.chunk_duration_s
                 + simulator.timestamp_by_fine_cpu[0, target_idm]
                 + (simulator.nt // 2) * simulator.time_step_s)
    toas = (first_toa, first_toa + 0.1, first_toa + 0.15)
    heterogeneous = (45.0, 30.0, 20.0)

    signal, argmax = simulator.signal_template(
        toas, injected_snrs=heterogeneous, time_chunk_index=3)
    default_signal, default_argmax = simulator.signal_template(
        toas, time_chunk_index=3)
    explicit_default, explicit_default_argmax = simulator.signal_template(
        toas, injected_snrs=(45.0, 45.0, 45.0), time_chunk_index=3)

    assert signal.dtype == np.float32 and argmax.dtype == np.uint32
    assert signal.shape == argmax.shape == (simulator.ndm, simulator.nt)
    assert not np.array_equal(signal, default_signal)
    assert default_signal is explicit_default
    assert default_argmax is explicit_default_argmax

    generated, generated_argmax, chunk = simulator.generate_map(
        toas, injected_snrs=heterogeneous, trial=0, time_chunk_index=3)
    assert chunk == 3
    assert generated.dtype == np.float32 and generated_argmax.dtype == np.uint32
    assert np.array_equal(generated_argmax, argmax)
    assert np.allclose(
        generated,
        signal + simulator.background(0) - simulator.background_parameters.location,
    )

    target_idm = int(np.argmin(np.abs(simulator.dm_axis_cpu - 100.0)))
    chunk_start_s = 3 * simulator.chunk_duration_s
    coordinates = (
        simulator.timestamp_by_fine_cpu[:, target_idm, None]
        + np.arange(simulator.nt)[None, :] * simulator.time_step_s
        + chunk_start_s
    )
    local_peaks = []
    for toa in toas:
        _, itime = np.unravel_index(
            np.argmin(np.abs(coordinates - toa)), coordinates.shape)
        window = signal[target_idm, max(0, itime - 1):itime + 2]
        local_peaks.append(float(np.max(window)))
    assert local_peaks[0] > local_peaks[1] > local_peaks[2] > 0

    _expect_raises(
        ValueError,
        lambda: simulator.signal_template(toas, injected_snrs=(45.0, 30.0)),
        "expected 3",
    )
    _expect_raises(
        ValueError,
        lambda: simulator.signal_template(toas, injected_snrs=45.0),
        "one value per TOA",
    )
    for invalid in ((45.0, 0.0, 20.0), (45.0, np.nan, 20.0)):
        _expect_raises(
            ValueError,
            lambda invalid=invalid: simulator.signal_template(
                toas, injected_snrs=invalid),
            "finite and positive",
        )


def main():
    test_heterogeneous_injected_snrs()
    print("fast_snrmap heterogeneous-S/N tests passed")


if __name__ == "__main__":
    main()
