"""Small executable checks for the migrated toy experiment entry points."""
import csv
from pathlib import Path

import numpy as np
import pytest
import yaml

from . import test_peakfinder_recall as recall
from . import test_peakfinder_runtime as runtime
from . import test_peakfinder_separability as separability
from .experiment_common import prepare_plan
from .fast_snrmap import FastSnrMapSimulator
from .peakfinders import enumerate_plan_subbands
from .producer_metadata import ARGMAX_ENCODING

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def toy_inputs(tmp_path):
    config_path = ROOT / 'configs/dedispersion/toy_off.yml'
    metadata = yaml.safe_load((ROOT / 'configs/xengine_metadata.yml').read_text())
    metadata.update(zone_nfreq=[640], zone_freq_edges=[400, 800],
                    freq_channels=list(range(640)), beam_ids=[100],
                    beam_positions_x=[0.0], beam_positions_y=[0.0], noise_variance=1.0)
    metadata_path = tmp_path / 'metadata.yml'
    metadata_path.write_text(yaml.safe_dump(metadata))
    return str(config_path), str(metadata_path)


@pytest.mark.parametrize('module,section,options', [
    (recall, 'recall', ['--trials', '1', '--parameter-seed', '137']),
    (separability, 'separability', ['--trials', '1', '--separations-ms', '10',
                                 '--parameter-seed', '137']),
    (runtime, 'runtime', ['--ndm', '64', '--nt', '64', '--warmup', '1', '--iterations', '1']),
])
def test_toy_campaign_entry_points(module, section, options, toy_inputs, tmp_path, monkeypatch):
    config_path, metadata_path = toy_inputs
    _, _, _, expected_dcores = prepare_plan(config_path, metadata_path, cuda_device_id=0)
    observed = []
    if section != 'runtime':
        original = module.run_simulated_peakfinders

        def capture_producer(**kwargs):
            decoded, info = original(**kwargs)
            assert tuple(info['dcores']) == expected_dcores
            assert info['argmax_encoding'] == ARGMAX_ENCODING
            assert info['containment_checked_chunks'] == [0, 1, 2, 3]
            for values in decoded.values():
                assert np.isfinite(values['dm']).all()
                assert np.isfinite(values['toa_ref_s']).all()
                assert values['argmax_token'].dtype == np.uint32
            observed.append(info)
            return decoded, info

        monkeypatch.setattr(module, 'run_simulated_peakfinders', capture_producer)
    output = tmp_path / section
    args = ['--config', config_path, '--metadata', metadata_path,
            '--results-dir', str(output), '--device', '0', *options]
    module.main(args)
    metadata = yaml.safe_load((output / 'metadata.yaml').read_text())
    assert metadata['suite']['schema_version'] == 5
    record = metadata[section]
    assert record['complete'] is True
    assert tuple(record['scientific_parameters']['dcores']) == expected_dcores
    assert record['scientific_parameters']['argmax_encoding'] == ARGMAX_ENCODING
    assert [entry['dcore'] for entry in record['plan']] == list(expected_dcores)
    rows_before = (output / (section + '.csv')).read_bytes()
    rows = list(csv.DictReader(rows_before.decode().splitlines()))
    assert len(rows) == 1
    assert rows[0]['schema_version'] == '5'
    if section != 'runtime':
        assert len(observed) == 1
    # A resume must preserve completed data and avoid simulating it again.
    module.main([*args, '--resume'])
    assert (output / (section + '.csv')).read_bytes() == rows_before
    if section != 'runtime':
        assert len(observed) == 1


def test_fast_simulator_preserves_nonzero_extra_dm(toy_inputs):
    config, xmd, plan, dcores = prepare_plan(*toy_inputs, cuda_device_id=0)
    tree = plan.trees[0]
    _, bands, full_index = enumerate_plan_subbands(plan, 0, dcores=dcores)
    full_band = bands[full_index]
    multiplet = int(np.flatnonzero(
        np.asarray(tree.frequency_subbands.m_to_n) == full_band.band_index)[0])
    token = (multiplet << 16) | (1 << 24)
    integer = plan.decode_argmax(token, 0, dcores[0], 240, 0)
    target_dm = plan.decode_argmax2(0, *integer)[2]
    simulator = FastSnrMapSimulator(
        plan, 0, full_band, dcores=dcores, xp=np,
        time_sample_s=float(config.time_sample_ms) / 1000, nt_in=int(plan.nt_in),
        reference_freq_mhz=400.0, injected_dm=target_dm,
        injected_snr=50.0, injected_width_s=0.002,
    )
    assert simulator.extra_dm == 1
    assert np.all(simulator.fine_tokens_cpu >> np.uint32(24) == 1)
    assert np.min(np.abs(simulator.dm_axis_cpu - target_dm)) < 1e-12
    diagnostic = simulator.diagnostics()
    assert diagnostic['dcores'] == list(dcores)
    assert diagnostic['argmax_encoding'] == ARGMAX_ENCODING
    for fine_index, token in enumerate(simulator.fine_tokens_cpu):
        for idm in (0, simulator.ndm // 2, simulator.ndm - 1):
            integer = plan.decode_argmax(int(token), 0, dcores[0], idm, 0)
            physical = plan.decode_argmax2(0, *integer)
            assert simulator.dm_axis_cpu[idm] == physical[2]
            assert simulator.timestamp_by_fine_cpu[fine_index, idm] == (
                physical[3] * simulator.time_sample_s)
