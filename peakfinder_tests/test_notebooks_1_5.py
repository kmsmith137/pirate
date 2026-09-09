"""Execute the active notebooks with small real 1.5 experiments and bad inputs."""
import json
import os
from pathlib import Path
import shutil

os.environ.setdefault('MPLBACKEND', 'Agg')

import asdf
import matplotlib.pyplot as plt
import numpy as np
import pytest
import yaml

from pirate_frb import DedispersionConfig, DedispersionPlan
from pirate_frb.tests.test_offline_grouper import _write_toy_acquisition, _rewrite_map
from . import test_peakfinder_recall as recall
from . import test_peakfinder_runtime as runtime
from . import test_peakfinder_separability as separability
from .producer_metadata import ARGMAX_ENCODING

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def render_without_gui(monkeypatch):
    def render():
        for number in plt.get_fignums():
            plt.figure(number).canvas.draw()
    monkeypatch.setattr(plt, 'show', render)


def _execute_notebook(name, overrides):
    path = ROOT / 'peakfinder_tests' / name
    nb = json.loads(path.read_text())
    namespace = {'__name__': '__notebook_validation__'}
    try:
        for cell in nb['cells']:
            if cell['cell_type'] != 'code':
                continue
            assert cell['outputs'] == [] and cell['execution_count'] is None
            exec(compile(''.join(cell['source']), f'{path}:{cell["id"]}', 'exec'), namespace)
            if cell['id'] == 'parameters':
                namespace.update(overrides)
        return namespace
    finally:
        plt.close('all')


@pytest.fixture(scope='module')
def experiments(tmp_path_factory):
    directory = tmp_path_factory.mktemp('pirate15-notebooks')
    config = ROOT / 'configs/dedispersion/toy_off.yml'
    metadata = yaml.safe_load((ROOT / 'configs/xengine_metadata.yml').read_text())
    metadata.update(zone_nfreq=[640], zone_freq_edges=[400, 800],
                    freq_channels=list(range(640)), beam_ids=[100],
                    beam_positions_x=[0.0], beam_positions_y=[0.0], noise_variance=1.0)
    metadata_path = directory / 'metadata.yml'
    metadata_path.write_text(yaml.safe_dump(metadata))
    campaign = directory / 'campaign'
    base = ['--config', str(config), '--metadata', str(metadata_path),
            '--results-dir', str(campaign), '--device', '0']
    runtime.main([*base, '--ndm', '64', '--nt', '64', '--warmup', '1', '--iterations', '1'])
    recall.main([*base, '--trials', '1', '--parameter-seed', '137'])
    separability.main([*base, '--trials', '1', '--parameter-seed', '137', '--separations-ms', '10'])
    maps = directory / 'maps'
    maps.mkdir()
    fixture = _write_toy_acquisition(str(maps), beams=(7,), chunks=(1,))
    return dict(directory=directory, config=config, metadata=metadata_path,
                campaign=campaign, fixture=fixture, map=Path(fixture.files[(7, 1)]))


def _map_options():
    return dict(GPU_DEVICE=0, TREE_INDEX=0, DM_REACH_VALUES=(1, 8),
                TIMING_WARMUP_RUNS=1, TIMING_MEASURED_RUNS=1)


def test_analytic_notebook_and_export(experiments):
    destination = experiments['directory'] / 'analytic.npz'
    state = _execute_notebook('inspect_peakfinders_on_fast_snrmap.ipynb', dict(
        **_map_options(), CONFIG_PATH=experiments['config'],
        XENGINE_METADATA_PATH=experiments['metadata'],
        SAVE_GENERATED_MAP=True, GENERATED_MAP_PATH=destination))
    toas = np.asarray(state['decoded']['toa_ref_s'])
    matched = [int(np.argmin(np.abs(toas - toa))) for toa in state['INJECTED_TOAS_S']]
    assert len(set(matched)) == 3
    assert all(abs(toas[i] - toa) < .003
               for i, toa in zip(matched, state['INJECTED_TOAS_S']))
    with np.load(destination, allow_pickle=False) as archive:
        assert str(archive['argmax_encoding']) == ARGMAX_ENCODING
        assert tuple(archive['dcores']) == tuple(state['dcores'])
        assert int(archive['time_chunk_index']) == 4  # Derived from 0.9984 ms cadence.
        assert np.array_equal(archive['out_argmax'], state['argmax_gpu'].get())
        config = DedispersionConfig.from_yaml_string(str(archive['config_yaml']))
        plan = DedispersionPlan.from_yaml_string(config, str(archive['plan_yaml']))
        assert plan.ntrees == state['plan'].ntrees


def test_saved_map_notebook_matches_cpp(experiments):
    state = _execute_notebook('inspect_peakfinders_on_snrmap.ipynb', dict(
        **_map_options(), ASDF_PATH=experiments['map']))
    fixture = experiments['fixture']
    values = state['decoded']
    assert len(values['snr']) == 1
    assert tuple(state['dcores']) == fixture.dcores
    assert int(values['argmax_token'][0]) == fixture.token
    integer = fixture.plan.decode_argmax(
        fixture.token, 0, fixture.dcores[0], fixture.idm, fixture.itime)
    physical = fixture.plan.decode_argmax2(0, *integer)
    assert values['dm'][0] == physical[2]
    assert np.isclose(values['toa_ref_s'][0],
        (fixture.plan.nt_in + physical[3]) * state['time_sample_s'], rtol=0, atol=1e-12)


@pytest.mark.parametrize('mutation,match', [
    (lambda root: root.update(format_version=2), 'format_version=3'),
    (lambda root: root.pop('dcores'), 'dcores'),
])
def test_saved_notebook_rejects_legacy_or_missing_metadata(experiments, tmp_path, mutation, match):
    path = tmp_path / experiments['map'].name
    shutil.copyfile(experiments['map'], path)
    _rewrite_map(str(path), mutation)
    with pytest.raises(ValueError, match=match):
        _execute_notebook('inspect_peakfinders_on_snrmap.ipynb', dict(
            **_map_options(), ASDF_PATH=path))


def test_campaign_notebook_uses_recorded_small_grids(experiments, monkeypatch):
    monkeypatch.setenv('PEAKFINDER_RESULTS_DIR', str(experiments['campaign']))
    state = _execute_notebook('analyze_peakfinder_tests.ipynb', {})
    assert state['SCHEMA_VERSION'] == 5
    assert state['METHODS'] == ('full_band_bowtie',)
    assert state['N_DM'] == (64,) and state['N_T'] == (64,)
    assert state['SEPARATIONS_MS'] == (10,)
    assert state['TIMED_CALLS'] == 1
    assert state['timing_mean_matrices']['full_band_bowtie'].shape == (1, 1)


@pytest.mark.parametrize('mutation,match', [
    (lambda m: m['suite'].update(schema_version=4), 'schema v5'),
    (lambda m: m['recall']['scientific_parameters'].pop('dcores'), 'Dcores'),
    (lambda m: m['runtime']['scientific_parameters'].update(ndm_values=[32, 64]), 'runtime'),
])
def test_campaign_notebook_rejects_incompatible_metadata(experiments, tmp_path, monkeypatch, mutation, match):
    for path in experiments['campaign'].iterdir():
        if path.is_file():
            shutil.copyfile(path, tmp_path / path.name)
    path = tmp_path / 'metadata.yaml'
    metadata = yaml.safe_load(path.read_text())
    mutation(metadata)
    path.write_text(yaml.safe_dump(metadata))
    monkeypatch.setenv('PEAKFINDER_RESULTS_DIR', str(tmp_path))
    with pytest.raises(ValueError, match=match):
        _execute_notebook('analyze_peakfinder_tests.ipynb', {})
