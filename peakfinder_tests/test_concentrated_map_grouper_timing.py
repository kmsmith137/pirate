"""Focused CPU-only tests for the concentrated-map stress benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from peakfinder_tests import benchmark_concentrated_map_grouper_timing as benchmark


def _specs():
    families = (0, 1, 1, 2)
    shapes = ((8, 5), (4, 3), (6, 4), (3, 2))
    return tuple(
        SimpleNamespace(
            tree_index=tree,
            primary_tree_index=families[tree],
            ndm=shape[0],
            ntime=shape[1],
            pixels_per_beam=shape[0] * shape[1],
        )
        for tree, shape in enumerate(shapes)
    )


def _clean_maps(specs):
    return tuple(
        np.zeros((1, spec.ndm, spec.ntime), dtype=np.float16)
        for spec in specs
    )


def test_layouts_select_exact_tree_sets():
    specs = _specs()
    assert benchmark.selected_tree_indices(specs, "single_map", 2) == (2,)
    assert benchmark.selected_tree_indices(specs, "single_family", 2) == (1, 2)
    assert benchmark.selected_tree_indices(specs, "all_maps", 2) == (0, 1, 2, 3)
    with pytest.raises(ValueError, match="outside"):
        benchmark.selected_tree_indices(specs, "single_map", 4)
    with pytest.raises(ValueError, match="unknown"):
        benchmark.selected_tree_indices(specs, "not-a-layout", 2)


@pytest.mark.parametrize(
    ("layout", "expected_trees"),
    [
        ("single_map", (2,)),
        ("single_family", (1, 2)),
        ("all_maps", (0, 1, 2, 3)),
    ],
)
def test_population_is_exact_deterministic_and_concentrated(
        layout, expected_trees):
    specs = _specs()
    clean = _clean_maps(specs)
    first, trees = benchmark.populate_host_maps(
        clean, specs, layout=layout, selected_tree=2,
        hot_pixel_count=5, hot_snr=32.0, base_seed=17,
    )
    repeated, repeated_trees = benchmark.populate_host_maps(
        clean, specs, layout=layout, selected_tree=2,
        hot_pixel_count=5, hot_snr=32.0, base_seed=17,
    )
    assert trees == repeated_trees == expected_trees
    assert sum(np.count_nonzero(array == np.float16(32.0)) for array in first) == 5
    for tree, (actual, again, original) in enumerate(zip(first, repeated, clean)):
        assert actual.dtype == np.float16
        assert np.array_equal(actual, again)
        if tree not in expected_trees:
            assert np.array_equal(actual, original)
    assert all(np.count_nonzero(array) == 0 for array in clean)


def test_population_rejects_capacity_and_schema_errors():
    specs = _specs()
    clean = _clean_maps(specs)
    with pytest.raises(ValueError, match="fewer than requested"):
        benchmark.populate_host_maps(
            clean, specs, layout="single_map", selected_tree=3,
            hot_pixel_count=7, hot_snr=32.0, base_seed=1,
        )
    wrong = list(clean)
    wrong[2] = wrong[2].astype(np.float32)
    with pytest.raises(ValueError, match="float16"):
        benchmark.populate_host_maps(
            wrong, specs, layout="single_map", selected_tree=2,
            hot_pixel_count=1, hot_snr=32.0, base_seed=1,
        )


def test_argument_validation_requires_timeout_or_explicit_unbounded_opt_in():
    parser = benchmark.build_parser()
    with pytest.raises(ValueError, match="timeout is disabled"):
        benchmark.validate_arguments(parser.parse_args([
            "--hot-pixel-counts", "6000",
        ]))
    args = benchmark.validate_arguments(parser.parse_args([
        "--hot-pixel-counts", "6000",
        "--gpu-timeout-ms", "25",
    ]))
    assert args.hot_pixel_counts == (6000,)
    assert args.gpu_timeout_ms == 25.0
    with pytest.raises(ValueError, match="float16"):
        benchmark.validate_arguments(parser.parse_args([
            "--hot-snr", "10.0001",
        ]))


def test_analysis_notebook_is_cleared_and_compilable():
    path = Path(__file__).with_name(
        "analyze_concentrated_map_grouper_timing.ipynb"
    )
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    code_cells = [
        cell for cell in notebook["cells"] if cell["cell_type"] == "code"
    ]
    assert code_cells
    for cell in code_cells:
        assert cell.get("execution_count") is None
        assert cell.get("outputs") == []
        compile("".join(cell["source"]), str(path), "exec")
