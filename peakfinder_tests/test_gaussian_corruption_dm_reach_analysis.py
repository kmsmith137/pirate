import json
from pathlib import Path


def test_dm_reach_analysis_notebook_is_focused_cleared_and_compilable():
    path = Path(__file__).with_name(
        "analyze_gaussian_corruption_dm_reach.ipynb"
    )
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4

    code_cells = [
        cell for cell in notebook["cells"] if cell["cell_type"] == "code"
    ]
    assert [cell["id"] for cell in code_cells] == [
        "load-campaigns",
        "validate-pairing",
        "survived-vs-injected-by-dm-reach",
    ]
    for cell in code_cells:
        assert cell["execution_count"] is None
        assert cell["outputs"] == []
        compile("".join(cell["source"]), f"{path.name}:{cell['id']}", "exec")

    loader = "".join(code_cells[0]["source"])
    for concept in (
        "DM_REACH_GRID = (1, 2, 4, 8, 16, 32)",
        "PIRATE_GAUSSIAN_DM_REACH_RESULTS_ROOT",
        "results_gaussian_corruption_timing",
        "campaign signature mismatch",
        "campaign is incomplete",
        "total_pixels_above_threshold",
        "total_peakfinder_candidates",
        "PIXELS_PER_BEAM = 983_040",
        "TOTAL_BEAMS = 60",
    ):
        assert concept in loader

    pairing = "".join(code_cells[1]["source"])
    assert "normalized.pop('dm_reach', None)" in pairing
    assert "PAIRED_WORKLOAD_FIELDS" in pairing
    assert "unpaired" in pairing

    plot = "".join(code_cells[2]["source"])
    for concept in (
        "ax.scatter(",
        "np.median",
        "dm_reach = {dm_reach}",
        "No suppression ($y=x$)",
        "ax.set_xscale('log')",
        "ax.set_yscale('log')",
        "60-beam total",
    ):
        assert concept in plot
    assert plot.count("plt.show()") == 1
