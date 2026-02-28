from pathlib import Path

from boardGPT.interpretability.ui.compare import compare_run_dirs
from boardGPT.interpretability.ui.run_index import discover_runs
from boardGPT.interpretability.ui.runs import launch_run


def test_b5c_run_launch_index_and_compare(tmp_path: Path):
    out_root = tmp_path / "runs"

    out1 = launch_run(mode="b3_demo", out_root=str(out_root), run_name="r1")
    out2 = launch_run(mode="b3_demo", out_root=str(out_root), run_name="r2")

    run1 = str(out1["run_dir"])
    run2 = str(out2["run_dir"])
    assert Path(run1).exists()
    assert Path(run2).exists()

    indexed = discover_runs(str(out_root))
    assert len(indexed) >= 2
    names = {x["run_name"] for x in indexed}
    assert "r1" in names
    assert "r2" in names

    cmp_out = compare_run_dirs(run1, run2)
    assert "global_delta" in cmp_out
    assert "layer_deltas" in cmp_out
