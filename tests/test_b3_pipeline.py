from pathlib import Path

from boardGPT.interpretability.cli import main as cli_main
from boardGPT.interpretability.pipeline import run_b3_demo_pipeline
from boardGPT.games.cli import main as games_cli_main


def test_b3_pipeline_outputs_expected_artifacts(tmp_path: Path):
    out_root = tmp_path / "out_interp"
    result = run_b3_demo_pipeline(
        out_root=str(out_root),
        run_name="test_run",
        seed=1,
        layers=(0, 1),
        target_cell=5,
        target_class=2,
        target_cells=(3, 5),
        max_examples=4,
    )

    run_dir_value = result.get("run_dir")
    assert isinstance(run_dir_value, str)
    run_dir = Path(run_dir_value)
    assert run_dir.exists()
    assert (run_dir / "config.json").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "report.md").exists()
    assert (run_dir / "metrics_overall.csv").exists()
    assert (run_dir / "metrics_by_layer.csv").exists()
    assert (run_dir / "top_contributors.csv").exists()


def test_b3_cli_smoke_and_report(tmp_path: Path):
    out_root = tmp_path / "out_interp_cli"
    rc = cli_main([
        "smoke",
        "--out-root",
        str(out_root),
        "--run-name",
        "cli_run",
        "--max-examples",
        "3",
    ])
    assert rc == 0

    run_dir = out_root / "cli_run"
    assert run_dir.exists()

    rc2 = cli_main([
        "report",
        "--run-dir",
        str(run_dir),
    ])
    assert rc2 == 0


def test_games_cli_interpretability_bridge(tmp_path: Path):
    out_root = tmp_path / "out_interp_games_cli"
    rc = games_cli_main([
        "othello",
        "interp",
        "smoke",
        "--out-root",
        str(out_root),
        "--run-name",
        "bridge_run",
        "--max-examples",
        "3",
    ])
    assert rc == 0
    assert (out_root / "bridge_run" / "report.md").exists()
