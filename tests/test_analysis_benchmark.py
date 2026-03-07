from pathlib import Path

from boardGPT.analysis import (
    PolicyConfig,
    WorkflowConfig,
    flatten_workflow_summary,
    run_workflow_benchmark,
)


def test_flatten_workflow_summary_extracts_core_metrics():
    summary = {
        "world": {"shape": (16,), "periodic": (True,), "vocab_size": 16, "dim": 1},
        "cooccurrence": {
            "symmetry_r2": 0.91,
            "symmetry_mse": 0.02,
            "fit": {"amplitude": 0.4, "sigma": 0.2, "offset": 0.0, "r2": 0.8},
        },
        "ablation": {
            "target_size": 6,
            "procrustes_mean": [0.8, 0.5, 0.3],
            "gram_rel_mse_mean": [1.2, 0.7, 0.4],
        },
    }
    row = flatten_workflow_summary(summary)
    assert row["world_dim"] == 1
    assert row["cooc_symmetry_r2"] == 0.91
    assert row["cooc_fit_sigma"] == 0.2
    assert row["ablation_last_procrustes"] == 0.3


def test_run_workflow_benchmark_writes_aggregates(tmp_path: Path):
    cfg_a = WorkflowConfig(
        world_shape=(12,),
        periodic=(True,),
        policy=PolicyConfig(length=20, num_sequences=96, seed=0),
        run_model=False,
    )
    cfg_b = WorkflowConfig(
        world_shape=(6, 6),
        periodic=(False, False),
        policy=PolicyConfig(length=20, num_sequences=96, seed=1),
        run_model=False,
    )

    summary = run_workflow_benchmark(
        benchmark_name="test_bench",
        configs=[("run_a", cfg_a), ("run_b", cfg_b)],
        output_root=str(tmp_path),
    )

    assert summary["num_runs"] == 2
    assert Path(summary["csv_path"]).exists()
    assert Path(summary["markdown_path"]).exists()
    assert Path(summary["summary_path"]).exists()
