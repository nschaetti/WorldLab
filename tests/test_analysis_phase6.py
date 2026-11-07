from pathlib import Path

from worldlab.board.analysis import PolicyConfig, WorkflowConfig, run_synthetic_symmetry_workflow


def test_workflow_runs_without_model_and_writes_summary(tmp_path: Path):
    cfg = WorkflowConfig(
        world_shape=(16,),
        periodic=(True,),
        policy=PolicyConfig(length=24, num_sequences=256, seed=0),
        run_model=False,
        output_dir=str(tmp_path),
    )
    summary = run_synthetic_symmetry_workflow(cfg)

    assert "cooccurrence" in summary
    assert "ablation" in summary
    assert "model" not in summary
    assert "output_path" in summary
    assert Path(summary["output_path"]).exists()


def test_workflow_runs_with_model_small_config():
    cfg = WorkflowConfig(
        world_shape=(12,),
        periodic=(True,),
        policy=PolicyConfig(length=20, num_sequences=128, seed=1),
        run_model=True,
    )
    summary = run_synthetic_symmetry_workflow(cfg)

    assert "model" in summary
    assert "decoding" in summary
    assert summary["model"]["train_loss_last"] >= 0
