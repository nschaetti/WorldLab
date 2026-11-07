from pathlib import Path

import torch
import yaml

from worldlab.board.interpretability.cli import main as interp_cli_main
from worldlab.board.interpretability.real_pipeline import run_b6_matrix_from_config_file
from worldlab.nn.board import GPT, GPTConfig


def _write_tiny_assets(tmp_path: Path) -> tuple[Path, Path, Path]:
    model_cfg = {
        "block_size": 8,
        "vocab_size": 61,
        "n_layer": 2,
        "n_head": 1,
        "n_embd": 16,
        "dropout": 0.0,
        "bias": True,
    }
    cfg_path = tmp_path / "model_config.yaml"
    cfg_path.write_text(yaml.safe_dump(model_cfg), encoding="utf-8")

    model = GPT(GPTConfig(**model_cfg), use_flash=False)
    model_path = tmp_path / "model.pt"
    torch.save(model.state_dict(), model_path)

    games = [
        ["d3", "c3", "c4", "e3"],
        ["c4", "c3", "d3", "e3"],
        ["f5", "f4", "e3", "d3"],
        ["d3", "c3", "f5", "f4"],
    ]
    data_path = tmp_path / "dataset.pkl"
    with data_path.open("wb") as f:
        import pickle

        pickle.dump(games, f)

    return cfg_path, model_path, data_path


def _write_matrix_config(tmp_path: Path, cfg_path: Path, model_path: Path, data_path: Path) -> Path:
    exp = {
        "experiment": {"name": "b6_test", "output_root": str(tmp_path / "out"), "validation_mode": "strict"},
        "model": {"model_path": str(model_path), "config_path": str(cfg_path), "device": "cpu"},
        "probe": {"type": "linear", "path": None},
        "dataset": {"path": str(data_path), "format": "pkl", "batch_size": 2, "max_examples": 4},
        "protocol": {
            "type": "metrics_only",
            "layers": [0],
            "position": "last",
            "target_class": 2,
            "target_cell": 5,
            "target_cells": [5],
            "max_examples": 4,
        },
        "intervention": {"steps": 6, "lr": 0.05, "reg_strength": 1e-4},
        "batch": {
            "seeds": [0],
            "checkpoints": [{"name": "tiny", "model_path": str(model_path), "config_path": str(cfg_path)}],
        },
        "matrix": {
            "type": "metrics_only",
            "layers": [0],
            "target_cells": [5, 9],
            "target_classes": [2],
            "positions": ["last"],
            "seeds": [0],
            "resume": True,
            "continue_on_error": False,
        },
        "report": {"top_k": 5, "export_csv": True, "export_markdown": True},
    }
    exp_path = tmp_path / "matrix_exp.yaml"
    exp_path.write_text(yaml.safe_dump(exp), encoding="utf-8")
    return exp_path


def test_b6_matrix_runner_and_resume(tmp_path: Path):
    cfg_path, model_path, data_path = _write_tiny_assets(tmp_path)
    exp_path = _write_matrix_config(tmp_path, cfg_path, model_path, data_path)

    out1 = run_b6_matrix_from_config_file(str(exp_path), run_name="mx", resume=True)
    matrix_dir = Path(out1["matrix_dir"])
    assert matrix_dir.exists()
    assert (matrix_dir / "matrix_runs.json").exists()
    assert (matrix_dir / "matrix_results.csv").exists()
    assert (matrix_dir / "matrix_summary.json").exists()
    assert out1["summary"]["num_runs_total"] == 2
    first_run_dir = Path(out1["runs"][0]["run_dir"])
    assert (first_run_dir / "reproducibility.json").exists()

    out2 = run_b6_matrix_from_config_file(str(exp_path), run_name="mx", resume=True)
    assert out2["summary"]["num_runs_resumed"] >= 2


def test_b6_cli_real_matrix(tmp_path: Path):
    cfg_path, model_path, data_path = _write_tiny_assets(tmp_path)
    exp_path = _write_matrix_config(tmp_path, cfg_path, model_path, data_path)

    rc = interp_cli_main(["real-matrix", "--exp-config", str(exp_path), "--run-name", "mx_cli", "--no-resume"])
    assert rc == 0
    matrix_dir = tmp_path / "out" / "mx_cli"
    assert (matrix_dir / "matrix_summary.json").exists()
