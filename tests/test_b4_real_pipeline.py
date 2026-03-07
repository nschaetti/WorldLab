import json
from pathlib import Path

import torch
import yaml

from boardGPT.interpretability.cli import main as interp_cli_main
from boardGPT.interpretability.real_pipeline import run_b4_from_config_file
from boardGPT.nn import GPT, GPTConfig


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


def _write_exp_config(tmp_path: Path, cfg_path: Path, model_path: Path, data_path: Path) -> Path:
    exp = {
        "experiment": {"name": "b4_test", "output_root": str(tmp_path / "out"), "validation_mode": "strict"},
        "model": {"model_path": str(model_path), "config_path": str(cfg_path), "device": "cpu"},
        "probe": {"type": "linear", "path": None},
        "dataset": {"path": str(data_path), "format": "pkl", "batch_size": 2, "max_examples": 4},
        "protocol": {
            "type": "metrics_and_attribution",
            "layers": [0, 1],
            "position": "last",
            "target_class": 2,
            "target_cell": 5,
            "target_cells": [3, 5],
            "max_examples": 4,
        },
        "intervention": {"steps": 8, "lr": 0.05, "reg_strength": 1e-4},
        "batch": {
            "seeds": [0],
            "checkpoints": [
                {"name": "tiny", "model_path": str(model_path), "config_path": str(cfg_path)}
            ],
        },
        "report": {"top_k": 5, "export_csv": True, "export_markdown": True},
    }
    exp_path = tmp_path / "exp.yaml"
    exp_path.write_text(yaml.safe_dump(exp), encoding="utf-8")
    return exp_path


def test_b4_from_config_file(tmp_path: Path):
    cfg_path, model_path, data_path = _write_tiny_assets(tmp_path)
    exp_path = _write_exp_config(tmp_path, cfg_path, model_path, data_path)

    out = run_b4_from_config_file(str(exp_path), run_name="single")
    run_dir = Path(out["run_dir"])
    assert run_dir.exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "report.md").exists()
    assert (run_dir / "metrics_by_layer.csv").exists()
    assert (run_dir / "reproducibility.json").exists()
    assert (run_dir / "reproducibility_report.md").exists()

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["run_integrity"] in {"clean", "degraded"}
    assert "telemetry" in summary
    assert "elapsed_total_s" in summary["telemetry"]
    assert "memory" in summary["telemetry"]
    assert "reproducibility" in summary
    assert summary["reproducibility"]["seed"]["seed"] == 0


def test_b4_cli_real_batch(tmp_path: Path):
    cfg_path, model_path, data_path = _write_tiny_assets(tmp_path)
    exp_path = _write_exp_config(tmp_path, cfg_path, model_path, data_path)

    rc = interp_cli_main(["real-batch", "--exp-config", str(exp_path), "--run-name", "batch_test"])
    assert rc == 0
    batch_dir = tmp_path / "out" / "batch_test"
    assert (batch_dir / "batch_summary.json").exists()


def test_b4_tolerant_with_auto_probe_training(tmp_path: Path):
    cfg_path, model_path, data_path = _write_tiny_assets(tmp_path)
    exp_path = _write_exp_config(tmp_path, cfg_path, model_path, data_path)
    exp = yaml.safe_load(exp_path.read_text(encoding="utf-8"))
    exp["experiment"]["validation_mode"] = "tolerant"
    exp["probe"]["path"] = None
    exp["probe"]["auto_train_if_missing"] = True
    exp["probe"]["auto_train_epochs"] = 1
    exp["probe"]["auto_train_max_examples"] = 4
    exp_path.write_text(yaml.safe_dump(exp), encoding="utf-8")

    out = run_b4_from_config_file(str(exp_path), run_name="auto_probe")
    summary = json.loads((Path(out["run_dir"]) / "summary.json").read_text(encoding="utf-8"))
    assert summary["probe"]["checkpoint_loaded"] is False
    assert summary["probe"]["auto_train"]["enabled"] is True
    assert summary["probe"]["auto_train"]["trained"] is True
