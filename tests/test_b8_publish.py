from pathlib import Path

import torch
import yaml

from boardGPT.interpretability.cli import main as interp_cli_main
from boardGPT.interpretability.publish import run_b8_publish_from_matrix_dir
from boardGPT.interpretability.real_pipeline import run_b6_matrix_from_config_file
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
    ]
    data_path = tmp_path / "dataset.pkl"
    with data_path.open("wb") as f:
        import pickle

        pickle.dump(games, f)

    return cfg_path, model_path, data_path


def _write_matrix_config(tmp_path: Path, cfg_path: Path, model_path: Path, data_path: Path) -> Path:
    exp = {
        "experiment": {
            "name": "b8_test",
            "output_root": str(tmp_path / "out"),
            "validation_mode": "strict",
            "seed": 0,
            "reproducibility": {"deterministic": True, "include_pip_freeze": False},
        },
        "model": {"model_path": str(model_path), "config_path": str(cfg_path), "device": "cpu"},
        "probe": {"type": "linear", "path": None},
        "dataset": {"path": str(data_path), "format": "pkl", "batch_size": 2, "max_examples": 3},
        "protocol": {
            "type": "metrics_only",
            "layers": [0],
            "position": "last",
            "target_class": 2,
            "target_cell": 5,
            "target_cells": [5],
            "max_examples": 3,
        },
        "intervention": {"steps": 5, "lr": 0.05, "reg_strength": 1e-4},
        "batch": {
            "seeds": [0],
            "checkpoints": [{"name": "tiny", "model_path": str(model_path), "config_path": str(cfg_path)}],
        },
        "matrix": {
            "type": "metrics_only",
            "layers": [0],
            "target_cells": [5],
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


def test_b8_publish_from_matrix_dir(tmp_path: Path):
    cfg_path, model_path, data_path = _write_tiny_assets(tmp_path)
    exp_path = _write_matrix_config(tmp_path, cfg_path, model_path, data_path)

    out_mx = run_b6_matrix_from_config_file(str(exp_path), run_name="mx_pub", resume=True)
    matrix_dir = Path(out_mx["matrix_dir"])

    out_pub = run_b8_publish_from_matrix_dir(str(matrix_dir), title="Test Publication")
    assert Path(out_pub["publication_markdown"]).exists()
    assert Path(out_pub["publication_notebook"]).exists()


def test_b8_cli_real_publish(tmp_path: Path):
    cfg_path, model_path, data_path = _write_tiny_assets(tmp_path)
    exp_path = _write_matrix_config(tmp_path, cfg_path, model_path, data_path)
    out_mx = run_b6_matrix_from_config_file(str(exp_path), run_name="mx_pub_cli", resume=True)
    matrix_dir = Path(out_mx["matrix_dir"])

    rc = interp_cli_main([
        "real-publish",
        "--matrix-dir",
        str(matrix_dir),
        "--title",
        "CLI Publication",
        "--markdown-name",
        "cli_pub.md",
        "--notebook-name",
        "cli_pub.ipynb",
    ])
    assert rc == 0
    assert (matrix_dir / "cli_pub.md").exists()
    assert (matrix_dir / "cli_pub.ipynb").exists()


def test_b8_cli_real_one_shot(tmp_path: Path):
    cfg_path, model_path, data_path = _write_tiny_assets(tmp_path)
    exp_path = _write_matrix_config(tmp_path, cfg_path, model_path, data_path)

    rc = interp_cli_main([
        "real-one-shot",
        "--exp-config",
        str(exp_path),
        "--run-name",
        "mx_one_shot",
        "--title",
        "One Shot Publication",
        "--markdown-name",
        "oneshot.md",
        "--notebook-name",
        "oneshot.ipynb",
    ])
    assert rc == 0
    matrix_dir = tmp_path / "out" / "mx_one_shot"
    assert (matrix_dir / "matrix_summary.json").exists()
    assert (matrix_dir / "oneshot.md").exists()
    assert (matrix_dir / "oneshot.ipynb").exists()
