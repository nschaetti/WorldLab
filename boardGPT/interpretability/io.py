"""
Run artifact I/O utilities for interpretability experiments.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def _to_jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_to_jsonable(v) for v in value]
    return value


def save_run_artifacts(
    run_dir: str,
    config: Dict[str, Any],
    summary: Dict[str, Any],
    results: Optional[list] = None,
    tensors: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, str]:
    """
    Save a run in a standard layout:
    - config.json
    - summary.json
    - results.jsonl (optional)
    - tensors/<name>.pt (optional)
    """
    out_dir = Path(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config_path = out_dir / "config.json"
    summary_path = out_dir / "summary.json"
    config_path.write_text(json.dumps(_to_jsonable(config), indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(_to_jsonable(summary), indent=2), encoding="utf-8")

    result_paths: Dict[str, str] = {
        "run_dir": str(out_dir),
        "config": str(config_path),
        "summary": str(summary_path),
    }

    if results is not None:
        results_path = out_dir / "results.jsonl"
        with results_path.open("w", encoding="utf-8") as f:
            for row in results:
                f.write(json.dumps(_to_jsonable(row)) + "\n")
        result_paths["results"] = str(results_path)

    if tensors:
        tensor_dir = out_dir / "tensors"
        tensor_dir.mkdir(parents=True, exist_ok=True)
        for name, tensor in tensors.items():
            torch.save(tensor.detach().cpu(), tensor_dir / f"{name}.pt")
        result_paths["tensors_dir"] = str(tensor_dir)

    return result_paths


def load_run_artifacts(run_dir: str) -> Dict[str, Any]:
    out_dir = Path(run_dir)
    if not out_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    config = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))

    results = []
    results_path = out_dir / "results.jsonl"
    if results_path.exists():
        with results_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))

    tensors: Dict[str, torch.Tensor] = {}
    tensor_dir = out_dir / "tensors"
    if tensor_dir.exists():
        for p in tensor_dir.glob("*.pt"):
            tensors[p.stem] = torch.load(p, map_location="cpu", weights_only=True)

    return {
        "config": config,
        "summary": summary,
        "results": results,
        "tensors": tensors,
    }
