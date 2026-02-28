"""
Real-model experiment pipeline (B4).
"""

from __future__ import annotations

import json
import pickle
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from boardGPT.games.othello.othello_simulator import create_move_mapping
from boardGPT.nn import GPT, GPTConfig

from .io import save_run_artifacts
from .probes import LinearBoardProbe, MLPBoardProbe
from .protocols import ProtocolConfig, run_b2_attribution_protocol, run_b2_metrics_protocol
from .report import export_metrics_csv, export_top_contributors_csv, generate_markdown_report


def load_exp_config(config_path: str) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Experiment config not found: {config_path}")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("Experiment config must be a mapping")
    return cfg


def _read_model_config(config_path: str) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Model config file not found: {config_path}")

    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".json"}:
        obj = json.loads(text)
    else:
        obj = yaml.safe_load(text)
    if not isinstance(obj, dict):
        raise ValueError(f"Model config at {config_path} must be a mapping")

    allowed = {
        "block_size",
        "vocab_size",
        "n_layer",
        "n_head",
        "n_embd",
        "dropout",
        "n_latent_token",
        "n_latent",
        "bias",
    }
    filtered = {k: v for k, v in obj.items() if k in allowed}
    if not filtered:
        raise ValueError(
            f"Model config at {config_path} does not include GPTConfig fields; "
            f"expected at least one of {sorted(allowed)}"
        )
    return filtered


def _load_model(model_cfg: Dict[str, Any]) -> GPT:
    model_path = str(model_cfg.get("model_path", ""))
    config_path = str(model_cfg.get("config_path", ""))
    device = str(model_cfg.get("device", "cpu"))

    if not model_path:
        raise ValueError("model.model_path is required")
    if not config_path:
        raise ValueError("model.config_path is required")

    config_dict = _read_model_config(config_path)
    gpt_cfg = GPTConfig(**config_dict)
    model = GPT(gpt_cfg, use_flash=False)

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    suffix = path.suffix.lower()
    if suffix in {".pt", ".pth"}:
        payload = torch.load(path, map_location=device, weights_only=True)
        if isinstance(payload, dict) and "model_state_dict" in payload:
            state_dict = payload["model_state_dict"]
        elif isinstance(payload, dict) and "model" in payload:
            state_dict = payload["model"]
        else:
            state_dict = payload
        if not isinstance(state_dict, dict):
            raise ValueError("PT/PTH checkpoint must contain a state_dict-like mapping")
        model.load_state_dict(state_dict, strict=False)
    elif suffix == ".safetensors":
        model.load_safetensors(str(path))
    else:
        raise ValueError(f"Unsupported model format: {suffix}. Use .pt/.pth/.safetensors")

    model.to(device)
    model.eval()
    return model


def _load_probe(probe_cfg: Dict[str, Any], input_dim: int, device: str) -> torch.nn.Module:
    probe_type = str(probe_cfg.get("type", "linear")).lower()
    if probe_type == "linear":
        probe = LinearBoardProbe(input_dim=input_dim)
    elif probe_type == "mlp":
        hidden_dim = int(probe_cfg.get("hidden_dim", 256))
        probe = MLPBoardProbe(input_dim=input_dim, hidden_dim=hidden_dim)
    else:
        raise ValueError(f"Unsupported probe type: {probe_type}")

    checkpoint = probe_cfg.get("path")
    if checkpoint:
        ckpt_path = Path(str(checkpoint))
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Probe checkpoint not found: {checkpoint}")
        payload = torch.load(ckpt_path, map_location=device, weights_only=True)
        if isinstance(payload, dict) and "state_dict" in payload:
            state = payload["state_dict"]
        else:
            state = payload
        if not isinstance(state, dict):
            raise ValueError("Probe checkpoint must contain a state_dict-like mapping")
        probe.load_state_dict(state, strict=False)

    probe.to(device)
    probe.eval()
    return probe


def _decode_move_token(token: Any) -> str:
    if isinstance(token, bytes):
        return token.decode("utf-8")
    return str(token)


def _game_to_token_ids(game: Any, move_to_id: Dict[str, int], block_size: int) -> Optional[List[int]]:
    seq: List[int] = [0]

    if isinstance(game, (list, tuple)):
        if len(game) == 0:
            return None
        if isinstance(game[0], (int, float)):
            raw = [int(x) for x in game]
            if raw and raw[0] == 0:
                seq = raw
            else:
                seq = [0] + raw
        else:
            for mv in game:
                mv_str = _decode_move_token(mv)
                if mv_str in move_to_id:
                    seq.append(move_to_id[mv_str])
    else:
        return None

    if len(seq) <= 1:
        return None

    if len(seq) > block_size:
        seq = seq[:block_size]
    elif len(seq) < block_size:
        seq = seq + [0] * (block_size - len(seq))
    return seq


def _load_dataset_tokens(dataset_cfg: Dict[str, Any], block_size: int) -> torch.Tensor:
    path = Path(str(dataset_cfg.get("path", "")))
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    fmt = str(dataset_cfg.get("format", path.suffix.lstrip(".").lower())).lower()
    if fmt not in {"bin", "pkl", "pickle"}:
        raise ValueError(f"Unsupported dataset format: {fmt}")

    with path.open("rb") as f:
        games = pickle.load(f)

    if not isinstance(games, (list, tuple)):
        raise ValueError("Dataset payload must be a sequence of games")

    move_to_id = create_move_mapping()
    max_examples = int(dataset_cfg.get("max_examples", len(games)))

    all_tokens: List[List[int]] = []
    for game in games:
        ids = _game_to_token_ids(game, move_to_id=move_to_id, block_size=block_size)
        if ids is None:
            continue
        all_tokens.append(ids)
        if len(all_tokens) >= max_examples:
            break

    if not all_tokens:
        raise ValueError("No valid game sequences extracted from dataset")

    return torch.tensor(all_tokens, dtype=torch.long)


def _build_loader(dataset_cfg: Dict[str, Any], block_size: int) -> DataLoader:
    batch_size = int(dataset_cfg.get("batch_size", 16))
    tokens = _load_dataset_tokens(dataset_cfg, block_size=block_size)
    return DataLoader(TensorDataset(tokens), batch_size=batch_size, shuffle=False)


def _build_protocol_config(cfg: Dict[str, Any]) -> ProtocolConfig:
    intervention = cfg.get("intervention", {})
    protocol = cfg.get("protocol", {})
    model = cfg.get("model", {})
    return ProtocolConfig(
        max_examples=int(protocol.get("max_examples", cfg.get("dataset", {}).get("max_examples", 128))),
        intervention_steps=int(intervention.get("steps", 20)),
        intervention_lr=float(intervention.get("lr", 0.08)),
        intervention_reg_strength=float(intervention.get("reg_strength", 1e-4)),
        clamp_norm=intervention.get("clamp_norm", None),
        device=str(model.get("device", "cpu")),
    )


def run_b4_single_from_config(config: Dict[str, Any], run_name: Optional[str] = None) -> Dict[str, Any]:
    model_cfg = config.get("model", {})
    dataset_cfg = config.get("dataset", {})
    protocol_cfg = config.get("protocol", {})
    report_cfg = config.get("report", {})
    probe_cfg = config.get("probe", {"type": "linear"})

    model = _load_model(model_cfg)
    probe = _load_probe(probe_cfg, input_dim=model.config.n_embd, device=str(model_cfg.get("device", "cpu")))

    loader_metrics = _build_loader(dataset_cfg, block_size=model.config.block_size)
    loader_attr = _build_loader(dataset_cfg, block_size=model.config.block_size)

    pconf = _build_protocol_config(config)
    layers = [int(x) for x in protocol_cfg.get("layers", [0, 1])]
    target_cell = int(protocol_cfg.get("target_cell", 5))
    target_class = int(protocol_cfg.get("target_class", 2))
    target_cells = [int(x) for x in protocol_cfg.get("target_cells", [target_cell])]
    position = protocol_cfg.get("position", "last")
    top_k = int(report_cfg.get("top_k", 15))

    mode = str(protocol_cfg.get("type", "metrics_and_attribution"))
    metrics_report: Optional[Dict[str, Any]] = None
    attribution_bundle: Optional[Dict[str, Any]] = None

    if mode in {"metrics_only", "metrics_and_attribution"}:
        metrics_report = run_b2_metrics_protocol(
            model=model,
            probe=probe,
            batches=loader_metrics,
            layers=layers,
            target_cell=target_cell,
            target_class=target_class,
            position=position,
            config=pconf,
        )

    if mode in {"attribution_only", "metrics_and_attribution"}:
        attribution_bundle = run_b2_attribution_protocol(
            model=model,
            probe=probe,
            batches=loader_attr,
            layers=layers,
            target_cells=target_cells,
            target_class=target_class,
            position=position,
            config=pconf,
            top_k=top_k,
        )

    run_root = Path(str(config.get("experiment", {}).get("output_root", "out_interp/b4")))
    run_name = run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = run_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    summary_obj: Dict[str, Any] = {
        "mode": mode,
        "run_name": run_name,
        "layers": layers,
        "target_class": target_class,
    }
    if metrics_report is not None:
        summary_obj["metrics"] = metrics_report.get("metrics", {})
    if attribution_bundle is not None:
        summary_obj["attribution"] = {
            "top_contributors": attribution_bundle.get("attribution", {}).get("top_contributors", [])
        }

    tensors = None
    attr_map = None
    if attribution_bundle is not None:
        attr_obj = attribution_bundle.get("attribution", {})
        candidate = attr_obj.get("attribution_map") if isinstance(attr_obj, dict) else None
        if torch.is_tensor(candidate):
            attr_map = candidate
            tensors = {"attribution_map": attr_map}

    per_example_results: Optional[List[Dict[str, Any]]] = None
    if metrics_report is not None:
        sweep = metrics_report.get("sweep", {})
        layer_results = sweep.get("layer_results", {}) if isinstance(sweep, dict) else {}
        first_layer = layers[0] if layers else 0
        first = layer_results.get(first_layer, {}) if isinstance(layer_results, dict) else {}
        rows = first.get("results") if isinstance(first, dict) else None
        if isinstance(rows, list):
            per_example_results = rows

    artifacts = save_run_artifacts(
        run_dir=str(run_dir),
        config=config,
        summary=summary_obj,
        results=per_example_results,
        tensors=tensors,
    )

    csv_files: Dict[str, str] = {}
    if metrics_report is not None and bool(report_cfg.get("export_csv", True)):
        csv_files.update(export_metrics_csv(str(run_dir), metrics_report))
    if attribution_bundle is not None and bool(report_cfg.get("export_csv", True)):
        attr_obj = attribution_bundle.get("attribution", {})
        if isinstance(attr_obj, dict):
            csv_files["top_contributors"] = export_top_contributors_csv(str(run_dir), attr_obj)

    report_md = None
    if bool(report_cfg.get("export_markdown", True)):
        report_md = generate_markdown_report(
            run_dir=str(run_dir),
            run_name=run_name,
            summary={
                "mode": mode,
                "dataset": dataset_cfg.get("path", ""),
                "model": model_cfg.get("model_path", ""),
                "layers": layers,
                "target_cell": target_cell,
                "target_class": target_class,
            },
            metrics_report=metrics_report or {"metrics": {"overall": {}, "by_layer": {}}},
            attribution_report=(attribution_bundle or {}).get("attribution", {}),
        )

    return {
        "run_dir": str(run_dir),
        "artifacts": artifacts,
        "csv": csv_files,
        "report_md": report_md,
        "metrics_report": metrics_report,
        "attribution_report": attribution_bundle,
    }


def run_b4_from_config_file(config_path: str, run_name: Optional[str] = None) -> Dict[str, Any]:
    cfg = load_exp_config(config_path)
    return run_b4_single_from_config(cfg, run_name=run_name)


def run_b4_batch_from_config_file(config_path: str, run_name: Optional[str] = None) -> Dict[str, Any]:
    cfg = load_exp_config(config_path)
    batch = cfg.get("batch", {})
    seeds = [int(x) for x in batch.get("seeds", [0])]
    checkpoints = batch.get("checkpoints", [])
    if not checkpoints:
        checkpoints = [{
            "name": "default",
            "model_path": cfg.get("model", {}).get("model_path"),
            "config_path": cfg.get("model", {}).get("config_path"),
        }]

    base_name = run_name or datetime.now().strftime("batch_%Y%m%d_%H%M%S")
    all_runs: List[Dict[str, Any]] = []

    for ckpt in checkpoints:
        ckpt_name = str(ckpt.get("name", "ckpt"))
        for seed in seeds:
            current = deepcopy(cfg)
            current.setdefault("model", {})
            current["model"]["model_path"] = ckpt.get("model_path", current["model"].get("model_path"))
            current["model"]["config_path"] = ckpt.get("config_path", current["model"].get("config_path"))
            current.setdefault("experiment", {})
            current["experiment"]["name"] = f"{base_name}_{ckpt_name}_s{seed}"

            out = run_b4_single_from_config(current, run_name=f"{base_name}_{ckpt_name}_s{seed}")

            metrics = out.get("metrics_report", {}).get("metrics", {}).get("overall", {})
            target_gain = metrics.get("target_gain", {}) if isinstance(metrics, dict) else {}
            all_runs.append(
                {
                    "checkpoint": ckpt_name,
                    "seed": seed,
                    "run_dir": out.get("run_dir"),
                    "success_rate": float(metrics.get("success_rate", 0.0)) if isinstance(metrics, dict) else 0.0,
                    "target_gain_mean": float(target_gain.get("mean", 0.0)) if isinstance(target_gain, dict) else 0.0,
                }
            )

    summary_by_checkpoint: Dict[str, Dict[str, float]] = {}
    for row in all_runs:
        ck = str(row["checkpoint"])
        bucket = summary_by_checkpoint.setdefault(ck, {"count": 0.0, "success_rate_sum": 0.0, "target_gain_sum": 0.0})
        bucket["count"] += 1.0
        bucket["success_rate_sum"] += float(row["success_rate"])
        bucket["target_gain_sum"] += float(row["target_gain_mean"])

    for ck, bucket in summary_by_checkpoint.items():
        count = max(bucket["count"], 1.0)
        bucket["success_rate_mean"] = bucket["success_rate_sum"] / count
        bucket["target_gain_mean"] = bucket["target_gain_sum"] / count

    output_root = Path(str(cfg.get("experiment", {}).get("output_root", "out_interp/b4")))
    batch_dir = output_root / base_name
    batch_dir.mkdir(parents=True, exist_ok=True)

    (batch_dir / "batch_runs.json").write_text(json.dumps(all_runs, indent=2), encoding="utf-8")
    (batch_dir / "batch_summary.json").write_text(json.dumps(summary_by_checkpoint, indent=2), encoding="utf-8")

    return {
        "batch_dir": str(batch_dir),
        "runs": all_runs,
        "summary": summary_by_checkpoint,
    }
