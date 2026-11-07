"""
Real-model experiment pipeline (B4).
"""

from __future__ import annotations

import array
import csv
import json
import pickle
import resource
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from worldlab.board.games.othello.othello_simulator import OthelloGame, create_id_to_move_mapping, create_move_mapping
from worldlab.nn.board import GPT, GPTConfig

from .activation_extractor import extract_dataset_residuals
from .io import save_run_artifacts
from .probes import LinearBoardProbe, MLPBoardProbe, eval_probe, fit_probe
from .protocols import ProtocolConfig, run_b2_attribution_protocol, run_b2_metrics_protocol
from .reproducibility import collect_environment_snapshot, set_global_determinism, write_reproducibility_pack
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


def _load_probe(probe_cfg: Dict[str, Any], input_dim: int, device: str) -> Tuple[torch.nn.Module, bool]:
    probe_type = str(probe_cfg.get("type", "linear")).lower()
    if probe_type == "linear":
        probe = LinearBoardProbe(input_dim=input_dim)
    elif probe_type == "mlp":
        hidden_dim = int(probe_cfg.get("hidden_dim", 256))
        probe = MLPBoardProbe(input_dim=input_dim, hidden_dim=hidden_dim)
    else:
        raise ValueError(f"Unsupported probe type: {probe_type}")

    checkpoint = probe_cfg.get("path")
    loaded_from_checkpoint = False
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
        loaded_from_checkpoint = True

    probe.to(device)
    probe.eval()
    return probe, loaded_from_checkpoint


def _decode_move_token(token: Any) -> str:
    if isinstance(token, bytes):
        return token.decode("utf-8")
    return str(token)


def _record_issue(
    validation_mode: str,
    warnings: List[str],
    message: str,
    strict_error: bool = True,
) -> None:
    mode = validation_mode.strip().lower()
    if strict_error and mode == "strict":
        raise ValueError(message)
    warnings.append(message)


def _infer_dataset_format(path: Path, explicit_format: Optional[str]) -> str:
    if explicit_format:
        return explicit_format.strip().lower()
    suffix = path.suffix.lower().lstrip(".")
    if suffix in {"pkl", "pickle", "bin", "npy", "npz", "json", "jsonl"}:
        return suffix
    return "pkl"


def _load_dataset_payload(path: Path, fmt: str) -> Any:
    if fmt in {"bin", "pkl", "pickle"}:
        with path.open("rb") as f:
            return pickle.load(f)
    if fmt == "npy":
        return np.load(path, allow_pickle=True)
    if fmt == "npz":
        obj = np.load(path, allow_pickle=True)
        if len(obj.files) == 1:
            return obj[obj.files[0]]
        return {k: obj[k] for k in obj.files}
    if fmt == "json":
        return json.loads(path.read_text(encoding="utf-8"))
    if fmt == "jsonl":
        rows: List[Any] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
        return rows
    raise ValueError(f"Unsupported dataset format: {fmt}")


def _iter_games_from_payload(payload: Any) -> Iterable[Any]:
    if isinstance(payload, dict):
        if "games" in payload and isinstance(payload["games"], (list, tuple, np.ndarray)):
            return payload["games"]
        if "data" in payload and isinstance(payload["data"], (list, tuple, np.ndarray)):
            return payload["data"]
        # fallback: values can be games in keyed datasets
        return list(payload.values())
    if isinstance(payload, (list, tuple, np.ndarray)):
        return payload
    raise ValueError("Unsupported dataset payload type; expected sequence or mapping with game lists")


def _normalize_game_object(game: Any) -> Any:
    if isinstance(game, dict):
        for key in ("moves", "sequence", "tokens", "game"):
            if key in game:
                return game[key]
    return game


def _game_to_token_ids(
    game: Any,
    move_to_id: Dict[str, int],
    block_size: int,
    validation_mode: str,
    warnings: List[str],
) -> Optional[List[int]]:
    seq: List[int] = [0]

    game = _normalize_game_object(game)

    if isinstance(game, array.array):
        game = list(game)

    if isinstance(game, np.ndarray):
        if game.ndim == 0:
            game = [int(game.item())]
        elif game.ndim == 1:
            game = game.tolist()
        else:
            _record_issue(validation_mode, warnings, f"Skipping game with ndarray ndim={game.ndim}", strict_error=False)
            return None

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
                    _record_issue(
                        validation_mode,
                        warnings,
                        f"Unknown move token '{mv_str}' encountered; skipped",
                        strict_error=False,
                    )
    else:
        return None

    if len(seq) <= 1:
        return None

    if len(seq) > block_size:
        seq = seq[:block_size]
    elif len(seq) < block_size:
        seq = seq + [0] * (block_size - len(seq))
    return seq


def _load_dataset_tokens(
    dataset_cfg: Dict[str, Any],
    block_size: int,
    validation_mode: str,
    warnings: List[str],
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    path = Path(str(dataset_cfg.get("path", "")))
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    fmt = _infer_dataset_format(path, dataset_cfg.get("format"))
    payload = _load_dataset_payload(path, fmt)
    games = list(_iter_games_from_payload(payload))

    move_to_id = create_move_mapping()
    max_examples = int(dataset_cfg.get("max_examples", len(games)))

    all_tokens: List[List[int]] = []
    skipped = 0
    for game in games:
        ids = _game_to_token_ids(
            game,
            move_to_id=move_to_id,
            block_size=block_size,
            validation_mode=validation_mode,
            warnings=warnings,
        )
        if ids is None:
            skipped += 1
            continue
        all_tokens.append(ids)
        if len(all_tokens) >= max_examples:
            break

    if not all_tokens:
        raise ValueError("No valid game sequences extracted from dataset")

    tensor = torch.tensor(all_tokens, dtype=torch.long)
    info = {
        "format": fmt,
        "num_games_in_payload": len(games),
        "num_examples_loaded": int(tensor.size(0)),
        "num_skipped": skipped,
    }
    return tensor, info


def _build_loader(
    dataset_cfg: Dict[str, Any],
    block_size: int,
    validation_mode: str,
    warnings: List[str],
) -> Tuple[DataLoader, Dict[str, Any], torch.Tensor]:
    batch_size = int(dataset_cfg.get("batch_size", 16))
    tokens, info = _load_dataset_tokens(
        dataset_cfg,
        block_size=block_size,
        validation_mode=validation_mode,
        warnings=warnings,
    )
    return DataLoader(TensorDataset(tokens), batch_size=batch_size, shuffle=False), info, tokens


def _memory_snapshot(device: str) -> Dict[str, float]:
    rss_mb = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0
    out: Dict[str, float] = {"cpu_maxrss_mb": rss_mb}
    if str(device).startswith("cuda") and torch.cuda.is_available():
        dev = torch.device(device)
        out["gpu_max_allocated_mb"] = float(torch.cuda.max_memory_allocated(dev)) / (1024.0 ** 2)
        out["gpu_max_reserved_mb"] = float(torch.cuda.max_memory_reserved(dev)) / (1024.0 ** 2)
    return out


def _token_ids_to_board_labels(tokens: torch.Tensor) -> torch.Tensor:
    """
    Convert tokenized move sequences to final-board cell labels [N,64].

    Sequence convention: 0 is BOS/padding and ignored for replay.
    """
    id_to_move = create_id_to_move_mapping()
    labels: List[torch.Tensor] = []

    for row in tokens.tolist():
        move_ids = [int(x) for x in row if int(x) != 0]
        moves = [id_to_move[mid] for mid in move_ids if mid in id_to_move]
        game = OthelloGame.load_moves(moves)
        board = game.flattened_board_state().astype(np.int64)
        labels.append(torch.tensor(board, dtype=torch.long))

    return torch.stack(labels, dim=0)


def _auto_train_probe_if_requested(
    probe: torch.nn.Module,
    model: GPT,
    probe_cfg: Dict[str, Any],
    protocol_cfg: Dict[str, Any],
    dataset_tokens: torch.Tensor,
    device: str,
    warnings: List[str],
    validation_mode: str,
) -> Dict[str, Any]:
    auto = bool(probe_cfg.get("auto_train_if_missing", False))
    if not auto:
        return {"enabled": False, "trained": False}

    epochs = int(probe_cfg.get("auto_train_epochs", 3))
    lr = float(probe_cfg.get("auto_train_lr", 1e-3))
    batch_size = int(probe_cfg.get("auto_train_batch_size", 32))
    max_examples = int(probe_cfg.get("auto_train_max_examples", min(int(dataset_tokens.size(0)), 256)))
    layer = int(probe_cfg.get("auto_train_layer", (protocol_cfg.get("layers") or [0])[0]))
    position = probe_cfg.get("auto_train_position", protocol_cfg.get("position", "last"))

    if str(position) != "last":
        _record_issue(
            validation_mode,
            warnings,
            "auto_train_probe currently supports position='last'; falling back to 'last'",
            strict_error=False,
        )
        position = "last"

    subset = dataset_tokens[:max_examples]
    if subset.size(0) < 2:
        _record_issue(validation_mode, warnings, "Not enough examples for probe auto-training", strict_error=False)
        return {"enabled": True, "trained": False}

    tmp_loader = DataLoader(TensorDataset(subset), batch_size=batch_size, shuffle=False)
    features = extract_dataset_residuals(
        model=model,
        batches=tmp_loader,
        layer=layer,
        position_policy="last",
        device=device,
        return_cpu=True,
    )
    labels = _token_ids_to_board_labels(subset)

    n = int(features.size(0))
    split = max(1, int(0.8 * n))
    if split >= n:
        split = n - 1
    train_ds = TensorDataset(features[:split], labels[:split])
    val_ds = TensorDataset(features[split:], labels[split:])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    history = fit_probe(
        probe=probe,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        learning_rate=lr,
        device=device,
    )
    eval_stats = eval_probe(probe=probe, data_loader=val_loader, device=device)
    probe.eval()
    return {
        "enabled": True,
        "trained": True,
        "epochs": epochs,
        "learning_rate": lr,
        "batch_size": batch_size,
        "max_examples": int(subset.size(0)),
        "layer": layer,
        "position": "last",
        "history": history,
        "val": {
            "loss": float(eval_stats["loss"]),
            "acc_global": float(eval_stats["acc_global"]),
        },
    }


def _validate_model_dataset_compat(
    model: GPT,
    dataset_tokens: torch.Tensor,
    dataset_block_size: int,
    validation_mode: str,
    warnings: List[str],
) -> None:
    expected_vocab = len(create_move_mapping())
    model_vocab = int(model.config.vocab_size)

    if model_vocab < expected_vocab:
        _record_issue(
            validation_mode,
            warnings,
            f"Model vocab_size={model_vocab} smaller than expected Othello move mapping size={expected_vocab}",
            strict_error=True,
        )
    elif model_vocab != expected_vocab:
        _record_issue(
            validation_mode,
            warnings,
            f"Model vocab_size={model_vocab} differs from Othello mapping size={expected_vocab}",
            strict_error=False,
        )

    max_token = int(dataset_tokens.max().item())
    if max_token >= model_vocab:
        _record_issue(
            validation_mode,
            warnings,
            f"Dataset contains token id {max_token} >= model vocab_size {model_vocab}",
            strict_error=True,
        )

    if int(model.config.block_size) != int(dataset_block_size):
        _record_issue(
            validation_mode,
            warnings,
            f"dataset.block_size={dataset_block_size} differs from model.block_size={model.config.block_size}; "
            "tokens are currently padded/truncated to model.block_size",
            strict_error=False,
        )


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
    experiment_cfg = config.get("experiment", {})

    seed = int(experiment_cfg.get("seed", 0))
    repro_cfg = experiment_cfg.get("reproducibility", {}) if isinstance(experiment_cfg, dict) else {}
    deterministic = bool(repro_cfg.get("deterministic", True))
    include_pip_freeze = bool(repro_cfg.get("include_pip_freeze", False))

    seed_info = set_global_determinism(seed=seed, deterministic=deterministic)
    env_snapshot = collect_environment_snapshot(include_pip_freeze=include_pip_freeze)

    validation_mode = str(experiment_cfg.get("validation_mode", "strict")).strip().lower()
    if validation_mode not in {"strict", "tolerant"}:
        raise ValueError(f"Unsupported validation_mode: {validation_mode}; expected 'strict' or 'tolerant'")

    warnings: List[str] = []
    stage_times: Dict[str, float] = {}
    t_start = time.perf_counter()
    model_device = str(model_cfg.get("device", "cpu"))
    if model_device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(torch.device(model_device))

    t0 = time.perf_counter()
    model = _load_model(model_cfg)
    stage_times["load_model_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    probe, probe_loaded_from_checkpoint = _load_probe(probe_cfg, input_dim=model.config.n_embd, device=model_device)
    stage_times["load_probe_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    loader_metrics, dataset_info, dataset_tokens = _build_loader(
        dataset_cfg,
        block_size=model.config.block_size,
        validation_mode=validation_mode,
        warnings=warnings,
    )
    loader_attr = DataLoader(TensorDataset(dataset_tokens), batch_size=int(dataset_cfg.get("batch_size", 16)), shuffle=False)
    stage_times["build_loaders_s"] = time.perf_counter() - t0

    _validate_model_dataset_compat(
        model=model,
        dataset_tokens=dataset_tokens,
        dataset_block_size=int(dataset_cfg.get("block_size", model.config.block_size)),
        validation_mode=validation_mode,
        warnings=warnings,
    )

    pconf = _build_protocol_config(config)
    layers = [int(x) for x in protocol_cfg.get("layers", [0, 1])]
    target_cell = int(protocol_cfg.get("target_cell", 5))
    target_class = int(protocol_cfg.get("target_class", 2))
    target_cells = [int(x) for x in protocol_cfg.get("target_cells", [target_cell])]
    position = protocol_cfg.get("position", "last")
    top_k = int(report_cfg.get("top_k", 15))

    auto_probe_info = {"enabled": False, "trained": False}
    if not probe_loaded_from_checkpoint:
        t0 = time.perf_counter()
        auto_probe_info = _auto_train_probe_if_requested(
            probe=probe,
            model=model,
            probe_cfg=probe_cfg,
            protocol_cfg=protocol_cfg,
            dataset_tokens=dataset_tokens,
            device=model_device,
            warnings=warnings,
            validation_mode=validation_mode,
        )
        stage_times["auto_train_probe_s"] = time.perf_counter() - t0

    mode = str(protocol_cfg.get("type", "metrics_and_attribution"))
    metrics_report: Optional[Dict[str, Any]] = None
    attribution_bundle: Optional[Dict[str, Any]] = None

    if mode in {"metrics_only", "metrics_and_attribution"}:
        t0 = time.perf_counter()
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
        stage_times["metrics_protocol_s"] = time.perf_counter() - t0

    if mode in {"attribution_only", "metrics_and_attribution"}:
        t0 = time.perf_counter()
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
        stage_times["attribution_protocol_s"] = time.perf_counter() - t0

    run_root = Path(str(config.get("experiment", {}).get("output_root", "out_interp/b4")))
    run_name = run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = run_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    telemetry = {
        "stage_times_s": stage_times,
        "elapsed_total_s": time.perf_counter() - t_start,
        "memory": _memory_snapshot(model_device),
    }

    integrity = "clean" if len(warnings) == 0 else "degraded"

    summary_obj: Dict[str, Any] = {
        "mode": mode,
        "run_name": run_name,
        "layers": layers,
        "target_class": target_class,
        "validation_mode": validation_mode,
        "run_integrity": integrity,
        "warnings": warnings,
        "telemetry": telemetry,
        "dataset": dataset_info,
        "probe": {
            "type": str(probe_cfg.get("type", "linear")),
            "checkpoint_loaded": probe_loaded_from_checkpoint,
            "auto_train": auto_probe_info,
        },
        "reproducibility": {
            "seed": seed_info,
            "deterministic": deterministic,
            "include_pip_freeze": include_pip_freeze,
        },
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

    t0 = time.perf_counter()
    artifacts = save_run_artifacts(
        run_dir=str(run_dir),
        config=config,
        summary=summary_obj,
        results=per_example_results,
        tensors=tensors,
    )
    stage_times["save_artifacts_s"] = time.perf_counter() - t0

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

    repro_artifacts = write_reproducibility_pack(
        run_dir=str(run_dir),
        config=config,
        seed_info=seed_info,
        env_snapshot=env_snapshot,
    )

    # refresh final telemetry with potential extra stages after summary initial build
    summary_obj["telemetry"] = {
        "stage_times_s": stage_times,
        "elapsed_total_s": time.perf_counter() - t_start,
        "memory": _memory_snapshot(model_device),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary_obj, indent=2), encoding="utf-8")

    return {
        "run_dir": str(run_dir),
        "artifacts": artifacts,
        "csv": csv_files,
        "report_md": report_md,
        "reproducibility": repro_artifacts,
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
            current["experiment"]["seed"] = int(seed)

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


def _default_checkpoints(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    batch = cfg.get("batch", {}) if isinstance(cfg, dict) else {}
    checkpoints = batch.get("checkpoints", []) if isinstance(batch, dict) else []
    if checkpoints:
        return [dict(x) for x in checkpoints if isinstance(x, dict)]

    model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    return [
        {
            "name": "default",
            "model_path": model_cfg.get("model_path"),
            "config_path": model_cfg.get("config_path"),
        }
    ]


def _extract_metrics_from_summary(summary: Dict[str, Any]) -> Dict[str, float]:
    metrics = summary.get("metrics", {}) if isinstance(summary, dict) else {}
    overall = metrics.get("overall", {}) if isinstance(metrics, dict) else {}
    if not isinstance(overall, dict):
        overall = {}
    target_gain = overall.get("target_gain", {}) if isinstance(overall, dict) else {}
    if not isinstance(target_gain, dict):
        target_gain = {}
    telemetry = summary.get("telemetry", {}) if isinstance(summary, dict) else {}
    if not isinstance(telemetry, dict):
        telemetry = {}

    return {
        "success_rate": float(overall.get("success_rate", 0.0)),
        "target_gain_mean": float(target_gain.get("mean", 0.0)),
        "elapsed_total_s": float(telemetry.get("elapsed_total_s", 0.0)),
    }


def _matrix_run_name(
    base_name: str,
    checkpoint: str,
    seed: int,
    layer: int,
    target_cell: int,
    target_class: int,
    position: Any,
) -> str:
    pos = str(position).replace(" ", "")
    return f"{base_name}_{checkpoint}_s{seed}_l{layer}_cell{target_cell}_cls{target_class}_p{pos}"


def run_b6_matrix_from_config_file(
    config_path: str,
    run_name: Optional[str] = None,
    resume: bool = True,
) -> Dict[str, Any]:
    """
    Run a systematic B6 experiment matrix from config.

    Matrix dimensions (config section `matrix`):
    - `layers`
    - `target_cells`
    - `target_classes`
    - `positions`
    - `seeds`
    - `checkpoints` (optional override; defaults to `batch.checkpoints`)

    Each grid point launches one B4 run with a deterministic run name.
    Results are aggregated into JSON + CSV artifacts.
    """
    cfg = load_exp_config(config_path)
    matrix = cfg.get("matrix", {}) if isinstance(cfg, dict) else {}
    if not isinstance(matrix, dict):
        raise ValueError("`matrix` section must be a mapping")

    protocol = cfg.get("protocol", {}) if isinstance(cfg, dict) else {}
    batch = cfg.get("batch", {}) if isinstance(cfg, dict) else {}

    layers = [int(x) for x in matrix.get("layers", protocol.get("layers", [0]))]
    target_cells = [int(x) for x in matrix.get("target_cells", protocol.get("target_cells", [protocol.get("target_cell", 5)]))]
    target_classes = [int(x) for x in matrix.get("target_classes", [protocol.get("target_class", 2)])]
    positions_raw = matrix.get("positions", [protocol.get("position", "last")])
    positions = [str(x) if not isinstance(x, int) else int(x) for x in positions_raw]
    seeds = [int(x) for x in matrix.get("seeds", batch.get("seeds", [0]))]
    checkpoints = matrix.get("checkpoints", _default_checkpoints(cfg))
    if not isinstance(checkpoints, list) or not checkpoints:
        raise ValueError("No checkpoints available for matrix run")

    continue_on_error = bool(matrix.get("continue_on_error", True))
    if "resume" in matrix:
        resume = bool(matrix.get("resume"))

    base_name = run_name or datetime.now().strftime("matrix_%Y%m%d_%H%M%S")
    output_root = Path(str(cfg.get("experiment", {}).get("output_root", "out_interp/b6")))
    matrix_dir = output_root / base_name
    matrix_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []

    for ckpt in checkpoints:
        if not isinstance(ckpt, dict):
            continue
        ckpt_name = str(ckpt.get("name", "ckpt"))
        ckpt_model_path = ckpt.get("model_path")
        ckpt_config_path = ckpt.get("config_path")

        for seed in seeds:
            for layer in layers:
                for target_cell in target_cells:
                    for target_class in target_classes:
                        for position in positions:
                            run_id = _matrix_run_name(
                                base_name=base_name,
                                checkpoint=ckpt_name,
                                seed=seed,
                                layer=layer,
                                target_cell=target_cell,
                                target_class=target_class,
                                position=position,
                            )
                            run_dir = matrix_dir / run_id
                            summary_path = run_dir / "summary.json"

                            if resume and summary_path.exists():
                                try:
                                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                                    metrics = _extract_metrics_from_summary(summary)
                                    rows.append(
                                        {
                                            "run_id": run_id,
                                            "status": "resumed",
                                            "checkpoint": ckpt_name,
                                            "seed": seed,
                                            "layer": layer,
                                            "target_cell": target_cell,
                                            "target_class": target_class,
                                            "position": position,
                                            "run_dir": str(run_dir),
                                            "success_rate": metrics["success_rate"],
                                            "target_gain_mean": metrics["target_gain_mean"],
                                            "elapsed_total_s": metrics["elapsed_total_s"],
                                            "error": "",
                                        }
                                    )
                                    continue
                                except Exception:
                                    # corrupted resume artifact -> re-run
                                    pass

                            current = deepcopy(cfg)
                            current.setdefault("experiment", {})
                            current["experiment"]["name"] = run_id
                            current["experiment"]["output_root"] = str(matrix_dir)
                            current["experiment"]["seed"] = int(seed)

                            current.setdefault("model", {})
                            current["model"]["model_path"] = ckpt_model_path or current["model"].get("model_path")
                            current["model"]["config_path"] = ckpt_config_path or current["model"].get("config_path")

                            current.setdefault("protocol", {})
                            current["protocol"]["layers"] = [layer]
                            current["protocol"]["target_cell"] = target_cell
                            current["protocol"]["target_cells"] = [target_cell]
                            current["protocol"]["target_class"] = target_class
                            current["protocol"]["position"] = position
                            if "type" in matrix:
                                current["protocol"]["type"] = str(matrix.get("type"))

                            try:
                                out = run_b4_single_from_config(current, run_name=run_id)
                                summary = json.loads((Path(out["run_dir"]) / "summary.json").read_text(encoding="utf-8"))
                                metrics = _extract_metrics_from_summary(summary)
                                rows.append(
                                    {
                                        "run_id": run_id,
                                        "status": "completed",
                                        "checkpoint": ckpt_name,
                                        "seed": seed,
                                        "layer": layer,
                                        "target_cell": target_cell,
                                        "target_class": target_class,
                                        "position": position,
                                        "run_dir": str(out.get("run_dir", run_dir)),
                                        "success_rate": metrics["success_rate"],
                                        "target_gain_mean": metrics["target_gain_mean"],
                                        "elapsed_total_s": metrics["elapsed_total_s"],
                                        "error": "",
                                    }
                                )
                            except Exception as exc:
                                rows.append(
                                    {
                                        "run_id": run_id,
                                        "status": "failed",
                                        "checkpoint": ckpt_name,
                                        "seed": seed,
                                        "layer": layer,
                                        "target_cell": target_cell,
                                        "target_class": target_class,
                                        "position": position,
                                        "run_dir": str(run_dir),
                                        "success_rate": 0.0,
                                        "target_gain_mean": 0.0,
                                        "elapsed_total_s": 0.0,
                                        "error": str(exc),
                                    }
                                )
                                if not continue_on_error:
                                    raise

    # Aggregations
    by_checkpoint: Dict[str, Dict[str, float]] = {}
    by_layer: Dict[str, Dict[str, float]] = {}
    completed_rows = [r for r in rows if r.get("status") in {"completed", "resumed"}]

    for r in completed_rows:
        ck = str(r.get("checkpoint", ""))
        lk = str(r.get("layer", ""))
        for key, container in ((ck, by_checkpoint), (lk, by_layer)):
            bucket = container.setdefault(key, {"count": 0.0, "success_rate_sum": 0.0, "target_gain_sum": 0.0})
            bucket["count"] += 1.0
            bucket["success_rate_sum"] += float(r.get("success_rate", 0.0))
            bucket["target_gain_sum"] += float(r.get("target_gain_mean", 0.0))

    for container in (by_checkpoint, by_layer):
        for _, bucket in container.items():
            c = max(bucket.get("count", 0.0), 1.0)
            bucket["success_rate_mean"] = bucket.get("success_rate_sum", 0.0) / c
            bucket["target_gain_mean"] = bucket.get("target_gain_sum", 0.0) / c

    # Persist artifacts
    runs_json = matrix_dir / "matrix_runs.json"
    results_csv = matrix_dir / "matrix_results.csv"
    summary_json = matrix_dir / "matrix_summary.json"

    runs_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    with results_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "run_id",
            "status",
            "checkpoint",
            "seed",
            "layer",
            "target_cell",
            "target_class",
            "position",
            "run_dir",
            "success_rate",
            "target_gain_mean",
            "elapsed_total_s",
            "error",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    summary_obj = {
        "num_runs_total": len(rows),
        "num_runs_completed": len([r for r in rows if r.get("status") == "completed"]),
        "num_runs_resumed": len([r for r in rows if r.get("status") == "resumed"]),
        "num_runs_failed": len([r for r in rows if r.get("status") == "failed"]),
        "by_checkpoint": by_checkpoint,
        "by_layer": by_layer,
    }
    summary_json.write_text(json.dumps(summary_obj, indent=2), encoding="utf-8")

    return {
        "matrix_dir": str(matrix_dir),
        "runs": rows,
        "summary": summary_obj,
        "artifacts": {
            "matrix_runs_json": str(runs_json),
            "matrix_results_csv": str(results_csv),
            "matrix_summary_json": str(summary_json),
        },
    }
