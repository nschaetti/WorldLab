"""
Model and probe loaders for Streamlit interpretability UI.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml

from worldlab.nn import LinearBoardProbe, MLPBoardProbe
from worldlab.nn.board import GPT, GPTConfig


def _read_config_file(config_path: str) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    text = path.read_text(encoding="utf-8")
    data: Any
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        data = yaml.safe_load(text)

    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping: {config_path}")

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
    filtered = {k: v for k, v in data.items() if k in allowed}
    if not filtered:
        raise ValueError("Config does not include GPTConfig fields")
    return filtered


def load_model_from_paths(model_path: str, config_path: str, device: str = "cpu") -> GPT:
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    cfg = GPTConfig(**_read_config_file(config_path))
    model = GPT(cfg, use_flash=False)

    suffix = model_file.suffix.lower()
    if suffix in {".pt", ".pth"}:
        payload = torch.load(model_file, map_location=device, weights_only=False)
        if isinstance(payload, dict) and "model_state_dict" in payload:
            state = payload["model_state_dict"]
        elif isinstance(payload, dict) and "model" in payload:
            state = payload["model"]
        else:
            state = payload
        if not isinstance(state, dict):
            raise ValueError("Checkpoint does not contain a valid model state dict")
        model.load_state_dict(state, strict=False)
    elif suffix == ".safetensors":
        model.load_safetensors(str(model_file))
    else:
        raise ValueError("Unsupported model format; use .pt/.pth/.safetensors")

    model.to(device)
    model.eval()
    return model


def _extract_probe_state(payload: Any) -> Dict[str, torch.Tensor]:
    if isinstance(payload, dict) and "state_dict" in payload and isinstance(payload["state_dict"], dict):
        return payload["state_dict"]
    if isinstance(payload, dict):
        tensor_values = [v for v in payload.values() if torch.is_tensor(v)]
        if tensor_values:
            return payload
    raise ValueError("Probe checkpoint must contain a state_dict-like mapping")


def load_probe_from_path(
    probe_path: str,
    input_dim: int,
    probe_type: str = "linear",
    hidden_dim: int = 256,
    device: str = "cpu",
) -> torch.nn.Module:
    probe_type = probe_type.lower().strip()
    if probe_type == "linear":
        probe = LinearBoardProbe(input_dim=input_dim)
    elif probe_type == "mlp":
        probe = MLPBoardProbe(input_dim=input_dim, hidden_dim=hidden_dim)
    else:
        raise ValueError(f"Unsupported probe type: {probe_type}")

    path = Path(probe_path)
    if not path.exists():
        raise FileNotFoundError(f"Probe checkpoint not found: {probe_path}")

    payload = torch.load(path, map_location=device, weights_only=False)
    state = _extract_probe_state(payload)
    probe.load_state_dict(state, strict=False)
    probe.to(device)
    probe.eval()
    return probe


def build_untrained_probe(input_dim: int, probe_type: str = "linear", hidden_dim: int = 256) -> torch.nn.Module:
    probe_type = probe_type.lower().strip()
    if probe_type == "linear":
        return LinearBoardProbe(input_dim=input_dim)
    if probe_type == "mlp":
        return MLPBoardProbe(input_dim=input_dim, hidden_dim=hidden_dim)
    raise ValueError(f"Unsupported probe type: {probe_type}")
