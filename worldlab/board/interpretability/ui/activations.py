"""
Activation extraction and formatting helpers for UI.
"""

from __future__ import annotations

from typing import Dict

import torch

from worldlab.board.interpretability.activation_extractor import extract_residuals
from worldlab.nn.board import GPT


def get_residual_vector(
    model: GPT,
    tokens: torch.Tensor,
    layer: int,
    position: int,
    device: str = "cpu",
) -> torch.Tensor:
    residuals = extract_residuals(model=model, token_batch=tokens, layer=layer, device=device)
    if position < 0 or position >= residuals.size(1):
        raise ValueError(f"Position {position} out of range [0, {residuals.size(1) - 1}]")
    return residuals[0, position, :].detach().cpu()


def summarize_vector(vector: torch.Tensor) -> Dict[str, float]:
    if vector.dim() != 1:
        raise ValueError("Expected a 1D activation vector")
    return {
        "dim": float(vector.numel()),
        "norm": float(vector.norm().item()),
        "mean": float(vector.mean().item()),
        "std": float(vector.std().item()),
        "min": float(vector.min().item()),
        "max": float(vector.max().item()),
    }
