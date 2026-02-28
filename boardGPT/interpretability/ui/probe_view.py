"""
Probe decoding helpers for UI.
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn


CLASS_TO_SYMBOL = {
    0: ".",
    1: "W",
    2: "B",
}


@torch.no_grad()
def run_probe_on_vector(probe: nn.Module, vector: torch.Tensor, device: str = "cpu") -> Dict[str, torch.Tensor]:
    if vector.dim() != 1:
        raise ValueError("Expected 1D activation vector")
    probe = probe.to(device)
    probe.eval()
    x = vector.unsqueeze(0).to(device)
    logits = probe(x)
    probs = torch.softmax(logits, dim=-1)
    pred = torch.argmax(probs, dim=-1)
    conf = torch.max(probs, dim=-1).values
    return {
        "logits": logits[0].detach().cpu(),
        "probs": probs[0].detach().cpu(),
        "pred": pred[0].detach().cpu(),
        "conf": conf[0].detach().cpu(),
    }


def _cell_index_from_row_col(row: int, col: int) -> int:
    # board labels convention is column-major: a1,a2,...,a8,b1,...,h8
    return col * 8 + row


def format_board_symbols(pred: torch.Tensor) -> List[List[str]]:
    if pred.shape != (64,):
        raise ValueError(f"Expected predicted classes with shape (64,), got {tuple(pred.shape)}")
    grid: List[List[str]] = []
    for row in range(8):
        line: List[str] = []
        for col in range(8):
            idx = _cell_index_from_row_col(row, col)
            label = int(pred[idx].item())
            line.append(CLASS_TO_SYMBOL.get(label, "?"))
        grid.append(line)
    return grid


def format_board_confidence(conf: torch.Tensor) -> List[List[float]]:
    if conf.shape != (64,):
        raise ValueError(f"Expected confidence with shape (64,), got {tuple(conf.shape)}")
    grid: List[List[float]] = []
    for row in range(8):
        line: List[float] = []
        for col in range(8):
            idx = _cell_index_from_row_col(row, col)
            line.append(float(conf[idx].item()))
        grid.append(line)
    return grid
