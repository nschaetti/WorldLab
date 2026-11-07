"""
Activation patching helpers for Streamlit UI (B5-B).
"""

from __future__ import annotations

from typing import Dict, Optional, Union

import torch
import torch.nn as nn

from worldlab.board.interpretability.interventions import (
    InterventionConfig,
    InterventionTarget,
    apply_residual_delta,
    run_probe_intervention,
)
from worldlab.board.interpretability.ui.inference import run_next_token_inference
from worldlab.board.interpretability.ui.probe_view import run_probe_on_vector
from worldlab.nn.board import GPT


PositionPolicy = Union[str, int]


@torch.no_grad()
def _summarize_logits(logits: torch.Tensor, top_k: int = 10) -> Dict[str, object]:
    last_logits = logits[:, -1, :]
    probs = torch.softmax(last_logits, dim=-1)
    k = max(1, min(top_k, probs.size(-1)))
    top_probs, top_ids = torch.topk(probs, k=k, dim=-1)
    items = [
        {"token_id": float(tid), "prob": float(prob)}
        for tid, prob in zip(top_ids[0].tolist(), top_probs[0].tolist())
    ]
    pred_id = int(torch.argmax(last_logits, dim=-1)[0].item())
    pred_prob = float(probs[0, pred_id].item())
    return {
        "prediction_token_id": pred_id,
        "prediction_prob": pred_prob,
        "top_k": items,
    }


def run_activation_patch_experiment(
    model: GPT,
    probe: nn.Module,
    tokens: torch.Tensor,
    layer: int,
    position: PositionPolicy,
    target_cell: int,
    target_class: int,
    steps: int,
    lr: float,
    reg_strength: float,
    clamp_norm: Optional[float],
    device: str = "cpu",
) -> Dict[str, object]:
    """
    Run an end-to-end patch experiment and return before/after model+probe stats.
    """
    if tokens.dim() != 2 or tokens.size(0) != 1:
        raise ValueError("tokens must have shape [1, T]")

    target = InterventionTarget(
        cell_index=int(target_cell),
        target_class=int(target_class),
        position=position,
    )
    config = InterventionConfig(
        steps=int(steps),
        lr=float(lr),
        reg_strength=float(reg_strength),
        clamp_norm=clamp_norm,
        device=device,
    )

    before_inference = run_next_token_inference(model=model, tokens=tokens, top_k=10)
    intervention = run_probe_intervention(
        model=model,
        probe=probe,
        token_batch=tokens,
        layer=int(layer),
        target=target,
        config=config,
        device=device,
    )

    delta = intervention.get("delta")
    if not torch.is_tensor(delta):
        raise RuntimeError("run_probe_intervention did not return a tensor delta")

    applied = apply_residual_delta(
        model=model,
        token_batch=tokens,
        layer=int(layer),
        delta=delta,
        position=position,
        device=device,
    )
    after_inference = _summarize_logits(applied["logits"], top_k=10)

    pos_idx = int(intervention.get("position_index", tokens.size(1) - 1))
    residuals = applied["residuals"]
    post_vec = residuals[0, pos_idx, :].detach().cpu()
    probe_after = run_probe_on_vector(probe=probe, vector=post_vec, device=device)

    return {
        "target": {
            "cell": int(target_cell),
            "class": int(target_class),
            "layer": int(layer),
            "position": position,
        },
        "before_inference": before_inference,
        "after_inference": after_inference,
        "intervention": intervention,
        "probe_after": probe_after,
    }
