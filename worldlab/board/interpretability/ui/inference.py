"""
Inference helpers for interpretability UI.
"""

from __future__ import annotations

from typing import Dict, List

import torch

from worldlab.nn.board import GPT


@torch.no_grad()
def run_next_token_inference(model: GPT, tokens: torch.Tensor, top_k: int = 10) -> Dict[str, object]:
    if tokens.dim() != 2:
        raise ValueError(f"Expected tokens shape [B,T], got {tuple(tokens.shape)}")

    _, logits, _ = model(idx=tokens)
    last_logits = logits[:, -1, :]
    probs = torch.softmax(last_logits, dim=-1)

    k = max(1, min(top_k, probs.size(-1)))
    top_probs, top_ids = torch.topk(probs, k=k, dim=-1)

    top_items: List[Dict[str, float]] = []
    for token_id, prob in zip(top_ids[0].tolist(), top_probs[0].tolist()):
        top_items.append({"token_id": float(token_id), "prob": float(prob)})

    pred_id = int(torch.argmax(last_logits, dim=-1)[0].item())
    pred_prob = float(probs[0, pred_id].item())

    return {
        "prediction_token_id": pred_id,
        "prediction_prob": pred_prob,
        "top_k": top_items,
        "last_logits": last_logits.detach().cpu(),
    }
