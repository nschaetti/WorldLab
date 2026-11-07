"""
Latent intervention primitives for BoardGPT interpretability workflows.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

from worldlab.nn.board import GPT

from .activation_extractor import extract_residuals


PositionPolicy = Union[str, int]


@dataclass
class InterventionTarget:
    """
    Target specification for board-state intervention.
    """

    cell_index: int
    target_class: int
    position: PositionPolicy = "last"


@dataclass
class InterventionConfig:
    """
    Optimization hyperparameters for residual intervention.
    """

    steps: int = 200
    lr: float = 5e-2
    reg_strength: float = 1e-3
    clamp_norm: Optional[float] = None
    device: Optional[Union[str, torch.device]] = None


def _validate_target(target: InterventionTarget) -> None:
    if target.cell_index < 0 or target.cell_index >= 64:
        raise ValueError(f"cell_index must be in [0, 63], got {target.cell_index}")
    if target.target_class < 0 or target.target_class >= 3:
        raise ValueError(f"target_class must be in [0, 2], got {target.target_class}")


def _resolve_position_index(seq_len: int, position: PositionPolicy) -> int:
    if position == "last":
        return seq_len - 1
    if isinstance(position, int):
        if position < 0 or position >= seq_len:
            raise ValueError(f"position {position} out of range [0, {seq_len - 1}]")
        return position
    raise ValueError("position must be 'last' or an integer")


def _ensure_2d_activations(activations: torch.Tensor) -> torch.Tensor:
    if activations.dim() == 1:
        return activations.unsqueeze(0)
    if activations.dim() == 2:
        return activations
    raise ValueError(f"Expected activations with shape [D] or [B,D], got {tuple(activations.shape)}")


def _probe_target_probs(probe: nn.Module, activations: torch.Tensor, target: InterventionTarget) -> torch.Tensor:
    logits = probe(activations)
    if logits.dim() != 3:
        raise ValueError(f"Probe output must be [B,64,3], got {tuple(logits.shape)}")
    if logits.size(1) != 64 or logits.size(2) != 3:
        raise ValueError(f"Probe output second/third dims must be (64,3), got {tuple(logits.shape)}")
    target_logits = logits[:, target.cell_index, :]
    return target_logits.softmax(dim=-1)[:, target.target_class]


def optimize_residual_delta(
    probe: nn.Module,
    base_activations: torch.Tensor,
    target: InterventionTarget,
    config: Optional[InterventionConfig] = None,
) -> Dict[str, object]:
    """
    Optimize an additive residual delta to increase probe confidence on a target.

    Args:
        probe: Probe model that maps activations to board logits [B,64,3].
        base_activations: Activation vectors [D] or [B,D].
        target: Target cell/class and sequence position.
        config: Optimization hyperparameters.

    Returns:
        Dictionary containing delta, optimized activations and optimization traces.
    """
    _validate_target(target)
    config = InterventionConfig() if config is None else config
    acts = _ensure_2d_activations(base_activations)

    if config.device is None:
        config_device = acts.device
    else:
        config_device = torch.device(config.device)

    probe = probe.to(config_device)
    probe.eval()

    acts = acts.to(config_device)
    delta = torch.zeros_like(acts, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=config.lr)

    target_idx = torch.full((acts.size(0),), target.target_class, dtype=torch.long, device=config_device)

    with torch.no_grad():
        before_probs = _probe_target_probs(probe, acts, target)
        before_mean = before_probs.mean().item()

    loss_trace: List[float] = []
    prob_trace: List[float] = []

    for _ in range(config.steps):
        optimizer.zero_grad()
        intervened = acts + delta
        logits = probe(intervened)
        cell_logits = logits[:, target.cell_index, :]

        loss_target = nn.functional.cross_entropy(cell_logits, target_idx)
        loss_reg = config.reg_strength * torch.mean(delta.pow(2))
        loss = loss_target + loss_reg
        loss.backward()
        optimizer.step()

        if config.clamp_norm is not None:
            with torch.no_grad():
                norms = delta.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                factors = torch.clamp(config.clamp_norm / norms, max=1.0)
                delta.mul_(factors)

        with torch.no_grad():
            probs = _probe_target_probs(probe, acts + delta, target)
            loss_trace.append(loss.item())
            prob_trace.append(probs.mean().item())

    optimized = (acts + delta).detach()
    delta_detached = delta.detach()

    with torch.no_grad():
        after_probs = _probe_target_probs(probe, optimized, target)
        after_mean = after_probs.mean().item()

    return {
        "delta": delta_detached,
        "optimized_activations": optimized,
        "before_target_prob": before_mean,
        "after_target_prob": after_mean,
        "target_prob_gain": after_mean - before_mean,
        "delta_norm": torch.mean(delta_detached.norm(dim=-1)).item(),
        "loss_trace": loss_trace,
        "target_prob_trace": prob_trace,
    }


@torch.no_grad()
def apply_residual_delta(
    model: GPT,
    token_batch: torch.Tensor,
    layer: int,
    delta: torch.Tensor,
    position: PositionPolicy = "last",
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Run model once with an additive intervention on residual stream.

    Args:
        model: BoardGPT GPT model.
        token_batch: Input token IDs [B,T].
        layer: Residual layer index in [0, n_layer-1].
        delta: Delta tensor [D] or [B,D].
        position: "last" or a fixed token index.
        device: Compute device.

    Returns:
        Dict with `logits` and `residuals` tensors.
    """
    if layer < 0 or layer >= model.config.n_layer:
        raise ValueError(f"layer must be in [0, {model.config.n_layer - 1}], got {layer}")

    token_batch = token_batch.long()
    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)

    delta = delta.to(device)
    if delta.dim() == 1:
        delta = delta.unsqueeze(0)
    if delta.dim() != 2:
        raise ValueError(f"delta must have shape [D] or [B,D], got {tuple(delta.shape)}")

    was_training = model.training
    model.eval()

    def _hook(x: torch.Tensor) -> torch.Tensor:
        out = x.clone()
        idx = _resolve_position_index(out.size(1), position)
        if delta.size(0) == 1:
            out[:, idx, :] = out[:, idx, :] + delta[0]
        elif delta.size(0) == out.size(0):
            out[:, idx, :] = out[:, idx, :] + delta
        else:
            raise ValueError(
                f"delta batch dimension mismatch: delta has {delta.size(0)} rows, "
                f"residual batch has {out.size(0)} rows"
            )
        return out

    hook_point = model.residual_hook[layer]
    hook_point.add_hook(_hook)
    try:
        _, logits, obj = model(idx=token_batch.to(device), to_return=[f"residuals{layer}"])
        if len(obj) != 1:
            raise RuntimeError(f"Expected one residual tensor from model, got {len(obj)}")
        residuals = obj[0]
    finally:
        hook_point.fns.remove(_hook)
        if was_training:
            model.train()

    return {
        "logits": logits,
        "residuals": residuals,
    }


def run_probe_intervention(
    model: GPT,
    probe: nn.Module,
    token_batch: torch.Tensor,
    layer: int,
    target: InterventionTarget,
    config: Optional[InterventionConfig] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, object]:
    """
    End-to-end single intervention run:
    1) extract residual
    2) optimize delta against probe target
    3) apply delta in model forward
    4) compute before/after probe target metrics
    """
    _validate_target(target)
    config = InterventionConfig() if config is None else config
    if token_batch.dim() != 2:
        raise ValueError(f"token_batch must be [B,T], got {tuple(token_batch.shape)}")
    if token_batch.size(0) != 1:
        raise ValueError("run_probe_intervention expects a single sequence batch with shape [1, T]")

    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)

    base_residuals = extract_residuals(model=model, token_batch=token_batch, layer=layer, device=device)
    pos_idx = _resolve_position_index(base_residuals.size(1), target.position)
    base_act = base_residuals[0, pos_idx, :]

    optimization = optimize_residual_delta(
        probe=probe,
        base_activations=base_act,
        target=target,
        config=InterventionConfig(
            steps=config.steps,
            lr=config.lr,
            reg_strength=config.reg_strength,
            clamp_norm=config.clamp_norm,
            device=device,
        ),
    )

    delta = optimization["delta"]
    if not torch.is_tensor(delta):
        raise RuntimeError("Optimization did not return a tensor delta")
    intervened = apply_residual_delta(
        model=model,
        token_batch=token_batch,
        layer=layer,
        delta=delta,
        position=target.position,
        device=device,
    )

    post_residuals = intervened["residuals"]
    post_act = post_residuals[0, pos_idx, :].unsqueeze(0)
    base_act_batch = base_act.unsqueeze(0)

    probe = probe.to(device)
    probe.eval()

    before_probs = _probe_target_probs(probe, base_act_batch.to(device), target)
    after_probs = _probe_target_probs(probe, post_act.to(device), target)

    before_logits = probe(base_act_batch.to(device))[0, target.cell_index, :]
    after_logits = probe(post_act.to(device))[0, target.cell_index, :]

    before_board_logits = probe(base_act_batch.to(device))[0]
    after_board_logits = probe(post_act.to(device))[0]
    before_board_probs = before_board_logits.softmax(dim=-1)
    after_board_probs = after_board_logits.softmax(dim=-1)

    before_pred = int(before_logits.argmax().item())
    after_pred = int(after_logits.argmax().item())

    return {
        "before_target_prob": float(before_probs.item()),
        "after_target_prob": float(after_probs.item()),
        "target_prob_gain": float(after_probs.item() - before_probs.item()),
        "before_pred_class": before_pred,
        "after_pred_class": after_pred,
        "target_class": target.target_class,
        "target_cell": target.cell_index,
        "layer": layer,
        "success": after_pred == target.target_class,
        "delta": delta.detach().cpu(),
        "delta_norm": float(delta.detach().norm(dim=-1).mean().item()),
        "position_index": pos_idx,
        "before_target_logit": float(before_logits[target.target_class].item()),
        "after_target_logit": float(after_logits[target.target_class].item()),
        "target_logit_gain": float((after_logits[target.target_class] - before_logits[target.target_class]).item()),
        "before_board_probs": before_board_probs.detach().cpu(),
        "after_board_probs": after_board_probs.detach().cpu(),
        "loss_trace": optimization["loss_trace"],
        "target_prob_trace": optimization["target_prob_trace"],
    }
