"""Optimization utilities shared by WorldLab experiments."""

import math

import torch


def warmup_cosine_lr(step, peak_lr, min_lr, warmup_steps, total_steps, *, start_factor=None):
    """Compute the learning rate at an optimizer step, with a clamped cosine tail."""
    if step < warmup_steps:
        if start_factor is None:
            return peak_lr * (step + 1) / (warmup_steps + 1)
        return peak_lr * (start_factor + (1 - start_factor) * step / warmup_steps)
    duration = total_steps - warmup_steps
    if duration <= 0:
        return peak_lr if step == warmup_steps else min_lr
    progress = min(1.0, (step - warmup_steps) / duration)
    return min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (peak_lr - min_lr)


def optimizer_step(optimizer, *, scaler=None, parameters=None, grad_clip=0.0):
    """Finish a step after one or more backward passes, then clear gradients."""
    if grad_clip:
        if scaler is not None:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(parameters, grad_clip)
    if scaler is None:
        optimizer.step()
    else:
        scaler.step(optimizer)
        scaler.update()
    optimizer.zero_grad(set_to_none=True)


def configure_optimizers(module, weight_decay, learning_rate, betas, device_type):
    """Build AdamW with decay on matrices and embeddings, preserving parameter order."""
    decay, no_decay = [], []
    for parameter in module.parameters():
        if parameter.requires_grad:
            (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(
        groups, lr=learning_rate, betas=betas, fused=True if device_type == "cuda" else None,
    )
    return optimizer, sum(p.numel() for p in decay), sum(p.numel() for p in no_decay)
