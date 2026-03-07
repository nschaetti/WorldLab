"""
Minimal GPT training helpers for synthetic sequence worlds (Phase 3).

The goal is to provide a lightweight, educational training loop that can be used
inside notebooks to study how representation geometry evolves under controlled
co-occurrence statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch

from boardGPT.nn import GPT, GPTConfig


@dataclass(frozen=True)
class SyntheticGPTTrainConfig:
    """
    Training configuration for synthetic next-token modeling.

    Attributes:
        batch_size: Mini-batch size.
        max_steps: Number of optimizer steps.
        learning_rate: AdamW learning rate.
        weight_decay: AdamW weight decay.
        grad_clip: Optional gradient clipping threshold.
        seed: Random seed.
        device: Compute device.
        log_interval: Print interval in steps.
    """

    batch_size: int = 64
    max_steps: int = 200
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: Optional[float] = 1.0
    seed: int = 0
    device: str = "cpu"
    log_interval: int = 50


def build_tiny_gpt_for_synthetic(
    vocab_size: int,
    block_size: int,
    n_layer: int = 4,
    n_head: int = 4,
    n_embd: int = 128,
    dropout: float = 0.0,
    bias: bool = True,
    use_flash: bool = False,
) -> GPT:
    """
    Build a compact GPT suitable for synthetic experiments.
    """
    cfg = GPTConfig(
        block_size=block_size,
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout,
        bias=bias,
    )
    return GPT(config=cfg, use_flash=use_flash)


def _sample_batch(
    sequences: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    n, t = sequences.shape
    if t < 2:
        raise ValueError(f"sequences length must be >= 2, got {t}")

    idx = rng.integers(0, n, size=batch_size)
    x = sequences[idx, :-1]
    y = sequences[idx, 1:]
    return (
        torch.from_numpy(x).long().to(device),
        torch.from_numpy(y).long().to(device),
    )


def train_gpt_on_synthetic_sequences(
    model: GPT,
    sequences: Union[np.ndarray, torch.Tensor],
    config: SyntheticGPTTrainConfig,
) -> list[float]:
    """
    Train GPT on synthetic sequences with next-token loss.

    Args:
        model: GPT model.
        sequences: Integer array ``[N, T]``.
        config: Training hyperparameters.

    Returns:
        List of per-step training losses.
    """
    if config.batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {config.batch_size}")
    if config.max_steps <= 0:
        raise ValueError(f"max_steps must be > 0, got {config.max_steps}")

    if torch.is_tensor(sequences):
        seq_np = sequences.detach().cpu().numpy().astype(np.int64)
    else:
        seq_np = np.asarray(sequences, dtype=np.int64)

    if seq_np.ndim != 2:
        raise ValueError(f"sequences must be [N, T], got shape {seq_np.shape}")
    # We train with x = seq[:, :-1], so only T-1 tokens are forwarded.
    if (seq_np.shape[1] - 1) > model.config.block_size:
        raise ValueError(
            f"input length {seq_np.shape[1] - 1} exceeds model block_size {model.config.block_size}"
        )

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    rng = np.random.default_rng(config.seed)

    device = torch.device(config.device)
    model = model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    losses: list[float] = []
    for step in range(1, config.max_steps + 1):
        x, y = _sample_batch(seq_np, batch_size=config.batch_size, rng=rng, device=device)

        _, logits, _ = model(idx=x, to_return=None)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            y.reshape(-1),
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if config.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip)
        optimizer.step()

        loss_value = float(loss.detach().cpu().item())
        losses.append(loss_value)

        if config.log_interval > 0 and (step % config.log_interval == 0 or step == 1):
            print(f"[synthetic-train] step={step:04d} loss={loss_value:.6f}")

    return losses
