"""
Probe models and training utilities for board-state decoding.
"""

from typing import Dict, Iterable, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .spec import OTHELLO_NUM_BOARD_CELLS


class LinearBoardProbe(nn.Module):
    """
    Linear probe from activation vector to 64x3 board logits.
    """

    def __init__(self, input_dim: int, num_cells: int = OTHELLO_NUM_BOARD_CELLS, num_classes: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.num_cells = num_cells
        self.num_classes = num_classes
        self.proj = nn.Linear(input_dim, num_cells * num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.proj(x)
        return logits.view(*x.shape[:-1], self.num_cells, self.num_classes)


class MLPBoardProbe(nn.Module):
    """
    Two-layer MLP probe from activation vector to 64x3 board logits.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_cells: int = OTHELLO_NUM_BOARD_CELLS,
        num_classes: int = 3,
        activation: nn.Module = nn.GELU(),
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_cells = num_cells
        self.num_classes = num_classes
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_cells * num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        return logits.view(*x.shape[:-1], self.num_cells, self.num_classes)


def _unpack_batch(
    batch: Union[torch.Tensor, Sequence[torch.Tensor]]
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if torch.is_tensor(batch):
        return batch, None
    if isinstance(batch, (list, tuple)) and len(batch) >= 2 and torch.is_tensor(batch[0]) and torch.is_tensor(batch[1]):
        return batch[0], batch[1]
    raise ValueError("Expected batch as tensor or (features, labels)")


def _compute_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if logits.dim() != 3:
        raise ValueError(f"Expected logits shape [B, cells, classes], got {tuple(logits.shape)}")
    if targets.dim() != 2:
        raise ValueError(f"Expected targets shape [B, cells], got {tuple(targets.shape)}")
    if logits.size(1) != targets.size(1):
        raise ValueError(
            f"Mismatched board cells count between logits ({logits.size(1)}) and targets ({targets.size(1)})"
        )
    return nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1).long())


@torch.no_grad()
def _compute_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> Tuple[float, torch.Tensor]:
    preds = logits.argmax(dim=-1)
    correct = preds.eq(targets)
    acc_global = correct.float().mean().item()
    acc_per_cell = correct.float().mean(dim=0)
    return acc_global, acc_per_cell


def fit_probe(
    probe: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 10,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, list]:
    """
    Train a probe on activations->board labels.

    Returns a small history dict with train/val losses and validation accuracy.
    """
    if device is None:
        device = next(probe.parameters()).device
    device = torch.device(device)
    probe.to(device)

    optimizer = torch.optim.AdamW(probe.parameters(), lr=learning_rate, weight_decay=weight_decay)
    history: Dict[str, list] = {"train_loss": [], "val_loss": [], "val_acc": []}

    for _ in range(epochs):
        probe.train()
        loss_sum = 0.0
        count = 0
        for batch in train_loader:
            features, targets = _unpack_batch(batch)
            if targets is None:
                raise ValueError("fit_probe requires labels in train_loader")
            features = features.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = probe(features)
            loss = _compute_loss(logits, targets)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            count += 1

        history["train_loss"].append(loss_sum / max(count, 1))

        if val_loader is not None:
            eval_stats = eval_probe(probe=probe, data_loader=val_loader, device=device)
            history["val_loss"].append(eval_stats["loss"])
            history["val_acc"].append(eval_stats["acc_global"])

    return history


@torch.no_grad()
def eval_probe(
    probe: nn.Module,
    data_loader: DataLoader,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Union[float, torch.Tensor]]:
    """
    Evaluate a probe and return loss + accuracy metrics.
    """
    if device is None:
        device = next(probe.parameters()).device
    device = torch.device(device)
    probe.to(device)
    probe.eval()

    loss_sum = 0.0
    acc_sum = 0.0
    count = 0
    acc_per_cell_sum = None

    for batch in data_loader:
        features, targets = _unpack_batch(batch)
        if targets is None:
            raise ValueError("eval_probe requires labels in data_loader")
        features = features.to(device)
        targets = targets.to(device)

        logits = probe(features)
        loss = _compute_loss(logits, targets)
        acc_global, acc_per_cell = _compute_accuracy(logits, targets)

        loss_sum += loss.item()
        acc_sum += acc_global
        acc_per_cell_sum = acc_per_cell if acc_per_cell_sum is None else acc_per_cell_sum + acc_per_cell
        count += 1

    if count == 0:
        raise ValueError("eval_probe received an empty data_loader")

    return {
        "loss": loss_sum / count,
        "acc_global": acc_sum / count,
        "acc_per_cell": (acc_per_cell_sum / count).detach().cpu(),
    }


@torch.no_grad()
def predict_board(
    probe: nn.Module,
    activations: torch.Tensor,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """
    Predict board labels from activations.

    Returns:
        Tensor of predicted classes with shape [..., 64].
    """
    if device is None:
        device = next(probe.parameters()).device
    device = torch.device(device)

    probe.eval()
    logits = probe(activations.to(device))
    return logits.argmax(dim=-1).detach().cpu()
