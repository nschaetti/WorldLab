"""Neural probes that decode an Othello board from hidden activations."""

import torch
from torch import nn


class LinearBoardProbe(nn.Module):
    """
    Linear probe from activation vector to 64x3 board logits.
    """

    def __init__(self, input_dim: int, num_cells: int = 64, num_classes: int = 3):
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
        num_cells: int = 64,
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
