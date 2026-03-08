"""
Lightweight probing heads for analysing transformer representations on boards.

The probes reshape hidden states into board-shaped tensors to predict cell
occupancy or move classes, enabling interpretability experiments.
"""

import torch
import torch.nn as nn


class BoardLinearProbe(nn.Module):
    """
    Linear probes for stacked residuals.

    Input:
        x: [B, n_layers, T, d_model]
    Output:
        logits: [B, n_layers, T, board_size, n_classes]
    """
    def __init__(
            self,
            d_model: int,
            board_size: int,
            n_classes: int,
    ):
        super().__init__()
        self.linear = nn.Linear(
            d_model,
            board_size * n_classes
        )
    # end def __init__

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x (Tensor): [B, n_layers, T, d_model]
        Returns:
            logits (Tensor): [B, n_layers, T, 64, 3]
        """
        B, L, T, D = x.shape
        layer_out = self.linear(x)          # [B, T, 192]
        layer_out = layer_out.view(B, T, 64, 3)       # [B, T, 64, 3]
        return layer_out                   # [B, L, T, 64, 3]
    # end def forward

# end class BoardLinearProbe



class BoardLinearProbe(nn.Module):
    """
    Linear probes for stacked residuals.

    Input:
        x: [B, n_layers, T, d_model]
    Output:
        logits: [B, n_layers, T, board_size, n_classes]
    """
    def __init__(
            self,
            d_model: int,
            board_size: int,
            n_classes: int,
    ):
        super().__init__()
        self.linear = nn.Linear(
            d_model,
            board_size * n_classes
        )
    # end def __init__

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x (Tensor): [B, T, d_model]
        Returns:
            logits (Tensor): [B, T, 64, 3]
        """
        B, T, D = x.shape
        layer_out = self.linear(x)          # [B, T, 192]
        layer_out = layer_out.view(B, T, 64, 3)       # [B, T, 64, 3]
        return layer_out                   # [B, T, 64, 3]
    # end def forward

# end class BoardLinearProbe



class BoardMLPProbe(nn.Module):
    """
    Non-linear probe (MLP) for stacked residuals.

    Input:
        x: [B, n_layers, T, d_model]
    Output:
        logits: [B, n_layers, T, board_size, n_classes]
    """

    def __init__(
        self,
        d_model: int,
        board_size: int,
        n_classes: int,
        hidden_dim: int = 512,
        n_hidden_layers: int = 2,
        activation: str = "gelu",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.board_size = board_size
        self.n_classes = n_classes

        # Choix de l'activation
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
            "silu": nn.SiLU(),
        }
        act = activations.get(activation, nn.GELU())

        # Construction dynamique du MLP
        layers = [nn.Linear(d_model, hidden_dim), act]
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(act)
        # end for
        layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, board_size * n_classes))

        self.mlp = nn.Sequential(*layers)
    # end def __init__

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): [B, n_layers, T, d_model]
        Returns:
            logits (Tensor): [B, n_layers, T, board_size, n_classes]
        """
        B, L, T, D = x.shape

        # Fusionne toutes les dimensions sauf D pour passage dans le MLP
        out = self.mlp(x)                              # [B, L, T, board_size*n_classes]
        out = out.view(B, L, T, self.board_size, self.n_classes)
        return out
    # end def forward

# end class BoardMLPProbe



