"""Modality-independent implementation of LeJEPA (arXiv:2511.08544v3, Algorithm 2)."""

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import groupby

import torch
from torch import Tensor, nn

from .sigreg import SIGReg, _working_dtype


class ProjectionMLP(nn.Sequential):
    """Linear projection with optional Linear/BatchNorm/ReLU hidden layers."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: Sequence[int] = (2048, 2048)):
        layers = []
        for hidden_dim in hidden_dims:
            layers.extend((nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU()))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, output_dim))
        super().__init__(*layers)


@dataclass
class LeJEPAOutput:
    loss: Tensor
    prediction: Tensor
    sigreg: Tensor


class LeJEPAObjective(nn.Module):
    """Predict the mean global embedding from every view of the same sample.

    Inputs have shape (views, batch, features), with global views first.
    The paper's Algorithm 2 averages over features as well as samples and
    views. The target remains in the gradient graph by default.
    """

    def __init__(self, weight: float = 0.05, sigreg: SIGReg | None = None):
        super().__init__()
        if not 0 <= weight <= 1:
            raise ValueError("SIGReg weight must lie in [0, 1]")
        self.weight = weight
        self.sigreg = sigreg if sigreg is not None else SIGReg()

    def forward(
        self,
        projections: Tensor,
        num_global: int,
        *,
        step: int | None = None,
        predictions: Tensor | None = None,
        target: Tensor | None = None,
    ) -> LeJEPAOutput:
        if projections.ndim != 3 or projections.size(0) < 2:
            raise ValueError("Expected at least two views shaped (views, batch, features)")
        if not 1 <= num_global <= projections.size(0):
            raise ValueError("num_global must lie between 1 and the number of views")
        projections = projections.to(_working_dtype(projections))
        if predictions is None:
            predictions = projections
        elif predictions.shape != projections.shape:
            raise ValueError("Predictions must match projections")
        if target is None:
            target = projections[:num_global].mean(dim=0)
        elif target.shape != projections.shape[1:]:
            raise ValueError("Target must have shape (batch, features)")
        prediction = (predictions.to(projections) - target.to(projections)).square().mean()
        regularization = self.sigreg(projections, step=step)
        loss = (1 - self.weight) * prediction + self.weight * regularization
        return LeJEPAOutput(loss, prediction, regularization)


class LeJEPA(nn.Module):
    """Wrap an encoder returning (batch, features) and a shared projector.

    Pass a (views, batch, ...) tensor or a sequence of (batch, ...) tensors.
    Views of the same consecutive shape are encoded together, so global and
    local crops can have different resolutions without padding.
    """

    def __init__(
        self, encoder: nn.Module, projector: nn.Module, objective: LeJEPAObjective | None = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.projector = projector
        self.objective = objective if objective is not None else LeJEPAObjective()

    def encode(self, samples: Tensor) -> Tensor:
        return self.encoder(samples)

    def encode_views(self, views: Tensor | Sequence[Tensor]) -> tuple[Tensor, Tensor]:
        if not len(views) or any(view.size(0) != views[0].size(0) for view in views):
            raise ValueError("Views must share the same batch size")
        batch = views[0].size(0)
        features, projections = [], []
        for _, group in groupby(views, key=lambda view: view.shape):
            group = list(group)
            encoded = self.encode(torch.cat(group))
            projected = self.projector(encoded)
            features.append(encoded.reshape(len(group), batch, -1))
            projections.append(projected.reshape(len(group), batch, -1))
        return torch.cat(features), torch.cat(projections)

    def forward(
        self, views: Tensor | Sequence[Tensor], num_global: int, *, step: int | None = None,
    ) -> LeJEPAOutput:
        _, projections = self.encode_views(views)
        return self.objective(projections, num_global, step=step)
