"""Modality-independent implementation of LeJEPA (arXiv:2511.08544v3, Algorithm 2)."""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import groupby

import torch
from torch import Tensor, nn

from .sigreg import SIGReg, _working_dtype


class ProjectionMLP(nn.Sequential):
    """Linear projection with optional Linear/BatchNorm/ReLU hidden layers.

    The network maps an input of ``input_dim`` features to ``output_dim``
    features through a configurable MLP. Every hidden layer is a
    Linear→BatchNorm1d→ReLU stack, and the output layer is a bare Linear.

    Examples:
        >>> mlp = ProjectionMLP(64, 16, hidden_dims=(128, 128))
        >>> mlp(torch.randn(4, 64)).shape
        torch.Size([4, 16])
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: Sequence[int] = (2048, 2048)):
        """Build the projection MLP.

        Args:
            input_dim (int): Size of the input feature dimension.
            output_dim (int): Size of the output projection dimension.
            hidden_dims (Sequence[int], optional): Widths of the hidden
                Linear/BatchNorm/ReLU layers, in order. An empty sequence
                yields a single Linear from ``input_dim`` to ``output_dim``.
                Defaults to (2048, 2048).
        """
        layers = []
        for hidden_dim in hidden_dims:
            layers.extend((nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU()))
            input_dim = hidden_dim
        # end for
        layers.append(nn.Linear(input_dim, output_dim))
        super().__init__(*layers)
    # end def __init__
# end class ProjectionMLP


@dataclass
class LeJEPAOutput:
    """Container for the LeJEPA objective outputs.

    Attributes:
        loss (Tensor): Scalar total loss, ``prediction_scale * prediction
            + weight * sigreg``.
        prediction (Tensor): Scalar mean squared error of the embeddings
            prediction term.
        sigreg (Tensor): Scalar SIGReg regularizer value.
    """

    loss: Tensor
    prediction: Tensor
    sigreg: Tensor
# end class LeJEPAOutput


class LeJEPAObjective(nn.Module):
    """Predict the mean global embedding from every view of the same sample.

    Inputs have shape (views, batch, features), with global views first.
    The paper's Algorithm 2 averages over features as well as samples and
    views. The target remains in the gradient graph by default. The PRH
    recipe pools views for SIGReg and adds its weighted value to prediction.

    Attributes:
        weight (float): SIGReg mixture weight in [0, 1].
        pool_views (bool): If True, flatten views and batch before SIGReg.
        additive (bool): If True, combine losses additively instead of the
            ``(1 - weight) * prediction`` scaling.
        sigreg (SIGReg): The underlying SIGReg regularizer module.
    """

    def __init__(
            self,
            weight: float = 0.05,
            sigreg: SIGReg | None = None, *,
            pool_views: bool = False,
            additive: bool = False,
    ):
        """Initialize the loss weights and the SIGReg module.

        Args:
            weight (float, optional): SIGReg mixture weight, must lie in
                [0, 1]. Defaults to 0.05.
            sigreg (SIGReg | None, optional): Preconfigured SIGReg module;
                a default ``SIGReg()`` is created when absent. Defaults to
                None.
            pool_views (bool, optional): Flatten (views, batch) into a
                single sample axis before SIGReg. Defaults to False.
            additive (bool, optional): Use ``prediction + weight * sigreg``
                instead of ``(1 - weight) * prediction``. Defaults to False.

        Raises:
            ValueError: If ``weight`` is outside [0, 1].
        """
        super().__init__()
        if not 0 <= weight <= 1:
            raise ValueError("SIGReg weight must lie in [0, 1]")
        # end if
        self.weight = weight
        self.pool_views = pool_views
        self.additive = additive
        self.sigreg = sigreg if sigreg is not None else SIGReg()
    # end def __init__

    def forward(
            self,
            projections: Tensor,
            num_global: int,
            *,
            step: int | None = None,
            predictions: Tensor | None = None,
            target: Tensor | None = None,
    ) -> LeJEPAOutput:
        """Compute the joint prediction and SIGReg loss.

        Args:
            projections (Tensor): Embeddings shaped (views, batch,
                features), with global views first.
            num_global (int): Number of leading views to average into the
                prediction target, between 1 and ``projections.size(0)``.
            step (int | None, optional): Optimization step forwarded to
                SIGReg for deterministic direction sampling in distributed
                mode. Defaults to None.
            predictions (Tensor | None, optional): Optional predictor
                output shaped like ``projections``; ``projections`` itself
                is used when absent. Defaults to None.
            target (Tensor | None, optional): Explicit prediction target
                shaped (batch, features); when absent, the mean of the
                first ``num_global`` views is used. Defaults to None.

        Returns:
            LeJEPAOutput: Scalar ``loss``, ``prediction`` and ``sigreg``
            values.

        Raises:
            ValueError: If the view count is below 2, ``num_global`` is out
                of range, or the shapes of ``predictions`` or ``target``
                do not match ``projections``.
        """
        if projections.ndim != 3 or projections.size(0) < 2:
            raise ValueError("Expected at least two views shaped (views, batch, features)")
        # end if

        if not 1 <= num_global <= projections.size(0):
            raise ValueError("num_global must lie between 1 and the number of views")
        # end if

        # projections, shape (views, batch, features)
        projections = projections.to(_working_dtype(projections))
        if predictions is None:
            predictions = projections
        elif predictions.shape != projections.shape:
            raise ValueError("Predictions must match projections")
        # end if

        if target is None:
            # Mean of the global views, shape (batch, features)
            target = projections[:num_global].mean(dim=0)
        elif target.shape != projections.shape[1:]:
            raise ValueError("Target must have shape (batch, features)")
        # end if

        # Scalar squared error against the target, shape ()
        prediction = (predictions.to(projections) - target.to(projections)).square().mean()

        if self.pool_views:
            # Pooled samples, shape (views * batch, features)
            samples = projections.flatten(0, 1)
        else:
            # Per-view samples, shape (views, batch, features)
            samples = projections
        # end if

        # Scalar SIGReg value, shape ()
        regularization = self.sigreg(samples, step=step)

        # Mixture coefficient: additive or convex complement, scalar
        prediction_scale = 1.0 if self.additive else 1 - self.weight

        # Scalar total loss, shape ()
        loss = prediction_scale * prediction + self.weight * regularization

        return LeJEPAOutput(loss, prediction, regularization)
    # end def forward

# end class LeJEPAObjective


class LeJEPA(nn.Module):
    """Wrap an encoder returning (batch, features) and a shared projector.

    Pass a (views, batch, ...) tensor or a sequence of (batch, ...) tensors.
    Views of the same consecutive shape are encoded together, so global and
    local crops can have different resolutions without padding.

    Attributes:
        encoder (nn.Module): Backbone mapping (batch, ...) to (batch, F).
        projector (nn.Module): Head mapping (batch, F) to (batch, D).
        objective (LeJEPAObjective): Loss module computing the LeJEPA
            objective from projections.
    """

    def __init__(
            self,
            encoder: nn.Module,
            projector: nn.Module,
            objective: LeJEPAObjective | None = None,
    ):
        """Assemble the encoder, projector and objective.

        Args:
            encoder (nn.Module): Module mapping a (batch, ...) tensor to
                features shaped (batch, F).
            projector (nn.Module): Module mapping features to projections
                shaped (batch, D).
            objective (LeJEPAObjective | None, optional): Loss module; a
                default ``LeJEPAObjective()`` is created when absent.
                Defaults to None.
        """
        super().__init__()
        self.encoder = encoder
        self.projector = projector
        self.objective = objective if objective is not None else LeJEPAObjective()
    # end def __init__

    def encode(self, samples: Tensor) -> Tensor:
        """Encode a batch of samples with the backbone.

        Args:
            samples (Tensor): Sample batch shaped (batch, ...), where
                ``...`` is the encoder input layout.

        Returns:
            Tensor: Encoder features shaped (batch, F), where F is the
            encoder's output feature size.
        """
        return self.encoder(samples)
        # end return
    # end def encode

    def _encoded_groups(
            self, views: Tensor | Sequence[Tensor],
    ) -> Iterator[tuple[Tensor, int, int]]:
        """Encode consecutive same-shape groups of views.

        Args:
            views (Tensor | Sequence[Tensor]): Either a (views, batch, ...)
            tensor, viewed as ``views`` tensors of shape (batch, ...), or a
            sequence of (batch, ...) tensors of possibly mixed shapes.

        Yields:
            tuple[Tensor, int, int]: For each consecutive group of equal
            input shapes: the encoded and concatenated features shaped
            (group_views * batch, F), the number of views in the group and
            the shared batch size.

        Raises:
            ValueError: If the view list is empty or views disagree on the
                batch size.
        """
        if not len(views) or any(view.size(0) != views[0].size(0) for view in views):
            raise ValueError("Views must share the same batch size")
        # end if
        for _, group in groupby(views, key=lambda view: view.shape):
            group = list(group)
            # (group_views, batch, ...) -> (group_views * batch, ...)
            encoded = self.encode(torch.cat(group))
            # Batch and view bookkeeping, scalars
            yield encoded, len(group), views[0].size(0)
        # end for
    # end def _encoded_groups

    def encode_views(
            self,
            views: Tensor | Sequence[Tensor]
    ) -> tuple[Tensor, Tensor]:
        """Encode all views and project them, stacking views per sample.

        Args:
            views (Tensor | Sequence[Tensor]): Input views, as accepted by
                :meth:`_encoded_groups`; all views must share a batch size.

        Returns:
            tuple[Tensor, Tensor]: The backbone features and the projected
            embeddings, both shaped (views, batch, F) and
            (views, batch, D) respectively.
        """
        features, projections = [], []
        for encoded, count, batch in self._encoded_groups(views):
            # (group_views * batch, F) @ projector -> (group_views * batch, D)
            projected = self.projector(encoded)
            # Regroup the views axis: (group_views, batch, F) and (group_views, batch, D)
            features.append(encoded.reshape(count, batch, -1))
            projections.append(projected.reshape(count, batch, -1))
        # end for
        # (views, batch, F) and (views, batch, D)
        return torch.cat(features), torch.cat(projections)
        # end return
    # end def encode_views

    def forward(
        self, views: Tensor | Sequence[Tensor], num_global: int, *, step: int | None = None,
    ) -> LeJEPAOutput:
        """Encode the views, project and compute the LeJEPA loss.

        Args:
            views (Tensor | Sequence[Tensor]): Input views, as accepted by
                :meth:`encode_views`, shaped (views, batch, ...) or a
                sequence of (batch, ...) tensors.
            num_global (int): Number of leading global views averaged into
                the prediction target.
            step (int | None, optional): Optimization step forwarded to
                SIGReg. Defaults to None.

        Returns:
            LeJEPAOutput: Scalar ``loss``, ``prediction`` and ``sigreg``
            values.
        """
        _, projections = self.encode_views(views)
        # Projections shaped (views, batch, D) -> scalar loss components
        return self.objective(projections, num_global, step=step)
    # end def forward
# end class LeJEPA
