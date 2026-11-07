"""Sliced Epps–Pulley regularization (LeJEPA, arXiv:2511.08544v3, Algorithm 1)."""

import math

import torch
from torch import Tensor, distributed as dist, nn
from torch.distributed.nn.functional import all_reduce
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def _working_dtype(tensor: Tensor) -> torch.dtype:
    return torch.float64 if tensor.dtype == torch.float64 else torch.float32


def _characteristic_sums(samples: Tensor, knots: Tensor) -> Tensor:
    phases = samples.unsqueeze(-1) * knots
    return torch.stack((phases.cos().sum(dim=-3), phases.sin().sum(dim=-3)))


def _projected_sums(embeddings: Tensor, directions: Tensor, knots: Tensor) -> Tensor:
    # Non-reentrant checkpointing must replay the projection at the same precision.
    with torch.autocast(embeddings.device.type, enabled=False):
        return _characteristic_sums(embeddings @ directions, knots)


class EppsPulley(nn.Module):
    """Standard-normal test on samples with shape (..., batch, slices).

    Returns (..., slices). The grid covers the positive half of the domain
    by default. With symmetric_grid=True, num_points counts the full grid
    instead, as in the paper's pseudocode.
    """

    def __init__(self, num_points: int = 17, t_max: float = 3.0, *, symmetric_grid: bool = False):
        super().__init__()
        if num_points < 2 or not math.isfinite(t_max) or t_max <= 0:
            raise ValueError("Epps–Pulley requires at least two knots and finite t_max > 0")
        start = -t_max if symmetric_grid else 0.0
        grid = torch.linspace(start, t_max, num_points, dtype=torch.float64)
        window = torch.exp(-0.5 * grid.square())
        # Both grids integrate across [-t_max, t_max], exploiting an even integrand.
        weights = torch.full_like(grid, 2 * t_max / (num_points - 1))
        weights[0] *= 0.5
        weights[-1] *= 0.5
        self.register_buffer("grid", grid, persistent=False)
        self.register_buffer("normal_cf", window, persistent=False)
        self.register_buffer("weights", weights * window, persistent=False)

    @staticmethod
    def sample_count(samples: Tensor, distributed: bool) -> Tensor:
        count = torch.tensor(samples.size(-2), device=samples.device, dtype=torch.int64)
        if distributed:
            dist.all_reduce(count)
        return count

    def statistic(self, sums: Tensor, count: Tensor, distributed: bool) -> Tensor:
        if distributed:
            sums = all_reduce(sums)
        real, imag = sums / count
        error = (real - self.normal_cf.to(real)).square() + imag.square()
        return (error * self.weights.to(error)).sum(dim=-1) * count

    def forward(self, samples: Tensor, *, distributed: bool = False) -> Tensor:
        if samples.ndim < 2 or not samples.is_floating_point() or samples.size(-2) < 1:
            raise ValueError("Expected floating point samples shaped (..., batch, slices)")
        samples = samples.to(_working_dtype(samples))
        count = self.sample_count(samples, distributed)
        return self.statistic(_characteristic_sums(samples, self.grid.to(samples)), count, distributed)


class SIGReg(nn.Module):
    """Apply Epps–Pulley to random unit projections of (..., batch, features).

    With DDP, pass the same explicit step on each rank. The collective reduces
    empirical characteristic-function sums and supports unequal rank batches.
    The differentiable SUM reduction assumes DDP averages parameter gradients.
    Set chunk_size to cap temporary memory by recomputing each slice chunk
    during backpropagation.
    """

    def __init__(
        self,
        num_slices: int = 1024,
        num_points: int = 17,
        t_max: float = 5.0,
        *,
        seed: int = 0,
        chunk_size: int | None = 64,
        symmetric_grid: bool = False,
    ):
        super().__init__()
        if num_slices < 1 or (chunk_size is not None and chunk_size < 1):
            raise ValueError("num_slices and chunk_size must be positive")
        self.num_slices = num_slices
        self.seed = seed
        self.chunk_size = chunk_size
        self.test = EppsPulley(num_points, t_max, symmetric_grid=symmetric_grid)

    def directions(self, embeddings: Tensor, step: int | None = None) -> Tensor:
        generator = None
        if step is not None:
            generator = torch.Generator(device=embeddings.device)
            generator.manual_seed((self.seed + step) % (2**63))
        vectors = torch.randn(
            embeddings.size(-1), self.num_slices, generator=generator,
            device=embeddings.device, dtype=_working_dtype(embeddings),
        )
        return F.normalize(vectors, dim=0)

    def forward(
        self,
        embeddings: Tensor,
        *,
        step: int | None = None,
        directions: Tensor | None = None,
        distributed: bool | None = None,
    ) -> Tensor:
        if (embeddings.ndim < 2 or not embeddings.is_floating_point()
                or min(embeddings.shape) < 1):
            raise ValueError("Expected non-empty floating point embeddings (..., batch, features)")
        if distributed is None:
            distributed = dist.is_initialized() and dist.get_world_size() > 1
        if directions is None:
            if distributed and step is None:
                raise ValueError("Distributed SIGReg needs a shared step or shared directions")
            directions = self.directions(embeddings, step)
        if directions.ndim != 2 or directions.size(0) != embeddings.size(-1) or directions.size(1) < 1:
            raise ValueError("Expected directions shaped (features, slices)")

        embeddings = embeddings.to(_working_dtype(embeddings))
        directions = directions.to(embeddings)
        knots = self.test.grid.to(embeddings)
        count = self.test.sample_count(embeddings, distributed)
        chunks = directions.split(self.chunk_size or directions.size(1), dim=1)
        characteristic_sums = []
        for chunk in chunks:
            if self.chunk_size and torch.is_grad_enabled() and embeddings.requires_grad:
                sums = checkpoint(_projected_sums, embeddings, chunk, knots, use_reentrant=False)
            else:
                sums = _projected_sums(embeddings, chunk, knots)
            characteristic_sums.append(sums)
        sums = torch.cat(characteristic_sums, dim=-2)
        return self.test.statistic(sums, count, distributed).mean()
