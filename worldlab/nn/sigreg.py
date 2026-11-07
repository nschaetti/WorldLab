"""Sliced Epps–Pulley regularization (LeJEPA, arXiv:2511.08544v3, Algorithm 1)."""

import math

import torch
from torch import Tensor, distributed as dist, nn
from torch.distributed.nn.functional import all_reduce
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def _working_dtype(tensor: Tensor) -> torch.dtype:
    """Return the working compute dtype for a tensor.

    SIGReg computations run in float32 by default. Only float64 inputs
    keep double precision throughout so that higher accuracy is possible
    when requested explicitly.

    Args:
        tensor (Tensor): Input tensor of any floating point or other dtype.

    Returns:
        torch.dtype: ``torch.float64`` for float64 inputs, ``torch.float32``
        otherwise.

    Examples:
        >>> _working_dtype(torch.empty(2, dtype=torch.float64))
        torch.float64
    """
    return torch.float64 if tensor.dtype == torch.float64 else torch.float32
    # end return
# end def _working_dtype


def _characteristic_sums(samples: Tensor, knots: Tensor) -> Tensor:
    """Compute real and imaginary empirical characteristic function sums.

    The characteristic function of the samples is evaluated at each knot of
    the integration grid and the sums over the batch dimension are returned.

    Args:
        samples (Tensor): Sample values shaped (..., batch, slices).
        knots (Tensor): Integration grid shaped (num_points,).

    Returns:
        Tensor: Stacked real and imaginary sums shaped
        (2, ..., batch, num_points), with the real part at index 0 and the
        imaginary part at index 1.
    """
    # (..., batch, slices) * (num_points,) -> (..., batch, slices, num_points)
    phases = samples.unsqueeze(-1) * knots
    # Sum over slices (dim=-3 after the broadcast expansion), -> (2, ..., batch, num_points)
    return torch.stack((phases.cos().sum(dim=-3), phases.sin().sum(dim=-3)))
    # end return
# end def _characteristic_sums


def _projected_sums(embeddings: Tensor, directions: Tensor, knots: Tensor) -> Tensor:
    """Project embeddings onto random directions, then compute characteristic sums.

    Projection happens outside autocast so that non-reentrant checkpointing
    replays the projection and the matrix product at the same precision.

    Args:
        embeddings (Tensor): Embeddings shaped (..., batch, features).
        directions (Tensor): Unit projection vectors shaped (features, slices).
        knots (Tensor): Integration grid shaped (num_points,).

    Returns:
        Tensor: Stacked real and imaginary characteristic function sums
        shaped (2, ..., batch, num_points).
    """
    # Non-reentrant checkpointing must replay the projection at the same precision.
    with torch.autocast(embeddings.device.type, enabled=False):
        # (..., batch, features) @ (features, slices) -> (..., batch, slices)
        return _characteristic_sums(embeddings @ directions, knots)
    # end with
# end def _projected_sums


class EppsPulley(nn.Module):
    """Standard-normal goodness-of-fit test on uniform random projections.

    Given samples shaped (..., batch, slices), evaluates the empirical
    characteristic function of each sample against the standard normal
    characteristic function and integrates the squared error weighted by
    a Gaussian window over the knot grid. The result is the Epps–Pulley
    statistic computed per sample.

    Attributes:
        grid (Tensor): Knot grid of shape (num_points,), registered as a
            non-persistent buffer.
        normal_cf (Tensor): Standard normal characteristic function values
            on the grid, shape (num_points,), non-persistent buffer.
        weights (Tensor): Integration weights including the Gaussian
            window, shape (num_points,), non-persistent buffer.

    The grid covers the positive half of the domain by default. With
    ``symmetric_grid=True``, ``num_points`` counts the full grid instead,
    as in the paper's pseudocode.
    """

    def __init__(self, num_points: int = 17, t_max: float = 3.0, *, symmetric_grid: bool = False):
        """Build the knot grid, Gaussian window and trapezoidal weights.

        Args:
            num_points (int, optional): Number of integration knots.
                Must be at least 2. Defaults to 17.
            t_max (float, optional): Positive integration bound; the grid
                spans ``[-t_max, t_max]``, exploiting an even integrand.
                Defaults to 3.0.
            symmetric_grid (bool, optional): If True, spread ``num_points``
                knots over the full ``[-t_max, t_max]`` range instead of the
                positive half. Defaults to False.

        Raises:
            ValueError: If ``num_points < 2`` or ``t_max`` is not finite
                and positive.
        """
        super().__init__()
        if num_points < 2 or not math.isfinite(t_max) or t_max <= 0:
            raise ValueError("Epps–Pulley requires at least two knots and finite t_max > 0")
        # end if
        # Grid on the half domain [0, t_max] (or the full domain when symmetric).
        start = -t_max if symmetric_grid else 0.0
        # (num_points,)
        grid = torch.linspace(start, t_max, num_points, dtype=torch.float64)
        # Standard normal characteristic function exp(-t^2 / 2), shape (num_points,)
        window = torch.exp(-0.5 * grid.square())
        # Both grids integrate across [-t_max, t_max], exploiting an even integrand.
        # (num_points,)
        weights = torch.full_like(grid, 2 * t_max / (num_points - 1))
        weights[0] *= 0.5
        weights[-1] *= 0.5
        self.register_buffer("grid", grid, persistent=False)
        self.register_buffer("normal_cf", window, persistent=False)
        self.register_buffer("weights", weights * window, persistent=False)
    # end def __init__

    @staticmethod
    def sample_count(samples: Tensor, distributed: bool) -> Tensor:
        """Count the per-slice sample size, summed across ranks if distributed.

        Args:
            samples (Tensor): Sample values whose next-to-last dimension is
                the batch, shaped (..., batch, slices).
            distributed (bool): If True, all-reduce the local count so that
                every rank sees the global batch size.

        Returns:
            Tensor: Scalar int64 tensor holding the (global) batch size.
        """
        # Scalar sample count for this rank, shape ()
        count = torch.tensor(samples.size(-2), device=samples.device, dtype=torch.int64)
        if distributed:
            dist.all_reduce(count)
        # end if
        return count
        # end return
    # end def sample_count

    def statistic(self, sums: Tensor, count: Tensor, distributed: bool) -> Tensor:
        """Compute the weighted Epps–Pulley statistic.

        Args:
            sums (Tensor): Stacked real and imaginary characteristic sums
                shaped (2, ..., batch, num_points), optionally reduced
                across ranks when ``distributed`` is True.
            count (Tensor): Scalar (global) batch size, as returned by
                :meth:`sample_count`.
            distributed (bool): If True, all-reduce the sums across ranks
                with a differentiable SUM reduction before normalizing.

        Returns:
            Tensor: Per-sample Epps–Pulley statistic shaped (..., batch),
            scaled by the (global) sample count.
        """
        if distributed:
            sums = all_reduce(sums)
        # end if
        # Normalize by the (global) batch size: real and imag each (..., batch, num_points)
        real, imag = sums / count
        # Squared complex error against the normal cf, shape (..., batch, num_points)
        error = (real - self.normal_cf.to(real)).square() + imag.square()
        # Integrate over knots then scale by the sample count, -> (..., batch)
        return (error * self.weights.to(error)).sum(dim=-1) * count
        # end return
    # end def statistic

    def forward(self, samples: Tensor, *, distributed: bool = False) -> Tensor:
        """Evaluate the Epps–Pulley statistic on raw samples.

        Args:
            samples (Tensor): Floating point samples shaped
                (..., batch, slices).
            distributed (bool, optional): If True, count and reduce across
                ranks. Defaults to False.

        Returns:
            Tensor: Per-sample statistic shaped (..., batch).

        Raises:
            ValueError: If ``samples`` is not floating point, has fewer
                than 2 dimensions or an empty batch dimension.
        """
        if samples.ndim < 2 or not samples.is_floating_point() or samples.size(-2) < 1:
            raise ValueError("Expected floating point samples shaped (..., batch, slices)")
        # end if
        # Promote to the working dtype, shape (..., batch, slices)
        samples = samples.to(_working_dtype(samples))
        # Scalar (global) batch size, shape ()
        count = self.sample_count(samples, distributed)
        # Characteristic sums, shape (2, ..., batch, num_points)
        sums = _characteristic_sums(samples, self.grid.to(samples))
        # Per-sample statistic, shape (..., batch)
        return self.statistic(sums, count, distributed)
        # end return
    # end def forward
# end class EppsPulley


class SIGReg(nn.Module):
    """Apply Epps–Pulley to random unit projections of (..., batch, features).

    The statistic is the mean Epps–Pulley value over ``num_slices`` random
    unit directions, giving a scalar dependence-like regularizer that
    vanishes for isotropic Gaussian embeddings.

    With DDP, pass the same explicit ``step`` on each rank. The collective
    reduces empirical characteristic-function sums and supports unequal rank
    batches. The differentiable SUM reduction assumes DDP averages parameter
    gradients.

    Set ``chunk_size`` to cap temporary memory by recomputing each slice
    chunk during backpropagation.

    Attributes:
        num_slices (int): Number of random projection directions.
        seed (int): Base seed for the direction sampler; the step offsets
            the base seed when ``step`` is given.
        chunk_size (int | None): Slices per checkpointed chunk, or None
            to project all slices at once.
        test (EppsPulley): The underlying standard-normal test module.
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
        """Initialize the projection count, grid test and chunking policy.

        Args:
            num_slices (int, optional): Number of random unit projection
                directions. Must be at least 1. Defaults to 1024.
            num_points (int, optional): Number of Epps–Pulley integration
                knots. Defaults to 17.
            t_max (float, optional): Integration bound of the knot grid.
                Defaults to 5.0.
            seed (int, optional): Base seed for the direction sampler.
                In distributed runs, pass ``step`` to ``forward`` so every
                rank samples identical directions. Defaults to 0.
            chunk_size (int | None, optional): May be None to disable
                chunking; must be positive when given. Defaults to 64.
            symmetric_grid (bool, optional): Passed through to
                :class:`EppsPulley`. Defaults to False.

        Raises:
            ValueError: If ``num_slices < 1`` or ``chunk_size < 1``.
        """
        super().__init__()
        if num_slices < 1 or (chunk_size is not None and chunk_size < 1):
            raise ValueError("num_slices and chunk_size must be positive")
        # end if

        self.num_slices = num_slices
        self.seed = seed
        self.chunk_size = chunk_size
        self.test = EppsPulley(num_points, t_max, symmetric_grid=symmetric_grid)
    # end def __init__

    def directions(self, embeddings: Tensor, step: int | None = None) -> Tensor:
        """Sample deterministic unit projection directions.

        Args:
            embeddings (Tensor): Tensor whose trailing dimension defines
                the direction space, shaped (..., batch, features).
            step (int | None, optional): Optimization step; when given, the
                sampling seed is ``self.seed + step`` so directions rotate
                across steps while staying deterministic and rank-shared.
                Defaults to None.

        Returns:
            Tensor: Unit vectors sampled from a standard normal and L2
            normalized, shaped (features, num_slices).
        """
        generator = None
        if step is not None:
            generator = torch.Generator(device=embeddings.device)
            generator.manual_seed((self.seed + step) % (2**63))
        # end if
        # (features, num_slices)
        vectors = torch.randn(
            embeddings.size(-1), self.num_slices, generator=generator,
            device=embeddings.device, dtype=_working_dtype(embeddings),
        )
        # L2-normalized columns, shape (features, num_slices)
        return F.normalize(vectors, dim=0)
        # end return
    # end def directions

    def forward(
        self,
        embeddings: Tensor,
        *,
        step: int | None = None,
        directions: Tensor | None = None,
        distributed: bool | None = None,
    ) -> Tensor:
        """Compute the mean sliced Epps–Pulley statistic of embeddings.

        Random unit directions project embeddings down to one dimension per
        slice, then the Epps–Pulley statistic tests each projection against
        a standard normal. Slices are processed in chunks of ``chunk_size``
        with gradient checkpointing whenever gradients are enabled.

        Args:
            embeddings (Tensor): Floating point embeddings shaped
                (..., batch, features), with at least a batch rank.
            step (int | None, optional): Shared optimization step required
                in distributed mode when ``directions`` is not provided.
                Defaults to None.
            directions (Tensor | None, optional): Precomputed projection
                vectors shaped (features, slices); when given, ``step`` is
                not required for determinism. Defaults to None.
            distributed (bool | None, optional): Force the distributed
                reduction mode, defaults to True when DDP is initialized
                with more than one rank. Defaults to None.

        Returns:
            Tensor: Scalar mean Epps–Pulley statistic over all slices.

        Raises:
            ValueError: If embeddings are not floating point, are empty,
                or ``directions`` is malformed; or if distributed mode
                requires a shared ``step``.
        """
        if (embeddings.ndim < 2 or not embeddings.is_floating_point()
                or min(embeddings.shape) < 1):
            raise ValueError("Expected non-empty floating point embeddings (..., batch, features)")
        # end if
        if distributed is None:
            distributed = dist.is_initialized() and dist.get_world_size() > 1
        # end if
        if directions is None:
            if distributed and step is None:
                raise ValueError("Distributed SIGReg needs a shared step or shared directions")
            # end if
            directions = self.directions(embeddings, step)
        # end if
        if directions.ndim != 2 or directions.size(0) != embeddings.size(-1) or directions.size(1) < 1:
            raise ValueError("Expected directions shaped (features, slices)")
        # end if

        # Promote to the working dtype, embedding shape (..., batch, features)
        embeddings = embeddings.to(_working_dtype(embeddings))
        # directions shape (features, num_slices)
        directions = directions.to(embeddings)
        # Knot grid, shape (num_points,)
        knots = self.test.grid.to(embeddings)
        # Scalar (global) batch size, shape ()
        count = self.test.sample_count(embeddings, distributed)
        # Split directions into chunks of at most chunk_size, tuple of (features, chunk)
        chunks = directions.split(self.chunk_size or directions.size(1), dim=1)
        characteristic_sums = []
        for chunk in chunks:
            if self.chunk_size and torch.is_grad_enabled() and embeddings.requires_grad:
                # Checkpointed chunk, -> (2, ..., batch, chunk)
                sums = checkpoint(_projected_sums, embeddings, chunk, knots, use_reentrant=False)
            else:
                # Direct chunk, -> (2, ..., batch, chunk)
                sums = _projected_sums(embeddings, chunk, knots)
            # end if
            characteristic_sums.append(sums)
        # end for
        # Concatenate chunks on the slice axis, -> (2, ..., batch, num_slices)
        sums = torch.cat(characteristic_sums, dim=-2)
        # Mean statistic over slices, scalar (...)
        return self.test.statistic(sums, count, distributed).mean()
        # end return
    # end def forward
# end class SIGReg
