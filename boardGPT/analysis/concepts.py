"""
Latent concept specifications and distance utilities.

In the symmetry analysis workflow, a "concept" is a set of tokens endowed with
coordinates in a latent space plus a distance function and boundary condition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class ConceptSpec:
    """
    Geometry metadata for a subset of tokens.

    Attributes:
        name: Human-readable concept name.
        token_ids: Token IDs participating in the concept (shape ``[N]``).
        coordinates: Latent coordinates per token (shape ``[N, D]``).
        periodic: Boolean periodic mask per dimension (shape ``[D]``).
        periods: Period value for periodic dimensions; ignored for open ones.
    """

    name: str
    token_ids: np.ndarray
    coordinates: np.ndarray
    periodic: np.ndarray
    periods: np.ndarray

    @property
    def num_tokens(self) -> int:
        return int(self.token_ids.shape[0])

    @property
    def dim(self) -> int:
        return int(self.coordinates.shape[1])


def _normalize_periodic_spec(
    dim: int,
    periodic: Optional[Sequence[bool]] = None,
    periods: Optional[Sequence[float]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize periodic flags and periods to dense arrays."""
    if periodic is None:
        periodic_arr = np.zeros((dim,), dtype=bool)
    else:
        periodic_arr = np.asarray(periodic, dtype=bool)
        if periodic_arr.shape != (dim,):
            raise ValueError(f"periodic must have shape ({dim},), got {periodic_arr.shape}")

    if periods is None:
        periods_arr = np.ones((dim,), dtype=np.float64)
    else:
        periods_arr = np.asarray(periods, dtype=np.float64)
        if periods_arr.shape != (dim,):
            raise ValueError(f"periods must have shape ({dim},), got {periods_arr.shape}")
        if np.any(periods_arr <= 0):
            raise ValueError("periods must be strictly positive")

    return periodic_arr, periods_arr


def lattice_1d_concept(
    num_tokens: int,
    name: str = "lattice_1d",
    periodic: bool = False,
    period: float = 2.0,
    token_offset: int = 0,
    normalize_to_unit_interval: bool = True,
) -> ConceptSpec:
    """
    Create a 1D lattice concept.

    Args:
        num_tokens: Number of tokens in the concept.
        name: Concept name.
        periodic: Whether boundary is periodic.
        period: Period value if periodic.
        token_offset: Offset added to token IDs.
        normalize_to_unit_interval: If True, coordinates are rescaled to ``[-1, 1]``.

    Returns:
        ``ConceptSpec`` for a 1D lattice.
    """
    if num_tokens <= 0:
        raise ValueError("num_tokens must be > 0")

    token_ids = np.arange(token_offset, token_offset + num_tokens, dtype=np.int64)
    x = np.arange(num_tokens, dtype=np.float64)
    if normalize_to_unit_interval:
        if num_tokens == 1:
            x = np.array([0.0], dtype=np.float64)
        else:
            x = 2.0 * (x / (num_tokens - 1)) - 1.0
    coords = x.reshape(-1, 1)
    periodic_arr = np.array([periodic], dtype=bool)
    periods_arr = np.array([period], dtype=np.float64)
    return ConceptSpec(name=name, token_ids=token_ids, coordinates=coords, periodic=periodic_arr, periods=periods_arr)


def lattice_nd_concept(
    shape: Sequence[int],
    name: str = "lattice_nd",
    periodic: Optional[Sequence[bool]] = None,
    periods: Optional[Sequence[float]] = None,
    token_offset: int = 0,
    normalize_to_unit_interval: bool = True,
) -> ConceptSpec:
    """
    Create an N-dimensional lattice concept.

    Args:
        shape: Number of sites per dimension (e.g., ``(8, 8)``).
        name: Concept name.
        periodic: Optional periodic mask per dimension.
        periods: Optional period per dimension.
        token_offset: Offset added to token IDs.
        normalize_to_unit_interval: If True each axis is mapped to ``[-1,1]``.

    Returns:
        ``ConceptSpec`` for a lattice in ``D=len(shape)`` dimensions.
    """
    shape_arr = np.asarray(shape, dtype=np.int64)
    if shape_arr.ndim != 1 or np.any(shape_arr <= 0):
        raise ValueError(f"shape must be 1D with positive entries, got {shape}")

    dim = int(shape_arr.shape[0])
    periodic_arr, periods_arr = _normalize_periodic_spec(dim=dim, periodic=periodic, periods=periods)

    axes = []
    for size in shape_arr.tolist():
        x = np.arange(size, dtype=np.float64)
        if normalize_to_unit_interval:
            if size == 1:
                x = np.array([0.0], dtype=np.float64)
            else:
                x = 2.0 * (x / (size - 1)) - 1.0
        axes.append(x)

    mesh = np.meshgrid(*axes, indexing="ij")
    coords = np.stack([m.reshape(-1) for m in mesh], axis=1)
    token_ids = np.arange(token_offset, token_offset + coords.shape[0], dtype=np.int64)
    return ConceptSpec(name=name, token_ids=token_ids, coordinates=coords, periodic=periodic_arr, periods=periods_arr)


def pairwise_distances(spec: ConceptSpec) -> np.ndarray:
    """
    Compute pairwise distances under open/periodic boundary conditions.

    Periodic dimensions use the minimal wrapped displacement:
        ``delta = min(|x-y|, period - |x-y|)``.

    Returns:
        Distance matrix of shape ``[N, N]``.
    """
    x = spec.coordinates
    n, dim = x.shape
    diff = x[:, None, :] - x[None, :, :]
    abs_diff = np.abs(diff)

    if np.any(spec.periodic):
        for d in range(dim):
            if spec.periodic[d]:
                p = spec.periods[d]
                wrapped = p - abs_diff[:, :, d]
                abs_diff[:, :, d] = np.minimum(abs_diff[:, :, d], wrapped)

    return np.sqrt(np.sum(abs_diff * abs_diff, axis=-1))
