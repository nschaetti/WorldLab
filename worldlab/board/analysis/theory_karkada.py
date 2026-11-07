"""
Phase 2 theory utilities for symmetry-in-statistics analysis.

This module implements the practical numerical core used to compare observed
co-occurrence structure against translation-symmetric kernel models.

Scope of this implementation
----------------------------
- distance-binned translation-symmetry diagnostics,
- exponential-kernel fitting from empirical profiles,
- kernel-matrix reconstruction from fitted parameters,
- eigendecomposition/PCA geometry extraction from Gram matrices.

These tools are model-agnostic and work for synthetic worlds, OthelloGPT
representations, and other token representation systems.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class BinnedProfile:
    """
    Distance-binned summary of a matrix.

    Attributes:
        bin_edges: Bin edges of shape ``[B+1]``.
        bin_centers: Bin centers of shape ``[B]``.
        bin_means: Mean matrix value in each distance bin (shape ``[B]``).
        bin_counts: Number of samples per bin (shape ``[B]``).
    """

    bin_edges: np.ndarray
    bin_centers: np.ndarray
    bin_means: np.ndarray
    bin_counts: np.ndarray


@dataclass(frozen=True)
class ExponentialFitResult:
    """
    Fit result for an exponential kernel profile.

    Model:
        ``k(d) = offset + amplitude * exp(-d / sigma)``

    Attributes:
        amplitude: Fitted amplitude.
        sigma: Fitted decay scale.
        offset: Fitted constant offset.
        mse: Mean squared error on fitted points.
        r2: Coefficient of determination on fitted points.
    """

    amplitude: float
    sigma: float
    offset: float
    mse: float
    r2: float


@dataclass(frozen=True)
class SymmetryDiagnostic:
    """
    Translation-symmetry diagnostic summary.

    Attributes:
        profile: Binned profile object.
        predicted: Per-entry matrix reconstructed from binned profile.
        r2: Variance explained by distance-only prediction.
        mse: Mean squared error of distance-only prediction.
    """

    profile: BinnedProfile
    predicted: np.ndarray
    r2: float
    mse: float


def _validate_square_same_shape(matrix: np.ndarray, distances: np.ndarray) -> None:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"matrix must be square 2D, got shape {matrix.shape}")
    if distances.shape != matrix.shape:
        raise ValueError(
            f"distances must have same shape as matrix: {matrix.shape}, got {distances.shape}"
        )


def distance_binned_profile(
    matrix: np.ndarray,
    distances: np.ndarray,
    num_bins: int = 32,
    include_diagonal: bool = False,
) -> BinnedProfile:
    """
    Compute a distance-binned profile of matrix values.

    Args:
        matrix: Matrix ``A`` to summarize (typically Gram, PMI or M* submatrix).
        distances: Pairwise distance matrix with same shape as ``matrix``.
        num_bins: Number of distance bins.
        include_diagonal: Whether to include ``i=j`` entries.

    Returns:
        A ``BinnedProfile`` object.
    """
    _validate_square_same_shape(matrix=matrix, distances=distances)
    if num_bins < 2:
        raise ValueError(f"num_bins must be >= 2, got {num_bins}")

    n = matrix.shape[0]
    tri_k = 0 if include_diagonal else 1
    ii, jj = np.triu_indices(n, k=tri_k)

    d = distances[ii, jj]
    a = matrix[ii, jj]

    d_min = float(np.min(d))
    d_max = float(np.max(d))
    if not np.isfinite(d_min) or not np.isfinite(d_max):
        raise ValueError("distance entries must be finite")

    if d_max == d_min:
        d_max = d_min + 1e-9

    edges = np.linspace(d_min, d_max, num_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    sums = np.zeros((num_bins,), dtype=np.float64)
    counts = np.zeros((num_bins,), dtype=np.int64)

    bin_ids = np.searchsorted(edges, d, side="right") - 1
    bin_ids = np.clip(bin_ids, 0, num_bins - 1)
    np.add.at(sums, bin_ids, a)
    np.add.at(counts, bin_ids, 1)

    means = np.zeros((num_bins,), dtype=np.float64)
    nonzero = counts > 0
    means[nonzero] = sums[nonzero] / counts[nonzero]

    return BinnedProfile(
        bin_edges=edges,
        bin_centers=centers,
        bin_means=means,
        bin_counts=counts,
    )


def _distance_only_prediction_from_profile(
    distances: np.ndarray,
    profile: BinnedProfile,
) -> np.ndarray:
    """Map each distance entry to the corresponding binned mean."""
    bin_ids = np.searchsorted(profile.bin_edges, distances, side="right") - 1
    bin_ids = np.clip(bin_ids, 0, profile.bin_means.shape[0] - 1)
    return profile.bin_means[bin_ids]


def translation_symmetry_diagnostic(
    matrix: np.ndarray,
    distances: np.ndarray,
    num_bins: int = 32,
    include_diagonal: bool = False,
) -> SymmetryDiagnostic:
    """
    Quantify how well matrix entries are explained by distance alone.

    Method:
    1) bin entries by distance,
    2) replace each entry by the mean value of its distance bin,
    3) compute ``R^2`` and MSE between original matrix and this distance-only model.

    Returns:
        ``SymmetryDiagnostic`` with profile and scores.
    """
    profile = distance_binned_profile(
        matrix=matrix,
        distances=distances,
        num_bins=num_bins,
        include_diagonal=include_diagonal,
    )
    pred = _distance_only_prediction_from_profile(distances=distances, profile=profile)

    n = matrix.shape[0]
    tri_k = 0 if include_diagonal else 1
    ii, jj = np.triu_indices(n, k=tri_k)
    y = matrix[ii, jj]
    yhat = pred[ii, jj]

    mse = float(np.mean((y - yhat) ** 2))
    var = float(np.var(y))
    r2 = 1.0 - mse / (var + 1e-12)

    return SymmetryDiagnostic(profile=profile, predicted=pred, r2=float(r2), mse=mse)


def fit_exponential_profile(
    distances: np.ndarray,
    values: np.ndarray,
    sigma_grid: Optional[np.ndarray] = None,
    fit_offset: bool = True,
) -> ExponentialFitResult:
    """
    Fit ``offset + amplitude * exp(-d/sigma)`` on 1D samples ``(d, value)``.

    The fit is solved by grid-search over ``sigma`` and closed-form least squares
    for linear coefficients ``(offset, amplitude)`` at each sigma.

    Args:
        distances: Distance samples ``d`` (1D).
        values: Value samples ``y`` (1D), same shape as ``distances``.
        sigma_grid: Optional grid of positive sigma values.
        fit_offset: If False, force offset to zero.

    Returns:
        ``ExponentialFitResult``.
    """
    d = np.asarray(distances, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    if d.shape != y.shape or d.ndim != 1:
        raise ValueError(f"distances and values must be 1D and same shape, got {d.shape}, {y.shape}")

    mask = np.isfinite(d) & np.isfinite(y)
    d = d[mask]
    y = y[mask]
    if d.size < 3:
        raise ValueError("need at least 3 valid samples to fit exponential profile")

    d_max = float(np.max(d))
    d_pos = d[d > 0]
    d_min_pos = float(np.min(d_pos)) if d_pos.size > 0 else max(d_max * 1e-3, 1e-3)

    if sigma_grid is None:
        low = max(d_min_pos * 0.5, 1e-4)
        high = max(d_max * 2.0, low * 10.0)
        sigma_grid = np.geomspace(low, high, 128)
    else:
        sigma_grid = np.asarray(sigma_grid, dtype=np.float64)
        sigma_grid = sigma_grid[np.isfinite(sigma_grid) & (sigma_grid > 0)]
        if sigma_grid.size == 0:
            raise ValueError("sigma_grid must contain positive finite values")

    best_sse = np.inf
    best = None

    for sigma in sigma_grid:
        k = np.exp(-d / sigma)
        if fit_offset:
            x = np.stack([np.ones_like(k), k], axis=1)
            coeff, *_ = np.linalg.lstsq(x, y, rcond=None)
            offset = float(coeff[0])
            amplitude = float(coeff[1])
            yhat = x @ coeff
        else:
            denom = float(np.dot(k, k)) + 1e-12
            amplitude = float(np.dot(k, y) / denom)
            offset = 0.0
            yhat = amplitude * k

        sse = float(np.sum((y - yhat) ** 2))
        if sse < best_sse:
            best_sse = sse
            best = (amplitude, float(sigma), offset, yhat)

    assert best is not None
    amplitude, sigma, offset, yhat = best
    mse = float(np.mean((y - yhat) ** 2))
    var = float(np.var(y))
    r2 = 1.0 - mse / (var + 1e-12)
    return ExponentialFitResult(amplitude=amplitude, sigma=sigma, offset=offset, mse=mse, r2=float(r2))


def fit_exponential_kernel_from_matrix(
    matrix: np.ndarray,
    distances: np.ndarray,
    num_bins: int = 32,
    include_diagonal: bool = False,
    fit_offset: bool = True,
) -> ExponentialFitResult:
    """
    Fit an exponential kernel to a matrix using its distance-binned profile.

    This is a high-level convenience wrapper:
    1) compute profile ``mean value vs distance``,
    2) fit ``offset + amplitude * exp(-d/sigma)``.
    """
    profile = distance_binned_profile(
        matrix=matrix,
        distances=distances,
        num_bins=num_bins,
        include_diagonal=include_diagonal,
    )

    valid = profile.bin_counts > 0
    return fit_exponential_profile(
        distances=profile.bin_centers[valid],
        values=profile.bin_means[valid],
        fit_offset=fit_offset,
    )


def build_exponential_kernel_matrix(
    distances: np.ndarray,
    amplitude: float,
    sigma: float,
    offset: float = 0.0,
) -> np.ndarray:
    """
    Construct matrix ``K_ij = offset + amplitude * exp(-dist_ij / sigma)``.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    return offset + amplitude * np.exp(-distances / sigma)


def pca_from_gram(
    gram: np.ndarray,
    rank: Optional[int] = None,
    center: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract PCA coordinates from a Gram matrix.

    Args:
        gram: Symmetric Gram-like matrix of shape ``[N, N]``.
        rank: Optional number of components to keep.
        center: If True, apply double-centering ``P G P``.

    Returns:
        Tuple ``(coords, eigvals, eigvecs)`` where:
        - ``coords`` has shape ``[N, r]``,
        - ``eigvals`` has shape ``[r]``,
        - ``eigvecs`` has shape ``[N, r]``.
    """
    g = np.asarray(gram, dtype=np.float64)
    if g.ndim != 2 or g.shape[0] != g.shape[1]:
        raise ValueError(f"gram must be square 2D, got shape {g.shape}")

    n = g.shape[0]
    if center:
        p = np.eye(n) - np.ones((n, n), dtype=np.float64) / n
        g = p @ g @ p

    evals, evecs = np.linalg.eigh(g)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    positive = evals > 0
    evals = evals[positive]
    evecs = evecs[:, positive]

    if rank is not None:
        if rank <= 0:
            raise ValueError(f"rank must be > 0, got {rank}")
        evals = evals[:rank]
        evecs = evecs[:, :rank]

    coords = evecs * np.sqrt(evals[None, :])
    return coords, evals, evecs
