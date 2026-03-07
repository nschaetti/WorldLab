"""
Linear decoding analysis utilities (Phase 4).

This module evaluates how well latent coordinates can be linearly decoded from
representation features as a function of projection rank. It mirrors the
"decoding vs rank" analyses used in symmetry-geometry papers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class DecodingCurveResult:
    """
    Aggregated decoding error curves across random train/test splits.

    Attributes:
        ranks: Evaluated ranks.
        train_mse_mean: Mean train MSE for each rank.
        train_mse_std: Std train MSE for each rank.
        test_mse_mean: Mean test MSE for each rank.
        test_mse_std: Std test MSE for each rank.
        train_rel_mse_mean: Mean train relative MSE, normalized by target variance.
        test_rel_mse_mean: Mean test relative MSE, normalized by target variance.
        num_trials: Number of random splits used.
        train_size: Number of train samples per trial.
        ridge_alpha: Ridge coefficient used.
    """

    ranks: np.ndarray
    train_mse_mean: np.ndarray
    train_mse_std: np.ndarray
    test_mse_mean: np.ndarray
    test_mse_std: np.ndarray
    train_rel_mse_mean: np.ndarray
    test_rel_mse_mean: np.ndarray
    num_trials: int
    train_size: int
    ridge_alpha: float


def _as_2d_targets(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        return y[:, None]
    if y.ndim == 2:
        return y
    raise ValueError(f"targets must be 1D or 2D, got shape {y.shape}")


def fit_linear_decoder(
    x: np.ndarray,
    y: np.ndarray,
    ridge_alpha: float = 0.0,
) -> np.ndarray:
    """
    Fit linear decoder ``y_hat = x @ w``.

    Args:
        x: Feature matrix of shape ``[N, R]``.
        y: Target matrix of shape ``[N, D]`` (or vector ``[N]``).
        ridge_alpha: Ridge regularization coefficient. ``0`` means ridgeless
            solution via pseudoinverse.

    Returns:
        Weight matrix ``w`` of shape ``[R, D]``.
    """
    x = np.asarray(x, dtype=np.float64)
    y2 = _as_2d_targets(y)

    if x.ndim != 2:
        raise ValueError(f"x must be 2D, got shape {x.shape}")
    if x.shape[0] != y2.shape[0]:
        raise ValueError(f"x and y sample count mismatch: {x.shape[0]} vs {y2.shape[0]}")
    if ridge_alpha < 0:
        raise ValueError(f"ridge_alpha must be >= 0, got {ridge_alpha}")

    if ridge_alpha == 0.0:
        return np.linalg.pinv(x) @ y2

    xtx = x.T @ x
    reg = ridge_alpha * np.eye(xtx.shape[0], dtype=np.float64)
    xty = x.T @ y2
    return np.linalg.solve(xtx + reg, xty)


def predict_linear_decoder(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Compute ``x @ w`` with shape checks."""
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    if x.ndim != 2 or w.ndim != 2:
        raise ValueError(f"x and w must be 2D, got {x.shape} and {w.shape}")
    if x.shape[1] != w.shape[0]:
        raise ValueError(f"dimension mismatch: x has {x.shape[1]} features, w has {w.shape[0]} rows")
    return x @ w


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error over all entries."""
    yt = _as_2d_targets(y_true)
    yp = _as_2d_targets(y_pred)
    if yt.shape != yp.shape:
        raise ValueError(f"shape mismatch: {yt.shape} vs {yp.shape}")
    return float(np.mean((yt - yp) ** 2))


def relative_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Relative MSE normalized by mean target power.

    Formula:
        ``rel_mse = E||y - y_hat||^2 / (E||y||^2 + eps)``
    """
    yt = _as_2d_targets(y_true)
    yp = _as_2d_targets(y_pred)
    if yt.shape != yp.shape:
        raise ValueError(f"shape mismatch: {yt.shape} vs {yp.shape}")
    num = np.mean(np.sum((yt - yp) ** 2, axis=1))
    den = np.mean(np.sum(yt ** 2, axis=1)) + 1e-12
    return float(num / den)


def evaluate_linear_decoding_vs_rank(
    features: np.ndarray,
    targets: np.ndarray,
    ranks: Sequence[int],
    num_trials: int = 50,
    train_size: Optional[int] = None,
    ridge_alpha: float = 0.0,
    seed: int = 0,
) -> DecodingCurveResult:
    """
    Evaluate train/test linear decoding error as function of feature rank.

    Args:
        features: Feature matrix ``[N, R_max]`` (typically top PCA coordinates).
        targets: Targets ``[N]`` or ``[N, D]``.
        ranks: Iterable of evaluated ranks ``r`` (1 <= r <= R_max).
        num_trials: Number of random train/test splits.
        train_size: Train sample count per trial. Defaults to ``N//2``.
        ridge_alpha: Ridge coefficient (0.0 means ridgeless).
        seed: RNG seed.

    Returns:
        ``DecodingCurveResult`` with mean/std curves.
    """
    x = np.asarray(features, dtype=np.float64)
    y = _as_2d_targets(targets)

    if x.ndim != 2:
        raise ValueError(f"features must be 2D, got shape {x.shape}")
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"sample mismatch: features {x.shape[0]} vs targets {y.shape[0]}")
    if num_trials <= 0:
        raise ValueError(f"num_trials must be > 0, got {num_trials}")
    if ridge_alpha < 0:
        raise ValueError(f"ridge_alpha must be >= 0, got {ridge_alpha}")

    n, r_max = x.shape
    ranks_arr = np.asarray(list(ranks), dtype=np.int64)
    if ranks_arr.ndim != 1 or ranks_arr.size == 0:
        raise ValueError("ranks must be a non-empty 1D sequence")
    if np.any(ranks_arr <= 0) or np.any(ranks_arr > r_max):
        raise ValueError(f"ranks must be in [1, {r_max}], got {ranks_arr}")

    if train_size is None:
        train_size = n // 2
    if train_size <= 0 or train_size >= n:
        raise ValueError(f"train_size must be in [1, {n-1}], got {train_size}")

    rng = np.random.default_rng(seed)

    train_mse_trials = np.zeros((num_trials, ranks_arr.size), dtype=np.float64)
    test_mse_trials = np.zeros((num_trials, ranks_arr.size), dtype=np.float64)
    train_rel_trials = np.zeros((num_trials, ranks_arr.size), dtype=np.float64)
    test_rel_trials = np.zeros((num_trials, ranks_arr.size), dtype=np.float64)

    for trial in range(num_trials):
        perm = rng.permutation(n)
        train_idx = perm[:train_size]
        test_idx = perm[train_size:]

        y_train = y[train_idx]
        y_test = y[test_idx]

        for k, r in enumerate(ranks_arr.tolist()):
            xr_train = x[train_idx, :r]
            xr_test = x[test_idx, :r]

            w = fit_linear_decoder(x=xr_train, y=y_train, ridge_alpha=ridge_alpha)
            pred_train = predict_linear_decoder(xr_train, w)
            pred_test = predict_linear_decoder(xr_test, w)

            train_mse_trials[trial, k] = mse(y_train, pred_train)
            test_mse_trials[trial, k] = mse(y_test, pred_test)
            train_rel_trials[trial, k] = relative_mse(y_train, pred_train)
            test_rel_trials[trial, k] = relative_mse(y_test, pred_test)

    return DecodingCurveResult(
        ranks=ranks_arr,
        train_mse_mean=np.mean(train_mse_trials, axis=0),
        train_mse_std=np.std(train_mse_trials, axis=0),
        test_mse_mean=np.mean(test_mse_trials, axis=0),
        test_mse_std=np.std(test_mse_trials, axis=0),
        train_rel_mse_mean=np.mean(train_rel_trials, axis=0),
        test_rel_mse_mean=np.mean(test_rel_trials, axis=0),
        num_trials=num_trials,
        train_size=train_size,
        ridge_alpha=ridge_alpha,
    )
