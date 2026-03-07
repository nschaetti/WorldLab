"""
Collective-effect ablations and reconstruction tools (Phase 5).

This module implements experiments inspired by the robustness analyses in
Karkada et al. (2026):

1) remove direct couplings inside a target concept subset,
2) attempt to reconstruct target geometry from helper tokens,
3) quantify reconstruction quality with Gram and Procrustes metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Optional, Sequence

import numpy as np


FactorMode = Literal["absolute", "psd"]
HelperStrategy = Literal["top_coupling", "random"]


@dataclass(frozen=True)
class ReconstructionMetrics:
    """
    Reconstruction quality summary for a target subset.

    Attributes:
        procrustes_error: Shape mismatch after optimal rigid alignment.
        gram_mse: MSE between reconstructed and reference target Gram matrices.
        gram_rel_mse: Relative Gram MSE normalized by reference power.
    """

    procrustes_error: float
    gram_mse: float
    gram_rel_mse: float


@dataclass(frozen=True)
class HelperCurveResult:
    """
    Aggregated helper-size reconstruction curve.

    Attributes:
        helper_sizes: Evaluated helper counts.
        procrustes_mean: Mean Procrustes error per helper size.
        procrustes_std: Std Procrustes error per helper size.
        gram_rel_mse_mean: Mean relative Gram MSE per helper size.
        gram_rel_mse_std: Std relative Gram MSE per helper size.
        strategy: Helper selection strategy.
        num_trials: Number of trials per helper size.
    """

    helper_sizes: np.ndarray
    procrustes_mean: np.ndarray
    procrustes_std: np.ndarray
    gram_rel_mse_mean: np.ndarray
    gram_rel_mse_std: np.ndarray
    strategy: str
    num_trials: int


def _as_index_array(indices: Sequence[int]) -> np.ndarray:
    arr = np.asarray(indices, dtype=np.int64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("indices must be a non-empty 1D sequence")
    if np.any(arr < 0):
        raise ValueError("indices must be non-negative")
    return arr


def ablate_target_block(
    matrix: np.ndarray,
    target_indices: Sequence[int],
    value: float = 0.0,
) -> np.ndarray:
    """
    Ablate all entries in the target-target block.

    Args:
        matrix: Symmetric matrix ``[V, V]``.
        target_indices: Target concept token indices.
        value: Fill value for target-target entries.

    Returns:
        Copy of matrix with target block set to ``value``.
    """
    m = np.asarray(matrix, dtype=np.float64)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError(f"matrix must be square 2D, got {m.shape}")

    t = _as_index_array(target_indices)
    if np.max(t) >= m.shape[0]:
        raise ValueError("target index out of range")

    out = np.array(m, copy=True)
    out[np.ix_(t, t)] = value
    return out


def select_helpers_by_coupling(
    matrix: np.ndarray,
    target_indices: Sequence[int],
    top_k: int,
    candidate_indices: Optional[Sequence[int]] = None,
    use_absolute: bool = True,
) -> np.ndarray:
    """
    Select helper tokens with strongest coupling to target subset.

    Score for candidate token ``j`` is:
        ``sum_i |M_ji|`` over target indices if ``use_absolute=True``,
        otherwise ``sum_i M_ji^2``.
    """
    m = np.asarray(matrix, dtype=np.float64)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError(f"matrix must be square 2D, got {m.shape}")

    t = _as_index_array(target_indices)
    v = m.shape[0]

    if candidate_indices is None:
        mask = np.ones((v,), dtype=bool)
        mask[t] = False
        candidates = np.where(mask)[0]
    else:
        candidates = _as_index_array(candidate_indices)
        candidates = np.array([x for x in candidates.tolist() if x not in set(t.tolist())], dtype=np.int64)
        if candidates.size == 0:
            raise ValueError("candidate_indices yields empty helper candidate set")

    if top_k <= 0:
        raise ValueError(f"top_k must be > 0, got {top_k}")
    if top_k > candidates.size:
        top_k = int(candidates.size)

    block = m[np.ix_(candidates, t)]
    if use_absolute:
        scores = np.sum(np.abs(block), axis=1)
    else:
        scores = np.sum(block * block, axis=1)

    order = np.argsort(scores)[::-1]
    return candidates[order[:top_k]]


def _factorize_symmetric(
    matrix: np.ndarray,
    rank: Optional[int],
    mode: FactorMode,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Factorize symmetric matrix into coordinates.

    - ``mode='psd'`` keeps only positive eigenvalues,
    - ``mode='absolute'`` keeps largest-magnitude eigenvalues and uses
      ``sqrt(abs(lambda))`` (matching ``|M*|``-style embedding geometry).
    """
    a = np.asarray(matrix, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"matrix must be square 2D, got {a.shape}")

    evals, evecs = np.linalg.eigh(a)
    order = np.argsort(np.abs(evals))[::-1] if mode == "absolute" else np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    if mode == "psd":
        keep = evals > 0
        evals = evals[keep]
        evecs = evecs[:, keep]
        scales = np.sqrt(evals)
    elif mode == "absolute":
        scales = np.sqrt(np.abs(evals))
    else:
        raise ValueError(f"unknown mode: {mode}")

    if rank is not None:
        if rank <= 0:
            raise ValueError(f"rank must be > 0, got {rank}")
        evals = evals[:rank]
        evecs = evecs[:, :rank]
        scales = scales[:rank]

    coords = evecs * scales[None, :]
    recon = coords @ coords.T
    return coords, evals, recon


def procrustes_error(
    reference: np.ndarray,
    estimate: np.ndarray,
    center: bool = True,
    scale: bool = True,
) -> float:
    """
    Orthogonal Procrustes mismatch between two coordinate sets.

    Returns:
        ``||X - YR||_F / ||X||_F`` after optional centering/scaling.
    """
    x = np.asarray(reference, dtype=np.float64)
    y = np.asarray(estimate, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError(f"reference and estimate must be [N,Dx],[N,Dy], got {x.shape}, {y.shape}")

    if center:
        x = x - np.mean(x, axis=0, keepdims=True)
        y = y - np.mean(y, axis=0, keepdims=True)

    if scale:
        nx = np.linalg.norm(x)
        ny = np.linalg.norm(y)
        if nx > 0:
            x = x / nx
        if ny > 0:
            y = y / ny

    c = y.T @ x
    u, _, vt = np.linalg.svd(c, full_matrices=False)
    r = u @ vt
    y_aligned = y @ r
    return float(np.linalg.norm(x - y_aligned) / (np.linalg.norm(x) + 1e-12))


def compute_reconstruction_metrics(
    reference_coords: np.ndarray,
    estimated_coords: np.ndarray,
    reference_gram: np.ndarray,
    estimated_gram: np.ndarray,
) -> ReconstructionMetrics:
    """Compute Procrustes + Gram reconstruction metrics."""
    p_err = procrustes_error(reference=reference_coords, estimate=estimated_coords)
    diff = reference_gram - estimated_gram
    mse = float(np.mean(diff * diff))
    rel = float(np.mean(diff * diff) / (np.mean(reference_gram * reference_gram) + 1e-12))
    return ReconstructionMetrics(procrustes_error=p_err, gram_mse=mse, gram_rel_mse=rel)


def reconstruct_target_with_helpers(
    matrix: np.ndarray,
    target_indices: Sequence[int],
    helper_indices: Sequence[int],
    rank: Optional[int] = None,
    mode: FactorMode = "absolute",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reconstruct target geometry from a target+helpers submatrix.

    Returns:
        ``(target_coords, target_gram_recon, used_indices)``.
    """
    m = np.asarray(matrix, dtype=np.float64)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError(f"matrix must be square 2D, got {m.shape}")

    t = _as_index_array(target_indices)
    h = _as_index_array(helper_indices) if len(helper_indices) > 0 else np.zeros((0,), dtype=np.int64)

    used = np.concatenate([t, h], axis=0)
    if np.max(used) >= m.shape[0]:
        raise ValueError("index out of range")
    if np.unique(used).size != used.size:
        raise ValueError("target_indices and helper_indices must be disjoint and unique")

    sub = m[np.ix_(used, used)]
    coords_sub, _, recon_sub = _factorize_symmetric(sub, rank=rank, mode=mode)

    nt = t.size
    coords_t = coords_sub[:nt]
    gram_t = recon_sub[:nt, :nt]
    return coords_t, gram_t, used


def helper_reconstruction_curve(
    original_matrix: np.ndarray,
    ablated_matrix: np.ndarray,
    target_indices: Sequence[int],
    helper_sizes: Sequence[int],
    strategy: HelperStrategy = "top_coupling",
    num_trials: int = 20,
    rank: Optional[int] = None,
    mode: FactorMode = "absolute",
    seed: int = 0,
) -> HelperCurveResult:
    """
    Evaluate target-geometry reconstruction as helper pool size grows.

    Reference geometry is computed from ``original_matrix`` on the target subset.
    Reconstruction uses ``ablated_matrix`` and selected helpers.
    """
    orig = np.asarray(original_matrix, dtype=np.float64)
    abl = np.asarray(ablated_matrix, dtype=np.float64)
    if orig.shape != abl.shape:
        raise ValueError(f"matrix shape mismatch: {orig.shape} vs {abl.shape}")

    t = _as_index_array(target_indices)
    v = orig.shape[0]
    if np.max(t) >= v:
        raise ValueError("target index out of range")

    helper_sizes_arr = np.asarray(list(helper_sizes), dtype=np.int64)
    if helper_sizes_arr.ndim != 1 or helper_sizes_arr.size == 0:
        raise ValueError("helper_sizes must be non-empty 1D sequence")
    if np.any(helper_sizes_arr < 0):
        raise ValueError("helper_sizes must be >= 0")

    pool_mask = np.ones((v,), dtype=bool)
    pool_mask[t] = False
    helper_pool = np.where(pool_mask)[0]

    max_h = int(np.max(helper_sizes_arr))
    if max_h > helper_pool.size:
        raise ValueError(f"max helper size {max_h} exceeds available pool {helper_pool.size}")

    ref_coords, _, ref_recon = _factorize_symmetric(orig[np.ix_(t, t)], rank=rank, mode=mode)
    ref_gram = ref_recon

    rng = np.random.default_rng(seed)
    trials = max(1, num_trials)

    p_mean = np.zeros_like(helper_sizes_arr, dtype=np.float64)
    p_std = np.zeros_like(helper_sizes_arr, dtype=np.float64)
    g_mean = np.zeros_like(helper_sizes_arr, dtype=np.float64)
    g_std = np.zeros_like(helper_sizes_arr, dtype=np.float64)

    for i, h in enumerate(helper_sizes_arr.tolist()):
        p_trials = np.zeros((trials,), dtype=np.float64)
        g_trials = np.zeros((trials,), dtype=np.float64)

        for k in range(trials):
            if h == 0:
                helpers = np.zeros((0,), dtype=np.int64)
            elif strategy == "top_coupling":
                helpers = select_helpers_by_coupling(
                    matrix=abl,
                    target_indices=t,
                    top_k=h,
                    candidate_indices=helper_pool,
                    use_absolute=True,
                )
            elif strategy == "random":
                helpers = rng.choice(helper_pool, size=h, replace=False)
            else:
                raise ValueError(f"unknown strategy: {strategy}")

            est_coords, est_gram, _ = reconstruct_target_with_helpers(
                matrix=abl,
                target_indices=t,
                helper_indices=helpers,
                rank=rank,
                mode=mode,
            )

            metrics = compute_reconstruction_metrics(
                reference_coords=ref_coords,
                estimated_coords=est_coords,
                reference_gram=ref_gram,
                estimated_gram=est_gram,
            )
            p_trials[k] = metrics.procrustes_error
            g_trials[k] = metrics.gram_rel_mse

        p_mean[i] = float(np.mean(p_trials))
        p_std[i] = float(np.std(p_trials))
        g_mean[i] = float(np.mean(g_trials))
        g_std[i] = float(np.std(g_trials))

    return HelperCurveResult(
        helper_sizes=helper_sizes_arr,
        procrustes_mean=p_mean,
        procrustes_std=p_std,
        gram_rel_mse_mean=g_mean,
        gram_rel_mse_std=g_std,
        strategy=strategy,
        num_trials=trials,
    )
