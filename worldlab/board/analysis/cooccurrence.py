"""
Weighted co-occurrence statistics for token sequences.

Why this module exists
----------------------
The central object in Karkada et al. (2026) is a pairwise co-occurrence matrix
from which PMI-like quantities and spectral geometry are derived. This module
implements those statistics in a reusable and well-documented way.

Notation
--------
Given token sequences over a vocabulary of size ``V``:

- ``counts_ij``: weighted skip-gram co-occurrence counts,
- ``P_ij``: normalized co-occurrence probabilities,
- ``P_i``: unigram probabilities, ``P_i = sum_j P_ij``,
- ``PMI_ij = log((P_ij + eps) / (P_i P_j + eps))``,
- ``M*_ij = 2 * (P_ij - P_i P_j) / (P_ij + P_i P_j + eps)``.

``M*`` is the normalized excess co-occurrence matrix used in the paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np


@dataclass
class CooccurrenceStats:
    """
    Container for co-occurrence-derived statistics.

    Attributes:
        counts_ij: Raw weighted co-occurrence counts of shape ``[V, V]``.
        p_ij: Normalized co-occurrence probabilities ``P_ij``.
        p_i: Unigram probabilities ``P_i`` (shape ``[V]``).
        pmi: Pointwise mutual information matrix.
        m_star: Normalized excess co-occurrence matrix ``M*``.
        total_pair_weight: Sum of all weighted pair contributions.
    """

    counts_ij: np.ndarray
    p_ij: np.ndarray
    p_i: np.ndarray
    pmi: np.ndarray
    m_star: np.ndarray
    total_pair_weight: float


def _resolve_distance_weights(window_size: int, distance_weights: Optional[Sequence[float]]) -> np.ndarray:
    """
    Build the per-distance weighting array ``w[d]`` for ``d in [1, window_size]``.

    Args:
        window_size: Maximum skip distance.
        distance_weights: Optional length-``window_size`` sequence. If ``None``,
            the default triangular profile is used: ``w[d] = window_size + 1 - d``.

    Returns:
        NumPy array of shape ``[window_size]`` with strictly non-negative weights.

    Raises:
        ValueError: If ``window_size < 1``, wrong length, or negative weights.
    """
    if window_size < 1:
        raise ValueError(f"window_size must be >= 1, got {window_size}")

    if distance_weights is None:
        weights = np.arange(window_size, 0, -1, dtype=np.float64)
    else:
        weights = np.asarray(distance_weights, dtype=np.float64)
        if weights.shape != (window_size,):
            raise ValueError(
                f"distance_weights must have shape ({window_size},), got {weights.shape}"
            )

    if np.any(weights < 0):
        raise ValueError("distance_weights must be non-negative")
    return weights


def _iter_valid_tokens(tokens: Sequence[int], vocab_size: int) -> np.ndarray:
    """
    Convert an integer sequence to a validated NumPy array.

    Tokens outside ``[0, vocab_size-1]`` are ignored.
    """
    arr = np.asarray(tokens, dtype=np.int64)
    if arr.ndim != 1:
        raise ValueError(f"each sequence must be 1D, got shape {arr.shape}")
    mask = (arr >= 0) & (arr < vocab_size)
    return arr[mask]


def compute_weighted_cooccurrence_counts(
    sequences: Iterable[Sequence[int]],
    vocab_size: int,
    window_size: int = 16,
    distance_weights: Optional[Sequence[float]] = None,
    symmetric: bool = True,
) -> tuple[np.ndarray, float]:
    """
    Compute weighted skip-gram co-occurrence counts.

    For each token position ``t`` and distance ``d`` in ``[1, L]``:

    - forward pair: ``(x_t, x_{t+d})`` with weight ``w[d]``,
    - if ``symmetric=True``, backward pair is added implicitly by mirroring,
      making counts symmetric by construction.

    Args:
        sequences: Iterable of token sequences (integer IDs).
        vocab_size: Vocabulary size ``V``.
        window_size: Context window size ``L``.
        distance_weights: Optional custom weights ``w[d]``.
        symmetric: If True, also add reverse direction contribution.

    Returns:
        Tuple ``(counts_ij, total_pair_weight)``.
    """
    if vocab_size <= 0:
        raise ValueError(f"vocab_size must be > 0, got {vocab_size}")

    w = _resolve_distance_weights(window_size=window_size, distance_weights=distance_weights)
    counts = np.zeros((vocab_size, vocab_size), dtype=np.float64)
    total_pair_weight = 0.0

    for seq in sequences:
        x = _iter_valid_tokens(seq, vocab_size=vocab_size)
        n = x.shape[0]
        if n <= 1:
            continue

        for d in range(1, window_size + 1):
            weight = float(w[d - 1])
            if weight == 0.0 or d >= n:
                continue

            left = x[:-d]
            right = x[d:]

            np.add.at(counts, (left, right), weight)
            total_pair_weight += weight * left.shape[0]

            if symmetric:
                np.add.at(counts, (right, left), weight)
                total_pair_weight += weight * right.shape[0]

    return counts, total_pair_weight


def _safe_normalize(counts: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize a non-negative matrix into a probability matrix."""
    total = float(np.sum(counts))
    return counts / (total + eps)


def compute_pmi(p_ij: np.ndarray, p_i: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Compute elementwise PMI matrix.

    Formula:
        ``PMI_ij = log((P_ij + eps) / (P_i * P_j + eps))``
    """
    baseline = np.outer(p_i, p_i)
    return np.log((p_ij + eps) / (baseline + eps))


def compute_m_star(p_ij: np.ndarray, p_i: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Compute normalized excess co-occurrence matrix ``M*``.

    Formula:
        ``M*_ij = 2 * (P_ij - P_i P_j) / (P_ij + P_i P_j + eps)``
    """
    baseline = np.outer(p_i, p_i)
    num = p_ij - baseline
    den = p_ij + baseline + eps
    return 2.0 * num / den


def build_cooccurrence_statistics(
    sequences: Iterable[Sequence[int]],
    vocab_size: int,
    window_size: int = 16,
    distance_weights: Optional[Sequence[float]] = None,
    symmetric: bool = True,
    eps: float = 1e-12,
) -> CooccurrenceStats:
    """
    End-to-end helper that computes all Phase 1 statistics.

    Args:
        sequences: Iterable of token sequences.
        vocab_size: Vocabulary size.
        window_size: Context window size.
        distance_weights: Optional custom weights for each distance.
        symmetric: Whether to count reverse context pairs as well.
        eps: Numeric stabilizer used in PMI and ``M*`` formulas.

    Returns:
        A ``CooccurrenceStats`` object.
    """
    counts_ij, total_pair_weight = compute_weighted_cooccurrence_counts(
        sequences=sequences,
        vocab_size=vocab_size,
        window_size=window_size,
        distance_weights=distance_weights,
        symmetric=symmetric,
    )
    p_ij = _safe_normalize(counts_ij, eps=eps)
    p_i = np.sum(p_ij, axis=1)
    pmi = compute_pmi(p_ij=p_ij, p_i=p_i, eps=eps)
    m_star = compute_m_star(p_ij=p_ij, p_i=p_i, eps=eps)

    return CooccurrenceStats(
        counts_ij=counts_ij,
        p_ij=p_ij,
        p_i=p_i,
        pmi=pmi,
        m_star=m_star,
        total_pair_weight=total_pair_weight,
    )
