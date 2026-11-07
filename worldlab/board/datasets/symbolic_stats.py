"""
Utilities for co-occurrence analysis on symbolic sequence datasets.
"""

from __future__ import annotations

import pickle
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, Hashable, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


Symbol = Hashable


def _as_list_sequence(sequence: Any) -> List[Any]:
    if isinstance(sequence, np.ndarray):
        return sequence.tolist()
    if isinstance(sequence, (list, tuple)):
        return list(sequence)
    return list(sequence)


def _normalize_symbol(symbol: Any, decode_bytes: bool = True) -> Any:
    if decode_bytes and isinstance(symbol, (bytes, bytearray, np.bytes_)):
        return bytes(symbol).decode("utf-8")
    return symbol


def load_symbolic_pickle(path: str | Path, decode_bytes: bool = True) -> List[List[Any]]:
    """
    Load a pickle file containing symbolic sequences.

    The payload is expected to be an iterable of sequences. Sequences can be
    Python lists/tuples or numpy arrays.
    """
    p = Path(path)
    with p.open("rb") as f:
        payload = pickle.load(f)

    if not isinstance(payload, (list, tuple)):
        raise ValueError("Pickle payload must be a list/tuple of sequences")

    sequences: List[List[Any]] = []
    for sequence in payload:
        values = _as_list_sequence(sequence)
        normalized = [_normalize_symbol(v, decode_bytes=decode_bytes) for v in values]
        sequences.append(normalized)
    return sequences


def _build_vocabulary(sequences: Sequence[Sequence[Symbol]], vocabulary: Optional[Sequence[Symbol]]) -> List[Symbol]:
    if vocabulary is not None:
        return list(vocabulary)

    seen: "OrderedDict[Symbol, None]" = OrderedDict()
    for sequence in sequences:
        for token in sequence:
            if token not in seen:
                seen[token] = None
    return list(seen.keys())


def compute_cooccurrence_matrix(
    sequences: Sequence[Sequence[Symbol]],
    vocabulary: Optional[Sequence[Symbol]] = None,
    mode: str = "same_sequence",
    max_distance: int = 1,
    include_self: bool = False,
    symmetric: bool = True,
) -> Tuple[np.ndarray, List[Symbol]]:
    """
    Compute a token co-occurrence matrix.

    Args:
        sequences: Symbolic sequences.
        vocabulary: Optional fixed token order.
        mode: "same_sequence" or "adjacent".
        max_distance: Maximum token distance for mode="adjacent".
        include_self: Include diagonal counts.
        symmetric: If True, keep a symmetric matrix.

    Returns:
        (matrix, labels)
    """
    labels = _build_vocabulary(sequences=sequences, vocabulary=vocabulary)
    token_to_idx = {token: idx for idx, token in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)

    if mode not in {"same_sequence", "adjacent"}:
        raise ValueError("mode must be 'same_sequence' or 'adjacent'")
    if mode == "adjacent" and max_distance < 1:
        raise ValueError("max_distance must be >= 1 for adjacent mode")

    if mode == "same_sequence":
        for sequence in sequences:
            unique_idxs: List[int] = []
            present = set()
            for token in sequence:
                idx = token_to_idx.get(token)
                if idx is None or idx in present:
                    continue
                present.add(idx)
                unique_idxs.append(idx)

            if include_self:
                for idx in unique_idxs:
                    matrix[idx, idx] += 1

            for i in range(len(unique_idxs)):
                idx_i = unique_idxs[i]
                for j in range(i + 1, len(unique_idxs)):
                    idx_j = unique_idxs[j]
                    matrix[idx_i, idx_j] += 1
                    matrix[idx_j, idx_i] += 1
        return matrix, labels

    for sequence in sequences:
        idx_sequence = [token_to_idx[token] for token in sequence if token in token_to_idx]
        n = len(idx_sequence)
        for i in range(n):
            idx_i = idx_sequence[i]
            j_max = min(n, i + max_distance + 1)
            for j in range(i + 1, j_max):
                idx_j = idx_sequence[j]
                if not include_self and idx_i == idx_j:
                    continue
                matrix[idx_i, idx_j] += 1
                if symmetric:
                    matrix[idx_j, idx_i] += 1

    return matrix, labels


def top_cooccurring_pairs(
    matrix: np.ndarray,
    labels: Sequence[Symbol],
    top_k: int = 20,
    exclude_diagonal: bool = True,
) -> List[Tuple[Symbol, Symbol, int]]:
    """
    Return the top co-occurring token pairs from a matrix.
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    if matrix.shape[0] != len(labels):
        raise ValueError("labels length must match matrix size")

    results: List[Tuple[Symbol, Symbol, int]] = []
    n = matrix.shape[0]
    start_j = 1 if exclude_diagonal else 0
    for i in range(n):
        for j in range(i + start_j, n):
            if exclude_diagonal and i == j:
                continue
            value = int(matrix[i, j])
            if value <= 0:
                continue
            results.append((labels[i], labels[j], value))

    results.sort(key=lambda row: row[2], reverse=True)
    return results[:top_k]


def grouped_label_order(
    labels: Sequence[Symbol],
    group_key: Optional[Callable[[Symbol], str]] = None,
) -> Tuple[List[int], List[Tuple[str, int, int]]]:
    """
    Return indices that group labels and boundary metadata.

    Groups are defined from label prefixes by default (first character, uppercased).
    """
    if group_key is None:
        def group_key(token: Symbol) -> str:
            text = str(token)
            return text[0].upper() if text else "?"

    decorated: List[Tuple[str, str, int]] = []
    for idx, label in enumerate(labels):
        key = group_key(label)
        decorated.append((key, str(label), idx))

    decorated.sort(key=lambda x: (x[0], x[1]))
    order = [idx for _, _, idx in decorated]

    groups: List[Tuple[str, int, int]] = []
    if not decorated:
        return order, groups

    current_group = decorated[0][0]
    start = 0
    for pos, (group, _, _) in enumerate(decorated):
        if group != current_group:
            groups.append((current_group, start, pos))
            current_group = group
            start = pos
    groups.append((current_group, start, len(decorated)))
    return order, groups


def plot_grouped_cooccurrence_matrix(
    matrix: np.ndarray,
    labels: Sequence[Symbol],
    title: str = "Token Co-occurrence (Grouped)",
    max_ticks: int = 60,
    cmap: str = "YlOrRd",
    log_scale: bool = True,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot a grouped co-occurrence heatmap with visible token clusters.
    """
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    if matrix.shape[0] != len(labels):
        raise ValueError("labels length must match matrix size")

    order, groups = grouped_label_order(labels)
    ordered_matrix = matrix[np.ix_(order, order)]
    ordered_labels = [str(labels[i]) for i in order]

    data = np.log1p(ordered_matrix) if log_scale else ordered_matrix

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 9))

    try:
        import seaborn as sns
        sns.heatmap(data, cmap=cmap, cbar=True, square=True, ax=ax)
    except ModuleNotFoundError:  # pragma: no cover
        img = ax.imshow(data, cmap=cmap)
        plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04)

    n = len(ordered_labels)
    if n <= max_ticks:
        ticks = np.arange(n) + 0.5
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(ordered_labels, rotation=90, fontsize=7)
        ax.set_yticklabels(ordered_labels, fontsize=7)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    for _, start, _ in groups:
        if start > 0:
            ax.axhline(start, color="white", linewidth=1.2)
            ax.axvline(start, color="white", linewidth=1.2)

    for group, start, end in groups:
        center = (start + end) / 2.0
        ax.text(center, -0.6, group, ha="center", va="bottom", fontsize=9, transform=ax.transData)
        ax.text(-0.6, center, group, ha="right", va="center", fontsize=9, transform=ax.transData)

    suffix = " (log1p)" if log_scale else ""
    ax.set_title(f"{title}{suffix}")
    ax.set_xlabel("Tokens (grouped)")
    ax.set_ylabel("Tokens (grouped)")
    return ax
