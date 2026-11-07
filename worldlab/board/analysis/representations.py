"""
Representation extraction and geometry helpers (Phase 3).

This module bridges trained models and the symmetry analysis stack by providing:
- token-level representation aggregation,
- Gram/PCA computation,
- direct comparison against distance-based diagnostics from Phase 2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch

from worldlab.nn.board import GPT

from .theory_karkada import SymmetryDiagnostic, pca_from_gram, translation_symmetry_diagnostic


@dataclass(frozen=True)
class RepresentationBundle:
    """
    Standard representation geometry container.

    Attributes:
        representations: Token representations with shape ``[V, D]``.
        gram: Gram matrix ``R R^T`` of shape ``[V, V]``.
        pca_coords: PCA coordinates from Gram decomposition ``[V, r]``.
        pca_eigvals: Positive eigenvalues used in ``pca_coords``.
    """

    representations: np.ndarray
    gram: np.ndarray
    pca_coords: np.ndarray
    pca_eigvals: np.ndarray


@dataclass(frozen=True)
class RepresentationSymmetryReport:
    """
    Combined geometry/symmetry report for a representation set.

    Attributes:
        bundle: RepresentationBundle.
        symmetry: Distance-only diagnostic computed on Gram matrix.
    """

    bundle: RepresentationBundle
    symmetry: SymmetryDiagnostic


def _to_numpy(x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    raise TypeError(f"unsupported type: {type(x)}")


def compute_gram_matrix(representations: Union[np.ndarray, torch.Tensor], center: bool = False) -> np.ndarray:
    """
    Compute Gram matrix from representation rows.

    Args:
        representations: Matrix ``R`` with shape ``[V, D]``.
        center: If True, mean-center rows before Gram computation.

    Returns:
        ``G = R R^T`` with shape ``[V, V]``.
    """
    r = _to_numpy(representations).astype(np.float64)
    if r.ndim != 2:
        raise ValueError(f"representations must be 2D, got shape {r.shape}")
    if center:
        r = r - np.mean(r, axis=0, keepdims=True)
    return r @ r.T


def build_representation_bundle(
    representations: Union[np.ndarray, torch.Tensor],
    center_for_gram: bool = False,
    pca_rank: Optional[int] = None,
    pca_center_gram: bool = True,
) -> RepresentationBundle:
    """
    Build standardized representation geometry artifacts.
    """
    r = _to_numpy(representations).astype(np.float64)
    gram = compute_gram_matrix(r, center=center_for_gram)
    coords, eigvals, _ = pca_from_gram(gram=gram, rank=pca_rank, center=pca_center_gram)
    return RepresentationBundle(
        representations=r,
        gram=gram,
        pca_coords=coords,
        pca_eigvals=eigvals,
    )


@torch.no_grad()
def extract_mean_token_representations_from_model(
    model: GPT,
    sequences: Union[np.ndarray, torch.Tensor],
    layer: int,
    vocab_size: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    batch_size: int = 128,
    include_last_position: bool = False,
) -> np.ndarray:
    """
    Extract mean hidden representation per token ID from a GPT layer.

    Method:
    - run model and collect residuals at chosen layer,
    - for each token occurrence, aggregate its hidden vector,
    - return mean vector per token.

    Args:
        model: GPT model.
        sequences: Integer token matrix ``[N, T]``.
        layer: Residual layer index.
        vocab_size: Optional override for vocabulary size.
        device: Inference device.
        batch_size: Batch size for forward passes.
        include_last_position: If False, ignore position ``T-1`` since it has no
            next-token target in standard causal LM training.

    Returns:
        NumPy array of shape ``[V, D]``.
    """
    x = _to_numpy(sequences).astype(np.int64)
    if x.ndim != 2:
        raise ValueError(f"sequences must be [N,T], got shape {x.shape}")

    if vocab_size is None:
        vocab_size = int(model.config.vocab_size)
    if vocab_size <= 0:
        raise ValueError(f"vocab_size must be > 0, got {vocab_size}")

    if layer < 0 or layer >= model.config.n_layer:
        raise ValueError(f"layer must be in [0, {model.config.n_layer - 1}], got {layer}")

    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")

    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)

    was_training = model.training
    model.eval()

    d_model = int(model.config.n_embd)
    sums = np.zeros((vocab_size, d_model), dtype=np.float64)
    counts = np.zeros((vocab_size,), dtype=np.int64)

    # If sequences are one token longer than model context (typical next-token
    # setup with x=seq[:-1], y=seq[1:]), we can safely trim the final token while
    # preserving the intended set of supervised positions.
    force_include_last = False
    if x.shape[1] > model.config.block_size:
        if x.shape[1] == model.config.block_size + 1 and not include_last_position:
            x = x[:, :-1]
            force_include_last = True
        else:
            raise ValueError(
                f"sequence length {x.shape[1]} exceeds model block_size {model.config.block_size}"
            )

    n = x.shape[0]
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        batch = torch.from_numpy(x[start:stop]).long().to(device)
        _, _, obj = model(idx=batch, to_return=[f"residuals{layer}"])
        if len(obj) != 1:
            raise RuntimeError(f"expected one residual tensor, got {len(obj)}")

        residual = obj[0].detach().cpu().numpy()  # [B, T, D]
        toks = batch.detach().cpu().numpy()       # [B, T]

        effective_include_last = include_last_position or force_include_last
        t_limit = residual.shape[1] if effective_include_last else max(residual.shape[1] - 1, 0)
        for b in range(residual.shape[0]):
            for t in range(t_limit):
                tok = int(toks[b, t])
                if 0 <= tok < vocab_size:
                    sums[tok] += residual[b, t]
                    counts[tok] += 1

    if was_training:
        model.train()

    out = np.zeros_like(sums)
    nz = counts > 0
    out[nz] = sums[nz] / counts[nz, None]
    return out


def representation_symmetry_report(
    representations: Union[np.ndarray, torch.Tensor],
    distances: np.ndarray,
    pca_rank: Optional[int] = 8,
    num_bins: int = 32,
    center_for_gram: bool = False,
    pca_center_gram: bool = True,
    include_diagonal: bool = False,
) -> RepresentationSymmetryReport:
    """
    Compute combined representation bundle and translation-symmetry diagnostic.
    """
    bundle = build_representation_bundle(
        representations=representations,
        center_for_gram=center_for_gram,
        pca_rank=pca_rank,
        pca_center_gram=pca_center_gram,
    )
    sym = translation_symmetry_diagnostic(
        matrix=bundle.gram,
        distances=distances,
        num_bins=num_bins,
        include_diagonal=include_diagonal,
    )
    return RepresentationSymmetryReport(bundle=bundle, symmetry=sym)
