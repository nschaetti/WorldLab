"""Paired and rotation-free metrics for the Platonic Representation Hypothesis."""

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


def _paired(a, b, minimum=2):
    if (a.ndim != 2 or b.ndim != 2 or len(a) != len(b) or len(a) < minimum
            or min(a.size(1), b.size(1)) < 1):
        raise ValueError(f"Expected paired feature matrices with at least {minimum} rows")


def _orthogonal(a, b):
    u, _, vh = torch.linalg.svd(a.T @ b, full_matrices=False)
    return u @ vh


@dataclass
class Rotation:
    """Centred Procrustes map. Unequal widths give a partial isometry."""

    matrix: Tensor
    source_mean: Tensor
    target_mean: Tensor

    @classmethod
    def fit(cls, source, target):
        _paired(source, target)
        a, b = source.double(), target.double()
        mean_a, mean_b = a.mean(0), b.mean(0)
        return cls(_orthogonal(a - mean_a, b - mean_b), mean_a, mean_b)

    def __call__(self, source):
        return (source.to(self.matrix) - self.source_mean) @ self.matrix + self.target_mean


def _neighbors(features, k, block_size):
    features = F.normalize(features, dim=1)
    indices = []
    for start in range(0, len(features), block_size):
        scores = features[start:start + block_size] @ features.T
        rows = torch.arange(len(scores), device=scores.device)
        scores[rows, start + rows] = -torch.inf
        indices.append(scores.topk(k, dim=1).indices)
    return torch.cat(indices)


def _overlap(a, b):
    # Each neighbour appears once. Comparison uses O(N k²) memory.
    return (a[:, :, None] == b[:, None, :]).any(-1).float().mean().item()


def mutual_knn(a, b, k=10, *, block_size=1024):
    """Mean fraction of shared cosine neighbours, excluding each query itself."""
    _paired(a, b)
    if min(k, block_size) < 1:
        raise ValueError("k and block_size must be positive")
    k = min(k, len(a) - 1)
    return _overlap(_neighbors(a, k, block_size), _neighbors(b, k, block_size))


def _recall(a, b, block_size):
    a, b = F.normalize(a, dim=1), F.normalize(b, dim=1)
    counts = torch.zeros(2, device=a.device, dtype=torch.int64)
    for start in range(0, len(a), block_size):
        scores = a[start:start + block_size] @ b.T
        rows = torch.arange(len(scores), device=scores.device)
        # Ties count against the correct pair, including collapsed encoders.
        ranks = (scores >= scores[rows, start + rows, None]).sum(1)
        counts += torch.stack(((ranks <= 1).sum(), (ranks <= 5).sum()))
    return (counts.double() / len(a)).tolist()


def _whitener(centered):
    covariance = centered.T @ centered / (len(centered) - 1)
    eye = torch.eye(len(covariance), dtype=covariance.dtype, device=covariance.device)
    covariance = 0.9 * covariance + 0.1 * covariance.diagonal().mean() * eye
    values, vectors = torch.linalg.eigh(covariance)
    floor = torch.finfo(values.dtype).eps * values.abs().max().clamp_min(1)
    return (vectors * values.clamp_min(floor).rsqrt()) @ vectors.T


def paired_alignment(fit_a, fit_b, test_a, test_b, *, k=10, block_size=1024, test_means=False):
    """Fit one rotation on the fit pairs and score separate test pairs."""
    _paired(fit_a, fit_b)
    _paired(test_a, test_b)
    if fit_a.size(1) != test_a.size(1) or fit_b.size(1) != test_b.size(1):
        raise ValueError("Fit and test feature widths must match for each modality")
    rotation = Rotation.fit(fit_a, fit_b)
    va, vb = fit_a.double() - rotation.source_mean, fit_b.double() - rotation.target_mean
    ta = test_a.double() - (test_a.double().mean(0) if test_means else rotation.source_mean)
    tb = test_b.double() - (test_b.double().mean(0) if test_means else rotation.target_mean)
    rotated = ta @ rotation.matrix
    r1, r5 = _recall(rotated, tb, block_size)
    wa, wb = _whitener(va), _whitener(vb)
    wr1, wr5 = _recall(ta @ wa @ _orthogonal(va @ wa, vb @ wb), tb @ wb, block_size)
    return {
        "mknn": mutual_knn(ta, tb, k, block_size=block_size),
        "R@1": r1, "R@5": r5, "wR@1": wr1, "wR@5": wr5,
        "cos": F.cosine_similarity(rotated, tb, dim=1).mean().item(),
        "fit_pairs": len(va), "test_pairs": len(ta),
    }, rotation


def _prepare(features):
    features = features.float()
    flat = features.abs().flatten()
    quantile = torch.quantile(flat[::max(1, (len(flat) + 9_999_999) // 10_000_000)], 0.95)
    return F.normalize(features.clamp(-quantile, quantile), dim=1)


def _u_centered_gram(features):
    gram = (features.double() @ features.double().T).fill_diagonal_(0)
    n = len(gram)
    sums = gram.sum(1)
    centered = gram - (sums[:, None] + sums[None, :]) / (n - 2) + sums.sum() / ((n - 1) * (n - 2))
    return centered.fill_diagonal_(0)


def prh_score(layers_a, layers_b, *, k=10, metric="mknn", block_size=1024):
    """Max over layer pairs after clipping at the 95th absolute percentile."""
    if metric not in {"mknn", "cka"} or not layers_a or not layers_b or min(k, block_size) < 1:
        raise ValueError("Expected layers, metric mknn or cka, and positive k and block_size")
    minimum = 4 if metric == "cka" else 2
    for layer in (*layers_a, *layers_b):
        _paired(layers_a[0], layer, minimum)
    if metric == "mknn":
        k = min(k, len(layers_a[0]) - 1)
        a = [_neighbors(_prepare(layer), k, block_size) for layer in layers_a]
        b = [_neighbors(_prepare(layer), k, block_size) for layer in layers_b]
        return max((_overlap(x, y), i, j) for i, x in enumerate(a) for j, y in enumerate(b))

    def normalized_gram(features):
        gram = _u_centered_gram(_prepare(features))
        return F.normalize(gram.flatten(), dim=0)

    # Hold one source Gram at a time, so memory does not scale with depth.
    best = (-float("inf"), 0, 0)
    for i, layer in enumerate(layers_a):
        source = normalized_gram(layer)
        for j, target in enumerate(layers_b):
            best = max(best, (torch.dot(source, normalized_gram(target)).item(), i, j))
    return best


def scene_explained_fraction(features, k):
    """Estimate tr(sqrt(C))/k after PCA whitening and finite-view correction.

    Input is (scenes, observations, features), with the same number of
    observations per scene. See Theorem 3 and Appendix C of the PRH paper.
    """
    if features.ndim != 3 or min(features.shape[:2]) < 2 or not 1 <= k <= features.size(-1):
        raise ValueError("Expected at least two scenes and two observations, with valid k")
    n, views, _ = features.shape
    flat = features.flatten(0, 1).double()
    flat = flat - flat.mean(0)
    values, vectors = torch.linalg.eigh(flat.T @ flat / len(flat))
    if values[-k] <= values[-1] * 1e-12:
        raise ValueError("The selected dimensions exceed the effective feature rank")
    white = ((flat @ vectors[:, -k:]) / values[-k:].sqrt()).reshape(n, views, k)
    means = white.mean(1)
    residual = (white - means[:, None]).flatten(0, 1)
    corrected = means.T @ means / n - residual.T @ residual / (n * (views - 1) * views)
    return torch.linalg.eigvalsh(corrected).clamp(0, 1).sqrt().mean().item()
