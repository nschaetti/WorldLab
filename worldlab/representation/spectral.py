"""Finite-world spectral JEPA and the kernel mismatch bound (PRH §2)."""

import math

import torch


def _kernel(pairs):
    pairs = torch.as_tensor(pairs, dtype=torch.float64)
    if pairs.ndim != 2 or pairs.size(0) != pairs.size(1) or pairs.size(0) < 2:
        raise ValueError("Expected a square positive-pair distribution")
    if (not torch.isfinite(pairs).all() or (pairs < 0).any()
            or not torch.allclose(pairs, pairs.T) or not torch.allclose(pairs.sum(), pairs.new_tensor(1))):
        raise ValueError("Positive pairs must be nonnegative, symmetric and sum to one")
    p = pairs.sum(1)
    if (p <= 0).any():
        raise ValueError("Each world state must have positive probability")
    root = p.sqrt()
    return pairs / (root[:, None] * root[None, :]), p


def spectral_solution(pairs, k):
    """Top k nonconstant eigenfunctions, the whitened squared-loss JEPA solution."""
    kernel, p = _kernel(pairs)
    if not 1 <= k < len(p):
        raise ValueError("Expected 1 <= k < number of states")
    root = p.sqrt()
    # Remove only the constant direction, including for disconnected worlds.
    basis = torch.linalg.qr(root[:, None], mode="complete").Q[:, 1:]
    values, vectors = torch.linalg.eigh(basis.T @ kernel @ basis)
    return (basis @ vectors[:, -k:].flip(1)) / root[:, None], values.flip(0)


def _weighted_distance(a, b, p):
    u, _, vh = torch.linalg.svd(a.T @ (p[:, None] * b))
    return (p @ ((a @ (u @ vh) - b).square().sum(1))).sqrt().item()


def mismatch_bound(pairs_a, pairs_b, k):
    """Return actual distance, spectral gap and Theorem 2's upper bound."""
    ka, p = _kernel(pairs_a)
    kb, q = _kernel(pairs_b)
    if p.shape != q.shape or not torch.allclose(p, q):
        raise ValueError("The bound requires a common world marginal")
    a, _ = spectral_solution(pairs_a, k)
    b, eigenvalues = spectral_solution(pairs_b, k)
    distance = _weighted_distance(a, b, p)
    if k == len(p) - 1:
        return distance, math.inf, 0.0
    gap = (eigenvalues[k - 1] - eigenvalues[k]).item()
    delta = ka - kb
    perturbation = min(math.sqrt(k) * torch.linalg.matrix_norm(delta, ord=2).item(),
                       torch.linalg.matrix_norm(delta).item())
    bound = 2 * math.sqrt(2) * perturbation / gap if gap > 0 else math.inf
    return distance, gap, bound


def coverage_bound(observation_pairs, scene_ids, k, *, tolerance=1e-9):
    """Bound the deviation from the scene embedding for arbitrary positive views.

    The normalized observation kernel splits into functions of the scene and
    nuisance functions with zero conditional mean. The bound requires a
    spectral gap separating the retained scene directions from both blocks.
    """
    kernel, rho = _kernel(observation_pairs)
    scene_ids = torch.as_tensor(scene_ids, device=kernel.device)
    if scene_ids.shape != rho.shape:
        raise ValueError("Provide one scene ID per observation")
    labels, indices = torch.unique(scene_ids, return_inverse=True)
    scenes = len(labels)
    if not 1 <= k < scenes:
        raise ValueError("Expected 1 <= k < number of scenes")
    p = torch.zeros(scenes, dtype=rho.dtype, device=rho.device).scatter_add_(0, indices, rho)
    indicators = torch.nn.functional.one_hot(indices, scenes).to(rho)
    scene_basis = indicators * (rho / p[indices]).sqrt()[:, None]
    nuisance_basis = torch.linalg.qr(scene_basis, mode="complete").Q[:, scenes:]

    scene_pairs = indicators.T @ torch.as_tensor(observation_pairs).to(kernel) @ indicators
    target, eigenvalues = spectral_solution(scene_pairs, k)
    embedding, _ = spectral_solution(observation_pairs, k)
    actual = _weighted_distance(embedding, target[indices], rho)

    next_scene = eigenvalues[k].item() if k < scenes - 1 else -math.inf
    if nuisance_basis.size(1):
        coupling = nuisance_basis.T @ kernel @ scene_basis
        nuisance = torch.linalg.eigvalsh(nuisance_basis.T @ kernel @ nuisance_basis)[-1].item()
        op = torch.linalg.matrix_norm(coupling, ord=2).item()
        hs = torch.linalg.matrix_norm(coupling).item()
    else:
        nuisance, op, hs = -math.inf, 0.0, 0.0
    gap = eigenvalues[k - 1].item() - max(next_scene, nuisance)
    bound = 2 * math.sqrt(2) * min(math.sqrt(k) * op, math.sqrt(2) * hs) / gap if gap > tolerance else math.inf
    return {"distance": actual, "gap": gap, "bound": bound, "nuisance": nuisance,
            "coupling_operator": op, "coupling_hilbert_schmidt": hs}
