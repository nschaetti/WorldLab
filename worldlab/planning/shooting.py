"""Optimize action blocks through a batched scalar terminal cost."""

import torch


@torch.no_grad()
def cem(cost, initial, *, samples=300, elites=30, iterations=30, variance=1.0, generator=None):
    """Diagonal Gaussian CEM. Initial mean is (B, horizon, action_dim)."""
    if not 2 <= elites <= samples or iterations < 1 or variance <= 0:
        raise ValueError("Expected 2 <= elites <= samples, positive iterations and positive variance")
    mean = initial
    std = torch.full_like(mean, variance**0.5)
    for _ in range(iterations):
        noise = torch.randn((mean.size(0), samples, *mean.shape[1:]),
                            device=mean.device, dtype=mean.dtype, generator=generator)
        candidates = mean[:, None] + std[:, None] * noise
        candidates[:, 0] = mean
        scores = cost(candidates)
        indices = scores.topk(elites, dim=1, largest=False).indices
        best = candidates[torch.arange(mean.size(0), device=mean.device)[:, None], indices]
        mean = best.mean(dim=1)
        std = best.std(dim=1).clamp_min(1e-6)
    return mean


@torch.enable_grad()
def gradient_plan(cost, initial, *, samples=100, iterations=30, lr=0.1, method="adam", generator=None):
    """Optimize parallel Gaussian restarts, then select the lowest-cost plan."""
    if samples < 1 or iterations < 1 or lr <= 0:
        raise ValueError("Expected positive samples, iterations and learning rate")
    optimizers = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW,
                  "sgd": torch.optim.SGD, "rmsprop": torch.optim.RMSprop}
    if method not in optimizers:
        raise ValueError(f"Unknown gradient optimizer: {method}")
    noise = torch.randn((initial.size(0), samples, *initial.shape[1:]),
                        device=initial.device, dtype=initial.dtype, generator=generator)
    actions = torch.nn.Parameter(initial.detach()[:, None] + noise)
    with torch.no_grad():
        actions[:, 0] = initial
    optimizer = optimizers[method]([actions], lr=lr)
    for _ in range(iterations):
        actions.grad, = torch.autograd.grad(cost(actions).sum(), actions)
        optimizer.step()
    with torch.no_grad():
        indices = cost(actions).argmin(dim=1)
        return actions[torch.arange(actions.size(0), device=actions.device), indices]
