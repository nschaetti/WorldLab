"""
Synthetic sequence generation from explicit probability policies.

This module is designed for controlled experiments where the sequence law is
known by construction. It lets you test when a GPT can recover latent 1D/2D/3D
geometry from co-occurrence structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .concepts import ConceptSpec, pairwise_distances


@dataclass(frozen=True)
class SyntheticWorld:
    """
    Synthetic token world with latent coordinates.

    Attributes:
        concept: Geometry of tokens.
        token_names: Optional human-readable names, shape ``[V]``.
    """

    concept: ConceptSpec
    token_names: Optional[list[str]] = None

    @property
    def vocab_size(self) -> int:
        return int(self.concept.num_tokens)


@dataclass(frozen=True)
class PolicyConfig:
    """
    Policy hyperparameters for synthetic sequence generation.

    The next-token logits are composed from multiple optional terms:

    1) Transition kernel from current token (Markov term),
    2) Attraction to previously visited tokens (history growth term),
    3) Optional directional drift in latent space,
    4) Exploration mixing with uniform distribution.

    Args:
        length: Number of tokens per sequence.
        num_sequences: Number of sequences.
        seed: RNG seed.
        first_token_logits: Optional start logits over tokens (shape ``[V]``).
        transition_kernel: One of ``{"none", "exponential", "gaussian", "power_law"}``.
        transition_scale: Scale parameter for kernels (sigma-like).
        transition_power: Exponent used by ``power_law`` kernel.
        transition_strength: Weight of transition logits.
        hard_transition_radius: If provided, tokens farther than this radius from
            current token have zero probability.
        history_strength: Weight of attraction to visited-token set.
        history_scale: Scale for history kernel.
        history_hard_radius: If provided and history is non-empty, only tokens
            within this distance of *some* visited token are allowed.
        history_kernel: Kernel type for history attraction.
        drift_vector: Optional latent drift direction (shape ``[D]``).
        drift_strength: Weight of directional drift.
        temperature: Softmax temperature.
        epsilon_uniform: Uniform exploration mixing in ``[0,1)``.
    """

    length: int
    num_sequences: int
    seed: int = 0
    first_token_logits: Optional[np.ndarray] = None
    transition_kernel: str = "exponential"
    transition_scale: float = 0.25
    transition_power: float = 2.0
    transition_strength: float = 1.0
    hard_transition_radius: Optional[float] = None
    history_strength: float = 0.0
    history_scale: float = 0.25
    history_hard_radius: Optional[float] = None
    history_kernel: str = "exponential"
    drift_vector: Optional[np.ndarray] = None
    drift_strength: float = 0.0
    temperature: float = 1.0
    epsilon_uniform: float = 0.0


@dataclass
class SyntheticSequenceDataset:
    """
    Generated synthetic sequences and generation metadata.

    Attributes:
        sequences: Integer token sequences, shape ``[N, T]``.
        first_token_probs: Start distribution used by the generator, shape ``[V]``.
        transition_probs: Base Markov transition matrix from transition term only,
            shape ``[V, V]``.
        config: Policy used for generation.
        world: World used for generation.
    """

    sequences: np.ndarray
    first_token_probs: np.ndarray
    transition_probs: np.ndarray
    config: PolicyConfig
    world: SyntheticWorld


def _validate_config(world: SyntheticWorld, cfg: PolicyConfig) -> None:
    if cfg.length <= 0:
        raise ValueError(f"length must be > 0, got {cfg.length}")
    if cfg.num_sequences <= 0:
        raise ValueError(f"num_sequences must be > 0, got {cfg.num_sequences}")
    if cfg.temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {cfg.temperature}")
    if not (0.0 <= cfg.epsilon_uniform < 1.0):
        raise ValueError(f"epsilon_uniform must be in [0,1), got {cfg.epsilon_uniform}")

    if cfg.first_token_logits is not None:
        first = np.asarray(cfg.first_token_logits, dtype=np.float64)
        if first.shape != (world.vocab_size,):
            raise ValueError(
                f"first_token_logits must have shape ({world.vocab_size},), got {first.shape}"
            )

    if cfg.drift_vector is not None:
        drift = np.asarray(cfg.drift_vector, dtype=np.float64)
        if drift.shape != (world.concept.dim,):
            raise ValueError(
                f"drift_vector must have shape ({world.concept.dim},), got {drift.shape}"
            )


def _kernel(distance: np.ndarray, kind: str, scale: float, power: float = 2.0) -> np.ndarray:
    if scale <= 0:
        raise ValueError(f"scale must be > 0, got {scale}")

    if kind == "none":
        return np.zeros_like(distance, dtype=np.float64)
    if kind == "exponential":
        return np.exp(-distance / scale)
    if kind == "gaussian":
        return np.exp(-0.5 * (distance / scale) ** 2)
    if kind == "power_law":
        return 1.0 / (1.0 + (distance / scale) ** power)
    raise ValueError(f"unknown kernel kind: {kind}")


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    x = logits / temperature
    x = x - np.max(x)
    ex = np.exp(x)
    s = np.sum(ex)
    if s <= 0 or not np.isfinite(s):
        raise RuntimeError("invalid softmax normalization")
    return ex / s


def _safe_distribution(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = float(np.sum(probs))
    if total <= 0:
        probs = np.ones_like(probs) / probs.size
        return probs
    return probs / total


def _build_first_token_probs(world: SyntheticWorld, cfg: PolicyConfig) -> np.ndarray:
    if cfg.first_token_logits is None:
        probs = np.ones((world.vocab_size,), dtype=np.float64) / world.vocab_size
    else:
        probs = _softmax(np.asarray(cfg.first_token_logits, dtype=np.float64), temperature=cfg.temperature)
    if cfg.epsilon_uniform > 0:
        uniform = np.ones_like(probs) / probs.size
        probs = (1.0 - cfg.epsilon_uniform) * probs + cfg.epsilon_uniform * uniform
    return _safe_distribution(probs)


def _build_transition_logits(world: SyntheticWorld, cfg: PolicyConfig) -> np.ndarray:
    dist = pairwise_distances(world.concept)
    base = _kernel(
        distance=dist,
        kind=cfg.transition_kernel,
        scale=cfg.transition_scale,
        power=cfg.transition_power,
    )
    logits = cfg.transition_strength * base

    if cfg.hard_transition_radius is not None:
        logits = np.where(dist <= cfg.hard_transition_radius, logits, -np.inf)

    if cfg.drift_vector is not None and cfg.drift_strength != 0.0:
        coords = world.concept.coordinates
        drift = np.asarray(cfg.drift_vector, dtype=np.float64)
        drift = drift / (np.linalg.norm(drift) + 1e-12)
        delta = coords[None, :, :] - coords[:, None, :]
        directional = np.tensordot(delta, drift, axes=([2], [0]))
        logits = logits + cfg.drift_strength * directional

    return logits


def _history_logits(
    world: SyntheticWorld,
    cfg: PolicyConfig,
    visited: np.ndarray,
) -> np.ndarray:
    if cfg.history_strength == 0.0 or visited.size == 0:
        return np.zeros((world.vocab_size,), dtype=np.float64)

    coords = world.concept.coordinates
    visited_coords = coords[visited]
    diff = coords[:, None, :] - visited_coords[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    min_dist = np.min(dist, axis=1)

    score = _kernel(
        distance=min_dist,
        kind=cfg.history_kernel,
        scale=cfg.history_scale,
        power=cfg.transition_power,
    )
    logits = cfg.history_strength * score

    if cfg.history_hard_radius is not None:
        logits = np.where(min_dist <= cfg.history_hard_radius, logits, -np.inf)

    return logits


def _rowwise_softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    probs = np.zeros_like(logits, dtype=np.float64)
    for i in range(logits.shape[0]):
        row = logits[i]
        finite = np.isfinite(row)
        if not np.any(finite):
            probs[i] = np.ones((row.shape[0],), dtype=np.float64) / row.shape[0]
            continue
        masked = np.where(finite, row, -1e12)
        probs[i] = _softmax(masked, temperature=temperature)
    return probs


def generate_synthetic_sequences(
    world: SyntheticWorld,
    config: PolicyConfig,
) -> SyntheticSequenceDataset:
    """
    Generate synthetic sequences under an explicit probability policy.

    Educational interpretation of the policy
    ----------------------------------------
    At each step, token probabilities are assembled from interpretable terms:

    ``logits_t = transition(current_token) + history_attraction(visited_set) + drift``

    then turned into a distribution with softmax and optional uniform exploration.

    This allows controlled experiments like:
    - force local growth around visited tokens,
    - create periodic 1D random walks,
    - induce anisotropic 2D/3D motion,
    - test robustness to noise.
    """
    _validate_config(world=world, cfg=config)
    rng = np.random.default_rng(config.seed)

    first_probs = _build_first_token_probs(world=world, cfg=config)
    transition_logits = _build_transition_logits(world=world, cfg=config)
    transition_probs = _rowwise_softmax(transition_logits, temperature=config.temperature)

    sequences = np.zeros((config.num_sequences, config.length), dtype=np.int64)

    for n in range(config.num_sequences):
        first_tok = int(rng.choice(world.vocab_size, p=first_probs))
        sequences[n, 0] = first_tok
        visited = [first_tok]

        for t in range(1, config.length):
            prev = int(sequences[n, t - 1])

            logits = np.array(transition_logits[prev], copy=True)
            logits += _history_logits(
                world=world,
                cfg=config,
                visited=np.asarray(visited, dtype=np.int64),
            )

            finite = np.isfinite(logits)
            if not np.any(finite):
                probs = np.ones((world.vocab_size,), dtype=np.float64) / world.vocab_size
            else:
                logits = np.where(finite, logits, -1e12)
                probs = _softmax(logits, temperature=config.temperature)

            if config.epsilon_uniform > 0:
                uniform = np.ones_like(probs) / probs.size
                probs = (1.0 - config.epsilon_uniform) * probs + config.epsilon_uniform * uniform
                probs = _safe_distribution(probs)

            token_t = int(rng.choice(world.vocab_size, p=probs))
            sequences[n, t] = token_t
            visited.append(token_t)

    return SyntheticSequenceDataset(
        sequences=sequences,
        first_token_probs=first_probs,
        transition_probs=transition_probs,
        config=config,
        world=world,
    )
