# Symmetry Reproduction - Phase 1

This document describes the first implementation block for reproducing and
extending the analysis from Karkada et al. (2026), with an educational focus.

## Goal of Phase 1

Build three reusable foundations:

1. **Co-occurrence statistics** (`P_ij`, `P_i`, PMI, `M*`),
2. **Latent concept geometry** (token coordinates and distances),
3. **Synthetic sequence worlds** from explicit probability policies.

Phase 1 does **not** yet include the full theoretical fitting/closed-form mode
prediction layer. It prepares all required inputs for that layer.

## New modules

- `boardGPT/analysis/cooccurrence.py`
  - Weighted skip-gram counting,
  - Probability normalization,
  - PMI and `M*` computation.

- `boardGPT/analysis/concepts.py`
  - `ConceptSpec` dataclass,
  - 1D and ND lattice concept builders,
  - open/periodic pairwise distances.

- `boardGPT/analysis/synthetic_sequences.py`
  - `SyntheticWorld` and `PolicyConfig`,
  - custom policy-driven sequence generation,
  - optional history-growth behavior ("close to already visited tokens").

## Educational interpretation

### 1) Co-occurrence block

Given integer token sequences, we compute:

- `counts_ij`: weighted pair counts in a context window,
- `P_ij = counts_ij / sum(counts_ij)`,
- `P_i = sum_j P_ij`,
- `PMI_ij = log((P_ij + eps)/(P_i P_j + eps))`,
- `M*_ij = 2*(P_ij - P_iP_j)/(P_ij + P_iP_j + eps)`.

This mirrors the central statistics used in the paper.

### 2) Concept block

`ConceptSpec` defines a geometric interpretation of tokens:

- which tokens are in concept `S`,
- coordinates `x_i` in dimension `D`,
- boundary conditions (open or periodic per axis).

Distance matrices are then used to test translation symmetry and fit kernels.

### 3) Synthetic-world block

You can generate datasets from custom policies, independent of Othello rules.

At each step:

`logits_t = transition(current_token) + history_attraction(visited_set) + drift`

Then apply softmax (+ optional uniform exploration).

This supports experiments like:

- ring/line/grid random walks,
- hard local growth around already visited tokens,
- anisotropic trajectories,
- noisy vs clean co-occurrence structure.

## Minimal usage

```python
from boardGPT.analysis import (
    lattice_nd_concept,
    SyntheticWorld,
    PolicyConfig,
    generate_synthetic_sequences,
    build_cooccurrence_statistics,
)

concept = lattice_nd_concept(shape=(8, 8), periodic=(False, False), name="grid2d")
world = SyntheticWorld(concept=concept)
cfg = PolicyConfig(
    length=60,
    num_sequences=1000,
    seed=0,
    transition_kernel="exponential",
    transition_scale=0.2,
    history_strength=0.8,
    history_hard_radius=0.35,
)

dataset = generate_synthetic_sequences(world, cfg)
stats = build_cooccurrence_statistics(dataset.sequences, vocab_size=world.vocab_size)
```

## What comes in Phase 2

- translation-symmetry diagnostics,
- exponential-kernel fitting (open/periodic),
- theoretical mode prediction and direct comparison plots.
