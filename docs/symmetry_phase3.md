# Symmetry Reproduction - Phase 3

Phase 3 connects synthetic worlds and theory diagnostics to actual GPT
representations.

## Goal

Train a GPT on synthetic sequences, extract token-level hidden
representations, and evaluate whether the learned geometry follows the expected
distance-based structure.

## New modules

- `boardGPT/analysis/synthetic_gpt.py`
  - tiny GPT builder for synthetic experiments,
  - minimal next-token training loop.

- `boardGPT/analysis/representations.py`
  - token-level representation extraction from layer residuals,
  - Gram/PCA bundle builder,
  - symmetry report on representation Gram matrices.

## Core workflow

1. Generate synthetic sequences from a known policy (Phase 1).
2. Train a compact GPT on those sequences.
3. Extract mean representation per token at a chosen layer.
4. Build Gram/PCA and compute translation-symmetry diagnostics.
5. Compare with fitted-kernel theory from Phase 2.

## Important interpretation choice

`extract_mean_token_representations_from_model(...)` aggregates hidden vectors
by **input token identity** (which token is observed at that position).

This is useful to study whether token identity embeddings become geometrically
organized by latent space under a given sequence policy.

## Suggested first experiment

- World: 1D periodic ring,
- Policy: local exponential transitions,
- Expectation: top PCA coordinates resemble sinusoidal/circular organization.
