# Symmetry Reproduction - Phase 4

Phase 4 adds linear decoding curves as a function of projection rank.

## Goal

Measure how well latent variables (1D/2D/3D coordinates, mobility, etc.) can be
decoded from representation features when increasing the number of included PCA
components.

This is the experimental counterpart of the coordinate-decoding analyses in the
paper.

## New module

- `worldlab/board/analysis/decoding.py`

### Main APIs

- `fit_linear_decoder(x, y, ridge_alpha)`
  - Ridgeless (pseudo-inverse) when `ridge_alpha=0`,
  - Ridge closed form when `ridge_alpha>0`.

- `evaluate_linear_decoding_vs_rank(features, targets, ranks, ...)`
  - Random train/test splits,
  - mean/std train and test MSE curves,
  - relative MSE curves normalized by target power.

## Recommended usage pattern

1. Build token representations and PCA coordinates from Phase 3.
2. Select target latent coordinates (e.g., `x_i`, `(x_i, y_i)`).
3. Call `evaluate_linear_decoding_vs_rank(...)` for a range of ranks.
4. Compare:
   - ridgeless (`ridge_alpha=0`) to inspect interpolation effects,
   - ridge (`ridge_alpha>0`) for stabilized generalization.

## Interpretation

- Decreasing error with rank indicates information is progressively captured by
  additional modes.
- Under ridgeless fitting, a peak near interpolation threshold can appear.
- Relative MSE facilitates cross-task comparison (different target scales).
