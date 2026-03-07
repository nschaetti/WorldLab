# Symmetry Reproduction - Phase 2

Phase 2 adds the first theory-vs-data comparison layer on top of Phase 1.

## Goal

Given a matrix of interest (typically `M*`, PMI, or representation Gram matrix)
and a concept distance matrix, we want to answer:

1. **How translation-symmetric is this matrix?**
2. **Can an exponential kernel explain the observed profile?**
3. **What geometry is induced by the corresponding Gram matrix?**

## New module

- `boardGPT/analysis/theory_karkada.py`

### Main APIs

- `distance_binned_profile(matrix, distances, ...)`
  - Builds `mean(matrix_ij)` as a function of distance bins.

- `translation_symmetry_diagnostic(matrix, distances, ...)`
  - Reconstructs entries from distance-only profile,
  - Returns MSE and `R^2` (variance explained by distance-only model).

- `fit_exponential_profile(distances, values, ...)`
  - Fits `offset + amplitude * exp(-d/sigma)`.

- `fit_exponential_kernel_from_matrix(matrix, distances, ...)`
  - Convenience wrapper: profile + exponential fit.

- `build_exponential_kernel_matrix(distances, amplitude, sigma, offset)`
  - Builds model kernel matrix from fitted parameters.

- `pca_from_gram(gram, rank, center)`
  - Extracts PCA coordinates from Gram matrix by eigendecomposition.

## Reading the outputs

- **High symmetry `R^2`** in `translation_symmetry_diagnostic` means matrix values
  are mostly controlled by distance alone.
- **High fit `R^2`** in `fit_exponential_profile` means an exponential kernel is
  a good low-parameter description.
- **PCA from fitted kernel Gram** can be directly compared to empirical PCA.

## Why this matters for your project

This gives a direct bridge between:

- **data statistics** (co-occurrences),
- **theory objects** (distance kernels),
- **model geometry** (PCA manifolds from Gram matrices).

It is exactly the core methodology needed before extending to layer-wise GPT
representations and full figure reproduction.
