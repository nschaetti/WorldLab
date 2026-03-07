# Symmetry Math Notes

Compact reference of the main quantities used in the implementation.

## Co-occurrence statistics

Given weighted skip-gram counts `C_ij`:

- `P_ij = C_ij / sum_{a,b} C_ab`
- `P_i = sum_j P_ij`
- `PMI_ij = log((P_ij + eps) / (P_i P_j + eps))`
- `M*_ij = 2 * (P_ij - P_i P_j) / (P_ij + P_i P_j + eps)`

## Translation symmetry diagnostic

Given matrix `A_ij` and distances `d_ij`:

1. Bin entries by distance,
2. Replace each `A_ij` by mean value of its bin,
3. Report:
   - `MSE = E[(A_ij - Ahat_ij)^2]`,
   - `R^2 = 1 - MSE / Var(A_ij)`.

High `R^2` means values are mostly controlled by distance alone.

## Exponential kernel fit

Fit profile `y(d)` with:

- `y(d) = b + a * exp(-d / sigma)`.

Implementation uses grid search over `sigma` and least-squares solve for
`(b, a)` at each grid point.

## Gram -> PCA geometry

From Gram matrix `G` (optionally centered):

- eigendecompose `G = U diag(lambda) U^T`,
- keep positive eigenvalues,
- coordinates `W = U diag(sqrt(lambda))`.

Rows of `W` are token coordinates in PCA space.

## Linear decoding

Given features `X in R^{N x r}` and targets `Y in R^{N x D}`:

- ridgeless: `W = pinv(X) Y`,
- ridge: `W = (X^T X + alpha I)^(-1) X^T Y`.

Metrics:

- `MSE = E||Y - XW||^2`,
- `relative MSE = E||Y - XW||^2 / (E||Y||^2 + eps)`.

## Helper reconstruction

1. Ablate target-target block in matrix,
2. Select helper tokens (top coupling or random),
3. Factorize target+helpers submatrix,
4. Compare reconstructed target geometry with reference using:
   - Procrustes error,
   - Gram relative MSE.
