# Symmetry Reproduction - Phase 5

Phase 5 implements collective-effect ablations and helper-based reconstruction.

## Goal

Replicate the key robustness idea:

- remove direct target-target couplings,
- recover target geometry using helper tokens,
- quantify how reconstruction improves with helper count.

## New module

- `worldlab/board/analysis/ablations.py`

### Main APIs

- `ablate_target_block(matrix, target_indices, value=0.0)`
  - zero (or overwrite) target-target block.

- `select_helpers_by_coupling(matrix, target_indices, top_k, ...)`
  - chooses helper tokens most coupled to target set.

- `reconstruct_target_with_helpers(matrix, target_indices, helper_indices, ...)`
  - factorizes target+helpers submatrix,
  - returns reconstructed target coordinates and Gram.

- `procrustes_error(reference, estimate)`
  - geometric mismatch after optimal rotation.

- `helper_reconstruction_curve(original_matrix, ablated_matrix, ...)`
  - curve of reconstruction quality vs helper size,
  - strategies: `top_coupling` and `random`.

## Practical interpretation

- If helper-based reconstruction improves strongly with helper size, this is
  evidence that the target structure is carried collectively by broader
  vocabulary interactions.
- Comparing `top_coupling` vs `random` helpers measures how specific helper
  selection contributes to recovery quality.
