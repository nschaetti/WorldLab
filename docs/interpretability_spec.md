# Othello Interpretability Spec

This document defines the conventions used by `boardGPT.interpretability` for Jalon A.

## Token and move conventions

- Vocabulary size is 61.
- Token `0` is reserved for `BOS`.
- Move tokens are `1..60` and map to legal board coordinates excluding the 4 center squares (`d4`, `d5`, `e4`, `e5`).
- Notation uses lowercase algebraic coordinates like `a1`, `d3`, `h8`.

Helpers:

- `get_othello_move_mappings()` returns `(move_to_id, id_to_move)`.
- `move_ids_to_notation(ids, skip_bos=True)` converts token IDs to move strings.

## Board label convention

Board labels are a flat vector of length 64 in column-major coordinate order:

- `a1, a2, ..., a8, b1, ..., h8`

Cell classes are:

- `0`: empty
- `1`: white
- `2`: black

Helpers:

- `notation_to_board_labels(moves)`
- `move_ids_to_board_labels(ids, skip_bos=True)`

## Activation extraction convention

Residual activations are extracted from BoardGPT GPT with:

- `to_return=["residuals{layer}"]`
- layer index range: `[0, n_layer - 1]`

Extractor APIs:

- `extract_residuals(model, token_batch, layer, device=None)` -> `[B, T, D]`
- `extract_dataset_residuals(..., position_policy="all"|"last"|int)` -> concatenated tensor

## Probe output convention

Probe logits are shaped as:

- `[B, 64, 3]`

Predictions are class IDs with shape:

- `[B, 64]`

Implemented probes:

- `LinearBoardProbe(input_dim)`
- `MLPBoardProbe(input_dim, hidden_dim)`
