# B4 Experiment Config

This file describes the expected config shape for real-model experiments.

Example config: `config/interpretability/b4_example.yaml`

## Sections

- `experiment`: global metadata and output root.
- `model`: default checkpoint and runtime settings.
- `probe`: probe architecture and optional checkpoint path.
- `dataset`: source path and dataloader parameters.
- `protocol`: B2 protocol parameters (layers, targets, position).
- `intervention`: latent intervention optimizer settings.
- `batch`: optional multi-seed and multi-checkpoint sweep.
- `report`: artifact/report output options.

## Recommended defaults

- `protocol.position: "last"`
- `intervention.steps: 20-40` for quick runs, `80+` for deeper runs
- `dataset.max_examples: 64-256` for iteration speed
- `batch.seeds: [0,1,2]` for first stability checks

## Typical run matrix

- Quick sanity: 1 checkpoint x 1 seed x 64 examples
- Stability: 1 checkpoint x 3 seeds x 128 examples
- Comparison: 2+ checkpoints x 3 seeds x 128+ examples
