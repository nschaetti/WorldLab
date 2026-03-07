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
- `matrix`: optional B6 systematic grid sweep (layers x cells x classes x seeds x checkpoints).
- `report`: artifact/report output options.

## Validation mode

Set `experiment.validation_mode` to control integrity behavior:

- `strict`: fail fast on critical mismatches (recommended for benchmark/paper runs).
- `tolerant`: record warnings and continue when possible (recommended for UI exploration).

## Reproducibility (B7)

Use `experiment.seed` and `experiment.reproducibility`:

- `seed`: global seed applied to Python/NumPy/Torch.
- `reproducibility.deterministic`: toggles deterministic backend settings.
- `reproducibility.include_pip_freeze`: if true, writes `requirements_frozen.txt`.

Each run now writes:

- `reproducibility.json`
- `reproducibility_report.md`
- optional `requirements_frozen.txt`

Run outputs include:

- `run_integrity`: `clean` or `degraded`
- `warnings`: list of non-fatal issues
- `telemetry`: stage timings and memory snapshot

## Probe fallback

If `probe.path` is empty, you can enable optional fallback probe training:

- `probe.auto_train_if_missing: true`
- tune `auto_train_epochs`, `auto_train_lr`, `auto_train_batch_size`,
  `auto_train_max_examples`, `auto_train_layer`, `auto_train_position`.

When enabled, the pipeline will:

1. extract residual activations from the configured layer,
2. derive final-board labels from the move sequences,
3. train the probe for a few epochs,
4. record training/eval stats in `summary.json`.

## Dataset format support

`dataset.format` can be one of:

- `bin`, `pkl`, `pickle` (pickle payload)
- `npy`, `npz`
- `json`, `jsonl`

For structured payloads, the loader also accepts mappings that contain game
lists under keys like `games` or `data`.

## Recommended defaults

- `protocol.position: "last"`
- `intervention.steps: 20-40` for quick runs, `80+` for deeper runs
- `dataset.max_examples: 64-256` for iteration speed
- `batch.seeds: [0,1,2]` for first stability checks

## Typical run matrix

- Quick sanity: 1 checkpoint x 1 seed x 64 examples
- Stability: 1 checkpoint x 3 seeds x 128 examples
- Comparison: 2+ checkpoints x 3 seeds x 128+ examples

## B6 matrix runner

Use `real-matrix` to run a systematic grid over:

- `matrix.layers`
- `matrix.target_cells`
- `matrix.target_classes`
- `matrix.positions`
- `matrix.seeds`
- checkpoints from `matrix.checkpoints` (or `batch.checkpoints` fallback)

Outputs are written in a dedicated matrix directory:

- `matrix_runs.json` (all points and statuses)
- `matrix_results.csv` (flat table)
- `matrix_summary.json` (aggregations by checkpoint/layer)

CLI:

```bash
python tools/interpretability_cli.py real-matrix --exp-config config/interpretability/b4_example.yaml
```

Disable resume and force rerun:

```bash
python tools/interpretability_cli.py real-matrix --exp-config config/interpretability/b4_example.yaml --no-resume
```

## B8 publication pack

Generate a publication-style markdown and notebook from a completed matrix run:

```bash
python tools/interpretability_cli.py real-publish --matrix-dir out_interp/b4/<matrix_run_name>
```

Optional naming:

```bash
python tools/interpretability_cli.py real-publish \
  --matrix-dir out_interp/b4/<matrix_run_name> \
  --title "BoardGPT Interpretability Matrix" \
  --markdown-name b8_report.md \
  --notebook-name b8_report.ipynb
```

Outputs are written in the matrix directory.

## One-shot B6 -> B8

Run matrix and publication generation in one command:

```bash
python tools/interpretability_cli.py real-one-shot \
  --exp-config config/interpretability/b4_example.yaml \
  --run-name matrix_pub_run
```

Force full recomputation of matrix points:

```bash
python tools/interpretability_cli.py real-one-shot \
  --exp-config config/interpretability/b4_example.yaml \
  --run-name matrix_pub_run \
  --no-resume
```
