# Symmetry Reproduction - Phase 6

Phase 6 packages the previous phases into a reproducible workflow and learning
index.

## Additions

- `boardGPT/analysis/workflows.py`
  - `WorkflowConfig`
  - `run_synthetic_symmetry_workflow(...)`

- `tools/symmetry_workflow.py`
  - CLI entry point for end-to-end synthetic runs.

- `boardGPT/analysis/benchmark.py`
  - multi-run benchmark aggregation utilities.

- `tools/symmetry_benchmark.py`
  - CLI for benchmark campaigns (CSV/Markdown/JSON outputs).

- `notebooks/symmetry/00_learning_path.ipynb`
  - central notebook index and quick-start.

## What the workflow runs

1. Build synthetic world and generate sequences,
2. Compute co-occurrence + symmetry diagnostics + exponential fit,
3. Run helper-ablation curve,
4. Optionally train tiny GPT,
5. Optionally extract representations and run decoding curves,
6. Optionally save a JSON summary.

## CLI examples

1D periodic:

```bash
python tools/symmetry_workflow.py --shape 32 --periodic 1 --run-model --output out/symmetry/demo_1d
```

2D open:

```bash
python tools/symmetry_workflow.py --shape 8,8 --periodic 0,0 --run-model --output out/symmetry/demo_2d
```

Benchmark campaign:

```bash
python tools/symmetry_benchmark.py --name campaign_default --preset default --output out/symmetry_benchmarks
```
