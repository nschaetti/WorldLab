# Symmetry Reproduction Guide (Phases 1-6)

This guide is the entry point for the full educational pipeline.

## What is implemented

- **Phase 1**: co-occurrence stats + concept geometry + synthetic policies,
- **Phase 2**: translation-symmetry diagnostics + exponential kernel fitting,
- **Phase 3**: GPT training and representation extraction,
- **Phase 4**: linear decoding curves vs rank,
- **Phase 5**: helper-based ablation and reconstruction,
- **Phase 6**: reproducible workflow orchestration and documentation.

## Reading path

1. `docs/symmetry_phase1.md`
2. `docs/symmetry_phase2.md`
3. `docs/symmetry_phase3.md`
4. `docs/symmetry_phase4.md`
5. `docs/symmetry_phase5.md`

Then run notebooks in order:

1. `notebooks/symmetry/00_learning_path.ipynb`
2. `notebooks/symmetry/01_phase1_foundations.ipynb`
3. `notebooks/symmetry/02_phase2_theory_diagnostics.ipynb`
4. `notebooks/symmetry/03_phase3_gpt_representations.ipynb`
5. `notebooks/symmetry/04_phase4_decoding_vs_rank.ipynb`
6. `notebooks/symmetry/05_phase5_helper_ablations.ipynb`

## End-to-end quick run

You can launch a full synthetic workflow from CLI:

```bash
python tools/symmetry_workflow.py --shape 32 --periodic 1 --run-model --output out/symmetry/demo
```

For a 2D world:

```bash
python tools/symmetry_workflow.py --shape 8,8 --periodic 0,0 --run-model --output out/symmetry/grid2d
```

Outputs include a JSON summary with symmetry, decoding, and ablation metrics.

## Multi-run benchmark campaign

To compare multiple worlds/policies in one shot:

```bash
python tools/symmetry_benchmark.py --name campaign_default --preset default --output out/symmetry_benchmarks
```

With model analysis enabled:

```bash
python tools/symmetry_benchmark.py --name campaign_with_model --preset default --run-model --output out/symmetry_benchmarks
```

Grid campaign example:

```bash
python tools/symmetry_benchmark.py --name campaign_grid --preset grid --dims 1,2 --sizes 8,12 --scales 0.18,0.3 --output out/symmetry_benchmarks
```

Benchmark artifacts:

- `benchmark_results.csv`
- `benchmark_report.md`
- `benchmark_summary.json`

## Suggested first scientific checks

1. **1D periodic local policy**: expect circular/Fourier structure.
2. **1D open local policy**: expect open manifold.
3. **2D local policy**: expect smooth low-frequency spatial modes.
4. **Ablation + helpers**: expect reconstruction quality to improve with helper count.
