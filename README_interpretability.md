# BoardGPT Interpretability Quickstart

This document summarizes the A/B1/B2/B3 interpretability stack.

## Components

- `boardGPT/interpretability/spec.py`: Othello token/board conventions.
- `boardGPT/interpretability/activation_extractor.py`: residual extraction APIs.
- `boardGPT/interpretability/probes.py`: linear and MLP probes.
- `boardGPT/interpretability/interventions.py`: latent intervention primitives.
- `boardGPT/interpretability/protocols.py`: B1/B2 protocol runners.
- `boardGPT/interpretability/metrics_intervention.py`: intervention metrics.
- `boardGPT/interpretability/attribution.py`: attribution maps and top contributors.
- `boardGPT/interpretability/io.py`: artifact persistence.
- `boardGPT/interpretability/report.py`: CSV + markdown reports.
- `boardGPT/interpretability/pipeline.py`: B3 end-to-end pipeline orchestrator.

## Run the smoke test

```bash
python tools/smoke_test_interpretability.py
```

## B5-A Streamlit lab

```bash
streamlit run apps/interpretability_lab.py
```

Workflow in the app:

1. Load model and optional probe from sidebar.
2. Enter sequence (`token_ids` or Othello `moves`) and parse input.
3. Run model prediction (top-k next token).
4. Extract residual activations (layer + position).
5. Run probe dashboard to visualize board prediction and confidence.
6. Use the Activation Patching tab to run before/after intervention impact.
7. Use Runs & Compare tab to launch B3/B4 runs, explore artifacts, and compare two runs.

## B3 CLI examples

All interpretability commands are inference-only (evaluation/intervention). They do not train model weights.

```bash
python tools/interpretability_cli.py smoke
python tools/interpretability_cli.py b2-metrics --layers 0,1
python tools/interpretability_cli.py b2-attrib --layers 0,1 --target-cells 3,5,9
python tools/interpretability_cli.py report --run-dir out_interp/jalon_b3/<run_name>
python tools/interpretability_cli.py real-metrics --exp-config config/interpretability/b4_example.yaml
python tools/interpretability_cli.py real-attrib --exp-config config/interpretability/b4_example.yaml
python tools/interpretability_cli.py real-batch --exp-config config/interpretability/b4_example.yaml
```

You can also use the games CLI bridge:

```bash
python -m boardGPT.games.cli othello interp smoke
python -m boardGPT.games.cli othello interp b2-metrics --layers 0,1
python -m boardGPT.games.cli othello interp report --run-dir out_interp/jalon_b3/<run_name>
python -m boardGPT.games.cli othello interp real-batch --exp-config config/interpretability/b4_example.yaml
```

## Artifacts layout

Each B3 run creates a directory under `out_interp/jalon_b3/<run_name>` with:

- `config.json`
- `summary.json`
- `results.jsonl`
- `tensors/*.pt`
- `metrics_overall.csv`
- `metrics_by_layer.csv`
- `top_contributors.csv`
- `report.md`

## B4 config template

- `config/interpretability/b4_example.yaml`
- `docs/interpretability_b4_config.md`

## Notebooks

- `notebooks/jalon_a_smoke_demo.ipynb`
- `notebooks/jalon_a_probe_workflow.ipynb`
- `notebooks/jalon_b1_interventions_demo.ipynb`
- `notebooks/jalon_b2_metrics_attribution_demo.ipynb`
- `notebooks/jalon_b3_run_analysis_template.ipynb`
- `notebooks/jalon_b3_run_from_notebook.ipynb`
- `notebooks/jalon_b3_compare_runs.ipynb`
