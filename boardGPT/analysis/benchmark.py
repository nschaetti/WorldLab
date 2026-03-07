"""
Benchmark utilities for running multiple symmetry workflows.

This module wraps the Phase 6 workflow runner to support reproducible campaign
execution over a grid of synthetic world/policy settings.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import csv
import json

import numpy as np

from .workflows import WorkflowConfig, run_synthetic_symmetry_workflow


@dataclass(frozen=True)
class BenchmarkRunRecord:
    """A single benchmark run record with raw summary and flattened metrics."""

    run_name: str
    run_dir: str
    summary: dict[str, Any]
    flat_metrics: dict[str, Any]


def _to_scalar(value: Any) -> Any:
    """Convert NumPy values/arrays to compact scalar/string-friendly values."""
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return "[]"
        if value.ndim == 1 and value.size <= 8:
            return json.dumps(value.tolist())
        return float(value.reshape(-1)[-1])
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return "[]"
        if len(value) <= 8 and all(isinstance(x, (int, float, bool, str)) for x in value):
            return json.dumps(list(value))
        return _to_scalar(value[-1])
    if isinstance(value, dict):
        return json.dumps(value)
    return str(value)


def _last_numeric(value: Any) -> Any:
    """Return the last element for sequence-like metrics, else scalar conversion."""
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        return _to_scalar(value.reshape(-1)[-1])
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return None
        return _to_scalar(value[-1])
    return _to_scalar(value)


def flatten_workflow_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """
    Flatten the workflow summary to a one-row metric dictionary.

    The output is designed for CSV export and quick ranking/filtering.
    """
    world = summary.get("world", {})
    cooc = summary.get("cooccurrence", {})
    fit = cooc.get("fit", {}) if isinstance(cooc, dict) else {}
    ab = summary.get("ablation", {})
    model = summary.get("model", {})
    dec = summary.get("decoding", {})

    row: dict[str, Any] = {
        "world_shape": _to_scalar(world.get("shape")),
        "world_periodic": _to_scalar(world.get("periodic")),
        "world_vocab_size": _to_scalar(world.get("vocab_size")),
        "world_dim": _to_scalar(world.get("dim")),
        "cooc_symmetry_r2": _to_scalar(cooc.get("symmetry_r2")),
        "cooc_symmetry_mse": _to_scalar(cooc.get("symmetry_mse")),
        "cooc_fit_amplitude": _to_scalar(fit.get("amplitude")),
        "cooc_fit_sigma": _to_scalar(fit.get("sigma")),
        "cooc_fit_offset": _to_scalar(fit.get("offset")),
        "cooc_fit_r2": _to_scalar(fit.get("r2")),
        "ablation_target_size": _to_scalar(ab.get("target_size")),
        "ablation_last_procrustes": _last_numeric(ab.get("procrustes_mean")),
        "ablation_last_gram_rel_mse": _last_numeric(ab.get("gram_rel_mse_mean")),
    }

    if model:
        row.update(
            {
                "model_train_loss_first": _to_scalar(model.get("train_loss_first")),
                "model_train_loss_last": _to_scalar(model.get("train_loss_last")),
                "model_symmetry_r2": _to_scalar(model.get("symmetry_r2")),
                "model_symmetry_mse": _to_scalar(model.get("symmetry_mse")),
            }
        )
    else:
        row.update(
            {
                "model_train_loss_first": None,
                "model_train_loss_last": None,
                "model_symmetry_r2": None,
                "model_symmetry_mse": None,
            }
        )

    if dec:
        row.update(
            {
                "decoding_last_train_rel_mse": _last_numeric(dec.get("train_rel_mse_mean")),
                "decoding_last_test_rel_mse": _last_numeric(dec.get("test_rel_mse_mean")),
                "decoding_num_trials": _to_scalar(dec.get("num_trials")),
            }
        )
    else:
        row.update(
            {
                "decoding_last_train_rel_mse": None,
                "decoding_last_test_rel_mse": None,
                "decoding_num_trials": None,
            }
        )

    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if len(rows) == 0:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _to_scalar(row.get(k)) for k in fieldnames})


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    if len(rows) == 0:
        path.write_text("# Symmetry Benchmark\n\nNo runs.\n", encoding="utf-8")
        return

    cols = [
        "run_name",
        "world_shape",
        "cooc_symmetry_r2",
        "cooc_fit_sigma",
        "ablation_last_procrustes",
        "model_symmetry_r2",
        "decoding_last_test_rel_mse",
    ]

    lines = ["# Symmetry Benchmark", "", "| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in rows:
        values = [str(_to_scalar(row.get(c))) for c in cols]
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_workflow_benchmark(
    benchmark_name: str,
    configs: Iterable[tuple[str, WorkflowConfig]],
    output_root: str,
) -> dict[str, Any]:
    """
    Run a benchmark over named workflow configurations.

    Args:
        benchmark_name: Name for this benchmark campaign.
        configs: Iterable of ``(run_name, WorkflowConfig)``.
        output_root: Directory where run artifacts are written.

    Returns:
        Benchmark summary dictionary with CSV/Markdown artifact paths.
    """
    out_root = Path(output_root) / benchmark_name
    out_root.mkdir(parents=True, exist_ok=True)

    records: list[BenchmarkRunRecord] = []
    table_rows: list[dict[str, Any]] = []

    for run_name, cfg in configs:
        run_dir = out_root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        cfg2 = WorkflowConfig(**{**cfg.__dict__, "output_dir": str(run_dir)})
        summary = run_synthetic_symmetry_workflow(cfg2)
        flat = flatten_workflow_summary(summary)

        table_row = {"run_name": run_name, **flat}
        table_rows.append(table_row)
        records.append(
            BenchmarkRunRecord(
                run_name=run_name,
                run_dir=str(run_dir),
                summary=summary,
                flat_metrics=flat,
            )
        )

    csv_path = out_root / "benchmark_results.csv"
    md_path = out_root / "benchmark_report.md"
    summary_path = out_root / "benchmark_summary.json"

    _write_csv(csv_path, table_rows)
    _write_markdown(md_path, table_rows)

    benchmark_summary = {
        "benchmark_name": benchmark_name,
        "num_runs": len(table_rows),
        "output_root": str(out_root),
        "csv_path": str(csv_path),
        "markdown_path": str(md_path),
        "runs": [
            {
                "run_name": r.run_name,
                "run_dir": r.run_dir,
                "flat_metrics": r.flat_metrics,
            }
            for r in records
        ],
    }
    summary_path.write_text(json.dumps(benchmark_summary, indent=2), encoding="utf-8")
    benchmark_summary["summary_path"] = str(summary_path)
    return benchmark_summary
