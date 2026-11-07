"""
Reporting helpers for interpretability runs (B3).
"""

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def export_metrics_csv(run_dir: str, metrics_report: Dict[str, object]) -> Dict[str, str]:
    """
    Export common B2 metrics tables as CSV files.
    """
    out_dir = Path(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_obj = metrics_report.get("metrics")
    if not isinstance(metrics_obj, dict):
        raise ValueError("metrics_report must contain a dict under key 'metrics'")

    overall = metrics_obj.get("overall", {})
    by_layer = metrics_obj.get("by_layer", {})

    overall_csv = out_dir / "metrics_overall.csv"
    with overall_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        if isinstance(overall, dict):
            for key, value in overall.items():
                writer.writerow([key, _fmt(value)])

    layer_csv = out_dir / "metrics_by_layer.csv"
    with layer_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "success_rate", "target_gain_mean", "delta_norm_mean", "off_target_abs_mean"])
        if isinstance(by_layer, dict):
            for layer, row in by_layer.items():
                if not isinstance(row, dict):
                    continue
                gain = row.get("target_gain", {}) if isinstance(row.get("target_gain"), dict) else {}
                dnorm = row.get("delta_norm", {}) if isinstance(row.get("delta_norm"), dict) else {}
                off = row.get("off_target", {}) if isinstance(row.get("off_target"), dict) else {}
                writer.writerow(
                    [
                        layer,
                        _fmt(row.get("success_rate", 0.0)),
                        _fmt(gain.get("mean", 0.0)),
                        _fmt(dnorm.get("mean", 0.0)),
                        _fmt(off.get("mean_abs_off_target_change", 0.0)),
                    ]
                )

    return {
        "metrics_overall": str(overall_csv),
        "metrics_by_layer": str(layer_csv),
    }


def export_top_contributors_csv(run_dir: str, attribution_report: Dict[str, object]) -> str:
    """
    Export attribution top contributors as CSV.
    """
    out_dir = Path(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    contrib_csv = out_dir / "top_contributors.csv"
    with contrib_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "layer", "cell", "score", "abs_score"])
        top = attribution_report.get("top_contributors")
        if isinstance(top, list):
            for idx, row in enumerate(top, start=1):
                if not isinstance(row, dict):
                    continue
                writer.writerow(
                    [
                        idx,
                        _fmt(row.get("layer", "")),
                        _fmt(row.get("cell", "")),
                        _fmt(row.get("score", "")),
                        _fmt(row.get("abs_score", "")),
                    ]
                )

    return str(contrib_csv)


def generate_markdown_report(
    run_dir: str,
    run_name: str,
    summary: Dict[str, Any],
    metrics_report: Dict[str, object],
    attribution_report: Dict[str, object],
) -> str:
    """
    Generate a concise markdown report in run directory.
    """
    out_dir = Path(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_obj = metrics_report.get("metrics")
    overall = metrics_obj.get("overall", {}) if isinstance(metrics_obj, dict) else {}
    by_layer = metrics_obj.get("by_layer", {}) if isinstance(metrics_obj, dict) else {}

    top = attribution_report.get("top_contributors", [])
    if not isinstance(top, list):
        top = []

    lines: List[str] = []
    lines.append(f"# Interpretability Report - {run_name}")
    lines.append("")
    lines.append("## Run Summary")
    lines.append("")
    for key, value in summary.items():
        lines.append(f"- {key}: {_fmt(value)}")
    lines.append("")
    lines.append("## Overall Metrics")
    lines.append("")
    if isinstance(overall, dict):
        for key, value in overall.items():
            lines.append(f"- {key}: {_fmt(value)}")
    lines.append("")
    lines.append("## Metrics by Layer")
    lines.append("")
    lines.append("| layer | success_rate | target_gain_mean | delta_norm_mean |")
    lines.append("|---:|---:|---:|---:|")
    if isinstance(by_layer, dict):
        for layer, row in by_layer.items():
            if not isinstance(row, dict):
                continue
            gain = row.get("target_gain", {}) if isinstance(row.get("target_gain"), dict) else {}
            dnorm = row.get("delta_norm", {}) if isinstance(row.get("delta_norm"), dict) else {}
            lines.append(
                f"| {layer} | {_fmt(row.get('success_rate', 0.0))} | {_fmt(gain.get('mean', 0.0))} | {_fmt(dnorm.get('mean', 0.0))} |"
            )
    lines.append("")
    lines.append("## Top Contributors")
    lines.append("")
    lines.append("| rank | layer | cell | score | abs_score |")
    lines.append("|---:|---:|---:|---:|---:|")
    for idx, row in enumerate(top[:15], start=1):
        if not isinstance(row, dict):
            continue
        lines.append(
            f"| {idx} | {_fmt(row.get('layer', ''))} | {_fmt(row.get('cell', ''))} | {_fmt(row.get('score', ''))} | {_fmt(row.get('abs_score', ''))} |"
        )

    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(report_path)
