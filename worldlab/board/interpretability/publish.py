"""
Publication-style report/notebook generator for internal sharing (B8).

This module consumes B6 matrix artifacts and produces:
- a concise markdown synthesis report,
- a notebook scaffold with loading + plotting cells.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [dict(r) for r in reader]


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def generate_publication_markdown(
    matrix_dir: str,
    title: str = "Interpretability Matrix Report",
    output_name: str = "publication_report.md",
    top_k_rows: int = 20,
) -> str:
    """
    Generate publication-style markdown report from B6 matrix artifacts.

    Expected inputs in `matrix_dir`:
    - `matrix_summary.json`
    - `matrix_results.csv`
    """
    out_dir = Path(matrix_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = _read_json(out_dir / "matrix_summary.json")
    rows = _read_csv_rows(out_dir / "matrix_results.csv")

    completed = [r for r in rows if str(r.get("status", "")) in {"completed", "resumed"}]
    failed = [r for r in rows if str(r.get("status", "")) == "failed"]

    ranked = sorted(completed, key=lambda r: _to_float(r.get("success_rate"), 0.0), reverse=True)
    top_rows = ranked[: max(1, top_k_rows)] if ranked else []

    by_ckpt = summary.get("by_checkpoint", {}) if isinstance(summary, dict) else {}
    by_layer = summary.get("by_layer", {}) if isinstance(summary, dict) else {}

    lines: List[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append(f"- matrix_dir: `{out_dir}`")
    lines.append(f"- num_runs_total: {summary.get('num_runs_total', len(rows))}")
    lines.append(f"- num_runs_completed: {summary.get('num_runs_completed', len(completed))}")
    lines.append(f"- num_runs_resumed: {summary.get('num_runs_resumed', 0)}")
    lines.append(f"- num_runs_failed: {summary.get('num_runs_failed', len(failed))}")
    lines.append("")

    lines.append("## Aggregates by Checkpoint")
    lines.append("")
    lines.append("| checkpoint | count | success_rate_mean | target_gain_mean |")
    lines.append("|---|---:|---:|---:|")
    if isinstance(by_ckpt, dict):
        for ckpt, obj in sorted(by_ckpt.items(), key=lambda kv: str(kv[0])):
            if not isinstance(obj, dict):
                continue
            lines.append(
                "| {ckpt} | {count:.0f} | {sr:.6f} | {tg:.6f} |".format(
                    ckpt=ckpt,
                    count=_to_float(obj.get("count", 0.0)),
                    sr=_to_float(obj.get("success_rate_mean", 0.0)),
                    tg=_to_float(obj.get("target_gain_mean", 0.0)),
                )
            )
    lines.append("")

    lines.append("## Aggregates by Layer")
    lines.append("")
    lines.append("| layer | count | success_rate_mean | target_gain_mean |")
    lines.append("|---:|---:|---:|---:|")
    if isinstance(by_layer, dict):
        for layer, obj in sorted(by_layer.items(), key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else 10**9):
            if not isinstance(obj, dict):
                continue
            lines.append(
                "| {layer} | {count:.0f} | {sr:.6f} | {tg:.6f} |".format(
                    layer=layer,
                    count=_to_float(obj.get("count", 0.0)),
                    sr=_to_float(obj.get("success_rate_mean", 0.0)),
                    tg=_to_float(obj.get("target_gain_mean", 0.0)),
                )
            )
    lines.append("")

    lines.append(f"## Top {len(top_rows)} Runs by Success Rate")
    lines.append("")
    lines.append("| run_id | checkpoint | layer | cell | class | position | success_rate | target_gain_mean |")
    lines.append("|---|---|---:|---:|---:|---|---:|---:|")
    for r in top_rows:
        lines.append(
            "| {run_id} | {ckpt} | {layer} | {cell} | {cls} | {pos} | {sr:.6f} | {tg:.6f} |".format(
                run_id=str(r.get("run_id", "")),
                ckpt=str(r.get("checkpoint", "")),
                layer=str(r.get("layer", "")),
                cell=str(r.get("target_cell", "")),
                cls=str(r.get("target_class", "")),
                pos=str(r.get("position", "")),
                sr=_to_float(r.get("success_rate", 0.0)),
                tg=_to_float(r.get("target_gain_mean", 0.0)),
            )
        )
    lines.append("")

    if failed:
        lines.append("## Failed Runs")
        lines.append("")
        lines.append("| run_id | checkpoint | layer | cell | class | error |")
        lines.append("|---|---|---:|---:|---:|---|")
        for r in failed[:50]:
            lines.append(
                "| {run_id} | {ckpt} | {layer} | {cell} | {cls} | {err} |".format(
                    run_id=str(r.get("run_id", "")),
                    ckpt=str(r.get("checkpoint", "")),
                    layer=str(r.get("layer", "")),
                    cell=str(r.get("target_cell", "")),
                    cls=str(r.get("target_class", "")),
                    err=str(r.get("error", "")).replace("\n", " ")[:200],
                )
            )
        lines.append("")

    report_path = out_dir / output_name
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(report_path)


def generate_publication_notebook(
    matrix_dir: str,
    notebook_name: str = "publication_report.ipynb",
    title: str = "Interpretability Matrix Analysis",
) -> str:
    """
    Generate a lightweight notebook template bound to matrix artifacts.
    """
    out_dir = Path(matrix_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"# {title}\n",
                    "\n",
                    "This notebook was auto-generated from B6 matrix artifacts.\n",
                    f"Matrix directory: `{out_dir}`\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from pathlib import Path\n",
                    "import json\n",
                    "import pandas as pd\n",
                    "\n",
                    f"matrix_dir = Path(r\"{str(out_dir)}\")\n",
                    "summary = json.loads((matrix_dir / 'matrix_summary.json').read_text(encoding='utf-8'))\n",
                    "df = pd.read_csv(matrix_dir / 'matrix_results.csv')\n",
                    "summary, df.head()\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "completed = df[df['status'].isin(['completed', 'resumed'])].copy()\n",
                    "completed.sort_values('success_rate', ascending=False).head(20)\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Optional plots\n",
                    "import matplotlib.pyplot as plt\n",
                    "\n",
                    "if not completed.empty:\n",
                    "    by_layer = completed.groupby('layer')['success_rate'].mean().reset_index()\n",
                    "    plt.figure(figsize=(6, 4))\n",
                    "    plt.plot(by_layer['layer'], by_layer['success_rate'], marker='o')\n",
                    "    plt.xlabel('layer')\n",
                    "    plt.ylabel('mean success_rate')\n",
                    "    plt.title('Success Rate by Layer')\n",
                    "    plt.grid(True, alpha=0.3)\n",
                    "    plt.show()\n",
                ],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.x"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    nb_path = out_dir / notebook_name
    nb_path.write_text(json.dumps(nb, indent=2), encoding="utf-8")
    return str(nb_path)


def run_b8_publish_from_matrix_dir(
    matrix_dir: str,
    title: str = "Interpretability Matrix Report",
    markdown_name: str = "publication_report.md",
    notebook_name: str = "publication_report.ipynb",
) -> Dict[str, str]:
    """
    Generate markdown + notebook publication pack from a matrix directory.
    """
    md = generate_publication_markdown(
        matrix_dir=matrix_dir,
        title=title,
        output_name=markdown_name,
    )
    nb = generate_publication_notebook(
        matrix_dir=matrix_dir,
        notebook_name=notebook_name,
        title=title,
    )
    return {
        "matrix_dir": str(Path(matrix_dir)),
        "publication_markdown": md,
        "publication_notebook": nb,
    }
