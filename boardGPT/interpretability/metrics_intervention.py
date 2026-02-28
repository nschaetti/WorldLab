"""
Metrics utilities for intervention experiments (B2).
"""

from statistics import mean, median
from typing import Dict, Iterable, List

import torch


def _as_list(results: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    values = list(results)
    if not values:
        raise ValueError("Expected at least one intervention result")
    return values


def _quantile(values: List[float], q: float) -> float:
    if not values:
        raise ValueError("Cannot compute quantile on empty list")
    sorted_vals = sorted(values)
    idx = int(round((len(sorted_vals) - 1) * q))
    return float(sorted_vals[idx])


def compute_success_rate(results: Iterable[Dict[str, object]]) -> float:
    rows = _as_list(results)
    return float(sum(1 for r in rows if bool(r.get("success", False))) / len(rows))


def compute_target_gain_stats(results: Iterable[Dict[str, object]]) -> Dict[str, float]:
    rows = _as_list(results)
    gains = [float(r.get("target_prob_gain", 0.0)) for r in rows]
    m = mean(gains)
    var = mean([(g - m) ** 2 for g in gains])
    return {
        "mean": float(m),
        "std": float(var ** 0.5),
        "median": float(median(gains)),
        "min": float(min(gains)),
        "max": float(max(gains)),
        "p25": _quantile(gains, 0.25),
        "p75": _quantile(gains, 0.75),
    }


def compute_delta_norm_stats(results: Iterable[Dict[str, object]]) -> Dict[str, float]:
    rows = _as_list(results)
    norms = [float(r.get("delta_norm", 0.0)) for r in rows]
    m = mean(norms)
    var = mean([(n - m) ** 2 for n in norms])
    return {
        "mean": float(m),
        "std": float(var ** 0.5),
        "min": float(min(norms)),
        "max": float(max(norms)),
    }


def compute_off_target_impact(results: Iterable[Dict[str, object]]) -> Dict[str, float]:
    """
    Estimate collateral effect outside target cell.

    Requires each result to include:
    - before_board_probs: Tensor/List [64,3]
    - after_board_probs: Tensor/List [64,3]
    - target_cell
    - target_class
    """
    rows = _as_list(results)

    mean_abs_impacts: List[float] = []
    mean_signed_impacts: List[float] = []

    for r in rows:
        before = r.get("before_board_probs")
        after = r.get("after_board_probs")
        target_cell = int(r.get("target_cell", -1))
        target_class = int(r.get("target_class", 0))
        if before is None or after is None:
            continue

        before_t = before if torch.is_tensor(before) else torch.tensor(before)
        after_t = after if torch.is_tensor(after) else torch.tensor(after)

        if before_t.shape != (64, 3) or after_t.shape != (64, 3):
            continue

        delta = after_t[:, target_class] - before_t[:, target_class]
        mask = torch.ones(64, dtype=torch.bool)
        if 0 <= target_cell < 64:
            mask[target_cell] = False
        off = delta[mask]
        mean_abs_impacts.append(float(off.abs().mean().item()))
        mean_signed_impacts.append(float(off.mean().item()))

    if not mean_abs_impacts:
        return {
            "mean_abs_off_target_change": 0.0,
            "mean_signed_off_target_change": 0.0,
            "num_valid_examples": 0.0,
        }

    return {
        "mean_abs_off_target_change": float(mean(mean_abs_impacts)),
        "mean_signed_off_target_change": float(mean(mean_signed_impacts)),
        "num_valid_examples": float(len(mean_abs_impacts)),
    }


def aggregate_by_layer(layer_report: Dict[str, object]) -> Dict[int, Dict[str, object]]:
    layer_results = layer_report.get("layer_results")
    if not isinstance(layer_results, dict):
        raise ValueError("Expected layer_report['layer_results'] to be a dict")

    output: Dict[int, Dict[str, object]] = {}
    for layer, report in layer_results.items():
        if not isinstance(report, dict) or "results" not in report:
            continue
        results = report["results"]
        if not isinstance(results, list) or not results:
            continue
        output[int(layer)] = {
            "success_rate": compute_success_rate(results),
            "target_gain": compute_target_gain_stats(results),
            "delta_norm": compute_delta_norm_stats(results),
            "off_target": compute_off_target_impact(results),
        }
    return output


def aggregate_by_cell(results: Iterable[Dict[str, object]]) -> Dict[int, Dict[str, object]]:
    rows = _as_list(results)
    grouped: Dict[int, List[Dict[str, object]]] = {}
    for r in rows:
        cell = int(r.get("target_cell", -1))
        grouped.setdefault(cell, []).append(r)

    out: Dict[int, Dict[str, object]] = {}
    for cell, items in grouped.items():
        out[cell] = {
            "count": len(items),
            "success_rate": compute_success_rate(items),
            "target_gain_mean": float(compute_target_gain_stats(items)["mean"]),
        }
    return out
