"""
Run comparison helpers for Streamlit B5-C.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def compare_run_dirs(run_a: str, run_b: str) -> Dict[str, Any]:
    """
    Compare two run directories using summary + CSV artifacts.
    """
    a = Path(run_a)
    b = Path(run_b)
    if not a.exists() or not b.exists():
        raise FileNotFoundError("Both run directories must exist")

    sa = _read_json(a / "summary.json")
    sb = _read_json(b / "summary.json")

    oa: Dict[str, Any] = {}
    ob: Dict[str, Any] = {}

    if isinstance(sa, dict) and isinstance(sa.get("metrics"), dict):
        candidate = (sa.get("metrics") or {}).get("overall")
        if isinstance(candidate, dict):
            oa = candidate
    elif isinstance(sa, dict) and isinstance(sa.get("overall"), dict):
        candidate = sa.get("overall")
        if isinstance(candidate, dict):
            oa = candidate

    if isinstance(sb, dict) and isinstance(sb.get("metrics"), dict):
        candidate = (sb.get("metrics") or {}).get("overall")
        if isinstance(candidate, dict):
            ob = candidate
    elif isinstance(sb, dict) and isinstance(sb.get("overall"), dict):
        candidate = sb.get("overall")
        if isinstance(candidate, dict):
            ob = candidate

    oa_target_gain: Dict[str, Any] = {}
    ob_target_gain: Dict[str, Any] = {}
    oa_tg_candidate = oa.get("target_gain")
    ob_tg_candidate = ob.get("target_gain")
    if isinstance(oa_tg_candidate, dict):
        oa_target_gain = oa_tg_candidate
    if isinstance(ob_tg_candidate, dict):
        ob_target_gain = ob_tg_candidate

    global_delta = {
        "success_rate_delta": _to_float(oa.get("success_rate"), 0.0) * -1.0 + _to_float(ob.get("success_rate"), 0.0),
        "target_gain_mean_delta": _to_float(ob_target_gain.get("mean"), 0.0)
        - _to_float(oa_target_gain.get("mean"), 0.0),
    }

    la_rows = _read_csv(a / "metrics_by_layer.csv")
    lb_rows = _read_csv(b / "metrics_by_layer.csv")
    by_layer_a = {r.get("layer", ""): r for r in la_rows}
    by_layer_b = {r.get("layer", ""): r for r in lb_rows}
    layers = sorted(set(by_layer_a.keys()) | set(by_layer_b.keys()), key=lambda x: int(x) if str(x).isdigit() else 9999)

    layer_deltas: List[Dict[str, Any]] = []
    for layer in layers:
        ra = by_layer_a.get(layer, {})
        rb = by_layer_b.get(layer, {})
        layer_deltas.append(
            {
                "layer": layer,
                "success_rate_a": _to_float(ra.get("success_rate")),
                "success_rate_b": _to_float(rb.get("success_rate")),
                "success_rate_delta": _to_float(rb.get("success_rate")) - _to_float(ra.get("success_rate")),
                "target_gain_mean_a": _to_float(ra.get("target_gain_mean")),
                "target_gain_mean_b": _to_float(rb.get("target_gain_mean")),
                "target_gain_mean_delta": _to_float(rb.get("target_gain_mean")) - _to_float(ra.get("target_gain_mean")),
            }
        )

    top_a = _read_csv(a / "top_contributors.csv")
    top_b = _read_csv(b / "top_contributors.csv")

    def _key(row: Dict[str, str]) -> Tuple[str, str]:
        return (str(row.get("layer", "")), str(row.get("cell", "")))

    keys_a = {_key(r) for r in top_a}
    keys_b = {_key(r) for r in top_b}
    overlap = sorted(keys_a & keys_b)

    return {
        "run_a": str(a),
        "run_b": str(b),
        "global_delta": global_delta,
        "layer_deltas": layer_deltas,
        "top_overlap_count": len(overlap),
        "top_overlap": [{"layer": k[0], "cell": k[1]} for k in overlap[:30]],
    }
