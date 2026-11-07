"""
Run discovery/index helpers for Streamlit B5-C.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def read_run_meta(run_dir: str) -> Dict[str, Any]:
    """Read optional UI metadata (`run_meta.json`) for a run directory."""
    path = Path(run_dir) / "run_meta.json"
    obj = _read_json(path)
    return obj if isinstance(obj, dict) else {}


def write_run_meta(run_dir: str, tags: List[str], notes: str) -> str:
    """Persist UI metadata (`tags`, `notes`) next to run artifacts."""
    run_path = Path(run_dir)
    if not run_path.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    clean_tags = [str(t).strip() for t in tags if str(t).strip()]
    payload = {
        "tags": clean_tags,
        "notes": str(notes or ""),
    }
    meta_path = run_path / "run_meta.json"
    meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(meta_path)


def parse_tags(text: str) -> List[str]:
    """Parse comma-separated tags string into a normalized list."""
    return [x.strip() for x in str(text).split(",") if x.strip()]


def discover_runs(root: str) -> List[Dict[str, Any]]:
    """
    Discover run directories that contain `summary.json`.
    """
    base = Path(root)
    if not base.exists():
        return []

    out: List[Dict[str, Any]] = []
    for summary_path in base.rglob("summary.json"):
        run_dir = summary_path.parent
        summary = _read_json(summary_path) or {}
        config = _read_json(run_dir / "config.json") or {}
        meta = _read_json(run_dir / "run_meta.json") or {}

        metrics = summary.get("metrics") if isinstance(summary, dict) else {}
        overall = metrics.get("overall") if isinstance(metrics, dict) else {}
        success_rate = overall.get("success_rate") if isinstance(overall, dict) else None

        entry = {
            "run_name": run_dir.name,
            "run_dir": str(run_dir),
            "updated_ts": run_dir.stat().st_mtime,
            "has_report": (run_dir / "report.md").exists(),
            "has_metrics_csv": (run_dir / "metrics_by_layer.csv").exists(),
            "has_top_csv": (run_dir / "top_contributors.csv").exists(),
            "mode": summary.get("mode") if isinstance(summary, dict) else None,
            "success_rate": success_rate,
            "model_path": ((config.get("model") or {}).get("model_path") if isinstance(config, dict) else None),
            "dataset_path": ((config.get("dataset") or {}).get("path") if isinstance(config, dict) else None),
            "tags": ", ".join(meta.get("tags", [])) if isinstance(meta.get("tags", []), list) else "",
            "notes": str(meta.get("notes", "")) if isinstance(meta, dict) else "",
        }
        out.append(entry)

    out.sort(key=lambda x: x.get("updated_ts", 0), reverse=True)
    return out


def read_run_text_preview(run_dir: str, file_name: str, max_chars: int = 3000) -> str:
    path = Path(run_dir) / file_name
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) > max_chars:
        return text[:max_chars] + "\n..."
    return text
