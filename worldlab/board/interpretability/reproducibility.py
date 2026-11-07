"""
Reproducibility helpers for interpretability workflows (B7).

This module centralizes deterministic-runtime setup and reproducibility artifact
generation for experiment runs.
"""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


def set_global_determinism(seed: int, deterministic: bool = True) -> Dict[str, Any]:
    """
    Set global random seeds and deterministic backend settings.

    Returns a dictionary describing the applied settings.
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    applied = {
        "seed": seed,
        "deterministic": bool(deterministic),
    }

    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        try:
            torch.use_deterministic_algorithms(True)
            applied["torch_deterministic_algorithms"] = True
        except Exception as exc:
            applied["torch_deterministic_algorithms"] = f"unavailable: {exc}"

        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            applied["cudnn_deterministic"] = True
            applied["cudnn_benchmark"] = False
        except Exception as exc:
            applied["cudnn_settings"] = f"unavailable: {exc}"
    else:
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass

    return applied


def _safe_subprocess(cmd: list[str], cwd: Optional[Path] = None) -> Optional[str]:
    try:
        out = subprocess.check_output(cmd, cwd=str(cwd) if cwd else None, stderr=subprocess.DEVNULL)
    except Exception:
        return None
    return out.decode("utf-8", errors="replace").strip()


def collect_environment_snapshot(include_pip_freeze: bool = False) -> Dict[str, Any]:
    """
    Collect environment metadata useful for reproducibility reports.
    """
    repo_root = Path(__file__).resolve().parents[3]
    snapshot: Dict[str, Any] = {
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "platform": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "torch": {
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            "cuda_version": torch.version.cuda,
        },
        "numpy": {
            "version": np.__version__,
        },
        "git": {
            "commit": _safe_subprocess(["git", "rev-parse", "HEAD"], cwd=repo_root),
            "branch": _safe_subprocess(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root),
            "status_short": _safe_subprocess(["git", "status", "--short"], cwd=repo_root),
        },
    }

    if torch.cuda.is_available():
        devices = []
        for i in range(torch.cuda.device_count()):
            devices.append(
                {
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                }
            )
        snapshot["torch"]["cuda_devices"] = devices

    if include_pip_freeze:
        freeze = _safe_subprocess([sys.executable, "-m", "pip", "freeze"], cwd=repo_root)
        snapshot["pip_freeze"] = freeze.splitlines() if isinstance(freeze, str) else []

    return snapshot


def write_reproducibility_pack(
    run_dir: str,
    config: Dict[str, Any],
    seed_info: Dict[str, Any],
    env_snapshot: Dict[str, Any],
) -> Dict[str, str]:
    """
    Write reproducibility artifacts in run directory.

    Files written:
    - reproducibility.json
    - reproducibility_report.md
    - requirements_frozen.txt (optional, if available in env snapshot)
    """
    out_dir = Path(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    obj = {
        "seed": seed_info,
        "config_repro": (config.get("experiment", {}) or {}).get("reproducibility", {}),
        "environment": env_snapshot,
    }

    repro_json = out_dir / "reproducibility.json"
    repro_json.write_text(json.dumps(obj, indent=2), encoding="utf-8")

    lines = [
        "# Reproducibility Report",
        "",
        "## Determinism",
        "",
    ]
    for k, v in seed_info.items():
        lines.append(f"- {k}: {v}")

    lines += [
        "",
        "## Environment",
        "",
        f"- Python: {env_snapshot.get('python', {}).get('version', '')}",
        f"- Platform: {env_snapshot.get('platform', {}).get('platform', '')}",
        f"- Torch: {env_snapshot.get('torch', {}).get('version', '')}",
        f"- CUDA available: {env_snapshot.get('torch', {}).get('cuda_available', False)}",
        f"- NumPy: {env_snapshot.get('numpy', {}).get('version', '')}",
        "",
        "## Git",
        "",
        f"- Branch: {env_snapshot.get('git', {}).get('branch')}",
        f"- Commit: {env_snapshot.get('git', {}).get('commit')}",
    ]

    report_md = out_dir / "reproducibility_report.md"
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    paths = {
        "reproducibility_json": str(repro_json),
        "reproducibility_report_md": str(report_md),
    }

    freeze = env_snapshot.get("pip_freeze") if isinstance(env_snapshot, dict) else None
    if isinstance(freeze, list) and freeze:
        req_path = out_dir / "requirements_frozen.txt"
        req_path.write_text("\n".join(str(x) for x in freeze) + "\n", encoding="utf-8")
        paths["requirements_frozen"] = str(req_path)

    return paths
