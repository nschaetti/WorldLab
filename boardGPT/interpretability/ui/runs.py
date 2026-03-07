"""
Run launcher helpers for Streamlit B5-C.
"""

from __future__ import annotations

import time
from typing import Dict, Optional

from boardGPT.interpretability.pipeline import run_b3_demo_pipeline
from boardGPT.interpretability.real_pipeline import (
    run_b4_batch_from_config_file,
    run_b4_from_config_file,
    run_b6_matrix_from_config_file,
)


def launch_run(
    mode: str,
    out_root: str,
    run_name: Optional[str] = None,
    exp_config: Optional[str] = None,
) -> Dict[str, object]:
    """
    Launch a run in synchronous mode and return metadata.
    """
    mode = mode.strip().lower()
    t0 = time.time()

    if mode == "b3_demo":
        output = run_b3_demo_pipeline(out_root=out_root, run_name=run_name)
        run_dir = str(output.get("run_dir", ""))
    elif mode == "b4_real":
        if not exp_config:
            raise ValueError("exp_config is required for b4_real mode")
        output = run_b4_from_config_file(exp_config, run_name=run_name)
        run_dir = str(output.get("run_dir", ""))
    elif mode == "b4_batch":
        if not exp_config:
            raise ValueError("exp_config is required for b4_batch mode")
        output = run_b4_batch_from_config_file(exp_config, run_name=run_name)
        run_dir = str(output.get("batch_dir", ""))
    elif mode == "b6_matrix":
        if not exp_config:
            raise ValueError("exp_config is required for b6_matrix mode")
        output = run_b6_matrix_from_config_file(exp_config, run_name=run_name, resume=True)
        run_dir = str(output.get("matrix_dir", ""))
    else:
        raise ValueError(f"Unsupported run mode: {mode}")

    elapsed_s = time.time() - t0
    return {
        "mode": mode,
        "run_dir": run_dir,
        "elapsed_s": elapsed_s,
        "output": output,
    }
