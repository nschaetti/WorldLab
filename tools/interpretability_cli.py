#!/usr/bin/env python3
"""
Thin wrapper to launch BoardGPT interpretability CLI.

Examples:
  python tools/interpretability_cli.py smoke
  python tools/interpretability_cli.py b2-metrics --layers 0,1
  python tools/interpretability_cli.py report --run-dir out_interp/jalon_b3/<run>
"""

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from boardGPT.interpretability.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
