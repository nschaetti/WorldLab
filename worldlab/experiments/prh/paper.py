"""Use the authors' exact registry and advanced experiment entry points."""

import csv
import shlex
import subprocess
import sys
from pathlib import Path

UPSTREAM_REVISION = "a2b9949fa907d4fbe562742d4b1ffa7205ab7427"
SCRIPTS = {
    "train": ("train.py",),
    "evaluate": ("evaluate.py",),
    "caption": ("caption.py",),
    "tables": ("tables.py",),
    "figures": ("figures.py",),
    "data": ("-m", "prh.data"),
}


def _checkout(path):
    path = Path(path).expanduser().resolve()
    result = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True)
    if result.returncode or result.stdout.strip() != UPSTREAM_REVISION:
        raise ValueError(f"Expected the official prh-proof checkout at revision {UPSTREAM_REVISION}")
    return path


def upstream(args):
    root = _checkout(args.checkout)
    command = [sys.executable, *SCRIPTS[args.script], *args.arguments]
    subprocess.run(command, cwd=root, check=True)


def registry(args):
    root = _checkout(args.checkout)
    with (root / "paper/runs.csv").open(newline="", encoding="utf-8") as stream:
        matches = [row for row in csv.DictReader(stream) if row["tag"] == args.tag]
    if not matches:
        raise ValueError(f"No registered paper run named {args.tag!r}")
    if len(matches) != 1:
        raise ValueError("Run tag is ambiguous")
    command = [sys.executable, "train.py", *shlex.split(matches[0]["train_args"]), *args.arguments]
    print(shlex.join(command), flush=True)
    if args.run:
        subprocess.run(command, cwd=root, check=True)
